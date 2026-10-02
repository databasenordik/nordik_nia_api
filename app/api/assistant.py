import asyncio
import json
import re
from collections.abc import AsyncIterator
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from app.concurrency.cancellation import CancellationToken
from app.config import get_settings
from app.data_gateway.memory import MemoryDataGateway
from app.data_gateway.protocol import DataGateway
from app.data_gateway.repository import AssistantDataGateway
from app.db.pool import get_pool
from app.execution.turn import AssistantTurnService, TurnResult
from app.llm.factory import allowed_models, get_reasoning_provider
from app.memory.feedback import (
    MAX_ATTACHMENTS,
    MAX_ATTACHMENTS_TOTAL_BYTES,
    MAX_COMMENT,
    MAX_NAME,
    FeedbackReport,
    decode_attachment,
    default_feedback_store,
)
from app.memory.store import default_memory_store
from app.observability.audit import default_auditor
from app.security.access_scope import AccessScope
from app.security.deps import get_access_scope
from app.tts.base import ensure_no_tts_in_text_mode

router = APIRouter(prefix="/api/assistant", tags=["assistant"])
_SSE_HEARTBEAT_SECONDS = 1.0
# Starlette renamed the 422 constant (UNPROCESSABLE_ENTITY -> _CONTENT) and deprecates the old
# name; the number itself is stable across the versions this runs on.
_UNPROCESSABLE = 422


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None
    selected_file_id: int | None = None
    mode: str = "text"
    # Which model answers this turn. Validated against the allowlist rather than trusted:
    # the name reaches the provider directly, and the host exposes hundreds of models.
    # An unrecognised name falls back to the default instead of failing the turn.
    model: str | None = Field(default=None, max_length=100)


def _gateway() -> DataGateway:
    try:
        return AssistantDataGateway(get_pool())
    except RuntimeError as exc:
        if get_settings().app_env == "development":
            return MemoryDataGateway()
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="dataset catalog is unavailable",
        ) from exc


def _memory_store():
    return default_memory_store()


def _feedback_store():
    return default_feedback_store()


def _turn_service(
    gateway: DataGateway = Depends(_gateway),
    store=Depends(_memory_store),
) -> AssistantTurnService:
    return AssistantTurnService(
        gateway,
        store=store,
        reasoner=get_reasoning_provider(),
    )


def _serialize_turn(result: TurnResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "answer": result.answer,
        "planner_type": result.planner_type,
        "reasoning_calls": result.reasoning_calls,
        "facts": result.facts,
        "rows": result.rows,
        "list_results": result.list_results,
        "detail": result.detail,
        "plan": result.plan.model_dump(mode="json") if result.plan else None,
        "evidence": result.evidence,
        "methods_run": result.methods_run,
        "parallel_step_peak": result.parallel_step_peak,
        "critical_path_ms": result.critical_path_ms,
        "conversation_id": result.conversation_id,
        "selected_file_id": result.selected_file_id,
        "suggested_file_id": result.suggested_file_id,
        "error_code": result.error_code,
        "citations": result.citations,
        "inference": result.inference,
        # Voice clients should speak this, not `answer`: it is citation-free and
        # length-bounded, which is the difference between a 3s and a 12s reply.
        "spoken_text": result.spoken_text,
        "trace": result.trace,
        "debug_trace": result.trace,
        "planner_attempts": result.planner_attempts,
        "planner_retry_count": result.planner_retry_count,
        "review_attempts": result.review_attempts,
        "review_status": result.review_status,
        "planner_total_model_calls": result.planner_total_model_calls,
        "planner_failure_stage": result.planner_failure_stage,
        "planner_provider": result.planner_provider,
        "planner_model": result.planner_model,
        "planner_usage": result.planner_usage,
    }


@router.get("/conversations")
async def list_conversations(
    scope: AccessScope = Depends(get_access_scope),
    store=Depends(_memory_store),
    gateway: DataGateway = Depends(_gateway),
) -> dict[str, Any]:
    records, catalog = await asyncio.gather(
        store.list_for_principal(scope),
        gateway.load_catalog(scope),
    )
    return {
        "conversations": [
            {
                "id": item.id,
                "title": item.title,
                "active_frame_key": item.active_frame_key,
                "selected_file_id": item.selected_file_id,
                "selected_dataset_label": _dataset_label(catalog, item.selected_file_id),
                "legacy_unscoped": item.selected_file_id is None,
                "preview": (item.recent_turns[-1]["text"] if item.recent_turns else ""),
                "turn_count": len(item.recent_turns),
            }
            for item in records
        ]
    }


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    scope: AccessScope = Depends(get_access_scope),
    store=Depends(_memory_store),
    gateway: DataGateway = Depends(_gateway),
) -> dict[str, Any]:
    try:
        record = await store.get(scope, conversation_id)
    except PermissionError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="conversation not found") from exc
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="conversation not found")
    catalog = await gateway.load_catalog(scope)
    return {
        "id": record.id,
        "title": record.title,
        "active_frame_key": record.active_frame_key,
        "selected_file_id": record.selected_file_id,
        "selected_dataset_label": _dataset_label(catalog, record.selected_file_id),
        "legacy_unscoped": record.selected_file_id is None,
        "rolling_summary": record.rolling_summary,
        "turns": [_public_turn(turn) for turn in record.recent_turns],
    }


class RenameConversationRequest(BaseModel):
    # Auto-titles are the first question truncated to 80; a chosen one gets a little more
    # room, and a bound so the sidebar cannot be filled with a pasted essay.
    title: str = Field(min_length=1, max_length=120)


@router.patch("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def rename_conversation(
    conversation_id: str,
    body: RenameConversationRequest,
    scope: AccessScope = Depends(get_access_scope),
    store=Depends(_memory_store),
) -> Response:
    """Retitle one of this principal's conversations.

    404 for anything that is not theirs, the same as delete: a different status would say
    which ids are real. Only the title moves; the turns underneath it are untouched.
    """
    title = body.title.strip()
    if not title:
        raise HTTPException(_UNPROCESSABLE, detail="title is required")
    try:
        renamed = await store.rename(scope, conversation_id, title)
    except PermissionError:
        renamed = False
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="conversation not found") from exc
    if not renamed:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="conversation not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: str,
    scope: AccessScope = Depends(get_access_scope),
    store=Depends(_memory_store),
) -> Response:
    """Forget one of this principal's conversations.

    A conversation that belongs to someone else answers 404, the same as one that never
    existed: a different status would let anyone probe which ids are real. The audit trail
    is not part of what is forgotten, and the deletion is itself recorded -- the sidebar is
    the researcher's to tidy, the record of what was asked is not theirs to erase.
    """
    try:
        deleted = await store.delete(scope, conversation_id)
    except PermissionError:
        deleted = False
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="conversation not found") from exc
    if not deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="conversation not found")
    await default_auditor().record(
        event_type="conversation_deleted",
        principal_id=scope.principal_id,
        authorization_outcome="allowed",
        conversation_id=conversation_id,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


class FeedbackAttachmentRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=200)
    content_type: str = Field(min_length=1, max_length=100)
    # Base64 is a third larger than the file it carries; the decoded size is checked after.
    data_base64: str = Field(min_length=1, max_length=(MAX_ATTACHMENTS_TOTAL_BYTES * 4) // 3 + 8)


class FeedbackRequest(BaseModel):
    # Who filed it is required: a report nobody can be asked about cannot be followed up. What
    # they saw is optional -- the chat is attached, and often a screenshot says it better.
    comment: str = Field(default="", max_length=MAX_COMMENT)
    reporter_name: str = Field(min_length=1, max_length=MAX_NAME)
    conversation_id: str | None = None
    app_version: str = Field(default="", max_length=64)
    # Minutes to add to UTC to get the reporter's clock. The title carries a time someone
    # will read next to their own memory of filing it, so it should say what their clock
    # said. Bounded to the range real offsets occupy; absent means UTC.
    tz_offset_minutes: int | None = Field(default=None, ge=-840, le=840)
    # A screenshot of what went wrong says more than a description of it.
    attachments: list[FeedbackAttachmentRequest] = Field(
        default_factory=list, max_length=MAX_ATTACHMENTS
    )


def reported_title(reporter_name: str, filed_at: datetime) -> str:
    """What a reported conversation is called afterwards.

    Named so the sidebar sorts the complaints to hand and says who to ask about each one.
    The endpoint refuses a report without a name; a blank one here still yields the marker
    and the time rather than an empty slot in the title.
    """
    stamp = filed_at.strftime("%d/%m %H:%M")
    name = " ".join(reporter_name.split())
    return f"Reported - {name} - {stamp}" if name else f"Reported - {stamp}"


@router.post("/feedback", status_code=status.HTTP_204_NO_CONTENT)
async def submit_feedback(
    body: FeedbackRequest,
    scope: AccessScope = Depends(get_access_scope),
    sink=Depends(_feedback_store),
    store=Depends(_memory_store),
) -> Response:
    """A tester's report: what they saw, and who saw it.

    Signed in, because an anonymous write endpoint on a records system is a place to dump
    text. The name is whatever the tester typed and is never treated as identity -- the
    principal is recorded separately, from the session.

    The conversation is renamed to mark it, so whoever reads the report can find the chat it
    is about in a sidebar of questions that all look alike. Renaming is best-effort: the
    report is the thing worth keeping, and losing it because a title could not be written
    would be the wrong way round.
    """
    reporter_name = body.reporter_name.strip()
    if not reporter_name:
        raise HTTPException(_UNPROCESSABLE, detail="name is required")
    comment = body.comment.strip()
    try:
        attachments = tuple(
            decode_attachment(item.filename, item.content_type, item.data_base64)
            for item in body.attachments
        )
    except ValueError as exc:
        raise HTTPException(_UNPROCESSABLE, detail=str(exc)) from exc
    if sum(len(item.content) for item in attachments) > MAX_ATTACHMENTS_TOTAL_BYTES:
        raise HTTPException(
            _UNPROCESSABLE,
            detail="Those files are too large together. Attach fewer or smaller ones.",
        )
    await sink.record(
        scope,
        FeedbackReport(
            comment=comment,
            reporter_name=reporter_name,
            conversation_id=body.conversation_id,
            app_version=body.app_version.strip(),
            attachments=attachments,
        ),
    )
    if body.conversation_id:
        filed_at = datetime.now(UTC) + timedelta(minutes=body.tz_offset_minutes or 0)
        with suppress(Exception):
            # rename answers False for a conversation that is not this principal's, which is
            # the same silence a wrong id gets. Nothing here tells the caller either way.
            await store.rename(
                scope, body.conversation_id, reported_title(reporter_name, filed_at)
            )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/datasets")
async def list_datasets(
    scope: AccessScope = Depends(get_access_scope),
    gateway: DataGateway = Depends(_gateway),
) -> dict[str, Any]:
    try:
        datasets = await gateway.list_datasets(scope)
        field_groups = await asyncio.gather(
            *(gateway.get_dataset_fields(scope, int(item["file_id"])) for item in datasets)
        )
    except Exception as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="dataset catalog is unavailable",
        ) from exc
    enriched = []
    for dataset, fields in zip(datasets, field_groups, strict=False):
        item = dict(dataset)
        item["fields"] = fields
        item["supported_operations"] = _supported_operations(fields)
        item["starter_prompts"] = _starter_prompts(item, fields)
        enriched.append(item)
    return {
        "datasets": enriched,
        "scope_fingerprint": scope.fingerprint(),
        # The models this researcher may pick between, default first. Served from the
        # authenticated capability endpoint rather than /health, which is public.
        "models": [
            {"id": name, "selected": name == get_settings().reasoning_model}
            for name in allowed_models()
        ],
    }


@router.post("/query")
async def query(
    body: QueryRequest,
    request: Request,
    scope: AccessScope = Depends(get_access_scope),
    service: AssistantTurnService = Depends(_turn_service),
):
    ensure_no_tts_in_text_mode(body.mode)
    accept = request.headers.get("accept", "")
    if "text/event-stream" in accept:
        return StreamingResponse(
            _stream_query(body, scope, service),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    result = await service.answer(
        scope,
        body.question,
        mode=body.mode,
        conversation_id=body.conversation_id,
        selected_file_id=body.selected_file_id,
        model=body.model,
    )
    return _serialize_turn(result)


async def _stream_query(
    body: QueryRequest,
    scope: AccessScope,
    service: AssistantTurnService,
) -> AsyncIterator[str]:
    """Start the response before planning and cancel unused work on disconnect."""
    cancellation = CancellationToken()
    task: asyncio.Task[TurnResult] | None = None
    progress_item: asyncio.Task[str] | None = None
    progress: asyncio.Queue[str] = asyncio.Queue()
    emitted: set[str] = set()

    async def on_progress(event: str, data: dict[str, Any]) -> None:
        emitted.add(event)
        await progress.put(_sse(event, data))

    yield _sse(
        "plan_started",
        {
            "planner_type": "pending",
            "conversation_id": body.conversation_id,
            "mode": body.mode,
            "selected_file_id": body.selected_file_id,
        },
    )
    try:
        task = asyncio.create_task(
            service.answer(
                scope,
                body.question,
                mode=body.mode,
                conversation_id=body.conversation_id,
                selected_file_id=body.selected_file_id,
        model=body.model,
                cancellation=cancellation,
                on_progress=on_progress,
            )
        )
        while not task.done() or not progress.empty():
            progress_item = asyncio.create_task(progress.get())
            done, _pending = await asyncio.wait(
                {task, progress_item},
                timeout=_SSE_HEARTBEAT_SECONDS,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if progress_item in done:
                yield progress_item.result()
                progress_item = None
                continue
            progress_item.cancel()
            with suppress(asyncio.CancelledError):
                await progress_item
            progress_item = None
            if task in done:
                break
            # SSE comments keep proxies and browsers from treating a long model
            # or database call as an idle connection. Clients intentionally ignore it.
            yield ": keep-alive\n\n"
        result = await task
        async for event in _as_sse(
            result,
            include_plan_started=False,
            skip_events=emitted,
        ):
            yield event
    finally:
        if progress_item is not None and not progress_item.done():
            progress_item.cancel()
            with suppress(asyncio.CancelledError):
                await progress_item
        if task is not None and not task.done():
            cancellation.cancel("client_disconnected")
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


async def _as_sse(
    result: TurnResult,
    *,
    include_plan_started: bool = True,
    skip_events: set[str] | None = None,
) -> AsyncIterator[str]:
    skipped = skip_events or set()
    if include_plan_started:
        yield _sse(
            "plan_started",
            {
                "planner_type": result.planner_type or "none",
                "conversation_id": result.conversation_id,
                "mode": "text",
            },
        )
    if result.plan and "plan_validated" not in skipped:
        yield _sse(
            "plan_validated",
            {
                "ops": result.plan.op_names(),
                "file_ids": list(result.plan.scope.file_ids),
                "reasoning_calls": result.reasoning_calls,
            },
        )
    if (result.methods_run or result.evidence) and "retrieval_completed" not in skipped:
        yield _sse(
            "retrieval_completed",
            {"methods_run": result.methods_run, "evidence_count": len(result.evidence)},
        )
    if "computed_fact" not in skipped:
        for fact in result.facts:
            yield _sse("computed_fact", fact)
    if "answer_delta" not in skipped:
        for chunk in _answer_chunks(result.answer):
            yield _sse("answer_delta", {"text": chunk})
    for citation in result.citations:
        item = next((row for row in result.evidence if row.get("source_id") == citation), None)
        yield _sse("citation", {"source_id": citation, "fields": (item or {}).get("fields") or {}})
    if result.conversation_id:
        yield _sse(
            "memory_update",
            {
                "conversation_id": result.conversation_id,
                "selected_file_id": result.selected_file_id,
                "citations": result.citations,
            },
        )
    yield _sse("answer_completed", _serialize_turn(result))


def _answer_chunks(text: str) -> list[str]:
    if not text:
        return [""]
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if len(sentences) > 1:
        return [sentence if sentence.endswith((".", "!", "?")) else f"{sentence} " for sentence in sentences]
    words = text.split(" ")
    chunks: list[str] = []
    for index in range(0, len(words), 8):
        piece = " ".join(words[index : index + 8])
        if index + 8 < len(words):
            piece += " "
        chunks.append(piece)
    return chunks or [text]


def _public_turn(turn: dict[str, Any]) -> dict[str, Any]:
    citations = turn.get("citations") or []
    if isinstance(citations, str):
        citations = [item for item in citations.split(",") if item]
    return {
        "role": turn.get("role"),
        "text": turn.get("text") or "",
        "mode": turn.get("mode") or "text",
        "interrupted": bool(turn.get("interrupted")),
        "citations": citations,
    }


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _dataset_label(catalog, file_id: int | None) -> str | None:
    if file_id is None:
        return None
    dataset = catalog.dataset(file_id)
    return dataset.user_facing_label if dataset is not None else None


def _supported_operations(fields: list[dict[str, Any]]) -> list[str]:
    operations = {
        "describe",
        "count",
        "list",
        "distinct",
        "search",
        "sort",
        "paginate",
        "project_fields",
    }
    for field in fields:
        operators = set(field.get("allowed_operators") or ())
        if field.get("aggregatable"):
            operations.update({"group", "minimum", "maximum"})
        if {"IS_UNKNOWN", "IS_KNOWN"} & operators:
            operations.add("missing_values")
        if {"BEFORE", "AFTER", "DATE_RANGE"} & operators:
            operations.add("ranges")
    return sorted(operations)


def _starter_prompts(
    dataset: dict[str, Any],
    fields: list[dict[str, Any]],
) -> list[dict[str, str]]:
    label = str(dataset.get("user_facing_label") or "selected list")
    semantics = {str(item.get("semantic_field")) for item in fields}
    prompts = [
        {
            "title": "Explore this list",
            "caption": f"See the fields available in {label}",
            "prompt": "What information is available in this list?",
        },
        {
            "title": "Count records",
            "caption": f"Count the records in {label}",
            "prompt": "How many records are in this list?",
        },
    ]
    if "cause_of_death" in semantics:
        prompts.extend(
            [
                {
                    "title": "Causes of death",
                    "caption": "List distinct recorded causes",
                    "prompt": "What different causes of death are recorded?",
                },
                {
                    "title": "Death years",
                    "caption": "Find the year with the most deaths",
                    "prompt": "Which year has the most confirmed deaths?",
                },
            ]
        )
    else:
        if "community" in semantics:
            prompts.append(
                {
                    "title": "Browse communities",
                    "caption": "List distinct communities",
                    "prompt": "List the distinct communities in this list.",
                }
            )
        prompts.append(
            {
                "title": "Browse students",
                "caption": "Start with the first records",
                "prompt": "List the first 25 students and their recorded details.",
            }
        )
    return prompts[:4]
