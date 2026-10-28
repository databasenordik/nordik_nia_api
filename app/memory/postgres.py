from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import asyncpg

from app.memory.types import ConversationRecord, QueryFrame, ResultSet
from app.security.access_scope import AccessScope


class PostgresMemoryStore:
    """Durable assistant-owned conversation state shared by text and voice."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def for_user_file(self, scope: AccessScope, file_id: int, conversation_id: str | None = None) -> ConversationRecord:
        if scope.user_id is None or file_id not in scope.allowed_file_ids:
            raise PermissionError("user and authorized file are required")
        if not conversation_id:
            async with self._pool.acquire() as conn:
                conversation_id = await conn.fetchval(
                    """
                    INSERT INTO assistant.conversations (principal_id, user_id, selected_file_id)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (user_id, file_id) WHERE user_id IS NOT NULL
                    DO UPDATE SET user_id = EXCLUDED.user_id
                    WHERE assistant.conversations.principal_id = EXCLUDED.principal_id
                    RETURNING id::text
                    """, scope.principal_id, scope.user_id, file_id,
                )
            if conversation_id is None:
                raise PermissionError("conversation not found")
        record = await self.get(scope, conversation_id)
        if record is None or record.selected_file_id != file_id:
            raise PermissionError("conversation not found")
        return record

    async def history_for_file(self, scope: AccessScope, file_id: int, conversation_id: str | None, limit: int) -> dict:
        # Read persisted messages, not the 12-message model context snapshot.
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT id::text AS id FROM assistant.conversations
                   WHERE principal_id = $1 AND user_id = $2 AND file_id = $3
                     AND ($4::uuid IS NULL OR id = $4::uuid)""",
                scope.principal_id, scope.user_id, file_id,
                _validated_id(conversation_id) if conversation_id else None,
            )
            if row is None:
                if conversation_id:
                    raise PermissionError("conversation not found")
                return {"conversation_id": None, "file_id": file_id, "entries": [], "has_more": False}
            rows = await conn.fetch(
                """
                SELECT q.exchange_id::text AS id, q.exchange_id::text AS exchange_id,
                       q.id::text AS question_id, a.id::text AS answer_id,
                       q.text AS question, a.text AS answer, q.created_at,
                       a.created_at AS answered_at, q.communities
                FROM assistant.messages q
                JOIN assistant.messages a ON a.conversation_id = q.conversation_id
                    AND a.exchange_id = q.exchange_id AND a.role = 'assistant'
                WHERE q.conversation_id = $1::uuid AND q.role = 'user'
                  AND ($2::integer <> 94 OR a.row_access_revision = assistant_api.potential_access_revision())
                ORDER BY CASE WHEN a.turn_id ~ '^turn-[0-9]+$'
                              THEN substring(a.turn_id from '[0-9]+')::bigint ELSE 0 END DESC,
                         a.created_at DESC, a.id DESC
                LIMIT $3
                """, row["id"], file_id, limit + 1,
            )
        return {"conversation_id": row["id"], "file_id": file_id,
                "entries": [dict(r) for r in reversed(rows[:limit])], "has_more": len(rows) > limit}

    async def get_or_create(self, scope: AccessScope, conversation_id: str | None) -> ConversationRecord:
        if conversation_id:
            conversation_id = _validated_id(conversation_id)
            existing = await self.get(scope, conversation_id)
            if existing is not None:
                return existing
        record = ConversationRecord(
            id=conversation_id or _new_id(),
            principal_id=scope.principal_id,
            user_id=scope.user_id,
        )
        await self.save(record)
        return record

    async def save(self, record: ConversationRecord) -> None:
        _prepare_turns(record)
        async with self._pool.acquire() as conn:
            if record.selected_file_id == 94 and record.row_access_revision is None:
                record.row_access_revision = await conn.fetchval("SELECT assistant_api.potential_access_revision()")
            snapshot = _snapshot(record)
            async with conn.transaction():
                owner = await conn.fetchrow(
                    """
                    INSERT INTO assistant.conversations
                        (id, principal_id, selected_file_id, title, current_mode, user_id, updated_at)
                    VALUES ($1::uuid, $2, $3, $4, $5, $6, now())
                    ON CONFLICT (id) DO UPDATE SET
                        selected_file_id = COALESCE(
                            assistant.conversations.selected_file_id,
                            EXCLUDED.selected_file_id
                        ),
                        title = EXCLUDED.title,
                        current_mode = EXCLUDED.current_mode,
                        updated_at = now()
                    WHERE assistant.conversations.principal_id = EXCLUDED.principal_id
                      AND assistant.conversations.user_id IS NOT DISTINCT FROM EXCLUDED.user_id
                      AND (
                            assistant.conversations.selected_file_id IS NULL
                            OR EXCLUDED.selected_file_id IS NULL
                            OR assistant.conversations.selected_file_id = EXCLUDED.selected_file_id
                      )
                    RETURNING principal_id, selected_file_id
                    """,
                    _validated_id(record.id),
                    record.principal_id,
                    record.selected_file_id,
                    record.title,
                    _current_mode(record),
                    record.user_id,
                )
                # ON CONFLICT ... WHERE returns no row when the UUID is already
                # owned by another principal.  Check this before replacing any
                # messages so a guessed conversation UUID cannot cross tenants.
                if owner is None or owner["principal_id"] != record.principal_id:
                    raise PermissionError("conversation is not in this access scope")
                for turn in record.recent_turns:
                    await conn.execute(
                        """
                        INSERT INTO assistant.messages
                            (id, conversation_id, turn_id, role, text, mode, interrupted, created_at,
                             exchange_id, communities, row_access_revision)
                        VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8::timestamptz,
                                $9::uuid, $10::text[], $11)
                        ON CONFLICT (id) DO UPDATE SET
                            role = EXCLUDED.role,
                            text = EXCLUDED.text,
                            mode = EXCLUDED.mode,
                            interrupted = EXCLUDED.interrupted,
                            exchange_id = EXCLUDED.exchange_id,
                            communities = EXCLUDED.communities,
                            row_access_revision = EXCLUDED.row_access_revision
                        WHERE assistant.messages.conversation_id = EXCLUDED.conversation_id
                        """,
                        _validated_id(str(turn["message_id"])),
                        record.id,
                        str(turn["turn_id"]),
                        str(turn.get("role") or "assistant"),
                        str(turn.get("text") or ""),
                        str(turn.get("mode") or "text"),
                        bool(turn.get("interrupted")),
                        _as_datetime(turn["created_at"]),
                        _validated_id(turn["exchange_id"]),
                        list(turn.get("communities") or []),
                        record.row_access_revision,
                    )
                await conn.execute(
                    """
                    INSERT INTO assistant.conversation_state
                        (conversation_id, rolling_summary, active_query_state, last_source_ids, updated_at)
                    VALUES ($1::uuid, $2, $3::jsonb, $4::text[], now())
                    ON CONFLICT (conversation_id) DO UPDATE SET
                        rolling_summary = EXCLUDED.rolling_summary,
                        active_query_state = EXCLUDED.active_query_state,
                        last_source_ids = EXCLUDED.last_source_ids,
                        updated_at = now()
                    """,
                    record.id,
                    record.rolling_summary,
                    json.dumps(snapshot),
                    _last_source_ids(record),
                )

    async def list_for_principal(self, scope: AccessScope) -> list[ConversationRecord]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                WITH release AS MATERIALIZED (SELECT assistant_api.potential_access_revision() AS revision)
                SELECT c.id::text AS id, c.principal_id, c.user_id, c.selected_file_id,
                       c.title,
                       CASE WHEN c.selected_file_id = 94 AND s.active_query_state->>'row_access_revision' IS DISTINCT FROM release.revision
                            THEN '' ELSE s.rolling_summary END AS rolling_summary,
                       CASE WHEN c.selected_file_id = 94 AND s.active_query_state->>'row_access_revision' IS DISTINCT FROM release.revision
                            THEN s.active_query_state - ARRAY['rolling_summary','recent_turns','frames','result_sets','last_action_keys','active_frame_key']
                            ELSE s.active_query_state END AS active_query_state,
                       release.revision AS current_access_revision
                FROM assistant.conversations c
                CROSS JOIN release
                LEFT JOIN assistant.conversation_state s ON s.conversation_id = c.id
                WHERE c.principal_id = $1
                  AND c.user_id IS NOT DISTINCT FROM $2::bigint
                ORDER BY c.updated_at DESC
                """,
                scope.principal_id,
                scope.user_id,
            )
        return [_from_row(dict(row), scope.principal_id) for row in rows]

    async def get(self, scope: AccessScope, conversation_id: str) -> ConversationRecord | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                WITH release AS MATERIALIZED (SELECT assistant_api.potential_access_revision() AS revision)
                SELECT c.id::text AS id, c.principal_id, c.user_id, c.selected_file_id,
                       c.title,
                       CASE WHEN c.selected_file_id = 94 AND s.active_query_state->>'row_access_revision' IS DISTINCT FROM release.revision
                            THEN '' ELSE s.rolling_summary END AS rolling_summary,
                       CASE WHEN c.selected_file_id = 94 AND s.active_query_state->>'row_access_revision' IS DISTINCT FROM release.revision
                            THEN s.active_query_state - ARRAY['rolling_summary','recent_turns','frames','result_sets','last_action_keys','active_frame_key']
                            ELSE s.active_query_state END AS active_query_state,
                       messages.stored_messages, release.revision AS current_access_revision
                FROM assistant.conversations c
                CROSS JOIN release
                LEFT JOIN assistant.conversation_state s ON s.conversation_id = c.id
                LEFT JOIN LATERAL (
                    SELECT jsonb_agg(to_jsonb(recent) ORDER BY recent.sequence) AS stored_messages
                    FROM (
                        SELECT m.id::text AS message_id, m.exchange_id::text AS exchange_id,
                               m.communities, m.turn_id, m.role, m.text,
                               m.mode, m.interrupted, m.created_at,
                               CASE WHEN m.turn_id ~ '^turn-[0-9]+$'
                                    THEN substring(m.turn_id from '[0-9]+')::integer
                                    ELSE 0 END AS sequence
                        FROM assistant.messages m
                        WHERE m.conversation_id = c.id
                          AND (c.selected_file_id IS DISTINCT FROM 94 OR m.role = 'user'
                               OR m.row_access_revision = release.revision)
                          AND to_jsonb(m.communities) = COALESCE(s.active_query_state->'community_filter', '[]'::jsonb)
                        ORDER BY sequence DESC, m.created_at DESC, m.id DESC
                        LIMIT 12
                    ) recent
                ) messages ON true
                WHERE c.id = $1::uuid
                """,
                _validated_id(conversation_id),
            )
        if row is None:
            return None
        if row["principal_id"] != scope.principal_id or dict(row).get("user_id") != scope.user_id:
            raise PermissionError("conversation is not in this access scope")
        return _from_row(dict(row), scope.principal_id)

    async def delete(self, scope: AccessScope, conversation_id: str) -> bool:
        """Forget one conversation, with its state, messages, frames and result sets.

        The principal is part of the DELETE rather than a check before it, so there is no
        window between deciding and deleting, and a conversation belonging to someone else
        is indistinguishable from one that never existed. Every child table cascades;
        assistant.audit_events deliberately does not -- it records that a question was asked
        and answered, which a researcher tidying their sidebar does not get to erase.
        """
        async with self._pool.acquire() as conn:
            deleted = await conn.fetchval(
                """
                DELETE FROM assistant.conversations
                WHERE id = $1::uuid AND principal_id = $2 AND user_id IS NOT DISTINCT FROM $3::bigint
                RETURNING 1
                """,
                _validated_id(conversation_id),
                scope.principal_id,
                scope.user_id,
            )
        return deleted is not None

    async def rename(self, scope: AccessScope, conversation_id: str, title: str) -> bool:
        """Retitle one conversation, if it is this principal's.

        The principal is part of the UPDATE for the same reason it is part of the DELETE:
        someone else's conversation is indistinguishable from one that never existed, and
        there is no gap between deciding and writing. A title is only what the sidebar shows
        -- the turns underneath it are untouched.
        """
        async with self._pool.acquire() as conn:
            updated = await conn.fetchval(
                """
                UPDATE assistant.conversations
                SET title = $3, updated_at = now()
                WHERE id = $1::uuid AND principal_id = $2 AND user_id IS NOT DISTINCT FROM $4::bigint
                RETURNING 1
                """,
                _validated_id(conversation_id),
                scope.principal_id,
                title,
                scope.user_id,
            )
        return updated is not None


def snapshot_record(record: ConversationRecord) -> dict[str, Any]:
    return _snapshot(record)


def record_from_snapshot(payload: dict[str, Any], principal_id: str) -> ConversationRecord:
    return _from_snapshot(payload, principal_id)


def _new_id() -> str:
    from app.memory.types import new_conversation_id

    return new_conversation_id()


def _validated_id(value: str) -> str:
    try:
        return str(UUID(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("conversation_id must be a UUID") from exc


def _current_mode(record: ConversationRecord) -> str:
    if not record.recent_turns:
        return "text"
    return str(record.recent_turns[-1].get("mode") or "text")


def _last_source_ids(record: ConversationRecord) -> list[str]:
    frame = record.active_frame()
    return list(frame.last_source_ids) if frame else []


def _snapshot(record: ConversationRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "principal_id": record.principal_id,
        "user_id": record.user_id,
        "community_filter": list(record.community_filter),
        "selected_file_id": record.selected_file_id,
        "title": record.title,
        "active_frame_key": record.active_frame_key,
        "frames": {key: value.model_dump(mode="json") for key, value in record.frames.items()},
        "result_sets": {key: value.model_dump(mode="json") for key, value in record.result_sets.items()},
        "recent_turns": record.recent_turns,
        "rolling_summary": record.rolling_summary,
        "row_access_revision": record.row_access_revision,
        "last_action_keys": list(record.last_action_keys),
        "user_display_name": record.user_display_name,
        "message_sequence": record.message_sequence,
    }


def _from_row(row: dict[str, Any], principal_id: str) -> ConversationRecord:
    raw = row.get("active_query_state") or {}
    if isinstance(raw, str):
        raw = json.loads(raw)
    if raw:
        record = _from_snapshot(raw, principal_id)
        # The relational columns are authoritative. Older JSON snapshots do not contain the
        # dataset binding, and the title is edited in place by rename: a snapshot is written
        # when a turn is taken, so reading the title out of it meant a rename was stored and
        # then never shown -- the row said one thing and every endpoint said another.
        record.selected_file_id = row.get("selected_file_id") or record.selected_file_id
        record.user_id = row.get("user_id")
        record.title = row.get("title") or record.title
        if record.selected_file_id == 94:
            record.row_access_revision = row.get("current_access_revision")
        _enrich_turns(record, row.get("stored_messages"))
        return record
    return ConversationRecord(
        id=str(row["id"]),
        principal_id=principal_id,
        user_id=row.get("user_id"),
        selected_file_id=row.get("selected_file_id"),
        title=row.get("title"),
        rolling_summary=row.get("rolling_summary") or "",
        row_access_revision=row.get("current_access_revision") if row.get("selected_file_id") == 94 else None,
    )


def _from_snapshot(payload: dict[str, Any], principal_id: str) -> ConversationRecord:
    frames = {
        key: QueryFrame.model_validate(value) for key, value in (payload.get("frames") or {}).items()
    }
    result_sets = {
        key: ResultSet.model_validate(value) for key, value in (payload.get("result_sets") or {}).items()
    }
    return ConversationRecord(
        id=str(payload.get("id") or ""),
        # Ownership comes from the conversations row / validated caller, never
        # from a mutable JSON snapshot.
        principal_id=principal_id,
        selected_file_id=payload.get("selected_file_id"),
        community_filter=tuple(payload.get("community_filter") or ()),
        title=payload.get("title"),
        active_frame_key=payload.get("active_frame_key"),
        frames=frames,
        result_sets=result_sets,
        recent_turns=list(payload.get("recent_turns") or []),
        rolling_summary=payload.get("rolling_summary") or "",
        row_access_revision=payload.get("row_access_revision"),
        last_action_keys=list(payload.get("last_action_keys") or []),
        user_display_name=payload.get("user_display_name"),
        message_sequence=int(payload.get("message_sequence") or 0),
    )


def _prepare_turns(record: ConversationRecord) -> None:
    next_sequence = record.message_sequence
    question_exchange_id = None
    for turn in record.recent_turns:
        raw_turn_id = str(turn.get("turn_id") or "")
        if raw_turn_id.startswith("turn-"):
            try:
                next_sequence = max(next_sequence, int(raw_turn_id[5:]) + 1)
            except ValueError:
                pass
        if not turn.get("message_id"):
            turn["message_id"] = str(uuid4())
        if not turn.get("turn_id"):
            turn["turn_id"] = f"turn-{next_sequence}"
            next_sequence += 1
        if not turn.get("created_at"):
            turn["created_at"] = datetime.now(UTC).isoformat()
        if not turn.get("exchange_id"):
            turn["exchange_id"] = (
                question_exchange_id if turn.get("role") == "assistant" and question_exchange_id
                else turn["message_id"]
            )
        question_exchange_id = turn["exchange_id"] if turn.get("role") == "user" else None
    record.message_sequence = max(record.message_sequence, next_sequence)


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _enrich_turns(record: ConversationRecord, stored: Any) -> None:
    if not stored:
        _prepare_turns(record)
        return
    if isinstance(stored, str):
        stored = json.loads(stored)
    rows = list(stored or [])
    if len(rows) == len(record.recent_turns):
        for turn, row in zip(record.recent_turns, rows, strict=False):
            turn.setdefault("message_id", row.get("message_id"))
            turn.setdefault("exchange_id", row.get("exchange_id"))
            turn.setdefault("communities", row.get("communities") or [])
            turn.setdefault("turn_id", row.get("turn_id"))
            turn.setdefault("created_at", str(row.get("created_at")))
    _prepare_turns(record)
