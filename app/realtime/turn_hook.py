from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from app.concurrency.cancellation import CancellationToken
from app.execution.turn import AssistantTurnService, TurnResult
from app.security.access_scope import AccessScope

SpokenPhraseFn = Callable[[str], Awaitable[None]]


@dataclass
class TurnHookResult:
    reply: str
    core: TurnResult
    retrieval_completed: bool = True


@dataclass
class AssistantTurnHook:
    """Runs Assistant Core after a finalized user turn and before any spoken reply."""

    turn_service: AssistantTurnService
    started_replies: list[str] = field(default_factory=list)

    async def on_user_turn_completed(
        self,
        transcript: str,
        scope: AccessScope,
        *,
        conversation_id: str | None = None,
        selected_file_id: int | None = None,
        cancellation: CancellationToken | None = None,
        on_spoken_phrase: SpokenPhraseFn | None = None,
    ) -> TurnHookResult:
        if not transcript.strip():
            return TurnHookResult(reply="", core=TurnResult(status="empty", answer=""), retrieval_completed=False)
        result = await self.turn_service.answer(
            scope,
            transcript,
            mode="voice",
            conversation_id=conversation_id,
            selected_file_id=selected_file_id,
            cancellation=cancellation,
            on_spoken_phrase=on_spoken_phrase,
        )
        self.started_replies.append(result.answer)
        return TurnHookResult(reply=result.answer, core=result, retrieval_completed=True)
