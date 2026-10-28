from __future__ import annotations

from typing import Protocol

from app.memory.types import ConversationRecord, new_conversation_id
from app.security.access_scope import AccessScope


class MemoryStore(Protocol):
    async def for_user_file(self, scope: AccessScope, file_id: int, conversation_id: str | None = None) -> ConversationRecord: ...

    async def history_for_file(self, scope: AccessScope, file_id: int, conversation_id: str | None, limit: int) -> dict: ...

    async def get_or_create(self, scope: AccessScope, conversation_id: str | None) -> ConversationRecord: ...

    async def save(self, record: ConversationRecord) -> None: ...

    async def list_for_principal(self, scope: AccessScope) -> list[ConversationRecord]: ...

    async def get(self, scope: AccessScope, conversation_id: str) -> ConversationRecord | None: ...

    async def delete(self, scope: AccessScope, conversation_id: str) -> bool: ...

    async def rename(self, scope: AccessScope, conversation_id: str, title: str) -> bool: ...


class InMemoryMemoryStore:
    """Process-local store. Same AccessScope principal_id isolation as Postgres."""

    def __init__(self) -> None:
        self._records: dict[str, ConversationRecord] = {}
        self._messages: dict[str, dict[str, dict]] = {}

    async def for_user_file(self, scope: AccessScope, file_id: int, conversation_id: str | None = None) -> ConversationRecord:
        if scope.user_id is None or file_id not in scope.allowed_file_ids:
            raise PermissionError("user and authorized file are required")
        if conversation_id:
            record = await self.get(scope, conversation_id)
            if record is None or record.selected_file_id != file_id:
                raise PermissionError("conversation not found")
            return record
        for record in self._records.values():
            if (record.principal_id, record.user_id, record.selected_file_id) == (scope.principal_id, scope.user_id, file_id):
                return record
        record = ConversationRecord(id=new_conversation_id(), principal_id=scope.principal_id,
                                    user_id=scope.user_id, selected_file_id=file_id)
        await self.save(record)
        return record

    async def history_for_file(self, scope: AccessScope, file_id: int, conversation_id: str | None, limit: int) -> dict:
        records = await self.list_for_principal(scope)
        record = next((r for r in records if r.selected_file_id == file_id and (not conversation_id or r.id == conversation_id)), None)
        if record is None and conversation_id:
            raise PermissionError("conversation not found")
        pairs = history_pairs(list(self._messages.get(record.id, {}).values())) if record else []
        return {"conversation_id": record.id if record else None, "file_id": file_id,
                "entries": pairs[-limit:], "has_more": len(pairs) > limit}

    async def get_or_create(self, scope: AccessScope, conversation_id: str | None) -> ConversationRecord:
        if conversation_id and conversation_id in self._records:
            record = self._records[conversation_id]
            if record.principal_id != scope.principal_id or record.user_id != scope.user_id:
                raise PermissionError("conversation is not in this access scope")
            return record
        record = ConversationRecord(id=conversation_id or new_conversation_id(), principal_id=scope.principal_id, user_id=scope.user_id)
        self._records[record.id] = record
        return record

    async def save(self, record: ConversationRecord) -> None:
        from app.memory.postgres import _prepare_turns

        _prepare_turns(record)
        messages = self._messages.setdefault(record.id, {})
        for turn in record.recent_turns:
            messages[turn["message_id"]] = {**turn, "row_access_revision": record.row_access_revision}
        self._records[record.id] = record

    async def list_for_principal(self, scope: AccessScope) -> list[ConversationRecord]:
        return [item for item in self._records.values() if item.principal_id == scope.principal_id and item.user_id == scope.user_id]

    async def get(self, scope: AccessScope, conversation_id: str) -> ConversationRecord | None:
        record = self._records.get(conversation_id)
        if record is None:
            return None
        if record.principal_id != scope.principal_id or record.user_id != scope.user_id:
            raise PermissionError("conversation is not in this access scope")
        return record

    async def delete(self, scope: AccessScope, conversation_id: str) -> bool:
        """Forget one conversation. False when there was nothing of this principal's to forget."""
        record = self._records.get(conversation_id)
        if record is None:
            return False
        if record.principal_id != scope.principal_id or record.user_id != scope.user_id:
            raise PermissionError("conversation is not in this access scope")
        del self._records[conversation_id]
        self._messages.pop(conversation_id, None)
        return True

    async def rename(self, scope: AccessScope, conversation_id: str, title: str) -> bool:
        record = self._records.get(conversation_id)
        if record is None:
            return False
        if record.principal_id != scope.principal_id or record.user_id != scope.user_id:
            raise PermissionError("conversation is not in this access scope")
        record.title = title
        return True


_DEFAULT_STORE = InMemoryMemoryStore()


def history_pairs(turns: list[dict]) -> list[dict]:
    pairs = []
    questions = {}
    for turn in turns:
        exchange_id = turn.get("exchange_id")
        if turn.get("role") == "user":
            questions[exchange_id] = turn
        elif turn.get("role") == "assistant" and exchange_id in questions:
            question = questions.pop(exchange_id)
            pairs.append({"id": exchange_id, "exchange_id": exchange_id,
                          "question_id": question.get("message_id"), "answer_id": turn.get("message_id"),
                          "question": question.get("text", ""), "answer": turn.get("text", ""),
                          "created_at": question.get("created_at"), "answered_at": turn.get("created_at"),
                          "communities": question.get("communities") or []})
    return pairs


def default_memory_store() -> MemoryStore:
    """Prefer durable Postgres so text and voice workers share one thread."""
    try:
        from app.db.pool import get_pool
        from app.memory.postgres import PostgresMemoryStore

        return PostgresMemoryStore(get_pool())
    except Exception:
        return _DEFAULT_STORE
