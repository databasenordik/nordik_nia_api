from __future__ import annotations

from typing import Protocol

from app.memory.types import ConversationRecord, new_conversation_id
from app.security.access_scope import AccessScope


class MemoryStore(Protocol):
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

    async def get_or_create(self, scope: AccessScope, conversation_id: str | None) -> ConversationRecord:
        if conversation_id and conversation_id in self._records:
            record = self._records[conversation_id]
            if record.principal_id != scope.principal_id:
                raise PermissionError("conversation is not in this access scope")
            return record
        record = ConversationRecord(id=conversation_id or new_conversation_id(), principal_id=scope.principal_id)
        self._records[record.id] = record
        return record

    async def save(self, record: ConversationRecord) -> None:
        self._records[record.id] = record

    async def list_for_principal(self, scope: AccessScope) -> list[ConversationRecord]:
        return [item for item in self._records.values() if item.principal_id == scope.principal_id]

    async def get(self, scope: AccessScope, conversation_id: str) -> ConversationRecord | None:
        record = self._records.get(conversation_id)
        if record is None:
            return None
        if record.principal_id != scope.principal_id:
            raise PermissionError("conversation is not in this access scope")
        return record

    async def delete(self, scope: AccessScope, conversation_id: str) -> bool:
        """Forget one conversation. False when there was nothing of this principal's to forget."""
        record = self._records.get(conversation_id)
        if record is None:
            return False
        if record.principal_id != scope.principal_id:
            raise PermissionError("conversation is not in this access scope")
        del self._records[conversation_id]
        return True

    async def rename(self, scope: AccessScope, conversation_id: str, title: str) -> bool:
        record = self._records.get(conversation_id)
        if record is None:
            return False
        if record.principal_id != scope.principal_id:
            raise PermissionError("conversation is not in this access scope")
        record.title = title
        return True


_DEFAULT_STORE = InMemoryMemoryStore()


def default_memory_store() -> MemoryStore:
    """Prefer durable Postgres so text and voice workers share one thread."""
    try:
        from app.db.pool import get_pool
        from app.memory.postgres import PostgresMemoryStore

        return PostgresMemoryStore(get_pool())
    except Exception:
        return _DEFAULT_STORE
