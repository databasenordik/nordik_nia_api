from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AccessScope(BaseModel):
    """Immutable authorization projection consumed by Assistant Core.

    Adapters produce this object. Planning, retrieval, the data gateway,
    memory, citations, LiveKit, whisper, xAI, and Kokoro must accept this
    type and must not import adapter-specific session or JWT types.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    principal_id: str = Field(min_length=1)
    allowed_file_ids: tuple[int, ...]
    can_use_private_files: bool = False
    history_allowed: bool = False
    display_name: str | None = None

    @field_validator("allowed_file_ids", mode="before")
    @classmethod
    def _tuple_ids(cls, value: Any) -> tuple[int, ...]:
        if value is None:
            return ()
        return tuple(int(item) for item in value)

    def allows_file(self, file_id: int, *, private: bool = False) -> bool:
        if file_id not in self.allowed_file_ids:
            return False
        if private and not self.can_use_private_files:
            return False
        return True

    def narrow(self, requested_file_ids: list[int] | tuple[int, ...] | None) -> "AccessScope":
        """Return a scope that is never broader than this one."""
        if not requested_file_ids:
            return self
        narrowed = tuple(file_id for file_id in requested_file_ids if file_id in self.allowed_file_ids)
        return AccessScope(
            principal_id=self.principal_id,
            allowed_file_ids=narrowed,
            can_use_private_files=self.can_use_private_files,
            history_allowed=self.history_allowed,
            display_name=self.display_name,
        )

    def fingerprint(self) -> str:
        ids = ",".join(str(item) for item in sorted(self.allowed_file_ids))
        return (
            f"{self.principal_id}|files={ids}|private={int(self.can_use_private_files)}"
            f"|history={int(self.history_allowed)}"
        )
