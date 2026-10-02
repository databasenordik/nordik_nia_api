from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RetrievalMethod = Literal["exact", "fts", "trgm", "fuzzy", "alias"]


class RetrievalHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    file_id: int
    version: int
    source_row_id: int
    canonical_name: str | None = None
    canonical_community: str | None = None
    canonical_school: str | None = None
    method: RetrievalMethod
    score: float
    matched_field: str | None = None
    snippet: str | None = None
    record: dict[str, Any] = Field(default_factory=dict)

    def source_id(self) -> str:
        return f"file:{self.file_id}:v{self.version}:row:{self.source_row_id}"

    def identity(self) -> tuple[int, int]:
        return (self.file_id, self.source_row_id)


class EvidenceItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str
    file_id: int
    version: int
    source_row_id: int
    fields: dict[str, Any]
    raw_fields: dict[str, Any] = Field(default_factory=dict)


class EvidenceBranch(BaseModel):
    """Evidence owned by one TurnPlan action. Do not merge across branches."""

    model_config = ConfigDict(frozen=True)

    action_id: str
    goal: str
    facts: tuple[str, ...] = ()
    items: tuple[EvidenceItem, ...] = ()


class EvidencePacket(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str
    facts: tuple[str, ...] = ()
    items: tuple[EvidenceItem, ...] = ()
    token_estimate: int = 0
    truncated: bool = False
    action_id: str | None = None
    branches: tuple[EvidenceBranch, ...] = ()
