from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PlannedPredicate(BaseModel):
    field: str
    operator: str
    value: Any = None


class PlannedStep(BaseModel):
    id: str
    op: str
    input: str | list[str] | None = "current_records"
    where: list[PlannedPredicate] = Field(default_factory=list)
    fields: list[str] = Field(default_factory=list)
    query: str | None = None
    limit: int | None = None
    file_ids: list[int] | None = None


class PlannedQuery(BaseModel):
    """Constrained planner output. No SQL and no application table names."""

    file_ids: list[int]
    goals: list[str] = Field(default_factory=list)
    steps: list[PlannedStep]
    needs_synthesis: bool = False


class MemoryUpdate(BaseModel):
    active_file_ids: list[int] = Field(default_factory=list)
    active_topics: list[str] = Field(default_factory=list)
    last_source_ids: list[str] = Field(default_factory=list)


class SynthesisAnswer(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    inference: bool = False
    memory_update: MemoryUpdate = Field(default_factory=MemoryUpdate)
