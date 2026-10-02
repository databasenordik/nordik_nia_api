from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.planning.plan_schema import Predicate
from app.planning.turn_schema import QueryAction, ResultWindow


class QueryState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_ids: tuple[int, ...]
    version_mode: str = "current"
    filters: list[Predicate] = Field(default_factory=list)
    goal: str = "exact_count"
    retrieval_query: str | None = None
    limit: int | None = None
    offset: int | None = None
    exhaustive: bool = False
    presentation: str | None = None
    sort_by: str | None = None
    sort_direction: str | None = None
    sample: bool = False
    requested_fields: list[str] = Field(default_factory=list)
    window: ResultWindow | None = None
    total_count: int | None = None
    returned_count: int | None = None
    has_more: bool = False
    action_id: str | None = None
    source_action: QueryAction | None = None


class ResultSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    file_ids: tuple[int, ...]
    ordered_source_ids: list[str] = Field(default_factory=list)
    source_turn_id: str
    record_type: str = "research_rows"
    action_id: str | None = None
    total_count: int | None = None
    returned_count: int = 0
    offset: int = 0
    limit: int | None = None
    has_more: bool = False
    sort_by: str | None = None
    sort_direction: str | None = None
    sample: bool = False
    requested_fields: list[str] = Field(default_factory=list)
    window: ResultWindow | None = None


class QueryFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_key: str
    topic: str
    query: QueryState
    result_set_id: str | None = None
    last_source_ids: list[str] = Field(default_factory=list)
    action_id: str | None = None
    # The words that produced this result. A researcher who pastes back what the answer
    # should have been is correcting this question, and it is asked again against the list.
    question: str | None = Field(default=None, max_length=4000)


class ConversationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    principal_id: str
    # A conversation is permanently bound to one authorized dataset.  Null is
    # retained only for conversations created before dataset scoping existed.
    selected_file_id: int | None = None
    title: str | None = None
    active_frame_key: str | None = None
    frames: dict[str, QueryFrame] = Field(default_factory=dict)
    result_sets: dict[str, ResultSet] = Field(default_factory=dict)
    recent_turns: list[dict[str, Any]] = Field(default_factory=list)
    rolling_summary: str = ""
    last_action_keys: list[str] = Field(default_factory=list)
    user_display_name: str | None = None
    message_sequence: int = Field(default=0, ge=0)

    def active_frame(self) -> QueryFrame | None:
        if not self.active_frame_key:
            return None
        return self.frames.get(self.active_frame_key)


def new_conversation_id() -> str:
    return str(uuid4())


def frame_key_for(file_ids: tuple[int, ...]) -> str:
    if file_ids == (49,):
        return "master"
    if file_ids == (91,):
        return "confirmed"
    if file_ids == (93,):
        return "additional"
    if file_ids == (94,):
        return "potential"
    return "files-" + "-".join(str(item) for item in file_ids)


def parse_source_id(source_id: str) -> tuple[int, int] | None:
    # file:49:v3:row:1102
    parts = source_id.split(":")
    if len(parts) != 5 or parts[0] != "file" or parts[3] != "row":
        return None
    try:
        return int(parts[1]), int(parts[4])
    except ValueError:
        return None


def make_source_id(file_id: int, version: int, source_row_id: int) -> str:
    return f"file:{file_id}:v{version}:row:{source_row_id}"
