from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.config import get_settings
from app.memory.store import default_memory_store
from app.security.access_scope import AccessScope
from app.security.deps import get_access_scope
from app.security.livekit_token import (
    AGENT_NAME,
    TOKEN_TTL_SECONDS,
    assigned_room_name,
    mint_livekit_token,
)

router = APIRouter(prefix="/api/livekit", tags=["livekit"])


class TokenRequest(BaseModel):
    conversation_id: str | None = None
    selected_file_id: int | None = None
    room_name: str | None = Field(default=None, max_length=128)


class TokenResponse(BaseModel):
    server_url: str
    room_name: str
    participant_identity: str
    participant_token: str
    selected_file_id: int
    agent_name: str = AGENT_NAME
    ttl_seconds: int = TOKEN_TTL_SECONDS


@router.post("/token", response_model=TokenResponse)
async def create_token(
    body: TokenRequest,
    scope: AccessScope = Depends(get_access_scope),
    store=Depends(default_memory_store),
) -> TokenResponse:
    settings = get_settings()
    selected_file_id = body.selected_file_id or settings.default_people_file_id
    if selected_file_id not in scope.allowed_file_ids:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail={
                "code": "dataset_unavailable",
                "message": "The selected list is unavailable or unauthorized.",
            },
        )
    if body.conversation_id:
        try:
            record = await store.get(scope, body.conversation_id)
        except (PermissionError, ValueError) as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="conversation not found") from exc
        if record is not None and record.selected_file_id not in {None, selected_file_id}:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                detail={
                    "code": "dataset_scope_conflict",
                    "message": "This conversation is tied to another list.",
                },
            )
        if record is not None and record.selected_file_id is None and record.recent_turns:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                detail={
                    "code": "legacy_unscoped_conversation",
                    "message": "Start a new dataset-scoped chat before connecting voice.",
                },
            )
    selected_scope = scope.narrow((selected_file_id,))
    try:
        room_name = assigned_room_name(
            selected_scope,
            conversation_id=body.conversation_id,
            requested=body.room_name,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    identity = f"user-{scope.principal_id}"
    token = mint_livekit_token(
        selected_scope,
        room_name=room_name,
        identity=identity,
        conversation_id=body.conversation_id,
        selected_file_id=selected_file_id,
    )
    return TokenResponse(
        server_url=settings.livekit_url,
        room_name=room_name,
        participant_identity=identity,
        participant_token=token,
        selected_file_id=selected_file_id,
    )
