from __future__ import annotations

import json
import re
import time
import uuid

import jwt

from app.config import get_settings
from app.security.access_scope import AccessScope

TOKEN_TTL_SECONDS = 15 * 60
AGENT_NAME = "nia-research-assistant"
_ROOM_SAFE = re.compile(r"[^a-zA-Z0-9_-]+")


def principal_room_prefix(scope: AccessScope) -> str:
    slug = _ROOM_SAFE.sub("-", scope.principal_id).strip("-")[:32] or "user"
    return f"nia-{slug}"


def assigned_room_name(
    scope: AccessScope,
    *,
    conversation_id: str | None = None,
    requested: str | None = None,
) -> str:
    """Rooms are always namespaced to the authenticated principal."""
    prefix = principal_room_prefix(scope)
    if requested:
        if requested != prefix and not requested.startswith(prefix + "-"):
            raise ValueError("room_name is not allowed for this principal")
        return requested
    tail = _ROOM_SAFE.sub("-", (conversation_id or uuid.uuid4().hex)[:32]).strip("-") or uuid.uuid4().hex[:8]
    return f"{prefix}-{tail}"


def mint_livekit_token(
    scope: AccessScope,
    *,
    room_name: str,
    identity: str,
    conversation_id: str | None = None,
    selected_file_id: int | None = None,
    ttl_seconds: int = TOKEN_TTL_SECONDS,
) -> str:
    settings = get_settings()
    now = int(time.time())
    metadata = {
        "principal_id": scope.principal_id,
        "allowed_file_ids": list(scope.allowed_file_ids),
        "can_use_private_files": scope.can_use_private_files,
        "history_allowed": scope.history_allowed,
        "conversation_id": conversation_id,
        "selected_file_id": selected_file_id,
        "scope_fingerprint": scope.fingerprint(),
    }
    payload = {
        "iss": settings.livekit_api_key,
        "sub": identity,
        "name": identity,
        "nbf": now,
        "exp": now + ttl_seconds,
        "jti": str(uuid.uuid4()),
        "video": {
            "room": room_name,
            "roomJoin": True,
            "canPublish": True,
            "canSubscribe": True,
            "canPublishData": True,
        },
        "roomConfig": {"agents": [{"agentName": AGENT_NAME}]},
        "metadata": json.dumps(metadata),
    }
    return jwt.encode(payload, settings.livekit_api_secret, algorithm="HS256")


def decode_livekit_token(token: str) -> dict:
    settings = get_settings()
    return jwt.decode(
        token,
        settings.livekit_api_secret,
        algorithms=["HS256"],
        issuer=settings.livekit_api_key,
    )


def metadata_dict(metadata: str | dict | None) -> dict:
    if metadata is None:
        raise ValueError("missing participant metadata")
    if isinstance(metadata, str):
        if not metadata.strip():
            raise ValueError("missing participant metadata")
        return json.loads(metadata)
    return metadata


def scope_from_metadata(metadata: str | dict | None) -> AccessScope:
    payload = metadata_dict(metadata)
    return AccessScope(
        principal_id=str(payload["principal_id"]),
        allowed_file_ids=tuple(payload.get("allowed_file_ids") or ()),
        can_use_private_files=bool(payload.get("can_use_private_files", False)),
        history_allowed=bool(payload.get("history_allowed", False)),
    )


def conversation_id_from_metadata(metadata: str | dict | None) -> str | None:
    payload = metadata_dict(metadata)
    value = payload.get("conversation_id")
    return str(value) if value else None


def selected_file_id_from_metadata(metadata: str | dict | None) -> int | None:
    payload = metadata_dict(metadata)
    value = payload.get("selected_file_id")
    return int(value) if value is not None else None
