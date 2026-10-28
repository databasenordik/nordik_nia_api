from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from app.config import get_settings
from app.security.access_scope import AccessScope
from app.security.auth_adapter import AuthAdapter, get_auth_adapter
from app.security.deps import get_access_scope
from app.security.standalone_jwt import AuthError

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


class SessionResponse(BaseModel):
    principal_id: str
    allowed_file_ids: list[int]
    can_use_private_files: bool
    history_allowed: bool


def _serialize(scope: AccessScope) -> SessionResponse:
    return SessionResponse(
        principal_id=scope.principal_id,
        allowed_file_ids=list(scope.allowed_file_ids),
        can_use_private_files=scope.can_use_private_files,
        history_allowed=scope.history_allowed,
    )


def _set_session_cookie(response: Response, adapter: AuthAdapter, scope: AccessScope) -> None:
    settings = get_settings()
    response.set_cookie(
        key=settings.auth_cookie_name,
        value=adapter.issue_token(scope),
        max_age=settings.jwt_ttl_seconds,
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite=settings.auth_cookie_samesite.lower(),
        domain=settings.auth_cookie_domain or None,
        path="/",
    )


def _clear_session_cookie(response: Response) -> None:
    settings = get_settings()
    response.delete_cookie(
        key=settings.auth_cookie_name,
        domain=settings.auth_cookie_domain or None,
        path="/",
        secure=settings.auth_cookie_secure,
        httponly=True,
        samesite=settings.auth_cookie_samesite.lower(),
    )


@router.post("/login", response_model=SessionResponse)
async def login(
    body: LoginRequest,
    response: Response,
    adapter: AuthAdapter = Depends(get_auth_adapter),
) -> SessionResponse:
    try:
        scope = await adapter.authenticate_password(body.username, body.password)
    except AuthError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    _set_session_cookie(response, adapter, scope)
    return _serialize(scope)


@router.post("/session", response_model=SessionResponse)
async def exchange_website_session(
    response: Response,
    scope: AccessScope = Depends(get_access_scope),
    adapter: AuthAdapter = Depends(get_auth_adapter),
) -> SessionResponse:
    """Exchange a signed website AccessScope bearer token for a browser cookie."""
    _set_session_cookie(response, adapter, scope)
    return _serialize(scope)


@router.get("/session", response_model=SessionResponse)
async def browser_session(
    request: Request,
    response: Response,
    adapter: AuthAdapter = Depends(get_auth_adapter),
):
    """Restore a browser session without logging an expected 401 on first visit."""
    settings = get_settings()
    token = request.cookies.get(settings.auth_cookie_name)
    if not token:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    try:
        scope = adapter.resolve_scope(token)
    except AuthError:
        expired = Response(status_code=status.HTTP_204_NO_CONTENT)
        _clear_session_cookie(expired)
        return expired
    _set_session_cookie(response, adapter, scope)
    return _serialize(scope)


@router.get("/me", response_model=SessionResponse)
async def me(
    response: Response,
    scope: AccessScope = Depends(get_access_scope),
    adapter: AuthAdapter = Depends(get_auth_adapter),
) -> SessionResponse:
    _set_session_cookie(response, adapter, scope)
    return _serialize(scope)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(response: Response) -> None:
    _clear_session_cookie(response)
