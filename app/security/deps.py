from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from app.config import get_settings
from app.security.access_scope import AccessScope
from app.security.auth_adapter import AuthAdapter, get_auth_adapter
from app.security.standalone_jwt import AuthError


def get_access_scope(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    adapter: AuthAdapter = Depends(get_auth_adapter),
) -> AccessScope:
    settings = get_settings()
    cookie = request.cookies.get(settings.auth_cookie_name)
    credentials = authorization or cookie
    if not credentials:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="missing authorization")
    if not authorization and cookie:
        _validate_cookie_origin(request)
    try:
        return adapter.resolve_scope(credentials)
    except AuthError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc


def _validate_cookie_origin(request: Request) -> None:
    if request.method in {"GET", "HEAD", "OPTIONS"}:
        return
    settings = get_settings()
    origin = request.headers.get("origin")
    fetch_site = request.headers.get("sec-fetch-site")
    allowed = {item.rstrip("/") for item in settings.cors_origin_list}
    if origin and origin.rstrip("/") not in allowed:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="untrusted request origin")
    # Non-browser clients such as voice_cli do not send Fetch Metadata headers.
    # Browsers do, so a browser mutation without an Origin is rejected.
    if not origin and fetch_site:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="missing request origin")
