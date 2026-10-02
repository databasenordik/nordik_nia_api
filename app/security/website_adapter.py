import jwt

from app.config import Settings
from app.security.access_scope import AccessScope
from app.security.auth_adapter import AuthAdapter
from app.security.standalone_jwt import AuthError, StandaloneJwtAuthAdapter


class WebsiteAuthAdapter(AuthAdapter):
    """Validate the minimal signed AccessScope produced by the NIA website.

    Password authentication remains the website's responsibility. After an
    upstream token is validated, ``issue_token`` creates the short-lived
    assistant session token stored in the browser's HttpOnly cookie.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._session_tokens = StandaloneJwtAuthAdapter(settings)

    async def authenticate_password(self, username: str, password: str) -> AccessScope:
        raise AuthError("password login is handled by the NIA website")

    def issue_token(self, scope: AccessScope) -> str:
        return self._session_tokens.issue_token(scope)

    def resolve_scope(self, credentials: str) -> AccessScope:
        # Requests after exchange carry the assistant's own session token.
        try:
            return self._session_tokens.resolve_scope(credentials)
        except AuthError:
            pass

        token = credentials.removeprefix("Bearer ").strip()
        if len(self._settings.website_jwt_secret) < 32:
            raise AuthError("website token verification is not configured")
        try:
            payload = jwt.decode(
                token,
                self._settings.website_jwt_secret,
                algorithms=["HS256"],
                issuer=self._settings.website_jwt_issuer,
                audience=self._settings.website_jwt_audience,
                options={"require": ["exp", "iat", "sub"]},
            )
        except jwt.PyJWTError as exc:
            raise AuthError("invalid website token") from exc

        file_ids = payload.get("allowed_file_ids")
        if not isinstance(file_ids, list) or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in file_ids
        ):
            raise AuthError("website token has an invalid AccessScope")
        private = payload.get("can_use_private_files", False)
        history = payload.get("history_allowed", False)
        if not isinstance(private, bool) or not isinstance(history, bool):
            raise AuthError("website token has an invalid AccessScope")

        try:
            return AccessScope(
                principal_id=str(payload["sub"]),
                allowed_file_ids=tuple(file_ids),
                can_use_private_files=private,
                history_allowed=history,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AuthError("website token has an invalid AccessScope") from exc
