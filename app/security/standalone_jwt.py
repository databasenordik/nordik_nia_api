import hmac
from datetime import UTC, datetime, timedelta

import jwt

from app.config import Settings
from app.security.access_scope import AccessScope
from app.security.auth_adapter import AuthAdapter


class AuthError(Exception):
    pass


class StandaloneJwtAuthAdapter(AuthAdapter):
    """Phase 1 adapter. Issues and verifies standalone JWTs that carry AccessScope."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _accounts(self) -> list[tuple[str, str, str]]:
        """Every sign-in this process accepts, as (username, password, principal_id).

        The configured pair comes first and is unchanged. Extras exist because the testing
        build publishes one login on its sign-in screen while development keeps its own, and
        one shared password for both would mean rotating the published one locks out the
        person who has to rotate it.
        """
        primary = (
            self._settings.standalone_demo_username,
            self._settings.standalone_demo_password,
            self._settings.standalone_demo_principal_id,
        )
        accounts = [primary] if all(primary) else []
        for extra in self._settings.extra_accounts:
            # An extra without its own principal shares the configured one, so a tester's
            # conversation stays visible to whoever has to look into the report about it.
            principal = extra["principal_id"] or self._settings.standalone_demo_principal_id
            if not principal:
                continue
            accounts.append((extra["username"], extra["password"], principal))
        return accounts

    async def authenticate_password(self, username: str, password: str) -> AccessScope:
        # Usernames compare without surrounding space or case, as usernames normally do.
        # These are printed on the sign-in screen to be copied, so they arrive with a
        # trailing space from a sloppy selection, or capitalised by a keyboard that thinks it
        # is starting a sentence -- "User" was refused with the same blank "sign-in failed" a
        # wrong password gets. The password is still compared exactly.
        attempted = username.strip().casefold()
        matched: tuple[str, str, str] | None = None
        for candidate in self._accounts():
            # compare_digest on both halves, and no early exit once something matches: the
            # time this takes should not describe which usernames exist.
            name_ok = hmac.compare_digest(attempted, candidate[0].strip().casefold())
            password_ok = hmac.compare_digest(password, candidate[1])
            if name_ok and password_ok:
                matched = candidate
        if matched is None:
            raise AuthError("invalid credentials")
        return AccessScope(
            principal_id=matched[2],
            allowed_file_ids=tuple(self._settings.demo_allowed_file_ids),
            can_use_private_files=self._settings.standalone_demo_can_use_private_files,
            history_allowed=self._settings.standalone_demo_history_allowed,
            display_name=username,
        )

    def issue_token(self, scope: AccessScope) -> str:
        now = datetime.now(UTC)
        payload = {
            "iss": self._settings.jwt_issuer,
            "sub": scope.principal_id,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(seconds=self._settings.jwt_ttl_seconds)).timestamp()),
            "allowed_file_ids": list(scope.allowed_file_ids),
            "can_use_private_files": scope.can_use_private_files,
            "history_allowed": scope.history_allowed,
            "display_name": scope.display_name,
        }
        return jwt.encode(payload, self._settings.jwt_secret, algorithm="HS256")

    def resolve_scope(self, credentials: str) -> AccessScope:
        token = credentials.removeprefix("Bearer ").strip()
        try:
            payload = jwt.decode(
                token,
                self._settings.jwt_secret,
                algorithms=["HS256"],
                issuer=self._settings.jwt_issuer,
            )
        except jwt.PyJWTError as exc:
            raise AuthError("invalid token") from exc

        return AccessScope(
            principal_id=str(payload["sub"]),
            allowed_file_ids=tuple(payload.get("allowed_file_ids") or ()),
            can_use_private_files=bool(payload.get("can_use_private_files", False)),
            history_allowed=bool(payload.get("history_allowed", False)),
            display_name=str(payload["display_name"]) if payload.get("display_name") else None,
        )
