from abc import ABC, abstractmethod
from functools import lru_cache

from app.config import Settings, get_settings
from app.security.access_scope import AccessScope


class AuthAdapter(ABC):
    """Replaceable authentication boundary.

    Phase 1: StandaloneJwtAuthAdapter
    Phase 2: WebsiteAuthAdapter

    Assistant Core depends only on AccessScope produced by this interface.
    """

    @abstractmethod
    async def authenticate_password(self, username: str, password: str) -> AccessScope:
        """Standalone login. Website adapter may reject this path."""

    @abstractmethod
    def issue_token(self, scope: AccessScope) -> str:
        """Return a bearer token the standalone UI can send back."""

    @abstractmethod
    def resolve_scope(self, credentials: str) -> AccessScope:
        """Validate incoming credentials and return an immutable AccessScope."""


def build_auth_adapter(settings: Settings | None = None) -> AuthAdapter:
    cfg = settings or get_settings()
    adapter_name = cfg.auth_adapter.strip().lower()
    if adapter_name == "website":
        from app.security.website_adapter import WebsiteAuthAdapter

        return WebsiteAuthAdapter(cfg)
    from app.security.standalone_jwt import StandaloneJwtAuthAdapter

    return StandaloneJwtAuthAdapter(cfg)


@lru_cache
def get_auth_adapter() -> AuthAdapter:
    """FastAPI-safe provider. Do not add parameters; they become request fields."""
    return build_auth_adapter()
