import json
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    nia_transport: Literal["http", "grpc"] = "http"
    nia_grpc_principal_id: str = ""
    nia_grpc_allowed_file_ids: str = "49,91"
    nia_grpc_can_use_private_files: bool = False
    frontend_origin: str = "http://localhost:3000"
    cors_origins: str = "http://localhost:3000"

    assistant_database_url: str = ""
    assistant_migrator_database_url: str = ""
    assistant_runtime_password: str = ""
    assistant_migrator_password: str = ""
    assistant_schema: str = "assistant"
    assistant_api_schema: str = "assistant_api"
    apply_app_stubs: bool = True

    auth_adapter: str = "standalone"
    jwt_secret: str = ""
    jwt_issuer: str = "nia-standalone"
    jwt_ttl_seconds: int = 43200
    auth_cookie_name: str = "nia_session"
    auth_cookie_secure: bool = False
    auth_cookie_samesite: str = "lax"
    auth_cookie_domain: str = ""

    # The existing website signs a minimal AccessScope JWT. The assistant never
    # reads website users, roles, password hashes, OTPs, or session tables.
    website_jwt_secret: str = ""
    website_jwt_issuer: str = "nia-website"
    website_jwt_audience: str = "nia-assistant"

    standalone_demo_username: str = ""
    standalone_demo_password: str = ""
    standalone_demo_principal_id: str = ""
    # Additional sign-ins, as a JSON array of {username, password, principal_id?}. The pair
    # above stays the primary account and needs no change; this is for the case the testing
    # build created, where testers share one published login while development keeps its own.
    # JSON rather than a delimited string because a password may contain any character, and a
    # separator that silently truncates one is a login that fails for no visible reason.
    standalone_extra_accounts: str = ""
    standalone_demo_allowed_file_ids: str = "49,91"
    standalone_demo_can_use_private_files: bool = False
    standalone_demo_history_allowed: bool = False

    default_people_file_id: int = 49
    enable_history_queries: bool = False
    enable_attachment_content: bool = False

    livekit_url: str = "ws://localhost:7880"
    livekit_health_url: str = "http://localhost:7880"
    livekit_api_key: str = ""
    livekit_api_secret: str = ""

    xai_api_key: str = ""
    # HTTP-compatible providers require an explicitly configured base URL.
    reasoning_base_url: str = ""
    reasoning_provider: str = "http"
    reasoning_model: str = "grok-4.20-0309-non-reasoning"
    # Offered to the researcher in the UI. An allowlist because the model name is passed
    # straight to the provider: without one, a caller could name any of the ~491 models the
    # host exposes. Ordered fastest first -- the newer models reason before answering, which
    # on our planner schema costs 6s -> 35s per call.
    reasoning_model_choices: tuple[str, ...] = (
        "grok-4.20-0309-non-reasoning",
        "grok-4.5",
        "grok-4.6-fast",
        "grok-4.6",
    )
    # Which of those answer directly. Everything else reasons first and needs a longer
    # budget: measured p95 87s against 21s. Declared rather than guessed from the name, so
    # adding a model means saying which kind it is.
    direct_answer_models: tuple[str, ...] = ("grok-4.20-0309-non-reasoning",)
    reasoning_request_timeout_ms: int = 150000
    tts_provider: str = "xai"
    xai_tts_voice: str = "rex"
    xai_tts_male_voice: str = "rex"
    xai_tts_female_voice: str = "ara"
    xai_tts_language: str = "en"
    xai_tts_format: str = "pcm"
    xai_tts_streaming: bool = True
    xai_tts_url: str = "wss://api.x.ai/v1/tts"
    xai_tts_sample_rate: int = 24000
    xai_tts_first_audio_timeout_ms: int = 4000
    tts_queue_max_phrases: int = 8

    whisper_model: str = "base.en"
    whisper_service_url: str = "http://localhost:8090"
    voice_endpoint_min_delay_seconds: float = 0.35
    voice_endpoint_max_delay_seconds: float = 3.0

    redis_url: str = ""
    kokoro_url: str = "http://localhost:8880"
    kokoro_voice: str = "am_adam"
    kokoro_male_voice: str = "am_adam"
    kokoro_female_voice: str = "af_heart"
    kokoro_format: str = "pcm"

    max_evidence_tokens: int = 3000
    max_retrieval_results: int = 12
    recent_turn_count: int = 6
    db_pool_min: int = 2
    db_pool_max: int = 12
    db_query_concurrency: int = 8
    db_query_timeout_ms: int = 2000
    query_dag_max_concurrency: int = 8
    retrieval_max_concurrency: int = 6
    retrieval_branch_timeout_ms: int = 1500
    model_max_concurrency_per_turn: int = 2
    model_max_concurrency_global: int = 16
    whisper_max_concurrent_sessions: int = 4
    # A reasoning model spends 2,000+ tokens thinking before it answers; measured at 35s on
    # our planner schema. At 15s every one of them fails rather than merely lagging.
    xai_request_timeout_ms: int = 60000
    # Planning is structured extraction against a schema, not writing. Sampling made the
    # same question compile to different plans run to run -- on a 10-case probe, 4 cases
    # flipped across three identical runs -- which makes a benchmark unable to attribute a
    # regression and makes NIA's own answers irreproducible. Seed is sent when set, so a
    # replay can be pinned exactly.
    # A misspelt name is answered rather than reported missing. One edit away and
    # unambiguous is a typo and is corrected in place, with the correction stated; anything
    # less certain is put back as a question.
    name_correction: bool = True
    name_correction_max_distance: int = Field(default=2, ge=1, le=4)
    name_correction_limit: int = Field(default=6, ge=1, le=25)
    # A word match on narrative text is checked against the records before it runs: moved to
    # the narrative field that actually holds the words when the chosen one has none, and
    # narrowed to the direction the question gives a word ("transferred to", not "from").
    text_search_refinement: bool = True
    # Counting people rather than rows reads every matching row, so it is limited to lists
    # small enough for that to be cheap -- which are also the ones with blank and note rows.
    roster_max_rows: int = Field(default=1000, ge=0)
    # A short text-match answer is read record by record against the words of the question the
    # search could not check ("other schools"), beside the exact count. One bounded call.
    text_match_review: bool = True
    text_match_review_timeout_s: float = Field(default=30.0, gt=0)
    model_temperature: float = 0.0
    model_seed: int | None = None
    readiness_timeout_ms: int = 1500

    rate_limit_enabled: bool = True
    auth_rate_limit_per_minute: int = 10
    query_rate_limit_per_minute: int = 60
    livekit_rate_limit_per_minute: int = 20
    tts_rate_limit_per_minute: int = 60
    trust_proxy_headers: bool = False

    max_plan_steps: int = 24
    max_parallel_branches: int = 8
    max_inputs_per_step: int = 8
    max_files_per_plan: int = 8
    max_query_text: int = 500

    result_set_retention_hours: int = 72
    query_trace_retention_days: int = 14
    audit_retention_days: int = 90
    conversation_retention_days: int = 0
    background_worker_concurrency: int = 4

    planner_mode: Literal["legacy", "ai"] = "ai"
    planner_min_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    planner_requirement_check: bool = True
    planner_self_review: bool = True
    planner_review_mode: Literal["off", "always"] = "always"
    planner_model: str = ""
    planner_review_model: str = ""

    # A list small enough to read in full is answered from its rows rather than by
    # compiling a query. Qualification is by measurement, not by file id, so a list that
    # grows past the threshold falls back to the query path on its own.
    # Off by default, enabled in the deployment, exactly as planner_mode was introduced.
    # Every test fixture is a handful of rows, so a row-count threshold would divert the
    # whole suite away from the query pipeline it exists to test.
    whole_list_answer: bool = False
    whole_list_max_rows: int = Field(default=200, ge=1)
    whole_list_max_chars: int = Field(default=240_000, ge=1000)

    fail_closed_on_privilege_error: bool = Field(
        default=True,
        description="Refuse to start if assistant_runtime can read a forbidden table.",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def extra_accounts(self) -> list[dict[str, str]]:
        """Parsed STANDALONE_EXTRA_ACCOUNTS, or nothing if it is unset or malformed.

        A broken value must not take the primary account down with it, so a parse failure
        yields no extra accounts rather than raising: the configured sign-in still works and
        the operator sees the extras missing rather than a service that will not start.
        """
        raw = self.standalone_extra_accounts.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return []
        if not isinstance(parsed, list):
            return []
        accounts: list[dict[str, str]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            username = str(item.get("username") or "")
            password = str(item.get("password") or "")
            if not username or not password:
                continue
            accounts.append(
                {
                    "username": username,
                    "password": password,
                    "principal_id": str(item.get("principal_id") or ""),
                }
            )
        return accounts

    @property
    def demo_allowed_file_ids(self) -> list[int]:
        return [
            int(item.strip())
            for item in self.standalone_demo_allowed_file_ids.split(",")
            if item.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()


def validate_production_settings(settings: Settings) -> None:
    """Reject known development credentials and unsafe cookie settings."""
    if settings.app_env != "production":
        return

    errors: list[str] = []
    if len(settings.jwt_secret) < 32:
        errors.append("JWT_SECRET must be a unique secret of at least 32 characters")
    if not settings.auth_cookie_secure:
        errors.append("AUTH_COOKIE_SECURE must be true")
    if settings.auth_cookie_samesite.lower() not in {"lax", "strict", "none"}:
        errors.append("AUTH_COOKIE_SAMESITE must be lax, strict, or none")
    if "*" in settings.cors_origin_list:
        errors.append("CORS_ORIGINS must not contain '*' when credentials are enabled")
    if not settings.livekit_api_secret:
        errors.append("LIVEKIT_API_SECRET is required")

    if settings.auth_adapter.strip().lower() == "website":
        if len(settings.website_jwt_secret) < 32:
            errors.append("WEBSITE_JWT_SECRET must be at least 32 characters")
        if not settings.website_jwt_issuer.strip():
            errors.append("WEBSITE_JWT_ISSUER is required")
        if not settings.website_jwt_audience.strip():
            errors.append("WEBSITE_JWT_AUDIENCE is required")
    elif not all(
        (
            settings.standalone_demo_username.strip(),
            settings.standalone_demo_password,
            settings.standalone_demo_principal_id.strip(),
        )
    ):
        errors.append("standalone authentication credentials must be configured")

    if errors:
        raise RuntimeError("unsafe production configuration: " + "; ".join(errors))
