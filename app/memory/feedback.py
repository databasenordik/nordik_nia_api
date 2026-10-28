"""Where a tester's report goes.

The testing build asks people to click Report, write their name and what they saw, and
submit. This is the only writer of assistant.feedback, which had sat unused since the early
tables.

Same shape as the conversation store above it: a Postgres implementation for the running
stack and a process-local one for tests, chosen the same way, so a report is never lost to a
missing pool during a test run.
"""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Protocol

from app.security.access_scope import AccessScope

# A report is written by hand, so the ceilings are about keeping a stray paste out of the
# database rather than about what anyone is likely to type.
MAX_NAME = 120
MAX_COMMENT = 4000
# A screenshot or two and a short document. The whole request passes through the frontend
# host's proxy, which refuses bodies of a few megabytes, and base64 adds a third on top, so
# the total is held under that rather than per file alone. The page shrinks screenshots first.
MAX_ATTACHMENTS = 3
MAX_ATTACHMENTS_TOTAL_BYTES = 3 * 1024 * 1024

# What a tester is likely to attach, each with the bytes its format starts with. The declared
# type is checked against the content, so a report cannot store something else under a
# harmless name.
_SIGNATURES: dict[str, tuple[bytes, ...]] = {
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/gif": (b"GIF87a", b"GIF89a"),
    "image/webp": (b"RIFF",),
    "application/pdf": (b"%PDF-",),
    "text/plain": (),
}


@dataclass(frozen=True)
class FeedbackAttachment:
    filename: str
    content_type: str
    content: bytes


@dataclass(frozen=True)
class FeedbackReport:
    comment: str
    reporter_name: str = ""
    conversation_id: str | None = None
    principal_id: str = ""
    app_version: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    attachments: tuple[FeedbackAttachment, ...] = ()


def decode_attachment(filename: str, content_type: str, data_base64: str) -> FeedbackAttachment:
    """A tester's file, checked to be what it says it is. Raises ValueError with a reason."""
    name = _safe_filename(filename)
    kind = content_type.split(";", 1)[0].strip().lower()
    if kind not in _SIGNATURES:
        raise ValueError(
            f"{name} can't be attached: use a PNG, JPEG, GIF or WebP image, a PDF, or plain text."
        )
    try:
        content = base64.b64decode(data_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{name} could not be read.") from exc
    if not content:
        raise ValueError(f"{name} is empty.")
    if len(content) > MAX_ATTACHMENTS_TOTAL_BYTES:
        raise ValueError(f"{name} is too large to attach.")
    if not _content_matches(kind, content):
        raise ValueError(f"{name} does not look like the kind of file it says it is.")
    return FeedbackAttachment(filename=name, content_type=kind, content=content)


def _content_matches(kind: str, content: bytes) -> bool:
    if kind == "text/plain":
        try:
            content.decode("utf-8")
        except UnicodeDecodeError:
            return False
        return True
    if kind == "image/webp" and content[8:12] != b"WEBP":
        return False
    return any(content.startswith(signature) for signature in _SIGNATURES[kind])


def _safe_filename(filename: str) -> str:
    base = PurePosixPath((filename or "").replace("\\", "/")).name
    cleaned = re.sub(r"[\x00-\x1f\x7f]", "", base).strip()
    return cleaned[:200] or "attachment"


class FeedbackStore(Protocol):
    async def record(self, scope: AccessScope, report: FeedbackReport) -> None: ...


class InMemoryFeedbackStore:
    def __init__(self) -> None:
        self.reports: list[FeedbackReport] = []

    async def record(self, scope: AccessScope, report: FeedbackReport) -> None:
        self.reports.append(report)


class PostgresFeedbackStore:
    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def record(self, scope: AccessScope, report: FeedbackReport) -> None:
        """Store one report, keeping the conversation link only when it is the writer's own.

        An unrecognised or someone else's conversation id is dropped rather than refused:
        the finding is the point, and losing it to a bad link would be the wrong trade. The
        foreign key would reject it anyway, taking the report with it.
        """
        conversation_id = report.conversation_id or None
        if conversation_id is not None:
            async with self._pool.acquire() as conn:
                owned = await conn.fetchval(
                    """
                    SELECT 1 FROM assistant.conversations
                    WHERE id = $1::uuid AND principal_id = $2
                    """,
                    conversation_id,
                    scope.principal_id,
                )
            if not owned:
                conversation_id = None
        async with self._pool.acquire() as conn:
            # One transaction: a report whose files failed to store would read as complete
            # when it is not.
            async with conn.transaction():
                feedback_id = await conn.fetchval(
                    """
                    INSERT INTO assistant.feedback
                        (conversation_id, comment, reporter_name, principal_id, app_version,
                         category, rating)
                    VALUES ($1::uuid, $2, $3, $4, $5, 'tester_report', NULL)
                    RETURNING id
                    """,
                    conversation_id,
                    report.comment or None,
                    report.reporter_name or None,
                    scope.principal_id,
                    report.app_version or None,
                )
                if report.attachments:
                    await conn.executemany(
                        """
                        INSERT INTO assistant.feedback_attachments
                            (feedback_id, filename, content_type, byte_size, content)
                        VALUES ($1, $2, $3, $4, $5)
                        """,
                        [
                            (
                                feedback_id,
                                item.filename,
                                item.content_type,
                                len(item.content),
                                item.content,
                            )
                            for item in report.attachments
                        ],
                    )


_DEFAULT_FEEDBACK_STORE = InMemoryFeedbackStore()


def default_feedback_store() -> FeedbackStore:
    try:
        from app.db.pool import get_pool

        return PostgresFeedbackStore(get_pool())
    except Exception:
        return _DEFAULT_FEEDBACK_STORE
