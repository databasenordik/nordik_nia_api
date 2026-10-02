from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

_FORBIDDEN_ATTRS = frozenset(
    {
        "evidence",
        "evidence_packet",
        "rows",
        "sql",
        "password",
        "otp",
        "authorization",
        "raw_fields",
    }
)

_current: ContextVar[str | None] = ContextVar("nia_span_id", default=None)


def _safe_attrs(attributes: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in attributes.items() if key.lower() not in _FORBIDDEN_ATTRS}


def _otel_tracer():
    try:
        from opentelemetry import trace
    except ImportError:
        return None
    return trace.get_tracer("nia.assistant")


def _otel_error_status():
    from opentelemetry.trace import Status, StatusCode

    return Status(StatusCode.ERROR)


def _otel_value(value: Any) -> Any:
    if isinstance(value, str | bool | int | float):
        return value
    return str(value)


@dataclass
class Span:
    name: str
    span_id: str = field(default_factory=lambda: uuid4().hex)
    parent_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.perf_counter)
    ended_at: float | None = None
    status: str = "ok"

    @property
    def duration_ms(self) -> float:
        if self.ended_at is None:
            return 0.0
        return (self.ended_at - self.started_at) * 1000


class Tracer:
    """In-process spans. Uses OpenTelemetry when the SDK is installed."""

    def __init__(self) -> None:
        self.spans: list[Span] = []
        self._otel = _otel_tracer()

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        parent = _current.get()
        record = Span(name=name, parent_id=parent, attributes=_safe_attrs(attributes))
        self.spans.append(record)
        token = _current.set(record.span_id)
        otel_span = None
        if self._otel is not None:
            otel_span = self._otel.start_span(name)
            for key, value in record.attributes.items():
                otel_span.set_attribute(key, _otel_value(value))
        try:
            yield record
        except Exception:
            record.status = "error"
            if otel_span is not None:
                otel_span.set_status(_otel_error_status())
            raise
        finally:
            record.ended_at = time.perf_counter()
            if otel_span is not None:
                for key, value in _safe_attrs(record.attributes).items():
                    otel_span.set_attribute(key, _otel_value(value))
                otel_span.end()
            _current.reset(token)

    def by_name(self, name: str) -> list[Span]:
        return [item for item in self.spans if item.name == name]

    def reset(self) -> None:
        self.spans.clear()


_TRACER = Tracer()


def get_tracer() -> Tracer:
    return _TRACER


def reset_tracer() -> None:
    _TRACER.reset()
