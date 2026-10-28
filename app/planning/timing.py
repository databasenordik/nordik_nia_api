from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

# Per-layer planning latency. This exists so a turn can report where its time
# went (normalize / direct route / deterministic / semantic / constrained) without
# the planner importing an observability backend. Values are milliseconds.


@dataclass(frozen=True)
class PlanningTimings:
    """Immutable ordered record of planning stage durations."""

    stages: tuple[tuple[str, float], ...] = ()

    def as_dict(self) -> dict[str, float]:
        return {name: round(value, 3) for name, value in self.stages}

    def get(self, name: str, default: float = 0.0) -> float:
        for stage, value in self.stages:
            if stage == name:
                return value
        return default

    @property
    def total_ms(self) -> float:
        return round(sum(value for _name, value in self.stages), 3)

    @property
    def slowest(self) -> tuple[str, float] | None:
        if not self.stages:
            return None
        return max(self.stages, key=lambda item: item[1])


class TimingRecorder:
    """Collects stage durations for one planning pass.

    Stages are recorded even when the measured block raises, so a failed layer
    still shows the time it consumed.
    """

    def __init__(self) -> None:
        self._stages: list[tuple[str, float]] = []

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            self._stages.append((name, elapsed_ms))

    def record(self, name: str, elapsed_ms: float) -> None:
        self._stages.append((name, float(elapsed_ms)))

    def finish(self) -> PlanningTimings:
        return PlanningTimings(stages=tuple(self._stages))
