"""harness.models — Data classes for build / run / verify results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Status(Enum):
    """Outcome status for any harness operation."""

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    TIMEOUT = "timeout"
    SKIP = "skip"


@dataclass
class BuildResult:
    """Result of compiling a kernel spec."""

    status: Status
    duration_seconds: float
    stdout: str
    stderr: str
    executable_path: Optional[str] = None


@dataclass
class RunResult:
    """Result of executing a compiled kernel."""

    status: Status
    configuration: str  # e.g. "correctness", "performance"
    duration_seconds: float
    exit_code: int
    stdout: str
    stderr: str
    cpu_time_seconds: float | None = None
    kernel_time_seconds: float | None = None


@dataclass
class VerificationResult:
    """Result of verifying a kernel run against its spec strategies."""

    status: Status
    strategy_used: str  # which strategy matched
    details: str


@dataclass
class MetricResult:
    """A single extracted performance metric."""

    name: str
    value: float
    unit: str


@dataclass
class SpecResult:
    """Aggregated result for a full spec pipeline (build → run → verify)."""

    spec_id: str
    build: BuildResult
    runs: dict[str, RunResult] = field(default_factory=dict)
    verification: Optional[VerificationResult] = None
    metrics: list[MetricResult] = field(default_factory=list)
