"""Normalize execution mode requests to the Rust row strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..models import Diagnostic


ExecutionStrategy = Literal["columnar", "row"]
ExecutionEngine = Literal["rust"]
SUPPORTED_EXECUTION_MODES = ("auto", "rust", "python")


@dataclass(frozen=True)
class ExecutionStrategyDecision:
    """The normalized strategy and implementation choice for one request."""

    requested_mode: str
    strategy: ExecutionStrategy
    engine: ExecutionEngine
    reason_codes: tuple[str, ...]


class StrategyRequestError(ValueError):
    """Raised when a request mode cannot produce an execution decision."""

    def __init__(self, diagnostic: Diagnostic) -> None:
        self.diagnostic = diagnostic
        super().__init__(diagnostic.message)


def build_strategy_request_diagnostic(requested_mode: object) -> Diagnostic:
    """Build the request diagnostic for an unsupported execution mode."""

    if isinstance(requested_mode, str):
        display_mode = requested_mode.strip() or "<empty>"
    else:
        display_mode = repr(requested_mode)
    supported = ", ".join(SUPPORTED_EXECUTION_MODES)
    return Diagnostic(
        code="REQ_INVALID_EXECUTION_MODE",
        severity="error",
        stage="validate",
        message=(
            f"Unknown execution mode {display_mode!r}; expected one of: {supported}."
        ),
    )


def normalize_requested_mode(requested_mode: object) -> str:
    """Normalize a public execution mode or raise its request diagnostic."""

    normalized = requested_mode.strip().lower() if isinstance(requested_mode, str) else ""
    if normalized not in SUPPORTED_EXECUTION_MODES:
        raise StrategyRequestError(build_strategy_request_diagnostic(requested_mode))
    return normalized


def decide_execution_strategy(requested_mode: object) -> ExecutionStrategyDecision:
    """Resolve ``auto``, ``rust``, and ``python`` requests to the Rust row engine."""

    normalized = normalize_requested_mode(requested_mode)
    reason_code = {
        "auto": "ROW_ONLY_STRATEGY",
        "rust": "EXPLICIT_RUST_ROW",
        "python": "LEGACY_PYTHON_BACKEND_ALIAS",
    }[normalized]
    return ExecutionStrategyDecision(
        requested_mode=normalized,
        strategy="row",
        engine="rust",
        reason_codes=(reason_code,),
    )


__all__ = [
    "ExecutionEngine",
    "ExecutionStrategy",
    "ExecutionStrategyDecision",
    "SUPPORTED_EXECUTION_MODES",
    "StrategyRequestError",
    "build_strategy_request_diagnostic",
    "decide_execution_strategy",
    "normalize_requested_mode",
]
