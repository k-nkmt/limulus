"""Rust runtime adapter for the direct row execution path."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..models import DataSetRef, Diagnostic
from .contracts import (
    RuntimeExecutionResult,
    TypedRustExecutionContext,
    normalize_phase_metrics,
)
from .typed_bridge import TypedRustArrowIOBridge


class RustRuntimeBackend:
    name = "rust"

    def __init__(
        self,
        executor: Any | None = None,
        *,
        typed_bridge: TypedRustArrowIOBridge | None = None,
    ) -> None:
        self._typed_bridge = typed_bridge or TypedRustArrowIOBridge()
        self._executor = executor

    def is_available(self) -> bool:
        if self._executor is None:
            return False
        if hasattr(self._executor, "is_available"):
            return bool(self._executor.is_available())
        return True

    def execute_typed(
        self,
        context: TypedRustExecutionContext,
    ) -> RuntimeExecutionResult:
        """Execute one versioned typed program through the native adapter."""

        if self._typed_bridge is None or self._executor is None:
            return self._result(
                outputs={},
                diagnostics=(
                    Diagnostic(
                        code="RUNTIME_BACKEND_CAPABILITY_MISSING",
                        severity="error",
                        message="Typed Rust runtime backend is not available in this build.",
                    ),
                ),
            )

        payload, diagnostics = self._typed_bridge.build_payload(context)
        if diagnostics:
            return self._result(outputs={}, diagnostics=tuple(diagnostics))
        if payload is None:
            return self._result(
                outputs={},
                diagnostics=(
                    Diagnostic(
                        code="RUNTIME_RUST_BRIDGE_PAYLOAD_BUILD_FAILED",
                        severity="error",
                        message="Typed Rust runtime bridge could not construct payload.",
                    ),
                ),
            )
        if not hasattr(self._executor, "execute_block"):
            return self._result(
                outputs={},
                diagnostics=(
                    Diagnostic(
                        code="RUNTIME_BACKEND_CAPABILITY_MISSING",
                        severity="error",
                        message=(
                            "Rust runtime executor does not implement "
                            "execute_block."
                        ),
                    ),
                ),
            )
        return self._executor.execute_block(payload)

    @staticmethod
    def _result(
        *,
        outputs: Mapping[str, DataSetRef],
        diagnostics: tuple[Diagnostic, ...],
        native_invoked: bool = False,
        phase_metrics: Any = None,
    ) -> RuntimeExecutionResult:
        return RuntimeExecutionResult(
            outputs=outputs,
            diagnostics=diagnostics,
            actual_engine="rust",
            native_invoked=native_invoked,
            phase_metrics=normalize_phase_metrics(phase_metrics),
        )


__all__ = ["RustRuntimeBackend"]
