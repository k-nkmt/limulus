"""Runtime backend adapters and availability checks for Python and Rust execution."""

from __future__ import annotations

from typing import Any, Callable

from .backend_dispatch_policy import BackendDispatchPolicy
from ..models import DataSetRef, Diagnostic
from .contracts import RuntimeExecuteFn, RuntimeExecutionContext
from .rust_bridge import RustArrowIOBridge


class PythonRuntimeBackend:
    name = "python"

    def __init__(
        self,
        execute_impl: RuntimeExecuteFn,
    ) -> None:
        self._execute_impl = execute_impl

    def is_available(self) -> bool:
        return True

    def execute(
        self,
        context: RuntimeExecutionContext,
    ) -> tuple[dict[str, DataSetRef], list[Diagnostic]]:
        return self._execute_impl(context)


class RustRuntimeBackend:
    name = "rust"

    def __init__(
        self,
        bridge: RustArrowIOBridge | None = None,
        executor: Any | None = None,
        function_registry_keys: tuple[str, ...] = (),
        function_registry_keys_provider: Callable[[], tuple[str, ...]] | None = None,
    ) -> None:
        self._bridge = bridge
        self._executor = executor
        self._function_registry_keys = function_registry_keys
        self._function_registry_keys_provider = function_registry_keys_provider

    def is_available(self) -> bool:
        if self._bridge is None or self._executor is None:
            return False
        if hasattr(self._executor, "is_available"):
            return bool(self._executor.is_available())
        return True

    def execute(
        self,
        context: RuntimeExecutionContext,
    ) -> tuple[dict[str, DataSetRef], list[Diagnostic]]:
        if self._bridge is None or self._executor is None:
            return {}, [
                Diagnostic(
                    code="RUNTIME_BACKEND_CAPABILITY_MISSING",
                    severity="error",
                    message="Rust runtime backend is not available in this build.",
                )
            ]

        function_registry_keys = self._function_registry_keys
        if self._function_registry_keys_provider is not None:
            function_registry_keys = self._function_registry_keys_provider()

        payload, diagnostics = self._bridge.build_payload(context, function_registry_keys)
        if diagnostics:
            return {}, diagnostics
        if payload is None:
            return {}, [
                Diagnostic(
                    code="RUNTIME_RUST_BRIDGE_PAYLOAD_BUILD_FAILED",
                    severity="error",
                    message="Rust runtime bridge could not construct payload.",
                )
            ]

        if not hasattr(self._executor, "execute_block"):
            return {}, [
                Diagnostic(
                    code="RUNTIME_BACKEND_CAPABILITY_MISSING",
                    severity="error",
                    message="Rust runtime executor does not implement execute_block.",
                )
            ]

        return self._executor.execute_block(payload)


class RuntimeBackendSelector:
    def __init__(self, python_backend: PythonRuntimeBackend, rust_backend: RustRuntimeBackend) -> None:
        self._python_backend = python_backend
        self._rust_backend = rust_backend

    def select(
        self,
        preferred_backend: str,
        context: RuntimeExecutionContext | None = None,
    ) -> PythonRuntimeBackend | RustRuntimeBackend:
        engine = BackendDispatchPolicy.choose_engine(
            context,
            preferred_backend,
            rust_available=self._rust_backend.is_available(),
        )
        if engine == "rust":
            return self._rust_backend
        return self._python_backend


__all__ = [
    "PythonRuntimeBackend",
    "RuntimeBackendSelector",
    "RustRuntimeBackend",
]