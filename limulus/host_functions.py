"""Request-scoped host callable snapshots for the ``apply`` boundary.

This module intentionally accepts caller locals/globals as values supplied by
the public entrypoint.  It does not inspect the stack; public API capture is a
separate integration responsibility.  The service copies callable values into
an immutable :class:`~limulus.models.HostFunctionContext` and validates the
optional explicit ``functions`` mapping before a request reaches execution.
"""

from __future__ import annotations

import builtins
from collections.abc import Mapping
from dataclasses import dataclass, field
import importlib
from typing import Any
from types import MappingProxyType

from .models import ApplyCallable, Diagnostic, HostFunctionContext


@dataclass(frozen=True)
class HostFunctionContextCapture:
    """Result of one request's host-function snapshot and validation."""

    context: HostFunctionContext = field(default_factory=HostFunctionContext)
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    @property
    def has_errors(self) -> bool:
        return any(diagnostic.severity == "error" for diagnostic in self.diagnostics)


# Short alias for callers that prefer the service name used in the design
# diagrams.  Keeping both names makes the result contract explicit without
# exposing implementation details such as frame objects.
HostFunctionCapture = HostFunctionContextCapture


_RESERVED_BUILTIN_NAMES = (
    "abs",
    "all",
    "any",
    "bool",
    "float",
    "int",
    "len",
    "max",
    "min",
    "round",
    "str",
    "sum",
)
_RESERVED_BUILTINS = {
    name: getattr(builtins, name) for name in _RESERVED_BUILTIN_NAMES
}


@dataclass(frozen=True)
class HostFunctionResolution:
    """Result of resolving all apply targets for one execution request."""

    registry: Mapping[str, ApplyCallable] = field(
        default_factory=lambda: MappingProxyType({})
    )
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.registry, Mapping):
            raise TypeError("host function resolution registry must be a mapping")
        registry = {
            key: value
            for key, value in self.registry.items()
            if isinstance(key, str) and callable(value)
        }
        object.__setattr__(self, "registry", MappingProxyType(registry))
        if not isinstance(self.diagnostics, tuple):
            object.__setattr__(self, "diagnostics", tuple(self.diagnostics))


class HostFunctionResolver:
    """Resolve apply target names using one deterministic request context."""

    def resolve(
        self,
        target_names: Any,
        context: HostFunctionContext,
    ) -> HostFunctionResolution:
        """Resolve each unique target once, failing closed on any error.

        Resolution priority is reserved builtins, explicit functions, caller
        locals, caller globals, then a qualified module target.  The context
        is already request-scoped and immutable; this method never inspects a
        frame or searches an ambient stack.
        """

        if not isinstance(context, HostFunctionContext):
            return HostFunctionResolution(
                diagnostics=(
                    _target_unresolved_diagnostic(
                        target="<context>",
                        source="resolver",
                        reason="host function context is invalid",
                    ),
                )
            )

        if isinstance(target_names, str):
            names = (target_names,)
        else:
            try:
                names = tuple(target_names)
            except TypeError:
                names = (target_names,)

        registry: dict[str, ApplyCallable] = {}
        diagnostics: list[Diagnostic] = []
        seen: set[object] = set()
        for target in names:
            # Target names come from the parser and are strings.  Keep an
            # invalid value diagnosable instead of allowing it to become a
            # surprising dictionary key or importlib argument.
            if not isinstance(target, str):
                diagnostics.append(
                    _target_unresolved_diagnostic(
                        target=repr(target),
                        source="resolver",
                        reason="target name must be a string",
                    )
                )
                continue
            if target in seen:
                continue
            seen.add(target)

            resolved = self._resolve_one(target, context)
            if isinstance(resolved, Diagnostic):
                diagnostics.append(resolved)
                continue
            registry[target] = resolved

        if diagnostics:
            return HostFunctionResolution(diagnostics=tuple(diagnostics))
        return HostFunctionResolution(registry=registry)

    def _resolve_one(
        self,
        target: str,
        context: HostFunctionContext,
    ) -> ApplyCallable | Diagnostic:
        builtin = _RESERVED_BUILTINS.get(target)
        if builtin is not None:
            return builtin

        for source, mapping in (
            ("explicit mapping", context.explicit),
            ("caller locals", context.caller_locals),
            ("caller globals", context.caller_globals),
        ):
            if target not in mapping:
                continue
            candidate = mapping[target]
            if callable(candidate):
                return candidate
            return _target_invalid_diagnostic(
                target=target,
                source=source,
                reason="resolved value is not callable",
            )

        if "." not in target:
            return _target_unresolved_diagnostic(
                target=target,
                source="resolver",
                reason=(
                    "target was not found in reserved builtins, explicit mapping, "
                    "caller locals, or caller globals"
                ),
            )

        module_name, attribute_name = target.rsplit(".", 1)
        if not module_name or not attribute_name:
            return _target_unresolved_diagnostic(
                target=target,
                source="qualified module",
                reason="qualified target must contain a module and attribute",
            )
        if _is_dunder_attribute(attribute_name):
            return _target_unresolved_diagnostic(
                target=target,
                source="qualified module",
                reason="dunder attribute traversal is not allowed",
            )

        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            return _target_unresolved_diagnostic(
                target=target,
                source="qualified module",
                reason=f"module import failed: {type(error).__name__}",
            )

        try:
            candidate = getattr(module, attribute_name)
        except AttributeError:
            return _target_unresolved_diagnostic(
                target=target,
                source="qualified module",
                reason="attribute was not found",
            )
        if not callable(candidate):
            return _target_invalid_diagnostic(
                target=target,
                source="qualified module",
                reason="resolved value is not callable",
            )
        return candidate


def _is_dunder_attribute(attribute_name: str) -> bool:
    return attribute_name.startswith("__") or attribute_name.endswith("__")


def _target_unresolved_diagnostic(
    *,
    target: str,
    source: str,
    reason: str,
) -> Diagnostic:
    return Diagnostic(
        code="RUNTIME_APPLY_TARGET_UNRESOLVED",
        severity="error",
        location=f"apply:{target}",
        stage="resolve",
        message=f"apply target {target!r} is unresolved (source={source}; reason={reason})",
    )


def _target_invalid_diagnostic(
    *,
    target: str,
    source: str,
    reason: str,
) -> Diagnostic:
    return Diagnostic(
        code="RUNTIME_APPLY_TARGET_INVALID",
        severity="error",
        location=f"apply:{target}",
        stage="resolve",
        message=f"apply target {target!r} is invalid (source={source}; reason={reason})",
    )


def _invalid_functions_diagnostic(message: str) -> Diagnostic:
    return Diagnostic(
        code="RUNTIME_APPLY_FUNCTIONS_INVALID",
        severity="error",
        location="request.functions",
        stage="validate",
        message=message,
    )


def _snapshot_caller_callables(value: object) -> dict[str, ApplyCallable]:
    """Shallow-copy callable values from a caller-provided scope mapping."""

    if not isinstance(value, Mapping):
        return {}
    return {
        key: candidate
        for key, candidate in value.items()
        if isinstance(key, str) and callable(candidate)
    }


def validate_explicit_functions(
    functions: object,
) -> tuple[dict[str, ApplyCallable], tuple[Diagnostic, ...]]:
    """Validate and snapshot an explicit ``functions`` mapping.

    Validation is fail-closed: any invalid key or value discards the entire
    mapping instead of returning a usable partial registry.  This allows a
    public caller to pass the diagnostics upstream without accidentally
    executing a request with only part of its intended function set.
    """

    if not isinstance(functions, Mapping):
        return {}, (
            _invalid_functions_diagnostic(
                "functions must be a mapping of string names to callable values; "
                f"received {type(functions).__name__}"
            ),
        )

    snapshot: dict[str, ApplyCallable] = {}
    for key, candidate in functions.items():
        if not isinstance(key, str):
            return {}, (
                _invalid_functions_diagnostic(
                    "functions mapping contains a non-string key "
                    f"(string key required): {key!r}"
                ),
            )
        if not callable(candidate):
            return {}, (
                _invalid_functions_diagnostic(
                    f"functions mapping value for {key!r} must be callable"
                ),
            )
        snapshot[key] = candidate

    return snapshot, ()


def capture_host_function_context(
    caller_locals: object,
    caller_globals: object,
    functions: object | None = None,
) -> HostFunctionContextCapture:
    """Capture one immutable request-scoped host-function context.

    ``caller_locals`` and ``caller_globals`` are low-level inputs supplied by a
    direct public caller.  They are copied and filtered to callable/string
    entries; this function never calls ``inspect`` and never stores a frame.
    When ``functions`` is provided, it must be a valid mapping of string names
    to callables.  Invalid explicit input returns an empty context and a
    ``RUNTIME_APPLY_FUNCTIONS_INVALID`` diagnostic.
    """

    if functions is None:
        return HostFunctionContextCapture(
            context=HostFunctionContext(
                caller_locals=_snapshot_caller_callables(caller_locals),
                caller_globals=_snapshot_caller_callables(caller_globals),
            )
        )

    explicit, diagnostics = validate_explicit_functions(functions)
    if diagnostics:
        # Do not retain even the caller snapshots when request validation has
        # failed.  The caller must handle the diagnostic before executing.
        return HostFunctionContextCapture(diagnostics=diagnostics)

    return HostFunctionContextCapture(
        context=HostFunctionContext(
            explicit=explicit,
            caller_locals=_snapshot_caller_callables(caller_locals),
            caller_globals=_snapshot_caller_callables(caller_globals),
        )
    )


# Explicit aliases used by integration code and tests while the public API
# wiring was introduced.  They all share the same implementation and
# therefore the same fail-closed/request-isolated semantics.
capture = capture_host_function_context
snapshot_caller_callables = _snapshot_caller_callables


__all__ = [
    "HostFunctionCapture",
    "HostFunctionContextCapture",
    "HostFunctionResolution",
    "HostFunctionResolver",
    "capture",
    "capture_host_function_context",
    "snapshot_caller_callables",
    "validate_explicit_functions",
]
