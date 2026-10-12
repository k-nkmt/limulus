"""Contracts shared across strategy decisions, Rust handoff, and phase metrics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

from ..models import (
    DataSetRef,
    Diagnostic,
    ExecuteRequest,
    PreparationKind,
)
from ..syntax.models import BlockSyntax, DatasetOptionSyntax, SourceDocument


NATIVE_RUNTIME_MODULE_GROUP = (
    "ast",
    "diagnostics",
    "expressions",
    "runtime",
    "io",
    "output",
)

PHASE_METRIC_NAMES = (
    "source_access_ms",
    "row_execution_ms",
    "output_export_ms",
)
LEGACY_PHASE_METRIC_ALIASES = {
    "input_import_ms": "source_access_ms",
}


@runtime_checkable
class TypedOutputPlan(Protocol):
    """Settled output facts supplied by the execution output-plan owner."""

    @property
    def targets(self) -> tuple[str, ...]: ...

    @property
    def default_target(self) -> str | None: ...

    @property
    def suppress_implicit_output(self) -> bool: ...

    @property
    def effective_output_options(self) -> Mapping[str, DatasetOptionSyntax]: ...

    @property
    def diagnostics(self) -> tuple[Diagnostic, ...]: ...


@dataclass(frozen=True)
class TypedRustExecutionContext:
    """Pure syntax and resolved runtime facts for the internal typed bridge."""

    source: SourceDocument
    block: BlockSyntax
    request: ExecuteRequest
    resolved_inputs: Mapping[str, DataSetRef]
    preparation_kind: PreparationKind
    output_plan: TypedOutputPlan
    execution_plan: Mapping[str, Any] | None = None
    execution_readiness: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceDocument):
            raise TypeError("source must be a SourceDocument")
        if not isinstance(self.block, BlockSyntax):
            raise TypeError("block must be a BlockSyntax")
        if not isinstance(self.request, ExecuteRequest):
            raise TypeError("request must be an ExecuteRequest")
        if not isinstance(self.resolved_inputs, Mapping):
            raise TypeError("resolved_inputs must be a mapping")
        if not all(
            isinstance(name, str) and isinstance(reference, DataSetRef)
            for name, reference in self.resolved_inputs.items()
        ):
            raise TypeError("resolved_inputs must map names to DataSetRef values")
        if self.preparation_kind not in {"none", "prepared_set", "prepared_merge"}:
            raise ValueError(
                f"unsupported preparation kind: {self.preparation_kind!r}"
            )
        if not isinstance(self.output_plan, TypedOutputPlan):
            raise TypeError("output_plan must provide the typed output-plan contract")
        if self.block.span.source_id != self.source.source_id:
            raise ValueError("block and source document ids must match")
        if self.block.span.start < 0 or self.block.span.end > len(self.source.text):
            raise ValueError("block span is outside the source document")
        object.__setattr__(
            self,
            "resolved_inputs",
            MappingProxyType(dict(self.resolved_inputs)),
        )
        if self.execution_plan is not None:
            if not isinstance(self.execution_plan, Mapping):
                raise TypeError("execution_plan must be a mapping or None")
            object.__setattr__(
                self,
                "execution_plan",
                MappingProxyType(dict(self.execution_plan)),
            )
        if self.execution_readiness is not None:
            if not isinstance(self.execution_readiness, Mapping):
                raise TypeError("execution_readiness must be a mapping or None")
            object.__setattr__(
                self,
                "execution_readiness",
                MappingProxyType(dict(self.execution_readiness)),
            )


@dataclass(frozen=True)
class TypedRustExecutionPayload:
    """Typed native program, Arrow streams, and resolved execution facts."""

    native_program: Mapping[str, Any]
    input_streams: Mapping[str, Any]
    native_program_json: str = ""
    preparation_kind: PreparationKind = "none"
    apply_registry: Mapping[str, Any] = field(default_factory=dict)
    format_catalog_payload: Mapping[str, Any] = field(default_factory=dict)
    execution_plan: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.preparation_kind not in {"none", "prepared_set", "prepared_merge"}:
            raise ValueError(
                f"unsupported preparation kind: {self.preparation_kind!r}"
            )
        if self.apply_registry is None:
            object.__setattr__(self, "apply_registry", {})
        if self.format_catalog_payload is None:
            object.__setattr__(self, "format_catalog_payload", {})
        if self.execution_plan is not None and not isinstance(
            self.execution_plan, Mapping
        ):
            raise TypeError("execution_plan must be a mapping or None")


@dataclass(frozen=True)
class RuntimeExecutionResult:
    """Typed result envelope returned by the Rust runtime adapter.

    Runtime observations travel with the execution result instead of being
    read from adapter-owned side channels by the coordinator.
    """

    outputs: Mapping[str, DataSetRef]
    diagnostics: tuple[Diagnostic, ...]
    actual_engine: Literal["rust"]
    native_invoked: bool
    phase_metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.actual_engine != "rust":
            raise ValueError("actual_engine must be 'rust'")
        if not isinstance(self.diagnostics, tuple):
            object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(
            self,
            "phase_metrics",
            normalize_phase_metrics(self.phase_metrics),
        )


def normalize_phase_metrics(
    raw_metrics: Any,
    *,
    include_legacy_aliases: bool = False,
) -> dict[str, float]:
    if not isinstance(raw_metrics, Mapping):
        return {}

    metrics: dict[str, float] = {}
    for key in PHASE_METRIC_NAMES:
        value = raw_metrics.get(key)
        if isinstance(value, (int, float)):
            metrics[key] = float(value)

    for legacy_name, canonical_name in LEGACY_PHASE_METRIC_ALIASES.items():
        if canonical_name in metrics:
            continue
        value = raw_metrics.get(legacy_name)
        if isinstance(value, (int, float)):
            metrics[canonical_name] = float(value)

    if include_legacy_aliases:
        for legacy_name, canonical_name in LEGACY_PHASE_METRIC_ALIASES.items():
            value = metrics.get(canonical_name)
            if value is not None:
                metrics[legacy_name] = value
    return metrics


__all__ = [
    "LEGACY_PHASE_METRIC_ALIASES",
    "NATIVE_RUNTIME_MODULE_GROUP",
    "PHASE_METRIC_NAMES",
    "PreparationKind",
    "RuntimeExecutionResult",
    "normalize_phase_metrics",
]
