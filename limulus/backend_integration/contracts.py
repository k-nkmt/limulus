"""Contracts shared across backend selection, Rust handoff, and phase metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

from ..models import DataSetRef, Diagnostic, ExecuteRequest


NATIVE_RUNTIME_MODULE_GROUP = (
    "ast",
    "diagnostics",
    "expressions",
    "runtime",
    "io",
    "output",
)

STANDARD_SOURCE_ACCESS_MODE = "borrowed_arrow_view"
CANONICAL_COMPATIBILITY_PATH_REASONS = (
    "end_var_survivor_count_materialization",
    "dataset_option_materialization",
    "output_dataset_option_materialization",
    "unplanned_output_handoff",
    "legacy_output_materialization",
)
PHASE_METRIC_NAMES = (
    "source_access_ms",
    "row_execution_ms",
    "output_export_ms",
)
LEGACY_PHASE_METRIC_ALIASES = {
    "input_import_ms": "source_access_ms",
}
LEGACY_OUTPUT_MATERIALIZATION_REASON = "legacy_output_materialization"


@dataclass(frozen=True)
class RuntimeExecutionContext:
    request: ExecuteRequest
    ast_statements: Sequence[Any]
    resolved_inputs: Mapping[str, DataSetRef]
    resolved_output_targets: tuple[str, ...]
    execution_plan: Any | None = None
    transport_mode: str = "standard_arrow_stream"
    builder_mode: str | None = None
    rewrite_metadata: Any | None = None
    python_limited_mode: bool = False


@dataclass(frozen=True)
class RustExecutionPayload:
    ast_statements: Sequence[Any]
    output_targets: tuple[str, ...]
    input_streams: Mapping[str, Any]
    legacy_inputs: Mapping[str, Any] = None
    apply_registry: Mapping[str, Any] = None
    format_catalog_payload: Mapping[str, Any] = None
    prepared_merge_mode: bool = False
    function_registry_keys: tuple[str, ...] = ()
    execution_plan: Any | None = None
    row_loop_plan: Any | None = None
    source_access_mode: str = STANDARD_SOURCE_ACCESS_MODE
    compatibility_path_reason: str | None = None
    transport_mode: str = "standard_arrow_stream"
    builder_mode: str | None = None
    rewrite_metadata: Any | None = None
    python_limited_mode: bool = False

    def __post_init__(self) -> None:
        if self.legacy_inputs is None:
            object.__setattr__(self, "legacy_inputs", {})
        if self.apply_registry is None:
            object.__setattr__(self, "apply_registry", {})
        if self.format_catalog_payload is None:
            object.__setattr__(self, "format_catalog_payload", {})


RuntimeExecuteFn = Callable[[RuntimeExecutionContext], tuple[dict[str, DataSetRef], list[Diagnostic]]]


@dataclass(frozen=True)
class SourceOwnerEvidenceContract:
    source_access_mode: str = STANDARD_SOURCE_ACCESS_MODE
    compatibility_path_reason: str | None = None
    phase_metrics: Mapping[str, float] = field(default_factory=dict)

    @property
    def path_classification(self) -> str:
        return classify_source_access_path(
            source_access_mode=self.source_access_mode,
            compatibility_path_reason=self.compatibility_path_reason,
        )


def normalize_compatibility_path_reason(reason: Any) -> str | None:
    if isinstance(reason, str) and reason in CANONICAL_COMPATIBILITY_PATH_REASONS:
        return reason
    return None


def canonicalize_compatibility_path_reason(
    reason: Any,
    *,
    prepared_merge_mode: bool = False,
    transport_mode: str | None = None,
) -> str | None:
    if prepared_merge_mode or transport_mode == "diagnostic_memory_rows":
        return LEGACY_OUTPUT_MATERIALIZATION_REASON
    return normalize_compatibility_path_reason(reason)


def normalize_source_access_mode(
    source_access_mode: Any,
    compatibility_path_reason: Any,
) -> str:
    return STANDARD_SOURCE_ACCESS_MODE


def classify_source_access_path(
    *,
    source_access_mode: Any,
    compatibility_path_reason: Any,
) -> str:
    if normalize_compatibility_path_reason(compatibility_path_reason) is not None:
        return "unsupported_cleanup_path"
    return "standard_success_path"


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


def build_source_owner_evidence_contract(
    *,
    source_access_mode: Any,
    compatibility_path_reason: Any,
    phase_metrics: Any,
) -> SourceOwnerEvidenceContract:
    normalized_reason = normalize_compatibility_path_reason(compatibility_path_reason)
    normalized_mode = normalize_source_access_mode(
        source_access_mode=source_access_mode,
        compatibility_path_reason=normalized_reason,
    )
    return SourceOwnerEvidenceContract(
        source_access_mode=normalized_mode,
        compatibility_path_reason=normalized_reason,
        phase_metrics=normalize_phase_metrics(phase_metrics),
    )


class MetricsAdapter:
    @staticmethod
    def source_owner_metadata(
        *,
        raw_metadata: Any = None,
        source_access_mode: Any = STANDARD_SOURCE_ACCESS_MODE,
        compatibility_path_reason: Any = None,
        phase_metrics: Any = None,
        prepared_merge_mode: bool = False,
        transport_mode: str | None = None,
    ) -> dict[str, Any]:
        metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
        raw_reason = metadata.get("compatibility_path_reason", compatibility_path_reason)
        normalized_reason = canonicalize_compatibility_path_reason(
            raw_reason,
            prepared_merge_mode=prepared_merge_mode,
            transport_mode=transport_mode,
        )
        raw_phase_metrics = phase_metrics if phase_metrics is not None else metadata.get("phase_metrics")
        evidence = build_source_owner_evidence_contract(
            source_access_mode=metadata.get("source_access_mode", source_access_mode),
            compatibility_path_reason=normalized_reason,
            phase_metrics=raw_phase_metrics,
        )
        return {
            "source_access_mode": evidence.source_access_mode,
            "compatibility_path_reason": evidence.compatibility_path_reason,
            "path_classification": evidence.path_classification,
            "phase_metrics": dict(evidence.phase_metrics),
        }


__all__ = [
    "CANONICAL_COMPATIBILITY_PATH_REASONS",
    "LEGACY_OUTPUT_MATERIALIZATION_REASON",
    "LEGACY_PHASE_METRIC_ALIASES",
    "MetricsAdapter",
    "NATIVE_RUNTIME_MODULE_GROUP",
    "PHASE_METRIC_NAMES",
    "RuntimeExecuteFn",
    "RuntimeExecutionContext",
    "RustExecutionPayload",
    "STANDARD_SOURCE_ACCESS_MODE",
    "SourceOwnerEvidenceContract",
    "build_source_owner_evidence_contract",
    "canonicalize_compatibility_path_reason",
    "classify_source_access_path",
    "normalize_compatibility_path_reason",
    "normalize_phase_metrics",
    "normalize_source_access_mode",
]
