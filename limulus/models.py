from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Literal

from .naming import _dataset_key


ApplyCallable = Callable[..., object]


def _immutable_callable_mapping(mapping: Mapping[str, ApplyCallable]) -> Mapping[str, ApplyCallable]:
    """Create a shallow, callable-only immutable mapping snapshot.

    ``HostFunctionContext`` deliberately stores values rather than frame or
    mapping objects.  The host-functions service performs request validation
    and fail-closed diagnostics before constructing the context; this helper
    keeps direct construction safe as well by discarding non-callable entries.
    """

    if not isinstance(mapping, Mapping):
        raise TypeError("host function context mappings must be mappings")
    return MappingProxyType(
        {
            key: value
            for key, value in mapping.items()
            if isinstance(key, str) and callable(value)
        }
    )


@dataclass(frozen=True)
class HostFunctionContext:
    """Immutable request-scoped callable snapshots for ``apply`` resolution.

    The context contains no frame reference and no mutable registry.  Each
    mapping is copied on construction and exposed through ``MappingProxyType``;
    callers may therefore reuse or mutate their source dictionaries without
    changing an already-created request context.
    """

    explicit: Mapping[str, ApplyCallable] = field(
        default_factory=lambda: MappingProxyType({})
    )
    caller_locals: Mapping[str, ApplyCallable] = field(
        default_factory=lambda: MappingProxyType({})
    )
    caller_globals: Mapping[str, ApplyCallable] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        for name in ("explicit", "caller_locals", "caller_globals"):
            object.__setattr__(self, name, _immutable_callable_mapping(getattr(self, name)))


PreparationKind = Literal["none", "prepared_set", "prepared_merge"]


@dataclass(frozen=True)
class PreparationProvenance:
    """Typed provenance carried separately from ``location``/payload.

    Prepared kind and case-preserving internal variable names travel here, so
    ``DataSetRef.location`` stays opaque user-supplied text and consumers never
    need to parse markers out of it.
    """

    kind: PreparationKind = "none"
    internal_variable_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"none", "prepared_set", "prepared_merge"}:
            raise ValueError(f"unsupported preparation kind: {self.kind!r}")
        object.__setattr__(self, "internal_variable_names", tuple(self.internal_variable_names))


@dataclass(frozen=True)
class DataSetRef:
    kind: str
    location: str
    payload: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    preparation: PreparationProvenance = field(default_factory=PreparationProvenance)


@dataclass(frozen=True)
class SourceMetadataSnapshot:
    """Lightweight source metadata captured before SET/MERGE collapse.

    The snapshot intentionally stores only an Arrow schema and the
    ``DataSetRef.metadata`` mapping.  It never owns a table, row buffer, or
    Python row list, so preserving lineage cannot change the transport
    contract or duplicate execution data.
    """

    source_ordinal: int
    source_name: str
    schema: Any
    dataset_ref_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source_ordinal < 0:
            raise ValueError("source_ordinal must be non-negative")
        if not isinstance(self.dataset_ref_metadata, Mapping):
            raise TypeError("dataset_ref_metadata must be a mapping")
        object.__setattr__(
            self,
            "dataset_ref_metadata",
            MappingProxyType(dict(self.dataset_ref_metadata)),
        )


@dataclass(frozen=True)
class SourceMetadataLineage:
    """Ordered metadata snapshots for DSL source occurrences."""

    ordered_sources: tuple[SourceMetadataSnapshot, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ordered_sources", tuple(self.ordered_sources))


@dataclass(frozen=True)
class NormalizedExecutionInputs:
    """Request-scoped Arrow inputs and the public presentation preference.

    ``inputs`` is the carrier used by preparation and the Rust bridge.  The
    presentation flag is deliberately kept outside that mapping so a public
    memory representation cannot change the native transport contract.
    """

    inputs: Mapping[str, DataSetRef] = field(default_factory=dict)
    preserve_memory_presentation: bool = False


@dataclass(frozen=True)
class PreparedExecutionInputs:
    """Preparation result carried from shared input processing to handoff."""

    inputs: Mapping[str, DataSetRef] = field(default_factory=dict)
    preparation_kind: PreparationKind = "none"
    source_metadata_lineage: SourceMetadataLineage = field(default_factory=SourceMetadataLineage)

    def __post_init__(self) -> None:
        if self.preparation_kind not in {"none", "prepared_set", "prepared_merge"}:
            raise ValueError(f"unsupported preparation kind: {self.preparation_kind!r}")


@dataclass(frozen=True)
class DiagnosticSpan:
    start: int
    end: int
    line: int
    column: int
    end_line: int | None = None
    end_column: int | None = None
    source_id: str = "<dsl>"


@dataclass(frozen=True)
class DiagnosticLabel:
    span: DiagnosticSpan
    message: str = ""
    kind: str = "primary"


@dataclass(frozen=True)
class Diagnostic:
    code: str
    severity: str
    message: str
    location: str = ""
    stage: str = ""
    span: DiagnosticSpan | None = None
    labels: tuple[DiagnosticLabel, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    source_text: str | None = None


@dataclass(frozen=True)
class CompatibilityNotice:
    id: str
    category: str
    summary: str
    link: str = ""


@dataclass(frozen=True)
class ExecuteRequest:
    dsl_text: str
    inputs: Mapping[str, DataSetRef] = field(default_factory=dict)
    output_targets: tuple[str, ...] = field(default_factory=tuple)
    options: Mapping[str, Any] = field(default_factory=dict)
    host_function_context: HostFunctionContext = field(default_factory=HostFunctionContext)
    runtime_backend: str | None = None


@dataclass(frozen=True)
class ExecuteResponse:
    outputs: Mapping[str, DataSetRef] = field(default_factory=dict)
    outputs_arrow: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)
    notices: tuple[CompatibilityNotice, ...] = field(default_factory=tuple)

    @property
    def has_errors(self) -> bool:
        return any(diag.severity == "error" for diag in self.diagnostics)


@dataclass(frozen=True)
class OutputConversionResult:
    outputs: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    @property
    def has_errors(self) -> bool:
        return any(diag.severity == "error" for diag in self.diagnostics)


@dataclass(frozen=True)
class LogEntry:
    code: str
    severity: str
    message: str
    location: str = ""
    stage: str = ""
    span: DiagnosticSpan | None = None
    labels: tuple[DiagnosticLabel, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    source_text: str | None = None


@dataclass(frozen=True)
class RenderRequest:
    source_id: str = "<dsl>"
    source_text: str | None = None
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)


@dataclass
class SubmitResult:
    success: bool
    datasets: Mapping[str, Any] = field(default_factory=dict)
    log: tuple[LogEntry, ...] = field(default_factory=tuple)
    elapsed_seq: float = 0.0
    display_result: bool = field(default=True, repr=False)

    def __bool__(self) -> bool:
        return self.success

    def __repr__(self) -> str:
        if not self.display_result:
            return ""
        return (
            "SubmitResult("
            f"success={self.success}, "
            f"datasets={self.datasets}, "
            f"log={self.log}, "
            f"elapsed_seq={self.elapsed_seq}"
            ")"
        )

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        printer.text(repr(self))

    def format_log(self) -> str:
        lines: list[str] = [
            f"success: {self.success}",
            f"{round(self.elapsed_seq, 2)} seconds elapsed",
        ]
        error_count = sum(1 for entry in self.log if entry.severity == "error")
        if error_count:
            lines.append(f"Error entries: {error_count}")
        if not self.log:
            lines.append("(no log entries)")
            return "\n".join(lines)

        for entry in self.log:
            from .renderer import render_diagnostics

            diagnostic = Diagnostic(
                code=entry.code,
                severity=entry.severity,
                message=entry.message,
                location=entry.location,
                stage=entry.stage,
                span=entry.span,
                labels=entry.labels,
                notes=entry.notes,
                source_text=entry.source_text,
            )
            source_id = entry.span.source_id if entry.span is not None else "<dsl>"
            rendered = render_diagnostics(
                RenderRequest(
                    source_id=source_id,
                    source_text=entry.source_text,
                    diagnostics=(diagnostic,),
                )
            )
            lines.append(rendered)
        return "\n".join(lines)

    def print_log(self) -> None:
        print(self.format_log())


class DatasetCatalog(Mapping[str, Any]):
    def __init__(self) -> None:
        self._datasets: MutableMapping[str, Any] = {}
        self._display_names: MutableMapping[str, str] = {}

    def set(self, name: str, table: Any) -> None:
        normalized = self._normalize(name)
        self._datasets[normalized] = table
        self._display_names[normalized] = self._display_name(name)

    def update(self, datasets: Mapping[str, Any]) -> None:
        for name, table in datasets.items():
            self.set(name, table)

    def delete(self, name: str, *, missing_ok: bool = True) -> bool:
        normalized = self._normalize(name)
        if normalized in self._datasets:
            del self._datasets[normalized]
            self._display_names.pop(normalized, None)
            return True
        if missing_ok:
            return False
        raise KeyError(name)

    def __getitem__(self, name: str) -> Any:
        return self._datasets[self._normalize(name)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._display_names.values())

    def __len__(self) -> int:
        return len(self._datasets)

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return self._normalize(name) in self._datasets

    @staticmethod
    def _normalize(name: str) -> str:
        return _dataset_key(name)

    @staticmethod
    def _display_name(name: str) -> str:
        normalized = name.strip()
        if normalized[:5].upper() == "WORK.":
            return normalized[5:]
        return normalized


__all__ = [
    "ApplyCallable",
    "DataSetRef",
    "PreparationKind",
    "SourceMetadataSnapshot",
    "SourceMetadataLineage",
    "NormalizedExecutionInputs",
    "PreparedExecutionInputs",
    "DiagnosticSpan",
    "DiagnosticLabel",
    "Diagnostic",
    "HostFunctionContext",
    "ExecuteRequest",
    "ExecuteResponse",
    "OutputConversionResult",
    "CompatibilityNotice",
    "LogEntry",
    "RenderRequest",
    "SubmitResult",
    "DatasetCatalog",
]
