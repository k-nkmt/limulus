"""Immutable contracts for the typed execution coordinator and stage owners."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from ..models import (
    DataSetRef,
    Diagnostic,
    ExecuteRequest,
    ExecuteResponse,
    NormalizedExecutionInputs,
    PreparedExecutionInputs,
    SourceMetadataLineage,
    SourceMetadataSnapshot,
)
from ..parser import ParsedStatement
from ..syntax.models import BlockSyntax, SourceDocument, SourceStatementSyntax
from .rewrites import RewritePlan, TypedHelperPlan
from .statement_expressions import StatementExpressionSlot
from .strategy import ExecutionStrategyDecision

if TYPE_CHECKING:
    from .typed_output_planning import TypedOutputTargetPlan


class _ImmutableDict(dict[Any, Any]):
    """Read-only dict-compatible mapping for native serializers.

    ``MappingProxyType`` is the preferred boundary representation, but a few
    existing native serialization paths require nested values to satisfy
    ``isinstance(value, dict)``. This narrow dict subclass keeps that
    compatibility without exposing mutation through the snapshot.
    """

    def __init__(self, value: Mapping[Any, Any]) -> None:
        dict.__init__(self, value)

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        raise TypeError("mapping is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __ior__(self, value: Any) -> "_ImmutableDict":
        self._immutable(value)
        return self


class _ImmutableList(list[Any]):
    """Read-only list-compatible sequence for native serializers."""

    def __init__(self, value: Any) -> None:
        list.__init__(self, value)

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        raise TypeError("sequence is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable

    def __iadd__(self, value: Any) -> "_ImmutableList":
        self._immutable(value)
        return self

    def __imul__(self, value: Any) -> "_ImmutableList":
        self._immutable(value)
        return self


@dataclass(frozen=True)
class HelperColumnPlan:
    """Planner-owned helper-column layout shared by plan producers/consumers."""

    arrow_columns: tuple[str, ...]
    fallback_columns: tuple[str, ...]
    source_aliases: tuple[str, ...]
    requires_row_prepare: bool
    fallback_reason: str | None


def _empty_mapping() -> Mapping[str, Any]:
    return MappingProxyType({})


def _snapshot_value(value: Any) -> Any:
    """Deep-freeze plan containers without copying opaque/heavy carriers.

    Execution payloads (including Arrow tables) are deliberately retained by
    identity. Container shells and embedded dataclass plans are recreated
    recursively; scalar and opaque values remain request-owned values.
    """

    if isinstance(value, list):
        return _ImmutableList(_snapshot_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_snapshot_value(item) for item in value)
    if isinstance(value, Mapping):
        snapshot = {key: _snapshot_value(item) for key, item in value.items()}
        if isinstance(value, dict):
            return _ImmutableDict(snapshot)
        return MappingProxyType(snapshot)
    if is_dataclass(value) and not isinstance(value, type):
        try:
            return replace(
                value,
                **{
                    item.name: _snapshot_value(getattr(value, item.name))
                    for item in dataclass_fields(value)
                },
            )
        except (TypeError, ValueError):
            # Opaque dataclasses may expose constructor-incompatible fields;
            # preserve their identity rather than changing their semantics.
            return value
    return value


def _snapshot_mapping(value: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return MappingProxyType(
        {key: _snapshot_value(item) for key, item in value.items()}
    )


def _snapshot_dataset_ref(value: Any) -> Any:
    """Copy a DataSetRef shell while retaining its payload object."""

    if not isinstance(value, DataSetRef):
        return value
    metadata = value.metadata
    if isinstance(metadata, Mapping):
        metadata = _snapshot_value(metadata)
    return DataSetRef(
        kind=value.kind,
        location=value.location,
        payload=value.payload,
        metadata=metadata,
        # Provenance is an immutable frozen value: carry it by reference so the
        # snapshot never rebuilds it from defaults (payload is likewise shared).
        preparation=value.preparation,
    )


def _snapshot_dataset_mapping(
    value: Mapping[str, DataSetRef], *, name: str
) -> Mapping[str, DataSetRef]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return MappingProxyType(
        {key: _snapshot_dataset_ref(item) for key, item in value.items()}
    )


def _snapshot_execute_request(value: ExecuteRequest) -> ExecuteRequest:
    if not isinstance(value, ExecuteRequest):
        raise TypeError("request must be ExecuteRequest")
    return ExecuteRequest(
        dsl_text=value.dsl_text,
        inputs=_snapshot_dataset_mapping(value.inputs, name="request.inputs"),
        output_targets=tuple(value.output_targets),
        options=_snapshot_mapping(value.options, name="request.options"),
        host_function_context=value.host_function_context,
        runtime_backend=value.runtime_backend,
    )


def _snapshot_lineage(value: SourceMetadataLineage) -> SourceMetadataLineage:
    if not isinstance(value, SourceMetadataLineage):
        raise TypeError("source_metadata_lineage must be SourceMetadataLineage")
    return SourceMetadataLineage(
        ordered_sources=tuple(
            SourceMetadataSnapshot(
                source_ordinal=source.source_ordinal,
                source_name=source.source_name,
                schema=source.schema,
                dataset_ref_metadata=_snapshot_value(source.dataset_ref_metadata),
            )
            for source in value.ordered_sources
        )
    )


def _snapshot_prepared_inputs(value: PreparedExecutionInputs) -> PreparedExecutionInputs:
    if not isinstance(value, PreparedExecutionInputs):
        raise TypeError("prepared_inputs must be PreparedExecutionInputs")
    return PreparedExecutionInputs(
        inputs=_snapshot_dataset_mapping(
            value.inputs, name="prepared_inputs.inputs"
        ),
        preparation_kind=value.preparation_kind,
        source_metadata_lineage=_snapshot_lineage(value.source_metadata_lineage),
    )


@dataclass(frozen=True)
class PipelineExecutionRequest:
    """Request and catalog snapshots consumed by one pipeline execution."""

    request: ExecuteRequest
    registered_tables: Mapping[str, DataSetRef] = field(default_factory=_empty_mapping)
    generated_tables: Mapping[str, DataSetRef] = field(default_factory=_empty_mapping)
    runtime_backend_preference: str = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(self, "request", _snapshot_execute_request(self.request))
        object.__setattr__(
            self,
            "registered_tables",
            _snapshot_dataset_mapping(
                self.registered_tables, name="registered_tables"
            ),
        )
        object.__setattr__(
            self,
            "generated_tables",
            _snapshot_dataset_mapping(
                self.generated_tables, name="generated_tables"
            ),
        )


@dataclass(frozen=True)
class StrategyDecisionRequest:
    """Immutable request for the one request-scoped strategy decision."""

    runtime_backend_preference: object

    @property
    def preference(self) -> object:
        """Alias used by strategy-focused callers without another state field."""

        return self.runtime_backend_preference


@dataclass(frozen=True)
class StrategyDecisionResult:
    """Strategy decision or its request diagnostic, without side channels."""

    strategy_decision: ExecutionStrategyDecision | None = None
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.diagnostics, tuple):
            object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def decision(self) -> ExecutionStrategyDecision | None:
        """Short alias for consumers that refer to the value as a decision."""

        return self.strategy_decision


@dataclass(frozen=True)
class TypedValidationRequest:
    """Internal typed inputs for one block validation pass.

    The validation entry point receives the pure ``BlockSyntax`` plus the
    request-scoped ``SourceDocument`` that owns its spans, and the same
    request-scoped catalog snapshots as the rest of the typed pipeline.
    """

    block: BlockSyntax
    source: SourceDocument
    explicit_inputs: Mapping[str, DataSetRef] = field(default_factory=_empty_mapping)
    available_inputs: Mapping[str, DataSetRef] = field(default_factory=_empty_mapping)
    registered_tables: Mapping[str, DataSetRef] = field(default_factory=_empty_mapping)
    explicit_output_targets: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.block, BlockSyntax):
            raise TypeError("block must be a BlockSyntax")
        if not isinstance(self.source, SourceDocument):
            raise TypeError("source must be a SourceDocument")
        block_span = self.block.span
        if block_span.source_id != self.source.source_id:
            raise ValueError(
                f"block span source_id {block_span.source_id!r} does not match "
                f"source document {self.source.source_id!r}"
            )
        if block_span.end > len(self.source.text):
            raise ValueError(
                f"block span [{block_span.start}, {block_span.end}) is outside "
                f"the source document (length {len(self.source.text)})"
            )
        object.__setattr__(
            self,
            "explicit_inputs",
            _snapshot_dataset_mapping(self.explicit_inputs, name="explicit_inputs"),
        )
        object.__setattr__(
            self,
            "available_inputs",
            _snapshot_dataset_mapping(self.available_inputs, name="available_inputs"),
        )
        object.__setattr__(
            self,
            "registered_tables",
            _snapshot_dataset_mapping(self.registered_tables, name="registered_tables"),
        )
        object.__setattr__(
            self,
            "explicit_output_targets",
            tuple(str(target) for target in self.explicit_output_targets),
        )


@dataclass(frozen=True)
class TypedOutputPlanRequest:
    """Internal typed request for resolving output targets and options."""

    block: BlockSyntax
    source: SourceDocument
    explicit_output_targets: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.block, BlockSyntax):
            raise TypeError("block must be a BlockSyntax")
        if not isinstance(self.source, SourceDocument):
            raise TypeError("source must be a SourceDocument")
        if self.block.span.source_id != self.source.source_id:
            raise ValueError(
                f"block span source_id {self.block.span.source_id!r} does not match "
                f"source document {self.source.source_id!r}"
            )
        if self.block.span.end > len(self.source.text):
            raise ValueError(
                f"block span [{self.block.span.start}, {self.block.span.end}) is outside "
                f"the source document (length {len(self.source.text)})"
            )
        object.__setattr__(
            self,
            "explicit_output_targets",
            tuple(str(target) for target in self.explicit_output_targets),
        )


@dataclass(frozen=True)
class ValidationResult:
    """Diagnostics returned by the validation owner without side channels."""

    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.diagnostics, tuple):
            object.__setattr__(self, "diagnostics", tuple(self.diagnostics))


@dataclass(frozen=True)
class TypedInputResolutionRequest:
    """Structural SET/MERGE references and request-scoped catalog snapshots."""

    block: BlockSyntax
    explicit_inputs: Mapping[str, DataSetRef] = field(default_factory=_empty_mapping)
    available_inputs: Mapping[str, DataSetRef] = field(default_factory=_empty_mapping)
    registered_tables: Mapping[str, DataSetRef] = field(default_factory=_empty_mapping)
    synthetic_inputs: Mapping[str, DataSetRef] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        if not isinstance(self.block, BlockSyntax):
            raise TypeError("block must be a BlockSyntax")
        for name in (
            "explicit_inputs",
            "available_inputs",
            "registered_tables",
            "synthetic_inputs",
        ):
            object.__setattr__(
                self,
                name,
                _snapshot_dataset_mapping(getattr(self, name), name=name),
            )

    @property
    def source_occurrences(self) -> tuple[Any, ...]:
        source_statement = next(
            (
                statement
                for statement in self.block.statements
                if statement.kind in {"SET", "MERGE"}
            ),
            None,
        )
        if source_statement is None or not isinstance(
            source_statement.payload, SourceStatementSyntax
        ):
            return ()
        return tuple(source_statement.payload.refs)


@dataclass(frozen=True)
class InputResolutionResult:
    """Normalized Arrow carriers and lineage returned by input preparation."""

    resolved_inputs: NormalizedExecutionInputs = field(
        default_factory=NormalizedExecutionInputs
    )
    source_metadata_lineage: SourceMetadataLineage = field(
        default_factory=SourceMetadataLineage
    )
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.resolved_inputs, NormalizedExecutionInputs):
            raise TypeError("resolved_inputs must be NormalizedExecutionInputs")
        object.__setattr__(
            self,
            "resolved_inputs",
            NormalizedExecutionInputs(
                inputs=_snapshot_dataset_mapping(
                    self.resolved_inputs.inputs,
                    name="resolved_inputs.inputs",
                ),
                preserve_memory_presentation=bool(
                    self.resolved_inputs.preserve_memory_presentation
                ),
            ),
        )
        if not isinstance(self.source_metadata_lineage, SourceMetadataLineage):
            raise TypeError(
                "source_metadata_lineage must be SourceMetadataLineage"
            )
        object.__setattr__(
            self,
            "diagnostics",
            tuple(self.diagnostics),
        )

    @property
    def inputs(self) -> Mapping[str, DataSetRef]:
        """Convenience view of the normalized Arrow carrier mapping."""

        return self.resolved_inputs.inputs

    @property
    def preserve_memory_presentation(self) -> bool:
        """Whether explicit memory inputs keep their public presentation."""

        return self.resolved_inputs.preserve_memory_presentation

    @property
    def normalized_inputs(self) -> NormalizedExecutionInputs:
        """Descriptive alias for callers that name the Arrow carrier directly."""

        return self.resolved_inputs

    @property
    def lineage(self) -> SourceMetadataLineage:
        """Short alias for the ordered source metadata snapshot."""

        return self.source_metadata_lineage


@dataclass(frozen=True)
class InputPreparationRequest:
    """Immutable inputs for materializing one already-planned rewrite.

    Planning and materialization intentionally have separate request types.  A
    preparation service may shape carriers and apply input-stage policy, but it
    must consume the exact :class:`RewritePlan` supplied by the plan owner.
    """

    ast_statements: tuple[ParsedStatement, ...]
    input_resolution: InputResolutionResult
    rewrite_plan: RewritePlan
    by_keys: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        statements = tuple(self.ast_statements)
        if not all(isinstance(statement, ParsedStatement) for statement in statements):
            raise TypeError("ast_statements must contain ParsedStatement values")
        object.__setattr__(self, "ast_statements", statements)
        if not isinstance(self.input_resolution, InputResolutionResult):
            raise TypeError("input_resolution must be InputResolutionResult")
        if not isinstance(self.rewrite_plan, RewritePlan):
            raise TypeError("rewrite_plan must be an immutable RewritePlan")
        if self.by_keys is not None:
            by_keys = tuple(self.by_keys)
            if not all(isinstance(name, str) and name for name in by_keys):
                raise TypeError("by_keys must contain non-empty strings")
            object.__setattr__(self, "by_keys", by_keys)

    @property
    def input_resolution_result(self) -> InputResolutionResult:
        """Descriptive alias retained for stage consumers."""

        return self.input_resolution

    @property
    def resolved_inputs(self) -> NormalizedExecutionInputs:
        """Convenience view of the normalized source carriers."""

        return self.input_resolution.resolved_inputs

    @property
    def source_metadata_lineage(self) -> SourceMetadataLineage:
        """Lineage captured by resolution, before carrier materialization."""

        return self.input_resolution.source_metadata_lineage


@dataclass(frozen=True)
class TypedInputPreparationRequest:
    """Internal typed request for input-owner materialization."""

    block: BlockSyntax
    helper_plan: TypedHelperPlan
    input_resolution: InputResolutionResult

    def __post_init__(self) -> None:
        if not isinstance(self.block, BlockSyntax):
            raise TypeError("block must be BlockSyntax")
        if not isinstance(self.helper_plan, TypedHelperPlan):
            raise TypeError("helper_plan must be TypedHelperPlan")
        if not isinstance(self.input_resolution, InputResolutionResult):
            raise TypeError("input_resolution must be InputResolutionResult")


@dataclass(frozen=True)
class TypedInputPreparationResult:
    """Prepared carriers and the structurally updated typed block."""

    block: BlockSyntax
    prepared: PreparedExecutionInputs = field(default_factory=PreparedExecutionInputs)
    pre_evaluation_targets: tuple[StatementExpressionSlot, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.prepared, PreparedExecutionInputs):
            raise TypeError("prepared must be PreparedExecutionInputs")
        object.__setattr__(
            self,
            "prepared",
            PreparedExecutionInputs(
                inputs=_snapshot_dataset_mapping(self.prepared.inputs, name="prepared.inputs"),
                preparation_kind=self.prepared.preparation_kind,
                source_metadata_lineage=_snapshot_lineage(
                    self.prepared.source_metadata_lineage
                ),
            ),
        )
        if not isinstance(self.block, BlockSyntax):
            raise TypeError("block must be BlockSyntax")
        targets = tuple(self.pre_evaluation_targets)
        if not all(isinstance(target, StatementExpressionSlot) for target in targets):
            raise TypeError("pre_evaluation_targets must contain expression slots")
        object.__setattr__(self, "pre_evaluation_targets", targets)
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))


@dataclass(frozen=True)
class TypedOutputShapingRequest:
    """Typed inputs for output projection planning."""

    block: BlockSyntax
    source: SourceDocument
    output_target_plan: "TypedOutputTargetPlan"
    resolved_inputs: Mapping[str, DataSetRef]
    helper_plan: TypedHelperPlan

    def __post_init__(self) -> None:
        if not isinstance(self.block, BlockSyntax):
            raise TypeError("block must be BlockSyntax")
        if not isinstance(self.source, SourceDocument):
            raise TypeError("source must be SourceDocument")
        if self.block.span.source_id != self.source.source_id:
            raise ValueError("block and source document ids must match")
        from .typed_output_planning import TypedOutputTargetPlan

        if not isinstance(self.output_target_plan, TypedOutputTargetPlan):
            raise TypeError("output_target_plan must be TypedOutputTargetPlan")
        if not isinstance(self.helper_plan, TypedHelperPlan):
            raise TypeError("helper_plan must be TypedHelperPlan")
        object.__setattr__(
            self,
            "resolved_inputs",
            _snapshot_dataset_mapping(self.resolved_inputs, name="resolved_inputs"),
        )


@dataclass(frozen=True)
class TypedOutputShapingResult:
    """Typed projection plan or diagnostics when output policy is invalid."""

    output_plan: Any | None = None
    diagnostics: tuple[Diagnostic, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        if self.diagnostics and self.output_plan is not None:
            raise ValueError("diagnostic shaping results cannot carry an output plan")


@dataclass(frozen=True)
class TypedOutputFinalizeRequest:
    """Typed inputs for post-runtime output shaping."""

    block: BlockSyntax
    output_target_plan: "TypedOutputTargetPlan"
    output_plan: Any
    resolved_inputs: Mapping[str, DataSetRef]
    outputs: Mapping[str, DataSetRef]
    helper_plan: TypedHelperPlan
    source_metadata_lineage: SourceMetadataLineage = field(
        default_factory=SourceMetadataLineage
    )
    preparation_kind: str = "none"
    preserve_memory_presentation: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.block, BlockSyntax):
            raise TypeError("block must be BlockSyntax")
        from .typed_output_planning import TypedOutputTargetPlan

        if not isinstance(self.output_target_plan, TypedOutputTargetPlan):
            raise TypeError("output_target_plan must be TypedOutputTargetPlan")
        if not isinstance(self.helper_plan, TypedHelperPlan):
            raise TypeError("helper_plan must be TypedHelperPlan")
        object.__setattr__(
            self,
            "resolved_inputs",
            _snapshot_dataset_mapping(self.resolved_inputs, name="resolved_inputs"),
        )
        object.__setattr__(
            self,
            "outputs",
            _snapshot_dataset_mapping(self.outputs, name="outputs"),
        )
        if not isinstance(self.source_metadata_lineage, SourceMetadataLineage):
            raise TypeError("source_metadata_lineage must be SourceMetadataLineage")
        object.__setattr__(
            self,
            "source_metadata_lineage",
            _snapshot_lineage(self.source_metadata_lineage),
        )
        if self.preparation_kind not in {"none", "prepared_set", "prepared_merge"}:
            raise ValueError("unsupported preparation_kind")
        object.__setattr__(
            self,
            "preserve_memory_presentation",
            bool(self.preserve_memory_presentation),
        )


@dataclass(frozen=True)
class InputPreparationResult:
    """Prepared carriers and diagnostics returned by the input owner."""

    prepared: PreparedExecutionInputs = field(default_factory=PreparedExecutionInputs)
    prepared_statements: tuple[ParsedStatement, ...] = field(default_factory=tuple)
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.prepared, PreparedExecutionInputs):
            raise TypeError("prepared must be PreparedExecutionInputs")
        object.__setattr__(
            self,
            "prepared",
            PreparedExecutionInputs(
                inputs=_snapshot_dataset_mapping(self.prepared.inputs, name="prepared.inputs"),
                preparation_kind=self.prepared.preparation_kind,
                source_metadata_lineage=_snapshot_lineage(
                    self.prepared.source_metadata_lineage
                ),
            ),
        )
        statements = tuple(self.prepared_statements)
        if not all(isinstance(statement, ParsedStatement) for statement in statements):
            raise TypeError("prepared_statements must contain ParsedStatement values")
        object.__setattr__(self, "prepared_statements", statements)
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def prepared_inputs(self) -> PreparedExecutionInputs:
        """Alias used by consumers that name the carrier explicitly."""

        return self.prepared

    @property
    def inputs(self) -> Mapping[str, DataSetRef]:
        """Convenience view of the prepared dataset mapping."""

        return self.prepared.inputs


@dataclass(frozen=True)
class OutputHandoffResult:
    """Finalized outputs and their public/lazy presentations."""

    outputs: Mapping[str, DataSetRef] = field(default_factory=_empty_mapping)
    public_outputs: Mapping[str, DataSetRef] = field(default_factory=_empty_mapping)
    outputs_arrow: Mapping[str, Any] = field(default_factory=_empty_mapping)
    output_source_metadata_lineage: SourceMetadataLineage = field(
        default_factory=SourceMetadataLineage
    )
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "outputs",
            _snapshot_dataset_mapping(self.outputs, name="outputs"),
        )
        object.__setattr__(
            self,
            "public_outputs",
            _snapshot_dataset_mapping(self.public_outputs, name="public_outputs"),
        )
        if not isinstance(self.outputs_arrow, Mapping):
            raise TypeError("outputs_arrow must be a mapping")
        object.__setattr__(
            self,
            "outputs_arrow",
            MappingProxyType(self.outputs_arrow),
        )
        if not isinstance(
            self.output_source_metadata_lineage, SourceMetadataLineage
        ):
            raise TypeError(
                "output_source_metadata_lineage must be SourceMetadataLineage"
            )
        object.__setattr__(
            self,
            "output_source_metadata_lineage",
            _snapshot_lineage(self.output_source_metadata_lineage),
        )
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))


@dataclass(frozen=True)
class TypedPlanBuildRequest:
    """Settled typed execution facts consumed by the final plan owner."""

    block: BlockSyntax
    prepared: PreparedExecutionInputs
    helper_plan: TypedHelperPlan
    output_target_plan: "TypedOutputTargetPlan"
    output_handoff_plan: Any
    slot_registry: Mapping[str, Mapping[str, int]]
    strategy_decision: ExecutionStrategyDecision

    def __post_init__(self) -> None:
        if not isinstance(self.block, BlockSyntax):
            raise TypeError("block must be a BlockSyntax")
        if not isinstance(self.prepared, PreparedExecutionInputs):
            raise TypeError("prepared must be PreparedExecutionInputs")
        if not isinstance(self.helper_plan, TypedHelperPlan):
            raise TypeError("helper_plan must be TypedHelperPlan")
        from .typed_output_planning import TypedOutputTargetPlan

        if not isinstance(self.output_target_plan, TypedOutputTargetPlan):
            raise TypeError("output_target_plan must be TypedOutputTargetPlan")
        if self.output_target_plan.diagnostics:
            raise ValueError("cannot build a plan from an invalid typed output plan")
        if not isinstance(self.strategy_decision, ExecutionStrategyDecision):
            raise TypeError("strategy_decision must be ExecutionStrategyDecision")
        object.__setattr__(
            self,
            "slot_registry",
            _snapshot_value(self.slot_registry),
        )


@dataclass(frozen=True)
class TypedPlanBuildResult:
    """Final typed execution plan and its owning diagnostics."""

    execution_plan: Mapping[str, Any] | None = None
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.execution_plan is not None:
            if not isinstance(self.execution_plan, Mapping):
                raise TypeError("execution_plan must be a mapping or None")
            object.__setattr__(self, "execution_plan", _snapshot_value(self.execution_plan))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))


@dataclass(frozen=True)
class ExecutionObservations:
    """Request-scoped observations collected up to the last reached stage."""

    strategy_decision: ExecutionStrategyDecision | None = None
    runtime_backend: str | None = None
    phase_metrics: Mapping[str, float] = field(default_factory=_empty_mapping)
    prepared_inputs: PreparedExecutionInputs = field(
        default_factory=PreparedExecutionInputs
    )
    output_source_metadata_lineage: SourceMetadataLineage = field(
        default_factory=SourceMetadataLineage
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "phase_metrics",
            _snapshot_mapping(self.phase_metrics, name="phase_metrics"),
        )
        object.__setattr__(
            self,
            "prepared_inputs",
            _snapshot_prepared_inputs(self.prepared_inputs),
        )
        object.__setattr__(
            self,
            "output_source_metadata_lineage",
            _snapshot_lineage(self.output_source_metadata_lineage),
        )


@dataclass(frozen=True)
class PipelineExecutionResult:
    """Pipeline response plus state the facade may commit exactly once."""

    response: ExecuteResponse
    catalog_updates: Mapping[str, DataSetRef] = field(default_factory=_empty_mapping)
    observations: ExecutionObservations = field(default_factory=ExecutionObservations)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "catalog_updates",
            _snapshot_dataset_mapping(
                self.catalog_updates, name="catalog_updates"
            ),
        )
        if not isinstance(self.observations, ExecutionObservations):
            raise TypeError("observations must be ExecutionObservations")


__all__ = [
    "ExecutionObservations",
    "HelperColumnPlan",
    "InputPreparationRequest",
    "InputPreparationResult",
    "TypedInputPreparationRequest",
    "TypedInputPreparationResult",
    "InputResolutionResult",
    "OutputHandoffResult",
    "TypedOutputPlanRequest",
    "PipelineExecutionRequest",
    "PipelineExecutionResult",
    "StrategyDecisionRequest",
    "StrategyDecisionResult",
    "TypedValidationRequest",
    "ValidationResult",
]
