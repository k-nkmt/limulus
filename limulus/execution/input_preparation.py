"""Backend-neutral input preparation for SET/MERGE rows and dataset options."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import re
from typing import Any, cast

import pyarrow as pa
import pyarrow.compute as pc

from ..arrow_bridge import (
    MergeSchemaOwnership,
    analyze_merge_schema_ownership,
    materialize_merge_carrier,
)
from ..dataset_option_expressions import (
    DatasetWhereEvaluationResult,
    WhereComparison,
    WhereExpression,
    WhereLogicalExpression,
    WhereNotExpression,
    WhereNullPredicate,
    evaluate_dataset_where,
)
from ..io import ExecutorIOService
from ..io_adapters import (
    DataFrameAdapterPandas,
    DataInputAdapterArrow,
    DataOutputAdapterArrow,
)
from ..naming import _column_key
from ..models import (
    DataSetRef,
    Diagnostic,
    NormalizedExecutionInputs,
    PreparationKind,
    PreparedExecutionInputs,
    PreparationProvenance,
    SourceMetadataLineage,
    SourceMetadataSnapshot,
)
from ..parser import (
    DatasetReference,
    DatasetReferenceOptionSpec,
    ParsedStatement,
    SetStatementOptionSpec,
)
from ..syntax.models import (
    BlockSyntax,
    DataStatementSyntax,
    DatasetOptionSyntax,
    DatasetReferenceSyntax,
    ExpressionSyntax,
    LiteralSyntax,
    NameListStatementSyntax,
    SetStatementOptionSyntax,
    SourceStatementSyntax,
    UnarySyntax,
)
from .contracts import (
    InputPreparationRequest,
    InputPreparationResult,
    InputResolutionResult,
    TypedInputPreparationRequest,
    TypedInputPreparationResult,
    TypedInputResolutionRequest,
)
from .rewrites import (
    RewritePlan,
    TypedHelperPlan,
    build_by_rows_precondition_failed,
    build_internal_var_collision,
    build_merge_duplicate_column,
    build_row_sort_key,
    build_runtime_set_not_found,
    is_source_owned_lag_lead_spec,
    lag_lead_default_value,
    materialize_lag_lead_columns,
    prepare_merge_rows,
    prepare_set_rows,
    sortable_value,
    target_index_for_spec,
)


@dataclass(frozen=True)
class _ArrowMergeKeyPart:
    """One exact, hashable BY component from an Arrow scalar."""

    data_type: str
    value: object | None
    is_null: bool


@dataclass(frozen=True)
class _ArrowMergeKey:
    """A typed lexicographic MERGE BY key with raw temporal values."""

    parts: tuple[_ArrowMergeKeyPart, ...]


@dataclass(frozen=True)
class ArrowMergeSelectionPlan:
    """Source row selection for one BY MERGE carrier.

    The plan contains only source occurrence names, exact keys, and row
    indices.  It deliberately owns no Arrow arrays and no Python row values;
    a later carrier builder can gather each source column from these indices.
    ``None`` denotes a typed missing row, while ``present_by_source`` keeps
    IN= distinct from a carried-forward row index.
    """

    source_names: tuple[str, ...]
    output_keys: tuple[_ArrowMergeKey, ...]
    row_indices_by_source: tuple[tuple[int | None, ...], ...]
    present_by_source: tuple[tuple[bool, ...], ...]

    @property
    def row_count(self) -> int:
        if self.output_keys:
            return len(self.output_keys)
        return max((len(indices) for indices in self.row_indices_by_source), default=0)


def _arrow_merge_key_sort_key(key: _ArrowMergeKey) -> tuple[tuple[int, object], ...]:
    return tuple(
        (1, 0) if part.is_null else (0, part.value)
        for part in key.parts
    )


def _arrow_merge_type_is_nested(data_type: Any) -> bool:
    return any(
        predicate(data_type)
        for predicate in (
            pa.types.is_list,
            pa.types.is_large_list,
            pa.types.is_fixed_size_list,
            pa.types.is_struct,
            pa.types.is_map,
        )
    )


def _arrow_source_backed_type_is_nested(data_type: Any) -> bool:
    return any(
        predicate(data_type)
        for predicate in (
            pa.types.is_list,
            pa.types.is_large_list,
            pa.types.is_fixed_size_list,
            pa.types.is_struct,
        )
    )


def _arrow_merge_scalar_key_value(scalar: Any, data_type: Any) -> object | None:
    if not getattr(scalar, "is_valid", True):
        return None
    if pa.types.is_timestamp(data_type) or pa.types.is_time32(data_type) or pa.types.is_time64(data_type):
        # Arrow's Python scalar exposes timestamp/time as datetime/time.  Cast
        # through the physical int64 representation so BY never loses unit,
        # pre-epoch values, or sub-microsecond identity.
        return int(scalar.cast(pa.int64()).as_py())
    return scalar.as_py()


def _arrow_take_column(
    column: Any,
    row_indices: Sequence[int | None],
) -> Any:
    """Take source rows while preserving the Arrow column's physical type."""

    indices = pa.array(row_indices, type=pa.int64())
    return pc.take(column, indices)


def _arrow_combine_taken_chunks(array: Any) -> Any:
    if isinstance(array, pa.ChunkedArray):
        return array.combine_chunks()
    return array


def _arrow_type_is_null(data_type: Any) -> bool:
    return data_type == pa.null()


def _arrow_fields_are_compatible(expected: pa.Field, actual: pa.Field) -> bool:
    """Compare source fields while allowing case-only top-level renames."""

    return expected.with_name(actual.name).equals(actual, check_metadata=True)


def normalize_execution_inputs(
    inputs: Mapping[str, DataSetRef],
) -> tuple[NormalizedExecutionInputs | None, Diagnostic | None]:
    """Normalize supported input carriers to the Arrow execution contract.

    This is intentionally an input-stage helper.  It preserves the source
    ``location`` and metadata on the normalized shell while keeping the
    presentation preference outside the native transport mapping.
    """

    if not isinstance(inputs, Mapping):
        return None, Diagnostic(
            code="RUNTIME_RUST_BRIDGE_INPUT_NOT_ARROW",
            severity="error",
            location="dataset:<unknown>",
            message=(
                "Unable to normalize execution inputs to Arrow: expected a mapping of "
                f"DataSetRef values, got {type(inputs).__name__}."
            ),
        )

    preserve_memory_presentation = bool(inputs) and all(
        isinstance(dataset_ref, DataSetRef)
        and str(dataset_ref.kind).strip().lower() == "memory"
        for dataset_ref in inputs.values()
    )

    normalized: dict[str, DataSetRef] = {}
    for dataset_name, dataset_ref in inputs.items():
        source_location = (
            str(dataset_ref.location)
            if isinstance(dataset_ref, DataSetRef) and dataset_ref.location
            else f"dataset://{dataset_name}"
        )
        try:
            if not isinstance(dataset_ref, DataSetRef):
                raise TypeError(
                    "input value must be a DataSetRef, "
                    f"got {type(dataset_ref).__name__}"
                )

            def _mapping_sequence_to_arrow(
                sequence: Sequence[Mapping[str, Any]],
            ) -> Any:
                normalized_rows = [dict(item) for item in sequence]
                all_keys: list[str] = []
                seen_keys: set[str] = set()
                for row in normalized_rows:
                    for key in row:
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        all_keys.append(key)
                stabilized_rows = [
                    {key: row.get(key) for key in all_keys}
                    for row in normalized_rows
                ]
                return pa.Table.from_pylist(stabilized_rows)

            kind = str(dataset_ref.kind).strip().lower()
            payload = dataset_ref.payload
            if kind == "arrow_table":
                if isinstance(payload, pa.Table):
                    arrow_payload = payload
                elif isinstance(payload, Sequence) and not isinstance(
                    payload, (str, bytes, bytearray)
                ):
                    if not all(isinstance(item, Mapping) for item in payload):
                        raise TypeError(
                            "arrow_table sequence payload must be a sequence of mappings"
                        )
                    arrow_payload = _mapping_sequence_to_arrow(payload)
                elif payload is not None and hasattr(payload, "__arrow_c_stream__"):
                    arrow_payload = payload
                else:
                    raise TypeError(
                        "arrow_table payload must expose __arrow_c_stream__"
                    )
            elif kind == "polars":
                try:
                    import polars as pl
                except ImportError as error:
                    raise TypeError("polars is required for polars input format") from error
                if not isinstance(payload, pl.DataFrame):
                    raise TypeError("polars input requires a polars.DataFrame payload")
                arrow_payload = payload.to_arrow()
            elif kind == "pandas":
                try:
                    import pandas as pd
                except ImportError as error:
                    raise TypeError("pandas is required for pandas input format") from error
                if not isinstance(payload, pd.DataFrame):
                    raise TypeError("pandas input requires a pandas.DataFrame payload")
                arrow_payload = pa.Table.from_pandas(payload)
            elif kind == "memory":
                if payload is None:
                    rows: list[Mapping[str, Any]] = []
                elif isinstance(payload, Sequence) and not isinstance(
                    payload, (str, bytes, bytearray)
                ):
                    if not all(isinstance(item, Mapping) for item in payload):
                        raise TypeError("memory input payload must be a sequence of mappings")
                    rows = list(payload)
                else:
                    raise TypeError("memory input payload must be a sequence of mappings")
                arrow_payload = _mapping_sequence_to_arrow(rows)
            else:
                raise TypeError(f"unsupported input kind '{dataset_ref.kind}'")

            if not hasattr(arrow_payload, "__arrow_c_stream__"):
                raise TypeError("normalized payload does not expose __arrow_c_stream__")
            normalized[dataset_name] = DataSetRef(
                kind="arrow_table",
                location=dataset_ref.location,
                payload=arrow_payload,
                metadata=dataset_ref.metadata,
                # User-supplied provenance is origin information, not payload
                # derivation: normalization never resets it.
                preparation=dataset_ref.preparation,
            )
        except Exception as error:
            return None, Diagnostic(
                code="RUNTIME_RUST_BRIDGE_INPUT_NOT_ARROW",
                severity="error",
                location=f"dataset:{dataset_name}",
                message=(
                    f"Unable to normalize input '{dataset_name}' at '{source_location}' "
                    f"to Arrow: {error}"
                ),
            )

    return (
        NormalizedExecutionInputs(
            inputs=normalized,
            preserve_memory_presentation=preserve_memory_presentation,
        ),
        None,
    )


def _promote_mixed_string_tables(
    tables: Sequence[pa.Table],
    *,
    column_keys: set[str] | None = None,
) -> Sequence[pa.Table]:
    """Widen only shared string/large_string columns, after source options."""

    if len(tables) <= 1:
        return tables
    small_keys: set[str] = set()
    large_keys: set[str] = set()
    other_keys: set[str] = set()
    for table in tables:
        for field in table.schema:
            key = _column_key(field.name)
            if column_keys is not None and key not in column_keys:
                continue
            if pa.types.is_string(field.type):
                small_keys.add(key)
            elif pa.types.is_large_string(field.type):
                large_keys.add(key)
            elif not pa.types.is_null(field.type):
                other_keys.add(key)
    mixed_keys = (small_keys & large_keys) - other_keys
    if not mixed_keys:
        return tables

    promoted: list[pa.Table] = []
    for table in tables:
        for index, field in enumerate(table.schema):
            if _column_key(field.name) in mixed_keys and pa.types.is_string(field.type):
                target_field = field.with_type(pa.large_string())
                table = table.set_column(
                    index, target_field, table.column(index).cast(target_field.type, safe=True)
                )
        promoted.append(table)
    return promoted


class ExecutionInputPreparationService:
    """Authoritative owner of input lookup, Arrow normalization, and lineage."""

    def __init__(
        self,
        io_service: Any | None = None,
    ) -> None:
        self._io_service = io_service or ExecutorIOService(
            arrow_input=DataInputAdapterArrow(),
            arrow_output=DataOutputAdapterArrow(),
            pandas_adapter=DataFrameAdapterPandas(),
            supported_formats=("polars", "pandas", "arrow_table"),
        )

    def resolve_typed(
        self, request: TypedInputResolutionRequest
    ) -> InputResolutionResult:
        """Resolve structural SET/MERGE occurrences through the shared catalogs.

        Catalog priority, carrier normalization, lineage, and failure
        semantics are owned by this service; names and option snapshots come only
        from the typed source-reference nodes.
        """

        if not isinstance(request, TypedInputResolutionRequest):
            raise TypeError("request must be a TypedInputResolutionRequest")

        source_occurrences = request.source_occurrences
        source_names = tuple(str(reference.name) for reference in source_occurrences)
        synthetic_inputs = dict(request.synthetic_inputs)
        if not synthetic_inputs:
            synthetic_inputs = self._io_service.resolve_synthetic_inputs_for_names(
                requested_names=source_names,
                explicit_inputs=request.explicit_inputs,
                available_inputs=request.available_inputs,
                registered_tables=request.registered_tables,
                cache=None,
            )

        resolved_inputs, resolution_diagnostics = self._io_service.resolve_input_names(
            source_input_names=source_names,
            explicit_inputs=request.explicit_inputs,
            available_inputs=request.available_inputs,
            registered_tables=request.registered_tables,
            synthetic_inputs=synthetic_inputs,
        )
        if resolution_diagnostics:
            return self._failed_result(resolution_diagnostics)

        normalized, normalization_diagnostic = normalize_execution_inputs(
            resolved_inputs
        )
        if normalization_diagnostic is not None or normalized is None:
            diagnostics = (
                (normalization_diagnostic,)
                if normalization_diagnostic is not None
                else (
                    Diagnostic(
                        code="RUNTIME_RUST_BRIDGE_INPUT_NOT_ARROW",
                        severity="error",
                        location="dataset:<unknown>",
                        message="Unable to normalize input to Arrow.",
                    ),
                )
            )
            return self._failed_result(diagnostics)

        preserve_memory_presentation = bool(request.explicit_inputs) and all(
            str(dataset_ref.kind).strip().lower() == "memory"
            for dataset_ref in request.explicit_inputs.values()
        )
        normalized = NormalizedExecutionInputs(
            inputs=normalized.inputs,
            preserve_memory_presentation=preserve_memory_presentation,
        )
        lineage, lineage_diagnostic = capture_source_metadata_lineage(
            source_refs=source_occurrences,
            resolved_inputs=normalized.inputs,
        )
        if lineage_diagnostic is not None:
            return self._failed_result((lineage_diagnostic,))

        return InputResolutionResult(
            resolved_inputs=normalized,
            source_metadata_lineage=lineage,
            diagnostics=(),
        )

    def prepare(
        self,
        request: InputPreparationRequest,
    ) -> InputPreparationResult:
        """Materialize input-stage preparation from one immutable rewrite plan.

        This is the shared preparation core called by :meth:`prepare_typed`
        in the public execution path.

        This stage owns dataset-reference options, SET/MERGE input ordering,
        BY sorting, and preparation-carrier construction.  It deliberately
        does not instantiate or consult a planner: ``request.rewrite_plan`` is
        the decision produced by :class:`ExecutionPlanService` and is consumed
        as supplied.  SET and MERGE carrier transforms are returned with the
        prepared AST.
        """

        if not isinstance(request, InputPreparationRequest):
            raise TypeError("request must be InputPreparationRequest")
        if request.input_resolution.diagnostics:
            return self._failed_preparation_result(request.input_resolution.diagnostics)

        current_inputs = dict(request.resolved_inputs.inputs)
        source_metadata_lineage = request.source_metadata_lineage
        diagnostics: list[Diagnostic] = []
        source_statement = next(
            (
                statement
                for statement in request.ast_statements
                if statement.kind in {"SET", "MERGE"}
            ),
            None,
        )
        source_refs = list(getattr(source_statement, "dataset_refs", ()) or ())
        missing_sources = [
            str(source_ref.name)
            for source_ref in source_refs
            if _resolve_source_input_ref(current_inputs, str(source_ref.name)) is None
        ]
        if missing_sources:
            return self._failed_preparation_result(
                tuple(build_runtime_set_not_found(name) for name in missing_sources)
            )
        if not source_metadata_lineage.ordered_sources and source_refs:
            source_metadata_lineage, lineage_error = capture_source_metadata_lineage(
                source_refs=source_refs,
                resolved_inputs=current_inputs,
            )
            if lineage_error is not None:
                diagnostics.append(lineage_error)

        if diagnostics:
            return self._failed_preparation_result(diagnostics)

        if source_statement is not None and source_statement.kind == "MERGE":
            prepared_statements, prepared_inputs, merge_diagnostic = self._prepare_merge_for_execution(
                ast_statements=tuple(request.ast_statements),
                resolved_inputs=current_inputs,
                source_metadata_lineage=source_metadata_lineage,
                option_applier=try_apply_dataset_reference_options_to_arrow_table,
                by_keys=request.by_keys,
            )
            if merge_diagnostic is not None:
                if merge_diagnostic.severity == "error":
                    return self._failed_preparation_result((merge_diagnostic,))
                diagnostics.append(merge_diagnostic)
            explicit_kind = self._preparation_kind(prepared_inputs)
            conflict = self._aggregate_preparation_provenance(
                prepared_inputs, explicit_kind
            )
            if conflict is not None:
                return self._failed_preparation_result((conflict,))
            prepared = PreparedExecutionInputs(
                inputs=prepared_inputs,
                preparation_kind=explicit_kind,
                source_metadata_lineage=source_metadata_lineage,
            )
            return InputPreparationResult(
                prepared=prepared,
                prepared_statements=prepared_statements,
                diagnostics=tuple(diagnostics),
            )

        # This unit owns SET's carrier preprocessing and source statement
        # collapse; MERGE carriers and transformed statements are owned elsewhere.
        set_statement = next(
            (statement for statement in request.ast_statements if statement.kind == "SET"),
            None,
        )
        set_source_refs = list(getattr(set_statement, "dataset_refs", ()) or ())
        by_keys = (
            request.by_keys
            if request.by_keys is not None
            else self._extract_variable_list(request.ast_statements, "BY")
        )
        has_merge = any(statement.kind == "MERGE" for statement in request.ast_statements)
        has_lag_lead = getattr(request.rewrite_plan, "lag_lead_mode", "none") != "none"
        has_repeated_set_source_names = len(
            {_column_key(ref.name) for ref in set_source_refs}
        ) < len(set_source_refs)
        has_source_dataset_options = any(
            self._has_dataset_reference_options(getattr(ref, "options", None))
            for ref in set_source_refs
        )
        has_in_option_vars = any(
            getattr(getattr(ref, "options", None), "in_var", None) is not None
            for ref in set_source_refs
        )
        set_statement_options = getattr(set_statement, "statement_options", None)
        has_set_statement_options = bool(
            getattr(set_statement_options, "indsname_var", None)
            or getattr(set_statement_options, "end_var", None)
        )

        collision = self._detect_set_input_internal_variable_collisions(
            set_statement=set_statement,
            source_refs=set_source_refs,
            resolved_inputs=current_inputs,
            by_keys=by_keys,
        )
        if collision is not None:
            return self._failed_preparation_result((collision,))

        # Check shared SET schemas before concat.  Option-shaped sources are
        # checked by the Arrow option application below, after their shape is
        # known, matching the historical diagnostic ordering.
        if not has_source_dataset_options:
            set_schema_tables = self._arrow_set_schema_tables(
                source_refs=set_source_refs,
                resolved_inputs=current_inputs,
            )
            if set_schema_tables is not None:
                promoted_tables = _promote_mixed_string_tables(
                    [table for _, table in set_schema_tables]
                )
                for (name, table), promoted in zip(set_schema_tables, promoted_tables):
                    if promoted is not table:
                        current_inputs[name] = replace(current_inputs[name], payload=promoted)
                set_schema_tables = [
                    (name, table)
                    for (name, _), table in zip(set_schema_tables, promoted_tables)
                ]
                schema_diagnostic = self._set_schema_incompatibility_diagnostic(
                    set_schema_tables
                )
                if schema_diagnostic is not None:
                    return self._failed_preparation_result((schema_diagnostic,))

        if (
            not has_merge
            and not has_lag_lead
            and not by_keys
            and not has_repeated_set_source_names
            and not has_set_statement_options
        ):
            interleaved = self._try_interleave_set_inputs(
                request.ast_statements,
                current_inputs,
                by_keys,
            )
            if interleaved is not None:
                current_inputs = interleaved
        elif by_keys and not has_merge:
            sorted_inputs, sort_diagnostic = self._sort_inputs_by_by_keys(
                current_inputs, by_keys
            )
            if sort_diagnostic is not None:
                return self._failed_preparation_result((sort_diagnostic,))
            assert sorted_inputs is not None
            current_inputs = sorted_inputs

        # Source-local options, BY flags, and SET automatic variables are
        # materialized together so option ordering is fixed before any stream
        # interleave or row preparation can observe the source.
        should_prepare_set_rows = bool(
            set_source_refs
            and (
                has_repeated_set_source_names
                or has_source_dataset_options
                or has_in_option_vars
                or has_lag_lead
                or bool(by_keys)
                or has_set_statement_options
            )
        )
        if should_prepare_set_rows:
            prepared_inputs, prepare_error = self._prepare_set_rows_with_temporary_variables(
                ast_statements=request.ast_statements,
                resolved_inputs=current_inputs,
                by_keys=by_keys,
                has_lag_lead=has_lag_lead,
                option_applier=try_apply_dataset_reference_options_to_arrow_table,
            )
            if prepare_error is not None:
                return self._failed_preparation_result((prepare_error,))
            if prepared_inputs is not None:
                current_inputs = prepared_inputs

        explicit_kind = self._preparation_kind(current_inputs)
        conflict = self._aggregate_preparation_provenance(current_inputs, explicit_kind)
        if conflict is not None:
            return self._failed_preparation_result((conflict,))
        prepared = PreparedExecutionInputs(
            inputs=current_inputs,
            preparation_kind=explicit_kind,
            source_metadata_lineage=source_metadata_lineage,
        )
        return InputPreparationResult(
            prepared=prepared,
            prepared_statements=self._collapse_prepared_set_statement(
                ast_statements=tuple(request.ast_statements),
                resolved_inputs=current_inputs,
            ),
            diagnostics=tuple(diagnostics),
        )

    def prepare_typed(
        self,
        request: TypedInputPreparationRequest,
    ) -> TypedInputPreparationResult:
        """Materialize typed source options and helper columns through this owner.

        Convert structural source/input facts to preparation requests and
        delegate to :meth:`prepare`.
        """

        if not isinstance(request, TypedInputPreparationRequest):
            raise TypeError("request must be TypedInputPreparationRequest")
        targets = request.helper_plan.pre_evaluation_targets
        if request.input_resolution.diagnostics:
            return TypedInputPreparationResult(
                block=request.block,
                prepared=PreparedExecutionInputs(),
                pre_evaluation_targets=targets,
                diagnostics=request.input_resolution.diagnostics,
            )

        statements = self._typed_input_statements(request.block)
        by_keys = self._typed_by_keys(request.block)
        rewrite_plan = self._typed_input_rewrite_plan(request.helper_plan)
        base = self.prepare(
            InputPreparationRequest(
                ast_statements=statements,
                input_resolution=request.input_resolution,
                rewrite_plan=rewrite_plan,
                by_keys=by_keys,
            )
        )
        if base.diagnostics and any(
            diagnostic.severity == "error" for diagnostic in base.diagnostics
        ):
            return TypedInputPreparationResult(
                block=request.block,
                prepared=PreparedExecutionInputs(),
                pre_evaluation_targets=targets,
                diagnostics=base.diagnostics,
            )

        rewritten_inputs, rewrite_error = self._materialize_lag_lead_rewrite_inputs(
            ast_statements=base.prepared_statements,
            resolved_inputs=base.prepared.inputs,
            rewrite_plan=rewrite_plan,
            by_keys=by_keys,
        )
        if rewrite_error is not None:
            return TypedInputPreparationResult(
                block=request.block,
                prepared=PreparedExecutionInputs(),
                pre_evaluation_targets=targets,
                diagnostics=(*base.diagnostics, rewrite_error),
            )

        prepared = base.prepared
        if rewritten_inputs is not None:
            preparation_kind = self._preparation_kind(rewritten_inputs)
            conflict = self._aggregate_preparation_provenance(
                rewritten_inputs, preparation_kind
            )
            if conflict is not None:
                return TypedInputPreparationResult(
                    block=request.block,
                    prepared=PreparedExecutionInputs(),
                    pre_evaluation_targets=targets,
                    diagnostics=(*base.diagnostics, conflict),
                )
            prepared = PreparedExecutionInputs(
                inputs=rewritten_inputs,
                preparation_kind=cast(PreparationKind, preparation_kind),
                source_metadata_lineage=base.prepared.source_metadata_lineage,
            )

        return TypedInputPreparationResult(
            block=self._update_typed_prepared_source(
                request.block,
                original_statements=statements,
                prepared_statements=base.prepared_statements,
            ),
            prepared=prepared,
            pre_evaluation_targets=targets,
            diagnostics=base.diagnostics,
        )

    @staticmethod
    def _typed_input_statements(block: BlockSyntax) -> tuple[ParsedStatement, ...]:
        """Project only typed owner inputs; never read display/source text."""

        statements: list[ParsedStatement] = []
        for statement in block.statements:
            refs: tuple[DatasetReference, ...] = ()
            output_refs: tuple[DatasetReference, ...] = ()
            statement_options = SetStatementOptionSpec()
            payload = statement.payload
            if isinstance(payload, SourceStatementSyntax):
                refs = tuple(
                    DatasetReference(
                        name=reference.name,
                        options=DatasetReferenceOptionSpec(
                            in_var=reference.options.in_var,
                            keep_vars=reference.options.keep_vars,
                            drop_vars=reference.options.drop_vars,
                            where_expr=reference.options.where_expr,
                            rename_map=dict(reference.options.rename_map),
                            firstobs=reference.options.firstobs,
                            obs=reference.options.obs,
                            label=reference.options.label,
                        ),
                    )
                    for reference in payload.refs
                )
                statement_options = SetStatementOptionSpec(
                    indsname_var=payload.options.indsname_var,
                    end_var=payload.options.end_var,
                )
            elif isinstance(payload, DataStatementSyntax):
                output_refs = tuple(
                    DatasetReference(
                        name=reference.name,
                        options=DatasetReferenceOptionSpec(
                            in_var=reference.options.in_var,
                            keep_vars=reference.options.keep_vars,
                            drop_vars=reference.options.drop_vars,
                            where_expr=reference.options.where_expr,
                            rename_map=dict(reference.options.rename_map),
                            firstobs=reference.options.firstobs,
                            obs=reference.options.obs,
                            label=reference.options.label,
                        ),
                    )
                    for reference in payload.outputs
                )
            statements.append(
                ParsedStatement(
                    kind=statement.kind,
                    text="",
                    dataset_refs=refs,
                    output_refs=output_refs,
                    statement_options=statement_options,
                )
            )
        return tuple(statements)

    @staticmethod
    def _typed_by_keys(block: BlockSyntax) -> tuple[str, ...]:
        for statement in block.statements:
            if statement.kind == "BY" and isinstance(
                statement.payload, NameListStatementSyntax
            ):
                return tuple(name.name for name in statement.payload.names)
        return ()

    @staticmethod
    def _typed_input_rewrite_plan(helper_plan: TypedHelperPlan) -> RewritePlan:
        specs = tuple(
            {
                "function_name": spec.function_name,
                "argument_is_string_literal": spec.argument_is_string_literal,
                "source_name": spec.source_name,
                "offset": spec.offset,
                "is_source_owned": spec.is_source_owned,
                "typed_default_value": ExecutionInputPreparationService._typed_default_value(
                    spec.default_expression
                ),
                "helper_column": spec.helper_column,
            }
            for spec in helper_plan.lag_lead_specs
            if spec.is_source_owned
        )
        return RewritePlan(
            merge_mode=helper_plan.merge_mode,
            lag_lead_mode=helper_plan.lag_lead_mode,
            where_mode=helper_plan.where_mode,
            helper_source_columns=helper_plan.helper_source_columns,
            helper_runtime_columns=helper_plan.helper_runtime_columns,
            lag_lead_specs=specs,
        )

    @staticmethod
    def _typed_default_value(expression: ExpressionSyntax | None) -> Any:
        if expression is None:
            return None
        if isinstance(expression, UnarySyntax):
            if expression.operator != "-":
                return None
            operand = ExecutionInputPreparationService._typed_default_value(
                expression.operand
            )
            return -operand if isinstance(operand, (int, float)) else None
        if isinstance(expression, LiteralSyntax):
            if expression.kind == "missing":
                return None
            lexeme = expression.lexeme.strip()
            if expression.kind == "string":
                if (
                    len(lexeme) >= 2
                    and lexeme[0] in {"'", '"'}
                    and lexeme[-1] == lexeme[0]
                ):
                    # Match the existing source-owned helper default contract:
                    # preserve literal contents without interpreting escapes.
                    return lexeme[1:-1]
            if expression.kind not in {"numeric", "string"}:
                return None
            try:
                return int(lexeme)
            except ValueError:
                try:
                    return float(lexeme)
                except ValueError:
                    return None
        return None

    @classmethod
    def _update_typed_prepared_source(
        cls,
        block: BlockSyntax,
        *,
        original_statements: Sequence[ParsedStatement],
        prepared_statements: Sequence[ParsedStatement],
    ) -> BlockSyntax:
        """Mirror owner SET/MERGE source transformations structurally."""

        if len(original_statements) != len(prepared_statements):
            return block
        statements = list(block.statements)
        for index, statement in enumerate(statements):
            if not isinstance(statement.payload, SourceStatementSyntax):
                continue
            original = original_statements[index]
            prepared = prepared_statements[index]
            if original.kind not in {"SET", "MERGE"}:
                continue
            if prepared.kind == original.kind and prepared.dataset_refs == original.dataset_refs:
                continue

            source_payload = statement.payload
            remaining = list(source_payload.refs)
            typed_refs: list[DatasetReferenceSyntax] = []
            for reference in prepared.dataset_refs:
                match_index = next(
                    (
                        ref_index
                        for ref_index, candidate in enumerate(remaining)
                        if _column_key(candidate.name) == _column_key(reference.name)
                    ),
                    0 if remaining else None,
                )
                if match_index is None:
                    continue
                source_reference = remaining.pop(match_index)
                options = reference.options
                typed_refs.append(
                    replace(
                        source_reference,
                        name=reference.name,
                        options=DatasetOptionSyntax(
                            in_var=options.in_var,
                            keep_vars=options.keep_vars,
                            drop_vars=options.drop_vars,
                            where_expr=options.where_expr,
                            rename_map=tuple(options.rename_map.items()),
                            firstobs=options.firstobs,
                            obs=options.obs,
                            label=options.label,
                        ),
                    )
                )
            statements[index] = replace(
                statement,
                kind=prepared.kind,
                payload=SourceStatementSyntax(
                    refs=tuple(typed_refs),
                    options=SetStatementOptionSyntax(
                        indsname_var=prepared.statement_options.indsname_var,
                        end_var=prepared.statement_options.end_var,
                    ),
                ),
                legacy_display_span=None,
            )
        return replace(block, statements=tuple(statements))

    def project_prepared_internal_aliases_for_handoff(
        self,
        resolved_inputs: Mapping[str, DataSetRef],
    ) -> dict[str, DataSetRef]:
        """Drop duplicate case aliases from prepared Arrow carriers.

        Preparation retains both case variants of automatic BY variables so
        the planner can describe the physical carrier.  The native cursor
        resolves names case-insensitively, so only the first physical alias
        may cross the final handoff boundary.
        """

        projected_inputs: dict[str, DataSetRef] = {}
        for name, dataset_ref in resolved_inputs.items():
            payload = dataset_ref.payload
            provenance = getattr(dataset_ref, "preparation", None)
            if (
                dataset_ref.kind.strip().lower() != "arrow_table"
                or not isinstance(payload, pa.Table)
                or getattr(provenance, "kind", "none") == "none"
            ):
                projected_inputs[name] = dataset_ref
                continue

            # Case-preserving internal names come from the typed contract only;
            # ``location`` is never parsed for control information.
            internal_names = tuple(
                getattr(provenance, "internal_variable_names", ()) or ()
            )
            internal_keys = {_column_key(item) for item in internal_names}
            if not internal_keys:
                projected_inputs[name] = dataset_ref
                continue

            retained_indices: list[int] = []
            seen_internal_keys: set[str] = set()
            for index, column_name in enumerate(payload.schema.names):
                column_key = _column_key(column_name)
                if column_key in internal_keys:
                    if column_key in seen_internal_keys:
                        continue
                    seen_internal_keys.add(column_key)
                retained_indices.append(index)

            if len(retained_indices) == len(payload.schema.names):
                projected_inputs[name] = dataset_ref
                continue

            projected_inputs[name] = DataSetRef(
                kind="arrow_table",
                location=dataset_ref.location,
                payload=payload.select(retained_indices),
                metadata=dataset_ref.metadata,
                # Physical projection never rewrites provenance: the carrier
                # stays the same prepared carrier with the same internal names.
                preparation=dataset_ref.preparation,
            )

        return projected_inputs

    def _materialize_lag_lead_rewrite_inputs(
        self,
        *,
        ast_statements: Sequence[ParsedStatement],
        resolved_inputs: Mapping[str, DataSetRef],
        rewrite_plan: Any,
        by_keys: Sequence[str] | None = None,
    ) -> tuple[dict[str, DataSetRef] | None, Diagnostic | None]:
        lag_lead_specs = tuple(rewrite_plan.lag_lead_specs)
        if not lag_lead_specs:
            return None, None

        set_statement = next(
            (statement for statement in ast_statements if statement.kind == "SET"),
            None,
        )
        if set_statement is None:
            return None, None

        source_refs = list(getattr(set_statement, "dataset_refs", ()) or ())
        if not source_refs:
            return None, None
        by_keys = (
            tuple(by_keys)
            if by_keys is not None
            else self._extract_variable_list(ast_statements, "BY")
        )
        first_source_name = source_refs[0].name
        source_input = _resolve_source_input_ref(resolved_inputs, first_source_name)
        if source_input is None:
            return None, Diagnostic(
                code="RUNTIME_SET_DATASET_NOT_FOUND",
                severity="error",
                message=f"Input dataset is not provided: {first_source_name}",
            )

        source_names = {_column_key(str(ref.name)) for ref in source_refs}
        columnar_prepared_input = self._materialize_lag_lead_arrow_input_if_supported(
            source_refs=source_refs,
            by_keys=by_keys,
            source_name=first_source_name,
            source_input=source_input,
            lag_lead_specs=lag_lead_specs,
        )
        if columnar_prepared_input is not None:
            updated_inputs = {
                name: dataset
                for name, dataset in resolved_inputs.items()
                if _column_key(str(name)) not in source_names
            }
            updated_inputs[first_source_name] = columnar_prepared_input
            return updated_inputs, None

        existing_rows: list[dict[str, Any]] = []
        source_inputs_missing = [
            ref.name
            for ref in source_refs[1:]
            if _resolve_source_input_ref(resolved_inputs, ref.name) is None
        ]
        if source_inputs_missing:
            loaded_rows, load_error = self._io_service.load_input_rows(source_input)
            if load_error is not None:
                return None, load_error
            existing_rows = loaded_rows
        else:
            for source_ref in source_refs:
                input_ref = _resolve_source_input_ref(resolved_inputs, source_ref.name)
                if input_ref is None:
                    return None, Diagnostic(
                        code="RUNTIME_SET_DATASET_NOT_FOUND",
                        severity="error",
                        message=f"Input dataset is not provided: {source_ref.name}",
                    )
                loaded_rows, load_error = self._io_service.load_input_rows(input_ref)
                if load_error is not None:
                    return None, load_error
                existing_rows.extend(loaded_rows)

        if by_keys and len(existing_rows) > 1:
            existing_rows.sort(
                key=lambda row: build_row_sort_key(
                    row,
                    by_keys=by_keys,
                    resolve_row_value=self._resolve_row_value,
                )
            )

        if existing_rows and all(
            all(spec["helper_column"] in row for spec in lag_lead_specs)
            for row in existing_rows
        ):
            return dict(resolved_inputs), None

        rewritten_rows = materialize_lag_lead_columns(
            rows=existing_rows,
            specs=lag_lead_specs,
        )
        prepared_input = self._coerce_rewrite_input(
            source_name=first_source_name,
            source_input=source_input,
            rewritten_rows=rewritten_rows,
            internal_variable_names={spec["helper_column"] for spec in lag_lead_specs},
        )
        updated_inputs = {
            name: dataset
            for name, dataset in resolved_inputs.items()
            if _column_key(str(name)) not in source_names
        }
        updated_inputs[first_source_name] = prepared_input
        return updated_inputs, None

    def _materialize_lag_lead_arrow_input_if_supported(
        self,
        *,
        source_refs: Sequence[Any],
        by_keys: Sequence[str],
        source_name: str,
        source_input: DataSetRef,
        lag_lead_specs: Sequence[Mapping[str, Any]],
    ) -> DataSetRef | None:
        if source_input.kind != "arrow_table":
            return None

        is_prepared_carrier = (
            getattr(getattr(source_input, "preparation", None), "kind", "none")
            == "prepared_set"
        )
        if not is_prepared_carrier:
            if len(source_refs) != 1:
                return None
            option_spec = getattr(source_refs[0], "options", None)
            if option_spec is not None and (
                getattr(option_spec, "keep_vars", ())
                or getattr(option_spec, "drop_vars", ())
                or getattr(option_spec, "rename_map", {})
                or getattr(option_spec, "where_expr", None)
                or getattr(option_spec, "firstobs", None) is not None
                or getattr(option_spec, "obs", None) is not None
            ):
                return None

        table = source_input.payload
        if not hasattr(table, "schema") or not hasattr(table, "column"):
            return None

        source_owned_specs = tuple(
            spec for spec in lag_lead_specs if is_source_owned_lag_lead_spec(spec)
        )
        if not source_owned_specs:
            return None

        schema_names = tuple(getattr(table.schema, "names", ()) or ())
        if schema_names and all(spec["helper_column"] in schema_names for spec in source_owned_specs):
            return source_input

        if by_keys:
            resolved_sort_keys: list[tuple[str, str]] = []
            for by_key in by_keys:
                resolved_name = self._resolve_arrow_table_column_name(table, by_key)
                if resolved_name is None:
                    return None
                resolved_sort_keys.append((resolved_name, "ascending"))
            try:
                table = table.sort_by(resolved_sort_keys)
            except Exception:
                return None

        materialized_columns: list[tuple[str, Any]] = []
        for spec in source_owned_specs:
            resolved_source_name = self._resolve_arrow_table_column_name(
                table, spec["source_name"]
            )
            if resolved_source_name is None:
                return None
            values = table.column(resolved_source_name).combine_chunks().to_pylist()
            default_value = lag_lead_default_value(spec)
            helper_values = []
            for index in range(len(values)):
                target_index = target_index_for_spec(index=index, spec=spec)
                helper_values.append(
                    values[target_index] if 0 <= target_index < len(values) else default_value
                )
            materialized_columns.append((spec["helper_column"], pa.array(helper_values)))

        rewritten_table = table
        for helper_name, helper_array in materialized_columns:
            if helper_name in rewritten_table.schema.names:
                field_index = rewritten_table.schema.get_field_index(helper_name)
                rewritten_table = rewritten_table.set_column(
                    field_index, helper_name, helper_array
                )
            else:
                rewritten_table = rewritten_table.append_column(helper_name, helper_array)

        return self._coerce_rewrite_arrow_input(
            source_name=source_name,
            source_input=source_input,
            rewritten_table=rewritten_table,
            internal_variable_names={spec["helper_column"] for spec in source_owned_specs},
        )

    @staticmethod
    def _failed_preparation_result(
        diagnostics: Sequence[Diagnostic],
    ) -> InputPreparationResult:
        """Return an empty carrier whenever preparation fails."""

        return InputPreparationResult(
            prepared=PreparedExecutionInputs(),
            prepared_statements=(),
            diagnostics=tuple(diagnostics),
        )

    @staticmethod
    def _preparation_kind(inputs: Mapping[str, DataSetRef]) -> str:
        """Aggregate the typed carrier kinds into one request kind.

        ``none`` carriers do not participate; a single distinct prepared kind
        becomes the aggregate.  Mixed kinds are never silently resolved here:
        the aggregate validator owns that conflict.
        """

        observed_kinds: set[str] = set()
        for dataset_ref in inputs.values():
            provenance = getattr(dataset_ref, "preparation", None)
            kind = getattr(provenance, "kind", "none")
            if kind != "none":
                observed_kinds.add(str(kind))
        if len(observed_kinds) == 1:
            return next(iter(observed_kinds))
        return "none"

    @staticmethod
    def _aggregate_preparation_provenance(
        inputs: Mapping[str, DataSetRef],
        explicit_kind: str,
    ) -> Diagnostic | None:
        """Validate typed carrier provenance against the declared preparation kind.

        The aggregate is computed only from typed ``PreparationProvenance``
        values: no marker inference and no MERGE preference participate.  Mixed
        carrier kinds, or an aggregate that disagrees with the declared kind,
        are rejected with an input-preparation diagnostic instead of being
        silently overwritten.
        """

        observed_kinds: set[str] = set()
        for name, dataset_ref in inputs.items():
            provenance = getattr(dataset_ref, "preparation", None)
            kind = getattr(provenance, "kind", "none")
            if kind != "none":
                observed_kinds.add(str(kind))

        if not observed_kinds:
            aggregate_kind = "none"
        elif len(observed_kinds) == 1:
            aggregate_kind = next(iter(observed_kinds))
        else:
            return Diagnostic(
                code="RUNTIME_PREPARATION_PROVENANCE_CONFLICT",
                severity="error",
                location=f"datasets:{','.join(sorted(inputs))}" if inputs else "",
                message=(
                    "Prepared carriers mix preparation kinds "
                    f"{sorted(observed_kinds)} (datasets: "
                    f"{', '.join(sorted(inputs))}); request declares "
                    f"'{explicit_kind}'."
                ),
            )

        if aggregate_kind != explicit_kind:
            return Diagnostic(
                code="RUNTIME_PREPARATION_PROVENANCE_CONFLICT",
                severity="error",
                message=(
                    f"Aggregate prepared kind '{aggregate_kind}' does not match "
                    f"declared preparation kind '{explicit_kind}'."
                ),
            )
        return None

    def _prepare_merge_for_execution(
        self,
        *,
        ast_statements: tuple[ParsedStatement, ...],
        resolved_inputs: Mapping[str, DataSetRef],
        source_metadata_lineage: SourceMetadataLineage,
        option_applier: Any,
        by_keys: Sequence[str] | None = None,
    ) -> tuple[tuple[ParsedStatement, ...], dict[str, DataSetRef], Diagnostic | None]:
        merge_statement = next(
            (statement for statement in ast_statements if statement.kind == "MERGE"),
            None,
        )
        if merge_statement is None:
            return ast_statements, dict(resolved_inputs), None

        source_refs = list(getattr(merge_statement, "dataset_refs", ()) or ())
        if not source_refs:
            return ast_statements, dict(resolved_inputs), None

        by_keys = (
            tuple(by_keys)
            if by_keys is not None
            else self._extract_variable_list(ast_statements, "BY")
        )
        schema_ownership: MergeSchemaOwnership | None = None
        if len(source_metadata_lineage.ordered_sources) == len(source_refs):
            schema_ownership = analyze_merge_schema_ownership(
                source_metadata_lineage,
                by_keys=by_keys,
            )
            if schema_ownership.ambiguous_non_by_fields:
                duplicate_label = ", ".join(schema_ownership.ambiguous_non_by_fields)
                return (
                    ast_statements,
                    dict(resolved_inputs),
                    build_merge_duplicate_column(duplicate_label),
                )

        in_option_vars = [
            ref.options.in_var
            for ref in source_refs
            if getattr(getattr(ref, "options", None), "in_var", None) is not None
        ]
        statement_options = getattr(merge_statement, "statement_options", None)
        internal_variable_names = self._collect_internal_variable_names(
            in_option_vars=in_option_vars,
            indsname_var=getattr(statement_options, "indsname_var", None),
            end_var=getattr(statement_options, "end_var", None),
            by_keys=by_keys,
        )

        prepared_arrow_input, prepared_arrow_diagnostic = self._prepare_merge_arrow_input_if_supported(
            source_refs=source_refs,
            by_keys=by_keys,
            resolved_inputs=resolved_inputs,
            internal_variable_names=internal_variable_names,
            schema_ownership=schema_ownership,
            option_applier=option_applier,
        )
        if prepared_arrow_diagnostic is not None and (
            prepared_arrow_input is None
            or prepared_arrow_diagnostic.severity == "error"
        ):
            return ast_statements, dict(resolved_inputs), prepared_arrow_diagnostic
        if prepared_arrow_input is not None:
            first_source_name = str(source_refs[0].name)
            source_names = {_column_key(str(source_ref.name)) for source_ref in source_refs}
            updated_inputs = {
                name: dataset
                for name, dataset in resolved_inputs.items()
                if _column_key(str(name)) not in source_names
            }
            updated_inputs[first_source_name] = prepared_arrow_input
            replacement_statement = replace(
                merge_statement,
                kind="SET",
                text=f"set {first_source_name}",
                dataset_refs=(DatasetReference(name=first_source_name),),
            )
            updated_statements = tuple(
                replacement_statement if statement is merge_statement else statement
                for statement in ast_statements
            )
            return updated_statements, updated_inputs, prepared_arrow_diagnostic

        option_source = next(
            (
                ref
                for ref in source_refs
                if self._has_dataset_reference_options(getattr(ref, "options", None))
            ),
            None,
        )
        if option_source is not None:
            return (
                ast_statements,
                dict(resolved_inputs),
                self._dataset_option_arrow_required_diagnostic(option_source),
            )

        merge_resolved_inputs = dict(resolved_inputs)
        for source_ref in source_refs:
            source_name = str(source_ref.name)
            source_input = _resolve_source_input_ref(resolved_inputs, source_name)
            if source_input is not None:
                merge_resolved_inputs[source_name] = source_input

        merged_rows_with_source, merge_error = prepare_merge_rows(
            source_refs=source_refs,
            by_keys=by_keys,
            resolved_inputs=merge_resolved_inputs,
            internal_variable_names=internal_variable_names,
            load_input_rows=self._io_service.load_input_rows,
            resolve_row_key=self._resolve_row_key,
            resolve_row_value=self._resolve_row_value,
        )
        if merge_error is not None:
            return ast_statements, dict(resolved_inputs), merge_error

        merged_rows = [
            {key: value for key, value in dict(row).items() if value is not None}
            for _, row, _ in merged_rows_with_source
        ]
        first_source_name = str(source_refs[0].name)
        prepared_input = self._coerce_prepared_merge_input(
            source_name=first_source_name,
            merged_rows=merged_rows,
            source_inputs=resolved_inputs,
            internal_variable_names=internal_variable_names,
        )
        source_names = {_column_key(str(source_ref.name)) for source_ref in source_refs}
        updated_inputs = {
            name: dataset
            for name, dataset in resolved_inputs.items()
            if _column_key(str(name)) not in source_names
        }
        updated_inputs[first_source_name] = prepared_input
        replacement_statement = replace(
            merge_statement,
            kind="SET",
            text=f"set {first_source_name}",
            dataset_refs=(DatasetReference(name=first_source_name),),
        )
        updated_statements = tuple(
            replacement_statement if statement is merge_statement else statement
            for statement in ast_statements
        )
        return updated_statements, updated_inputs, None

    def _prepare_merge_arrow_input_if_supported(
        self,
        *,
        source_refs: Sequence[Any],
        by_keys: Sequence[str],
        resolved_inputs: Mapping[str, DataSetRef],
        internal_variable_names: set[str],
        schema_ownership: MergeSchemaOwnership | None = None,
        option_applier: Any,
    ) -> tuple[DataSetRef | None, Diagnostic | None]:
        if not by_keys:
            return self._prepare_merge_arrow_input_without_by(
                source_refs=source_refs,
                resolved_inputs=resolved_inputs,
                internal_variable_names=internal_variable_names,
                schema_ownership=schema_ownership,
                option_applier=try_apply_dataset_reference_options_to_arrow_table,
            )

        normalized_by_keys = {_column_key(key) for key in by_keys}
        source_infos: list[dict[str, Any]] = []
        canonical_by_names: dict[str, str] = {}
        non_key_columns_by_source: list[set[str]] = []
        by_field_observations: dict[str, list[tuple[str, str, str]]] = {
            _column_key(by_key): [] for by_key in by_keys
        }

        for source_ref in source_refs:
            input_ref = _resolve_source_input_ref(resolved_inputs, str(source_ref.name))
            if input_ref is None:
                return None, None
            if input_ref.kind != "arrow_table" or not isinstance(input_ref.payload, pa.Table):
                return None, None

            table, option_error = option_applier(
                input_ref.payload,
                source_name=source_ref.name,
                option_spec=source_ref.options,
            )
            if option_error is not None:
                return None, option_error
            if table is None:
                return None, None

            allow_internal_names = getattr(
                getattr(input_ref, "preparation", None), "kind", "none"
            ) == "prepared_merge"
            collision = self._detect_internal_variable_collision_in_schema(
                column_names=tuple(getattr(getattr(table, "schema", None), "names", ()) or ()),
                source_name=source_ref.name,
                internal_variable_names=internal_variable_names,
                allow_internal_names=allow_internal_names,
            )
            if collision is not None:
                return None, collision

            resolved_by_names: list[str] = []
            for by_key in by_keys:
                resolved_name = self._resolve_arrow_table_column_name(table, by_key)
                if resolved_name is None:
                    return None, build_by_precondition_failed(by_key, source_ref.name)
                canonical_by_names.setdefault(_column_key(by_key), resolved_name)
                resolved_by_names.append(resolved_name)

            non_key_columns = {
                _column_key(column_name)
                for column_name in getattr(table, "column_names", ())
                if _column_key(column_name) not in normalized_by_keys
            }
            non_key_columns_by_source.append(non_key_columns)
            source_infos.append(
                {
                    "source_name": str(source_ref.name),
                    "in_var": getattr(source_ref.options, "in_var", None),
                    "table": table,
                    "resolved_by_names": tuple(resolved_by_names),
                }
            )

        promoted_tables = _promote_mixed_string_tables(
            [info["table"] for info in source_infos], column_keys=normalized_by_keys
        )
        for info, table in zip(source_infos, promoted_tables):
            info["table"] = table
            for by_key, resolved_name in zip(by_keys, info["resolved_by_names"]):
                by_field_observations[_column_key(by_key)].append(
                    (
                        info["source_name"],
                        resolved_name,
                        str(table.schema.field(resolved_name).type),
                    )
                )

        for by_key in by_keys:
            by_schema_diagnostic = self._merge_by_schema_incompatibility_diagnostic(
                by_key=by_key,
                observations=by_field_observations.get(_column_key(by_key), ()),
            )
            if by_schema_diagnostic is not None:
                return None, by_schema_diagnostic

        duplicate_columns: set[str] = set()
        for left_index in range(len(non_key_columns_by_source)):
            for right_index in range(left_index + 1, len(non_key_columns_by_source)):
                duplicate_columns.update(
                    non_key_columns_by_source[left_index].intersection(
                        non_key_columns_by_source[right_index]
                    )
                )
        if duplicate_columns:
            duplicate_label = ", ".join(sorted(duplicate_columns))
            return None, build_merge_duplicate_column(duplicate_label)

        selection_plan, selection_diagnostic = self._build_arrow_merge_by_selection_plan(
            source_specs=source_infos,
            by_keys=by_keys,
        )
        if selection_diagnostic is not None and (
            selection_plan is None or selection_diagnostic.severity == "error"
        ):
            return None, selection_diagnostic
        assert selection_plan is not None
        output_by_keys = tuple(
            canonical_by_names.get(_column_key(key), key) for key in by_keys
        )

        output_arrays: list[Any] = []
        output_fields: list[pa.Field] = []
        for index, output_name in enumerate(output_by_keys):
            owner_sources: list[int] = []
            owner_row_indices: list[int] = []
            for output_row_index in range(selection_plan.row_count):
                for source_index, selected_rows in enumerate(
                    selection_plan.row_indices_by_source
                ):
                    selected_row_index = selected_rows[output_row_index]
                    if selected_row_index is None:
                        continue
                    owner_sources.append(source_index)
                    owner_row_indices.append(selected_row_index)
                    break
            source_column_names = tuple(
                info["resolved_by_names"][index] for info in source_infos
            )
            source_field = self._select_arrow_output_field(
                source_infos=source_infos,
                source_column_names=source_column_names,
            )
            if source_field is None:
                return None, Diagnostic(
                    code="RUNTIME_MERGE_ARROW_GATHER_FAILED",
                    severity="error",
                    location=f"column:{output_name}/selection:field",
                    message=f"No source Field owns MERGE column '{output_name}'.",
                )
            gathered, gather_error = self._gather_arrow_column_from_owners(
                source_infos=source_infos,
                owner_sources=owner_sources,
                owner_row_indices=owner_row_indices,
                source_column_names=source_column_names,
                output_column_name=output_name,
                output_type=source_field.type,
            )
            if gather_error is not None:
                return None, gather_error
            assert gathered is not None
            output_arrays.append(gathered)
            output_fields.append(
                pa.field(
                    output_name,
                    source_field.type,
                    nullable=source_field.nullable
                    or any(
                        key.parts[index].is_null
                        for key in selection_plan.output_keys
                    ),
                    metadata=(
                        dict(source_field.metadata)
                        if source_field.metadata is not None
                        else None
                    ),
                )
            )

        for source_index, info in enumerate(source_infos):
            selected_rows = selection_plan.row_indices_by_source[source_index]
            for column_name in getattr(info["table"], "column_names", ()):
                if _column_key(column_name) in normalized_by_keys:
                    continue
                gathered, gather_error = self._gather_arrow_column_for_selection(
                    source_name=str(info["source_name"]),
                    column_name=str(column_name),
                    column=info["table"].column(column_name),
                    row_indices=selected_rows,
                )
                if gather_error is not None:
                    return None, gather_error
                assert gathered is not None
                output_arrays.append(gathered)
                source_field = info["table"].schema.field(column_name)
                output_fields.append(
                    pa.field(
                        source_field.name,
                        source_field.type,
                        nullable=source_field.nullable or any(
                            row_index is None for row_index in selected_rows
                        ),
                        metadata=(
                            dict(source_field.metadata)
                            if source_field.metadata is not None
                            else None
                        ),
                    )
                )

        emitted_in_vars: set[str] = set()
        for source_index, info in enumerate(source_infos):
            in_var = info["in_var"]
            if not in_var or _column_key(in_var) in emitted_in_vars:
                continue
            emitted_in_vars.add(_column_key(in_var))
            output_arrays.append(
                pa.array(
                    [
                        1
                        if present
                        else 0
                        for present in selection_plan.present_by_source[source_index]
                    ],
                    type=pa.int8(),
                )
            )
            output_fields.append(pa.field(in_var, pa.int8(), nullable=False))

        for by_key_index, by_key in enumerate(by_keys):
            first_values = [
                1
                if output_row_index == 0
                or selection_plan.output_keys[output_row_index].parts[by_key_index]
                != selection_plan.output_keys[output_row_index - 1].parts[by_key_index]
                else 0
                for output_row_index in range(selection_plan.row_count)
            ]
            last_values = [
                1
                if output_row_index == selection_plan.row_count - 1
                or selection_plan.output_keys[output_row_index].parts[by_key_index]
                != selection_plan.output_keys[output_row_index + 1].parts[by_key_index]
                else 0
                for output_row_index in range(selection_plan.row_count)
            ]
            for prefix, values in (
                ("FIRST", first_values),
                ("LAST", last_values),
                ("first", first_values),
                ("last", last_values),
            ):
                output_arrays.append(pa.array(values, type=pa.int8()))
                output_fields.append(pa.field(f"{prefix}.{by_key}", pa.int8(), nullable=False))

        first_source_table = source_infos[0]["table"]
        output_table = pa.Table.from_arrays(
            output_arrays,
            schema=pa.schema(
                output_fields,
                metadata=first_source_table.schema.metadata,
            ),
        )
        if schema_ownership is not None:
            output_table = materialize_merge_carrier(
                output_table,
                type_hints=schema_ownership.type_hints,
                column_order=schema_ownership.column_order,
            )

        first_source_name = str(source_refs[0].name)
        first_source_input = _resolve_source_input_ref(resolved_inputs, first_source_name)
        if first_source_input is None:
            return None, None
        return (
            DataSetRef(
                kind="arrow_table",
                location=first_source_input.location,
                payload=output_table,
                metadata=first_source_input.metadata,
                preparation=self._prepared_provenance(
                    first_source_input, "prepared_merge", internal_variable_names
                ),
            ),
            selection_diagnostic,
        )

    @classmethod
    def _build_arrow_merge_by_selection_plan(
        cls,
        *,
        source_specs: Sequence[Mapping[str, Any]],
        by_keys: Sequence[str],
    ) -> tuple[ArrowMergeSelectionPlan | None, Diagnostic | None]:
        """Build exact BY alignment without materializing data columns.

        ``source_specs`` contains one entry per MERGE occurrence with a table,
        resolved BY column names, and source name.  Only BY scalars are
        inspected to build groups; non-key columns remain in their Arrow
        arrays for the gather stage.
        """

        source_names: list[str] = []
        grouped_sources: list[dict[_ArrowMergeKey, tuple[int, ...]]] = []
        canonical_key_types: list[str | None] = [None] * len(by_keys)
        for source_spec in source_specs:
            table = source_spec.get("table")
            resolved_by_names = tuple(source_spec.get("resolved_by_names", ()) or ())
            if not isinstance(table, pa.Table) or len(resolved_by_names) != len(by_keys):
                continue
            for key_index, column_name in enumerate(resolved_by_names):
                data_type = table.schema.field(column_name).type
                if _arrow_type_is_null(data_type):
                    continue
                canonical_key_types[key_index] = str(data_type)

        for source_spec in source_specs:
            source_name = str(source_spec.get("source_name", "<unknown>"))
            table = source_spec.get("table")
            resolved_by_names = tuple(source_spec.get("resolved_by_names", ()) or ())
            if not isinstance(table, pa.Table) or len(resolved_by_names) != len(by_keys):
                return None, Diagnostic(
                    code="RUNTIME_BY_PRECONDITION_FAILED",
                    severity="error",
                    location=f"dataset:{source_name}",
                    message=(
                        f"MERGE BY selection could not resolve all keys for source "
                        f"'{source_name}'."
                    ),
                )

            source_names.append(source_name)
            groups: dict[_ArrowMergeKey, list[int]] = {}
            row_count = int(getattr(table, "num_rows", 0) or 0)
            for row_index in range(row_count):
                key_parts: list[_ArrowMergeKeyPart] = []
                for key_index, (by_key, column_name) in enumerate(
                    zip(by_keys, resolved_by_names)
                ):
                    column = table.column(column_name)
                    data_type = column.type
                    if _arrow_merge_type_is_nested(data_type):
                        return None, Diagnostic(
                            code="RUNTIME_BY_PRECONDITION_FAILED",
                            severity="error",
                            location=f"dataset:{source_name}",
                            message=(
                                f"MERGE BY key '{by_key}' in source '{source_name}' "
                                "requires unsupported nested value interpretation."
                            ),
                        )
                    try:
                        scalar = column[row_index]
                        value = _arrow_merge_scalar_key_value(scalar, data_type)
                    except Exception as error:
                        return None, Diagnostic(
                            code="RUNTIME_BY_PRECONDITION_FAILED",
                            severity="error",
                            location=f"dataset:{source_name}",
                            message=(
                                f"MERGE BY key '{by_key}' in source '{source_name}' "
                                f"could not be read exactly: {type(error).__name__}: {error}."
                            ),
                        )
                    key_parts.append(
                        _ArrowMergeKeyPart(
                            data_type=canonical_key_types[key_index] or str(data_type),
                            value=value,
                            is_null=not getattr(scalar, "is_valid", True),
                        )
                    )

                key = _ArrowMergeKey(parts=tuple(key_parts))
                try:
                    groups.setdefault(key, []).append(row_index)
                except TypeError as error:
                    return None, Diagnostic(
                        code="RUNTIME_BY_PRECONDITION_FAILED",
                        severity="error",
                        location=f"dataset:{source_name}",
                        message=(
                            f"MERGE BY key in source '{source_name}' is not hashable "
                            f"without value conversion: {type(error).__name__}: {error}."
                        ),
                    )
            grouped_sources.append(
                {key: tuple(indices) for key, indices in groups.items()}
            )

        all_keys = {
            key
            for source_groups in grouped_sources
            for key in source_groups
        }
        try:
            ordered_keys = tuple(sorted(all_keys, key=_arrow_merge_key_sort_key))
        except Exception as error:
            return None, Diagnostic(
                code="RUNTIME_BY_PRECONDITION_FAILED",
                severity="error",
                location="statement:merge",
                message=(
                    "MERGE BY keys could not be ordered exactly: "
                    f"{type(error).__name__}: {error}."
                ),
            )

        output_keys: list[_ArrowMergeKey] = []
        row_indices_by_source: list[list[int | None]] = [
            [] for _ in grouped_sources
        ]
        present_by_source: list[list[bool]] = [[] for _ in grouped_sources]
        repeated_source_indices: set[int] = set()
        for key in ordered_keys:
            group_lengths = [len(groups.get(key, ())) for groups in grouped_sources]
            repeated_for_key = {
                source_index
                for source_index, group_length in enumerate(group_lengths)
                if group_length >= 2
            }
            if len(repeated_for_key) >= 2:
                repeated_source_indices.update(repeated_for_key)
            group_row_count = max(group_lengths, default=0)
            for group_offset in range(group_row_count):
                output_keys.append(key)
                for source_index, groups in enumerate(grouped_sources):
                    source_rows = groups.get(key, ())
                    if not source_rows:
                        row_indices_by_source[source_index].append(None)
                        present_by_source[source_index].append(False)
                        continue
                    row_indices_by_source[source_index].append(
                        source_rows[min(group_offset, len(source_rows) - 1)]
                    )
                    present_by_source[source_index].append(True)

        selection_plan = ArrowMergeSelectionPlan(
            source_names=tuple(source_names),
            output_keys=tuple(output_keys),
            row_indices_by_source=tuple(
                tuple(indices) for indices in row_indices_by_source
            ),
            present_by_source=tuple(
                tuple(present) for present in present_by_source
            ),
        )
        if len(repeated_source_indices) < 2:
            return selection_plan, None

        repeated_sources = tuple(
            source_names[index] for index in sorted(repeated_source_indices)
        )
        return selection_plan, Diagnostic(
            code="RUNTIME_MERGE_REPEATED_BY_VALUES",
            severity="warning",
            location="statement:merge",
            message=(
                "MERGE BY values repeat in multiple sources for BY columns "
                f"{', '.join(by_keys)}; affected sources: "
                f"{', '.join(repeated_sources)}. Existing occurrence alignment "
                "was retained."
            ),
        )

    @classmethod
    def _build_arrow_merge_ordinal_selection_plan(
        cls,
        *,
        source_specs: Sequence[Mapping[str, Any]],
    ) -> ArrowMergeSelectionPlan:
        """Build BY-less MERGE occurrence alignment from row ordinals only."""

        source_names: list[str] = []
        row_counts: list[int] = []
        for source_spec in source_specs:
            source_names.append(str(source_spec.get("source_name", "<unknown>")))
            table = source_spec.get("table")
            row_counts.append(int(getattr(table, "num_rows", 0) or 0))

        output_row_count = max(row_counts, default=0)
        row_indices_by_source = tuple(
            tuple(row_index if row_index < row_count else None for row_index in range(output_row_count))
            for row_count in row_counts
        )
        present_by_source = tuple(
            tuple(row_index < row_count for row_index in range(output_row_count))
            for row_count in row_counts
        )
        return ArrowMergeSelectionPlan(
            source_names=tuple(source_names),
            output_keys=(),
            row_indices_by_source=row_indices_by_source,
            present_by_source=present_by_source,
        )

    @staticmethod
    def _select_arrow_output_field(
        *,
        source_infos: Sequence[Mapping[str, Any]],
        source_column_names: Sequence[str | None],
    ) -> pa.Field | None:
        """Choose a concrete Field when another provider is null-typed."""

        observed_fields: list[pa.Field] = []
        for source_info, column_name in zip(source_infos, source_column_names):
            if column_name is None:
                continue
            observed_fields.append(source_info["table"].schema.field(column_name))
        for field in observed_fields:
            if not _arrow_type_is_null(field.type):
                return field
        return observed_fields[0] if observed_fields else None

    @classmethod
    def _gather_arrow_column_for_selection(
        cls,
        *,
        source_name: str,
        column_name: str,
        column: Any,
        row_indices: Sequence[int | None],
    ) -> tuple[Any | None, Diagnostic | None]:
        """Gather one source column and keep typed NULL missing positions."""

        try:
            return _arrow_take_column(column, row_indices), None
        except Exception as error:
            return None, Diagnostic(
                code="RUNTIME_MERGE_ARROW_GATHER_FAILED",
                severity="error",
                location=f"dataset:{source_name}/column:{column_name}/selection:take",
                message=(
                    f"Arrow take failed for MERGE selection in source '{source_name}', "
                    f"column '{column_name}': {type(error).__name__}: {error}."
                ),
            )

    @classmethod
    def _gather_arrow_column_from_owners(
        cls,
        *,
        source_infos: Sequence[Mapping[str, Any]],
        owner_sources: Sequence[int | None],
        owner_row_indices: Sequence[int | None],
        source_column_names: Sequence[str | None],
        output_column_name: str,
        output_type: Any | None = None,
    ) -> tuple[Any | None, Diagnostic | None]:
        """Gather a BY column from contiguous source-owner runs."""

        if not owner_sources:
            first_source_index = next(
                (
                    index
                    for index, column_name in enumerate(source_column_names)
                    if column_name is not None
                ),
                None,
            )
            if first_source_index is None:
                return None, Diagnostic(
                    code="RUNTIME_MERGE_ARROW_GATHER_FAILED",
                    severity="error",
                    location=f"column:{output_column_name}",
                    message=(
                        f"No source occurrence owns MERGE column '{output_column_name}'."
                    ),
                )
            first_column = source_infos[first_source_index]["table"].column(
                source_column_names[first_source_index]
            )
            empty_type = output_type if output_type is not None else first_column.type
            return pa.array([], type=empty_type), None

        segments: list[Any] = []
        run_start = 0
        while run_start < len(owner_sources):
            owner = owner_sources[run_start]
            run_end = run_start + 1
            while run_end < len(owner_sources) and owner_sources[run_end] == owner:
                run_end += 1
            if owner is None:
                first_source_index = next(
                    (
                        index
                        for index, column_name in enumerate(source_column_names)
                        if column_name is not None
                    ),
                    None,
                )
                if first_source_index is None:
                    return None, Diagnostic(
                        code="RUNTIME_MERGE_ARROW_GATHER_FAILED",
                        severity="error",
                        location=f"column:{output_column_name}",
                        message=(
                            f"No source occurrence owns MERGE column '{output_column_name}'."
                        ),
                    )
                reference_column = source_infos[first_source_index]["table"].column(
                    source_column_names[first_source_index]
                )
                missing_type = (
                    output_type if output_type is not None else reference_column.type
                )
                segments.append(
                    pa.array(
                        [None] * (run_end - run_start),
                        type=missing_type,
                    )
                )
                run_start = run_end
                continue
            source_info = source_infos[owner]
            source_column_name = source_column_names[owner]
            if source_column_name is None:
                return None, Diagnostic(
                    code="RUNTIME_MERGE_ARROW_GATHER_FAILED",
                    severity="error",
                    location=f"dataset:{source_info.get('source_name', '<unknown>')}/column:{output_column_name}",
                    message=(
                        f"Source occurrence does not contain MERGE column '{output_column_name}'."
                    ),
                )
            source_column = source_info["table"].column(source_column_name)
            segment, error = cls._gather_arrow_column_for_selection(
                source_name=str(source_info.get("source_name", "<unknown>")),
                column_name=output_column_name,
                column=source_column,
                row_indices=owner_row_indices[run_start:run_end],
            )
            if error is not None:
                return None, error
            assert segment is not None
            try:
                combined_segment = _arrow_combine_taken_chunks(segment)
                if (
                    output_type is not None
                    and _arrow_type_is_null(combined_segment.type)
                    and not _arrow_type_is_null(output_type)
                ):
                    combined_segment = combined_segment.cast(output_type)
                segments.append(combined_segment)
            except Exception as error:
                return None, Diagnostic(
                    code="RUNTIME_MERGE_ARROW_GATHER_FAILED",
                    severity="error",
                    location=(
                        f"dataset:{source_info.get('source_name', '<unknown>')}"
                        f"/column:{output_column_name}/selection:combine_chunks"
                    ),
                    message=(
                        f"Arrow chunks could not be combined for MERGE selection in "
                        f"source '{source_info.get('source_name', '<unknown>')}', "
                        f"column '{output_column_name}': {type(error).__name__}: {error}."
                    ),
                )
            run_start = run_end

        try:
            if len(segments) == 1:
                return segments[0], None
            return pa.concat_arrays(segments), None
        except Exception as error:
            source_aliases = ", ".join(
                str(source_info.get("source_name", "<unknown>"))
                for source_info in source_infos
            )
            return None, Diagnostic(
                code="RUNTIME_MERGE_ARROW_GATHER_FAILED",
                severity="error",
                location=(
                    f"dataset:{source_aliases}/column:{output_column_name}"
                    "/selection:combine"
                ),
                message=(
                    f"Arrow gather segments could not be combined for column "
                    f"'{output_column_name}' from sources '{source_aliases}': "
                    f"{type(error).__name__}: {error}."
                ),
            )

    def _prepare_merge_arrow_input_without_by(
        self,
        *,
        source_refs: Sequence[Any],
        resolved_inputs: Mapping[str, DataSetRef],
        internal_variable_names: set[str],
        schema_ownership: MergeSchemaOwnership | None = None,
        option_applier: Any,
    ) -> tuple[DataSetRef | None, Diagnostic | None]:
        """Prepare option-shaped Arrow MERGE occurrences by ordinal position."""

        first_source_name = str(source_refs[0].name)
        first_source_input = _resolve_source_input_ref(resolved_inputs, first_source_name)
        if first_source_input is None:
            return None, None

        prepared_sources: list[dict[str, Any]] = []
        ordered_columns: list[str] = []
        seen_columns: set[str] = set()
        in_vars: list[str] = []

        # Preserve occurrence identity: the same dataset may be referenced
        # repeatedly with different shaping and IN= options.
        for occurrence_index, source_ref in enumerate(source_refs):
            input_ref = _resolve_source_input_ref(resolved_inputs, str(source_ref.name))
            if input_ref is None:
                return None, None
            if input_ref.kind != "arrow_table" or not isinstance(input_ref.payload, pa.Table):
                return None, None

            table, option_error = option_applier(
                input_ref.payload,
                source_name=source_ref.name,
                option_spec=source_ref.options,
            )
            if option_error is not None:
                return None, option_error
            if table is None:
                return None, None

            allow_internal_names = getattr(
                getattr(input_ref, "preparation", None), "kind", "none"
            ) == "prepared_merge"
            collision = self._detect_internal_variable_collision_in_schema(
                column_names=tuple(getattr(getattr(table, "schema", None), "names", ()) or ()),
                source_name=source_ref.name,
                internal_variable_names=internal_variable_names,
                allow_internal_names=allow_internal_names,
            )
            if collision is not None:
                return None, collision

            for column_name in getattr(table, "column_names", ()):
                column_key = _column_key(column_name)
                if column_key in seen_columns:
                    continue
                seen_columns.add(column_key)
                ordered_columns.append(column_name)

            in_var = getattr(source_ref.options, "in_var", None)
            if in_var and in_var not in in_vars:
                in_vars.append(in_var)
            prepared_sources.append(
                {
                    "occurrence_index": occurrence_index,
                    "source_name": str(source_ref.name),
                    "in_var": in_var,
                    "table": table,
                }
            )

        for in_var in in_vars:
            if _column_key(in_var) not in seen_columns:
                seen_columns.add(_column_key(in_var))
                ordered_columns.append(in_var)

        ordinal_selection_plan = self._build_arrow_merge_ordinal_selection_plan(
            source_specs=prepared_sources,
        )
        compatibility_error = self._validate_ordinal_field_compatibility(
            source_infos=prepared_sources,
            ordered_columns=ordered_columns,
            in_vars=in_vars,
        )
        if compatibility_error is not None:
            return None, compatibility_error
        output_arrays: list[Any] = []
        output_fields: list[pa.Field] = []
        emitted_in_vars: set[str] = set()

        def append_in_var(in_var: str) -> None:
            in_var_key = _column_key(in_var)
            if in_var_key in emitted_in_vars:
                return
            emitted_in_vars.add(in_var_key)
            values = [0] * ordinal_selection_plan.row_count
            for source_index, source in enumerate(prepared_sources):
                if source["in_var"] != in_var:
                    continue
                values = [
                    1 if present else 0
                    for present in ordinal_selection_plan.present_by_source[source_index]
                ]
            output_arrays.append(pa.array(values, type=pa.int8()))
            output_fields.append(pa.field(in_var, pa.int8(), nullable=False))

        for column_name in ordered_columns:
            matching_in_var = next(
                (
                    in_var
                    for in_var in in_vars
                    if _column_key(in_var) == _column_key(column_name)
                ),
                None,
            )
            if matching_in_var is not None:
                append_in_var(matching_in_var)
                continue

            source_column_names: list[str | None] = []
            owner_sources: list[int | None] = [
                None
            ] * ordinal_selection_plan.row_count
            owner_row_indices: list[int | None] = [
                None
            ] * ordinal_selection_plan.row_count
            owner_field: pa.Field | None = None
            for source_index, source in enumerate(prepared_sources):
                resolved_column_name = self._resolve_arrow_table_column_name(
                    source["table"],
                    column_name,
                )
                source_column_names.append(resolved_column_name)
                if resolved_column_name is None:
                    continue
                for row_index, selected_row_index in enumerate(
                    ordinal_selection_plan.row_indices_by_source[source_index]
                ):
                    if selected_row_index is None:
                        continue
                    # Later occurrences intentionally overwrite earlier ones,
                    # matching the historical ordinal MERGE row contract.
                    owner_sources[row_index] = source_index
                    owner_row_indices[row_index] = selected_row_index

            owner_field = self._select_arrow_output_field(
                source_infos=prepared_sources,
                source_column_names=source_column_names,
            )
            if owner_field is None:
                return None, Diagnostic(
                    code="RUNTIME_MERGE_ARROW_GATHER_FAILED",
                    severity="error",
                    location=f"column:{column_name}",
                    message=f"No source occurrence owns MERGE column '{column_name}'.",
                )
            gathered, gather_error = self._gather_arrow_column_from_owners(
                source_infos=prepared_sources,
                owner_sources=owner_sources,
                owner_row_indices=owner_row_indices,
                source_column_names=source_column_names,
                output_column_name=column_name,
                output_type=owner_field.type,
            )
            if gather_error is not None:
                return None, gather_error
            assert gathered is not None
            output_arrays.append(gathered)
            output_fields.append(
                pa.field(
                    column_name,
                    owner_field.type,
                    nullable=owner_field.nullable
                    or any(owner is None for owner in owner_sources),
                    metadata=(
                        dict(owner_field.metadata)
                        if owner_field.metadata is not None
                        else None
                    ),
                )
            )

        for in_var in in_vars:
            append_in_var(in_var)

        output_table = pa.Table.from_arrays(
            output_arrays,
            schema=pa.schema(
                output_fields,
                metadata=prepared_sources[0]["table"].schema.metadata,
            ),
        )
        if schema_ownership is not None:
            output_table = materialize_merge_carrier(
                output_table,
                type_hints=schema_ownership.type_hints,
                column_order=schema_ownership.column_order,
            )

        return DataSetRef(
            kind="arrow_table",
            location=first_source_input.location,
            payload=output_table,
            metadata=first_source_input.metadata,
            preparation=self._prepared_provenance(
                first_source_input, "prepared_merge", internal_variable_names
            ),
        ), None

    @classmethod
    def _validate_ordinal_field_compatibility(
        cls,
        *,
        source_infos: Sequence[Mapping[str, Any]],
        ordered_columns: Sequence[str],
        in_vars: Sequence[str],
    ) -> Diagnostic | None:
        """Require one physical Arrow type for each shared ordinal field."""

        in_var_keys = {_column_key(name) for name in in_vars}
        for column_name in ordered_columns:
            if _column_key(column_name) in in_var_keys:
                continue
            observations: list[tuple[str, pa.Field]] = []
            for source_info in source_infos:
                resolved_name = cls._resolve_arrow_table_column_name(
                    source_info["table"],
                    column_name,
                )
                if resolved_name is None:
                    continue
                observations.append(
                    (
                        str(source_info.get("source_name", "<unknown>")),
                        source_info["table"].schema.field(resolved_name),
                    )
                )
            concrete_observations = [
                observation
                for observation in observations
                if not _arrow_type_is_null(observation[1].type)
            ]
            observed_types = {str(field.type) for _, field in concrete_observations}
            if len(observed_types) <= 1:
                if len(concrete_observations) <= 1 or not concrete_observations:
                    continue
                first_source, first_field = concrete_observations[0]
                if not _arrow_source_backed_type_is_nested(first_field.type):
                    continue
                incompatible = next(
                    (
                        item
                        for item in concrete_observations[1:]
                        if not _arrow_fields_are_compatible(first_field, item[1])
                    ),
                    None,
                )
                if incompatible is None:
                    continue
                second_source, second_field = incompatible
                return Diagnostic(
                    code="RUNTIME_MERGE_FIELD_INCOMPATIBLE",
                    severity="error",
                    location=f"dataset:{first_source}/column:{column_name}/selection:ordinal",
                    message=(
                        f"Ordinal MERGE column '{column_name}' has incompatible "
                        f"source Fields in '{first_source}' and '{second_source}': "
                        f"{first_field} vs {second_field}."
                    ),
                )
            first_source, first_field = concrete_observations[0]
            second_source, second_field = next(
                (
                    item
                    for item in concrete_observations[1:]
                    if str(item[1].type) != str(first_field.type)
                ),
                concrete_observations[1],
            )
            return Diagnostic(
                code="RUNTIME_MERGE_FIELD_TYPE_INCOMPATIBLE",
                severity="error",
                location=f"dataset:{first_source}/column:{column_name}/selection:ordinal",
                message=(
                    f"Ordinal MERGE column '{column_name}' has incompatible Arrow "
                    f"types in sources '{first_source}' and '{second_source}': "
                    f"{first_field.type} vs {second_field.type}."
                ),
            )
        return None

    @staticmethod
    def _merge_by_schema_incompatibility_diagnostic(
        *,
        by_key: str,
        observations: Sequence[tuple[str, str, str]],
    ) -> Diagnostic | None:
        """Build the existing BY precondition diagnostic for type conflicts."""

        non_null_observations = [
            observation for observation in observations if observation[2] != "null"
        ]
        observed_types = {observation[2] for observation in non_null_observations}
        if len(observed_types) <= 1:
            return None

        first_source, field_name, first_type = non_null_observations[0]
        second_source, _, second_type = next(
            observation
            for observation in non_null_observations[1:]
            if observation[2] != first_type
        )
        source_aliases = ", ".join(
            dict.fromkeys(observation[0] for observation in observations)
        )
        return Diagnostic(
            code="RUNTIME_BY_PRECONDITION_FAILED",
            severity="error",
            location=f"dataset:{first_source}/column:{field_name}/selection:by",
            message=(
                f"MERGE BY key '{by_key}' has incompatible types in sources "
                f"'{source_aliases}' for field '{field_name}': "
                f"'{first_source}'={first_type} vs '{second_source}'={second_type}."
            ),
        )

    @classmethod
    def _coerce_prepared_merge_input(
        cls,
        *,
        source_name: str,
        merged_rows: Sequence[Mapping[str, Any]],
        source_inputs: Mapping[str, DataSetRef],
        internal_variable_names: set[str] | None = None,
    ) -> DataSetRef:
        source_input = _resolve_source_input_ref(source_inputs, source_name)
        source_location = (
            source_input.location
            if source_input is not None
            else f"dataset://{source_name}"
        )
        return DataSetRef(
            kind="arrow_table",
            location=source_location,
            payload=cls._build_arrow_table_from_rows(merged_rows),
            metadata=source_input.metadata if source_input is not None else {},
            preparation=cls._prepared_provenance(
                source_input, "prepared_merge", internal_variable_names
            ),
        )

    @classmethod
    def _collapse_prepared_set_statement(
        cls,
        *,
        ast_statements: Sequence[ParsedStatement],
        resolved_inputs: Mapping[str, DataSetRef],
    ) -> tuple[ParsedStatement, ...]:
        """Replace a materialized SET stream with its single carrier source.

        The carrier already contains source occurrence order, repeated-source
        rows, automatic IN/INDSNAME/END values, and BY flags.  Leaving the
        original references in the AST would make the runtime read the
        source stream a second time.  A non-collapsed stream (for example a
        non-Arrow input without preparation requirements) keeps its original
        statement and remains owned by the runtime stream contract.
        """

        set_index = next(
            (index for index, statement in enumerate(ast_statements) if statement.kind == "SET"),
            None,
        )
        if set_index is None or any(statement.kind == "MERGE" for statement in ast_statements):
            return tuple(ast_statements)

        set_statement = ast_statements[set_index]
        source_refs = tuple(getattr(set_statement, "dataset_refs", ()) or ())
        if not source_refs:
            return tuple(ast_statements)

        first_source_name = str(source_refs[0].name)
        first_input = _resolve_source_input_ref(resolved_inputs, first_source_name)
        if (
            first_input is None
            or getattr(getattr(first_input, "preparation", None), "kind", "none")
            != "prepared_set"
        ):
            return tuple(ast_statements)

        prepared_input_keys = {_column_key(str(name)) for name in resolved_inputs}
        remaining_unique_sources = {
            _column_key(str(source_ref.name))
            for source_ref in source_refs[1:]
            if _column_key(str(source_ref.name)) != _column_key(first_source_name)
        }
        if remaining_unique_sources.intersection(prepared_input_keys):
            return tuple(ast_statements)

        replacement = replace(
            set_statement,
            text=f"set {first_source_name}",
            dataset_refs=(DatasetReference(name=first_source_name),),
        )
        return tuple(
            replacement if index == set_index else statement
            for index, statement in enumerate(ast_statements)
        )

    @staticmethod
    def _extract_variable_list(
        ast_statements: Sequence[ParsedStatement],
        kind: str,
    ) -> tuple[str, ...]:
        statement = next(
            (item for item in ast_statements if item.kind == kind),
            None,
        )
        if statement is None:
            return ()
        text = str(getattr(statement, "text", ""))
        return tuple(name for name in text[len(kind) :].strip().split() if name)

    @staticmethod
    def _resolve_row_key(row: Mapping[str, Any], name: str) -> str | None:
        if name in row:
            return name
        normalized = _column_key(name)
        for candidate in row.keys():
            if _column_key(candidate) == normalized:
                return candidate
        return None

    @classmethod
    def _resolve_row_value(cls, row: Mapping[str, Any], name: str) -> Any:
        resolved = cls._resolve_row_key(row, name)
        return row.get(resolved) if resolved is not None else None

    @staticmethod
    def _apply_source_option_column_order(
        columns: Sequence[str],
        source_spec: Mapping[str, Any],
    ) -> tuple[str, ...]:
        ordered = list(columns)
        keep_vars = tuple(source_spec.get("keep", ()) or ())
        if keep_vars:
            keep_keys = {_column_key(name) for name in keep_vars}
            ordered = [name for name in ordered if _column_key(name) in keep_keys]
        drop_vars = tuple(source_spec.get("drop", ()) or ())
        if drop_vars:
            drop_keys = {_column_key(name) for name in drop_vars}
            ordered = [name for name in ordered if _column_key(name) not in drop_keys]
        rename_map = dict(source_spec.get("rename", {}) or {})
        if rename_map:
            normalized_rename = {
                _column_key(source): str(target)
                for source, target in rename_map.items()
            }
            ordered = [normalized_rename.get(_column_key(name), name) for name in ordered]
        return tuple(ordered)

    def _detect_set_input_internal_variable_collisions(
        self,
        *,
        set_statement: ParsedStatement | None,
        source_refs: Sequence[Any],
        resolved_inputs: Mapping[str, DataSetRef],
        by_keys: Sequence[str],
    ) -> Diagnostic | None:
        if set_statement is None or not source_refs:
            return None
        in_option_vars = [
            ref.options.in_var
            for ref in source_refs
            if getattr(getattr(ref, "options", None), "in_var", None) is not None
        ]
        statement_options = getattr(set_statement, "statement_options", None)
        indsname_var = getattr(statement_options, "indsname_var", None)
        end_var = getattr(statement_options, "end_var", None)
        internal_variable_names = self._collect_internal_variable_names(
            in_option_vars=in_option_vars,
            indsname_var=indsname_var,
            end_var=end_var,
            by_keys=by_keys,
        )
        if not internal_variable_names:
            return None

        for source_ref in source_refs:
            input_ref = resolved_inputs.get(source_ref.name)
            if input_ref is None:
                continue
            if input_ref.kind != "arrow_table" or not isinstance(input_ref.payload, pa.Table):
                continue
            option_spec = getattr(source_ref, "options", None)
            option_columns = self._apply_source_option_column_order(
                tuple(getattr(input_ref.payload.schema, "names", ()) or ()),
                {
                    "keep": getattr(option_spec, "keep_vars", ()),
                    "drop": getattr(option_spec, "drop_vars", ()),
                    "rename": getattr(option_spec, "rename_map", {}),
                },
            )
            allow_internal_names = getattr(
                getattr(input_ref, "preparation", None), "kind", "none"
            ) == "prepared_merge"
            collision = self._detect_internal_variable_collision_in_schema(
                column_names=option_columns,
                source_name=source_ref.name,
                internal_variable_names=internal_variable_names,
                allow_internal_names=allow_internal_names,
            )
            if collision is not None:
                return collision
        return None

    @staticmethod
    def _detect_internal_variable_collision_in_schema(
        *,
        column_names: Sequence[str],
        source_name: str,
        internal_variable_names: set[str],
        allow_internal_names: bool,
    ) -> Diagnostic | None:
        if allow_internal_names or not internal_variable_names:
            return None
        normalized_internal_names = {_column_key(name) for name in internal_variable_names}
        collided_names = sorted(
            (name for name in column_names if _column_key(name) in normalized_internal_names),
            key=_column_key,
        )
        if not collided_names:
            return None
        return build_internal_var_collision(collided_names[0], source_name)

    @staticmethod
    def _collect_internal_variable_names(
        *,
        in_option_vars: Sequence[str],
        indsname_var: str | None,
        end_var: str | None,
        by_keys: Sequence[str],
    ) -> set[str]:
        names = {name for name in in_option_vars if name}
        if indsname_var:
            names.add(indsname_var)
        if end_var:
            names.add(end_var)
        for key in by_keys:
            names.update({f"FIRST.{key}", f"LAST.{key}", f"first.{key}", f"last.{key}"})
        return names

    @staticmethod
    def _has_dataset_reference_options(option_spec: Any | None) -> bool:
        if option_spec is None:
            return False
        return bool(
            getattr(option_spec, "keep_vars", ())
            or getattr(option_spec, "drop_vars", ())
            or getattr(option_spec, "where_expr", None) is not None
            or getattr(option_spec, "rename_map", {})
            or getattr(option_spec, "firstobs", None) is not None
            or getattr(option_spec, "obs", None) is not None
        )

    @classmethod
    def _dataset_option_arrow_required_diagnostic(cls, source_ref: Any) -> Diagnostic:
        source_name = str(getattr(source_ref, "name", "<unknown>"))
        option_spec = getattr(source_ref, "options", None)
        detail = (
            "WHERE= requires the shared Arrow preparation path"
            if getattr(option_spec, "where_expr", None) is not None
            else "dataset options require the shared Arrow preparation path"
        )
        return Diagnostic(
            code="RUNTIME_DATASET_OPTION_INVALID",
            severity="error",
            message=f"Dataset option {detail} for source '{source_name}'.",
            location=f"dataset:{source_name}",
        )

    def _try_interleave_set_inputs(
        self,
        ast_statements: Sequence[ParsedStatement],
        resolved_inputs: Mapping[str, DataSetRef],
        by_keys: Sequence[str],
    ) -> dict[str, DataSetRef] | None:
        """Collapse simple multi-source SET streams in Arrow space."""

        set_statement = next(
            (statement for statement in ast_statements if statement.kind == "SET"),
            None,
        )
        if set_statement is None:
            return None
        source_refs = tuple(getattr(set_statement, "dataset_refs", ()) or ())
        if len(source_refs) <= 1:
            return None
        if any(
            getattr(getattr(ref, "options", None), "in_var", None) is not None
            for ref in source_refs
        ):
            return None
        if any(
            self._has_dataset_reference_options(getattr(ref, "options", None))
            for ref in source_refs
        ):
            return None

        tables: list[pa.Table] = []
        for ref in source_refs:
            dataset = resolved_inputs.get(ref.name)
            if (
                dataset is None
                or dataset.kind.strip().lower() != "arrow_table"
                or not isinstance(dataset.payload, pa.Table)
            ):
                return None
            tables.append(dataset.payload)
        try:
            combined = pa.concat_tables(tables, promote_options="default")
            sort_keys = [
                (key, "ascending")
                for key in by_keys
                if key in combined.schema.names
            ]
            if sort_keys:
                combined = combined.sort_by(sort_keys)
        except Exception:
            return None

        first_name = source_refs[0].name
        source_names = {ref.name for ref in source_refs}
        result: dict[str, DataSetRef] = {}
        for name, dataset in resolved_inputs.items():
            if name in source_names:
                if name == first_name:
                    result[name] = DataSetRef(
                        kind="arrow_table",
                        location=str(dataset.location or ""),
                        payload=combined,
                        metadata=dataset.metadata,
                        preparation=self._prepared_provenance(
                            dataset, "prepared_set", None
                        ),
                    )
                continue
            result[name] = dataset
        return result

    @staticmethod
    def _arrow_set_schema_tables(
        *,
        source_refs: Sequence[Any],
        resolved_inputs: Mapping[str, DataSetRef],
    ) -> list[tuple[str, Any]] | None:
        if len(source_refs) <= 1:
            return None
        source_tables: list[tuple[str, Any]] = []
        for source_ref in source_refs:
            input_ref = resolved_inputs.get(source_ref.name)
            if input_ref is None:
                return None
            if (
                input_ref.kind.strip().lower() != "arrow_table"
                or not isinstance(input_ref.payload, pa.Table)
            ):
                return None
            source_tables.append((str(source_ref.name), input_ref.payload))
        return source_tables

    @staticmethod
    def _set_schema_incompatibility_diagnostic(
        source_tables: Sequence[tuple[str, Any]],
    ) -> Diagnostic | None:
        fields_by_key: dict[str, list[tuple[str, pa.Field]]] = {}
        for source_name, table in source_tables:
            schema = getattr(table, "schema", None)
            if schema is None:
                continue
            for field in schema:
                field_name = str(getattr(field, "name", ""))
                fields_by_key.setdefault(_column_key(field_name), []).append(
                    (source_name, field)
                )

        for field_observations in fields_by_key.values():
            non_null = [item for item in field_observations if not _arrow_type_is_null(item[1].type)]
            observed_types = {str(field.type) for _, field in non_null}
            if len(observed_types) <= 1:
                if len(non_null) <= 1 or not non_null:
                    continue
                first_source, first_field = non_null[0]
                if not _arrow_source_backed_type_is_nested(first_field.type):
                    continue
                second = next(
                    (
                        item
                        for item in non_null[1:]
                        if not _arrow_fields_are_compatible(first_field, item[1])
                    ),
                    None,
                )
                if second is None:
                    continue
                second_source, second_field = second
                source_aliases = ", ".join(
                    dict.fromkeys(item[0] for item in field_observations)
                )
                return Diagnostic(
                    code="RUNTIME_SET_SCHEMA_INCOMPATIBLE",
                    severity="error",
                    location=f"dataset:{first_source}/column:{first_field.name}",
                    message=(
                        f"SET sources '{source_aliases}' have incompatible Field "
                        f"ownership for field '{first_field.name}': "
                        f"'{first_source}'={first_field} vs "
                        f"'{second_source}'={second_field}."
                    ),
                )
            else:
                first_source, first_field = non_null[0]
                second_source, second_field = next(
                    item
                    for item in non_null[1:]
                    if str(item[1].type) != str(first_field.type)
                )
                source_aliases = ", ".join(
                    dict.fromkeys(item[0] for item in field_observations)
                )
                return Diagnostic(
                    code="RUNTIME_SET_SCHEMA_INCOMPATIBLE",
                    severity="error",
                    location=f"dataset:{first_source}",
                    message=(
                        f"SET sources '{source_aliases}' have incompatible types for field "
                        f"'{first_field.name}': '{first_source}'={first_field.type} vs "
                        f"'{second_source}'={second_field.type}."
                    ),
                )
        return None

    def _prepare_set_rows_with_temporary_variables(
        self,
        *,
        ast_statements: Sequence[ParsedStatement],
        resolved_inputs: Mapping[str, DataSetRef],
        by_keys: Sequence[str],
        has_lag_lead: bool,
        option_applier: Any,
    ) -> tuple[dict[str, DataSetRef] | None, Diagnostic | None]:
        set_statement = next(
            (statement for statement in ast_statements if statement.kind == "SET"),
            None,
        )
        if set_statement is None or any(
            statement.kind == "MERGE" for statement in ast_statements
        ):
            return None, None

        source_refs = list(getattr(set_statement, "dataset_refs", ()) or ())
        if not source_refs:
            return None, None
        has_repeated_source_names = len(
            {_column_key(ref.name) for ref in source_refs}
        ) < len(source_refs)
        in_option_vars = [
            ref.options.in_var
            for ref in source_refs
            if getattr(getattr(ref, "options", None), "in_var", None) is not None
        ]
        statement_options = getattr(set_statement, "statement_options", None)
        indsname_var = getattr(statement_options, "indsname_var", None)
        end_var = getattr(statement_options, "end_var", None)
        has_position_window_options = any(
            getattr(getattr(ref, "options", None), "firstobs", None) is not None
            or getattr(getattr(ref, "options", None), "obs", None) is not None
            for ref in source_refs
        )
        needs_prepare = bool(
            in_option_vars
            or indsname_var
            or end_var
            or by_keys
            or has_lag_lead
            or has_repeated_source_names
            or has_position_window_options
            or any(
                self._has_dataset_reference_options(getattr(ref, "options", None))
                for ref in source_refs
            )
        )
        if not needs_prepare:
            return None, None

        # A previous simple interleave may already have collapsed all source
        # streams into the first prepared carrier.  It is complete for this
        # unit; do not try to reload omitted source names and leak a partial
        # diagnostic.
        first_input = resolved_inputs.get(source_refs[0].name)
        if (
            first_input is not None
            and getattr(getattr(first_input, "preparation", None), "kind", "none")
            == "prepared_set"
            and all(ref.name not in resolved_inputs for ref in source_refs[1:])
        ):
            return None, None

        internal_names = self._collect_internal_variable_names(
            in_option_vars=in_option_vars,
            indsname_var=indsname_var,
            end_var=end_var,
            by_keys=by_keys,
        )
        prepared_arrow_inputs, prepared_arrow_error = self._prepare_set_arrow_inputs_if_supported(
            source_refs=source_refs,
            by_keys=by_keys,
            in_option_vars=in_option_vars,
            indsname_var=indsname_var,
            end_var=end_var,
            resolved_inputs=resolved_inputs,
            internal_variable_names=internal_names,
            option_applier=option_applier,
        )
        if prepared_arrow_error is not None:
            return None, prepared_arrow_error
        if prepared_arrow_inputs is not None:
            return prepared_arrow_inputs, None

        option_source = next(
            (
                ref
                for ref in source_refs
                if self._has_dataset_reference_options(getattr(ref, "options", None))
            ),
            None,
        )
        if option_source is not None:
            return None, self._dataset_option_arrow_required_diagnostic(option_source)

        prepared_rows, prepare_error = prepare_set_rows(
            source_refs=source_refs,
            by_keys=by_keys,
            in_option_vars=in_option_vars,
            indsname_var=indsname_var,
            end_var=end_var,
            resolved_inputs=resolved_inputs,
            internal_variable_names=internal_names,
            load_input_rows=self._io_service.load_input_rows,
            resolve_row_key=self._resolve_row_key,
            resolve_row_value=self._resolve_row_value,
        )
        if prepare_error is not None:
            return None, prepare_error
        first_source = source_refs[0].name
        first_input = resolved_inputs.get(first_source)
        if first_input is None:
            return None, None
        prepared_ref = self._coerce_rewrite_input(
            source_name=first_source,
            source_input=first_input,
            rewritten_rows=prepared_rows,
            internal_variable_names=internal_names,
        )
        source_names = {ref.name for ref in source_refs}
        updated_inputs = {
            name: dataset
            for name, dataset in resolved_inputs.items()
            if name not in source_names
        }
        updated_inputs[first_source] = prepared_ref
        return updated_inputs, None

    def _prepare_set_arrow_inputs_if_supported(
        self,
        *,
        source_refs: Sequence[Any],
        by_keys: Sequence[str],
        in_option_vars: Sequence[str],
        indsname_var: str | None,
        end_var: str | None,
        resolved_inputs: Mapping[str, DataSetRef],
        internal_variable_names: set[str],
        option_applier: Any,
    ) -> tuple[dict[str, DataSetRef] | None, Diagnostic | None]:
        prepared_tables: list[pa.Table] = []
        source_names = {ref.name for ref in source_refs}
        first_source_name = source_refs[0].name
        first_source_input = resolved_inputs.get(first_source_name)
        if first_source_input is None:
            return None, None

        for source_order, source_ref in enumerate(source_refs):
            input_ref = resolved_inputs.get(source_ref.name)
            if input_ref is None:
                return None, None
            if input_ref.kind != "arrow_table" or not isinstance(input_ref.payload, pa.Table):
                return None, None
            table, option_error = option_applier(
                input_ref.payload,
                source_name=source_ref.name,
                option_spec=source_ref.options,
            )
            if option_error is not None:
                return None, option_error
            if table is None:
                return None, None

            allow_internal_names = getattr(
                getattr(input_ref, "preparation", None), "kind", "none"
            ) == "prepared_merge"
            collision = self._detect_internal_variable_collision_in_schema(
                column_names=tuple(getattr(table.schema, "names", ()) or ()),
                source_name=source_ref.name,
                internal_variable_names=internal_variable_names,
                allow_internal_names=allow_internal_names,
            )
            if collision is not None:
                return None, collision

            row_count = int(getattr(table, "num_rows", 0) or 0)
            table = table.append_column(
                "__prepare_source_order__",
                pa.array([source_order] * row_count, type=pa.int64()),
            )
            table = table.append_column(
                "__prepare_row_order__",
                pa.array(list(range(row_count)), type=pa.int64()),
            )
            for in_var in in_option_vars:
                values = [
                    1 if getattr(source_ref.options, "in_var", None) == in_var else 0
                ] * row_count
                table = self._set_arrow_column(table, in_var, pa.array(values, type=pa.int8()))
            if indsname_var:
                table = self._set_arrow_column(
                    table,
                    indsname_var,
                    pa.array([source_ref.name] * row_count),
                )
            prepared_tables.append(table)

        prepared_tables = list(_promote_mixed_string_tables(prepared_tables))
        schema_diagnostic = self._set_schema_incompatibility_diagnostic(
            [
                (str(source_ref.name), table)
                for source_ref, table in zip(source_refs, prepared_tables)
            ]
        )
        if schema_diagnostic is not None:
            return None, schema_diagnostic
        try:
            combined = pa.concat_tables(prepared_tables, promote_options="default")
            if by_keys:
                sort_keys: list[tuple[str, str]] = []
                for by_key in by_keys:
                    resolved_name = self._resolve_arrow_table_column_name(combined, by_key)
                    if resolved_name is None:
                        return None, build_by_rows_precondition_failed(by_key)
                    sort_keys.append((resolved_name, "ascending"))
                sort_keys.extend(
                    [
                        ("__prepare_source_order__", "ascending"),
                        ("__prepare_row_order__", "ascending"),
                    ]
                )
                combined = combined.sort_by(sort_keys)
        except Exception:
            return None, None

        if end_var:
            row_count = int(getattr(combined, "num_rows", 0) or 0)
            end_values = [0] * row_count
            if end_values:
                end_values[-1] = 1
            combined = self._set_arrow_column(
                combined,
                end_var,
                pa.array(end_values, type=pa.int8()),
            )
        combined, by_error = self._annotate_arrow_by_group_flags(combined, by_keys=by_keys)
        if by_error is not None:
            return None, by_error
        visible_names = [
            name
            for name in getattr(combined, "column_names", ())
            if name not in {"__prepare_source_order__", "__prepare_row_order__"}
        ]
        combined = combined.select(visible_names)
        updated_inputs = {
            name: dataset
            for name, dataset in resolved_inputs.items()
            if name not in source_names
        }
        updated_inputs[first_source_name] = self._coerce_rewrite_arrow_input(
            source_name=first_source_name,
            source_input=first_source_input,
            rewritten_table=combined,
            internal_variable_names=internal_variable_names,
        )
        return updated_inputs, None

    @staticmethod
    def _set_arrow_column(table: pa.Table, name: str, values: Any) -> pa.Table:
        field_index = table.schema.get_field_index(name)
        if field_index >= 0:
            return table.set_column(field_index, name, values)
        return table.append_column(name, values)

    @classmethod
    def _resolve_arrow_table_column_name(cls, table: Any, name: str) -> str | None:
        schema_names = tuple(getattr(getattr(table, "schema", None), "names", ()) or ())
        if name in schema_names:
            return name
        normalized_name = _column_key(name)
        for candidate in schema_names:
            if _column_key(candidate) == normalized_name:
                return candidate
        return None

    @classmethod
    def _annotate_arrow_by_group_flags(
        cls,
        table: pa.Table,
        *,
        by_keys: Sequence[str],
    ) -> tuple[pa.Table, Diagnostic | None]:
        if not by_keys or int(getattr(table, "num_rows", 0) or 0) == 0:
            return table, None
        value_columns: list[list[Any]] = []
        for by_key in by_keys:
            resolved_name = cls._resolve_arrow_table_column_name(table, by_key)
            if resolved_name is None:
                return table, build_by_rows_precondition_failed(by_key)
            value_columns.append(table.column(resolved_name).combine_chunks().to_pylist())
        for by_key, values in zip(by_keys, value_columns):
            first_values: list[int] = []
            last_values: list[int] = []
            for index, current_value in enumerate(values):
                previous_value = values[index - 1] if index > 0 else object()
                next_value = values[index + 1] if index < len(values) - 1 else object()
                first_values.append(1 if current_value != previous_value else 0)
                last_values.append(1 if current_value != next_value else 0)
            table = cls._set_arrow_column(
                table,
                f"FIRST.{by_key}",
                pa.array(first_values, type=pa.int8()),
            )
            table = cls._set_arrow_column(
                table,
                f"LAST.{by_key}",
                pa.array(last_values, type=pa.int8()),
            )
            table = cls._set_arrow_column(
                table,
                f"first.{by_key}",
                pa.array(first_values, type=pa.int8()),
            )
            table = cls._set_arrow_column(
                table,
                f"last.{by_key}",
                pa.array(last_values, type=pa.int8()),
            )
        return table, None

    @staticmethod
    def _build_arrow_table_from_rows(rows: Sequence[Mapping[str, Any]]) -> pa.Table:
        normalized_rows = [dict(row) for row in rows]
        if not normalized_rows:
            return pa.Table.from_pylist([])
        all_keys: list[str] = []
        seen_keys: set[str] = set()
        for row in normalized_rows:
            for key in row:
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                all_keys.append(key)
        stabilized_rows = [{key: row.get(key) for key in all_keys} for row in normalized_rows]
        return pa.Table.from_pylist(stabilized_rows)

    @staticmethod
    def _prepared_provenance(
        source_input: DataSetRef | None,
        kind: str,
        internal_variable_names: set[str] | None,
    ) -> PreparationProvenance:
        """Build typed provenance for one newly generated prepared carrier.

        The carrier kind is the producing stage's kind unless the source was
        itself a prepared carrier (a later generation keeps its kind).  Internal
        names accumulate across generations so every internal column the
        carrier physically contains stays registered case-preserving.
        """

        source_provenance = getattr(source_input, "preparation", None)
        source_kind = getattr(source_provenance, "kind", "none")
        source_names = tuple(
            getattr(source_provenance, "internal_variable_names", ()) or ()
        )
        merged_names = set(source_names) | {
            name for name in (internal_variable_names or set()) if name
        }
        return PreparationProvenance(
            kind=str(source_kind) if source_kind != "none" else kind,
            internal_variable_names=tuple(sorted(merged_names)),
        )

    @classmethod
    def _coerce_rewrite_input(
        cls,
        *,
        source_name: str,
        source_input: DataSetRef,
        rewritten_rows: Sequence[Mapping[str, Any]],
        internal_variable_names: set[str] | None = None,
    ) -> DataSetRef:
        base_location = source_input.location or f"dataset://{source_name}"
        return DataSetRef(
            kind="arrow_table",
            # The user-supplied location is preserved byte-for-byte: prepared
            # kind and internal names travel in the typed contract only.
            location=base_location,
            payload=cls._build_arrow_table_from_rows(rewritten_rows),
            metadata=source_input.metadata,
            preparation=cls._prepared_provenance(
                source_input, "prepared_set", internal_variable_names
            ),
        )

    @staticmethod
    def _coerce_rewrite_arrow_input(
        *,
        source_name: str,
        source_input: DataSetRef,
        rewritten_table: pa.Table,
        internal_variable_names: set[str] | None = None,
    ) -> DataSetRef:
        base_location = source_input.location or f"dataset://{source_name}"
        return DataSetRef(
            kind="arrow_table",
            location=base_location,
            payload=rewritten_table,
            metadata=source_input.metadata,
            preparation=ExecutionInputPreparationService._prepared_provenance(
                source_input, "prepared_set", internal_variable_names
            ),
        )

    def _sort_inputs_by_by_keys(
        self,
        resolved_inputs: Mapping[str, DataSetRef],
        by_keys: Sequence[str],
    ) -> tuple[dict[str, DataSetRef] | None, Diagnostic | None]:
        sorted_inputs: dict[str, DataSetRef] = {}
        for name, ref in resolved_inputs.items():
            if ref.kind == "arrow_table" and isinstance(ref.payload, pa.Table):
                active_keys = []
                for by_key in by_keys:
                    resolved_key = next(
                        (
                            column_name
                            for column_name in ref.payload.column_names
                            if _column_key(column_name) == _column_key(by_key)
                        ),
                        None,
                    )
                    if resolved_key is None:
                        continue
                    data_type = ref.payload.schema.field(resolved_key).type
                    if _arrow_merge_type_is_nested(data_type):
                        return (
                            None,
                            self._nested_by_key_diagnostic(
                                dataset_name=name,
                                by_key=by_key,
                                resolved_column_name=resolved_key,
                                data_type=data_type,
                            ),
                        )
                    active_keys.append((resolved_key, "ascending"))
                if not active_keys:
                    sorted_inputs[name] = ref
                    continue
                try:
                    sorted_payload = ref.payload.sort_by(active_keys)
                except Exception as error:
                    first_key = active_keys[0][0] if active_keys else ""
                    return (
                        None,
                        self._nested_by_key_diagnostic(
                            dataset_name=name,
                            by_key=by_keys[0] if by_keys else first_key,
                            resolved_column_name=first_key,
                            data_type=(
                                ref.payload.schema.field(first_key).type
                                if first_key in ref.payload.column_names
                                else None
                            ),
                            underlying_error=error,
                        ),
                    )
                sorted_inputs[name] = DataSetRef(
                    kind="arrow_table",
                    location=ref.location,
                    payload=sorted_payload,
                    metadata=ref.metadata,
                    preparation=ref.preparation,
                )
                continue

            rows, error = self._io_service.load_input_rows(ref)
            if error is not None or not rows:
                sorted_inputs[name] = ref
                continue
            active_keys = [key for key in by_keys if self._resolve_row_key(rows[0], key) is not None]
            if not active_keys:
                sorted_inputs[name] = ref
                continue
            rows.sort(
                key=lambda row: build_row_sort_key(
                    row,
                    by_keys=tuple(active_keys),
                    resolve_row_value=self._resolve_row_value,
                )
            )
            sorted_inputs[name] = DataSetRef(
                kind=ref.kind,
                location=ref.location,
                payload=rows,
                metadata=ref.metadata,
            )
        return sorted_inputs, None

    @staticmethod
    def _nested_by_key_diagnostic(
        *,
        dataset_name: str,
        by_key: str,
        resolved_column_name: str,
        data_type: Any = None,
        underlying_error: Exception | None = None,
    ) -> Diagnostic:
        type_text = f" {data_type}" if data_type is not None else ""
        unsupported_text = (
            f"unsupported nested Arrow type{type_text}"
            if data_type is not None and _arrow_merge_type_is_nested(data_type)
            else f"Arrow type{type_text}"
        )
        underlying_text = (
            f" ({type(underlying_error).__name__}: {underlying_error})"
            if underlying_error is not None
            else ""
        )
        return Diagnostic(
            code="RUNTIME_BY_PRECONDITION_FAILED",
            severity="error",
            location=f"dataset:{dataset_name}/column:{resolved_column_name}",
            message=(
                f"SET BY key '{by_key}' in source '{dataset_name}' resolves to "
                f"column '{resolved_column_name}' with {unsupported_text}; "
                f"BY sorting does not support nested columns or the resolved "
                f"column type.{underlying_text}"
            ),
        )

    @staticmethod
    def _failed_result(
        diagnostics: Sequence[Diagnostic],
    ) -> InputResolutionResult:
        return InputResolutionResult(
            resolved_inputs=NormalizedExecutionInputs(),
            source_metadata_lineage=SourceMetadataLineage(),
            diagnostics=tuple(diagnostics),
        )


def capture_source_metadata_lineage(
    source_refs: Sequence[Any],
    resolved_inputs: Mapping[str, DataSetRef],
) -> tuple[SourceMetadataLineage, Diagnostic | None]:
    """Capture ordered, option-shaped source schemas before source collapse.

    Dataset references are resolved by the DSL occurrence rather than by the
    input mapping iteration order.  This preserves repeated source names and
    keeps the snapshot limited to schema/metadata; table values are never
    materialized into Python rows here.
    """

    snapshots: list[SourceMetadataSnapshot] = []
    for source_ordinal, source_ref in enumerate(source_refs):
        source_name = str(getattr(source_ref, "name", ""))
        input_ref = _resolve_source_input_ref(resolved_inputs, source_name)
        if input_ref is None:
            continue

        table = input_ref.payload
        if not isinstance(table, pa.Table):
            # Direct preparation helpers still accept memory refs for the
            # temporary row-based path.  The pipeline normalizes those refs
            # before this component, so there is no metadata snapshot to
            # capture for a non-Arrow direct call.
            continue

        shaped_schema, option_error = _project_arrow_schema_for_dataset_options(
            table.schema,
            source_name=source_name,
            option_spec=getattr(source_ref, "options", None),
        )
        if option_error is not None:
            return SourceMetadataLineage(tuple(snapshots)), option_error
        if shaped_schema is None:
            continue

        snapshots.append(
            SourceMetadataSnapshot(
                source_ordinal=source_ordinal,
                source_name=source_name,
                schema=shaped_schema,
                dataset_ref_metadata=input_ref.metadata,
            )
        )

    return SourceMetadataLineage(tuple(snapshots)), None


def _project_arrow_schema_for_dataset_options(
    schema: pa.Schema,
    *,
    source_name: str,
    option_spec: Any,
) -> tuple[pa.Schema | None, Diagnostic | None]:
    """Project lineage fields without consuming the input option twice.

    Lineage only needs the option-shaped fields.  Filtering rows or evaluating
    an Arrow mask here would repeat execution preparation work, so this helper
    validates and applies schema-only stages while the actual ``WHERE``/slice
    and array transformation remain owned by the single preparation call.
    """

    normalized = _normalize_dataset_option_spec(option_spec=option_spec)
    fields = list(schema)

    if normalized["keep_vars"]:
        keep_set = {_column_key(name) for name in normalized["keep_vars"]}
        fields = [field for field in fields if _column_key(field.name) in keep_set]

    if normalized["drop_vars"]:
        drop_set = {_column_key(name) for name in normalized["drop_vars"]}
        fields = [field for field in fields if _column_key(field.name) not in drop_set]

    if normalized["where_expr"] is not None:
        where_expression, where_error = _coerce_where_expression(
            normalized["where_expr"],
            source_name=source_name,
        )
        if where_error is not None:
            return None, where_error
        assert where_expression is not None
        where_error = _validate_where_expression_schema(
            where_expression,
            fields=fields,
            source_name=source_name,
        )
        if where_error is not None:
            return None, where_error

    rename_map = normalized["rename_map"]
    if rename_map:
        if len({_column_key(value) for value in rename_map.values()}) != len(rename_map):
            return None, Diagnostic(
                code="RUNTIME_DATASET_OPTION_INVALID",
                severity="error",
                message=(
                    f"Dataset option RENAME= has duplicate target names for source "
                    f"'{source_name}'."
                ),
            )
        resolved_rename_map: dict[str, str] = {}
        for old_name, new_name in rename_map.items():
            resolved_old_name = next(
                (
                    field.name
                    for field in fields
                    if _column_key(field.name) == _column_key(old_name)
                ),
                None,
            )
            if resolved_old_name is None:
                return None, Diagnostic(
                    code="RUNTIME_DATASET_OPTION_INVALID",
                    severity="error",
                    message=(
                        f"Dataset option RENAME= references unknown variable '{old_name}' "
                        f"for source '{source_name}'."
                    ),
                )
            resolved_rename_map[resolved_old_name] = str(new_name)
        fields = [
            pa.field(
                resolved_rename_map.get(field.name, field.name),
                field.type,
                nullable=field.nullable,
                metadata=dict(field.metadata) if field.metadata is not None else None,
            )
            for field in fields
        ]

    firstobs = normalized["firstobs"]
    obs = normalized["obs"]
    if firstobs is not None and firstobs <= 0:
        return None, Diagnostic(
            code="RUNTIME_DATASET_OPTION_INVALID",
            severity="error",
            message=(
                f"Dataset option FIRSTOBS= must be positive for source '{source_name}'."
            ),
        )
    if obs is not None and obs < 0:
        return None, Diagnostic(
            code="RUNTIME_DATASET_OPTION_INVALID",
            severity="error",
            message=(
                f"Dataset option OBS= must be non-negative for source '{source_name}'."
            ),
        )

    return pa.schema(
        fields,
        metadata=dict(schema.metadata) if schema.metadata is not None else None,
    ), None


def _validate_where_expression_schema(
    expression: WhereExpression,
    *,
    fields: Sequence[pa.Field],
    source_name: str,
) -> Diagnostic | None:
    """Validate WHERE shape and referenced columns without evaluating rows."""

    if isinstance(expression, (WhereComparison, WhereNullPredicate)):
        matches = [
            field.name
            for field in fields
            if _column_key(field.name) == _column_key(expression.column)
        ]
        if not matches:
            return _arrow_option_diagnostic(
                source_name=source_name,
                reason=(
                    f"WHERE= column '{expression.column}' was not found "
                    "(case-insensitive resolution)"
                ),
            )
        if len(matches) > 1:
            return _arrow_option_diagnostic(
                source_name=source_name,
                reason=(
                    f"WHERE= column resolution is ambiguous ({', '.join(matches)})"
                ),
            )
        return None

    if isinstance(expression, WhereNotExpression):
        return _validate_where_expression_schema(
            expression.operand,
            fields=fields,
            source_name=source_name,
        )

    if isinstance(expression, WhereLogicalExpression):
        if expression.operator not in {"and", "or"}:
            return _arrow_option_diagnostic(
                source_name=source_name,
                reason=f"WHERE= unsupported logical operator '{expression.operator}'",
            )
        if len(expression.operands) < 2:
            return _arrow_option_diagnostic(
                source_name=source_name,
                reason="WHERE= logical expression requires at least two operands",
            )
        for operand in expression.operands:
            diagnostic = _validate_where_expression_schema(
                operand,
                fields=fields,
                source_name=source_name,
            )
            if diagnostic is not None:
                return diagnostic
        return None

    return _arrow_option_diagnostic(
        source_name=source_name,
        reason=f"WHERE= unsupported expression kind {type(expression).__name__}",
    )


def _resolve_source_input_ref(
    resolved_inputs: Mapping[str, DataSetRef],
    source_name: str,
) -> DataSetRef | None:
    direct = resolved_inputs.get(source_name)
    if direct is not None:
        return direct
    source_key = _column_key(source_name)
    for candidate_name, candidate in resolved_inputs.items():
        if _column_key(str(candidate_name)) == source_key:
            return candidate
    return None


def _copy_arrow_schema(schema: pa.Schema) -> pa.Schema:
    """Copy schema/field metadata without retaining table or row buffers."""

    fields = [
        pa.field(
            field.name,
            field.type,
            nullable=field.nullable,
            metadata=dict(field.metadata) if field.metadata is not None else None,
        )
        for field in schema
    ]
    return pa.schema(
        fields,
        metadata=dict(schema.metadata) if schema.metadata is not None else None,
    )


def _resolve_dataset_option_row_key(row: Mapping[str, Any], name: str) -> str | None:
    if name in row:
        return name

    normalized = _column_key(name)
    for candidate in row.keys():
        if _column_key(candidate) == normalized:
            return candidate
    return None


def _normalize_dataset_option_spec(
    *,
    option_spec: Any,
) -> dict[str, Any]:
    return {
        "keep_vars": tuple(getattr(option_spec, "keep_vars", ()) or ()),
        "drop_vars": tuple(getattr(option_spec, "drop_vars", ()) or ()),
        "rename_map": dict(getattr(option_spec, "rename_map", {}) or {}),
        "where_expr": getattr(option_spec, "where_expr", None),
        "firstobs": getattr(option_spec, "firstobs", None),
        "obs": getattr(option_spec, "obs", None),
    }


def apply_dataset_reference_options_to_rows(
    rows: list[dict[str, Any]],
    *,
    source_name: str,
    option_spec: Any,
) -> tuple[list[dict[str, Any]], Diagnostic | None]:
    normalized = _normalize_dataset_option_spec(
        option_spec=option_spec,
    )
    if (
        not normalized["keep_vars"]
        and not normalized["drop_vars"]
        and not normalized["rename_map"]
        and not normalized["where_expr"]
        and normalized["firstobs"] is None
        and normalized["obs"] is None
    ):
        return list(rows), None

    # WHERE is an Arrow-only shared-input operation.  The row preparation
    # helper remains temporarily available for SET/MERGE orchestration, but it
    # must never evaluate or otherwise interpret a typed WHERE expression.
    if normalized["where_expr"] is not None:
        return [], Diagnostic(
            code="RUNTIME_DATASET_OPTION_INVALID",
            severity="error",
            message=(
                f"Dataset option WHERE= for source '{source_name}' requires the "
                "shared Arrow input preparation path."
            ),
        )

    processed: list[dict[str, Any]] = []

    if normalized["rename_map"] and len({_column_key(value) for value in normalized["rename_map"].values()}) != len(normalized["rename_map"]):
        return [], Diagnostic(
            code="RUNTIME_DATASET_OPTION_INVALID",
            severity="error",
            message=f"Dataset option RENAME= has duplicate target names for source '{source_name}'.",
        )

    for row in rows:
        working = dict(row)

        if normalized["keep_vars"]:
            keep_set = {_column_key(name) for name in normalized["keep_vars"]}
            working = {
                name: value
                for name, value in working.items()
                if _column_key(name) in keep_set
            }

        if normalized["drop_vars"]:
            drop_set = {_column_key(name) for name in normalized["drop_vars"]}
            working = {
                name: value
                for name, value in working.items()
                if _column_key(name) not in drop_set
            }

        if normalized["rename_map"]:
            resolved_rename_map: dict[str, str] = {}
            for old_name, new_name in normalized["rename_map"].items():
                resolved_old_name = _resolve_dataset_option_row_key(working, old_name)
                if resolved_old_name is None:
                    return [], Diagnostic(
                        code="RUNTIME_DATASET_OPTION_INVALID",
                        severity="error",
                        message=(
                            f"Dataset option RENAME= references unknown variable '{old_name}' "
                            f"for source '{source_name}'."
                        ),
                    )
                resolved_rename_map[resolved_old_name] = new_name
            renamed_row: dict[str, Any] = {}
            for key, value in working.items():
                renamed_row[resolved_rename_map.get(key, key)] = value
            working = renamed_row

        processed.append(working)

    firstobs = normalized["firstobs"]
    obs = normalized["obs"]
    if firstobs is not None and firstobs <= 0:
        return [], Diagnostic(
            code="RUNTIME_DATASET_OPTION_INVALID",
            severity="error",
            message=f"Dataset option FIRSTOBS= must be positive for source '{source_name}'.",
        )
    if obs is not None and obs < 0:
        return [], Diagnostic(
            code="RUNTIME_DATASET_OPTION_INVALID",
            severity="error",
            message=f"Dataset option OBS= must be non-negative for source '{source_name}'.",
        )
    start_index = max((firstobs or 1) - 1, 0)
    processed = processed[start_index:]
    if obs is not None:
        processed = processed[:obs]

    return processed, None


def _resolve_arrow_table_column_name(table: Any, name: str) -> str | None:
    schema_names = tuple(getattr(getattr(table, "schema", None), "names", ()) or ())
    if name in schema_names:
        return name

    normalized = _column_key(name)
    for candidate in schema_names:
        if _column_key(candidate) == normalized:
            return candidate
    return None


def try_apply_dataset_reference_options_to_arrow_table(
    table: Any,
    *,
    source_name: str,
    option_spec: Any,
) -> tuple[Any | None, Diagnostic | None]:
    normalized = _normalize_dataset_option_spec(
        option_spec=option_spec,
    )
    if not isinstance(table, pa.Table):
        return None, _arrow_option_diagnostic(
            source_name=source_name,
            reason=f"input is not an Arrow table ({type(table).__name__})",
        )
    if (
        not normalized["keep_vars"]
        and not normalized["drop_vars"]
        and not normalized["rename_map"]
        and not normalized["where_expr"]
        and normalized["firstobs"] is None
        and normalized["obs"] is None
    ):
        return table, None

    working = table

    if normalized["keep_vars"]:
        keep_set = {_column_key(name) for name in normalized["keep_vars"]}
        selected_names = [
            column_name
            for column_name in getattr(working, "column_names", ())
            if _column_key(column_name) in keep_set
        ]
        working = working.select(selected_names)

    if normalized["drop_vars"]:
        drop_set = {_column_key(name) for name in normalized["drop_vars"]}
        selected_names = [
            column_name
            for column_name in getattr(working, "column_names", ())
            if _column_key(column_name) not in drop_set
        ]
        working = working.select(selected_names)

    if normalized["where_expr"] is not None:
        where_expression, parse_diagnostic = _coerce_where_expression(
            normalized["where_expr"],
            source_name=source_name,
        )
        if parse_diagnostic is not None:
            return None, parse_diagnostic
        assert where_expression is not None  # narrowed by the diagnostic branch
        evaluation: DatasetWhereEvaluationResult = evaluate_dataset_where(
            where_expression,
            working,
            source_name=source_name,
        )
        if evaluation.diagnostics:
            return None, evaluation.diagnostics[0]
        if evaluation.mask is None:
            return None, _arrow_option_diagnostic(
                source_name=source_name,
                reason="WHERE= evaluator returned no mask",
            )
        try:
            working = working.filter(evaluation.mask)
        except Exception as error:
            return None, _arrow_option_diagnostic(
                source_name=source_name,
                reason=f"Arrow table filter failed: {type(error).__name__}: {error}",
            )

    if normalized["rename_map"]:
        if len({_column_key(value) for value in normalized["rename_map"].values()}) != len(normalized["rename_map"]):
            return None, Diagnostic(
                code="RUNTIME_DATASET_OPTION_INVALID",
                severity="error",
                message=f"Dataset option RENAME= has duplicate target names for source '{source_name}'.",
            )
        resolved_rename_map: dict[str, str] = {}
        for old_name, new_name in normalized["rename_map"].items():
            resolved_old_name = _resolve_arrow_table_column_name(working, old_name)
            if resolved_old_name is None:
                return None, Diagnostic(
                    code="RUNTIME_DATASET_OPTION_INVALID",
                    severity="error",
                    message=(
                        f"Dataset option RENAME= references unknown variable '{old_name}' "
                        f"for source '{source_name}'."
                    ),
                )
            resolved_rename_map[resolved_old_name] = new_name
        renamed_columns = [
            resolved_rename_map.get(column_name, column_name)
            for column_name in getattr(working, "column_names", ())
        ]
        original_metadata = working.schema.metadata
        working = working.rename_columns(renamed_columns)
        if original_metadata is not None:
            working = working.replace_schema_metadata(original_metadata)

    firstobs = normalized["firstobs"]
    obs = normalized["obs"]
    if firstobs is not None and firstobs <= 0:
        return None, Diagnostic(
            code="RUNTIME_DATASET_OPTION_INVALID",
            severity="error",
            message=f"Dataset option FIRSTOBS= must be positive for source '{source_name}'.",
        )
    if obs is not None and obs < 0:
        return None, Diagnostic(
            code="RUNTIME_DATASET_OPTION_INVALID",
            severity="error",
            message=f"Dataset option OBS= must be non-negative for source '{source_name}'.",
        )

    start_index = max((firstobs or 1) - 1, 0)
    length = None if obs is None else int(obs)
    working = working.slice(start_index, length)
    return working, None


def _coerce_where_expression(
    value: object,
    *,
    source_name: str,
) -> tuple[WhereExpression | None, Diagnostic | None]:
    if isinstance(value, (WhereComparison, WhereNullPredicate, WhereLogicalExpression, WhereNotExpression)):
        return value, None
    return None, _arrow_option_diagnostic(
        source_name=source_name,
        reason=f"WHERE= value is not a typed expression ({type(value).__name__})",
    )


def _arrow_option_diagnostic(*, source_name: str, reason: str) -> Diagnostic:
    return Diagnostic(
        code="RUNTIME_DATASET_OPTION_INVALID",
        severity="error",
        message=f"Dataset option WHERE= input preparation failed for source '{source_name}': {reason}",
    )
