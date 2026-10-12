"""Target-specific native output planning and output finalization.

The output service is the only producer of the Rust output handoff plan.  It
receives normalized execution inputs and an immutable rewrite decision, then
builds an independent projection/type/rename contract for every output
target, and finalizes the Rust-produced outputs into the public and lazy
presentations consumed by the execution boundary.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ..arrow_bridge import (
    _project_output_column_names,
    project_lineage_for_output,
    restore_arrow_schema_from_sources,
)
from ..dataset_option_expressions import where_expression_to_dict
from ..io import ExecutorIOService, _apply_arrow_metadata
from ..models import (
    DataSetRef,
    Diagnostic,
    SourceMetadataLineage,
)
from ..naming import _column_key, _dataset_key
from ..parser import ParsedStatement
from ..syntax.models import (
    ArrayElementTargetSyntax,
    ArrayStatementSyntax,
    AssignStatementSyntax,
    BlockSyntax,
    DatasetOptionSyntax,
    DoStatementSyntax,
    ElseStatementSyntax,
    IfStatementSyntax,
    LabelStatementSyntax,
    NameListStatementSyntax,
    NameTargetSyntax,
    RenameStatementSyntax,
    SourceStatementSyntax,
    StatementSyntax,
    SumStatementSyntax,
)
from .contracts import (
    OutputHandoffResult,
    TypedOutputFinalizeRequest,
    TypedOutputPlanRequest,
    TypedOutputShapingRequest,
    TypedOutputShapingResult,
    TypedValidationRequest,
)
from .rewrites import TypedHelperPlan
from .typed_output_planning import TypedOutputTargetPlan, plan_typed_output_targets
from .validation import (
    ExecutionValidationService,
    build_typed_validation_diagnostic,
    has_output_projection_collision,
)


@dataclass(frozen=True)
class OutputHandoffPlan:
    """Rust output contract, partitioned by normalized target name.

    The fields intentionally retain the existing shape consumed by
    ``ExecutionPlanService`` and the native adapter.
    """

    targets: tuple[str, ...]
    projected_columns_by_target: dict[str, tuple[str, ...]]
    rename_map_by_target: dict[str, dict[str, str]]
    type_expectations_by_target: dict[str, dict[str, str]]
    fallback_reason: str | None


@dataclass(frozen=True)
class OutputProjectionStage:
    keep: tuple[str, ...]
    drop: tuple[str, ...]
    rename: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "keep", tuple(str(name) for name in self.keep))
        object.__setattr__(self, "drop", tuple(str(name) for name in self.drop))
        object.__setattr__(
            self,
            "rename",
            MappingProxyType(
                {str(source): str(target) for source, target in self.rename.items()}
            ),
        )


@dataclass(frozen=True)
class OutputLineageProjectionPlan:
    stages_by_target: Mapping[str, tuple[OutputProjectionStage, ...]]

    def __post_init__(self) -> None:
        normalized = {
            str(target): tuple(stages)
            for target, stages in self.stages_by_target.items()
        }
        object.__setattr__(self, "stages_by_target", MappingProxyType(normalized))


class _LazyArrowOutputs(Mapping[str, Any]):
    """Lazy Arrow conversion owned by the output handoff service."""

    def __init__(self, outputs: Mapping[str, DataSetRef], converter: Any) -> None:
        self._outputs = outputs
        self._converter = converter
        self._cache: dict[str, Any] = {}

    def __getitem__(self, key: str) -> Any:
        if key in self._cache:
            return self._cache[key]
        dataset_ref = self._outputs.get(key)
        if dataset_ref is None:
            raise KeyError(key)
        arrow_table, error = self._converter(key, dataset_ref)
        if error is not None or arrow_table is None:
            raise KeyError(key)
        self._cache[key] = arrow_table
        return arrow_table

    def __iter__(self):
        return iter(self._outputs.keys())

    def __len__(self) -> int:
        return len(self._outputs)


class ExecutionOutputHandoffService:
    """Own target-specific Rust planning and post-runtime output handoff."""

    def __init__(
        self,
        *,
        io_service: ExecutorIOService | None = None,
        validation_service: ExecutionValidationService | None = None,
    ) -> None:
        self._io_service = io_service
        self._typed_validation_service = validation_service or ExecutionValidationService(
            io_service=io_service
        )

    def plan_typed(self, request: TypedOutputPlanRequest) -> TypedOutputTargetPlan:
        """Plan target registration and shared options from typed syntax only."""

        if not isinstance(request, TypedOutputPlanRequest):
            raise TypeError("request must be TypedOutputPlanRequest")

        validation_result = self._typed_validation_service.validate_typed_data_target_duplicates(
            TypedValidationRequest(
                block=request.block,
                source=request.source,
                explicit_output_targets=request.explicit_output_targets,
            )
        )
        if validation_result.diagnostics:
            return TypedOutputTargetPlan(
                targets=(),
                default_target=None,
                suppress_implicit_output=False,
                effective_output_options={},
                diagnostics=validation_result.diagnostics,
            )

        resolved_targets = self._typed_validation_service.resolve_typed_output_targets(
            request.block,
            request.explicit_output_targets,
        )
        return plan_typed_output_targets(
            block=request.block,
            source=request.source,
            resolved_output_targets=resolved_targets,
            normalize_target=self._normalize_target,
        )

    def plan_typed_shaping(
        self, request: TypedOutputShapingRequest
    ) -> TypedOutputShapingResult:
        """Plan projection and expected types from typed syntax and the option policy."""

        if not isinstance(request, TypedOutputShapingRequest):
            raise TypeError("request must be TypedOutputShapingRequest")
        target_plan = request.output_target_plan
        if target_plan.diagnostics:
            return TypedOutputShapingResult(diagnostics=target_plan.diagnostics)

        targets = tuple(self._normalize_target(target) for target in target_plan.targets)
        statements = request.block.statements
        helper_plan = request.helper_plan
        slot_registry = self.build_typed_slot_registry(
            block=request.block,
            resolved_inputs=request.resolved_inputs,
            helper_plan=helper_plan,
        )

        keep_vars = self._typed_name_list(statements, "KEEP")
        drop_vars = self._typed_name_list(statements, "DROP")
        base_rename_map = self._typed_rename_map(statements)
        base_excluded = self._typed_internal_output_names(
            statements=statements,
            resolved_inputs=request.resolved_inputs,
            additional_internal_names=helper_plan.helper_runtime_columns,
        )
        base_excluded.update(
            target
            for source, target in base_rename_map.items()
            if _column_key(source) in {_column_key(name) for name in base_excluded}
        )

        base_columns = self._filter_excluded_output_columns(
            columns=(
                *tuple(slot_registry.get("source_slots", {}).keys()),
                *tuple(slot_registry.get("mutable_slots", {}).keys()),
            ),
            excluded_names=base_excluded,
        )
        projected_base_unfiltered = self._project_output_columns(
            columns=base_columns,
            keep=keep_vars,
            drop=drop_vars,
            rename_map=base_rename_map,
        )
        rename_statement = next(
            (statement for statement in statements if statement.kind == "RENAME"),
            None,
        )
        if has_output_projection_collision(projected_base_unfiltered):
            return TypedOutputShapingResult(
                diagnostics=(
                    build_typed_validation_diagnostic(
                        code="RUNTIME_RENAME_STATEMENT_INVALID",
                        message="RENAME statement has duplicate target variable names.",
                        span=None if rename_statement is None else rename_statement.span,
                        source=request.source,
                        label_message="invalid rename",
                        stage="validate",
                    ),
                )
            )
        projected_base = self._filter_excluded_output_columns(
            columns=projected_base_unfiltered,
            excluded_names=base_excluded,
        )
        base_types = self._filter_excluded_type_expectations(
            type_expectations=self._infer_output_type_expectations(
                resolved_inputs=request.resolved_inputs,
                late_created_columns=tuple(slot_registry.get("mutable_slots", {}).keys()),
                rename_map=base_rename_map,
                keep=keep_vars,
                drop=drop_vars,
            ),
            excluded_names=base_excluded,
        )

        projected_columns_by_target: dict[str, tuple[str, ...]] = {}
        rename_map_by_target: dict[str, dict[str, str]] = {}
        type_expectations_by_target: dict[str, dict[str, str]] = {}
        for target in targets:
            options = self._typed_target_options(target_plan, target)
            target_keep = options.keep_vars
            target_drop = options.drop_vars
            target_rename = dict(options.rename_map)
            excluded = set(base_excluded)
            excluded.update(
                renamed
                for source, renamed in target_rename.items()
                if _column_key(source) in {_column_key(name) for name in excluded}
            )
            projected_target_unfiltered = self._project_output_columns(
                columns=projected_base,
                keep=target_keep,
                drop=target_drop,
                rename_map=target_rename,
            )
            if has_output_projection_collision(projected_target_unfiltered):
                data_statement = next(
                    (statement for statement in statements if statement.kind == "DATA"),
                    None,
                )
                return TypedOutputShapingResult(
                    diagnostics=(
                        build_typed_validation_diagnostic(
                            code="RUNTIME_DATASET_OPTION_INVALID",
                            message=(
                                "Dataset option RENAME= has duplicate target names "
                                f"for output target '{target}'."
                            ),
                            span=None if data_statement is None else data_statement.span,
                            source=request.source,
                            label_message="invalid rename",
                            stage="validate",
                        ),
                    )
                )
            projected_columns_by_target[target] = self._filter_excluded_output_columns(
                columns=projected_target_unfiltered,
                excluded_names=excluded,
            )
            rename_map_by_target[target] = base_rename_map | target_rename
            type_expectations_by_target[target] = self._filter_excluded_type_expectations(
                type_expectations=self._apply_type_projection(
                    type_expectations=base_types,
                    keep=target_keep,
                    drop=target_drop,
                    rename_map=target_rename,
                ),
                excluded_names=excluded,
            )

        return TypedOutputShapingResult(
            output_plan=OutputHandoffPlan(
                targets=targets,
                projected_columns_by_target=projected_columns_by_target,
                rename_map_by_target=rename_map_by_target,
                type_expectations_by_target=type_expectations_by_target,
                fallback_reason=None,
            )
        )

    def build_typed_slot_registry(
        self,
        *,
        block: BlockSyntax,
        resolved_inputs: Mapping[str, DataSetRef],
        helper_plan: TypedHelperPlan,
    ) -> dict[str, dict[str, int]]:
        """Expose the output owner's typed slot registry to final planning."""

        source_statement = next(
            (
                statement.payload
                for statement in block.statements
                if statement.kind in {"SET", "MERGE"}
                and isinstance(statement.payload, SourceStatementSyntax)
            ),
            None,
        )
        return self._build_slot_registry(
            resolved_inputs=resolved_inputs,
            source_specs=self._typed_source_specs(source_statement),
            runtime_helper_columns=helper_plan.helper_runtime_columns,
            typed_step_defined_columns=self._typed_step_defined_columns(
                block.statements
            ),
        )

    def finalize_typed(self, request: TypedOutputFinalizeRequest) -> OutputHandoffResult:
        """Finalize outputs using the same confirmed typed policy as planning."""

        if not isinstance(request, TypedOutputFinalizeRequest):
            raise TypeError("request must be TypedOutputFinalizeRequest")
        target_plan = request.output_target_plan
        if target_plan.diagnostics:
            return OutputHandoffResult(
                output_source_metadata_lineage=request.source_metadata_lineage,
                diagnostics=target_plan.diagnostics,
            )
        if not isinstance(request.output_plan, OutputHandoffPlan):
            raise TypeError("output_plan must be an OutputHandoffPlan")

        targets = tuple(self._normalize_target(target) for target in target_plan.targets)
        if request.output_plan.targets != targets:
            raise ValueError("output_plan targets must match output_target_plan")
        lineage_plan = self._build_typed_output_lineage_projection_plan(
            block=request.block,
            output_target_plan=target_plan,
            targets=targets,
        )
        exclusions = self._typed_internal_output_names_by_target(
            block=request.block,
            resolved_inputs=request.resolved_inputs,
            helper_runtime_columns=request.helper_plan.helper_runtime_columns,
            output_target_plan=target_plan,
            targets=targets,
        )
        by_keys = self._typed_name_list(request.block.statements, "BY")
        filtered_outputs, output_diagnostics = self._apply_output_stage_options(
            ast_statements=(),
            resolved_inputs=request.resolved_inputs,
            outputs=request.outputs,
            source_metadata_lineage=request.source_metadata_lineage,
            preparation_kind=request.preparation_kind,
            output_lineage_plan=lineage_plan,
            rewrite_plan=request.helper_plan,
            excluded_names_by_target=exclusions,
            typed_by_keys=by_keys,
        )
        if output_diagnostics:
            return OutputHandoffResult(
                output_source_metadata_lineage=request.source_metadata_lineage,
                diagnostics=output_diagnostics,
            )

        metadata_outputs, metadata_diagnostics = self._apply_typed_output_metadata(
            block=request.block,
            outputs=filtered_outputs,
            source_metadata_lineage=request.source_metadata_lineage,
            output_target_plan=target_plan,
        )
        if metadata_diagnostics:
            return OutputHandoffResult(
                output_source_metadata_lineage=request.source_metadata_lineage,
                diagnostics=metadata_diagnostics,
            )

        public_outputs, presentation_diagnostics = self._coerce_public_execute_outputs(
            outputs=metadata_outputs,
            preserve_memory_presentation=request.preserve_memory_presentation,
        )
        if presentation_diagnostics:
            return OutputHandoffResult(
                output_source_metadata_lineage=request.source_metadata_lineage,
                diagnostics=presentation_diagnostics,
            )
        return OutputHandoffResult(
            outputs=metadata_outputs,
            public_outputs=public_outputs,
            outputs_arrow=self.build_lazy_outputs(metadata_outputs),
            output_source_metadata_lineage=request.source_metadata_lineage,
            diagnostics=(),
        )

    def _build_typed_output_lineage_projection_plan(
        self,
        *,
        block: BlockSyntax,
        output_target_plan: TypedOutputTargetPlan,
        targets: tuple[str, ...],
    ) -> OutputLineageProjectionPlan:
        keep_vars = self._typed_name_list(block.statements, "KEEP")
        drop_vars = self._typed_name_list(block.statements, "DROP")
        rename_map = self._typed_rename_map(block.statements)
        stages_by_target: dict[str, tuple[OutputProjectionStage, ...]] = {}
        for target in targets:
            options = self._typed_target_options(output_target_plan, target)
            stages_by_target[target] = (
                OutputProjectionStage(
                    keep=keep_vars,
                    drop=drop_vars,
                    rename=rename_map,
                ),
                OutputProjectionStage(
                    keep=options.keep_vars,
                    drop=options.drop_vars,
                    rename=dict(options.rename_map),
                ),
            )
        return OutputLineageProjectionPlan(stages_by_target=stages_by_target)

    @staticmethod
    def _typed_name_list(statements: Sequence[StatementSyntax], kind: str) -> tuple[str, ...]:
        statement = next((item for item in statements if item.kind == kind), None)
        if statement is None or not isinstance(statement.payload, NameListStatementSyntax):
            return ()
        return tuple(item.name for item in statement.payload.names)

    @staticmethod
    def _typed_rename_map(statements: Sequence[StatementSyntax]) -> dict[str, str]:
        statement = next((item for item in statements if item.kind == "RENAME"), None)
        if statement is None or not isinstance(statement.payload, RenameStatementSyntax):
            return {}
        return dict(statement.payload.pairs)

    @staticmethod
    def _typed_target_options(
        output_target_plan: TypedOutputTargetPlan, target: str
    ) -> DatasetOptionSyntax:
        key = _dataset_key(target)
        if key not in output_target_plan.effective_output_options:
            raise ValueError(
                f"typed output policy is missing registered target {target!r}"
            )
        options = output_target_plan.effective_output_options[key]
        if not isinstance(options, DatasetOptionSyntax):
            raise TypeError("typed output policy values must be DatasetOptionSyntax")
        return options

    @staticmethod
    def _typed_source_specs(
        source_statement: SourceStatementSyntax | None,
    ) -> list[dict[str, Any]]:
        if source_statement is None:
            return []
        return [
            {
                "source": reference.name,
                "keep": list(reference.options.keep_vars),
                "drop": list(reference.options.drop_vars),
                "where": where_expression_to_dict(reference.options.where_expr),
                "rename": dict(reference.options.rename_map),
                "firstobs": reference.options.firstobs,
                "obs": reference.options.obs,
            }
            for reference in source_statement.refs
        ]

    @staticmethod
    def _typed_step_defined_columns(
        statements: Sequence[StatementSyntax],
    ) -> tuple[str, ...]:
        defined: list[str] = []
        seen: set[str] = set()

        def add(name: str | None) -> None:
            if name is None:
                return
            key = _column_key(name)
            if key not in seen:
                seen.add(key)
                defined.append(name)

        def add_action(action: Any) -> None:
            if not isinstance(action, StatementSyntax):
                return
            payload = action.payload
            if isinstance(payload, AssignStatementSyntax):
                target = payload.target
                if isinstance(target, (NameTargetSyntax, ArrayElementTargetSyntax)):
                    add(target.name)
            elif isinstance(payload, SumStatementSyntax):
                add(payload.target.name)
            elif isinstance(payload, IfStatementSyntax):
                add_action(payload.then_action)
            elif isinstance(payload, ElseStatementSyntax):
                add_action(payload.action)

        for statement in statements:
            payload = statement.payload
            if isinstance(payload, AssignStatementSyntax):
                target = payload.target
                if isinstance(target, (NameTargetSyntax, ArrayElementTargetSyntax)):
                    add(target.name)
            elif isinstance(payload, SumStatementSyntax):
                add(payload.target.name)
            elif isinstance(payload, IfStatementSyntax):
                add_action(payload.then_action)
            elif isinstance(payload, ElseStatementSyntax):
                add_action(payload.action)
            elif isinstance(payload, DoStatementSyntax):
                add(payload.loop_var)
            elif isinstance(payload, ArrayStatementSyntax):
                for name in payload.variables:
                    add(name)
            elif statement.kind == "RETAIN" and isinstance(
                payload, NameListStatementSyntax
            ):
                for name in payload.names:
                    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\.]*", name.name):
                        add(name.name)
        return tuple(defined)

    def _typed_internal_output_names(
        self,
        *,
        statements: Sequence[StatementSyntax],
        resolved_inputs: Mapping[str, DataSetRef],
        additional_internal_names: Sequence[str] = (),
    ) -> set[str]:
        source = next(
            (
                statement.payload
                for statement in statements
                if statement.kind in {"SET", "MERGE"}
                and isinstance(statement.payload, SourceStatementSyntax)
            ),
            None,
        )
        source = source if isinstance(source, SourceStatementSyntax) else None
        input_vars = tuple(
            reference.options.in_var
            for reference in (source.refs if source is not None else ())
            if reference.options.in_var
        )
        by_keys = self._typed_name_list(statements, "BY")
        options = source.options if source is not None else None
        excluded = self._collect_internal_variable_names(
            in_option_vars=input_vars,
            indsname_var=options.indsname_var if options is not None else None,
            end_var=options.end_var if options is not None else None,
            by_keys=by_keys,
        )
        for input_ref in resolved_inputs.values():
            excluded.update(input_ref.preparation.internal_variable_names)
        if source is not None:
            for reference in source.refs:
                for source_name, target_name in reference.options.rename_map:
                    if _column_key(source_name) in {
                        _column_key(name) for name in excluded
                    }:
                        excluded.add(target_name)
        for source_name, target_name in self._typed_rename_map(statements).items():
            if _column_key(source_name) in {_column_key(name) for name in excluded}:
                excluded.add(target_name)
        excluded.update(name for name in additional_internal_names if name)
        return excluded

    def _typed_internal_output_names_by_target(
        self,
        *,
        block: BlockSyntax,
        resolved_inputs: Mapping[str, DataSetRef],
        helper_runtime_columns: Sequence[str],
        output_target_plan: TypedOutputTargetPlan,
        targets: tuple[str, ...],
    ) -> dict[str, set[str]]:
        statements = block.statements
        base = self._typed_internal_output_names(
            statements=statements,
            resolved_inputs=resolved_inputs,
            additional_internal_names=helper_runtime_columns,
        )
        result: dict[str, set[str]] = {}
        for target in targets:
            excluded = set(base)
            options = self._typed_target_options(output_target_plan, target)
            for source, renamed in options.rename_map:
                if _column_key(source) in {_column_key(name) for name in excluded}:
                    excluded.add(renamed)
            result[target] = excluded
        return result

    def _apply_output_stage_options(
        self,
        *,
        ast_statements: Sequence[Any],
        resolved_inputs: Mapping[str, DataSetRef],
        outputs: Mapping[str, DataSetRef],
        rewrite_plan: Any,
        source_metadata_lineage: SourceMetadataLineage | None = None,
        preparation_kind: str | None = None,
        output_lineage_plan: OutputLineageProjectionPlan | None = None,
        excluded_names_by_target: Mapping[str, set[str]] | None = None,
        typed_by_keys: Sequence[str] | None = None,
    ) -> tuple[dict[str, DataSetRef], tuple[Diagnostic, ...]]:
        diagnostics: list[Diagnostic] = []
        excluded_names = (
            set()
            if excluded_names_by_target is not None
            else self._collect_internal_output_variable_names(
                ast_statements=ast_statements,
                resolved_inputs=resolved_inputs,
                additional_internal_names=tuple(rewrite_plan.helper_runtime_columns),
            )
        )
        import pyarrow as pa

        source_tables = tuple(
            dataset_ref.payload
            for dataset_ref in resolved_inputs.values()
            if dataset_ref.kind.strip().lower() == "arrow_table"
            and isinstance(dataset_ref.payload, pa.Table)
        )
        by_keys = (
            tuple(typed_by_keys)
            if typed_by_keys is not None
            else self._extract_variable_list(ast_statements, "BY")
        )

        filtered: dict[str, DataSetRef] = {}
        for target, dataset_ref in outputs.items():
            target_excluded = excluded_names
            if excluded_names_by_target is not None:
                target_excluded = excluded_names_by_target.get(target, set())
                if not target_excluded:
                    target_key = _dataset_key(target)
                    target_excluded = next(
                        (
                            values
                            for candidate, values in excluded_names_by_target.items()
                            if _dataset_key(candidate) == target_key
                        ),
                        set(),
                    )
            target_excluded_keys = {_column_key(name) for name in target_excluded}
            normalized_kind = dataset_ref.kind.strip().lower()

            if normalized_kind == "memory":
                if not target_excluded:
                    filtered[target] = dataset_ref
                    continue
                payload = dataset_ref.payload
                if isinstance(payload, Sequence) and not isinstance(
                    payload, (str, bytes, bytearray)
                ):
                    filtered_rows: list[dict[str, Any]] = []
                    for row in payload:
                        if not isinstance(row, Mapping):
                            diagnostics.append(
                                Diagnostic(
                                    code="RUNTIME_OUTPUT_TARGET_NOT_FOUND",
                                    severity="error",
                                    location=f"dataset:{target}",
                                    message="Output row must be a mapping.",
                                )
                            )
                            return {}, tuple(diagnostics)
                        filtered_rows.append(
                            {
                                key: value
                                for key, value in dict(row).items()
                                if (
                                    _column_key(str(key)) not in target_excluded_keys
                                    if excluded_names_by_target is not None
                                    else key not in target_excluded
                                )
                            }
                        )
                    filtered[target] = DataSetRef(
                        kind=dataset_ref.kind,
                        location=dataset_ref.location,
                        payload=filtered_rows,
                        metadata=dataset_ref.metadata,
                    )
                    continue
                filtered[target] = dataset_ref
                continue

            if normalized_kind == "arrow_table" and isinstance(
                dataset_ref.payload, pa.Table
            ):
                try:
                    projected_lineage = source_metadata_lineage
                    projected_by_keys = by_keys
                    if output_lineage_plan is not None and source_metadata_lineage is not None:
                        stages_by_target = getattr(
                            output_lineage_plan, "stages_by_target", {}
                        )
                        stages = stages_by_target.get(target)
                        if stages is None:
                            target_key = _dataset_key(target)
                            stages = next(
                                (
                                    candidate_stages
                                    for candidate_target, candidate_stages in stages_by_target.items()
                                    if _dataset_key(candidate_target) == target_key
                                ),
                                (),
                            )
                        projected_by_keys = _project_output_column_names(
                            by_keys,
                            stages=stages,
                        )
                        projection_result = project_lineage_for_output(
                            source_metadata_lineage,
                            stages=stages,
                            final_columns=dataset_ref.payload.schema.names,
                        )
                        if projection_result.invariant_error is not None:
                            diagnostics.append(
                                Diagnostic(
                                    code="CONVERT_OUTPUT_FAILED",
                                    severity="error",
                                    location=f"dataset:{target}",
                                    message=(
                                        "Failed to apply output metadata lineage projection: "
                                        f"{projection_result.invariant_error}"
                                    ),
                                )
                            )
                            return {}, tuple(diagnostics)
                        projected_lineage = projection_result.lineage
                    metadata_restored_table = restore_arrow_schema_from_sources(
                        dataset_ref.payload,
                        source_tables=source_tables,
                        lineage=projected_lineage,
                        preparation_kind=preparation_kind,
                        by_keys=projected_by_keys,
                    )
                    retained_column_indices = [
                        index
                        for index, name in enumerate(metadata_restored_table.schema.names)
                        if (
                            _column_key(name) not in target_excluded_keys
                            if excluded_names_by_target is not None
                            else name not in target_excluded
                        )
                    ]
                    filtered_table = metadata_restored_table.select(retained_column_indices)
                    filtered[target] = DataSetRef(
                        kind="arrow_table",
                        location=dataset_ref.location,
                        payload=filtered_table,
                        metadata=dataset_ref.metadata,
                    )
                except Exception as error:
                    diagnostics.append(
                        Diagnostic(
                            code="CONVERT_OUTPUT_FAILED",
                            severity="error",
                            location=f"dataset:{target}",
                            message=(
                                "Failed to apply output-stage internal variable filtering: "
                                f"{error}"
                            ),
                        )
                    )
                    return {}, tuple(diagnostics)
                continue

            filtered[target] = dataset_ref

        return filtered, tuple(diagnostics)

    def _apply_output_metadata(
        self,
        *,
        ast_statements: Sequence[Any],
        outputs: Mapping[str, DataSetRef],
        execution_plan: Mapping[str, Any] | None = None,
        source_metadata_lineage: SourceMetadataLineage | None = None,
    ) -> tuple[dict[str, DataSetRef], tuple[Diagnostic, ...]]:
        try:
            source_metadata: dict[str, Any] = {}
            if source_metadata_lineage is not None and source_metadata_lineage.ordered_sources:
                source_metadata = dict(
                    source_metadata_lineage.ordered_sources[0].dataset_ref_metadata
                )

            data_statement = next(
                (statement for statement in ast_statements if statement.kind == "DATA"),
                None,
            )
            label_statement = next(
                (statement for statement in ast_statements if statement.kind == "LABEL"),
                None,
            )
            if data_statement is None and label_statement is None and not source_metadata:
                return dict(outputs), ()

            column_labels = dict(getattr(label_statement, "label_map", {}))
            projection_spec: Mapping[str, Any] | None = None
            if isinstance(execution_plan, Mapping):
                raw_projection = execution_plan.get("post_projection")
                if isinstance(raw_projection, Mapping):
                    projection_spec = raw_projection
            if projection_spec is None:
                projection_spec = {
                    "keep": self._extract_variable_list(ast_statements, "KEEP"),
                    "drop": self._extract_variable_list(ast_statements, "DROP"),
                    "rename": dict(
                        getattr(
                            next(
                                (
                                    statement
                                    for statement in ast_statements
                                    if statement.kind == "RENAME"
                                ),
                                None,
                            ),
                            "rename_map",
                            {},
                        )
                    ),
                }
            column_labels = self._project_column_labels(
                column_labels=column_labels,
                keep_vars=(
                    projection_spec.get("keep", ())
                    if isinstance(projection_spec, Mapping)
                    else ()
                ),
                drop_vars=(
                    projection_spec.get("drop", ())
                    if isinstance(projection_spec, Mapping)
                    else ()
                ),
                rename_map=(
                    projection_spec.get("rename", {})
                    if isinstance(projection_spec, Mapping)
                    else {}
                ),
            )
            dataset_labels: dict[str, str] = {}
            if data_statement is not None:
                for output_ref in getattr(data_statement, "output_refs", ()):
                    label = getattr(getattr(output_ref, "options", None), "label", None)
                    if label:
                        dataset_labels[_dataset_key(output_ref.name)] = label

            updated: dict[str, DataSetRef] = {}
            for target, dataset_ref in outputs.items():
                metadata = dict(source_metadata)
                metadata.update(dataset_ref.metadata)
                dataset_label = dataset_labels.get(_dataset_key(target))
                if dataset_label:
                    metadata["memlabel"] = dataset_label
                if column_labels:
                    metadata["column_labels"] = dict(column_labels)
                payload = dataset_ref.payload
                if dataset_ref.kind.strip().lower() == "arrow_table":
                    payload = _apply_arrow_metadata(payload, metadata)
                updated[target] = DataSetRef(
                    kind=dataset_ref.kind,
                    location=dataset_ref.location,
                    payload=payload,
                    metadata=metadata,
                )
            return updated, ()
        except Exception as error:
            return {}, (
                Diagnostic(
                    code="CONVERT_OUTPUT_FAILED",
                    severity="error",
                    message=f"Failed to apply output metadata: {error}",
                ),
            )

    def _apply_typed_output_metadata(
        self,
        *,
        block: BlockSyntax,
        outputs: Mapping[str, DataSetRef],
        source_metadata_lineage: SourceMetadataLineage | None,
        output_target_plan: TypedOutputTargetPlan,
    ) -> tuple[dict[str, DataSetRef], tuple[Diagnostic, ...]]:
        try:
            source_metadata: dict[str, Any] = {}
            if source_metadata_lineage and source_metadata_lineage.ordered_sources:
                source_metadata = dict(
                    source_metadata_lineage.ordered_sources[0].dataset_ref_metadata
                )
            column_labels = source_metadata.get("column_labels", {})
            if not isinstance(column_labels, Mapping):
                column_labels = {}
            labels_statement = next(
                (
                    statement.payload
                    for statement in block.statements
                    if statement.kind == "LABEL"
                    and isinstance(statement.payload, LabelStatementSyntax)
                ),
                None,
            )
            if isinstance(labels_statement, LabelStatementSyntax):
                column_labels = dict(column_labels)
                column_labels.update(
                    {label.name: label.label for label in labels_statement.labels}
                )
            keep_vars = self._typed_name_list(block.statements, "KEEP")
            drop_vars = self._typed_name_list(block.statements, "DROP")
            rename_map = self._typed_rename_map(block.statements)
            updated: dict[str, DataSetRef] = {}
            for target, dataset_ref in outputs.items():
                options = self._typed_target_options(output_target_plan, target)
                projected_labels = self._project_column_labels(
                    column_labels=column_labels,
                    keep_vars=keep_vars,
                    drop_vars=drop_vars,
                    rename_map=rename_map,
                )
                projected_labels = self._project_column_labels(
                    column_labels=projected_labels,
                    keep_vars=options.keep_vars,
                    drop_vars=options.drop_vars,
                    rename_map=dict(options.rename_map),
                )
                metadata = dict(source_metadata)
                metadata.update(dataset_ref.metadata)
                if options.label:
                    metadata["memlabel"] = options.label
                if projected_labels:
                    metadata["column_labels"] = projected_labels
                payload = dataset_ref.payload
                if dataset_ref.kind.strip().lower() == "arrow_table":
                    payload = _apply_arrow_metadata(payload, metadata)
                updated[target] = DataSetRef(
                    kind=dataset_ref.kind,
                    location=dataset_ref.location,
                    payload=payload,
                    metadata=metadata,
                )
            return updated, ()
        except Exception as error:
            return {}, (
                Diagnostic(
                    code="CONVERT_OUTPUT_FAILED",
                    severity="error",
                    message=f"Failed to apply output metadata: {error}",
                ),
            )

    @staticmethod
    def _project_column_labels(
        *,
        column_labels: Mapping[str, str],
        keep_vars: Sequence[str],
        drop_vars: Sequence[str],
        rename_map: Mapping[str, str],
    ) -> dict[str, str]:
        if not column_labels:
            return {}

        entries = [(str(name), str(label)) for name, label in column_labels.items()]
        if keep_vars:
            keep_keys = {_column_key(name) for name in keep_vars}
            entries = [
                (name, label)
                for name, label in entries
                if _column_key(name) in keep_keys
            ]
        if drop_vars:
            drop_keys = {_column_key(name) for name in drop_vars}
            entries = [
                (name, label)
                for name, label in entries
                if _column_key(name) not in drop_keys
            ]
        if rename_map:
            normalized_rename = {
                _column_key(source): str(target)
                for source, target in rename_map.items()
            }
            entries = [
                (normalized_rename.get(_column_key(name), name), label)
                for name, label in entries
            ]
        return dict(entries)

    def _coerce_public_execute_outputs(
        self,
        *,
        outputs: Mapping[str, DataSetRef],
        preserve_memory_presentation: bool,
    ) -> tuple[dict[str, DataSetRef], tuple[Diagnostic, ...]]:
        if not preserve_memory_presentation:
            return dict(outputs), ()

        public_outputs: dict[str, DataSetRef] = {}
        for target, dataset_ref in outputs.items():
            normalized_kind = dataset_ref.kind.strip().lower()
            if normalized_kind != "arrow_table" or not hasattr(
                dataset_ref.payload, "to_pylist"
            ):
                public_outputs[target] = dataset_ref
                continue
            try:
                rows = dataset_ref.payload.to_pylist()
            except Exception as error:
                return {}, (
                    Diagnostic(
                        code="CONVERT_OUTPUT_FAILED",
                        severity="error",
                        location=f"dataset:{target}",
                        message=f"Failed to prepare public output: {error}",
                    ),
                )
            sparse_rows: list[Any] = []
            for row in rows:
                if isinstance(row, Mapping):
                    sparse_rows.append(
                        {
                            key: value
                            for key, value in dict(row).items()
                            if value is not None
                        }
                    )
                else:
                    sparse_rows.append(row)
            public_outputs[target] = DataSetRef(
                kind="memory",
                location=dataset_ref.location,
                payload=sparse_rows,
                metadata=dataset_ref.metadata,
            )
        return public_outputs, ()

    def build_lazy_outputs(
        self,
        outputs: Mapping[str, DataSetRef],
    ) -> Mapping[str, Any]:
        if self._io_service is None:
            return MappingProxyType({})
        return _LazyArrowOutputs(
            outputs,
            self._io_service.dataset_ref_to_arrow_table,
        )

    def _normalize_target(self, target: str) -> str:
        if self._io_service is not None:
            return self._io_service.normalize_dataset_name(target)
        normalized = str(target).strip()
        if normalized.lower().startswith("work."):
            return normalized[5:]
        return normalized

    @staticmethod
    def _source_specs(source_statement: ParsedStatement | None) -> list[dict[str, Any]]:
        if source_statement is None:
            return []
        source_specs: list[dict[str, Any]] = []
        for dataset_ref in tuple(getattr(source_statement, "dataset_refs", ()) or ()):
            options = getattr(dataset_ref, "options", None)
            source_specs.append(
                {
                    "source": dataset_ref.name,
                    "keep": list(getattr(options, "keep_vars", ()) or ()),
                    "drop": list(getattr(options, "drop_vars", ()) or ()),
                    "where": where_expression_to_dict(
                        getattr(options, "where_expr", None)
                    ),
                    "rename": dict(getattr(options, "rename_map", {}) or {}),
                    "firstobs": getattr(options, "firstobs", None),
                    "obs": getattr(options, "obs", None),
                }
            )
        return source_specs

    def _build_slot_registry(
        self,
        *,
        resolved_inputs: Mapping[str, DataSetRef],
        source_specs: Sequence[Mapping[str, Any]],
        runtime_helper_columns: Sequence[str] = (),
        typed_step_defined_columns: Sequence[str],
    ) -> dict[str, dict[str, int]]:
        source_columns: list[str] = []
        seen_source: set[str] = set()
        for source_spec in source_specs:
            source_name = source_spec.get("source")
            if not isinstance(source_name, str):
                continue
            input_ref = self._resolve_input_ref(resolved_inputs, source_name)
            if input_ref is None:
                continue
            ordered_columns = self._infer_dataset_column_order(input_ref)
            if ordered_columns is None:
                continue
            if self._uses_prepared_row_input(input_ref):
                planned_columns = ordered_columns
            else:
                planned_columns = self._apply_source_option_column_order(
                    ordered_columns, source_spec
                )
            for column_name in planned_columns:
                if column_name in seen_source:
                    continue
                seen_source.add(column_name)
                source_columns.append(column_name)

        mutable_columns = list(typed_step_defined_columns)
        seen_mutable = {_column_key(name) for name in mutable_columns}
        for helper_name in runtime_helper_columns:
            helper_key = _column_key(helper_name)
            if helper_key in seen_mutable:
                continue
            seen_mutable.add(helper_key)
            mutable_columns.append(helper_name)
        return {
            "source_slots": {name: index for index, name in enumerate(source_columns)},
            "mutable_slots": {name: index for index, name in enumerate(mutable_columns)},
            "automatic_slots": {"_N_": 0, "_ERROR_": 1},
        }

    @staticmethod
    def _resolve_input_ref(
        resolved_inputs: Mapping[str, DataSetRef],
        source_name: str,
    ) -> DataSetRef | None:
        direct = resolved_inputs.get(source_name)
        if direct is not None:
            return direct
        source_key = _dataset_key(source_name)
        for candidate_name, candidate in resolved_inputs.items():
            if _dataset_key(str(candidate_name)) == source_key:
                return candidate
        return None

    @staticmethod
    def _uses_prepared_row_input(dataset_ref: DataSetRef) -> bool:
        return (
            getattr(getattr(dataset_ref, "preparation", None), "kind", "none")
            != "none"
        )

    def _infer_dataset_column_order(
        self,
        input_ref: DataSetRef,
    ) -> tuple[str, ...] | None:
        payload = input_ref.payload
        if input_ref.kind.strip().lower() == "arrow_table" and hasattr(payload, "schema"):
            names = getattr(getattr(payload, "schema", None), "names", None)
            if names is not None:
                return tuple(str(name) for name in names)
        if self._io_service is None:
            return None
        rows, load_error = self._io_service.load_input_rows(input_ref)
        if load_error is not None:
            return None
        ordered: list[str] = []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            for name in row:
                key = _column_key(str(name))
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(str(name))
        return tuple(ordered)

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

    def _collect_internal_output_variable_names(
        self,
        *,
        ast_statements: Sequence[Any],
        resolved_inputs: Mapping[str, DataSetRef],
        additional_internal_names: Sequence[str] = (),
    ) -> set[str]:
        source_statement = next(
            (statement for statement in ast_statements if statement.kind in {"SET", "MERGE"}),
            None,
        )
        source_refs = tuple(getattr(source_statement, "dataset_refs", ()) or ())
        in_option_vars = [
            ref.options.in_var
            for ref in source_refs
            if getattr(getattr(ref, "options", None), "in_var", None) is not None
        ]
        by_keys = self._extract_variable_list(ast_statements, "BY")
        statement_options = getattr(source_statement, "statement_options", None)
        excluded = self._collect_internal_variable_names(
            in_option_vars=in_option_vars,
            indsname_var=getattr(statement_options, "indsname_var", None),
            end_var=getattr(statement_options, "end_var", None),
            by_keys=by_keys,
        )
        for input_ref in resolved_inputs.values():
            provenance = getattr(input_ref, "preparation", None)
            internal_names = tuple(
                getattr(provenance, "internal_variable_names", ()) or ()
            )
            excluded.update(name for name in internal_names if name)
        rename_statement = next(
            (statement for statement in ast_statements if statement.kind == "RENAME"),
            None,
        )
        if rename_statement is not None:
            for source, target in rename_statement.rename_map.items():
                if source in excluded:
                    excluded.add(target)
        data_statement = next(
            (statement for statement in ast_statements if statement.kind == "DATA"),
            None,
        )
        if data_statement is not None:
            for output_ref in getattr(data_statement, "output_refs", ()):
                option_spec = getattr(output_ref, "options", None)
                for source, target in dict(getattr(option_spec, "rename_map", {}) or {}).items():
                    if source in excluded:
                        excluded.add(target)
        excluded.update(name for name in additional_internal_names if name)
        return excluded

    @staticmethod
    def _project_output_columns(
        *,
        columns: Sequence[str],
        keep: Sequence[str],
        drop: Sequence[str],
        rename_map: Mapping[str, str],
    ) -> tuple[str, ...]:
        ordered = list(columns)
        if keep:
            keep_keys = {_column_key(name) for name in keep}
            ordered = [name for name in ordered if _column_key(name) in keep_keys]
        if drop:
            drop_keys = {_column_key(name) for name in drop}
            ordered = [name for name in ordered if _column_key(name) not in drop_keys]
        if rename_map:
            normalized_rename = {
                _column_key(source): str(target)
                for source, target in rename_map.items()
            }
            ordered = [normalized_rename.get(_column_key(name), name) for name in ordered]
        return tuple(ordered)

    @staticmethod
    def _filter_excluded_output_columns(
        *,
        columns: Sequence[str],
        excluded_names: set[str],
    ) -> tuple[str, ...]:
        excluded_keys = {_column_key(name) for name in excluded_names}
        ordered: list[str] = []
        seen: set[str] = set()
        for name in columns:
            key = _column_key(name)
            if key in excluded_keys or key in seen:
                continue
            seen.add(key)
            ordered.append(name)
        return tuple(ordered)

    @staticmethod
    def _filter_excluded_type_expectations(
        *,
        type_expectations: Mapping[str, str],
        excluded_names: set[str],
    ) -> dict[str, str]:
        excluded_keys = {_column_key(name) for name in excluded_names}
        filtered: dict[str, str] = {}
        for name, type_name in type_expectations.items():
            key = _column_key(name)
            if key in excluded_keys or key in {_column_key(existing) for existing in filtered}:
                continue
            filtered[name] = type_name
        return filtered

    def _infer_output_type_expectations(
        self,
        *,
        resolved_inputs: Mapping[str, DataSetRef],
        late_created_columns: Sequence[str],
        rename_map: Mapping[str, str],
        keep: Sequence[str],
        drop: Sequence[str],
    ) -> dict[str, str]:
        inferred: dict[str, str] = {}
        for dataset_ref in resolved_inputs.values():
            for name, type_name in self._infer_dataset_column_types(dataset_ref).items():
                inferred.setdefault(name, type_name)
        for name in late_created_columns:
            # A mutable slot can replace a source column with a different
            # scalar type during this block, so its source schema is not a
            # valid output type constraint.
            inferred[name] = "dynamic"
        return self._apply_type_projection(
            type_expectations=inferred,
            keep=keep,
            drop=drop,
            rename_map=rename_map,
        )

    @staticmethod
    def _apply_type_projection(
        *,
        type_expectations: Mapping[str, str],
        keep: Sequence[str],
        drop: Sequence[str],
        rename_map: Mapping[str, str],
    ) -> dict[str, str]:
        projected = dict(type_expectations)
        if keep:
            keep_keys = {_column_key(name) for name in keep}
            projected = {
                name: type_name
                for name, type_name in projected.items()
                if _column_key(name) in keep_keys
            }
        if drop:
            drop_keys = {_column_key(name) for name in drop}
            projected = {
                name: type_name
                for name, type_name in projected.items()
                if _column_key(name) not in drop_keys
            }
        if rename_map:
            normalized_rename = {
                _column_key(source): str(target)
                for source, target in rename_map.items()
            }
            renamed: dict[str, str] = {}
            for name, type_name in projected.items():
                renamed[normalized_rename.get(_column_key(name), name)] = type_name
            projected = renamed
        return projected

    def _infer_dataset_column_types(self, dataset_ref: DataSetRef) -> dict[str, str]:
        normalized_kind = dataset_ref.kind.strip().lower()
        payload = dataset_ref.payload
        if normalized_kind == "arrow_table" and hasattr(payload, "schema"):
            schema = getattr(payload, "schema", None)
            if schema is not None and hasattr(schema, "names"):
                return {str(field.name): str(field.type) for field in schema}

        if self._io_service is None:
            return {}
        rows, load_error = self._io_service.load_input_rows(dataset_ref)
        if load_error is not None:
            return {}
        inferred: dict[str, str] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            for name, value in row.items():
                if value is None or str(name) in inferred:
                    continue
                inferred[str(name)] = type(value).__name__
        return inferred


__all__ = [
    "ExecutionOutputHandoffService",
    "OutputHandoffPlan",
    "OutputHandoffResult",
    "OutputLineageProjectionPlan",
    "OutputProjectionStage",
    "TypedOutputTargetPlan",
]
