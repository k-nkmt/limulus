from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Protocol

from ..naming import _column_key
from ..backend_integration.backend_dispatch_policy import BackendDispatchPolicy
from ..backend_integration.contracts import RuntimeExecutionContext
from ..models import CompatibilityNotice, DataSetRef, Diagnostic, ExecuteRequest, ExecuteResponse
from ..parser import ParserExecutionContext, SplitStageParserService
from ..session_parsing import parse_simple_filter
from .rewrites import RewritePlanner

if TYPE_CHECKING:
    from .coordinator import DataStepExecutor


@dataclass(frozen=True)
class HelperColumnPlan:
    arrow_columns: tuple[str, ...]
    fallback_columns: tuple[str, ...]
    source_aliases: tuple[str, ...]
    requires_row_prepare: bool
    fallback_reason: str | None


@dataclass(frozen=True)
class OutputHandoffPlan:
    targets: tuple[str, ...]
    projected_columns_by_target: dict[str, tuple[str, ...]]
    rename_map_by_target: dict[str, dict[str, str]]
    type_expectations_by_target: dict[str, dict[str, str]]
    fallback_reason: str | None


_MISSING_VALUE_NOTICE = CompatibilityNotice(
    id="COMPAT_MISSING_VALUE_SEMANTICS",
    category="missing-values",
    summary="Missing Value Semantics Difference: missing values '.' may have different comparison and ordering behavior than Python/Arrow None/NaN.",
    link="compat://missing-value-semantics",
)


def _collect_compatibility_notices(dsl_text: str, inputs: Mapping[str, DataSetRef]) -> tuple[CompatibilityNotice, ...]:
    _missing_literal = re.compile(r"(?<![\w])\.(?![\w])")

    def _payload_contains_none(payload: object) -> bool:
        if payload is None:
            return False
        if isinstance(payload, Mapping):
            return any(value is None for value in payload.values())
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
            for item in payload:
                if isinstance(item, Mapping) and any(value is None for value in item.values()):
                    return True
                if item is None:
                    return True
        return False

    has_missing = bool(_missing_literal.search(dsl_text))
    if not has_missing:
        has_missing = any(_payload_contains_none(dataset.payload) for dataset in inputs.values())

    return (_MISSING_VALUE_NOTICE,) if has_missing else ()


class _LazyArrowOutputs(Mapping[str, Any]):
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


class MacroHook(Protocol):
    def evaluate(self, dsl_text: str) -> str:
        ...


class NoOpMacroHook:
    def evaluate(self, dsl_text: str) -> str:
        return dsl_text


class UnsupportedSyntaxMacroHook:
    def __init__(self, parser_service: SplitStageParserService | None = None) -> None:
        self._parser_service = parser_service or SplitStageParserService()

    def evaluate(self, dsl_text: str) -> str:
        regions = self._parser_service.extract_statement_regions(dsl_text)
        if regions is None:
            return dsl_text

        ranges = [self._expand_skip_region(dsl_text, region.start, region.end) for region in regions if region.kind == "SKIP"]
        if not ranges:
            return dsl_text

        masked = list(dsl_text)
        for start, end in ranges:
            for index in range(start, end):
                if masked[index] not in {"\n", "\r"}:
                    masked[index] = " "
        return "".join(masked)

    @staticmethod
    def _expand_skip_region(dsl_text: str, start: int, end: int) -> tuple[int, int]:
        if end < len(dsl_text) and dsl_text[end] == ";":
            return start, end + 1
        return start, end


class ExecutionPipelineCoordinator:
    _STAGE_SPLIT = "split blocks"
    _STAGE_MACRO = "macro hook"
    _STAGE_PARSE = "parse"
    _STAGE_VALIDATE = "validate"
    _STAGE_INPUT_OPTION = "resolve inputs"
    _STAGE_INPUT_PREPARE = "pre-processing inputs"
    _STAGE_EVAL_PREPARE = "pre-evaluations"
    _STAGE_PLAN = "plan generation"
    _STAGE_EXECUTE = "execute"
    _STAGE_OUTPUT_OPTION = "resolve outputs"
    _SUPPORTED_SOURCE_OPTION_FIELDS: tuple[str, ...] = ("keep", "drop", "rename", "where", "firstobs", "obs")
    _ROW_LOOP_UNSUPPORTED_FUNCTION_PATTERN = re.compile(r"\b(lag|lead|shift)\s*\(", re.IGNORECASE)

    def __init__(self, executor: DataStepExecutor, macro_hook: MacroHook | None = None) -> None:
        self._executor = executor
        self._macro_hook: MacroHook = macro_hook or NoOpMacroHook()
        self._rewrite_planner = RewritePlanner()

    def execute(self, request: ExecuteRequest) -> ExecuteResponse:
        request_diagnostics = self._validate_request(request)
        if request_diagnostics:
            return ExecuteResponse(diagnostics=self._tag_stage(request_diagnostics, self._STAGE_SPLIT))

        transformed_dsl, macro_diagnostics = self._apply_macro_hook(request.dsl_text)
        if macro_diagnostics:
            return ExecuteResponse(diagnostics=self._tag_stage(macro_diagnostics, self._STAGE_MACRO))

        split_blocks = self._executor._split_data_step_blocks(transformed_dsl)
        if not split_blocks:
            return ExecuteResponse(diagnostics=tuple())

        parser_backend = self._executor._parser_backend_selector.select(self._executor._parser_backend_preference)
        self._executor._last_parser_backend = parser_backend.name

        parsed_blocks: list[tuple[str, Any]] = []
        include_block_location = len(split_blocks) > 1
        for block_index, block_dsl in enumerate(split_blocks, start=1):
            parse_result = parser_backend.parse(ParserExecutionContext(dsl_text=block_dsl))
            if parse_result.has_errors:
                if parser_backend.name != "python":
                    fallback = self._executor._parser_backend_selector.select("python")
                    self._executor._last_parser_backend = fallback.name
                    parse_result = fallback.parse(ParserExecutionContext(dsl_text=block_dsl))
                if parse_result.has_errors:
                    return ExecuteResponse(
                        diagnostics=self._tag_stage(
                            self._executor._with_block_location(
                                diagnostics=parse_result.diagnostics,
                                block_index=block_index,
                                include_block_location=include_block_location,
                            ),
                            self._STAGE_PARSE,
                        )
                    )
            parsed_blocks.append((block_dsl, parse_result.ast.statements))

        required_input_names_by_block: list[set[str]] = []
        for _, ast_statements in parsed_blocks:
            source_statement = next(
                (statement for statement in ast_statements if statement.kind in {"SET", "MERGE"}),
                None,
            )
            names: set[str] = set()
            if source_statement is not None:
                names.update(self._executor._dataset_name_key(ref.name) for ref in source_statement.dataset_refs)
            required_input_names_by_block.append(names)

        future_required_after_block: list[set[str]] = [set() for _ in parsed_blocks]
        needed_future: set[str] = set()
        for index in range(len(parsed_blocks) - 1, -1, -1):
            future_required_after_block[index] = set(needed_future)
            needed_future.update(required_input_names_by_block[index])

        explicit_inputs = request.inputs or {}
        combined_outputs: dict[str, DataSetRef] = {}
        combined_notices: list[Any] = []
        combined_diagnostics: list[Diagnostic] = []
        available_inputs: dict[str, DataSetRef] = dict(self._executor._generated_tables)
        explicit_input_names = {self._executor._dataset_name_key(name) for name in explicit_inputs.keys()}

        for block_index, (block_dsl, ast_statements) in enumerate(parsed_blocks, start=1):
            validation_diagnostics = self._executor._validate_ast_block(
                ast_statements=ast_statements,
                dsl_text=block_dsl,
                explicit_inputs=explicit_inputs,
                available_inputs=available_inputs,
                explicit_output_targets=request.output_targets,
            )
            if validation_diagnostics:
                return ExecuteResponse(
                    diagnostics=self._tag_stage(
                        self._executor._with_block_location(
                            diagnostics=validation_diagnostics,
                            block_index=block_index,
                            include_block_location=include_block_location,
                        ),
                        self._STAGE_VALIDATE,
                    )
                )

            resolved_output_targets, output_target_diagnostics = self._executor._resolve_output_targets(
                ast_statements=ast_statements,
                explicit_output_targets=request.output_targets,
            )
            if output_target_diagnostics:
                return ExecuteResponse(
                    diagnostics=self._tag_stage(
                        self._executor._with_block_location(
                            diagnostics=tuple(output_target_diagnostics),
                            block_index=block_index,
                            include_block_location=include_block_location,
                        ),
                        self._STAGE_OUTPUT_OPTION,
                    )
                )

            resolved_inputs, resolution_diagnostics = self._executor._resolve_inputs(
                ast_statements=ast_statements,
                explicit_inputs=explicit_inputs,
                available_inputs=available_inputs,
            )
            if resolution_diagnostics:
                return ExecuteResponse(
                    diagnostics=self._tag_stage(
                        self._executor._with_block_location(
                            diagnostics=tuple(resolution_diagnostics),
                            block_index=block_index,
                            include_block_location=include_block_location,
                        ),
                        self._STAGE_INPUT_OPTION,
                    )
                )

            prepared_inputs, prepare_diagnostics = self._executor._prepare_runtime_inputs(
                ast_statements=ast_statements,
                resolved_inputs=resolved_inputs,
            )
            if prepare_diagnostics:
                return ExecuteResponse(
                    diagnostics=self._tag_stage(
                        self._executor._with_block_location(
                            diagnostics=tuple(prepare_diagnostics),
                            block_index=block_index,
                            include_block_location=include_block_location,
                        ),
                        self._STAGE_INPUT_PREPARE,
                    )
                )
            resolved_inputs = prepared_inputs

            prepared_statements, prepared_eval_inputs, eval_prepare_diagnostics = self._executor._prepare_runtime_evaluations(
                ast_statements=ast_statements,
                resolved_inputs=resolved_inputs,
            )
            if eval_prepare_diagnostics:
                return ExecuteResponse(
                    diagnostics=self._tag_stage(
                        self._executor._with_block_location(
                            diagnostics=tuple(eval_prepare_diagnostics),
                            block_index=block_index,
                            include_block_location=include_block_location,
                        ),
                        self._STAGE_EVAL_PREPARE,
                    )
                )
            _pre_prep_source = next(
                (s for s in ast_statements if getattr(s, "kind", None) in {"SET", "MERGE"}),
                None,
            )
            _pre_prep_hcp = self._plan_helper_columns(
                ast_statements=ast_statements,
                source_statement=_pre_prep_source,
                resolved_output_targets=resolved_output_targets,
                rewrite_plan=self._rewrite_planner.build_plan(
                    ast_statements=ast_statements,
                    source_statement=_pre_prep_source,
                ).as_dict(),
            )
            if _pre_prep_hcp.requires_row_prepare:
                combined_diagnostics.append(
                    Diagnostic(
                        code="HELPER_COLUMN_FALLBACK",
                        severity="info",
                        message=f"helper-column fallback: {_pre_prep_hcp.fallback_reason}",
                        stage="helper_column_plan",
                    )
                )
            ast_statements = prepared_statements
            resolved_inputs = prepared_eval_inputs
            execution_plan = self._generate_execution_plan(
                request=request,
                ast_statements=ast_statements,
                resolved_inputs=resolved_inputs,
                resolved_output_targets=resolved_output_targets,
            )

            execution_outputs, execution_diagnostics = self._executor._execute_program(
                RuntimeExecutionContext(
                    request=request,
                    ast_statements=ast_statements,
                    resolved_inputs=resolved_inputs,
                    resolved_output_targets=resolved_output_targets,
                    execution_plan=execution_plan,
                    builder_mode=(
                        execution_plan.get("row_loop_plan", {}).get("builder_mode")
                        if isinstance(execution_plan, Mapping)
                        else None
                    ),
                    rewrite_metadata=(
                        execution_plan.get("rewrite_plan")
                        if isinstance(execution_plan, Mapping)
                        else None
                    ),
                    python_limited_mode=BackendDispatchPolicy.requires_python_limited_backend(
                        SimpleNamespace(ast_statements=ast_statements, request=request)
                    ),
                )
            )
            if execution_diagnostics:
                return ExecuteResponse(
                    diagnostics=self._tag_stage(
                        self._executor._with_block_location(
                            diagnostics=tuple(execution_diagnostics),
                            block_index=block_index,
                            include_block_location=include_block_location,
                        ),
                        self._STAGE_EXECUTE,
                    )
                )

            output_stage_outputs, output_stage_diagnostics = self._executor._apply_output_stage_options(
                ast_statements=ast_statements,
                resolved_inputs=resolved_inputs,
                outputs=execution_outputs,
            )
            if output_stage_diagnostics:
                return ExecuteResponse(
                    diagnostics=self._tag_stage(
                        self._executor._with_block_location(
                            diagnostics=tuple(output_stage_diagnostics),
                            block_index=block_index,
                            include_block_location=include_block_location,
                        ),
                        self._STAGE_OUTPUT_OPTION,
                    )
                )
            execution_outputs = output_stage_outputs
            execution_outputs = self._executor._apply_output_metadata(
                ast_statements=ast_statements,
                outputs=execution_outputs,
                execution_plan=execution_plan,
            )

            for name, dataset in execution_outputs.items():
                combined_outputs[name] = dataset
                available_inputs[name] = dataset
                self._executor._generated_tables[name] = dataset

            notices = _collect_compatibility_notices(dsl_text=block_dsl, inputs=resolved_inputs)
            combined_notices.extend(notices)

            future_required = future_required_after_block[block_index - 1]
            future_needs_dictionary_catalog = any(
                required_name in {"DICTIONARY.TABLES", "DICTIONARY.COLUMNS"}
                for required_name in future_required
            )
            if future_needs_dictionary_catalog:
                continue

            if future_required:
                protected_inputs = explicit_input_names.union(future_required)
                available_inputs = {
                    name: dataset
                    for name, dataset in available_inputs.items()
                    if self._executor._dataset_name_key(name) in protected_inputs
                }
            else:
                available_inputs = {
                    name: dataset
                    for name, dataset in available_inputs.items()
                    if self._executor._dataset_name_key(name) in explicit_input_names
                }

        public_outputs = self._coerce_public_execute_outputs(request=request, outputs=combined_outputs)
        outputs_arrow = _LazyArrowOutputs(combined_outputs, self._executor._io_service.dataset_ref_to_arrow_table)
        return ExecuteResponse(
            outputs=public_outputs,
            outputs_arrow=outputs_arrow,
            diagnostics=tuple(combined_diagnostics),
            notices=tuple(combined_notices),
        )

    def _coerce_public_execute_outputs(
        self,
        *,
        request: ExecuteRequest,
        outputs: Mapping[str, DataSetRef],
    ) -> dict[str, DataSetRef]:
        explicit_inputs = getattr(request, "inputs", None) or {}
        if not explicit_inputs:
            return dict(outputs)
        if not all(
            isinstance(dataset_ref, DataSetRef) and dataset_ref.kind.strip().lower() == "memory"
            for dataset_ref in explicit_inputs.values()
        ):
            return dict(outputs)

        public_outputs: dict[str, DataSetRef] = {}
        for target, dataset_ref in outputs.items():
            normalized_kind = dataset_ref.kind.strip().lower()
            if normalized_kind != "arrow_table" or not hasattr(dataset_ref.payload, "to_pylist"):
                public_outputs[target] = dataset_ref
                continue
            try:
                rows = dataset_ref.payload.to_pylist()
            except Exception:
                public_outputs[target] = dataset_ref
                continue
            sparse_rows: list[dict[str, Any]] = []
            for row in rows:
                if isinstance(row, Mapping):
                    sparse_rows.append({key: value for key, value in dict(row).items() if value is not None})
                else:
                    sparse_rows.append(row)
            public_outputs[target] = DataSetRef(
                kind="memory",
                location=dataset_ref.location,
                payload=sparse_rows,
                metadata=dataset_ref.metadata,
            )
        return public_outputs

    def _generate_execution_plan(
        self,
        *,
        request: ExecuteRequest,
        ast_statements: Sequence[Any],
        resolved_inputs: Mapping[str, DataSetRef],
        resolved_output_targets: tuple[str, ...],
    ) -> Any | None:
        del request

        source_statement = next(
            (statement for statement in ast_statements if statement.kind in {"SET", "MERGE"}),
            None,
        )
        rewrite_plan = self._rewrite_planner.build_plan(
            ast_statements=ast_statements,
            source_statement=source_statement,
        ).as_dict()
        planned_ast_statements = (
            self._rewrite_planner.rewrite_lag_lead_statements(ast_statements)
            if rewrite_plan.get("lag_lead_mode") != "none"
            else tuple(ast_statements)
        )
        source_specs: list[dict[str, Any]] = []
        if source_statement is not None:
            dataset_refs = list(getattr(source_statement, "dataset_refs", ()) or ())
            for dataset_ref in dataset_refs:
                options = dataset_ref.options
                source_specs.append(
                    {
                        "source": dataset_ref.name,
                        "keep": list(options.keep_vars),
                        "drop": list(options.drop_vars),
                        "where": options.where_expr,
                        "rename": dict(options.rename_map),
                        "firstobs": options.firstobs,
                        "obs": options.obs,
                    }
                )

        pre_row_filter = self._rewrite_planner.build_pre_row_filter(
            ast_statements=planned_ast_statements,
            is_simple_filter_expression=self._is_simple_filter_expression,
        )

        keep_vars = self._executor._extract_variable_list(ast_statements, "KEEP")
        drop_vars = self._executor._extract_variable_list(ast_statements, "DROP")
        rename_statement = next((statement for statement in ast_statements if statement.kind == "RENAME"), None)
        rename_map = dict(rename_statement.rename_map) if rename_statement is not None else {}

        projection_present = bool(keep_vars or drop_vars or rename_map)
        post_projection = (
            {
                "keep": list(keep_vars),
                "drop": list(drop_vars),
                "rename": rename_map,
            }
            if projection_present
            else None
        )

        row_statements = [
            {"kind": statement.kind, "text": statement.text}
            for statement in planned_ast_statements
            if statement.kind not in {"WHERE", "KEEP", "DROP", "RENAME"}
        ]
        compatibility_path_plan = BackendDispatchPolicy.build_compatibility_path_plan(ast_statements).as_dict()

        source_policy = self._build_policy(
            stage="source_shaping",
            decision="apply",
            reason_code=None,
            details=None,
        )

        slot_registry = self._executor._build_slot_registry(
            ast_statements=ast_statements,
            resolved_inputs=resolved_inputs,
            source_specs=source_specs,
            runtime_helper_columns=tuple(rewrite_plan.get("helper_runtime_columns", ()) or ()),
        )
        helper_column_plan = self._plan_helper_columns(
            ast_statements=ast_statements,
            source_statement=source_statement,
            resolved_output_targets=resolved_output_targets,
            rewrite_plan=rewrite_plan,
        )
        output_handoff_plan = self._plan_output_handoff(
            ast_statements=ast_statements,
            resolved_inputs=resolved_inputs,
            resolved_output_targets=resolved_output_targets,
            slot_registry=slot_registry,
            post_projection=post_projection,
            excluded_output_variables=self._executor._collect_internal_output_variable_names(
                ast_statements=ast_statements,
                resolved_inputs=resolved_inputs,
                additional_internal_names=tuple(rewrite_plan.get("helper_runtime_columns", ()) or ()),
            ),
        )
        row_loop_guard = self._build_row_loop_guard(
            source_statement=source_statement,
            row_statements=row_statements,
            helper_column_plan=helper_column_plan,
            rewrite_plan=rewrite_plan,
        )
        execution_readiness = self._build_execution_readiness(
            source_policy=source_policy,
            pre_row_filter=pre_row_filter,
            row_loop_guard=row_loop_guard,
        )

        return {
            "stage_order": [
                "source_shaping",
                "helper_column_plan",
                "rewrite_plan",
                "pre_row_filter",
                "row_loop_guard",
                "row_execution",
                "post_projection",
                "output_handoff_plan",
            ],
            "source_shaping": {
                "sources": source_specs,
                "capability": {
                    "supported_option_fields": list(self._SUPPORTED_SOURCE_OPTION_FIELDS),
                    "supported": True,
                },
                "policy": source_policy,
            },
            "helper_column_plan": helper_column_plan,
            "rewrite_plan": rewrite_plan,
            "compatibility_path_plan": compatibility_path_plan,
            "pre_row_filter": pre_row_filter,
            "row_loop_guard": row_loop_guard,
            "row_loop_plan": self._build_row_loop_plan(
                helper_column_plan=helper_column_plan,
                pre_row_filter=pre_row_filter,
                output_handoff_plan=output_handoff_plan,
                row_loop_guard=row_loop_guard,
                slot_registry=slot_registry,
                execution_readiness=execution_readiness,
                rewrite_plan=rewrite_plan,
            ),
            "row_statements": row_statements,
            "post_projection": post_projection,
            "output_handoff_plan": output_handoff_plan,
            "slot_registry": slot_registry,
            "execution_readiness": execution_readiness,
        }

    def _build_row_loop_plan(
        self,
        *,
        helper_column_plan: HelperColumnPlan,
        pre_row_filter: Mapping[str, Any] | None,
        output_handoff_plan: OutputHandoffPlan,
        row_loop_guard: Mapping[str, Any],
        slot_registry: Mapping[str, Mapping[str, int]],
        execution_readiness: Mapping[str, Any],
        rewrite_plan: Mapping[str, Any],
    ) -> dict[str, Any]:
        source_slot_order = self._merge_row_loop_slot_order(
            tuple(slot_registry.get("source_slots", {}).keys()),
            helper_column_plan.arrow_columns,
            helper_column_plan.fallback_columns,
        )
        mutable_slot_order = tuple(slot_registry.get("mutable_slots", {}).keys())
        automatic_slot_order = tuple(slot_registry.get("automatic_slots", {}).keys())

        blocking_reason_codes = tuple(execution_readiness.get("blocking_reason_codes", ()) or ())
        guard_reason_codes = tuple(row_loop_guard.get("reason_codes", ()) or ())
        unsupported_reason = next(
            (
                reason
                for reason in (
                    *guard_reason_codes,
                    *blocking_reason_codes,
                    helper_column_plan.fallback_reason,
                    output_handoff_plan.fallback_reason,
                    pre_row_filter.get("policy", {}).get("reason_code") if pre_row_filter is not None else None,
                )
                if reason
            ),
            None,
        )

        return {
            "mode": "arrow_row_loop",
            "engine_mode": "unified_row_loop",
            "backend_mode": "rust_first",
            "where_mode": str(rewrite_plan.get("where_mode", "none")),
            "rewrite_mode": (
                "planner_owned_active"
                if rewrite_plan.get("merge_mode") != "none" or rewrite_plan.get("lag_lead_mode") != "none"
                else "planner_owned_pending"
            ),
            "cursor_kind": "arrow_row_cursor",
            "source_slot_order": source_slot_order,
            "mutable_slot_order": mutable_slot_order,
            "automatic_slot_order": automatic_slot_order,
            "builder_mode": "targeted_output_handoff",
            "output_mode": "targeted_output_handoff",
            "materialization_policy": "arrow_cursor",
            "unsupported_reason": unsupported_reason,
        }

    def _build_row_loop_guard(
        self,
        *,
        source_statement: Any | None,
        row_statements: Sequence[Mapping[str, Any]],
        helper_column_plan: HelperColumnPlan,
        rewrite_plan: Mapping[str, Any],
    ) -> dict[str, Any]:
        reason_codes: list[str] = []
        if helper_column_plan.requires_row_prepare:
            reason_codes.append("ROW_LOOP_UNSUPPORTED_RUNTIME_HELPER_COLUMNS")

        return {
            "decision": "apply" if not reason_codes else "error",
            "reason_codes": tuple(dict.fromkeys(reason_codes)),
        }

    def _merge_row_loop_slot_order(self, *slot_groups: Sequence[str]) -> tuple[str, ...]:
        ordered: list[str] = []
        seen: set[str] = set()
        for slot_group in slot_groups:
            for name in slot_group:
                key = _column_key(name)
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(name)
        return tuple(ordered)

    def _plan_output_handoff(
        self,
        *,
        ast_statements: Sequence[Any],
        resolved_inputs: Mapping[str, DataSetRef],
        resolved_output_targets: tuple[str, ...],
        slot_registry: Mapping[str, Mapping[str, int]],
        post_projection: Mapping[str, Any] | None,
        excluded_output_variables: set[str],
    ) -> OutputHandoffPlan:
        base_columns = self._filter_excluded_output_columns(
            columns=(
                *tuple(slot_registry.get("source_slots", {}).keys()),
                *tuple(slot_registry.get("mutable_slots", {}).keys()),
            ),
            excluded_names=excluded_output_variables,
        )
        if post_projection is None:
            post_projection = {"keep": [], "drop": [], "rename": {}}

        projected_base = self._filter_excluded_output_columns(
            columns=self._project_output_columns(
                columns=base_columns,
                keep=tuple(post_projection.get("keep", ()) or ()),
                drop=tuple(post_projection.get("drop", ()) or ()),
                rename_map=dict(post_projection.get("rename", {}) or {}),
            ),
            excluded_names=excluded_output_variables,
        )
        base_type_expectations = self._filter_excluded_type_expectations(
            type_expectations=self._infer_output_type_expectations(
                resolved_inputs=resolved_inputs,
                late_created_columns=tuple(slot_registry.get("mutable_slots", {}).keys()),
                rename_map=dict(post_projection.get("rename", {}) or {}),
                keep=tuple(post_projection.get("keep", ()) or ()),
                drop=tuple(post_projection.get("drop", ()) or ()),
            ),
            excluded_names=excluded_output_variables,
        )
        target_specs = self._collect_output_target_option_specs(
            ast_statements=ast_statements,
            resolved_output_targets=resolved_output_targets,
        )

        projected_columns_by_target: dict[str, tuple[str, ...]] = {}
        rename_map_by_target: dict[str, dict[str, str]] = {}
        type_expectations_by_target: dict[str, dict[str, str]] = {}
        base_rename_map = dict(post_projection.get("rename", {}) or {})

        for target in resolved_output_targets:
            target_spec = target_specs.get(
                self._executor._dataset_name_key(target),
                {"keep": [], "drop": [], "rename": {}},
            )
            projected_columns_by_target[target] = self._filter_excluded_output_columns(
                columns=self._project_output_columns(
                    columns=projected_base,
                    keep=tuple(target_spec.get("keep", ()) or ()),
                    drop=tuple(target_spec.get("drop", ()) or ()),
                    rename_map=dict(target_spec.get("rename", {}) or {}),
                ),
                excluded_names=excluded_output_variables,
            )
            rename_map_by_target[target] = base_rename_map | dict(target_spec.get("rename", {}) or {})
            type_expectations_by_target[target] = self._filter_excluded_type_expectations(
                type_expectations=self._apply_type_projection(
                    type_expectations=base_type_expectations,
                    keep=tuple(target_spec.get("keep", ()) or ()),
                    drop=tuple(target_spec.get("drop", ()) or ()),
                    rename_map=dict(target_spec.get("rename", {}) or {}),
                ),
                excluded_names=excluded_output_variables,
            )

        return OutputHandoffPlan(
            targets=resolved_output_targets,
            projected_columns_by_target=projected_columns_by_target,
            rename_map_by_target=rename_map_by_target,
            type_expectations_by_target=type_expectations_by_target,
            fallback_reason=None,
        )

    def _collect_output_target_option_specs(
        self,
        *,
        ast_statements: Sequence[Any],
        resolved_output_targets: tuple[str, ...],
    ) -> dict[str, dict[str, Any]]:
        data_statement = next((statement for statement in ast_statements if statement.kind == "DATA"), None)
        if data_statement is None:
            return {}

        refs = getattr(data_statement, "output_refs", ()) or getattr(data_statement, "dataset_refs", ())
        specs: dict[str, dict[str, Any]] = {}
        valid_targets = {self._executor._dataset_name_key(target) for target in resolved_output_targets}
        for ref in refs:
            ref_name = getattr(ref, "name", None)
            if not isinstance(ref_name, str):
                continue
            key = self._executor._dataset_name_key(ref_name)
            if key not in valid_targets:
                continue
            option_spec = getattr(ref, "options", None)
            if option_spec is None:
                continue
            specs[key] = {
                "keep": list(getattr(option_spec, "keep_vars", ()) or ()),
                "drop": list(getattr(option_spec, "drop_vars", ()) or ()),
                "rename": dict(getattr(option_spec, "rename_map", {}) or {}),
            }
        return specs

    def _project_output_columns(
        self,
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
            normalized_rename = {_column_key(source): str(target) for source, target in rename_map.items()}
            ordered = [normalized_rename.get(_column_key(name), name) for name in ordered]
        return tuple(ordered)

    def _filter_excluded_output_columns(self, *, columns: Sequence[str], excluded_names: set[str]) -> tuple[str, ...]:
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

    def _filter_excluded_type_expectations(
        self,
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
            inferred.setdefault(name, "dynamic")
        return self._apply_type_projection(
            type_expectations=inferred,
            keep=keep,
            drop=drop,
            rename_map=rename_map,
        )

    def _apply_type_projection(
        self,
        *,
        type_expectations: Mapping[str, str],
        keep: Sequence[str],
        drop: Sequence[str],
        rename_map: Mapping[str, str],
    ) -> dict[str, str]:
        projected = dict(type_expectations)
        if keep:
            keep_keys = {_column_key(name) for name in keep}
            projected = {name: type_name for name, type_name in projected.items() if _column_key(name) in keep_keys}
        if drop:
            drop_keys = {_column_key(name) for name in drop}
            projected = {name: type_name for name, type_name in projected.items() if _column_key(name) not in drop_keys}
        if rename_map:
            normalized_rename = {_column_key(source): str(target) for source, target in rename_map.items()}
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

        rows, load_error = self._executor._io_service.load_input_rows(dataset_ref)
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

    def _plan_helper_columns(
        self,
        *,
        ast_statements: Sequence[Any],
        source_statement: Any,
        resolved_output_targets: tuple[str, ...],
        rewrite_plan: Mapping[str, Any],
    ) -> HelperColumnPlan:
        arrow_columns: list[str] = list(rewrite_plan.get("helper_source_columns", ()) or ())
        fallback_columns: list[str] = []
        source_aliases: list[str] = []
        multi_output_with_by = (
            len(resolved_output_targets) > 1
            and any(getattr(s, "kind", None) == "BY" for s in ast_statements)
        )

        if source_statement is not None:
            dataset_refs = list(getattr(source_statement, "dataset_refs", ()) or ())
            for dataset_ref in dataset_refs:
                source_aliases.append(dataset_ref.name)

        if multi_output_with_by and not any(col.startswith("first.") or col.startswith("last.") for col in arrow_columns):
            arrow_columns.append("_first_last_routing_")
        requires_row_prepare = False
        fallback_reason: str | None = None

        return HelperColumnPlan(
            arrow_columns=tuple(arrow_columns),
            fallback_columns=tuple(fallback_columns),
            source_aliases=tuple(source_aliases),
            requires_row_prepare=requires_row_prepare,
            fallback_reason=fallback_reason,
        )

    def _is_simple_filter_expression(self, *, expression: str) -> bool:
        try:
            parse_simple_filter("datastep", expression)
            return True
        except Exception:
            return False

    def _build_policy(
        self,
        *,
        stage: str,
        decision: str,
        reason_code: str | None,
        details: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "stage": stage,
            "decision": decision,
            "reason_code": reason_code,
            "details": details,
        }

    def _build_execution_readiness(
        self,
        *,
        source_policy: Mapping[str, Any],
        pre_row_filter: Mapping[str, Any] | None,
        row_loop_guard: Mapping[str, Any],
    ) -> dict[str, Any]:
        decisions = [source_policy]
        if pre_row_filter is not None:
            decisions.append(pre_row_filter["policy"])
        decisions.append(
            {
                "stage": "row_loop_guard",
                "decision": row_loop_guard.get("decision", "apply"),
                "reason_code": (row_loop_guard.get("reason_codes") or (None,))[0],
                "details": None if row_loop_guard.get("decision") == "apply" else {"reasons": list(row_loop_guard.get("reason_codes", ()))},
            }
        )
        blocking = [
            policy
            for policy in decisions
            if policy.get("decision") in {"error", "fallback"}
        ]
        if blocking:
            blocking_reason_codes = [policy.get("reason_code") for policy in blocking if policy.get("reason_code")]
            return {
                "decision": "blocked",
                "policies": decisions,
                "blocking_policies": blocking,
                "blocking_reason_codes": blocking_reason_codes,
            }
        return {
            "decision": "ready",
            "policies": decisions,
            "blocking_policies": [],
            "blocking_reason_codes": [],
        }

    def _validate_request(self, request: ExecuteRequest) -> tuple[Diagnostic, ...]:
        diagnostics: list[Diagnostic] = []

        if not request.dsl_text or not request.dsl_text.strip():
            diagnostics.append(
                Diagnostic(
                    code="REQ_EMPTY_DSL",
                    severity="error",
                    message="dsl_text must not be empty.",
                )
            )

        explicit_inputs = request.inputs or {}

        if any(not name.strip() for name in explicit_inputs):
            diagnostics.append(
                Diagnostic(
                    code="REQ_INVALID_INPUTS",
                    severity="error",
                    message="inputs must contain non-empty dataset names.",
                )
            )

        if request.output_targets and any(not target.strip() for target in request.output_targets):
            diagnostics.append(
                Diagnostic(
                    code="REQ_INVALID_OUTPUT_TARGETS",
                    severity="error",
                    message="output_targets must contain non-empty target names.",
                )
            )

        return tuple(diagnostics)

    def _apply_macro_hook(self, dsl_text: str) -> tuple[str, tuple[Diagnostic, ...]]:
        try:
            return self._macro_hook.evaluate(dsl_text), ()
        except Exception as error:
            return "", (
                Diagnostic(
                    code="MACRO_EVALUATION_FAILED",
                    severity="error",
                    message=f"macro hook failed: {error}",
                ),
            )

    def _tag_stage(self, diagnostics: Sequence[Diagnostic], stage: str) -> tuple[Diagnostic, ...]:
        tagged: list[Diagnostic] = []
        for diagnostic in diagnostics:
            tagged.append(
                Diagnostic(
                    code=diagnostic.code,
                    severity=diagnostic.severity,
                    message=diagnostic.message,
                    location=diagnostic.location,
                    stage=diagnostic.stage or stage,
                    span=diagnostic.span,
                    labels=diagnostic.labels,
                    notes=diagnostic.notes,
                    source_text=diagnostic.source_text,
                )
            )
        return tuple(tagged)
