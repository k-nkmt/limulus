"""Python backend execution engine for the DataStep executor.

This module contains :class:`PythonBackendExecutionService`, which implements the
row-level PDV (Program Data Vector) loop for DataStep programs when the Python
runtime backend is selected.  It is separated from :mod:`limulus.executor` so that
the orchestration layer (``DataStepExecutor``) remains focused on backend selection,
I/O resolution, and multi-block coordination.
"""

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .backends import RuntimeExecutionContext
from .evaluator import ExpressionEvaluator
from .io import ExecutorIOService
from .executor_py_data import _PythonDatasetExecutionMixin
from .executor_py_stage import _PythonInputPreparationService, _PythonProgramExecutionService
from .models import DataSetRef, Diagnostic
from .executor_py_stmt import _PythonStatementExecutionMixin
from .runtime import PDVRuntimeService


class PythonBackendExecutionService(_PythonStatementExecutionMixin, _PythonDatasetExecutionMixin):
    """Python-backend execution engine for a single DataStep program block.

    Handles all row-level PDV loop execution when the Python runtime backend is
    selected.  Receives shared services (``runtime``, ``evaluator``, ``io_service``)
    from the parent :class:`~limulus.executor.DataStepExecutor` at construction time.
    """

    _SUM_STATEMENT = re.compile(r"^\s*([A-Za-z_][\w\.]*)\s*\+\s*(.+)$")
    _ASSIGN_STATEMENT = re.compile(r"^\s*(.+?)\s*=\s*(.+)$")
    _DO_TO_STATEMENT = re.compile(r"^do\s+([A-Za-z_][\w\.]*)\s*=\s*(.+?)\s+to\s+(.+)$", re.IGNORECASE)
    _ARRAY_DECLARATION = re.compile(
        r"^array\s+([A-Za-z_][\w\.]*)\s*(\[[^\]]+\])?\s+(\$\s+)?(.+)$",
        re.IGNORECASE,
    )
    _SIMPLE_WHERE_COMPARISON = re.compile(
        r"^\s*([A-Za-z_][\w\.]*)\s*(>=|<=|!=|=|>|<)\s*(.+?)\s*$",
        re.IGNORECASE,
    )
    _ROW_VIEW_FUNCTION_PATTERN = re.compile(r"\b(lag|lead)\s*\(", re.IGNORECASE)
    _MAX_DO_NESTING = 10
    _PREPARED_SET_ROWS_MARKER = "#prepared_set_rows"
    _PREPARED_MERGE_ROWS_MARKER = "#prepared_merge_rows"

    def __init__(
        self,
        runtime: PDVRuntimeService,
        evaluator: ExpressionEvaluator,
        io_service: ExecutorIOService,
    ) -> None:
        self._runtime = runtime
        self._evaluator = evaluator
        self._io_service = io_service
        self._if_then_do_condition_cache: dict[str, str | None] = {}
        self._if_then_action_cache: dict[tuple[str, str], tuple[str, str] | None] = {}
        self._subset_if_condition_cache: dict[tuple[str, str], str | None] = {}
        self._if_like_statement_cache: dict[tuple[str, str], tuple[str, Mapping[str, Any]] | None] = {}
        self._input_preparation = _PythonInputPreparationService(self)
        self._program_execution = _PythonProgramExecutionService(self)

    # ------------------------------------------------------------------
    # Primary entry point (called by PythonRuntimeBackend)
    # ------------------------------------------------------------------

    def execute(
        self,
        context: RuntimeExecutionContext,
    ) -> tuple[dict[str, DataSetRef], list[Diagnostic]]:
        ast_statements = context.ast_statements
        resolved_inputs = context.resolved_inputs
        resolved_output_targets = context.resolved_output_targets
        prepared, direct_execution, diagnostics = self._input_preparation.prepare(
            ast_statements=ast_statements,
            resolved_inputs=resolved_inputs,
            resolved_output_targets=resolved_output_targets,
        )
        if diagnostics:
            return {}, diagnostics
        if direct_execution is not None:
            return direct_execution
        if prepared is None:
            return {}, diagnostics

        return self._program_execution.execute(
            ast_statements=ast_statements,
            prepared=prepared,
            resolved_output_targets=resolved_output_targets,
        )

    def _execute_program_basic(
        self,
        *,
        ast_statements: Sequence[Any],
        rows: list[dict[str, Any]],
        resolved_output_targets: tuple[str, ...],
        source_statement: Any | None,
        merge_statement: Any | None,
        in_option_vars: Sequence[str],
        where_expression: str,
        drop_vars: tuple[str, ...],
        keep_vars: tuple[str, ...],
        rename_statement: Any,
        excluded_output_variables: set[str],
        output_dataset_options: Mapping[str, Any],
        passthrough_arrow_input: Any | None,
    ) -> tuple[dict[str, DataSetRef], list[Diagnostic]]:
        diagnostics: list[Diagnostic] = []
        if_chain = self._extract_if_chain(ast_statements)
        subset_if_conditions = self._extract_subset_if_conditions(ast_statements)
        has_if_explicit_output = False
        if if_chain is not None:
            has_if_explicit_output = any(
                "__output_target__" in assignments or "__output_default__" in assignments
                for _, assignments in if_chain[0]
            ) or (
                if_chain[1] is not None and (
                    "__output_target__" in if_chain[1] or "__output_default__" in if_chain[1]
                )
            )
        has_unconditional_delete = any(statement.kind == "DELETE" for statement in ast_statements)
        output_statement_targets = self._extract_output_targets(ast_statements)

        passthrough_targets = self._resolve_passthrough_targets(
            resolved_output_targets=resolved_output_targets,
            output_statement_targets=output_statement_targets,
            has_if_explicit_output=has_if_explicit_output,
        )
        can_passthrough = (
            merge_statement is None
            and passthrough_targets is not None
            and not in_option_vars
            and source_statement is not None
            and source_statement.statement_options.indsname_var is None
            and source_statement.statement_options.end_var is None
            and not where_expression
            and not drop_vars
            and not keep_vars
            and rename_statement is None
            and not excluded_output_variables
            and not output_dataset_options
            and if_chain is None
            and not subset_if_conditions
            and not has_unconditional_delete
        )
        if can_passthrough:
            if passthrough_arrow_input is not None:
                return (
                    {
                        target: DataSetRef(kind="arrow_table", location=f"dataset://{target}", payload=passthrough_arrow_input)
                        for target in passthrough_targets
                    },
                    diagnostics,
                )
            return (
                {
                    target: DataSetRef(kind="memory", location=f"dataset://{target}", payload=list(rows))
                    for target in passthrough_targets
                },
                diagnostics,
            )

        context = self._runtime.create_context()
        routed = self._runtime.create_output_buffers(resolved_output_targets)

        for row_index, row in enumerate(rows):
            self._runtime.set_row_view(row_index=row_index, rows=rows)
            self._runtime.begin_row(context)

            try:
                if where_expression and not self._runtime.passes_where(row, context, where_expression):
                    continue
            except self._runtime.RuntimeExecutionError as error:
                diagnostics.append(error.diagnostic)
                return {}, diagnostics

            working_row = dict(row)

            try:
                selected_output_target: str | None = None
                selected_default_output = False
                marked_for_delete = False
                marked_for_stop = False
                for subset_condition in subset_if_conditions:
                    if not self._runtime.passes_where(working_row, context, subset_condition):
                        marked_for_delete = True
                        break
                if if_chain is not None:
                    evaluated = self._runtime.evaluate_if_chain(
                        row=working_row,
                        context=context,
                        branches=if_chain[0],
                        else_assignments=if_chain[1],
                    )
                    selected_output_target = evaluated.pop("__output_target__", None)
                    selected_default_output = bool(evaluated.pop("__output_default__", False))
                    marked_for_delete = bool(evaluated.pop("__delete__", False))
                    marked_for_stop = bool(evaluated.pop("__stop__", False))
                    working_row = evaluated
            except self._runtime.RuntimeExecutionError as error:
                diagnostics.append(error.diagnostic)
                return {}, diagnostics

            if has_unconditional_delete or marked_for_delete:
                continue

            routed_row = self._runtime.apply_drop_keep(working_row, drop_vars=drop_vars, keep_vars=keep_vars)

            if rename_statement is not None:
                renamed_row, rename_error = self._apply_rename_statement(
                    row=routed_row,
                    rename_map=rename_statement.rename_map,
                )
                if rename_error is not None:
                    diagnostics.append(rename_error)
                    return {}, diagnostics
                routed_row = renamed_row

            routed_row = self._exclude_internal_output_variables(
                row=routed_row,
                excluded_names=excluded_output_variables,
            )

            target_candidates: list[str]
            if selected_output_target:
                if selected_output_target == "__default__":
                    target_candidates = [resolved_output_targets[0]]
                else:
                    target_candidates = [selected_output_target]
            elif selected_default_output:
                target_candidates = [resolved_output_targets[0]]
            elif has_if_explicit_output:
                continue
            elif output_statement_targets:
                target_candidates = list(output_statement_targets)
            else:
                target_candidates = [resolved_output_targets[0]]

            for target in target_candidates:
                target = self._resolve_declared_output_target(target, resolved_output_targets)
                projected, option_error = self._apply_output_dataset_options(
                    row=routed_row,
                    target=target,
                    output_dataset_options=output_dataset_options,
                )
                if option_error is not None:
                    diagnostics.append(option_error)
                    return {}, diagnostics
                self._runtime.route_output_record(
                    context=context,
                    row=projected,
                    target=target,
                    routed_outputs=routed,
                    diagnostics=diagnostics,
                )

            if marked_for_stop:
                break

        if diagnostics:
            return {}, diagnostics

        outputs = {
            target: DataSetRef(kind="memory", location=f"dataset://{target}", payload=records)
            for target, records in routed.items()
        }
        return outputs, diagnostics

    # ------------------------------------------------------------------
    # Arrow WHERE push-down (optional optimisation)
    # ------------------------------------------------------------------

    def _maybe_prefilter_rows_with_arrow_where(
        self,
        rows: list[dict[str, Any]],
        where_expression: str,
    ) -> list[dict[str, Any]]:
        if not rows or not where_expression.strip():
            return rows
        if os.getenv("LIMULUS_ENABLE_ARROW_WHERE_PUSHDOWN", "").strip() != "1":
            return rows
        if os.getenv("LIMULUS_DISABLE_ARROW_WHERE_PUSHDOWN", "").strip() == "1":
            return rows

        parsed = self._parse_simple_where_comparison(where_expression)
        if parsed is None:
            return rows
        variable_name, operator, scalar_value = parsed

        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.compute as pc  # type: ignore

            table = pa.Table.from_pylist(rows)
            if variable_name not in table.column_names:
                return rows

            column = table[variable_name]
            scalar = pa.scalar(scalar_value)
            if operator == ">":
                mask = pc.greater(column, scalar)
            elif operator == ">=":
                mask = pc.greater_equal(column, scalar)
            elif operator == "<":
                mask = pc.less(column, scalar)
            elif operator == "<=":
                mask = pc.less_equal(column, scalar)
            elif operator in {"=", "=="}:
                mask = pc.equal(column, scalar)
            else:
                mask = pc.not_equal(column, scalar)

            filtered = table.filter(mask)
            return filtered.to_pylist()
        except Exception:
            return rows

    def _parse_simple_where_comparison(self, expression: str) -> tuple[str, str, Any] | None:
        matched = self._SIMPLE_WHERE_COMPARISON.match(expression.strip())
        if matched is None:
            return None

        variable_name = matched.group(1)
        operator = matched.group(2)
        raw_value = matched.group(3).strip()

        if not raw_value:
            return None
        if any(token in raw_value.lower() for token in (" and ", " or ", "(", ")")):
            return None

        if (raw_value.startswith('"') and raw_value.endswith('"')) or (
            raw_value.startswith("'") and raw_value.endswith("'")
        ):
            return variable_name, operator, raw_value[1:-1]

        try:
            if "." in raw_value:
                return variable_name, operator, float(raw_value)
            return variable_name, operator, int(raw_value)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Advanced runtime (DO loops, ARRAY, SUM, RETAIN, nested IF/DO)
    # ------------------------------------------------------------------

    def _execute_program_advanced(
        self,
        ast_statements: Sequence[Any],
        rows: list[dict[str, Any]],
        resolved_output_targets: tuple[str, ...],
        where_expression: str,
        drop_vars: tuple[str, ...],
        keep_vars: tuple[str, ...],
        rename_statement: Any,
        excluded_output_variables: set[str],
        output_dataset_options: Mapping[str, Any],
    ) -> tuple[dict[str, DataSetRef], list[Diagnostic]]:
        diagnostics: list[Diagnostic] = []
        context = self._runtime.create_context()
        routed = self._runtime.create_output_buffers(resolved_output_targets)
        executable = self._extract_executable_statements(ast_statements)
        has_explicit_output_statement = any(statement.kind == "OUTPUT" for statement in executable)
        if not has_explicit_output_statement:
            for statement in executable:
                if statement.kind != "IF":
                    continue
                parsed_if = self._parse_if_like_statement_from_ast(statement, keyword="if")
                if parsed_if is None:
                    continue
                _, assignments = parsed_if
                if "__output_target__" in assignments or "__output_default__" in assignments:
                    has_explicit_output_statement = True
                    break
        sum_state: dict[str, Any] = {}
        retain_variable_names = set(self._extract_retain_variable_names(ast_statements))
        retain_variable_names.update(self._extract_sum_variable_names(ast_statements))
        retain_state: dict[str, Any] = {name: None for name in retain_variable_names}
        should_copy_source_row = self._uses_row_view_functions(ast_statements)

        for row_index, row in enumerate(rows):
            self._runtime.set_row_view(row_index=row_index, rows=rows)
            self._runtime.begin_row(context)

            try:
                if where_expression and not self._runtime.passes_where(row, context, where_expression):
                    continue
            except self._runtime.RuntimeExecutionError as error:
                diagnostics.append(error.diagnostic)
                return {}, diagnostics

            working_row = dict(row) if should_copy_source_row else row
            for name, value in retain_state.items():
                if name not in working_row:
                    working_row[name] = value
            emitted_rows: list[tuple[str | None, dict[str, Any]]] = []
            array_defs: dict[str, tuple[str, ...]] = {}
            row_deleted, stop_execution, execution_error = self._run_statement_block(
                statements=executable,
                index=0,
                stop_index=len(executable),
                row=working_row,
                context=context,
                emitted_rows=emitted_rows,
                array_defs=array_defs,
                sum_state=sum_state,
            )
            if execution_error is not None:
                diagnostics.append(execution_error)
                return {}, diagnostics

            for name in retain_state.keys():
                if name in working_row:
                    retain_state[name] = working_row[name]

            if not row_deleted and not emitted_rows and not has_explicit_output_statement:
                emitted_rows.append((None, dict(working_row)))

            for selected_output_target, emitted in emitted_rows:
                routed_row = self._runtime.apply_drop_keep(emitted, drop_vars=drop_vars, keep_vars=keep_vars)

                if rename_statement is not None:
                    renamed_row, rename_error = self._apply_rename_statement(
                        row=routed_row,
                        rename_map=rename_statement.rename_map,
                    )
                    if rename_error is not None:
                        diagnostics.append(rename_error)
                        return {}, diagnostics
                    routed_row = renamed_row

                routed_row = self._exclude_internal_output_variables(
                    row=routed_row,
                    excluded_names=excluded_output_variables,
                )

                target = selected_output_target or resolved_output_targets[0]
                target = self._resolve_declared_output_target(target, resolved_output_targets)
                projected, option_error = self._apply_output_dataset_options(
                    row=routed_row,
                    target=target,
                    output_dataset_options=output_dataset_options,
                )
                if option_error is not None:
                    diagnostics.append(option_error)
                    return {}, diagnostics
                self._runtime.route_output_record(
                    context=context,
                    row=projected,
                    target=target,
                    routed_outputs=routed,
                    diagnostics=diagnostics,
                )

            if stop_execution:
                break

        if diagnostics:
            return {}, diagnostics

        outputs = {
            target: DataSetRef(kind="memory", location=f"dataset://{target}", payload=records)
            for target, records in routed.items()
        }
        return outputs, diagnostics

    # ------------------------------------------------------------------
    # Output dataset options (DATA statement per-target options)
    # ------------------------------------------------------------------

    def _extract_prepared_set_rows(
        self,
        *,
        source_statement: Any,
        merge_statement: Any,
        source_refs: Sequence[Any],
        resolved_inputs: Mapping[str, DataSetRef],
    ) -> list[dict[str, Any]] | None:
        if merge_statement is not None:
            return None
        if not source_refs:
            return None

        first_source_name = source_refs[0].name
        first_input = resolved_inputs.get(first_source_name)
        if first_input is None:
            return None
        if self._PREPARED_SET_ROWS_MARKER not in (first_input.location or ""):
            return None
        if first_input.kind.strip().lower() != "memory":
            return None

        payload = first_input.payload
        if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
            return None
        if not all(isinstance(item, Mapping) for item in payload):
            return None

        return [dict(item) for item in payload]

    def _try_execute_direct_single_set_loop(
        self,
        *,
        ast_statements: Sequence[Any],
        source_statement: Any,
        merge_statement: Any,
        source_refs: Sequence[Any],
        by_keys: Sequence[str],
        in_option_vars: Sequence[str],
        internal_variable_names: set[str],
        resolved_inputs: Mapping[str, DataSetRef],
        resolved_output_targets: tuple[str, ...],
    ) -> tuple[dict[str, DataSetRef], list[Diagnostic]] | None:
        if not self._is_direct_single_set_loop_eligible(
            ast_statements=ast_statements,
            source_statement=source_statement,
            merge_statement=merge_statement,
            source_refs=source_refs,
            by_keys=by_keys,
            in_option_vars=in_option_vars,
        ):
            return None

        source_ref = source_refs[0]
        input_name = source_ref.name
        input_ref = resolved_inputs.get(input_name)
        if input_ref is None:
            return (
                {},
                [
                    Diagnostic(
                        code="RUNTIME_SET_DATASET_NOT_FOUND",
                        severity="error",
                        message=f"Input dataset is not provided: {input_name}",
                    )
                ],
            )

        if input_ref.kind.strip().lower() == "memory":
            return None

        row_iterable, iterate_error = self._iterate_input_rows(input_ref)
        if iterate_error is not None:
            return {}, [iterate_error]

        rename_statement = next((statement for statement in ast_statements if statement.kind == "RENAME"), None)
        if rename_statement is not None:
            rename_validation = self._validate_rename_statement(rename_statement.rename_map)
            if rename_validation is not None:
                return {}, [rename_validation]

        excluded_output_variables: set[str] = set()
        where_expression = self._extract_expression(ast_statements, "WHERE")
        drop_vars = self._extract_variable_list(ast_statements, "DROP")
        keep_vars = self._extract_variable_list(ast_statements, "KEEP")
        output_dataset_options = self._collect_output_dataset_options(
            ast_statements=ast_statements,
            declared_targets=resolved_output_targets,
        )
        if_chain = self._extract_if_chain(ast_statements)
        subset_if_conditions = self._extract_subset_if_conditions(ast_statements)
        has_if_explicit_output = False
        if if_chain is not None:
            has_if_explicit_output = any(
                "__output_target__" in assignments or "__output_default__" in assignments
                for _, assignments in if_chain[0]
            ) or (
                if_chain[1] is not None and (
                    "__output_target__" in if_chain[1] or "__output_default__" in if_chain[1]
                )
            )
        has_unconditional_delete = any(statement.kind == "DELETE" for statement in ast_statements)
        output_statement_targets = self._extract_output_targets(ast_statements)

        diagnostics: list[Diagnostic] = []
        context = self._runtime.create_context()
        routed = self._runtime.create_output_buffers(resolved_output_targets)

        for raw_row in row_iterable:
            if not isinstance(raw_row, Mapping):
                diagnostics.append(
                    Diagnostic(
                        code="IO_INPUT_ERROR",
                        severity="error",
                        message="Input row must be a mapping.",
                    )
                )
                return {}, diagnostics

            row = raw_row if isinstance(raw_row, dict) else dict(raw_row)

            option_row, option_error = self._apply_dataset_reference_options_to_row(
                row=row,
                source_name=input_name,
                option_spec=source_ref.options,
            )
            if option_error is not None:
                diagnostics.append(option_error)
                return {}, diagnostics
            if option_row is None:
                continue

            self._runtime.begin_row(context)

            try:
                if where_expression and not self._runtime.passes_where(option_row, context, where_expression):
                    continue
            except self._runtime.RuntimeExecutionError as error:
                diagnostics.append(error.diagnostic)
                return {}, diagnostics

            working_row = option_row

            try:
                selected_output_target: str | None = None
                selected_default_output = False
                marked_for_delete = False
                marked_for_stop = False
                for subset_condition in subset_if_conditions:
                    if not self._runtime.passes_where(working_row, context, subset_condition):
                        marked_for_delete = True
                        break
                if if_chain is not None:
                    evaluated = self._runtime.evaluate_if_chain(
                        row=working_row,
                        context=context,
                        branches=if_chain[0],
                        else_assignments=if_chain[1],
                    )
                    selected_output_target = evaluated.pop("__output_target__", None)
                    selected_default_output = bool(evaluated.pop("__output_default__", False))
                    marked_for_delete = bool(evaluated.pop("__delete__", False))
                    marked_for_stop = bool(evaluated.pop("__stop__", False))
                    working_row = evaluated
            except self._runtime.RuntimeExecutionError as error:
                diagnostics.append(error.diagnostic)
                return {}, diagnostics

            if has_unconditional_delete or marked_for_delete:
                continue

            routed_row = self._runtime.apply_drop_keep(working_row, drop_vars=drop_vars, keep_vars=keep_vars)

            if rename_statement is not None:
                renamed_row, rename_error = self._apply_rename_statement(
                    row=routed_row,
                    rename_map=rename_statement.rename_map,
                )
                if rename_error is not None:
                    diagnostics.append(rename_error)
                    return {}, diagnostics
                routed_row = renamed_row

            routed_row = self._exclude_internal_output_variables(
                row=routed_row,
                excluded_names=excluded_output_variables,
            )

            target_candidates: list[str]
            if selected_output_target:
                if selected_output_target == "__default__":
                    target_candidates = [resolved_output_targets[0]]
                else:
                    target_candidates = [selected_output_target]
            elif selected_default_output:
                target_candidates = [resolved_output_targets[0]]
            elif has_if_explicit_output:
                continue
            elif output_statement_targets:
                target_candidates = list(output_statement_targets)
            else:
                target_candidates = [resolved_output_targets[0]]

            for target in target_candidates:
                target = self._resolve_declared_output_target(target, resolved_output_targets)
                projected, option_error = self._apply_output_dataset_options(
                    row=routed_row,
                    target=target,
                    output_dataset_options=output_dataset_options,
                )
                if option_error is not None:
                    diagnostics.append(option_error)
                    return {}, diagnostics
                self._runtime.route_output_record(
                    context=context,
                    row=projected,
                    target=target,
                    routed_outputs=routed,
                    diagnostics=diagnostics,
                )

            if marked_for_stop:
                break

        if diagnostics:
            return {}, diagnostics

        outputs = {
            target: DataSetRef(kind="memory", location=f"dataset://{target}", payload=records)
            for target, records in routed.items()
        }
        return outputs, diagnostics

    def _is_direct_single_set_loop_eligible(
        self,
        *,
        ast_statements: Sequence[Any],
        source_statement: Any,
        merge_statement: Any,
        source_refs: Sequence[Any],
        by_keys: Sequence[str],
        in_option_vars: Sequence[str],
    ) -> bool:
        if merge_statement is not None:
            return False
        if len(source_refs) != 1:
            return False
        option_spec = getattr(source_refs[0], "options", None)
        if option_spec is not None and (
            getattr(option_spec, "firstobs", None) is not None
            or getattr(option_spec, "obs", None) is not None
        ):
            return False
        if by_keys:
            return False
        if in_option_vars:
            return False
        if source_statement.statement_options.indsname_var is not None:
            return False
        if source_statement.statement_options.end_var is not None:
            return False
        if self._uses_row_view_functions(ast_statements):
            return False

        has_advanced_runtime = any(
            statement.kind in {"SUM", "DO", "ARRAY", "ASSIGN", "RETAIN", "STOP"}
            for statement in ast_statements
        )
        has_advanced_runtime = has_advanced_runtime or any(
            statement.kind == "IF" and self._parse_if_then_do_condition(statement.text) is not None
            for statement in ast_statements
        )

        return not has_advanced_runtime

    def _collect_output_dataset_options(
        self,
        ast_statements: Sequence[Any],
        declared_targets: Sequence[str],
    ) -> dict[str, Any]:
        data_statement = next((statement for statement in ast_statements if statement.kind == "DATA"), None)
        if data_statement is None:
            return {}

        refs = getattr(data_statement, "output_refs", ()) or getattr(data_statement, "dataset_refs", ())
        if not refs:
            return {}

        options_by_target: dict[str, Any] = {}
        for ref in refs:
            option_spec = getattr(ref, "options", None)
            if option_spec is None:
                continue
            if (
                not getattr(option_spec, "keep_vars", ())
                and not getattr(option_spec, "drop_vars", ())
                and not getattr(option_spec, "rename_map", {})
            ):
                continue
            target = self._resolve_declared_output_target(ref.name, declared_targets)
            options_by_target[self._dataset_name_key(target)] = option_spec

        return options_by_target

    def _apply_output_dataset_options(
        self,
        row: Mapping[str, Any],
        target: str,
        output_dataset_options: Mapping[str, Any],
    ) -> tuple[dict[str, Any], Diagnostic | None]:
        option_spec = output_dataset_options.get(self._dataset_name_key(target))
        if option_spec is None:
            return row if isinstance(row, dict) else dict(row), None

        working = dict(row)
        keep_vars = tuple(getattr(option_spec, "keep_vars", ()))
        drop_vars = tuple(getattr(option_spec, "drop_vars", ()))
        rename_map = dict(getattr(option_spec, "rename_map", {}))

        if keep_vars:
            keep_set = set(keep_vars)
            working = {name: value for name, value in working.items() if name in keep_set}

        if drop_vars:
            drop_set = set(drop_vars)
            working = {name: value for name, value in working.items() if name not in drop_set}

        if rename_map:
            if len(set(rename_map.values())) != len(rename_map):
                return {}, Diagnostic(
                    code="RUNTIME_DATASET_OPTION_INVALID",
                    severity="error",
                    message=f"Output dataset option RENAME= has duplicate target names for '{target}'.",
                )
            for old_name in rename_map:
                if old_name not in working:
                    return {}, Diagnostic(
                        code="RUNTIME_DATASET_OPTION_INVALID",
                        severity="error",
                        message=(
                            f"Output dataset option RENAME= references unknown variable '{old_name}' "
                            f"for target '{target}'."
                        ),
                    )
            renamed: dict[str, Any] = {}
            for key, value in working.items():
                renamed[rename_map.get(key, key)] = value
            working = renamed

        return working, None

    def _validate_set_option_variables(self, set_statement: Any) -> Diagnostic | None:
        reserved = {"_N_", "_ERROR_"}

        for source_ref in set_statement.dataset_refs:
            in_var = source_ref.options.in_var
            if in_var and in_var in reserved:
                return Diagnostic(
                    code="RUNTIME_DATASET_OPTION_INVALID",
                    severity="error",
                    message=f"Dataset option IN= cannot use reserved variable name: {in_var}",
                )

        indsname_var = set_statement.statement_options.indsname_var
        if indsname_var and indsname_var in reserved:
            return Diagnostic(
                code="RUNTIME_DATASET_OPTION_INVALID",
                severity="error",
                message=f"SET statement option INDSNAME= cannot use reserved variable name: {indsname_var}",
            )

        end_var = set_statement.statement_options.end_var
        if end_var and end_var in reserved:
            return Diagnostic(
                code="RUNTIME_DATASET_OPTION_INVALID",
                severity="error",
                message=f"SET statement option END= cannot use reserved variable name: {end_var}",
            )

        used: list[str] = []
        for source_ref in set_statement.dataset_refs:
            if source_ref.options.in_var:
                used.append(source_ref.options.in_var)
        if indsname_var:
            used.append(indsname_var)
        if end_var:
            used.append(end_var)

        if len(used) != len(set(used)):
            return Diagnostic(
                code="RUNTIME_DATASET_OPTION_INVALID",
                severity="error",
                message="Dataset/SET option variables must be unique.",
            )

        return None

    def _validate_rename_statement(self, rename_map: Mapping[str, str]) -> Diagnostic | None:
        if len(set(rename_map.values())) != len(rename_map):
            return Diagnostic(
                code="RUNTIME_RENAME_STATEMENT_INVALID",
                severity="error",
                message="RENAME statement has duplicate target variable names.",
            )

        for source, target in rename_map.items():
            if target in rename_map and rename_map.get(target) == source:
                return Diagnostic(
                    code="RUNTIME_RENAME_STATEMENT_INVALID",
                    severity="error",
                    message="RENAME statement contains circular reference.",
                )

        return None

    def _apply_rename_statement(
        self,
        row: Mapping[str, Any],
        rename_map: Mapping[str, str],
    ) -> tuple[dict[str, Any], Diagnostic | None]:
        for old_name in rename_map:
            if old_name not in row:
                return {}, Diagnostic(
                    code="RUNTIME_RENAME_STATEMENT_INVALID",
                    severity="error",
                    message=f"RENAME statement references unknown variable: {old_name}",
                )

        renamed: dict[str, Any] = {}
        for key, value in row.items():
            renamed[rename_map.get(key, key)] = value

        return renamed, None

    # ------------------------------------------------------------------
    # Internal variable tracking
    # ------------------------------------------------------------------

    def _collect_internal_variable_names(
        self,
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
            names.add(f"FIRST.{key}")
            names.add(f"LAST.{key}")
            names.add(f"first.{key}")
            names.add(f"last.{key}")
        return names

    def _extract_prepared_internal_variable_names(
        self,
        *,
        source_refs: Sequence[Any],
        resolved_inputs: Mapping[str, DataSetRef],
    ) -> set[str]:
        if not source_refs:
            return set()

        first_source_name = source_refs[0].name
        first_input = resolved_inputs.get(first_source_name)
        if first_input is None:
            return set()

        location = first_input.location or ""
        if self._PREPARED_MERGE_ROWS_MARKER not in location:
            return set()

        marker = "|internal="
        if marker not in location:
            return set()

        raw = location.split(marker, maxsplit=1)[1]
        if not raw:
            return set()

        return {name for name in raw.split(",") if name}

    def _detect_internal_variable_collision(
        self,
        rows: Sequence[Mapping[str, Any]],
        source_name: str,
        internal_variable_names: set[str],
        allow_internal_names: bool = False,
    ) -> Diagnostic | None:
        if allow_internal_names or not internal_variable_names:
            return None

        for row in rows:
            collided_names = sorted(internal_variable_names.intersection(row.keys()))
            if not collided_names:
                continue
            variable_name = collided_names[0]
            return Diagnostic(
                code="RUNTIME_INTERNAL_VAR_NAME_COLLISION",
                severity="error",
                message=(
                    "Input dataset contains a reserved internal reference variable name: "
                    f"{variable_name} (source={source_name})"
                ),
            )
        return None

    def _build_excluded_output_variables(
        self,
        internal_variable_names: set[str],
        rename_map: Mapping[str, str],
    ) -> set[str]:
        excluded = set(internal_variable_names)
        for source, target in rename_map.items():
            if source in internal_variable_names:
                excluded.add(target)
        return excluded

    def _exclude_internal_output_variables(
        self,
        row: Mapping[str, Any],
        excluded_names: set[str],
    ) -> dict[str, Any]:
        if not excluded_names:
            return row if isinstance(row, dict) else dict(row)
        return {name: value for name, value in row.items() if name not in excluded_names}

    # ------------------------------------------------------------------
    # IF chain / subset IF extraction
    # ------------------------------------------------------------------

    def _extract_if_chain(
        self,
        ast_statements: Sequence[Any],
    ) -> tuple[list[tuple[str, Mapping[str, Any]]], Mapping[str, Any] | None] | None:
        branches: list[tuple[str, Mapping[str, Any]]] = []
        else_assignments: Mapping[str, Any] | None = None

        for statement in ast_statements:
            text = statement.text.strip()
            if statement.kind == "IF":
                parsed = self._parse_if_like_statement_from_ast(statement, keyword="if")
                if parsed is not None:
                    branches.append(parsed)
            elif statement.kind == "ELSE IF":
                parsed = self._parse_if_like_statement_from_ast(statement, keyword="else if")
                if parsed is not None:
                    branches.append(parsed)
            elif statement.kind == "ELSE":
                target = self._parse_else_output_target(text)
                if target is not None:
                    if target == "__default__":
                        else_assignments = {"__output_default__": True}
                    else:
                        else_assignments = {"__output_target__": target}

        if not branches and else_assignments is None:
            return None

        return branches, else_assignments

    def _extract_subset_if_conditions(self, ast_statements: Sequence[Any]) -> tuple[str, ...]:
        conditions: list[str] = []
        for statement in ast_statements:
            if statement.kind != "IF":
                continue
            if_spec = getattr(statement, "if_spec", None)
            if (
                if_spec is not None
                and getattr(if_spec, "is_subset", False)
                and getattr(if_spec, "condition", "")
            ):
                conditions.append(str(if_spec.condition))
                continue
            parsed = self._parse_subset_if_condition(statement.text, keyword="if")
            if parsed is not None:
                conditions.append(parsed)
        return tuple(conditions)

    def _parse_if_like_statement_from_ast(
        self,
        statement: Any,
        keyword: str,
    ) -> tuple[str, Mapping[str, Any]] | None:
        if_spec = getattr(statement, "if_spec", None)
        if if_spec is not None:
            condition = getattr(if_spec, "condition", "")
            action = getattr(if_spec, "then_action", None)
            if isinstance(condition, str) and condition and isinstance(action, str) and action:
                return self._parse_if_condition_and_action(condition, action)
        return self._parse_if_like_statement(statement.text, keyword=keyword)

    def _parse_if_like_statement(
        self,
        text: str,
        keyword: str,
    ) -> tuple[str, Mapping[str, Any]] | None:
        cache_key = (keyword, text)
        if cache_key in self._if_like_statement_cache:
            return self._if_like_statement_cache[cache_key]

        normalized = text.lower()
        prefix = f"{keyword} "
        if not normalized.startswith(prefix):
            self._if_like_statement_cache[cache_key] = None
            return None

        body = text[len(prefix):]
        then_index = body.lower().find(" then ")
        if then_index < 0:
            self._if_like_statement_cache[cache_key] = None
            return None

        raw_expression = body[:then_index].strip()
        action = body[then_index + len(" then ") :].strip()
        parsed = self._parse_if_condition_and_action(raw_expression, action)
        self._if_like_statement_cache[cache_key] = parsed
        return parsed

    def _parse_if_condition_and_action(
        self,
        raw_expression: str,
        action: str,
    ) -> tuple[str, Mapping[str, Any]] | None:
        if not raw_expression or not action:
            return None
        if action.lower() == "delete":
            return self._normalize_expression(raw_expression), {"__delete__": True}
        if action.lower() == "stop":
            return self._normalize_expression(raw_expression), {"__stop__": True}
        if action.lower() == "output":
            return self._normalize_expression(raw_expression), {"__output_default__": True}
        target = self._parse_output_target(action)
        if target is None:
            return None

        return self._normalize_expression(raw_expression), {"__output_target__": target}

    def _parse_else_output_target(self, text: str) -> str | None:
        lowered = text.lower()
        if not lowered.startswith("else "):
            return None
        return self._parse_output_target(text[len("else ") :].strip())

    def _parse_output_target(self, action_text: str) -> str | None:
        tokens = action_text.split()
        if len(tokens) == 1 and tokens[0].lower() == "output":
            return "__default__"
        if len(tokens) >= 2 and tokens[0].lower() == "output":
            return tokens[1]
        return None

    # ------------------------------------------------------------------
    # I/O service delegation wrappers
    # ------------------------------------------------------------------

    def _load_input_rows(self, input_ref: DataSetRef) -> tuple[list[dict[str, Any]], Diagnostic | None]:
        return self._io_service.load_input_rows(input_ref)

    def _iterate_input_rows(self, input_ref: DataSetRef) -> tuple[Iterable[dict[str, Any]], Diagnostic | None]:
        return self._io_service.iterate_input_rows(input_ref)

    def _extract_output_targets(self, ast_statements: Sequence[Any]) -> tuple[str, ...]:
        return self._io_service.extract_output_targets(ast_statements)

    def _resolve_declared_output_target(
        self,
        requested_target: str,
        declared_targets: Sequence[str],
    ) -> str:
        return self._io_service.resolve_declared_output_target(requested_target, declared_targets)

    def _dataset_name_key(self, name: str) -> str:
        return self._io_service.dataset_name_key(name)

    # ------------------------------------------------------------------
    # Utility helpers (also present in DataStepExecutor)
    # ------------------------------------------------------------------

    def _extract_expression(self, ast_statements: Sequence[Any], kind: str) -> str:
        statement = next((item for item in ast_statements if item.kind == kind), None)
        if statement is None:
            return ""
        keyword = kind.lower()
        expression = statement.text[len(keyword):].strip()
        return self._normalize_expression(expression)

    def _extract_variable_list(self, ast_statements: Sequence[Any], kind: str) -> tuple[str, ...]:
        statement = next((item for item in ast_statements if item.kind == kind), None)
        if statement is None:
            return ()
        keyword = kind.lower()
        variables = statement.text[len(keyword):].strip().split()
        return tuple(name for name in variables if name)

    def _resolve_passthrough_targets(
        self,
        resolved_output_targets: tuple[str, ...],
        output_statement_targets: tuple[str, ...],
        has_if_explicit_output: bool,
    ) -> tuple[str, ...] | None:
        if has_if_explicit_output:
            return None

        if output_statement_targets:
            resolved_targets = tuple(
                self._resolve_declared_output_target(target, resolved_output_targets)
                for target in output_statement_targets
            )
            if any(target not in resolved_output_targets for target in resolved_targets):
                return None
            return resolved_targets

        if not resolved_output_targets:
            return None

        return (resolved_output_targets[0],)
