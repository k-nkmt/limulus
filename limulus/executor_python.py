"""Python backend execution engine for the DataStep executor.

This module contains :class:`PythonBackendExecutionService`, which implements the
row-level PDV (Program Data Vector) loop for DataStep programs when the Python
runtime backend is selected.  It is separated from :mod:`limulus.executor` so that
the orchestration layer (``DataStepExecutor``) remains focused on backend selection,
I/O resolution, and multi-block coordination.
"""

import heapq
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .backends import RuntimeExecutionContext
from .evaluator import ExpressionEvaluator
from .io import ExecutorIOService
from .models import DataSetRef, Diagnostic
from .runtime import PDVRuntimeService


class PythonBackendExecutionService:
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
        diagnostics: list[Diagnostic] = []

        set_statement = next((statement for statement in ast_statements if statement.kind == "SET"), None)
        merge_statement = next((statement for statement in ast_statements if statement.kind == "MERGE"), None)

        source_statement = merge_statement or set_statement
        in_option_vars: list[str] = []
        internal_variable_names: set[str] = set()
        rows: list[dict[str, Any]] = []
        by_keys = self._extract_variable_list(ast_statements, "BY")

        if source_statement is None:
            rows = [{}]
        else:
            input_names = [ref.name for ref in source_statement.dataset_refs]
            if not input_names:
                source_tokens = source_statement.text.split()
                input_names = [source_tokens[1]] if len(source_tokens) >= 2 else []

            if not input_names:
                diagnostics.append(
                    Diagnostic(
                        code="RUNTIME_SET_DATASET_NOT_FOUND",
                        severity="error",
                        message="SET statement requires an input dataset name.",
                    )
                )
                return {}, diagnostics

            option_validation = self._validate_set_option_variables(source_statement)
            if option_validation is not None:
                diagnostics.append(option_validation)
                return {}, diagnostics

            in_option_vars = [
                ref.options.in_var
                for ref in source_statement.dataset_refs
                if ref.options.in_var is not None
            ]
            internal_variable_names = self._collect_internal_variable_names(
                in_option_vars=in_option_vars,
                indsname_var=source_statement.statement_options.indsname_var,
                end_var=source_statement.statement_options.end_var,
                by_keys=by_keys,
            )

            if source_statement.dataset_refs:
                source_refs = list(source_statement.dataset_refs)
            else:
                source_refs = [
                    type("_FallbackRef", (), {"name": name, "options": type("_FallbackOptions", (), {"in_var": None, "keep_vars": (), "drop_vars": (), "where_expr": None, "rename_map": {}})()})
                    for name in input_names
                ]

            prepared_internal_names = self._extract_prepared_internal_variable_names(
                source_refs=source_refs,
                resolved_inputs=resolved_inputs,
            )
            if prepared_internal_names:
                internal_variable_names.update(prepared_internal_names)

            prepared_rows = self._extract_prepared_set_rows(
                source_statement=source_statement,
                merge_statement=merge_statement,
                source_refs=source_refs,
                resolved_inputs=resolved_inputs,
            )
            if prepared_rows is not None:
                rows = prepared_rows
            else:
                direct_execution = self._try_execute_direct_single_set_loop(
                    ast_statements=ast_statements,
                    source_statement=source_statement,
                    merge_statement=merge_statement,
                    source_refs=source_refs,
                    by_keys=by_keys,
                    in_option_vars=in_option_vars,
                    internal_variable_names=internal_variable_names,
                    resolved_inputs=resolved_inputs,
                    resolved_output_targets=resolved_output_targets,
                )
                if direct_execution is not None:
                    return direct_execution

                rows_with_source: list[tuple[str, dict[str, Any], str | None]] = []

                if merge_statement is not None:
                    merged_rows, merge_error = self._build_merge_rows(
                        source_refs=source_refs,
                        by_keys=by_keys,
                        resolved_inputs=resolved_inputs,
                        internal_variable_names=internal_variable_names,
                    )
                    if merge_error is not None:
                        diagnostics.append(merge_error)
                        return {}, diagnostics
                    rows_with_source = merged_rows
                else:
                    needs_interleave = bool(by_keys) and len(source_refs) > 1
                    per_source: list[tuple[str, list[dict[str, Any]], str | None]] = []

                    for source_ref in source_refs:
                        input_name = source_ref.name
                        input_ref = resolved_inputs.get(input_name)
                        if input_ref is None:
                            diagnostics.append(
                                Diagnostic(
                                    code="RUNTIME_SET_DATASET_NOT_FOUND",
                                    severity="error",
                                    message=f"Input dataset is not provided: {input_name}",
                                )
                            )
                            return {}, diagnostics

                        loaded_rows, load_error = self._load_input_rows(input_ref)
                        if load_error is not None:
                            diagnostics.append(load_error)
                            return {}, diagnostics

                        allow_internal_names = (
                            isinstance(input_ref.location, str)
                            and self._PREPARED_MERGE_ROWS_MARKER in input_ref.location
                        )
                        collision = self._detect_internal_variable_collision(
                            rows=loaded_rows,
                            source_name=input_name,
                            internal_variable_names=internal_variable_names,
                            allow_internal_names=allow_internal_names,
                        )
                        if collision is not None:
                            diagnostics.append(collision)
                            return {}, diagnostics

                        option_rows, option_error = self._apply_dataset_reference_options(
                            rows=loaded_rows,
                            source_name=input_name,
                            option_spec=source_ref.options,
                        )
                        if option_error is not None:
                            diagnostics.append(option_error)
                            return {}, diagnostics

                        per_source.append((input_name, option_rows, source_ref.options.in_var))

                    if needs_interleave:
                        # Each source is already sorted; k-way merge suffices (O(n log k)).
                        def _by_key(
                            item: tuple[str, dict[str, Any], str | None],
                            _keys: list[str] = by_keys,
                        ) -> tuple:
                            row = item[1]
                            return tuple(
                                (1, 0.0, "") if (v := row.get(k)) is None
                                else (0, float(v), "") if isinstance(v, (int, float))
                                else (0, 0.0, str(v))
                                for k in _keys
                            )
                        rows_with_source = list(heapq.merge(
                            *[((n, r, iv) for r in rows) for n, rows, iv in per_source],
                            key=_by_key,
                        ))
                    else:
                        for name, option_rows, in_var in per_source:
                            for row in option_rows:
                                rows_with_source.append((name, row, in_var))

                for index, (source_name, row, row_in_var) in enumerate(rows_with_source):
                    needs_enrichment = bool(
                        in_option_vars
                        or source_statement.statement_options.indsname_var
                        or source_statement.statement_options.end_var
                    )
                    if not needs_enrichment:
                        rows.append(row)
                        continue

                    enriched = dict(row)
                    for in_var in in_option_vars:
                        if row_in_var is None and in_var in enriched:
                            continue
                        enriched[in_var] = 1 if row_in_var == in_var else 0

                    if source_statement.statement_options.indsname_var:
                        enriched[source_statement.statement_options.indsname_var] = source_name

                    if source_statement.statement_options.end_var:
                        enriched[source_statement.statement_options.end_var] = 1 if index == len(rows_with_source) - 1 else 0

                    rows.append(enriched)

                if merge_statement is None and by_keys:
                    rows.sort(key=lambda r: tuple(
                        (r.get(k) is None, str(r.get(k)) if r.get(k) is not None else "")
                        for k in by_keys
                    ))
                    rows, by_error = self._annotate_by_group_flags(rows=rows, by_keys=by_keys)
                    if by_error is not None:
                        diagnostics.append(by_error)
                        return {}, diagnostics

        rename_statement = next((statement for statement in ast_statements if statement.kind == "RENAME"), None)
        if rename_statement is not None:
            rename_validation = self._validate_rename_statement(rename_statement.rename_map)
            if rename_validation is not None:
                diagnostics.append(rename_validation)
                return {}, diagnostics

        excluded_output_variables: set[str] = set()

        where_expression = self._extract_expression(ast_statements, "WHERE")
        has_advanced_runtime = any(
            statement.kind in {"SUM", "DO", "ARRAY", "ASSIGN", "RETAIN", "STOP"}
            for statement in ast_statements
        )
        has_advanced_runtime = has_advanced_runtime or any(
            statement.kind == "IF" and self._parse_if_then_do_condition(statement.text) is not None
            for statement in ast_statements
        )
        drop_vars = self._extract_variable_list(ast_statements, "DROP")
        keep_vars = self._extract_variable_list(ast_statements, "KEEP")
        rows = self._maybe_prefilter_rows_with_arrow_where(rows=rows, where_expression=where_expression)
        output_dataset_options = self._collect_output_dataset_options(
            ast_statements=ast_statements,
            declared_targets=resolved_output_targets,
        )

        if has_advanced_runtime:
            return self._execute_program_advanced(
                ast_statements=ast_statements,
                rows=rows,
                resolved_output_targets=resolved_output_targets,
                where_expression=where_expression,
                drop_vars=drop_vars,
                keep_vars=keep_vars,
                rename_statement=rename_statement,
                excluded_output_variables=excluded_output_variables,
                output_dataset_options=output_dataset_options,
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

        passthrough_targets = self._resolve_passthrough_targets(
            resolved_output_targets=resolved_output_targets,
            output_statement_targets=output_statement_targets,
            has_if_explicit_output=has_if_explicit_output,
        )
        can_passthrough = (
            merge_statement is None
            and passthrough_targets is not None
            and not in_option_vars
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

    # ------------------------------------------------------------------
    # Statement helpers
    # ------------------------------------------------------------------

    def _extract_executable_statements(self, ast_statements: Sequence[Any]) -> list[Any]:
        return [
            statement
            for statement in ast_statements
            if statement.kind not in {"DATA", "SET", "MERGE", "BY", "WHERE", "DROP", "KEEP", "RENAME", "RUN"}
        ]

    def _uses_row_view_functions(self, ast_statements: Sequence[Any]) -> bool:
        for statement in ast_statements:
            statement_text = getattr(statement, "text", "")
            if isinstance(statement_text, str) and self._ROW_VIEW_FUNCTION_PATTERN.search(statement_text):
                return True
        return False

    def _run_statement_block(
        self,
        statements: Sequence[Any],
        index: int,
        stop_index: int,
        row: dict[str, Any],
        context: Any,
        emitted_rows: list[tuple[str | None, dict[str, Any]]],
        array_defs: dict[str, tuple[str, ...]],
        sum_state: dict[str, Any],
        nesting_level: int = 0,
    ) -> tuple[bool, bool, Diagnostic | None]:
        if nesting_level > self._MAX_DO_NESTING:
            return False, False, Diagnostic(
                code="RUNTIME_LOOP_NESTING_LIMIT_EXCEEDED",
                severity="error",
                message=f"DO block nesting exceeded limit: {self._MAX_DO_NESTING}",
            )

        cursor = index
        while cursor < stop_index:
            statement = statements[cursor]

            if statement.kind == "END":
                return False, False, None

            if statement.kind == "DO":
                end_index = self._find_matching_end(statements, cursor, stop_index)
                if end_index < 0:
                    return False, False, Diagnostic(
                        code="RUNTIME_LOOP_EVALUATION_ERROR",
                        severity="error",
                        message="DO statement is missing matching END.",
                    )

                spec = self._parse_do_to_spec(statement)
                if spec is None:
                    return False, False, Diagnostic(
                        code="RUNTIME_LOOP_EVALUATION_ERROR",
                        severity="error",
                        message=f"Unsupported DO syntax: {statement.text}",
                    )

                loop_var, start_expr, end_expr = spec
                start_value, start_error = self._evaluate_scalar(start_expr, row=row, context=context, array_defs=array_defs)
                if start_error is not None:
                    return False, False, start_error
                end_value, end_error = self._evaluate_scalar(end_expr, row=row, context=context, array_defs=array_defs)
                if end_error is not None:
                    return False, False, end_error

                try:
                    start_int = int(start_value)
                    end_int = int(end_value)
                except Exception:
                    return False, False, Diagnostic(
                        code="RUNTIME_LOOP_EVALUATION_ERROR",
                        severity="error",
                        message=f"DO bounds must be numeric: {statement.text}",
                    )

                for iteration, value in enumerate(range(start_int, end_int + 1), start=1):
                    if iteration > 10000:
                        return False, False, Diagnostic(
                            code="RUNTIME_LOOP_LIMIT_EXCEEDED",
                            severity="error",
                            message="DO TO loop exceeded safety iteration limit.",
                        )
                    row[loop_var] = value
                    deleted, stopped, nested_error = self._run_statement_block(
                        statements=statements,
                        index=cursor + 1,
                        stop_index=end_index,
                        row=row,
                        context=context,
                        emitted_rows=emitted_rows,
                        array_defs=array_defs,
                        sum_state=sum_state,
                        nesting_level=nesting_level + 1,
                    )
                    if nested_error is not None:
                        return False, False, nested_error
                    if stopped:
                        return deleted, True, None
                    if deleted:
                        return True, False, None

                cursor = end_index + 1
                continue

            if statement.kind == "ARRAY":
                declaration = self._parse_array_declaration(statement)
                if declaration is None:
                    return False, False, Diagnostic(
                        code="RUNTIME_ARRAY_INVALID",
                        severity="error",
                        message=f"Invalid ARRAY declaration: {statement.text}",
                    )
                array_name, variables = declaration
                array_defs[array_name] = variables
                cursor += 1
                continue

            if statement.kind == "RETAIN":
                cursor += 1
                continue

            if statement.kind == "SUM":
                sum_error = self._apply_sum_statement(
                    statement.text,
                    row=row,
                    context=context,
                    array_defs=array_defs,
                    sum_state=sum_state,
                )
                if sum_error is None:
                    matched = self._SUM_STATEMENT.match(statement.text)
                    if matched is not None:
                        sum_state[matched.group(1)] = row.get(matched.group(1))
                if sum_error is not None:
                    return False, False, sum_error
                cursor += 1
                continue

            if statement.kind == "ASSIGN":
                assign_error = self._apply_assignment_statement(statement.text, row=row, context=context, array_defs=array_defs)
                if assign_error is not None:
                    return False, False, assign_error
                cursor += 1
                continue

            if statement.kind == "IF":
                if_do_condition = self._parse_if_then_do_condition(statement.text)
                if if_do_condition is not None:
                    end_index = self._find_matching_end_for_if_do(statements, cursor, stop_index)
                    if end_index < 0:
                        return False, False, Diagnostic(
                            code="RUNTIME_LOOP_EVALUATION_ERROR",
                            severity="error",
                            message="IF THEN DO block is missing matching END.",
                        )

                    try:
                        matched = self._runtime.passes_where(row, context, if_do_condition)
                    except self._runtime.RuntimeExecutionError as error:
                        return False, False, error.diagnostic

                    if matched:
                        deleted, stopped, nested_error = self._run_statement_block(
                            statements=statements,
                            index=cursor + 1,
                            stop_index=end_index,
                            row=row,
                            context=context,
                            emitted_rows=emitted_rows,
                            array_defs=array_defs,
                            sum_state=sum_state,
                            nesting_level=nesting_level + 1,
                        )
                        if nested_error is not None:
                            return False, False, nested_error
                        if stopped:
                            return deleted, True, None
                        if deleted:
                            return True, False, None

                    cursor = end_index + 1
                    if cursor < stop_index and statements[cursor].kind == "ELSE":
                        else_text = statements[cursor].text.strip()
                        else_action = else_text[len("else"):].strip() if else_text.lower().startswith("else") else ""

                        if else_action.lower() == "do":
                            else_end_index = self._find_matching_end(statements, cursor, stop_index)
                            if else_end_index < 0:
                                return False, False, Diagnostic(
                                    code="RUNTIME_LOOP_EVALUATION_ERROR",
                                    severity="error",
                                    message="ELSE DO block is missing matching END.",
                                )
                            if not matched:
                                deleted, stopped, nested_error = self._run_statement_block(
                                    statements=statements,
                                    index=cursor + 1,
                                    stop_index=else_end_index,
                                    row=row,
                                    context=context,
                                    emitted_rows=emitted_rows,
                                    array_defs=array_defs,
                                    sum_state=sum_state,
                                    nesting_level=nesting_level + 1,
                                )
                                if nested_error is not None:
                                    return False, False, nested_error
                                if stopped:
                                    return deleted, True, None
                                if deleted:
                                    return True, False, None
                            cursor = else_end_index + 1
                            continue

                        if not matched and else_action:
                            deleted, action_error = self._execute_inline_action(
                                action_text=else_action,
                                row=row,
                                context=context,
                                emitted_rows=emitted_rows,
                                array_defs=array_defs,
                                sum_state=sum_state,
                            )
                            if action_error is not None:
                                return False, False, action_error
                            if deleted:
                                return True, False, None
                        cursor += 1
                        continue

                    continue

                parsed_if_action = self._parse_if_then_action(statement.text, keyword="if")
                if parsed_if_action is not None:
                    condition, action_text = parsed_if_action
                    try:
                        matched = self._runtime.passes_where(row, context, condition)
                    except self._runtime.RuntimeExecutionError as error:
                        return False, False, error.diagnostic

                    consumed_else = False
                    if matched:
                        deleted, action_error = self._execute_inline_action(
                            action_text=action_text,
                            row=row,
                            context=context,
                            emitted_rows=emitted_rows,
                            array_defs=array_defs,
                            sum_state=sum_state,
                        )
                        if action_error is not None:
                            return False, False, action_error
                        if row.pop("__stop__", False):
                            return False, True, None
                        if deleted:
                            return True, False, None
                        if cursor + 1 < stop_index and statements[cursor + 1].kind == "ELSE":
                            consumed_else = True
                    elif cursor + 1 < stop_index and statements[cursor + 1].kind == "ELSE":
                        else_text = statements[cursor + 1].text.strip()
                        else_action = else_text[len("else"):].strip()
                        deleted, action_error = self._execute_inline_action(
                            action_text=else_action,
                            row=row,
                            context=context,
                            emitted_rows=emitted_rows,
                            array_defs=array_defs,
                            sum_state=sum_state,
                        )
                        if action_error is not None:
                            return False, False, action_error
                        if row.pop("__stop__", False):
                            return False, True, None
                        if deleted:
                            return True, False, None
                        consumed_else = True

                    cursor += 2 if consumed_else else 1
                    continue

                subset_if_condition = self._parse_subset_if_condition(statement.text, keyword="if")
                if subset_if_condition is not None:
                    try:
                        matched = self._runtime.passes_where(row, context, subset_if_condition)
                    except self._runtime.RuntimeExecutionError as error:
                        return False, False, error.diagnostic
                    if not matched:
                        return True, False, None
                    cursor += 1
                    continue
                cursor += 1
                continue

            if statement.kind == "DELETE":
                return True, False, None

            if statement.kind == "STOP":
                return False, True, None

            if statement.kind == "OUTPUT":
                tokens = statement.text.split()
                emitted_rows.append((tokens[1] if len(tokens) > 1 else None, dict(row)))
                cursor += 1
                continue

            cursor += 1

        return False, False, None

    def _parse_if_then_action(self, text: str, keyword: str = "if") -> tuple[str, str] | None:
        cache_key = (keyword, text)
        cached = self._if_then_action_cache.get(cache_key)
        if cache_key in self._if_then_action_cache:
            return cached

        normalized = text.lower()
        prefix = f"{keyword} "
        if not normalized.startswith(prefix):
            self._if_then_action_cache[cache_key] = None
            return None

        body = text[len(prefix):]
        then_index = body.lower().find(" then ")
        if then_index < 0:
            self._if_then_action_cache[cache_key] = None
            return None

        raw_expression = body[:then_index].strip()
        action = body[then_index + len(" then "):].strip()
        if not action:
            self._if_then_action_cache[cache_key] = None
            return None
        parsed = (self._normalize_expression(raw_expression), action)
        self._if_then_action_cache[cache_key] = parsed
        return parsed

    def _parse_subset_if_condition(self, text: str, keyword: str = "if") -> str | None:
        cache_key = (keyword, text)
        if cache_key in self._subset_if_condition_cache:
            return self._subset_if_condition_cache[cache_key]

        normalized = text.lower()
        prefix = f"{keyword} "
        if not normalized.startswith(prefix):
            self._subset_if_condition_cache[cache_key] = None
            return None

        body = text[len(prefix):].strip()
        if not body or " then " in body.lower():
            self._subset_if_condition_cache[cache_key] = None
            return None
        parsed = self._normalize_expression(body)
        self._subset_if_condition_cache[cache_key] = parsed
        return parsed

    def _execute_inline_action(
        self,
        action_text: str,
        row: dict[str, Any],
        context: Any,
        emitted_rows: list[tuple[str | None, dict[str, Any]]],
        array_defs: Mapping[str, tuple[str, ...]],
        sum_state: dict[str, Any],
    ) -> tuple[bool, Diagnostic | None]:
        normalized = action_text.strip()
        if not normalized:
            return False, None

        lowered = normalized.lower()
        if lowered == "delete":
            return True, None
        if lowered == "stop":
            row["__stop__"] = True
            return False, None

        target = self._parse_output_target(normalized)
        if target is not None:
            if target == "__default__":
                emitted_rows.append((None, dict(row)))
            else:
                emitted_rows.append((target, dict(row)))
            return False, None

        if self._SUM_STATEMENT.match(normalized):
            sum_error = self._apply_sum_statement(
                normalized,
                row=row,
                context=context,
                array_defs=array_defs,
                sum_state=sum_state,
            )
            if sum_error is None:
                matched = self._SUM_STATEMENT.match(normalized)
                if matched is not None:
                    sum_state[matched.group(1)] = row.get(matched.group(1))
            return False, sum_error

        if self._ASSIGN_STATEMENT.match(normalized):
            assign_error = self._apply_assignment_statement(normalized, row=row, context=context, array_defs=array_defs)
            return False, assign_error

        return False, Diagnostic(
            code="RUNTIME_EXPRESSION_EVALUATION_ERROR",
            severity="error",
            message=f"Unsupported IF/ELSE action syntax: {action_text}",
        )

    def _find_matching_end(self, statements: Sequence[Any], do_index: int, stop_index: int) -> int:
        def opens_do_block(statement: Any) -> bool:
            if statement.kind == "DO":
                return True
            if statement.kind == "IF":
                return self._parse_if_then_do_condition(statement.text) is not None
            if statement.kind == "ELSE":
                else_text = statement.text.strip()
                else_action = else_text[len("else") :].strip() if else_text.lower().startswith("else") else ""
                return else_action.lower() == "do"
            return False

        depth = 0
        for cursor in range(do_index, stop_index):
            current = statements[cursor]
            if opens_do_block(current):
                depth += 1
            elif current.kind == "END":
                depth -= 1
                if depth == 0:
                    return cursor
        return -1

    def _find_matching_end_for_if_do(self, statements: Sequence[Any], if_index: int, stop_index: int) -> int:
        depth = 1
        for cursor in range(if_index + 1, stop_index):
            current = statements[cursor]
            if current.kind == "DO":
                depth += 1
            elif current.kind == "IF" and self._parse_if_then_do_condition(current.text) is not None:
                depth += 1
            elif current.kind == "END":
                depth -= 1
                if depth == 0:
                    return cursor
        return -1

    def _parse_do_to_spec(self, statement: Any) -> tuple[str, str, str] | None:
        do_spec = getattr(statement, "do_spec", None)
        if do_spec is not None:
            return (
                str(getattr(do_spec, "loop_var", "")).strip(),
                str(getattr(do_spec, "start_expr", "")).strip(),
                str(getattr(do_spec, "end_expr", "")).strip(),
            )

        do_text = getattr(statement, "text", statement)
        if not isinstance(do_text, str):
            return None

        matched = self._DO_TO_STATEMENT.match(do_text.strip())
        if matched is None:
            return None
        return matched.group(1), matched.group(2).strip(), matched.group(3).strip()

    def _parse_if_then_do_condition(self, if_text: str) -> str | None:
        if if_text in self._if_then_do_condition_cache:
            return self._if_then_do_condition_cache[if_text]

        stripped = if_text.strip()
        lowered = stripped.lower()
        if not lowered.startswith("if "):
            self._if_then_do_condition_cache[if_text] = None
            return None
        then_index = lowered.find(" then ")
        if then_index < 0:
            self._if_then_do_condition_cache[if_text] = None
            return None
        action = stripped[then_index + len(" then ") :].strip().lower()
        if action != "do":
            self._if_then_do_condition_cache[if_text] = None
            return None
        condition = stripped[len("if ") :then_index].strip()
        parsed = self._normalize_expression(condition)
        self._if_then_do_condition_cache[if_text] = parsed
        return parsed

    def _parse_array_declaration(self, statement: Any) -> tuple[str, tuple[str, ...]] | None:
        array_spec = getattr(statement, "array_spec", None)
        if array_spec is not None:
            variables = tuple(str(token) for token in getattr(array_spec, "variables", ()) if str(token))
            if not variables:
                return None
            return str(getattr(array_spec, "array_name", "")).strip(), variables

        array_text = getattr(statement, "text", statement)
        if not isinstance(array_text, str):
            return None

        matched = self._ARRAY_DECLARATION.match(array_text.strip())
        if matched is None:
            return None
        dimension_token = matched.group(2).strip() if matched.group(2) else None
        tokens = [token for token in matched.group(4).split() if token]
        if dimension_token is None and tokens and (
            re.match(r"^\[\s*\*\s*\]$", tokens[0])
            or re.match(r"^\[\d+\]$", tokens[0])
            or re.match(r"^\d+$", tokens[0])
        ):
            tokens = tokens[1:]
        variables = tuple(tokens)
        if not variables:
            return None
        return matched.group(1), variables

    # ------------------------------------------------------------------
    # RETAIN / SUM helpers
    # ------------------------------------------------------------------

    def _extract_retain_variable_names(self, ast_statements: Sequence[Any]) -> tuple[str, ...]:
        names: list[str] = []
        for statement in ast_statements:
            if statement.kind != "RETAIN":
                continue
            body = statement.text[len("retain"):].strip()
            names.extend(token for token in body.split() if token)
        return tuple(dict.fromkeys(names))

    def _extract_sum_variable_names(self, ast_statements: Sequence[Any]) -> tuple[str, ...]:
        names: list[str] = []
        for statement in ast_statements:
            if statement.kind != "SUM":
                continue
            matched = self._SUM_STATEMENT.match(statement.text)
            if matched is None:
                continue
            names.append(matched.group(1))
        return tuple(dict.fromkeys(names))

    def _apply_sum_statement(
        self,
        statement_text: str,
        row: dict[str, Any],
        context: Any,
        array_defs: Mapping[str, tuple[str, ...]],
        sum_state: Mapping[str, Any] | None = None,
    ) -> Diagnostic | None:
        matched = self._SUM_STATEMENT.match(statement_text)
        if matched is None:
            return Diagnostic(
                code="RUNTIME_SUM_STATEMENT_INVALID",
                severity="error",
                message=f"Invalid sum statement: {statement_text}",
            )

        variable_name = matched.group(1)
        expression = matched.group(2)
        add_value, add_error = self._evaluate_scalar(expression, row=row, context=context, array_defs=array_defs)
        if add_error is not None:
            return add_error

        current = row.get(variable_name)
        if self._is_missing_value(current) and sum_state is not None and variable_name in sum_state:
            current = sum_state.get(variable_name)
        current_numeric = 0.0 if self._is_missing_value(current) else float(current)
        add_numeric = 0.0 if self._is_missing_value(add_value) else float(add_value)
        summed = current_numeric + add_numeric
        row[variable_name] = int(summed) if float(summed).is_integer() else summed
        return None

    def _apply_assignment_statement(
        self,
        statement_text: str,
        row: dict[str, Any],
        context: Any,
        array_defs: Mapping[str, tuple[str, ...]],
    ) -> Diagnostic | None:
        matched = self._ASSIGN_STATEMENT.match(statement_text)
        if matched is None:
            return Diagnostic(
                code="RUNTIME_EXPRESSION_EVALUATION_ERROR",
                severity="error",
                message=f"Invalid assignment statement: {statement_text}",
            )

        left = matched.group(1).strip()
        right = matched.group(2).strip()
        right_value, right_error = self._evaluate_scalar(right, row=row, context=context, array_defs=array_defs)
        if right_error is not None:
            return right_error

        array_ref = self._parse_array_reference(left)
        if array_ref is None:
            row[left] = right_value
            return None

        array_name, index_expr = array_ref
        index_value, index_error = self._evaluate_scalar(index_expr, row=row, context=context, array_defs=array_defs)
        if index_error is not None:
            return index_error
        variable_name, variable_error = self._resolve_array_variable(array_name, index_value, array_defs)
        if variable_error is not None:
            return variable_error
        row[variable_name] = right_value
        return None

    # ------------------------------------------------------------------
    # Evaluator delegation wrappers
    # ------------------------------------------------------------------

    def _evaluate_scalar(
        self,
        expression: str,
        row: Mapping[str, Any],
        context: Any,
        array_defs: Mapping[str, tuple[str, ...]],
    ) -> tuple[Any, Diagnostic | None]:
        return self._evaluator.evaluate_scalar(expression, row=row, context=context, array_defs=array_defs)

    def _transform_array_expression(self, expression: str, array_defs: Mapping[str, tuple[str, ...]]) -> str:
        return self._evaluator.transform_array_expression(expression, array_defs=array_defs)

    def _transform_dotted_variable_expression(self, expression: str, row: Mapping[str, Any]) -> str:
        return self._evaluator.transform_dotted_variable_expression(expression, row=row)

    def _parse_array_reference(self, token: str) -> tuple[str, str] | None:
        return self._evaluator.parse_array_reference(token)

    def _resolve_array_variable(
        self,
        array_name: str,
        index_value: Any,
        array_defs: Mapping[str, tuple[str, ...]],
    ) -> tuple[str, Diagnostic | None]:
        return self._evaluator.resolve_array_variable(array_name, index_value, array_defs)

    def _is_missing_value(self, value: Any) -> bool:
        return self._evaluator.is_missing_value(value)

    def _normalize_expression(self, expression: str) -> str:
        return self._evaluator.normalize_expression(expression)

    def _normalize_concat_operator(self, expression: str) -> str:
        return self._evaluator.normalize_concat_operator(expression)

    # ------------------------------------------------------------------
    # MERGE / BY helpers
    # ------------------------------------------------------------------

    def _build_merge_rows(
        self,
        source_refs: Sequence[Any],
        by_keys: tuple[str, ...],
        resolved_inputs: Mapping[str, DataSetRef],
        internal_variable_names: set[str],
    ) -> tuple[list[tuple[str, dict[str, Any], str | None]], Diagnostic | None]:
        loaded_sources: list[tuple[str, str | None, list[dict[str, Any]]]] = []
        non_key_columns_by_source: list[set[str]] = []

        for source_ref in source_refs:
            input_name = source_ref.name
            input_ref = resolved_inputs.get(input_name)
            if input_ref is None:
                return [], Diagnostic(
                    code="RUNTIME_SET_DATASET_NOT_FOUND",
                    severity="error",
                    message=f"Input dataset is not provided: {input_name}",
                )

            loaded_rows, load_error = self._load_input_rows(input_ref)
            if load_error is not None:
                return [], load_error

            allow_internal_names = (
                isinstance(input_ref.location, str)
                and self._PREPARED_MERGE_ROWS_MARKER in input_ref.location
            )
            collision = self._detect_internal_variable_collision(
                rows=loaded_rows,
                source_name=input_name,
                internal_variable_names=internal_variable_names,
                allow_internal_names=allow_internal_names,
            )
            if collision is not None:
                return [], collision

            option_rows, option_error = self._apply_dataset_reference_options(
                rows=loaded_rows,
                source_name=input_name,
                option_spec=source_ref.options,
            )
            if option_error is not None:
                return [], option_error

            for row in option_rows:
                for by_key in by_keys:
                    if by_key not in row:
                        return [], Diagnostic(
                            code="RUNTIME_BY_PRECONDITION_FAILED",
                            severity="error",
                            message=(
                                f"BY key '{by_key}' is missing in source '{input_name}'."
                            ),
                        )

            non_key_columns = {
                column_name
                for row in option_rows
                for column_name in row.keys()
                if column_name not in by_keys
            }
            non_key_columns_by_source.append(non_key_columns)
            loaded_sources.append((input_name, source_ref.options.in_var, option_rows))

        duplicate_columns: set[str] = set()
        for left_index in range(len(non_key_columns_by_source)):
            for right_index in range(left_index + 1, len(non_key_columns_by_source)):
                duplicate_columns.update(
                    non_key_columns_by_source[left_index].intersection(non_key_columns_by_source[right_index])
                )

        if duplicate_columns:
            duplicate_label = ", ".join(sorted(duplicate_columns))
            return [], Diagnostic(
                code="RUNTIME_MERGE_DUPLICATE_COLUMN",
                severity="error",
                message=f"MERGE inputs contain duplicate non-BY columns: {duplicate_label}",
            )

        # BY-less MERGE: positional zip join (no BY statement)
        if not by_keys:
            all_source_rows = [rows for _, _, rows in loaded_sources]
            max_len = max((len(r) for r in all_source_rows), default=0)
            merged_rows_no_by: list[dict[str, Any]] = []
            merged_sources_no_by: list[str] = []
            merged_in_vars_no_by: list[str | None] = []
            for row_index in range(max_len):
                merged_row: dict[str, Any] = {}
                contributing: list[str] = []
                for source_name, in_var, rows in loaded_sources:
                    if row_index < len(rows):
                        merged_row.update(rows[row_index])
                        contributing.append(source_name)
                merged_rows_no_by.append(merged_row)
                merged_sources_no_by.append(",".join(contributing) if contributing else "")
                merged_in_vars_no_by.append(None)
                for source_name, in_var, _ in loaded_sources:
                    if in_var:
                        merged_row[in_var] = 1 if source_name in contributing else 0
            return list(zip(merged_sources_no_by, merged_rows_no_by, merged_in_vars_no_by)), None

        grouped_sources: list[dict[tuple[Any, ...], list[dict[str, Any]]]] = []
        merge_key_order: list[tuple[Any, ...]] = []

        for _, _, rows in loaded_sources:
            grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
            for row in rows:
                key = tuple(row[key_name] for key_name in by_keys)
                grouped.setdefault(key, []).append(row)
                if key not in merge_key_order:
                    merge_key_order.append(key)
            grouped_sources.append(grouped)

        merged_rows: list[dict[str, Any]] = []
        merged_sources: list[str] = []
        merged_in_vars: list[str | None] = []

        for key in merge_key_order:
            group_lengths = [len(grouped.get(key, [])) for grouped in grouped_sources]
            max_group_length = max(group_lengths) if group_lengths else 0

            for row_index in range(max_group_length):
                merged_row: dict[str, Any] = {}
                contributing_sources: list[str] = []
                contributing_in_vars: list[str | None] = []

                for source_index, (source_name, in_var, _) in enumerate(loaded_sources):
                    grouped_rows = grouped_sources[source_index].get(key, [])
                    if row_index < len(grouped_rows):
                        merged_row.update(grouped_rows[row_index])
                        contributing_sources.append(source_name)
                        contributing_in_vars.append(in_var)
                    elif grouped_rows:
                        # Carry forward last row values (PDV retain semantics for 1-to-many)
                        merged_row.update(grouped_rows[-1])
                        contributing_sources.append(source_name)
                        contributing_in_vars.append(in_var)

                for by_index, by_key in enumerate(by_keys):
                    merged_row[by_key] = key[by_index]

                merged_rows.append(merged_row)
                merged_sources.append(",".join(contributing_sources) if contributing_sources else "")
                merged_in_vars.append(None)

                for source_name, in_var, _ in loaded_sources:
                    if in_var:
                        merged_row[in_var] = 1 if source_name in contributing_sources else 0

        merged_rows, by_error = self._annotate_by_group_flags(rows=merged_rows, by_keys=by_keys)
        if by_error is not None:
            return [], by_error

        return list(zip(merged_sources, merged_rows, merged_in_vars)), None

    def _annotate_by_group_flags(
        self,
        rows: list[dict[str, Any]],
        by_keys: Sequence[str],
    ) -> tuple[list[dict[str, Any]], Diagnostic | None]:
        if not rows or not by_keys:
            return rows, None

        for by_key in by_keys:
            if any(by_key not in row for row in rows):
                return [], Diagnostic(
                    code="RUNTIME_BY_PRECONDITION_FAILED",
                    severity="error",
                    message=f"BY key '{by_key}' is missing in source rows.",
                )

        for by_key in by_keys:
            for index, row in enumerate(rows):
                previous_value = rows[index - 1].get(by_key) if index > 0 else object()
                next_value = rows[index + 1].get(by_key) if index < len(rows) - 1 else object()
                current_value = row.get(by_key)
                row[f"FIRST.{by_key}"] = 1 if current_value != previous_value else 0
                row[f"LAST.{by_key}"] = 1 if current_value != next_value else 0
                row[f"first.{by_key}"] = row[f"FIRST.{by_key}"]
                row[f"last.{by_key}"] = row[f"LAST.{by_key}"]

        return rows, None

    # ------------------------------------------------------------------
    # Dataset reference options (SET / MERGE source options)
    # ------------------------------------------------------------------

    def _apply_dataset_reference_options(
        self,
        rows: list[dict[str, Any]],
        source_name: str,
        option_spec: Any,
    ) -> tuple[list[dict[str, Any]], Diagnostic | None]:
        if (
            not option_spec.keep_vars
            and not option_spec.drop_vars
            and not option_spec.rename_map
            and not option_spec.where_expr
        ):
            return list(rows), None

        processed: list[dict[str, Any]] = []

        if option_spec.rename_map and len(set(option_spec.rename_map.values())) != len(option_spec.rename_map):
            return [], Diagnostic(
                code="RUNTIME_DATASET_OPTION_INVALID",
                severity="error",
                message=f"Dataset option RENAME= has duplicate target names for source '{source_name}'.",
            )

        for row in rows:
            working = dict(row)

            if option_spec.keep_vars:
                keep_set = set(option_spec.keep_vars)
                working = {name: value for name, value in working.items() if name in keep_set}

            if option_spec.drop_vars:
                drop_set = set(option_spec.drop_vars)
                working = {name: value for name, value in working.items() if name not in drop_set}

            if option_spec.rename_map:
                renamed_row: dict[str, Any] = {}
                for key, value in working.items():
                    renamed_row[option_spec.rename_map.get(key, key)] = value
                for old_name in option_spec.rename_map:
                    if old_name not in working:
                        return [], Diagnostic(
                            code="RUNTIME_DATASET_OPTION_INVALID",
                            severity="error",
                            message=(
                                f"Dataset option RENAME= references unknown variable '{old_name}' "
                                f"for source '{source_name}'."
                            ),
                        )
                working = renamed_row

            if option_spec.where_expr:
                try:
                    passes = bool(eval(option_spec.where_expr, {"__builtins__": {}}, dict(working)))
                except Exception as error:
                    return [], Diagnostic(
                        code="RUNTIME_DATASET_OPTION_INVALID",
                        severity="error",
                        message=(
                            f"Dataset option WHERE= evaluation failed for source '{source_name}': {error}"
                        ),
                    )
                if not passes:
                    continue

            processed.append(working)

        return processed, None

    def _apply_dataset_reference_options_to_row(
        self,
        row: Mapping[str, Any],
        source_name: str,
        option_spec: Any,
    ) -> tuple[dict[str, Any] | None, Diagnostic | None]:
        if (
            not option_spec.keep_vars
            and not option_spec.drop_vars
            and not option_spec.rename_map
            and not option_spec.where_expr
        ):
            return row if isinstance(row, dict) else dict(row), None

        if option_spec.rename_map and len(set(option_spec.rename_map.values())) != len(option_spec.rename_map):
            return None, Diagnostic(
                code="RUNTIME_DATASET_OPTION_INVALID",
                severity="error",
                message=f"Dataset option RENAME= has duplicate target names for source '{source_name}'.",
            )

        working = dict(row)

        if option_spec.keep_vars:
            keep_set = set(option_spec.keep_vars)
            working = {name: value for name, value in working.items() if name in keep_set}

        if option_spec.drop_vars:
            drop_set = set(option_spec.drop_vars)
            working = {name: value for name, value in working.items() if name not in drop_set}

        if option_spec.rename_map:
            renamed_row: dict[str, Any] = {}
            for key, value in working.items():
                renamed_row[option_spec.rename_map.get(key, key)] = value
            for old_name in option_spec.rename_map:
                if old_name not in working:
                    return None, Diagnostic(
                        code="RUNTIME_DATASET_OPTION_INVALID",
                        severity="error",
                        message=(
                            f"Dataset option RENAME= references unknown variable '{old_name}' "
                            f"for source '{source_name}'."
                        ),
                    )
            working = renamed_row

        if option_spec.where_expr:
            try:
                passes = bool(eval(option_spec.where_expr, {"__builtins__": {}}, dict(working)))
            except Exception as error:
                return None, Diagnostic(
                    code="RUNTIME_DATASET_OPTION_INVALID",
                    severity="error",
                    message=(
                        f"Dataset option WHERE= evaluation failed for source '{source_name}': {error}"
                    ),
                )
            if not passes:
                return None, None

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
