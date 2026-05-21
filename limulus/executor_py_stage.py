from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .models import DataSetRef, Diagnostic


@dataclass(frozen=True)
class _PreparedProgramInputs:
    rows: list[dict[str, Any]]
    source_statement: Any | None
    merge_statement: Any | None
    in_option_vars: tuple[str, ...]
    internal_variable_names: frozenset[str]
    by_keys: tuple[str, ...]
    passthrough_arrow_input: Any | None = None


class _PythonInputPreparationService:
    """Builds source rows for Python backend execution.

    Boundary:
    - input loading
    - prepared-row reuse
    - dataset option application
    - SET/MERGE row construction
    - internal helper variable enrichment
    """

    def __init__(self, owner: "PythonBackendExecutionService") -> None:
        self._owner = owner

    def prepare(
        self,
        *,
        ast_statements: Sequence[Any],
        resolved_inputs: Mapping[str, DataSetRef],
        resolved_output_targets: tuple[str, ...],
    ) -> tuple[_PreparedProgramInputs | None, tuple[dict[str, DataSetRef], list[Diagnostic]] | None, list[Diagnostic]]:
        diagnostics: list[Diagnostic] = []
        set_statement = next((statement for statement in ast_statements if statement.kind == "SET"), None)
        merge_statement = next((statement for statement in ast_statements if statement.kind == "MERGE"), None)
        source_statement = merge_statement or set_statement
        by_keys = self._owner._extract_variable_list(ast_statements, "BY")
        in_option_vars: list[str] = []
        internal_variable_names: set[str] = set()
        rows: list[dict[str, Any]] = []

        if source_statement is None:
            rows = [{}]
            return (
                _PreparedProgramInputs(
                    rows=rows,
                    source_statement=None,
                    merge_statement=None,
                    in_option_vars=tuple(),
                    internal_variable_names=frozenset(),
                    by_keys=tuple(by_keys),
                ),
                None,
                diagnostics,
            )

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
            return None, None, diagnostics

        option_validation = self._owner._validate_set_option_variables(source_statement)
        if option_validation is not None:
            diagnostics.append(option_validation)
            return None, None, diagnostics

        in_option_vars = [
            ref.options.in_var
            for ref in source_statement.dataset_refs
            if ref.options.in_var is not None
        ]
        internal_variable_names = self._owner._collect_internal_variable_names(
            in_option_vars=in_option_vars,
            indsname_var=source_statement.statement_options.indsname_var,
            end_var=source_statement.statement_options.end_var,
            by_keys=by_keys,
        )

        if source_statement.dataset_refs:
            source_refs = list(source_statement.dataset_refs)
        else:
            source_refs = [
                type(
                    "_FallbackRef",
                    (),
                    {
                        "name": name,
                        "options": type(
                            "_FallbackOptions",
                            (),
                            {
                                "in_var": None,
                                "keep_vars": (),
                                "drop_vars": (),
                                "where_expr": None,
                                "rename_map": {},
                            },
                        )(),
                    },
                )
                for name in input_names
            ]

        passthrough_arrow_input = self._resolve_passthrough_arrow_input(
            merge_statement=merge_statement,
            source_refs=source_refs,
            resolved_inputs=resolved_inputs,
        )

        prepared_internal_names = self._owner._extract_prepared_internal_variable_names(
            source_refs=source_refs,
            resolved_inputs=resolved_inputs,
        )
        if prepared_internal_names:
            internal_variable_names.update(prepared_internal_names)

        prepared_rows = self._owner._extract_prepared_set_rows(
            source_statement=source_statement,
            merge_statement=merge_statement,
            source_refs=source_refs,
            resolved_inputs=resolved_inputs,
        )
        if prepared_rows is not None:
            rows = prepared_rows
            passthrough_arrow_input = None
        else:
            direct_execution = self._owner._try_execute_direct_single_set_loop(
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
                return None, direct_execution, diagnostics

            rows_with_source: list[tuple[str, dict[str, Any], str | None]] = []
            if merge_statement is not None:
                merged_rows, merge_error = self._owner._build_merge_rows(
                    source_refs=source_refs,
                    by_keys=by_keys,
                    resolved_inputs=resolved_inputs,
                    internal_variable_names=internal_variable_names,
                )
                if merge_error is not None:
                    diagnostics.append(merge_error)
                    return None, None, diagnostics
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
                        return None, None, diagnostics

                    loaded_rows, load_error = self._owner._load_input_rows(input_ref)
                    if load_error is not None:
                        diagnostics.append(load_error)
                        return None, None, diagnostics

                    allow_internal_names = (
                        isinstance(input_ref.location, str)
                        and self._owner._PREPARED_MERGE_ROWS_MARKER in input_ref.location
                    )
                    collision = self._owner._detect_internal_variable_collision(
                        rows=loaded_rows,
                        source_name=input_name,
                        internal_variable_names=internal_variable_names,
                        allow_internal_names=allow_internal_names,
                    )
                    if collision is not None:
                        diagnostics.append(collision)
                        return None, None, diagnostics

                    option_rows, option_error = self._owner._apply_dataset_reference_options(
                        rows=loaded_rows,
                        source_name=input_name,
                        option_spec=source_ref.options,
                    )
                    if option_error is not None:
                        diagnostics.append(option_error)
                        return None, None, diagnostics

                    per_source.append((input_name, option_rows, source_ref.options.in_var))

                if needs_interleave:
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

                    rows_with_source = list(
                        heapq.merge(
                            *[((n, r, iv) for r in source_rows) for n, source_rows, iv in per_source],
                            key=_by_key,
                        )
                    )
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
                rows.sort(
                    key=lambda row: tuple(
                        (row.get(key) is None, str(row.get(key)) if row.get(key) is not None else "")
                        for key in by_keys
                    )
                )
                rows, by_error = self._owner._annotate_by_group_flags(rows=rows, by_keys=by_keys)
                if by_error is not None:
                    diagnostics.append(by_error)
                    return None, None, diagnostics

        return (
            _PreparedProgramInputs(
                rows=rows,
                source_statement=source_statement,
                merge_statement=merge_statement,
                in_option_vars=tuple(in_option_vars),
                internal_variable_names=frozenset(internal_variable_names),
                by_keys=tuple(by_keys),
                passthrough_arrow_input=passthrough_arrow_input,
            ),
            None,
            diagnostics,
        )

    def _resolve_passthrough_arrow_input(
        self,
        *,
        merge_statement: Any | None,
        source_refs: Sequence[Any],
        resolved_inputs: Mapping[str, DataSetRef],
    ) -> Any | None:
        if merge_statement is not None or len(source_refs) != 1:
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

        source_input = resolved_inputs.get(source_refs[0].name)
        if source_input is None or source_input.kind != "arrow_table":
            return None
        if not hasattr(source_input.payload, "schema"):
            return None
        return source_input.payload


class _PythonProgramExecutionService:
    """Runs prepared rows through the Python backend orchestration.

    Boundary:
    - statement-level execution mode selection
    - simple row loop orchestration
    - advanced runtime handoff
    - output routing preparation
    """

    def __init__(self, owner: "PythonBackendExecutionService") -> None:
        self._owner = owner

    def execute(
        self,
        *,
        ast_statements: Sequence[Any],
        prepared: _PreparedProgramInputs,
        resolved_output_targets: tuple[str, ...],
    ) -> tuple[dict[str, DataSetRef], list[Diagnostic]]:
        rename_statement = next((statement for statement in ast_statements if statement.kind == "RENAME"), None)
        if rename_statement is not None:
            rename_validation = self._owner._validate_rename_statement(rename_statement.rename_map)
            if rename_validation is not None:
                return {}, [rename_validation]

        excluded_output_variables: set[str] = set()
        where_expression = self._owner._extract_expression(ast_statements, "WHERE")
        has_advanced_runtime = any(
            statement.kind in {"SUM", "DO", "ARRAY", "ASSIGN", "RETAIN", "STOP"}
            for statement in ast_statements
        )
        has_advanced_runtime = has_advanced_runtime or any(
            statement.kind == "IF" and self._owner._parse_if_then_do_condition(statement) is not None
            for statement in ast_statements
        )
        drop_vars = self._owner._extract_variable_list(ast_statements, "DROP")
        keep_vars = self._owner._extract_variable_list(ast_statements, "KEEP")
        rows = self._owner._maybe_prefilter_rows_with_arrow_where(rows=prepared.rows, where_expression=where_expression)
        output_dataset_options = self._owner._collect_output_dataset_options(
            ast_statements=ast_statements,
            declared_targets=resolved_output_targets,
        )

        if has_advanced_runtime:
            return self._owner._execute_program_advanced(
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

        return self._owner._execute_program_basic(
            ast_statements=ast_statements,
            rows=rows,
            resolved_output_targets=resolved_output_targets,
            source_statement=prepared.source_statement,
            merge_statement=prepared.merge_statement,
            in_option_vars=prepared.in_option_vars,
            where_expression=where_expression,
            drop_vars=drop_vars,
            keep_vars=keep_vars,
            rename_statement=rename_statement,
            excluded_output_variables=excluded_output_variables,
            output_dataset_options=output_dataset_options,
            passthrough_arrow_input=prepared.passthrough_arrow_input,
        )