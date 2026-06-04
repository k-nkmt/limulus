"""Prepared-row orchestration helpers for the Python backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..backend_integration.backend_dispatch_policy import BackendDispatchPolicy
from ..models import DataSetRef, Diagnostic
from .input_preparation import _PreparedProgramInputs


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
        execution_plan: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, DataSetRef], list[Diagnostic]]:
        output_handoff_spec = self._owner._resolve_output_handoff_spec(
            execution_plan=execution_plan,
            resolved_output_targets=resolved_output_targets,
        )
        post_projection = self._owner._resolve_post_projection_spec(
            ast_statements=ast_statements,
            execution_plan=execution_plan,
        )
        rename_map = post_projection["rename"]

        excluded_output_variables: set[str] = set()
        where_expression = self._owner._extract_expression(ast_statements, "WHERE")
        where_statement = next((statement for statement in ast_statements if statement.kind == "WHERE"), None)
        compatibility_path_plan = BackendDispatchPolicy.build_compatibility_path_plan(ast_statements)
        drop_vars = tuple(post_projection["drop"])
        keep_vars = tuple(post_projection["keep"])
        rows = self._owner._maybe_prefilter_rows_with_arrow_where(rows=prepared.rows, where_expression=where_expression)
        rows, where_error = self._owner._prefilter_rows_with_runtime_where(
            rows=rows,
            where_expression=where_expression,
            where_statement=where_statement,
        )
        if where_error is not None:
            return {}, [where_error]
        projection_validation = self._owner._validate_post_projection_schema(
            rows=rows,
            keep_vars=keep_vars,
            drop_vars=drop_vars,
            rename_map=rename_map,
        )
        if projection_validation is not None:
            return {}, [projection_validation]
        output_dataset_options = self._owner._collect_output_dataset_options(
            ast_statements=ast_statements,
            declared_targets=resolved_output_targets,
        )

        if compatibility_path_plan.selected_path == "advanced":
            return self._owner._execute_program_advanced(
                ast_statements=ast_statements,
                rows=rows,
                resolved_output_targets=resolved_output_targets,
                where_expression=where_expression,
                drop_vars=drop_vars,
                keep_vars=keep_vars,
                rename_map=rename_map,
                excluded_output_variables=excluded_output_variables,
                output_dataset_options=output_dataset_options,
                output_handoff_spec=output_handoff_spec,
                prefer_arrow_output=prepared.prefer_arrow_output,
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
            rename_map=rename_map,
            excluded_output_variables=excluded_output_variables,
            output_dataset_options=output_dataset_options,
            passthrough_arrow_input=prepared.passthrough_arrow_input,
            output_handoff_spec=output_handoff_spec,
            prefer_arrow_output=prepared.prefer_arrow_output,
        )
