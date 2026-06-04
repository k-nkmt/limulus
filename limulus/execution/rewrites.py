from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from ..naming import _column_key


@dataclass(frozen=True)
class RewritePlan:
    merge_mode: str
    lag_lead_mode: str
    where_mode: str
    helper_source_columns: tuple[str, ...]
    helper_runtime_columns: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "merge_mode": self.merge_mode,
            "lag_lead_mode": self.lag_lead_mode,
            "where_mode": self.where_mode,
            "helper_source_columns": self.helper_source_columns,
            "helper_runtime_columns": self.helper_runtime_columns,
        }


@dataclass(frozen=True)
class HelperOwnershipDecision:
    helper_source_columns: tuple[str, ...]
    helper_runtime_columns: tuple[str, ...]


class HelperOwnershipPolicy:
    def classify(
        self,
        *,
        source_helper_columns: Sequence[str],
        lag_lead_runtime_columns: Sequence[str],
    ) -> HelperOwnershipDecision:
        return HelperOwnershipDecision(
            helper_source_columns=tuple(source_helper_columns),
            helper_runtime_columns=tuple(lag_lead_runtime_columns),
        )


class RewritePlanner:
    _LAG_LEAD_PATTERN = re.compile(r"\b(lag|lead|shift)\s*\(", re.IGNORECASE)
    _LAG_LEAD_CALL_PATTERN = re.compile(
        r"\b(?P<fn>lag|lead|shift)\s*\(\s*"
        r"(?P<arg>'[^']+'|\"[^\"]+\"|[A-Za-z_][\w\.]*)"
        r"\s*(?:,\s*(?P<offset>-?\d+))?"
        r"\s*(?:,\s*(?P<default>[^()]+?))?\s*\)",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        self._helper_ownership_policy = HelperOwnershipPolicy()

    def build_plan(
        self,
        *,
        ast_statements: Sequence[Any],
        source_statement: Any | None,
    ) -> RewritePlan:
        helper_source_columns: list[str] = []
        helper_runtime_columns: list[str] = []

        if source_statement is not None:
            dataset_refs = list(getattr(source_statement, "dataset_refs", ()) or ())
            for dataset_ref in dataset_refs:
                options = getattr(dataset_ref, "options", None)
                if options is None:
                    continue
                in_var = getattr(options, "in_var", None)
                if in_var:
                    helper_source_columns.append(in_var)

            statement_options = getattr(source_statement, "statement_options", None)
            if statement_options is not None:
                indsname_var = getattr(statement_options, "indsname_var", None)
                if indsname_var:
                    helper_source_columns.append(indsname_var)
                end_var = getattr(statement_options, "end_var", None)
                if end_var:
                    helper_source_columns.append(end_var)

        lag_lead_specs = self.collect_lag_lead_specs(ast_statements)
        source_owned_lag_lead_specs = tuple(
            spec for spec in lag_lead_specs if self._is_source_owned_lag_lead_spec(spec)
        )
        lag_lead_runtime_columns = self.collect_runtime_owned_lag_lead_columns(
            ast_statements,
            lag_lead_specs,
        )

        for by_name in self._extract_by_variables(ast_statements):
            helper_source_columns.append(f"first.{by_name}")
            helper_source_columns.append(f"last.{by_name}")
        helper_source_columns.extend(spec["helper_column"] for spec in source_owned_lag_lead_specs)

        ownership = self._helper_ownership_policy.classify(
            source_helper_columns=helper_source_columns,
            lag_lead_runtime_columns=lag_lead_runtime_columns,
        )

        merge_mode = "prepared_merge_rows" if getattr(source_statement, "kind", None) == "MERGE" else "none"
        lag_lead_mode = (
            "planner_precomputed_columns"
            if source_owned_lag_lead_specs
            else "runtime_owned_helpers" if lag_lead_runtime_columns else "none"
        )
        where_mode = "shared_pre_row_filter" if any(getattr(statement, "kind", None) == "WHERE" for statement in ast_statements) else "none"

        return RewritePlan(
            merge_mode=merge_mode,
            lag_lead_mode=lag_lead_mode,
            where_mode=where_mode,
            helper_source_columns=ownership.helper_source_columns,
            helper_runtime_columns=ownership.helper_runtime_columns,
        )

    def collect_runtime_owned_lag_lead_columns(
        self,
        ast_statements: Sequence[Any],
        lag_lead_specs: Sequence[dict[str, Any]],
    ) -> tuple[str, ...]:
        del ast_statements
        return tuple(
            spec["helper_column"]
            for spec in lag_lead_specs
            if not self._is_source_owned_lag_lead_spec(spec)
        )

    def uses_lag_lead_functions(self, ast_statements: Sequence[Any]) -> bool:
        for statement in ast_statements:
            statement_text = getattr(statement, "text", "")
            if isinstance(statement_text, str) and self._LAG_LEAD_PATTERN.search(statement_text):
                return True
        return False

    def build_pre_row_filter(
        self,
        *,
        ast_statements: Sequence[Any],
        is_simple_filter_expression: Callable[[str], bool],
    ) -> dict[str, Any] | None:
        where_statement = next((statement for statement in ast_statements if getattr(statement, "kind", None) == "WHERE"), None)
        if where_statement is None:
            return None

        expression = getattr(where_statement, "text", "")
        if not isinstance(expression, str):
            return None
        expression = expression[len("where") :].strip() if expression.lower().startswith("where") else expression.strip()
        if expression.endswith(";"):
            expression = expression[:-1].rstrip()

        filter_feasible = is_simple_filter_expression(expression=expression)
        return {
            "expression": expression,
            "capability": {
                "mode": "simple-comparison" if filter_feasible else "runtime-eval",
                "supported": True,
            },
            "policy": {
                "stage": "pre_row_filter",
                "decision": "apply",
                "reason_code": None,
                "details": None,
            },
        }

    def collect_lag_lead_specs(self, ast_statements: Sequence[Any]) -> tuple[dict[str, Any], ...]:
        specs: list[dict[str, Any]] = []
        occurrence = 0
        for statement in ast_statements:
            statement_text = getattr(statement, "text", "")
            if not isinstance(statement_text, str):
                continue
            for match in self._LAG_LEAD_CALL_PATTERN.finditer(statement_text):
                occurrence += 1
                function_name = match.group("fn").lower()
                raw_argument = match.group("arg")
                source_name = self._normalize_lag_lead_source_name(raw_argument)
                offset = int(match.group("offset") or 1)
                if function_name in {"lag", "lead"} and offset < 1:
                    continue
                helper_column = self._build_helper_column_name(function_name, source_name, offset, occurrence)
                specs.append(
                    {
                        "call_text": match.group(0),
                        "function_name": function_name,
                        "argument_is_string_literal": self._is_string_literal_argument(raw_argument),
                        "source_name": source_name,
                        "offset": offset,
                        "default_expr": (match.group("default") or "").strip() or None,
                        "helper_column": helper_column,
                    }
                )
        return tuple(specs)

    def rewrite_lag_lead_statements(self, ast_statements: Sequence[Any]) -> tuple[Any, ...]:
        specs = self.collect_lag_lead_specs(ast_statements)
        if not specs:
            return tuple(ast_statements)

        remaining_specs = iter(specs)

        def _replace_calls(text: str) -> str:
            def _replacement(match: re.Match[str]) -> str:
                spec = next(remaining_specs)
                if self._is_source_owned_lag_lead_spec(spec):
                    return spec["helper_column"]
                return match.group(0)

            return self._LAG_LEAD_CALL_PATTERN.sub(_replacement, text)

        rewritten: list[Any] = []
        for statement in ast_statements:
            text = getattr(statement, "text", "")
            if not isinstance(text, str) or not self._LAG_LEAD_PATTERN.search(text):
                rewritten.append(statement)
                continue
            rewritten.append(replace(statement, text=_replace_calls(text)))
        return tuple(rewritten)

    def materialize_lag_lead_columns(
        self,
        *,
        rows: Sequence[dict[str, Any]],
        ast_statements: Sequence[Any],
    ) -> list[dict[str, Any]]:
        specs = tuple(
            spec
            for spec in self.collect_lag_lead_specs(ast_statements)
            if self._is_source_owned_lag_lead_spec(spec)
        )
        if not specs:
            return [dict(row) for row in rows]

        materialized_rows = [dict(row) for row in rows]
        for index, row in enumerate(materialized_rows):
            for spec in specs:
                target_index = self._target_index_for_spec(index=index, spec=spec)
                if target_index < 0 or target_index >= len(materialized_rows):
                    row[spec["helper_column"]] = self._parse_default_expression(spec["default_expr"])
                    continue
                row[spec["helper_column"]] = self._resolve_row_value(materialized_rows[target_index], spec["source_name"])
        return materialized_rows

    def build_row_sort_key(
        self,
        row: Mapping[str, Any],
        *,
        by_keys: Sequence[str],
        resolve_row_value: Callable[[Mapping[str, Any], str], Any],
    ) -> tuple[Any, ...]:
        return tuple(self._sortable_value(resolve_row_value(row, key)) for key in by_keys)

    def prepare_merge_rows(
        self,
        *,
        source_refs: Sequence[Any],
        by_keys: Sequence[str],
        resolved_inputs: Mapping[str, Any],
        internal_variable_names: set[str],
        load_input_rows: Callable[[Any], tuple[list[dict[str, Any]], Any | None]],
        apply_dataset_reference_options: Callable[..., tuple[list[dict[str, Any]], Any | None]],
        resolve_row_key: Callable[[Mapping[str, Any], str], str | None],
        resolve_row_value: Callable[[Mapping[str, Any], str], Any],
        prepared_merge_marker: str,
    ) -> tuple[list[tuple[str, dict[str, Any], str | None]], Any | None]:
        loaded_sources: list[tuple[str, str | None, list[dict[str, Any]]]] = []
        non_key_columns_by_source: list[set[str]] = []
        normalized_by_keys = {_column_key(key) for key in by_keys}
        canonical_by_names: dict[str, str] = {}

        for source_ref in source_refs:
            input_name = source_ref.name
            input_ref = resolved_inputs.get(input_name)
            if input_ref is None:
                return [], self._build_runtime_set_not_found(input_name)

            loaded_rows, load_error = load_input_rows(input_ref)
            if load_error is not None:
                return [], load_error

            allow_internal_names = (
                isinstance(getattr(input_ref, "location", None), str)
                and prepared_merge_marker in str(input_ref.location)
            )
            collision = self._detect_internal_variable_collision(
                rows=loaded_rows,
                source_name=input_name,
                internal_variable_names=internal_variable_names,
                allow_internal_names=allow_internal_names,
            )
            if collision is not None:
                return [], collision

            option_rows, option_error = apply_dataset_reference_options(
                rows=loaded_rows,
                source_name=input_name,
                option_spec=source_ref.options,
            )
            if option_error is not None:
                return [], option_error

            for row in option_rows:
                for by_key in by_keys:
                    resolved_by_key = resolve_row_key(row, by_key)
                    if resolved_by_key is None:
                        return [], self._build_by_precondition_failed(by_key, input_name)
                    canonical_by_names.setdefault(_column_key(by_key), resolved_by_key)

            non_key_columns = {
                _column_key(column_name)
                for row in option_rows
                for column_name in row.keys()
                if _column_key(column_name) not in normalized_by_keys
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
            return [], self._build_merge_duplicate_column(duplicate_label)

        if not by_keys:
            all_source_rows = [rows for _, _, rows in loaded_sources]
            max_len = max((len(rows) for rows in all_source_rows), default=0)
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
        seen_merge_keys: set[tuple[Any, ...]] = set()

        for _, _, rows in loaded_sources:
            grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
            for row in rows:
                key = tuple(resolve_row_value(row, key_name) for key_name in by_keys)
                grouped.setdefault(key, []).append(row)
                if key not in seen_merge_keys:
                    seen_merge_keys.add(key)
                    merge_key_order.append(key)
            grouped_sources.append(grouped)

        merge_key_order.sort(key=lambda item: tuple(self._sortable_value(value) for value in item))
        output_by_keys = tuple(canonical_by_names.get(_column_key(key), key) for key in by_keys)

        merged_rows: list[dict[str, Any]] = []
        merged_sources: list[str] = []
        merged_in_vars: list[str | None] = []

        for key in merge_key_order:
            group_lengths = [len(grouped.get(key, [])) for grouped in grouped_sources]
            max_group_length = max(group_lengths) if group_lengths else 0

            for row_index in range(max_group_length):
                merged_row: dict[str, Any] = {}
                contributing_sources: list[str] = []

                for source_index, (source_name, in_var, _) in enumerate(loaded_sources):
                    grouped_rows = grouped_sources[source_index].get(key, [])
                    if row_index < len(grouped_rows):
                        merged_row.update(grouped_rows[row_index])
                        contributing_sources.append(source_name)
                    elif grouped_rows:
                        merged_row.update(grouped_rows[-1])
                        contributing_sources.append(source_name)

                for by_index, by_key in enumerate(output_by_keys):
                    normalized_by_key = _column_key(by_keys[by_index])
                    duplicate_by_names = [
                        name
                        for name in tuple(merged_row)
                        if _column_key(name) == normalized_by_key and name != by_key
                    ]
                    for duplicate_name in duplicate_by_names:
                        merged_row.pop(duplicate_name, None)
                    merged_row[by_key] = key[by_index]

                merged_rows.append(merged_row)
                merged_sources.append(",".join(contributing_sources) if contributing_sources else "")
                merged_in_vars.append(None)

                for source_name, in_var, _ in loaded_sources:
                    if in_var:
                        merged_row[in_var] = 1 if source_name in contributing_sources else 0

        merged_rows, by_error = self._annotate_by_group_flags(
            rows=merged_rows,
            by_keys=by_keys,
            resolve_row_key=resolve_row_key,
            resolve_row_value=resolve_row_value,
        )
        if by_error is not None:
            return [], by_error

        return list(zip(merged_sources, merged_rows, merged_in_vars)), None

    def prepare_set_rows(
        self,
        *,
        source_refs: Sequence[Any],
        by_keys: Sequence[str],
        in_option_vars: Sequence[str],
        indsname_var: str | None,
        end_var: str | None,
        resolved_inputs: Mapping[str, Any],
        internal_variable_names: set[str],
        load_input_rows: Callable[[Any], tuple[list[dict[str, Any]], Any | None]],
        apply_dataset_reference_options: Callable[..., tuple[list[dict[str, Any]], Any | None]],
        resolve_row_key: Callable[[Mapping[str, Any], str], str | None],
        resolve_row_value: Callable[[Mapping[str, Any], str], Any],
        prepared_merge_marker: str,
    ) -> tuple[list[dict[str, Any]], Any | None]:
        rows_with_source: list[tuple[str, dict[str, Any], str | None]] = []

        for source_ref in source_refs:
            input_name = source_ref.name
            input_ref = resolved_inputs.get(input_name)
            if input_ref is None:
                return [], self._build_runtime_set_not_found(input_name)

            loaded_rows, load_error = load_input_rows(input_ref)
            if load_error is not None:
                return [], load_error

            option_rows, option_error = apply_dataset_reference_options(
                rows=loaded_rows,
                source_name=input_name,
                option_spec=source_ref.options,
            )
            if option_error is not None:
                return [], option_error

            allow_internal_names = (
                isinstance(getattr(input_ref, "location", None), str)
                and prepared_merge_marker in str(input_ref.location)
            )
            collision = self._detect_internal_variable_collision(
                rows=option_rows,
                source_name=input_name,
                internal_variable_names=internal_variable_names,
                allow_internal_names=allow_internal_names,
            )
            if collision is not None:
                return [], collision

            for row in option_rows:
                rows_with_source.append((input_name, row, source_ref.options.in_var))

        if by_keys and len(rows_with_source) > 1:
            rows_with_source.sort(
                key=lambda item: self.build_row_sort_key(
                    item[1],
                    by_keys=by_keys,
                    resolve_row_value=resolve_row_value,
                )
            )

        prepared_rows: list[dict[str, Any]] = []
        for index, (source_name, row, row_in_var) in enumerate(rows_with_source):
            enriched = dict(row)

            for in_var in in_option_vars:
                if row_in_var is None and in_var in enriched:
                    continue
                enriched[in_var] = 1 if row_in_var == in_var else 0

            if indsname_var:
                enriched[indsname_var] = source_name

            if end_var:
                enriched[end_var] = 1 if index == len(rows_with_source) - 1 else 0

            prepared_rows.append(enriched)

        prepared_rows, by_error = self._annotate_by_group_flags(
            rows=prepared_rows,
            by_keys=by_keys,
            resolve_row_key=resolve_row_key,
            resolve_row_value=resolve_row_value,
        )
        if by_error is not None:
            return [], by_error

        return prepared_rows, None

    @staticmethod
    def _extract_by_variables(ast_statements: Sequence[Any]) -> tuple[str, ...]:
        statement = next((item for item in ast_statements if getattr(item, "kind", None) == "BY"), None)
        if statement is None:
            return ()
        text = getattr(statement, "text", "")
        if not isinstance(text, str):
            return ()
        variables = text[len("by"):].strip().split()
        return tuple(name for name in variables if name)

    @staticmethod
    def _normalize_lag_lead_source_name(argument: str) -> str:
        stripped = argument.strip()
        if (stripped.startswith("'") and stripped.endswith("'")) or (stripped.startswith('"') and stripped.endswith('"')):
            return stripped[1:-1]
        return stripped

    @staticmethod
    def _build_helper_column_name(function_name: str, source_name: str, offset: int, occurrence: int) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", source_name).strip("_") or "value"
        offset_token = RewritePlanner._offset_token(offset)
        return f"__rewrite_{function_name}_{sanitized.lower()}_{offset_token}_{occurrence}__"

    @staticmethod
    def _offset_token(offset: int) -> str:
        if offset < 0:
            return f"neg{abs(offset)}"
        return str(offset)

    @staticmethod
    def _is_string_literal_argument(argument: str) -> bool:
        stripped = argument.strip()
        return (stripped.startswith("'") and stripped.endswith("'")) or (
            stripped.startswith('"') and stripped.endswith('"')
        )

    def _is_source_owned_lag_lead_spec(self, spec: Mapping[str, Any]) -> bool:
        if spec.get("function_name") == "shift" and not spec.get("argument_is_string_literal"):
            return False
        default_expr = spec.get("default_expr")
        if default_expr is None:
            return True
        parsed_default = self._parse_default_expression(default_expr)
        if parsed_default is not None:
            return True
        return default_expr.strip() in {".", "None", "none", "null", "NULL"}

    @staticmethod
    def _target_index_for_spec(*, index: int, spec: Mapping[str, Any]) -> int:
        function_name = spec["function_name"]
        offset = int(spec["offset"])
        if function_name == "lag":
            return index - offset
        if function_name == "lead":
            return index + offset
        return index + offset

    @staticmethod
    def _parse_default_expression(default_expr: str | None) -> Any:
        if default_expr is None:
            return None
        stripped = default_expr.strip()
        if stripped in {".", "None", "none", "null", "NULL"}:
            return None
        if (stripped.startswith("'") and stripped.endswith("'")) or (stripped.startswith('"') and stripped.endswith('"')):
            return stripped[1:-1]
        try:
            return int(stripped)
        except ValueError:
            try:
                return float(stripped)
            except ValueError:
                return None

    @staticmethod
    def _resolve_row_value(row: dict[str, Any], source_name: str) -> Any:
        if source_name in row:
            return row[source_name]
        normalized = _column_key(source_name)
        for candidate, value in row.items():
            if _column_key(candidate) == normalized:
                return value
        return None

    @staticmethod
    def _sortable_value(value: Any) -> tuple[int, int, Any]:
        if value is None:
            return (1, 0, 0)
        if isinstance(value, bool):
            return (0, 0, int(value))
        if isinstance(value, (int, float)):
            return (0, 1, float(value))
        if isinstance(value, dt.datetime):
            return (0, 2, value.timestamp())
        if isinstance(value, dt.date):
            return (0, 3, value.toordinal())
        if isinstance(value, dt.time):
            return (0, 4, value.isoformat())
        return (0, 5, str(value))

    @staticmethod
    def _detect_internal_variable_collision(
        *,
        rows: Sequence[Mapping[str, Any]],
        source_name: str,
        internal_variable_names: set[str],
        allow_internal_names: bool,
    ) -> Any | None:
        if allow_internal_names or not internal_variable_names:
            return None
        for row in rows:
            collided_names = sorted(internal_variable_names.intersection(row.keys()))
            if not collided_names:
                continue
            variable_name = collided_names[0]
            return RewritePlanner._build_internal_var_collision(variable_name, source_name)
        return None

    @staticmethod
    def _annotate_by_group_flags(
        *,
        rows: list[dict[str, Any]],
        by_keys: Sequence[str],
        resolve_row_key: Callable[[Mapping[str, Any], str], str | None],
        resolve_row_value: Callable[[Mapping[str, Any], str], Any],
    ) -> tuple[list[dict[str, Any]], Any | None]:
        if not rows or not by_keys:
            return rows, None

        for by_key in by_keys:
            if any(resolve_row_key(row, by_key) is None for row in rows):
                return [], RewritePlanner._build_by_rows_precondition_failed(by_key)

        for by_key in by_keys:
            for index, row in enumerate(rows):
                previous_value = resolve_row_value(rows[index - 1], by_key) if index > 0 else object()
                next_value = resolve_row_value(rows[index + 1], by_key) if index < len(rows) - 1 else object()
                current_value = resolve_row_value(row, by_key)
                row[f"FIRST.{by_key}"] = 1 if current_value != previous_value else 0
                row[f"LAST.{by_key}"] = 1 if current_value != next_value else 0
                row[f"first.{by_key}"] = row[f"FIRST.{by_key}"]
                row[f"last.{by_key}"] = row[f"LAST.{by_key}"]
        return rows, None

    @staticmethod
    def _build_runtime_set_not_found(input_name: str) -> Any:
        from ..models import Diagnostic

        return Diagnostic(
            code="RUNTIME_SET_DATASET_NOT_FOUND",
            severity="error",
            message=f"Input dataset is not provided: {input_name}",
        )

    @staticmethod
    def _build_by_precondition_failed(by_key: str, input_name: str) -> Any:
        from ..models import Diagnostic

        return Diagnostic(
            code="RUNTIME_BY_PRECONDITION_FAILED",
            severity="error",
            message=f"BY key '{by_key}' is missing in source '{input_name}'.",
        )

    @staticmethod
    def _build_by_rows_precondition_failed(by_key: str) -> Any:
        from ..models import Diagnostic

        return Diagnostic(
            code="RUNTIME_BY_PRECONDITION_FAILED",
            severity="error",
            message=f"BY key '{by_key}' is missing in source rows.",
        )

    @staticmethod
    def _build_merge_duplicate_column(duplicate_label: str) -> Any:
        from ..models import Diagnostic

        return Diagnostic(
            code="RUNTIME_MERGE_DUPLICATE_COLUMN",
            severity="error",
            message=f"MERGE inputs contain duplicate non-BY columns: {duplicate_label}",
        )

    @staticmethod
    def _build_internal_var_collision(variable_name: str, source_name: str) -> Any:
        from ..models import Diagnostic

        return Diagnostic(
            code="RUNTIME_INTERNAL_VAR_NAME_COLLISION",
            severity="error",
            message=(
                "Input dataset contains a reserved internal reference variable name: "
                f"{variable_name} (source={source_name})"
            ),
        )


__all__ = ["HelperOwnershipDecision", "HelperOwnershipPolicy", "RewritePlan", "RewritePlanner"]
