"""Backend-neutral input preparation for SET/MERGE rows and dataset options."""

from __future__ import annotations

import heapq
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..naming import _column_key
from ..models import DataSetRef, Diagnostic
from ..runtime.arrow_cursor import ArrowTableRowCursor


def _resolve_dataset_option_row_key(row: Mapping[str, Any], name: str) -> str | None:
    if name in row:
        return name

    normalized = _column_key(name)
    for candidate in row.keys():
        if _column_key(candidate) == normalized:
            return candidate
    return None


def _build_dataset_option_scope(row: Mapping[str, Any]) -> dict[str, Any]:
    scope = dict(row)
    for column_name, value in row.items():
        scope.setdefault(_column_key(column_name), value)
        scope.setdefault(column_name.lower(), value)
    return scope


def _normalize_dataset_option_spec(
    *,
    option_spec: Any,
    source_shaping_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if source_shaping_spec is not None:
        return {
            "keep_vars": tuple(source_shaping_spec.get("keep", ()) or ()),
            "drop_vars": tuple(source_shaping_spec.get("drop", ()) or ()),
            "rename_map": dict(source_shaping_spec.get("rename", {}) or {}),
            "where_expr": source_shaping_spec.get("where"),
            "firstobs": source_shaping_spec.get("firstobs", getattr(option_spec, "firstobs", None)),
            "obs": source_shaping_spec.get("obs", getattr(option_spec, "obs", None)),
        }
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
    source_shaping_spec: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], Diagnostic | None]:
    normalized = _normalize_dataset_option_spec(
        option_spec=option_spec,
        source_shaping_spec=source_shaping_spec,
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

        if normalized["where_expr"]:
            try:
                passes = bool(eval(normalized["where_expr"], {"__builtins__": {}}, _build_dataset_option_scope(working)))
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


_SIMPLE_ARROW_DATASET_OPTION_COMPARISON = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_\.]*)\s*(>=|<=|==|=|!=|>|<)\s*(.+?)\s*$"
)


def _resolve_arrow_table_column_name(table: Any, name: str) -> str | None:
    schema_names = tuple(getattr(getattr(table, "schema", None), "names", ()) or ())
    if name in schema_names:
        return name

    normalized = _column_key(name)
    for candidate in schema_names:
        if _column_key(candidate) == normalized:
            return candidate
    return None


def _parse_simple_arrow_dataset_option_where(expression: str) -> tuple[str, str, Any] | None:
    matched = _SIMPLE_ARROW_DATASET_OPTION_COMPARISON.match(expression.strip())
    if matched is None:
        return None

    variable_name = matched.group(1)
    operator = matched.group(2)
    raw_value = matched.group(3).strip()
    if raw_value.startswith("(") and raw_value.endswith(")"):
        raw_value = raw_value[1:-1].strip()
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


def try_apply_dataset_reference_options_to_arrow_table(
    table: Any,
    *,
    source_name: str,
    option_spec: Any,
    source_shaping_spec: Mapping[str, Any] | None = None,
) -> tuple[Any | None, Diagnostic | None]:
    normalized = _normalize_dataset_option_spec(
        option_spec=option_spec,
        source_shaping_spec=source_shaping_spec,
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

    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.compute as pc  # type: ignore
    except Exception:
        return None, None

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

    if normalized["where_expr"]:
        parsed = _parse_simple_arrow_dataset_option_where(str(normalized["where_expr"]))
        if parsed is None:
            return None, None
        variable_name, operator, scalar_value = parsed
        resolved_name = _resolve_arrow_table_column_name(working, variable_name)
        if resolved_name is None:
            return None, Diagnostic(
                code="RUNTIME_DATASET_OPTION_INVALID",
                severity="error",
                message=(
                    f"Dataset option WHERE= references unknown variable '{variable_name}' "
                    f"for source '{source_name}'."
                ),
            )
        scalar = pa.scalar(scalar_value)
        column = working[resolved_name]
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
        working = working.filter(mask)

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
        working = working.rename_columns(renamed_columns)

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


class _PythonDatasetExecutionMixin:
    @staticmethod
    def _resolve_row_key(row: Mapping[str, Any], name: str) -> str | None:
        return _resolve_dataset_option_row_key(row, name)

    def _resolve_row_value(self, row: Mapping[str, Any], name: str) -> Any:
        resolved = self._resolve_row_key(row, name)
        if resolved is None:
            return None
        return row.get(resolved)

    @staticmethod
    def _build_case_insensitive_scope(row: Mapping[str, Any]) -> dict[str, Any]:
        return _build_dataset_option_scope(row)

    def _build_merge_rows(
        self,
        source_refs: Sequence[Any],
        by_keys: tuple[str, ...],
        resolved_inputs: Mapping[str, DataSetRef],
        internal_variable_names: set[str],
    ) -> tuple[list[tuple[str, dict[str, Any], str | None]], Diagnostic | None]:
        loaded_sources: list[tuple[str, str | None, list[dict[str, Any]]]] = []
        non_key_columns_by_source: list[set[str]] = []
        normalized_by_keys = {_column_key(key) for key in by_keys}
        canonical_by_names: dict[str, str] = {}

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
                    resolved_by_key = self._resolve_row_key(row, by_key)
                    if resolved_by_key is None:
                        return [], Diagnostic(
                            code="RUNTIME_BY_PRECONDITION_FAILED",
                            severity="error",
                            message=(
                                f"BY key '{by_key}' is missing in source '{input_name}'."
                            ),
                        )
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
            return [], Diagnostic(
                code="RUNTIME_MERGE_DUPLICATE_COLUMN",
                severity="error",
                message=f"MERGE inputs contain duplicate non-BY columns: {duplicate_label}",
            )

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
                key = tuple(self._resolve_row_value(row, key_name) for key_name in by_keys)
                grouped.setdefault(key, []).append(row)
                if key not in merge_key_order:
                    merge_key_order.append(key)
            grouped_sources.append(grouped)

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
                contributing_in_vars: list[str | None] = []

                for source_index, (source_name, in_var, _) in enumerate(loaded_sources):
                    grouped_rows = grouped_sources[source_index].get(key, [])
                    if row_index < len(grouped_rows):
                        merged_row.update(grouped_rows[row_index])
                        contributing_sources.append(source_name)
                        contributing_in_vars.append(in_var)
                    elif grouped_rows:
                        merged_row.update(grouped_rows[-1])
                        contributing_sources.append(source_name)
                        contributing_in_vars.append(in_var)

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
            if any(self._resolve_row_key(row, by_key) is None for row in rows):
                return [], Diagnostic(
                    code="RUNTIME_BY_PRECONDITION_FAILED",
                    severity="error",
                    message=f"BY key '{by_key}' is missing in source rows.",
                )

        for by_key in by_keys:
            for index, row in enumerate(rows):
                previous_value = self._resolve_row_value(rows[index - 1], by_key) if index > 0 else object()
                next_value = self._resolve_row_value(rows[index + 1], by_key) if index < len(rows) - 1 else object()
                current_value = self._resolve_row_value(row, by_key)
                row[f"FIRST.{by_key}"] = 1 if current_value != previous_value else 0
                row[f"LAST.{by_key}"] = 1 if current_value != next_value else 0
                row[f"first.{by_key}"] = row[f"FIRST.{by_key}"]
                row[f"last.{by_key}"] = row[f"LAST.{by_key}"]

        return rows, None

    def _apply_dataset_reference_options(
        self,
        rows: list[dict[str, Any]],
        source_name: str,
        option_spec: Any,
        source_shaping_spec: Mapping[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], Diagnostic | None]:
        return apply_dataset_reference_options_to_rows(
            rows,
            source_name=source_name,
            option_spec=option_spec,
            source_shaping_spec=source_shaping_spec,
        )

    @staticmethod
    def _normalize_dataset_option_spec(
        *,
        option_spec: Any,
        source_shaping_spec: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return _normalize_dataset_option_spec(
            option_spec=option_spec,
            source_shaping_spec=source_shaping_spec,
        )

    @staticmethod
    def _resolve_source_shaping_spec(
        *,
        execution_plan: Mapping[str, Any] | None,
        source_name: str,
    ) -> Mapping[str, Any] | None:
        if not execution_plan:
            return None
        source_shaping = execution_plan.get("source_shaping")
        if not isinstance(source_shaping, Mapping):
            return None
        sources = source_shaping.get("sources")
        if not isinstance(sources, Sequence):
            return None
        source_key = _column_key(source_name)
        for source_spec in sources:
            if not isinstance(source_spec, Mapping):
                continue
            candidate_name = source_spec.get("source")
            if isinstance(candidate_name, str) and _column_key(candidate_name) == source_key:
                return source_spec
        return None

    def _apply_dataset_reference_options_to_row(
        self,
        row: Mapping[str, Any],
        source_name: str,
        option_spec: Any,
        source_shaping_spec: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, Diagnostic | None]:
        normalized = self._normalize_dataset_option_spec(
            option_spec=option_spec,
            source_shaping_spec=source_shaping_spec,
        )
        if (
            not normalized["keep_vars"]
            and not normalized["drop_vars"]
            and not normalized["rename_map"]
            and not normalized["where_expr"]
            and normalized["firstobs"] is None
            and normalized["obs"] is None
        ):
            return row if isinstance(row, dict) else dict(row), None

        if normalized["rename_map"] and len({_column_key(value) for value in normalized["rename_map"].values()}) != len(normalized["rename_map"]):
            return None, Diagnostic(
                code="RUNTIME_DATASET_OPTION_INVALID",
                severity="error",
                message=f"Dataset option RENAME= has duplicate target names for source '{source_name}'.",
            )

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

        if normalized["where_expr"]:
            try:
                passes = bool(eval(normalized["where_expr"], {"__builtins__": {}}, self._build_case_insensitive_scope(working)))
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

        if normalized["rename_map"]:
            resolved_rename_map: dict[str, str] = {}
            for old_name, new_name in normalized["rename_map"].items():
                resolved_old_name = self._resolve_row_key(working, old_name)
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
            renamed_row: dict[str, Any] = {}
            for key, value in working.items():
                renamed_row[resolved_rename_map.get(key, key)] = value
            working = renamed_row

        return working, None


class _ArrowRowCarrier(Sequence[Mapping[str, Any]]):
    """Sequence view over an Arrow-like table without eager table.to_pylist()."""

    def __init__(self, table: Any) -> None:
        self._table = table
        self._cursor = ArrowTableRowCursor(table)
        self._num_rows = self._cursor.row_count

    @property
    def table(self) -> Any:
        return self._table

    @property
    def cursor(self) -> ArrowTableRowCursor:
        return self._cursor

    def __len__(self) -> int:
        return self._num_rows

    def __getitem__(self, index: int | slice) -> Mapping[str, Any] | list[Mapping[str, Any]]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        self._cursor.bind(index)
        return self._cursor.snapshot()


@dataclass(frozen=True)
class _PreparedProgramInputs:
    rows: Sequence[Mapping[str, Any]]
    source_statement: Any | None
    merge_statement: Any | None
    in_option_vars: tuple[str, ...]
    internal_variable_names: frozenset[str]
    by_keys: tuple[str, ...]
    passthrough_arrow_input: Any | None = None
    prefer_arrow_output: bool = False


class _PythonInputPreparationService:
    """Builds shared prepared row inputs for residual row-loop execution.

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
        execution_plan: Mapping[str, Any] | None = None,
    ) -> tuple[_PreparedProgramInputs | None, tuple[dict[str, DataSetRef], list[Diagnostic]] | None, list[Diagnostic]]:
        diagnostics: list[Diagnostic] = []
        if isinstance(execution_plan, Mapping):
            row_loop_guard = execution_plan.get("row_loop_guard")
            if isinstance(row_loop_guard, Mapping) and row_loop_guard.get("decision") == "error":
                reason_codes = tuple(row_loop_guard.get("reason_codes", ()) or ())
                reason_code = reason_codes[0] if reason_codes else "ROW_LOOP_UNSUPPORTED_PLAN"
                diagnostics.append(
                    Diagnostic(
                        code=str(reason_code),
                        severity="error",
                        message="Row loop guard rejected this plan; legacy prepared-row fallback is disabled.",
                    )
                )
                return None, None, diagnostics

            execution_readiness = execution_plan.get("execution_readiness")
            if isinstance(execution_readiness, Mapping) and execution_readiness.get("decision") == "blocked":
                blocking_reason_codes = tuple(execution_readiness.get("blocking_reason_codes", ()) or ())
                reason_code = (
                    blocking_reason_codes[0]
                    if blocking_reason_codes
                    else "ROW_LOOP_UNSUPPORTED_PLAN"
                )
                diagnostics.append(
                    Diagnostic(
                        code=str(reason_code),
                        severity="error",
                        message="Execution plan readiness is blocked for row-loop execution.",
                    )
                )
                return None, None, diagnostics

        set_statement = next((statement for statement in ast_statements if statement.kind == "SET"), None)
        merge_statement = next((statement for statement in ast_statements if statement.kind == "MERGE"), None)
        source_statement = merge_statement or set_statement
        by_keys = self._owner._extract_variable_list(ast_statements, "BY")
        in_option_vars: list[str] = []
        internal_variable_names: set[str] = set()
        rows: Sequence[Mapping[str, Any]] = []

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
                    prefer_arrow_output=False,
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
        arrow_row_carrier = self._resolve_arrow_row_carrier(
            merge_statement=merge_statement,
            source_refs=source_refs,
            source_statement=source_statement,
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
        elif arrow_row_carrier is not None:
            rows = arrow_row_carrier
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
                execution_plan=execution_plan,
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
                        source_shaping_spec=self._owner._resolve_source_shaping_spec(
                            execution_plan=execution_plan,
                            source_name=input_name,
                        ),
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
                prefer_arrow_output=bool(
                    passthrough_arrow_input is not None
                    or arrow_row_carrier is not None
                    or self._all_source_inputs_are_arrow_tables(
                        source_refs=source_refs,
                        resolved_inputs=resolved_inputs,
                    )
                ),
            ),
            None,
            diagnostics,
        )

    @staticmethod
    def _all_source_inputs_are_arrow_tables(
        *,
        source_refs: Sequence[Any],
        resolved_inputs: Mapping[str, DataSetRef],
    ) -> bool:
        saw_source = False
        for source_ref in source_refs:
            source_input = resolved_inputs.get(source_ref.name)
            if source_input is None:
                continue
            saw_source = True
            if source_input.kind != "arrow_table":
                return False
        return saw_source

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

    def _resolve_arrow_row_carrier(
        self,
        *,
        merge_statement: Any | None,
        source_refs: Sequence[Any],
        source_statement: Any | None,
        resolved_inputs: Mapping[str, DataSetRef],
    ) -> _ArrowRowCarrier | None:
        if merge_statement is not None or source_statement is None or len(source_refs) != 1:
            return None
        if source_statement.statement_options.indsname_var is not None:
            return None
        if source_statement.statement_options.end_var is not None:
            return None

        option_spec = getattr(source_refs[0], "options", None)
        if option_spec is not None and (
            getattr(option_spec, "keep_vars", ())
            or getattr(option_spec, "drop_vars", ())
            or getattr(option_spec, "rename_map", {})
            or getattr(option_spec, "where_expr", None)
            or getattr(option_spec, "in_var", None) is not None
            or getattr(option_spec, "firstobs", None) is not None
            or getattr(option_spec, "obs", None) is not None
        ):
            return None

        source_input = resolved_inputs.get(source_refs[0].name)
        if source_input is None or source_input.kind != "arrow_table":
            return None
        if not hasattr(source_input.payload, "schema"):
            return None
        if not hasattr(source_input.payload, "column"):
            return None
        return _ArrowRowCarrier(source_input.payload)
