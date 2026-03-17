from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .models import DataSetRef, Diagnostic


class _PythonDatasetExecutionMixin:
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
            and getattr(option_spec, "firstobs", None) is None
            and getattr(option_spec, "obs", None) is None
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

            processed.append(working)

        firstobs = getattr(option_spec, "firstobs", None)
        obs = getattr(option_spec, "obs", None)
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
            and getattr(option_spec, "firstobs", None) is None
            and getattr(option_spec, "obs", None) is None
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

        return working, None