"""Backend-neutral output accumulation and post-projection handoff helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..models import DataSetRef, Diagnostic


class _ArrowOutputAccumulator:
    def __init__(
        self,
        *,
        projected_columns: Sequence[str] = (),
        rename_map: Mapping[str, str] | None = None,
        type_expectations: Mapping[str, str] | None = None,
    ) -> None:
        self._column_order: list[str] = list(projected_columns)
        self._columns: dict[str, list[Any]] = {name: [] for name in self._column_order}
        self._row_count = 0
        self._planned_columns = bool(projected_columns)
        self._rename_map = dict(rename_map or {})
        self._rename_targets: dict[str, str] = {
            target_name: source_name for source_name, target_name in self._rename_map.items()
        }
        self._type_expectations = dict(type_expectations or {})

    def uses_planned_columns(self) -> bool:
        return self._planned_columns

    def append(self, row: Mapping[str, Any]) -> None:
        if self._planned_columns:
            for name in self._column_order:
                source_name = self._rename_targets.get(name, name)
                resolved_name = self._resolve_row_key(row, source_name)
                self._columns[name].append(row.get(resolved_name) if resolved_name is not None else None)
            self._row_count += 1
            return
        row_dict = row if isinstance(row, dict) else dict(row)
        for name in row_dict:
            if name in self._columns:
                continue
            self._column_order.append(name)
            self._columns[name] = [None] * self._row_count
        for name in self._column_order:
            self._columns[name].append(row_dict.get(name))
        self._row_count += 1

    def append_projected_values(self, projected_values: Mapping[str, Any]) -> None:
        if not self._planned_columns:
            self.append(projected_values)
            return
        for name in self._column_order:
            self._columns[name].append(projected_values.get(name))
        self._row_count += 1

    def to_arrow_table(self) -> Any:
        import pyarrow as pa

        arrays: dict[str, Any] = {}
        for name in self._column_order:
            values = self._columns[name]
            data_type = self._resolve_arrow_type(pa, self._type_expectations.get(name))
            arrays[name] = pa.array(values, type=data_type) if data_type is not None else pa.array(values)
        return pa.Table.from_pydict(arrays)

    def _resolve_arrow_type(self, pa: Any, type_name: str | None) -> Any | None:
        if not type_name or type_name == "dynamic":
            return None
        aliases = {
            "int": pa.int64(),
            "float": pa.float64(),
            "str": pa.string(),
            "string": pa.string(),
            "bool": pa.bool_(),
        }
        if type_name in aliases:
            return aliases[type_name]
        decimal_match = re.fullmatch(r"decimal128\((\d+),\s*(-?\d+)\)", type_name.strip().lower())
        if decimal_match is not None:
            return pa.decimal128(int(decimal_match.group(1)), int(decimal_match.group(2)))
        try:
            return pa.type_for_alias(type_name)
        except Exception:
            return None

    def _resolve_row_key(self, row: Mapping[str, Any], name: str) -> str | None:
        if name in row:
            return name
        lowered = name.lower()
        for key in row:
            if isinstance(key, str) and key.lower() == lowered:
                return key
        return None


class _PythonOutputRoutingMixin:
    def _target_uses_planned_output_handoff(
        self,
        *,
        target: str,
        routed_outputs: Mapping[str, list[dict[str, Any]] | _ArrowOutputAccumulator],
        output_handoff_spec: Mapping[str, Any] | None,
        prefer_arrow_output: bool,
    ) -> bool:
        if not prefer_arrow_output or output_handoff_spec is None:
            return False
        buffer = routed_outputs.get(target)
        return isinstance(buffer, _ArrowOutputAccumulator) and buffer.uses_planned_columns()

    def _build_projected_output_values(
        self,
        *,
        context: Any,
        row: Mapping[str, Any],
        target: str,
        output_handoff_spec: Mapping[str, Any],
    ) -> dict[str, Any]:
        projected_columns_by_target = dict(output_handoff_spec.get("projected_columns_by_target", {}) or {})
        rename_maps = dict(output_handoff_spec.get("rename_map_by_target", {}) or {})
        projected_columns = tuple(projected_columns_by_target.get(target, ()) or ())
        rename_inverse = {
            target_name: source_name
            for source_name, target_name in dict(rename_maps.get(target, {}) or {}).items()
        }
        projected_values: dict[str, Any] = {}
        for projected_name in projected_columns:
            source_name = rename_inverse.get(projected_name, projected_name)
            value = self._runtime.get_row_value(row, context, source_name)
            if value is None and source_name != projected_name:
                value = self._runtime.get_row_value(row, context, projected_name)
            projected_values[projected_name] = value
        return projected_values

    def _create_output_buffers(
        self,
        *,
        resolved_output_targets: tuple[str, ...],
        prefer_arrow_output: bool,
        output_handoff_spec: Mapping[str, Any] | None = None,
    ) -> dict[str, list[dict[str, Any]] | _ArrowOutputAccumulator]:
        if not prefer_arrow_output:
            return self._runtime.create_output_buffers(resolved_output_targets)
        projected_columns = {}
        rename_maps = {}
        type_expectations = {}
        if output_handoff_spec is not None:
            projected_columns = dict(output_handoff_spec.get("projected_columns_by_target", {}) or {})
            rename_maps = dict(output_handoff_spec.get("rename_map_by_target", {}) or {})
            type_expectations = dict(output_handoff_spec.get("type_expectations_by_target", {}) or {})
        return {
            target: _ArrowOutputAccumulator(
                projected_columns=tuple(projected_columns.get(target, ()) or ()),
                rename_map=rename_maps.get(target, {}),
                type_expectations=type_expectations.get(target, {}),
            )
            for target in resolved_output_targets
        }

    def _record_output_record(
        self,
        *,
        context: Any,
        row: Mapping[str, Any],
        target: str,
        routed_outputs: Mapping[str, list[dict[str, Any]] | _ArrowOutputAccumulator],
    ) -> Diagnostic | None:
        buffer = routed_outputs.get(target)
        if buffer is None:
            return Diagnostic(
                code="RUNTIME_OUTPUT_TARGET_NOT_FOUND",
                severity="error",
                location=self._runtime._row_location(context),
                message=f"Output target is not declared: {target}",
            )
        if isinstance(buffer, _ArrowOutputAccumulator):
            if buffer.uses_planned_columns():
                buffer.append_projected_values(row)
            else:
                buffer.append(row)
            return None
        buffer.append(dict(row))
        return None

    def _finalize_output_buffers(
        self,
        *,
        routed_outputs: Mapping[str, list[dict[str, Any]] | _ArrowOutputAccumulator],
        prefer_arrow_output: bool,
    ) -> dict[str, DataSetRef]:
        outputs: dict[str, DataSetRef] = {}
        for target, buffer in routed_outputs.items():
            if prefer_arrow_output and isinstance(buffer, _ArrowOutputAccumulator):
                outputs[target] = DataSetRef(
                    kind="arrow_table",
                    location=f"dataset://{target}",
                    payload=buffer.to_arrow_table(),
                )
                continue
            assert isinstance(buffer, list)
            outputs[target] = DataSetRef(kind="memory", location=f"dataset://{target}", payload=buffer)
        return outputs

    def _resolve_output_handoff_spec(
        self,
        *,
        execution_plan: Mapping[str, Any] | None,
        resolved_output_targets: Sequence[str],
    ) -> dict[str, Any] | None:
        if execution_plan is None:
            return None
        raw_plan = execution_plan.get("output_handoff_plan")
        if raw_plan is None:
            return None

        def _read_mapping(name: str) -> Mapping[str, Any]:
            if isinstance(raw_plan, Mapping):
                value = raw_plan.get(name)
            else:
                value = getattr(raw_plan, name, None)
            return value if isinstance(value, Mapping) else {}

        projected_raw = _read_mapping("projected_columns_by_target")
        rename_raw = _read_mapping("rename_map_by_target")
        types_raw = _read_mapping("type_expectations_by_target")

        return {
            "projected_columns_by_target": {
                target: tuple(projected_raw.get(target, ()) or ())
                for target in resolved_output_targets
            },
            "rename_map_by_target": {
                target: dict(rename_raw.get(target, {}) or {})
                for target in resolved_output_targets
            },
            "type_expectations_by_target": {
                target: dict(types_raw.get(target, {}) or {})
                for target in resolved_output_targets
            },
        }

    def _apply_output_handoff_plan(
        self,
        *,
        row: Mapping[str, Any],
        target: str,
        output_handoff_spec: Mapping[str, Any] | None,
        output_dataset_options: Mapping[str, Any],
        prefer_arrow_output: bool,
    ) -> tuple[dict[str, Any], Diagnostic | None]:
        if output_handoff_spec is None:
            return self._apply_output_dataset_options(
                row=row,
                target=target,
                output_dataset_options=output_dataset_options,
            )

        if prefer_arrow_output:
            return row if isinstance(row, dict) else dict(row), None

        rename_maps = output_handoff_spec.get("rename_map_by_target", {})
        projected_columns_by_target = output_handoff_spec.get("projected_columns_by_target", {})
        rename_map = dict(rename_maps.get(target, {}) or {})
        projected_columns = tuple(projected_columns_by_target.get(target, ()) or ())

        working = row if isinstance(row, dict) else dict(row)
        if rename_map:
            resolved_rename_map: dict[str, str] = {}
            for source_name, target_name in rename_map.items():
                resolved_source_name = self._resolve_row_key(working, source_name)
                if resolved_source_name is None:
                    continue
                resolved_rename_map[resolved_source_name] = target_name
            renamed: dict[str, Any] = {}
            for key, value in working.items():
                renamed[resolved_rename_map.get(key, key)] = value
            working = renamed
        if projected_columns:
            projected: dict[str, Any] = {}
            for name in projected_columns:
                resolved_name = self._resolve_row_key(working, name)
                if resolved_name is None:
                    continue
                projected[name] = working.get(resolved_name)
            working = projected
        return working if isinstance(working, dict) else dict(working), None

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
