"""Native block execution wrapper and diagnostics/metrics normalization helpers."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import pyarrow as pa

from ..models import DataSetRef, Diagnostic
from ..native_bridge import load_native_module
from ..parser import ast_statements_to_dict
from .contracts import (
    MetricsAdapter,
    NATIVE_RUNTIME_MODULE_GROUP,
    RustExecutionPayload,
    STANDARD_SOURCE_ACCESS_MODE,
    build_source_owner_evidence_contract,
    canonicalize_compatibility_path_reason,
    normalize_phase_metrics,
)
from .transport import _STANDARD_TRANSPORT_KIND


class RustNativeBlockExecutor:
    _POWER_PATTERN = re.compile(
        r"(?P<base>[A-Za-z_][\w\.]*)\s*\*\*\s*(?P<exp>-?\d+(?:\.\d+)?)"
    )
    _MISSING_LITERAL_PATTERN = re.compile(r"(?<![\w])\.(?![\w])")
    _ROUND_ASSIGN_PATTERN = re.compile(
        r"^\s*([A-Za-z_][\w\.]*)\s*=\s*round\s*\(",
        re.IGNORECASE,
    )
    native_runtime_module_group = NATIVE_RUNTIME_MODULE_GROUP

    def __init__(self) -> None:
        self.last_phase_metrics: dict[str, float] = {}
        self.last_source_owner_metadata: dict[str, Any] = {}

    @staticmethod
    def _normalize_diagnostic_code(code: str, message: str) -> str:
        if code == "RUNTIME_EXPRESSION_EVALUATION_ERROR" and "unsupported function:" in message.lower():
            return "RUNTIME_UNSUPPORTED_FUNCTION"
        return code

    def is_available(self) -> bool:
        native_module, _import_error = load_native_module()
        return native_module is not None

    def execute_block(
        self,
        payload: RustExecutionPayload,
    ) -> tuple[dict[str, DataSetRef], list[Diagnostic]]:
        self.last_phase_metrics = {}
        self.last_source_owner_metadata = {}
        normalize_round_outputs = self._requires_round_output_normalization(payload)
        round_output_columns = self._collect_round_output_columns(payload) if normalize_round_outputs else set()
        has_set_statement = any(getattr(statement, "kind", "") == "SET" for statement in payload.ast_statements)
        if not has_set_statement:
            return {}, [
                Diagnostic(
                    code="RUNTIME_RUST_STATEMENT_UNSUPPORTED",
                    severity="error",
                    location="statement:1",
                    message="Rust native runtime does not support DATA step execution without SET.",
                )
            ]

        native_module, _import_error = load_native_module()
        if native_module is None:
            return {}, [
                Diagnostic(
                    code="RUNTIME_BACKEND_CAPABILITY_MISSING",
                    severity="error",
                    message="Rust native runtime module is not available.",
                )
            ]

        native_payload, serialization_error = self._serialize_payload(payload)
        if serialization_error is not None:
            return {}, [serialization_error]

        try:
            native_result = native_module.execute_block(native_payload)
        except Exception as error:
            return {}, [
                Diagnostic(
                    code="RUNTIME_RUST_NATIVE_EXECUTION_FAILED",
                    severity="error",
                    message=f"Rust native execute_block failed: {error}",
                )
            ]

        if not isinstance(native_result, Mapping):
            return {}, [
                Diagnostic(
                    code="RUNTIME_RUST_NATIVE_EXECUTION_FAILED",
                    severity="error",
                    message="Rust native execute_block returned invalid result type.",
                )
            ]

        diagnostics = self._deserialize_diagnostics(native_result.get("diagnostics"))
        self.last_source_owner_metadata = self._deserialize_source_owner_metadata(
            raw_result=native_result,
            payload=payload,
        )
        if diagnostics:
            return {}, diagnostics
        self.last_phase_metrics = self._deserialize_phase_metrics(native_result.get("phase_metrics"))

        outputs: dict[str, DataSetRef] = {}
        if payload.prepared_merge_mode:
            raw_output_streams = native_result.get("output_streams")
            if isinstance(raw_output_streams, Mapping):
                for target, stream in raw_output_streams.items():
                    target_name = str(target)
                    try:
                        importer = getattr(pa.RecordBatchReader, "_import_from_c_capsule", pa.RecordBatchReader._import_from_c)
                        reader = importer(stream)
                        table = reader.read_all()
                        if normalize_round_outputs:
                            table = self._normalize_float_artifacts(table, round_output_columns)
                        table = self._restore_empty_output_schema(
                            table=table,
                            target_name=target_name,
                            execution_plan=payload.execution_plan,
                        )
                    except Exception as error:
                        return {}, [
                            Diagnostic(
                                code="RUNTIME_RUST_NATIVE_EXECUTION_FAILED",
                                severity="error",
                                location=f"dataset:{target_name}",
                                message=f"Failed to import Arrow C stream for '{target_name}': {error}",
                            )
                        ]
                    outputs[target_name] = DataSetRef(
                        kind=_STANDARD_TRANSPORT_KIND,
                        location=f"dataset://{target_name}",
                        payload=table,
                    )
                return outputs, diagnostics

        if payload.transport_mode == "diagnostic_memory_rows":
            raw_output_streams = native_result.get("output_streams")
            if isinstance(raw_output_streams, Mapping):
                for target, stream in raw_output_streams.items():
                    target_name = str(target)
                    try:
                        importer = getattr(pa.RecordBatchReader, "_import_from_c_capsule", pa.RecordBatchReader._import_from_c)
                        reader = importer(stream)
                        table = reader.read_all()
                        if normalize_round_outputs:
                            table = self._normalize_float_artifacts(table, round_output_columns)
                        rows = table.to_pylist()
                    except Exception as error:
                        return {}, [
                            Diagnostic(
                                code="RUNTIME_RUST_NATIVE_EXECUTION_FAILED",
                                severity="error",
                                location=f"dataset:{target_name}",
                                message=f"Failed to import Arrow C stream for '{target_name}': {error}",
                            )
                        ]
                    sparse_rows = [
                        {key: value for key, value in dict(row).items() if value is not None}
                        for row in rows
                        if isinstance(row, Mapping)
                    ]
                    outputs[target_name] = DataSetRef(
                        kind="memory",
                        location=f"dataset://{target_name}",
                        payload=sparse_rows,
                    )
                return outputs, diagnostics

        raw_output_streams = native_result.get("output_streams")
        if isinstance(raw_output_streams, Mapping):
            for target, stream in raw_output_streams.items():
                target_name = str(target)
                try:
                    importer = getattr(pa.RecordBatchReader, "_import_from_c_capsule", pa.RecordBatchReader._import_from_c)
                    reader = importer(stream)
                    table = reader.read_all()
                    table = self._restore_empty_output_schema(
                        table=table,
                        target_name=target_name,
                        execution_plan=payload.execution_plan,
                    )
                except Exception as error:
                    return {}, [
                        Diagnostic(
                            code="RUNTIME_RUST_NATIVE_EXECUTION_FAILED",
                            severity="error",
                            location=f"dataset:{target_name}",
                            message=f"Failed to import Arrow C stream for '{target_name}': {error}",
                        )
                    ]
                outputs[target_name] = DataSetRef(
                    kind=_STANDARD_TRANSPORT_KIND,
                    location=f"dataset://{target_name}",
                    payload=table,
                )
            return outputs, diagnostics

        return outputs, diagnostics

    def _restore_empty_output_schema(
        self,
        *,
        table: pa.Table,
        target_name: str,
        execution_plan: Any | None,
    ) -> pa.Table:
        if table.num_rows != 0 or table.num_columns != 0:
            return table

        target_spec = self._resolve_output_handoff_target_spec(
            execution_plan=execution_plan,
            target_name=target_name,
        )
        if target_spec is None:
            return table

        projected_columns = target_spec["projected_columns"]
        if not projected_columns:
            return table

        arrays: list[pa.Array] = []
        type_expectations = target_spec["type_expectations"]
        for column_name in projected_columns:
            arrow_type = self._resolve_arrow_type_name(type_expectations.get(column_name))
            arrays.append(pa.array([], type=arrow_type) if arrow_type is not None else pa.array([]))
        return pa.table(arrays, names=list(projected_columns))

    def _resolve_output_handoff_target_spec(
        self,
        *,
        execution_plan: Any | None,
        target_name: str,
    ) -> dict[str, Any] | None:
        if not isinstance(execution_plan, Mapping):
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
        types_raw = _read_mapping("type_expectations_by_target")
        return {
            "projected_columns": tuple(projected_raw.get(target_name, ()) or ()),
            "type_expectations": dict(types_raw.get(target_name, {}) or {}),
        }

    def _resolve_arrow_type_name(self, type_name: str | None) -> pa.DataType | None:
        if not type_name or type_name == "dynamic":
            return None
        aliases: dict[str, pa.DataType] = {
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

    def _requires_round_output_normalization(self, payload: RustExecutionPayload) -> bool:
        for statement in payload.ast_statements:
            statement_text = getattr(statement, "text", "")
            if isinstance(statement_text, str) and "round(" in statement_text.lower():
                return True
        return False

    def _collect_round_output_columns(self, payload: RustExecutionPayload) -> set[str]:
        columns: set[str] = set()
        for statement in payload.ast_statements:
            if getattr(statement, "kind", "") != "ASSIGN":
                continue
            statement_text = getattr(statement, "text", "")
            if not isinstance(statement_text, str):
                continue
            matched = self._ROUND_ASSIGN_PATTERN.match(statement_text)
            if matched is None:
                continue
            columns.add(matched.group(1))
        return columns

    def _normalize_float_artifacts(self, table: pa.Table, target_columns: set[str]) -> pa.Table:
        if not target_columns:
            return table

        rebuilt_columns: list[pa.Array] = []
        needs_rebuild = False

        for field, column in zip(table.schema, table.columns):
            if field.name not in target_columns:
                rebuilt_columns.append(column.combine_chunks())
                continue
            if not (pa.types.is_float32(field.type) or pa.types.is_float64(field.type)):
                rebuilt_columns.append(column.combine_chunks())
                continue

            values = column.to_pylist()
            normalized_values: list[float | None] = []
            field_changed = False
            for value in values:
                if value is None:
                    normalized_values.append(None)
                    continue
                if not isinstance(value, float) or not math.isfinite(value):
                    normalized_values.append(value)
                    continue

                normalized = value
                tolerance = 1e-12 * max(1.0, abs(value))
                for places in range(13):
                    rounded = round(value, places)
                    if abs(value - rounded) <= tolerance:
                        normalized = rounded
                        break

                if normalized == -0.0:
                    normalized = 0.0
                if normalized != value:
                    field_changed = True
                normalized_values.append(normalized)

            if field_changed:
                needs_rebuild = True
                rebuilt_columns.append(pa.array(normalized_values, type=field.type))
            else:
                rebuilt_columns.append(column.combine_chunks())

        if not needs_rebuild:
            return table
        return pa.table(rebuilt_columns, names=table.column_names)

    def _deserialize_phase_metrics(self, raw_metrics: Any) -> dict[str, float]:
        metrics = normalize_phase_metrics(raw_metrics, include_legacy_aliases=True)
        return {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }

    def _deserialize_source_owner_metadata(
        self,
        *,
        raw_result: Mapping[str, Any],
        payload: RustExecutionPayload,
    ) -> dict[str, Any]:
        return MetricsAdapter.source_owner_metadata(
            raw_metadata=raw_result,
            source_access_mode=payload.source_access_mode,
            compatibility_path_reason=payload.compatibility_path_reason,
            phase_metrics=raw_result.get("phase_metrics"),
            prepared_merge_mode=payload.prepared_merge_mode,
            transport_mode=payload.transport_mode,
        )

    def _serialize_payload(
        self,
        payload: RustExecutionPayload,
    ) -> tuple[dict[str, Any], Diagnostic | None]:
        ast_payload = ast_statements_to_dict(tuple(payload.ast_statements))
        self._normalize_power_operator_for_rust(ast_payload)
        try:
            ast_json = json.dumps(ast_payload, ensure_ascii=False)
        except Exception as error:
            return {}, Diagnostic(
                code="RUNTIME_RUST_AST_SERIALIZE_FAILED",
                severity="error",
                message=f"Failed to serialize AST for Rust runtime: {error}",
            )

        statements: list[dict[str, Any]] = []
        for statement in payload.ast_statements:
            statement_kind = getattr(statement, "kind", "")
            statement_text = getattr(statement, "text", "")
            if statement_kind == "ASSIGN":
                statement_text = self._normalize_assign_expression_for_rust(statement_text)
            statements.append(
                {
                    "kind": statement_kind,
                    "text": statement_text,
                }
            )

        if not payload.input_streams and payload.transport_mode != "diagnostic_memory_rows":
            return {}, Diagnostic(
                code="RUNTIME_RUST_BRIDGE_ARROW_EXPORT_MISSING",
                severity="error",
                message="Rust native runtime requires Arrow C input streams or prepared merge memory inputs.",
            )

        compatibility_path_reason = canonicalize_compatibility_path_reason(
            payload.compatibility_path_reason,
            prepared_merge_mode=payload.prepared_merge_mode,
            transport_mode=payload.transport_mode,
        )
        source_owner_evidence = build_source_owner_evidence_contract(
            source_access_mode=payload.source_access_mode,
            compatibility_path_reason=compatibility_path_reason,
            phase_metrics={},
        )
        serialized = {
            "ast": ast_payload,
            "ast_json": ast_json,
            "statements": statements,
            "output_targets": list(payload.output_targets),
            "input_streams": dict(payload.input_streams),
            "execution_plan": payload.execution_plan,
            "row_loop_plan": payload.row_loop_plan,
            "source_access_mode": source_owner_evidence.source_access_mode,
            "compatibility_path_reason": source_owner_evidence.compatibility_path_reason,
            "transport_mode": payload.transport_mode,
            "builder_mode": payload.builder_mode,
            "rewrite_metadata": payload.rewrite_metadata,
            "python_limited_mode": payload.python_limited_mode,
        }
        if payload.apply_registry:
            serialized["apply_registry"] = dict(payload.apply_registry)
        if payload.format_catalog_payload:
            serialized["format_catalog_payload"] = dict(payload.format_catalog_payload)
        if payload.prepared_merge_mode:
            serialized["prepared_merge_mode"] = True
        if payload.transport_mode == "diagnostic_memory_rows" and payload.legacy_inputs:
            serialized["inputs"] = dict(payload.legacy_inputs)

        return serialized, None

    def _normalize_power_operator_for_rust(self, ast_payload: dict[str, Any]) -> None:
        statements = ast_payload.get("statements")
        if not isinstance(statements, list):
            return
        for item in statements:
            if not isinstance(item, dict):
                continue
            if item.get("kind") != "ASSIGN":
                continue
            text = item.get("text")
            if not isinstance(text, str):
                continue
            item["text"] = self._normalize_assign_expression_for_rust(text)

    def _normalize_assign_expression_for_rust(self, text: str) -> str:
        return self._rewrite_missing_literal_in_assign(self._rewrite_power_expression_in_assign(text))

    def _rewrite_power_expression_in_assign(self, text: str) -> str:
        if "**" not in text or "=" not in text:
            return text
        left, right = text.split("=", 1)
        rewritten = right
        while True:
            updated = self._POWER_PATTERN.sub(r"pow(\g<base>, \g<exp>)", rewritten)
            if updated == rewritten:
                break
            rewritten = updated
        return f"{left.strip()} = {rewritten.strip()}"

    def _rewrite_missing_literal_in_assign(self, text: str) -> str:
        if "." not in text or "=" not in text:
            return text
        left, right = text.split("=", 1)
        rewritten = self._MISSING_LITERAL_PATTERN.sub("None", right)
        return f"{left.strip()} = {rewritten.strip()}"

    def _deserialize_diagnostics(self, value: Any) -> list[Diagnostic]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            return []

        diagnostics: list[Diagnostic] = []
        for item in value:
            if not isinstance(item, Mapping):
                continue
            message = str(item.get("message", "Rust native runtime returned diagnostic."))
            code = self._normalize_diagnostic_code(
                str(item.get("code", "RUNTIME_RUST_NATIVE_EXECUTION_FAILED")),
                message,
            )
            diagnostics.append(
                Diagnostic(
                    code=code,
                    severity=str(item.get("severity", "error")),
                    location=str(item.get("location", "")),
                    message=message,
                )
            )
        return diagnostics


__all__ = ["RustNativeBlockExecutor"]
