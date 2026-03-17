from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

import pyarrow as pa

from .models import DataSetRef, Diagnostic, ExecuteRequest
from .native_bridge import load_native_module
from .parser import DatasetReference, DatasetReferenceOptionSpec, ast_statements_to_dict


@dataclass(frozen=True)
class RuntimeExecutionContext:
    request: ExecuteRequest
    ast_statements: Sequence[Any]
    resolved_inputs: Mapping[str, DataSetRef]
    resolved_output_targets: tuple[str, ...]


@dataclass(frozen=True)
class RustExecutionPayload:
    ast_statements: Sequence[Any]
    output_targets: tuple[str, ...]
    input_streams: Mapping[str, Any]
    legacy_inputs: Mapping[str, Any] = None
    prepared_merge_mode: bool = False
    function_registry_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.legacy_inputs is None:
            object.__setattr__(self, "legacy_inputs", {})


RuntimeExecuteFn = Callable[[RuntimeExecutionContext], tuple[dict[str, DataSetRef], list[Diagnostic]]]


class RustArrowIOBridge:
    _APPLY_CALL_PATTERN = re.compile(r"\bapply\s*\(", re.IGNORECASE)
    _PREPARED_SET_ROWS_MARKER = "#prepared_set_rows"
    _PREPARED_MERGE_ROWS_MARKER = "#prepared_merge_rows"

    def build_payload(
        self,
        context: RuntimeExecutionContext,
        function_registry_keys: tuple[str, ...],
    ) -> tuple[RustExecutionPayload | None, list[Diagnostic]]:
        ast_statements = self._prepare_ast_statements(context)
        input_streams: dict[str, Any] = {}
        legacy_inputs: dict[str, Any] = {}
        prepared_merge_mode = False
        diagnostics: list[Diagnostic] = []

        for dataset_name, dataset_ref in context.resolved_inputs.items():
            normalized_kind = dataset_ref.kind.strip().lower()
            if normalized_kind == "memory":
                location = dataset_ref.location or ""
                payload = dataset_ref.payload
                if (
                    (
                        self._PREPARED_SET_ROWS_MARKER in location
                        or self._PREPARED_MERGE_ROWS_MARKER in location
                    )
                    and isinstance(payload, Sequence)
                    and not isinstance(payload, (str, bytes, bytearray))
                    and all(isinstance(item, Mapping) for item in payload)
                ):
                    prepared_merge_mode = self._PREPARED_MERGE_ROWS_MARKER in location
                    prepared_rows = [dict(item) for item in payload]
                    try:
                        all_keys: list[str] = []
                        seen_keys: set[str] = set()
                        for row in prepared_rows:
                            for key in row.keys():
                                if key in seen_keys:
                                    continue
                                seen_keys.add(key)
                                all_keys.append(key)

                        normalized_rows = [
                            {key: row.get(key) for key in all_keys}
                            for row in prepared_rows
                        ]
                        prepared_table = pa.Table.from_pylist(normalized_rows)
                        input_streams[dataset_name] = prepared_table.__arrow_c_stream__()
                    except Exception:
                        legacy_inputs[dataset_name] = prepared_rows
                    continue

                diagnostics.append(
                    Diagnostic(
                        code="RUNTIME_RUST_BRIDGE_INPUT_NOT_ARROW",
                        severity="error",
                        location=f"dataset:{dataset_name}",
                        message=f"Rust backend requires arrow_table input: {dataset_name}",
                    )
                )
                return None, diagnostics
            if normalized_kind != "arrow_table":
                diagnostics.append(
                    Diagnostic(
                        code="RUNTIME_RUST_BRIDGE_INPUT_NOT_ARROW",
                        severity="error",
                        location=f"dataset:{dataset_name}",
                        message=f"Rust backend requires arrow_table input: {dataset_name}",
                    )
                )
                return None, diagnostics

            payload = dataset_ref.payload
            if payload is None or not hasattr(payload, "__arrow_c_stream__"):
                diagnostics.append(
                    Diagnostic(
                        code="RUNTIME_RUST_BRIDGE_ARROW_EXPORT_MISSING",
                        severity="error",
                        location=f"dataset:{dataset_name}",
                        message=f"arrow_table payload does not expose __arrow_c_stream__: {dataset_name}",
                    )
                )
                return None, diagnostics
            try:
                input_streams[dataset_name] = payload.__arrow_c_stream__()
            except Exception as error:
                diagnostics.append(
                    Diagnostic(
                        code="RUNTIME_RUST_BRIDGE_ARROW_EXPORT_FAILED",
                        severity="error",
                        location=f"dataset:{dataset_name}",
                        message=f"Failed to export Arrow C Data stream for '{dataset_name}': {error}",
                    )
                )
                return None, diagnostics

        for index, statement in enumerate(ast_statements, start=1):
            statement_text = getattr(statement, "text", "")
            if isinstance(statement_text, str) and self._APPLY_CALL_PATTERN.search(statement_text):
                diagnostics.append(
                    Diagnostic(
                        code="RUNTIME_RUST_FUNCTION_UNSUPPORTED",
                        severity="error",
                        location=f"statement:{index}",
                        message="Rust backend does not support apply(); use python runtime backend.",
                    )
                )
                return None, diagnostics

        return (
            RustExecutionPayload(
                ast_statements=ast_statements,
                output_targets=context.resolved_output_targets,
                input_streams=input_streams,
                legacy_inputs=legacy_inputs,
                prepared_merge_mode=prepared_merge_mode,
                function_registry_keys=function_registry_keys,
            ),
            diagnostics,
        )

    def _prepare_ast_statements(
        self,
        context: RuntimeExecutionContext,
    ) -> tuple[Any, ...]:
        filtered_statements = tuple(
            statement
            for statement in context.ast_statements
            if getattr(statement, "kind", "") != "LABEL"
        )

        prepared_set_sources = {
            name
            for name, dataset_ref in context.resolved_inputs.items()
            if dataset_ref.kind.strip().lower() == "memory"
            and self._PREPARED_SET_ROWS_MARKER in (dataset_ref.location or "")
        }
        if not prepared_set_sources:
            return filtered_statements

        normalized: list[Any] = []
        for statement in filtered_statements:
            if getattr(statement, "kind", "") != "SET":
                normalized.append(statement)
                continue

            dataset_refs = tuple(
                DatasetReference(name=name, options=DatasetReferenceOptionSpec())
                for name in prepared_set_sources
            )
            normalized.append(
                replace(
                    statement,
                    text="set " + " ".join(ref.name for ref in dataset_refs),
                    dataset_refs=dataset_refs,
                )
            )

        return tuple(normalized)


class RustNativeBlockExecutor:
    _POWER_PATTERN = re.compile(
        r"(?P<base>[A-Za-z_][\w\.]*)\s*\*\*\s*(?P<exp>-?\d+(?:\.\d+)?)"
    )
    _ROUND_ASSIGN_PATTERN = re.compile(
        r"^\s*([A-Za-z_][\w\.]*)\s*=\s*round\s*\(",
        re.IGNORECASE,
    )

    def execute_block(
        self,
        payload: RustExecutionPayload,
    ) -> tuple[dict[str, DataSetRef], list[Diagnostic]]:
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
        if diagnostics:
            return {}, diagnostics

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

        if payload.legacy_inputs:
            raw_outputs = native_result.get("outputs")
            if isinstance(raw_outputs, Mapping):
                for target, rows in raw_outputs.items():
                    target_name = str(target)
                    if not isinstance(rows, Sequence):
                        continue
                    normalized_rows = [dict(row) for row in rows if isinstance(row, Mapping)]
                    outputs[target_name] = DataSetRef(
                        kind="memory",
                        location=f"dataset://{target_name}",
                        payload=normalized_rows,
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
                    if normalize_round_outputs:
                        table = self._normalize_float_artifacts(table, round_output_columns)
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
                    kind="arrow_table",
                    location=f"dataset://{target_name}",
                    payload=table,
                )
            return outputs, diagnostics

        raw_outputs = native_result.get("outputs")
        if isinstance(raw_outputs, Mapping):
            for target, rows in raw_outputs.items():
                target_name = str(target)
                if not isinstance(rows, Sequence):
                    continue
                normalized_rows = [dict(row) for row in rows if isinstance(row, Mapping)]
                outputs[target_name] = DataSetRef(
                    kind="memory",
                    location=f"dataset://{target_name}",
                    payload=normalized_rows,
                )

        return outputs, diagnostics

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
                statement_text = self._rewrite_power_expression_in_assign(statement_text)
            statements.append(
                {
                    "kind": statement_kind,
                    "text": statement_text,
                }
            )

        if not payload.input_streams and not payload.legacy_inputs:
            return {}, Diagnostic(
                code="RUNTIME_RUST_BRIDGE_ARROW_EXPORT_MISSING",
                severity="error",
                message="Rust native runtime requires Arrow C input streams or prepared merge memory inputs.",
            )

        serialized = {
            "ast": ast_payload,
            "ast_json": ast_json,
            "statements": statements,
            "output_targets": list(payload.output_targets),
            "input_streams": dict(payload.input_streams),
        }
        if payload.legacy_inputs:
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
            item["text"] = self._rewrite_power_expression_in_assign(text)

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

    def _deserialize_diagnostics(self, value: Any) -> list[Diagnostic]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            return []

        diagnostics: list[Diagnostic] = []
        for item in value:
            if not isinstance(item, Mapping):
                continue
            diagnostics.append(
                Diagnostic(
                    code=str(item.get("code", "RUNTIME_RUST_NATIVE_EXECUTION_FAILED")),
                    severity=str(item.get("severity", "error")),
                    location=str(item.get("location", "")),
                    message=str(item.get("message", "Rust native runtime returned diagnostic.")),
                )
            )
        return diagnostics


class PythonRuntimeBackend:
    name = "python"

    def __init__(
        self,
        execute_impl: RuntimeExecuteFn,
    ) -> None:
        self._execute_impl = execute_impl

    def is_available(self) -> bool:
        return True

    def execute(
        self,
        context: RuntimeExecutionContext,
    ) -> tuple[dict[str, DataSetRef], list[Diagnostic]]:
        return self._execute_impl(context)


class RustRuntimeBackend:
    name = "rust"

    def __init__(
        self,
        bridge: RustArrowIOBridge | None = None,
        executor: Any | None = None,
        function_registry_keys: tuple[str, ...] = (),
        function_registry_keys_provider: Callable[[], tuple[str, ...]] | None = None,
    ) -> None:
        self._bridge = bridge
        self._executor = executor
        self._function_registry_keys = function_registry_keys
        self._function_registry_keys_provider = function_registry_keys_provider

    def is_available(self) -> bool:
        return self._bridge is not None and self._executor is not None

    def execute(
        self,
        context: RuntimeExecutionContext,
    ) -> tuple[dict[str, DataSetRef], list[Diagnostic]]:
        if self._bridge is None or self._executor is None:
            return {}, [
                Diagnostic(
                    code="RUNTIME_BACKEND_CAPABILITY_MISSING",
                    severity="error",
                    message="Rust runtime backend is not available in this build.",
                )
            ]

        function_registry_keys = self._function_registry_keys
        if self._function_registry_keys_provider is not None:
            function_registry_keys = self._function_registry_keys_provider()

        payload, diagnostics = self._bridge.build_payload(context, function_registry_keys)
        if diagnostics:
            return {}, diagnostics
        if payload is None:
            return {}, [
                Diagnostic(
                    code="RUNTIME_RUST_BRIDGE_PAYLOAD_BUILD_FAILED",
                    severity="error",
                    message="Rust runtime bridge could not construct payload.",
                )
            ]

        if not hasattr(self._executor, "execute_block"):
            return {}, [
                Diagnostic(
                    code="RUNTIME_BACKEND_CAPABILITY_MISSING",
                    severity="error",
                    message="Rust runtime executor does not implement execute_block.",
                )
            ]

        return self._executor.execute_block(payload)

class RuntimeBackendSelector:
    def __init__(self, python_backend: PythonRuntimeBackend, rust_backend: RustRuntimeBackend) -> None:
        self._python_backend = python_backend
        self._rust_backend = rust_backend

    def select(self, preferred_backend: str) -> PythonRuntimeBackend | RustRuntimeBackend:
        preferred = preferred_backend.strip().lower()
        if preferred in {"auto", "rust"} and self._rust_backend.is_available():
            return self._rust_backend
        return self._python_backend
