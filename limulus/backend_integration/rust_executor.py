"""Native block execution wrapper and diagnostics/metrics normalization helpers."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import pyarrow as pa

from ..models import DataSetRef, Diagnostic
from ..naming import _dataset_key
from ..native_bridge import load_native_module
from .contracts import (
    NATIVE_RUNTIME_MODULE_GROUP,
    RuntimeExecutionResult,
    TypedRustExecutionPayload,
    normalize_phase_metrics,
)
from .transport import _STANDARD_TRANSPORT_KIND


class RustNativeBlockExecutor:
    native_runtime_module_group = NATIVE_RUNTIME_MODULE_GROUP

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
        payload: TypedRustExecutionPayload,
    ) -> RuntimeExecutionResult:
        """Call the public typed native entry and import its complete output set."""

        if not isinstance(payload, TypedRustExecutionPayload):
            raise TypeError("payload must be a TypedRustExecutionPayload")
        if not payload.native_program_json:
            return self._result(
                outputs={},
                diagnostics=(
                    Diagnostic(
                        code="RUNTIME_RUST_BRIDGE_PAYLOAD_BUILD_FAILED",
                        severity="error",
                        message="Typed native program JSON is required.",
                    ),
                ),
            )
        if not payload.input_streams:
            return self._result(
                outputs={},
                diagnostics=(
                    Diagnostic(
                        code="RUNTIME_RUST_BRIDGE_ARROW_EXPORT_MISSING",
                        severity="error",
                        message="Rust native runtime requires Arrow C input streams.",
                    ),
                ),
            )

        raw_targets = payload.native_program.get("output_targets")
        if not isinstance(raw_targets, Sequence) or isinstance(
            raw_targets, (str, bytes, bytearray)
        ):
            return self._result(
                outputs={},
                diagnostics=(
                    Diagnostic(
                        code="RUNTIME_RUST_BRIDGE_PAYLOAD_BUILD_FAILED",
                        severity="error",
                        message="Typed native program output_targets must be a sequence.",
                    ),
                ),
            )
        expected_targets = tuple(str(target) for target in raw_targets)

        native_module, _import_error = load_native_module()
        if native_module is None:
            return self._result(
                outputs={},
                diagnostics=(
                    Diagnostic(
                        code="RUNTIME_BACKEND_CAPABILITY_MISSING",
                        severity="error",
                        message="Rust native runtime module is not available.",
                    ),
                ),
            )

        native_payload: dict[str, Any] = {
            "native_program_json": payload.native_program_json,
            "input_streams": dict(payload.input_streams),
            "execution_plan": payload.execution_plan,
            "preparation_kind": payload.preparation_kind,
            "apply_registry": dict(payload.apply_registry),
            "format_catalog_payload": dict(payload.format_catalog_payload),
        }
        try:
            native_result = native_module.execute_block(native_payload)
        except Exception as error:
            return self._result(
                outputs={},
                diagnostics=(
                    Diagnostic(
                        code="RUNTIME_RUST_NATIVE_EXECUTION_FAILED",
                        severity="error",
                        message=f"Rust native execute_block failed: {error}",
                    ),
                ),
                native_invoked=True,
            )

        if not isinstance(native_result, Mapping):
            return self._result(
                outputs={},
                diagnostics=(
                    Diagnostic(
                        code="RUNTIME_RUST_NATIVE_EXECUTION_FAILED",
                        severity="error",
                        message=(
                            "Rust native execute_block returned invalid result type."
                        ),
                    ),
                ),
                native_invoked=True,
            )

        diagnostics = self._deserialize_diagnostics(native_result.get("diagnostics"))
        phase_metrics = self._deserialize_phase_metrics(native_result.get("phase_metrics"))
        if diagnostics:
            return self._result(
                outputs={},
                diagnostics=tuple(diagnostics),
                native_invoked=True,
                phase_metrics=phase_metrics,
            )

        raw_output_streams = native_result.get("output_streams")
        if not isinstance(raw_output_streams, Mapping):
            detail = (
                "missing"
                if raw_output_streams is None
                else f"invalid type {type(raw_output_streams).__name__}"
            )
            return self._result(
                outputs={},
                diagnostics=(
                    Diagnostic(
                        code="RUNTIME_RUST_NATIVE_EXECUTION_FAILED",
                        severity="error",
                        message=(
                            "Rust native execute_block returned no valid Arrow "
                            f"output_streams mapping ({detail})."
                        ),
                    ),
                ),
                native_invoked=True,
                phase_metrics=phase_metrics,
            )

        expected_target_keys = tuple(_dataset_key(target) for target in expected_targets)
        if len(set(expected_target_keys)) != len(expected_target_keys):
            return self._result(
                outputs={},
                diagnostics=(
                    Diagnostic(
                        code="RUNTIME_RUST_NATIVE_EXECUTION_FAILED",
                        severity="error",
                        message="Typed execution payload contains duplicate output targets.",
                    ),
                ),
                native_invoked=True,
                phase_metrics=phase_metrics,
            )
        available_streams: dict[str, object] = {}
        for target, stream in raw_output_streams.items():
            target_key = _dataset_key(str(target))
            if target_key in available_streams:
                return self._result(
                    outputs={},
                    diagnostics=(
                        Diagnostic(
                            code="RUNTIME_RUST_NATIVE_EXECUTION_FAILED",
                            severity="error",
                            location=f"dataset:{target}",
                            message=(
                                "Rust native execute_block returned duplicate "
                                "output targets."
                            ),
                        ),
                    ),
                    native_invoked=True,
                    phase_metrics=phase_metrics,
                )
            available_streams[target_key] = stream

        available_keys = set(available_streams)
        expected_keys = set(expected_target_keys)
        missing_targets = tuple(
            target
            for target, key in zip(expected_targets, expected_target_keys)
            if key not in available_keys
        )
        if missing_targets:
            target = missing_targets[0]
            return self._result(
                outputs={},
                diagnostics=(
                    Diagnostic(
                        code="RUNTIME_RUST_NATIVE_EXECUTION_FAILED",
                        severity="error",
                        location=f"dataset:{target}",
                        message=(
                            "Rust native execute_block did not return the expected "
                            f"Arrow output stream for target '{target}'."
                        ),
                    ),
                ),
                native_invoked=True,
                phase_metrics=phase_metrics,
            )
        unexpected = available_keys - expected_keys
        if unexpected:
            target_key = sorted(unexpected)[0]
            return self._result(
                outputs={},
                diagnostics=(
                    Diagnostic(
                        code="RUNTIME_RUST_NATIVE_EXECUTION_FAILED",
                        severity="error",
                        location=f"dataset:{target_key}",
                        message=(
                            "Rust native execute_block returned an unexpected "
                            f"Arrow output stream for target '{target_key}'."
                        ),
                    ),
                ),
                native_invoked=True,
                phase_metrics=phase_metrics,
            )

        outputs: dict[str, DataSetRef] = {}
        for target_name, target_key in zip(expected_targets, expected_target_keys):
            try:
                importer = getattr(
                    pa.RecordBatchReader,
                    "_import_from_c_capsule",
                    pa.RecordBatchReader._import_from_c,
                )
                reader = importer(available_streams[target_key])
                table = reader.read_all()
                table = self._restore_empty_output_schema(
                    table=table,
                    target_name=target_name,
                    execution_plan=payload.execution_plan,
                )
            except Exception as error:
                return self._result(
                    outputs={},
                    diagnostics=(
                        Diagnostic(
                            code="RUNTIME_RUST_NATIVE_EXECUTION_FAILED",
                            severity="error",
                            location=f"dataset:{target_name}",
                            message=(
                                f"Failed to import Arrow C stream for '{target_name}': {error}"
                            ),
                        ),
                    ),
                    native_invoked=True,
                    phase_metrics=phase_metrics,
                )
            outputs[target_name] = DataSetRef(
                kind=_STANDARD_TRANSPORT_KIND,
                location=f"dataset://{target_name}",
                payload=table,
            )
        return self._result(
            outputs=outputs,
            diagnostics=tuple(diagnostics),
            native_invoked=True,
            phase_metrics=phase_metrics,
        )

    @staticmethod
    def _result(
        *,
        outputs: Mapping[str, DataSetRef],
        diagnostics: Sequence[Diagnostic],
        native_invoked: bool = False,
        phase_metrics: Mapping[str, float] | None = None,
    ) -> RuntimeExecutionResult:
        return RuntimeExecutionResult(
            outputs=outputs,
            diagnostics=tuple(diagnostics),
            actual_engine="rust",
            native_invoked=native_invoked,
            phase_metrics=normalize_phase_metrics(phase_metrics),
        )

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
            try:
                arrow_type = self._resolve_arrow_type_name(type_expectations.get(column_name))
            except Exception as error:
                raise ValueError(
                    f"invalid Arrow type expectation for column '{column_name}': {error}"
                ) from error
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
        type_name = str(type_name).strip()
        aliases: dict[str, pa.DataType] = {
            "int": pa.int64(),
            "float": pa.float64(),
            "str": pa.string(),
            "string": pa.string(),
            "bool": pa.bool_(),
        }
        if type_name in aliases:
            return aliases[type_name]

        timestamp_match = re.fullmatch(
            r"timestamp\[(s|ms|us|ns)(?:,\s*tz=(.+))?\]",
            type_name,
        )
        if timestamp_match is not None:
            unit = timestamp_match.group(1)
            timezone = timestamp_match.group(2)
            if timezone is not None and not timezone:
                raise ValueError("timestamp timezone must not be empty")
            return pa.timestamp(unit, tz=timezone)
        if type_name.startswith("timestamp["):
            raise ValueError(f"invalid timestamp type expectation: {type_name}")

        time_match = re.fullmatch(r"time64\[(us|ns)\]", type_name)
        if time_match is not None:
            return pa.time64(time_match.group(1))
        if type_name.startswith("time64["):
            raise ValueError(f"invalid time64 type expectation: {type_name}")

        decimal_match = re.fullmatch(r"decimal128\((\d+),\s*(-?\d+)\)", type_name.strip().lower())
        if decimal_match is not None:
            return pa.decimal128(int(decimal_match.group(1)), int(decimal_match.group(2)))
        try:
            return pa.type_for_alias(type_name)
        except Exception:
            return None

    def _deserialize_phase_metrics(self, raw_metrics: Any) -> dict[str, float]:
        metrics = normalize_phase_metrics(raw_metrics, include_legacy_aliases=True)
        return {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }

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
