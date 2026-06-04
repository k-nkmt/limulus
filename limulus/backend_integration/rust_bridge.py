"""Arrow-to-Rust payload construction and transport classification helpers."""

from __future__ import annotations

import ast as py_ast
from collections.abc import Mapping
from dataclasses import replace
import re
from typing import Any

import pyarrow as pa

from .backend_dispatch_policy import BackendDispatchPolicy
from ..models import DataSetRef, Diagnostic
from ..parser import DatasetReference, DatasetReferenceOptionSpec
from .contracts import (
    NATIVE_RUNTIME_MODULE_GROUP,
    RuntimeExecutionContext,
    RustExecutionPayload,
    STANDARD_SOURCE_ACCESS_MODE,
)
from .transport import (
    _classify_transport_input,
    _PREPARED_MERGE_ROWS_MARKER,
    _PREPARED_SET_ROWS_MARKER,
    _SPECIAL_TRANSPORT_PATH,
    _STANDARD_TRANSPORT_KIND,
    _STANDARD_TRANSPORT_PATH,
    _uses_prepared_merge_transport,
    _uses_prepared_set_transport,
)


class RustArrowIOBridge:
    _APPLY_TARGET_PATTERN = re.compile(
        r"\bapply\s*\(\s*(?P<target>'(?:\\.|[^'])*'|\"(?:\\.|[^\"])*\")",
        re.IGNORECASE,
    )
    _PREPARED_SET_ROWS_MARKER = _PREPARED_SET_ROWS_MARKER
    _PREPARED_MERGE_ROWS_MARKER = _PREPARED_MERGE_ROWS_MARKER
    native_runtime_module_group = NATIVE_RUNTIME_MODULE_GROUP

    def build_payload(
        self,
        context: RuntimeExecutionContext,
        function_registry_keys: tuple[str, ...],
    ) -> tuple[RustExecutionPayload | None, list[Diagnostic]]:
        if isinstance(context.execution_plan, Mapping):
            execution_readiness = context.execution_plan.get("execution_readiness")
            if isinstance(execution_readiness, Mapping) and execution_readiness.get("decision") == "blocked":
                blocking_reason_codes = tuple(execution_readiness.get("blocking_reason_codes", ()) or ())
                reason_code = (
                    blocking_reason_codes[0]
                    if blocking_reason_codes
                    else "ROW_LOOP_UNSUPPORTED_PLAN"
                )
                return None, [
                    Diagnostic(
                        code=str(reason_code),
                        severity="error",
                        message="Execution plan readiness is blocked for row-loop execution.",
                    )
                ]

        ast_statements = self._prepare_ast_statements(context)
        input_streams: dict[str, Any] = {}
        legacy_inputs: dict[str, Any] = {}
        prepared_merge_mode = False
        transport_mode = context.transport_mode or "standard_arrow_stream"
        diagnostics: list[Diagnostic] = []
        apply_registry = self._build_apply_registry(ast_statements)

        for dataset_name, dataset_ref in context.resolved_inputs.items():
            transport_path = _classify_transport_input(dataset_ref)
            if transport_path == _SPECIAL_TRANSPORT_PATH:
                prepared_merge_mode = prepared_merge_mode or _uses_prepared_merge_transport(dataset_ref)
                if hasattr(dataset_ref.payload, "__arrow_c_stream__"):
                    input_streams[dataset_name] = dataset_ref.payload.__arrow_c_stream__()
                    continue

                prepared_rows = [dict(item) for item in dataset_ref.payload]
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
                    transport_mode = "diagnostic_memory_rows"
                continue

            if transport_path != _STANDARD_TRANSPORT_PATH:
                diagnostics.append(
                    Diagnostic(
                        code="RUNTIME_RUST_BRIDGE_INPUT_NOT_ARROW",
                        severity="error",
                        location=f"dataset:{dataset_name}",
                        message=f"Rust backend requires {_STANDARD_TRANSPORT_KIND} input: {dataset_name}",
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
                        message=(
                            f"{_STANDARD_TRANSPORT_KIND} payload does not expose __arrow_c_stream__: "
                            f"{dataset_name}"
                        ),
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

        source_access_mode, compatibility_path_reason = self._resolve_source_owner_metadata(
            context,
            ast_statements,
            prepared_merge_mode=prepared_merge_mode,
            transport_mode=transport_mode,
        )
        if compatibility_path_reason is not None:
            return None, [self._build_removed_owner_diagnostic(compatibility_path_reason)]

        request_options = getattr(context.request, "options", {}) if context.request is not None else {}
        format_catalog_payload = {}
        if isinstance(request_options, Mapping):
            raw_format_catalog_payload = request_options.get("format_catalog_payload")
            if isinstance(raw_format_catalog_payload, Mapping):
                format_catalog_payload = dict(raw_format_catalog_payload)

        return (
            RustExecutionPayload(
                ast_statements=ast_statements,
                output_targets=context.resolved_output_targets,
                input_streams=input_streams,
                legacy_inputs=legacy_inputs,
                apply_registry=apply_registry,
                format_catalog_payload=format_catalog_payload,
                prepared_merge_mode=prepared_merge_mode,
                function_registry_keys=function_registry_keys,
                execution_plan=context.execution_plan,
                row_loop_plan=(
                    context.execution_plan.get("row_loop_plan")
                    if isinstance(context.execution_plan, Mapping)
                    else None
                ),
                transport_mode=transport_mode,
                builder_mode=(
                    context.builder_mode
                    or (
                        context.execution_plan.get("row_loop_plan", {}).get("builder_mode")
                        if isinstance(context.execution_plan, Mapping)
                        else None
                    )
                ),
                source_access_mode=source_access_mode,
                compatibility_path_reason=compatibility_path_reason,
                rewrite_metadata=(
                    context.rewrite_metadata
                    if context.rewrite_metadata is not None
                    else (
                        context.execution_plan.get("rewrite_plan")
                        if isinstance(context.execution_plan, Mapping)
                        else None
                    )
                ),
                python_limited_mode=(
                    context.python_limited_mode
                    or BackendDispatchPolicy.requires_python_limited_backend(context)
                ),
            ),
            diagnostics,
        )

    def _build_apply_registry(
        self,
        ast_statements: tuple[Any, ...],
    ) -> dict[str, Any]:
        apply_registry: dict[str, Any] = {}
        for statement in ast_statements:
            statement_text = getattr(statement, "text", "")
            if not isinstance(statement_text, str) or "apply" not in statement_text.lower():
                continue
            for match in self._APPLY_TARGET_PATTERN.finditer(statement_text):
                target_literal = match.group("target")
                try:
                    function_name = py_ast.literal_eval(target_literal)
                except Exception:
                    continue
                if not isinstance(function_name, str) or function_name in apply_registry:
                    continue
                resolved = self._resolve_apply_target(function_name)
                if resolved is not None:
                    apply_registry[function_name] = resolved
        return apply_registry

    @staticmethod
    def _resolve_apply_target(function_name: str) -> Any | None:
        import builtins
        import importlib
        import inspect

        function = getattr(builtins, function_name, None)

        if function is None and "." in function_name:
            module_path, attr = function_name.rsplit(".", 1)
            try:
                module = importlib.import_module(module_path)
            except Exception:
                module = None
            if module is not None:
                function = getattr(module, attr, None)

        if function is None:
            for frame_info in inspect.stack()[1:]:
                frame = frame_info.frame
                if function_name in frame.f_locals:
                    function = frame.f_locals[function_name]
                    break
                if function_name in frame.f_globals:
                    function = frame.f_globals[function_name]
                    break
            if "frame_info" in locals():
                del frame_info

        return function

    def _resolve_source_owner_metadata(
        self,
        context: RuntimeExecutionContext,
        ast_statements: tuple[Any, ...],
        *,
        prepared_merge_mode: bool,
        transport_mode: str,
    ) -> tuple[str, str | None]:
        compatibility_path_reason = self._infer_compatibility_path_reason(
            context,
            ast_statements,
            prepared_merge_mode=prepared_merge_mode,
            transport_mode=transport_mode,
        )
        return STANDARD_SOURCE_ACCESS_MODE, compatibility_path_reason

    @staticmethod
    def _build_removed_owner_diagnostic(reason: str) -> Diagnostic:
        return Diagnostic(
            code="RUNTIME_RUST_SOURCE_ACCESS_UNSUPPORTED",
            severity="error",
            message=(
                "Rust backend no longer exposes migration-only source owner paths on the release surface; "
                f"trigger '{reason}' must use a borrowed success path or fail fast."
            ),
        )

    def _infer_compatibility_path_reason(
        self,
        context: RuntimeExecutionContext,
        ast_statements: tuple[Any, ...],
        *,
        prepared_merge_mode: bool,
        transport_mode: str,
    ) -> str | None:
        _ = prepared_merge_mode
        _ = transport_mode

        row_loop_plan = (
            context.execution_plan.get("row_loop_plan")
            if isinstance(context.execution_plan, Mapping)
            else None
        )
        has_unplanned_output_handoff = isinstance(row_loop_plan, Mapping) and (
            row_loop_plan.get("builder_mode") != "targeted_output_handoff"
            or row_loop_plan.get("output_mode") != "targeted_output_handoff"
        )
        # The cleanup phase keeps this path on the borrowed success surface.
        _ = has_unplanned_output_handoff
        return None

    @staticmethod
    def _dataset_options_are_passthrough(options: DatasetReferenceOptionSpec) -> bool:
        return (
            options.in_var is None
            and not options.keep_vars
            and not options.drop_vars
            and options.where_expr is None
            and not options.rename_map
            and options.firstobs is None
            and options.obs is None
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

        normalized: list[Any] = []
        for statement in filtered_statements:
            if getattr(statement, "kind", "") != "SET":
                normalized.append(statement)
                continue

            dataset_refs = tuple(getattr(statement, "dataset_refs", ()) or ())
            if dataset_refs:
                first_source_name = dataset_refs[0].name
                first_input = context.resolved_inputs.get(first_source_name)
                if first_input is not None and self._PREPARED_SET_ROWS_MARKER in (first_input.location or ""):
                    collapsed_refs = (
                        DatasetReference(name=first_source_name, options=DatasetReferenceOptionSpec()),
                    )
                    normalized.append(
                        replace(
                            statement,
                            text="set " + first_source_name,
                            dataset_refs=collapsed_refs,
                        )
                    )
                    continue

            normalized.append(statement)

        return tuple(normalized)


__all__ = ["RustArrowIOBridge"]