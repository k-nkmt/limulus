"""Internal transport bridge from syntax models to a typed native program."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import json
from typing import Any

from ..host_functions import HostFunctionContext, HostFunctionResolver
from ..models import Diagnostic
from ..syntax.models import (
    BlockSyntax,
    DatasetOptionSyntax,
    SourceStatementSyntax,
    StatementSyntax,
)
from ..syntax.native_payload import serialize_block, validate_native_program
from .bridge_support import export_arrow_inputs, validate_prepared_input_contract
from .contracts import (
    TypedRustExecutionContext,
    TypedRustExecutionPayload,
)


class TypedRustArrowIOBridge:
    """Build a typed native payload without changing the public runtime route."""

    def build_payload(
        self,
        context: TypedRustExecutionContext,
    ) -> tuple[TypedRustExecutionPayload | None, list[Diagnostic]]:
        if not isinstance(context, TypedRustExecutionContext):
            raise TypeError("context must be a TypedRustExecutionContext")

        readiness = context.execution_readiness
        if readiness is None and isinstance(context.execution_plan, Mapping):
            candidate = context.execution_plan.get("execution_readiness")
            readiness = candidate if isinstance(candidate, Mapping) else None
        if (
            isinstance(readiness, Mapping)
            and readiness.get("decision") == "blocked"
        ):
            blocking_reason_codes = tuple(
                readiness.get("blocking_reason_codes", ()) or ()
            )
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

        plan_diagnostics = tuple(context.output_plan.diagnostics)
        if plan_diagnostics:
            return None, list(plan_diagnostics)

        block = _without_label_statements(context.block)
        source_statement_kind, source_refs = _source_statement(block)
        preparation_diagnostics = validate_prepared_input_contract(
            resolved_inputs=context.resolved_inputs,
            preparation_kind=context.preparation_kind,
            source_statement_kind=source_statement_kind,
            source_refs=source_refs,
            validate_collapsed_shape=False,
        )
        if preparation_diagnostics:
            return None, preparation_diagnostics

        block = _normalize_prepared_set(
            block,
            resolved_inputs=context.resolved_inputs,
        )
        source_statement_kind, source_refs = _source_statement(block)
        preparation_diagnostics = validate_prepared_input_contract(
            resolved_inputs=context.resolved_inputs,
            preparation_kind=context.preparation_kind,
            source_statement_kind=source_statement_kind,
            source_refs=source_refs,
        )
        if preparation_diagnostics:
            return None, preparation_diagnostics

        target_names: list[str] = []
        seen_targets: set[str] = set()
        for statement in block.statements:
            for reference in statement.apply_references:
                target = reference.target
                if target in seen_targets:
                    continue
                seen_targets.add(target)
                target_names.append(target)

        host_context = context.request.host_function_context
        if not isinstance(host_context, HostFunctionContext):
            host_context = HostFunctionContext()
        resolution = HostFunctionResolver().resolve(target_names, host_context)
        if resolution.diagnostics:
            return None, list(resolution.diagnostics)

        output_plan = context.output_plan
        native_program = serialize_block(
            block,
            source=context.source,
            output_targets=output_plan.targets,
            default_target=output_plan.default_target,
            suppress_implicit_output=output_plan.suppress_implicit_output,
            effective_output_options=output_plan.effective_output_options,
        )
        validate_native_program(native_program)
        try:
            native_program_json = json.dumps(native_program, ensure_ascii=False)
        except Exception as error:
            return None, [
                Diagnostic(
                    code="RUNTIME_RUST_AST_SERIALIZE_FAILED",
                    severity="error",
                    message=f"Failed to serialize typed native program: {error}",
                )
            ]

        input_streams, export_diagnostics = export_arrow_inputs(
            context.resolved_inputs
        )
        if export_diagnostics:
            return None, export_diagnostics

        request_options = context.request.options
        format_catalog_payload: dict[str, Any] = {}
        if isinstance(request_options, Mapping):
            raw_catalog = request_options.get("format_catalog_payload")
            if isinstance(raw_catalog, Mapping):
                format_catalog_payload = dict(raw_catalog)

        return (
            TypedRustExecutionPayload(
                native_program=native_program,
                input_streams=input_streams,
                native_program_json=native_program_json,
                preparation_kind=context.preparation_kind,
                apply_registry=resolution.registry,
                format_catalog_payload=format_catalog_payload,
                execution_plan=context.execution_plan,
            ),
            [],
        )


def _without_label_statements(block: BlockSyntax) -> BlockSyntax:
    statements = tuple(
        statement for statement in block.statements if statement.kind != "LABEL"
    )
    if statements == block.statements:
        return block
    return replace(block, statements=statements)


def _source_statement(
    block: BlockSyntax,
) -> tuple[str | None, tuple[Any, ...]]:
    statement = next(
        (item for item in block.statements if item.kind in {"SET", "MERGE"}),
        None,
    )
    if statement is None or not isinstance(statement.payload, SourceStatementSyntax):
        return (statement.kind, ()) if statement is not None else (None, ())
    return statement.kind, tuple(statement.payload.refs)


def _normalize_prepared_set(
    block: BlockSyntax,
    *,
    resolved_inputs: Mapping[str, Any],
) -> BlockSyntax:
    """Mirror the prepared SET carrier structurally, preserving syntax facts."""

    statements: list[StatementSyntax] = []
    changed = False
    for statement in block.statements:
        payload = statement.payload
        if statement.kind != "SET" or not isinstance(
            payload, SourceStatementSyntax
        ) or not payload.refs:
            statements.append(statement)
            continue

        first_reference = payload.refs[0]
        first_input = next(
            (
                dataset_ref
                for dataset_name, dataset_ref in resolved_inputs.items()
                if str(dataset_name).casefold() == first_reference.name.casefold()
            ),
            None,
        )
        if (
            first_input is None
            or getattr(getattr(first_input, "preparation", None), "kind", "none")
            != "prepared_set"
        ):
            statements.append(statement)
            continue

        collapsed = replace(
            payload,
            refs=(replace(first_reference, options=DatasetOptionSyntax()),),
        )
        if collapsed == payload:
            statements.append(statement)
            continue
        # Replacing only the typed payload retains the enclosing span, origin,
        # and apply inventory exactly as the input-preparation owner supplied.
        statements.append(replace(statement, payload=collapsed))
        changed = True

    if not changed:
        return block
    return replace(block, statements=tuple(statements))


__all__ = ["TypedRustArrowIOBridge"]
