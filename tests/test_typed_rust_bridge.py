from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import inspect
from unittest.mock import patch

import pyarrow as pa
import pytest

from limulus.backend_integration import contracts as backend_contracts
from limulus.backend_integration import rust_executor as rust_executor_module
from limulus.backend_integration import typed_bridge
from limulus.backend_integration.contracts import TypedRustExecutionContext
from limulus.backend_integration.rust_runtime import RustRuntimeBackend
from limulus.backend_integration.typed_bridge import TypedRustArrowIOBridge
from limulus.execution import contracts as execution_contracts
from limulus.execution import coordinator as coordinator_module
from limulus.execution.runtime_handoff import ExecutionRuntimeHandoffService
from limulus.execution.typed_output_planning import TypedOutputTargetPlan
from limulus.host_functions import HostFunctionContext, HostFunctionResolver
from limulus.models import DataSetRef, Diagnostic, ExecuteRequest, PreparationProvenance
from limulus.syntax import parse_program
from limulus.syntax.native_payload import NativePayloadValidationError
from limulus.syntax.models import (
    ApplyReferenceSyntax,
    AssignStatementSyntax,
    DatasetOptionSyntax,
    DeferredExpressionError,
    DeferredStatementError,
    IfStatementSyntax,
    LiteralSyntax,
    SourceStatementSyntax,
)


@dataclass
class _ArrowExportProbe:
    events: list[str]
    table: pa.Table

    def __arrow_c_stream__(self):
        self.events.append("export")
        return self.table.__arrow_c_stream__()


def _typed_context(
    source_text: str,
    *,
    payload: object | None = None,
    host_functions: Mapping[str, Callable[..., object]] | None = None,
    output_plan: TypedOutputTargetPlan | None = None,
) -> TypedRustExecutionContext:
    parsed = parse_program(source_text)
    assert parsed.program is not None
    assert len(parsed.program.blocks) == 1
    return TypedRustExecutionContext(
        source=parsed.program.source,
        block=parsed.program.blocks[0],
        request=ExecuteRequest(
            dsl_text=source_text,
            host_function_context=HostFunctionContext(
                explicit=dict(host_functions or {}),
            ),
        ),
        resolved_inputs={
            "in": DataSetRef(
                kind="arrow_table",
                location="dataset://in",
                payload=payload if payload is not None else pa.table({"amount": [2]}),
            )
        },
        preparation_kind="none",
        output_plan=output_plan
        or TypedOutputTargetPlan(
            targets=("out",),
            default_target="out",
            suppress_implicit_output=False,
            effective_output_options={"out": DatasetOptionSyntax()},
        ),
    )


def test_typed_bridge_builds_native_program_and_resolves_saved_inventory_before_export() -> None:
    events: list[str] = []
    calls: list[tuple[tuple[str, ...], HostFunctionContext]] = []
    invoked: list[object] = []

    def custom(value: object) -> object:
        invoked.append(value)
        return value

    context = _typed_context(
        'data out; set in; result=apply("custom", amount); run;',
        payload=_ArrowExportProbe(events, pa.table({"amount": [2]})),
        host_functions={"custom": custom},
    )
    real_resolve = HostFunctionResolver.resolve

    def resolve_once(resolver, targets, host_context):
        events.append("resolve")
        calls.append((tuple(targets), host_context))
        return real_resolve(resolver, targets, host_context)

    with patch.object(
        HostFunctionResolver,
        "resolve",
        autospec=True,
        side_effect=resolve_once,
    ) as resolve:
        payload, diagnostics = TypedRustArrowIOBridge().build_payload(context)

    assert diagnostics == []
    assert payload is not None
    assert payload.native_program["version"] == 1
    assert payload.apply_registry["custom"] is custom
    assert payload.input_streams["in"] is not None
    assert calls == [(('custom',), context.request.host_function_context)]
    assert resolve.call_count == 1
    assert events == ["resolve", "export"]
    assert invoked == []


def test_typed_bridge_resolves_only_ordered_postrewrite_inventory() -> None:
    source_text = (
        "data out; set in; "
        "label note=\"apply('quoted', amount)\"; "
        "/* apply('comment', amount) */ "
        "prefixed=preapply('prefix', amount); "
        "first=apply('dup', amount); second=apply('dup', amount); "
        "third=apply('Case', amount); fourth=apply('case', amount); run;"
    )
    functions: dict[str, Callable[[object], object]] = {
        "dup": lambda value: value,
        "Case": lambda value: value,
        "case": lambda value: value,
    }
    context = _typed_context(source_text, host_functions=functions)
    references = [
        reference
        for statement in context.block.statements
        for reference in statement.apply_references
    ]
    assert [reference.target for reference in references] == [
        "dup",
        "dup",
        "Case",
        "case",
    ]
    label_statement = next(
        statement
        for statement in context.block.statements
        if statement.kind == "LABEL"
    )
    label_reference = ApplyReferenceSyntax(
        target="label_only",
        call_span=references[0].call_span,
        owner_span=label_statement.span,
    )
    label_statement = replace(
        label_statement,
        apply_references=(label_reference,),
    )
    context = replace(
        context,
        block=replace(
            context.block,
            statements=tuple(
                label_statement if statement.kind == "LABEL" else statement
                for statement in context.block.statements
            ),
        ),
    )

    events: list[object] = []
    real_resolve = HostFunctionResolver.resolve

    def capture_targets(resolver, targets, host_context):
        events.append(tuple(targets))
        return real_resolve(resolver, targets, host_context)

    with patch.object(
        HostFunctionResolver,
        "resolve",
        autospec=True,
        side_effect=capture_targets,
    ):
        payload, diagnostics = TypedRustArrowIOBridge().build_payload(context)

    assert diagnostics == []
    assert payload is not None
    assert list(payload.apply_registry) == ["dup", "Case", "case"]
    assert events == [("dup", "Case", "case")]
    assert all(
        statement["kind"] != "LABEL"
        for statement in payload.native_program["statements"]
    )


def test_typed_bridge_uses_rewritten_structure_when_display_still_has_removed_call() -> None:
    source_text = "data out; set in; result=apply('removed', amount); run;"
    context = _typed_context(source_text)
    assign = next(
        statement for statement in context.block.statements if statement.kind == "ASSIGN"
    )
    assert isinstance(assign.payload, AssignStatementSyntax)
    value_span = assign.payload.value.span
    rewritten_value = LiteralSyntax(kind="numeric", lexeme="1", span=value_span)
    rewritten_assign = replace(
        assign,
        payload=replace(assign.payload, value=rewritten_value),
        apply_references=(),
    )
    rewritten_block = replace(
        context.block,
        statements=tuple(
            rewritten_assign if statement is assign else statement
            for statement in context.block.statements
        ),
    )
    rewritten_context = replace(context, block=rewritten_block)

    payload, diagnostics = TypedRustArrowIOBridge().build_payload(rewritten_context)

    assert diagnostics == []
    assert payload is not None
    assert payload.apply_registry == {}
    serialized_assign = next(
        statement
        for statement in payload.native_program["statements"]
        if statement["kind"] == "ASSIGN"
    )
    assert "apply('removed', amount)" in serialized_assign["display_text"]
    assert serialized_assign["apply_references"] == []

    mutated_source = replace(
        rewritten_context.source,
        text=rewritten_context.source.text.replace("removed", "changed"),
    )
    mutated_context = replace(rewritten_context, source=mutated_source)
    mutated_payload, diagnostics = TypedRustArrowIOBridge().build_payload(
        mutated_context
    )
    assert diagnostics == []
    assert mutated_payload is not None
    assert mutated_payload.apply_registry == {}
    mutated_assign = next(
        statement
        for statement in mutated_payload.native_program["statements"]
        if statement["kind"] == "ASSIGN"
    )
    assert "apply('changed', amount)" in mutated_assign["display_text"]

    def without_display_text(value):
        if isinstance(value, dict):
            return {
                key: without_display_text(item)
                for key, item in value.items()
                if key != "display_text"
            }
        if isinstance(value, list):
            return [without_display_text(item) for item in value]
        return value

    assert without_display_text(mutated_payload.native_program) == without_display_text(
        payload.native_program
    )


def test_typed_bridge_resolves_deferred_or_unreached_apply_before_empty_arrow_export() -> None:
    for source_text in (
        "data out; set in; if 0 then x=apply('unreached', amount); run;",
        "data out; set in; x=apply('deferred', amount) + amount||; run;",
    ):
        events: list[str] = []
        context = _typed_context(
            source_text,
            payload=_ArrowExportProbe(events, pa.table({"amount": []})),
        )
        if "deferred" in source_text:
            assign = next(
                statement
                for statement in context.block.statements
                if statement.kind == "ASSIGN"
            )
            assert isinstance(assign.payload, AssignStatementSyntax)
            assert isinstance(assign.payload.value, DeferredExpressionError)

        real_resolve = HostFunctionResolver.resolve

        def record_resolve(resolver, targets, host_context):
            events.append("resolve")
            return real_resolve(resolver, targets, host_context)

        with patch.object(
            HostFunctionResolver,
            "resolve",
            autospec=True,
            side_effect=record_resolve,
        ):
            payload, diagnostics = TypedRustArrowIOBridge().build_payload(context)

        assert payload is None
        assert len(diagnostics) == 1
        assert diagnostics[0].code == "RUNTIME_APPLY_TARGET_UNRESOLVED"
        assert diagnostics[0].severity == "error"
        assert diagnostics[0].stage == "resolve"
        assert any(
            target in diagnostics[0].message
            for target in ("unreached", "deferred")
        )
        assert events == ["resolve"]


def test_typed_bridge_resolves_inventory_from_deferred_inline_action() -> None:
    events: list[str] = []
    context = _typed_context(
        'data out; if c then foo apply("faulted_action", amount); run;',
        payload=_ArrowExportProbe(events, pa.table({"amount": []})),
    )
    condition = next(
        statement for statement in context.block.statements if statement.kind == "IF"
    )
    assert isinstance(condition.payload, IfStatementSyntax)
    assert isinstance(condition.payload.then_action, DeferredStatementError)
    assert [reference.target for reference in condition.apply_references] == [
        "faulted_action"
    ]
    real_resolve = HostFunctionResolver.resolve

    def record_resolve(resolver, targets, host_context):
        events.append("resolve")
        return real_resolve(resolver, targets, host_context)

    with patch.object(
        HostFunctionResolver,
        "resolve",
        autospec=True,
        side_effect=record_resolve,
    ):
        payload, diagnostics = TypedRustArrowIOBridge().build_payload(context)

    assert payload is None
    assert len(diagnostics) == 1
    assert diagnostics[0].code == "RUNTIME_APPLY_TARGET_UNRESOLVED"
    assert diagnostics[0].stage == "resolve"
    assert "faulted_action" in diagnostics[0].message
    assert events == ["resolve"]


def test_invalid_output_plan_and_blocked_readiness_precede_bridge_side_effects() -> None:
    events: list[str] = []
    invalid_plan_diagnostic = Diagnostic(
        code="RUNTIME_DATASET_OPTION_INVALID",
        severity="error",
        message="invalid output plan",
        stage="validate",
    )
    invalid_plan = TypedOutputTargetPlan(
        targets=("out",),
        default_target="out",
        suppress_implicit_output=False,
        effective_output_options={"out": DatasetOptionSyntax()},
        diagnostics=(invalid_plan_diagnostic,),
    )
    context = _typed_context(
        "data out; set in; result=apply('must_not_resolve', amount); run;",
        payload=_ArrowExportProbe(events, pa.table({"amount": [2]})),
        output_plan=invalid_plan,
    )
    with patch.object(HostFunctionResolver, "resolve", autospec=True) as resolve:
        payload, diagnostics = TypedRustArrowIOBridge().build_payload(context)

    assert payload is None
    assert diagnostics == [invalid_plan_diagnostic]
    resolve.assert_not_called()
    assert events == []

    blocked_context = replace(
        context,
        execution_readiness={
            "decision": "blocked",
            "blocking_reason_codes": ("ROW_LOOP_UNSUPPORTED_PLAN",),
        },
    )
    with patch.object(HostFunctionResolver, "resolve", autospec=True) as resolve:
        payload, diagnostics = TypedRustArrowIOBridge().build_payload(blocked_context)

    assert payload is None
    assert len(diagnostics) == 1
    assert diagnostics[0].code == "ROW_LOOP_UNSUPPORTED_PLAN"
    assert diagnostics[0].message == "Execution plan readiness is blocked for row-loop execution."
    resolve.assert_not_called()
    assert events == []


def test_typed_bridge_validates_consumed_input_options_before_resolve_or_export() -> None:
    events: list[str] = []
    context = _typed_context(
        "data out; set in(firstobs=2); result=apply('must_not_resolve', amount); run;",
        payload=_ArrowExportProbe(events, pa.table({"amount": [2]})),
    )
    source_statement = next(
        statement
        for statement in context.block.statements
        if statement.kind == "SET"
    )
    assert isinstance(source_statement.payload, SourceStatementSyntax)
    prepared_inputs = dict(context.resolved_inputs)
    prepared_inputs["in"] = replace(
        prepared_inputs["in"],
        preparation=PreparationProvenance(kind="prepared_set"),
    )
    prepared_context = replace(
        context,
        resolved_inputs=prepared_inputs,
        preparation_kind="prepared_set",
    )

    with patch.object(HostFunctionResolver, "resolve", autospec=True) as resolve:
        payload, diagnostics = TypedRustArrowIOBridge().build_payload(prepared_context)

    assert payload is None
    assert len(diagnostics) == 1
    assert diagnostics[0].code == "RUNTIME_PREPARED_INPUT_OPTION_REMAINING"
    assert diagnostics[0].severity == "error"
    assert "still carries source-local dataset options" in diagnostics[0].message
    resolve.assert_not_called()
    assert events == []


def test_typed_bridge_structural_prepared_set_collapse_preserves_outer_facts() -> None:
    source_text = "data out; set in in; value=apply('host', amount); run;"
    host = lambda value: value
    context = _typed_context(source_text, host_functions={"host": host})
    original = next(
        statement for statement in context.block.statements if statement.kind == "SET"
    )
    source_payload = original.payload
    assert isinstance(source_payload, SourceStatementSyntax)
    first_reference, second_reference = source_payload.refs
    refs = (
        replace(
            first_reference,
            options=DatasetOptionSyntax(label="Python-owned label"),
        ),
        replace(
            second_reference,
            options=DatasetOptionSyntax(label="Python-owned label"),
        ),
    )
    carried_reference = ApplyReferenceSyntax(
        target="carried",
        call_span=next(
            statement
            for statement in context.block.statements
            if statement.kind == "ASSIGN"
        ).apply_references[0].call_span,
        owner_span=original.span,
    )
    original = replace(
        original,
        payload=replace(source_payload, refs=refs),
        origin=original.span,
        apply_references=(carried_reference,),
    )
    block = replace(
        context.block,
        statements=tuple(
            original if statement.kind == "SET" else statement
            for statement in context.block.statements
        ),
    )
    prepared_inputs = dict(context.resolved_inputs)
    prepared_inputs["in"] = replace(
        prepared_inputs["in"],
        preparation=PreparationProvenance(kind="prepared_set"),
    )
    context = replace(
        context,
        block=block,
        resolved_inputs=prepared_inputs,
        preparation_kind="prepared_set",
        request=replace(
            context.request,
            host_function_context=HostFunctionContext(
                explicit={"host": host, "carried": host}
            ),
        ),
    )

    payload, diagnostics = TypedRustArrowIOBridge().build_payload(context)

    assert diagnostics == []
    assert payload is not None
    serialized = next(
        statement
        for statement in payload.native_program["statements"]
        if statement["kind"] == "SET"
    )
    assert serialized["span"] == {
        "start": original.span.start,
        "end": original.span.end,
        "line": original.span.line,
        "column": original.span.column,
        "end_line": original.span.end_line,
        "end_column": original.span.end_column,
        "source_id": original.span.source_id,
    }
    assert serialized["origin"] == serialized["span"]
    assert [reference["target"] for reference in serialized["apply_references"]] == [
        "carried"
    ]
    source_refs = serialized["payload"]
    assert source_refs["tag"] == "source"
    assert [reference["name"] for reference in source_refs["refs"]] == ["in"]
    assert source_refs["refs"][0]["span"]["start"] == first_reference.span.start
    assert source_refs["refs"][0]["options"] == {
        "in_var": None,
        "keep_vars": [],
        "drop_vars": [],
        "where_expr": None,
        "rename_map": {},
        "firstobs": None,
        "obs": None,
    }
    assert len(source_payload.refs) == 2


def test_typed_bridge_accepts_prepared_merge_after_input_owner_collapses_it_to_set() -> None:
    source_text = "data out; merge in; value=1; run;"
    context = _typed_context(source_text)
    prepared_merge = DataSetRef(
        kind="arrow_table",
        location="dataset://in",
        payload=pa.table({"amount": [2]}),
        preparation=PreparationProvenance(kind="prepared_merge"),
    )
    statements = tuple(
        replace(statement, kind="SET")
        if statement.kind == "MERGE"
        else statement
        for statement in context.block.statements
    )
    collapsed_block = replace(context.block, statements=statements)
    prepared_context = replace(
        context,
        block=collapsed_block,
        resolved_inputs={"in": prepared_merge},
        preparation_kind="prepared_merge",
    )

    payload, diagnostics = TypedRustArrowIOBridge().build_payload(prepared_context)

    assert diagnostics == []
    assert payload is not None
    assert payload.preparation_kind == "prepared_merge"
    source = next(
        statement
        for statement in payload.native_program["statements"]
        if statement["kind"] == "SET"
    )
    assert [reference["name"] for reference in source["payload"]["refs"]] == ["in"]

    # An unconsumed MERGE is still rejected by the existing prepared-carrier
    # contract; the bridge does not reinterpret it as a collapsed SET.
    uncollapsed = replace(
        prepared_context,
        block=replace(
            prepared_context.block,
            statements=tuple(
                replace(statement, kind="MERGE")
                if statement.kind == "SET"
                else statement
                for statement in prepared_context.block.statements
            ),
        ),
    )
    rejected_payload, diagnostics = TypedRustArrowIOBridge().build_payload(uncollapsed)
    assert rejected_payload is None
    assert len(diagnostics) == 1
    assert diagnostics[0].code == "RUNTIME_PREPARED_INPUT_CONTRACT_INVALID"


def test_typed_bridge_forwards_final_output_plan_without_replanning() -> None:
    output_plan = TypedOutputTargetPlan(
        targets=("Out", "out", "not_registered"),
        default_target="explicit-default",
        suppress_implicit_output=True,
        effective_output_options={
            "Out": DatasetOptionSyntax(keep_vars=("A",)),
            "not_registered": DatasetOptionSyntax(),
        },
    )
    context = _typed_context(
        "data out; set in; run;",
        output_plan=output_plan,
    )
    context = replace(
        context,
        request=replace(
            context.request,
            options={"format_catalog_payload": {"FMT": {"kind": "format"}}},
        ),
    )

    payload, diagnostics = TypedRustArrowIOBridge().build_payload(context)

    assert diagnostics == []
    assert payload is not None
    assert payload.native_program["output_targets"] == [
        "Out",
        "out",
        "not_registered",
    ]
    assert payload.native_program["default_target"] == "explicit-default"
    assert payload.native_program["suppress_implicit_output"] is True
    assert list(payload.native_program["effective_output_options"]) == [
        "Out",
        "not_registered",
    ]
    assert payload.native_program["effective_output_options"]["Out"][
        "keep_vars"
    ] == ["A"]
    assert payload.native_program["effective_output_options"]["not_registered"][
        "keep_vars"
    ] == []
    assert payload.format_catalog_payload == {"FMT": {"kind": "format"}}


def test_typed_bridge_rejects_malformed_wire_before_arrow_export() -> None:
    events: list[str] = []
    output_plan = TypedOutputTargetPlan(
        targets=("out",),
        default_target="unregistered-default",
        suppress_implicit_output=False,
        effective_output_options={"orphan": DatasetOptionSyntax()},
    )
    context = _typed_context(
        "data out; set in; run;",
        payload=_ArrowExportProbe(events, pa.table({"amount": [2]})),
        output_plan=output_plan,
    )

    real_resolve = HostFunctionResolver.resolve

    def resolve_successfully(resolver, targets, host_context):
        return real_resolve(resolver, targets, host_context)

    with patch.object(
        HostFunctionResolver,
        "resolve",
        autospec=True,
        side_effect=resolve_successfully,
    ) as resolve:
        with pytest.raises(NativePayloadValidationError) as error:
            TypedRustArrowIOBridge().build_payload(context)

    assert error.value.reason == "field"
    assert error.value.path == "effective_output_options"
    resolve.assert_called_once()
    assert events == []


def test_typed_bridge_preserves_apply_resolution_priority_over_wire_rejection() -> None:
    events: list[str] = []
    output_plan = TypedOutputTargetPlan(
        targets=("out",),
        default_target="unregistered-default",
        suppress_implicit_output=False,
        effective_output_options={"orphan": DatasetOptionSyntax()},
    )
    context = _typed_context(
        "data out; set in; result=apply('missing', amount); run;",
        payload=_ArrowExportProbe(events, pa.table({"amount": [2]})),
        output_plan=output_plan,
    )

    payload, diagnostics = TypedRustArrowIOBridge().build_payload(context)

    assert payload is None
    assert len(diagnostics) == 1
    assert diagnostics[0].code == "RUNTIME_APPLY_TARGET_UNRESOLVED"
    assert diagnostics[0].stage == "resolve"
    assert events == []


def test_typed_bridge_does_not_import_parser_or_recover_inventory_from_display() -> None:
    source = inspect.getsource(typed_bridge)

    for forbidden in (
        "ParserService",
        "parse_program",
        "ast_to_dict",
        "statement_to_dict",
        "_extract_apply_targets",
        "py_ast",
        "re.compile",
    ):
        assert forbidden not in source
    assert "source.text" not in source


def test_typed_native_route_has_no_legacy_text_payload_or_power_rewrite() -> None:
    executor_source = inspect.getsource(rust_executor_module)
    coordinator_source = inspect.getsource(coordinator_module)

    for removed_name in (
        "execute_legacy_block",
        "execute_typed_block",
        "_serialize_payload",
        "_normalize_assign_expression_for_rust",
        "_rewrite_power_expression_in_assign",
        "_POWER_PATTERN",
    ):
        assert not hasattr(rust_executor_module.RustNativeBlockExecutor, removed_name)
        assert removed_name not in executor_source
    coordinator_tree = ast.parse(coordinator_source)
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.endswith(".rust_bridge")
        for node in ast.walk(coordinator_tree)
    )
    assert not any(
        isinstance(node, ast.Name) and node.id == "RustArrowIOBridge"
        for node in ast.walk(coordinator_tree)
    )
    for name in (
        "RuntimeExecutionContext",
        "RustExecutionPayload",
        "RuntimeExecuteFn",
    ):
        assert not hasattr(backend_contracts, name)
    for name in ("RuntimeHandoffRequest", "RuntimeHandoffResult"):
        assert not hasattr(execution_contracts, name)
    assert not hasattr(RustRuntimeBackend, "execute")
    assert not hasattr(ExecutionRuntimeHandoffService, "execute")
