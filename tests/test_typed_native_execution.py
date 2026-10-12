from __future__ import annotations

import json
from typing import Any, Literal, TypedDict, cast

import pyarrow as pa
import pytest

from limulus.backend_integration.contracts import TypedRustExecutionContext
from limulus.backend_integration.typed_bridge import TypedRustArrowIOBridge
from limulus.execution import DataStepExecutor
from limulus.execution.contracts import (
    TypedOutputFinalizeRequest,
    TypedOutputPlanRequest,
    TypedOutputShapingRequest,
)
from limulus.execution.output_handoff import ExecutionOutputHandoffService
from limulus.execution.planning import ExecutionPlanService
from limulus.execution.typed_output_planning import TypedOutputTargetPlan
from limulus.host_functions import HostFunctionContext
from limulus.models import (
    DataSetRef,
    ExecuteRequest,
    PreparationProvenance,
    SourceMetadataLineage,
    SourceMetadataSnapshot,
)
from limulus.native_bridge import load_native_module
from limulus.syntax import parse_program
from limulus.syntax.models import DatasetOptionSyntax


class TypedNativePayload(TypedDict):
    native_program_json: str
    input_streams: dict[str, Any]
    preparation_kind: Literal["none", "prepared_set", "prepared_merge"]
    execution_plan: dict[str, Any]
    apply_registry: dict[str, Any]
    format_catalog_payload: dict[str, Any]


def test_native_module_registers_typed_public_execute_block() -> None:
    native, error = load_native_module()
    assert native is not None, error
    assert callable(getattr(native, "execute_block", None))
    assert not hasattr(native, "execute_legacy_block")
    assert not hasattr(native, "execute_typed_block")


def _execution_plan_for_typed_input(
    dsl: str,
    table: pa.Table,
    output_targets: tuple[str, ...] = ("out",),
    host_functions: dict[str, object] | None = None,
) -> object:
    executor = DataStepExecutor(runtime_backend="rust")
    request = ExecuteRequest(
        dsl_text=dsl,
        inputs={"inp": DataSetRef(kind="arrow_table", location="dataset://inp", payload=table)},
        output_targets=output_targets,
        host_function_context=HostFunctionContext(explicit=host_functions or {}),
    )
    captured = []
    native_executor = executor._rust_backend._executor
    assert native_executor is not None
    execute = native_executor.execute_block

    def capture(context):
        captured.append(context)
        return execute(context)

    native_executor.execute_block = capture
    executor.execute(request)
    assert len(captured) == 1
    return captured[0].execution_plan


def _typed_payload(
    dsl: str,
    table: pa.Table,
    *,
    suppress_implicit_output: bool = False,
    host_functions: dict[str, object] | None = None,
    output_targets: tuple[str, ...] = ("out",),
    default_target: str | None = "out",
) -> dict[str, object]:
    parsed = parse_program(dsl)
    assert parsed.program is not None
    assert len(parsed.program.blocks) == 1
    context = TypedRustExecutionContext(
        source=parsed.program.source,
        block=parsed.program.blocks[0],
        request=ExecuteRequest(
            dsl_text=dsl,
            host_function_context=HostFunctionContext(explicit=host_functions or {}),
        ),
        resolved_inputs={
            "inp": DataSetRef(kind="arrow_table", location="dataset://inp", payload=table)
        },
        preparation_kind="none",
        output_plan=TypedOutputTargetPlan(
            targets=output_targets,
            default_target=default_target,
            suppress_implicit_output=suppress_implicit_output,
            effective_output_options={target: DatasetOptionSyntax() for target in output_targets},
        ),
    )
    native_payload, diagnostics = TypedRustArrowIOBridge().build_payload(context)
    assert diagnostics == []
    assert native_payload is not None
    return {
        "native_program_json": json.dumps(native_payload.native_program),
        "input_streams": dict(native_payload.input_streams),
        "preparation_kind": native_payload.preparation_kind,
        "execution_plan": _execution_plan_for_typed_input(
            dsl, table, output_targets, host_functions
        ),
        "apply_registry": dict(native_payload.apply_registry),
        "format_catalog_payload": dict(native_payload.format_catalog_payload),
    }


def _typed_payload_for_inputs(
    dsl: str,
    resolved_inputs: dict[str, DataSetRef],
    *,
    preparation_kind: Literal["none", "prepared_set", "prepared_merge"] = "none",
    output_targets: tuple[str, ...] = ("out",),
) -> TypedNativePayload:
    """Build a typed native payload for focused multi-source ingress tests."""

    parsed = parse_program(dsl)
    assert parsed.program is not None
    assert not parsed.has_errors, parsed.diagnostics
    assert len(parsed.program.blocks) == 1
    block = parsed.program.blocks[0]
    output_plan = ExecutionOutputHandoffService().plan_typed(
        TypedOutputPlanRequest(
            block=block,
            source=parsed.program.source,
            explicit_output_targets=output_targets,
        )
    )
    assert not output_plan.diagnostics
    context = TypedRustExecutionContext(
        source=parsed.program.source,
        block=block,
        request=ExecuteRequest(dsl_text=dsl),
        resolved_inputs=resolved_inputs,
        preparation_kind=preparation_kind,
        output_plan=output_plan,
    )
    native_payload, diagnostics = TypedRustArrowIOBridge().build_payload(context)
    assert diagnostics == []
    assert native_payload is not None
    return {
        "native_program_json": json.dumps(native_payload.native_program),
        "input_streams": dict(native_payload.input_streams),
        "preparation_kind": native_payload.preparation_kind,
        "execution_plan": {},
        "apply_registry": dict(native_payload.apply_registry),
        "format_catalog_payload": dict(native_payload.format_catalog_payload),
    }


def _typed_payload_with_planned_outputs(
    dsl: str,
    table: pa.Table,
    *,
    output_targets: tuple[str, ...] = (),
    host_functions: dict[str, object] | None = None,
) -> dict[str, object]:
    parsed = parse_program(dsl)
    assert parsed.program is not None
    assert not parsed.has_errors, parsed.diagnostics
    assert len(parsed.program.blocks) == 1
    block = parsed.program.blocks[0]
    service = ExecutionOutputHandoffService()
    output_plan = service.plan_typed(
        TypedOutputPlanRequest(
            block=block,
            source=parsed.program.source,
            explicit_output_targets=output_targets,
        )
    )
    assert not output_plan.diagnostics
    resolved_input = DataSetRef(
        kind="arrow_table", location="dataset://inp", payload=table
    )
    shaping = service.plan_typed_shaping(
        TypedOutputShapingRequest(
            block=block,
            source=parsed.program.source,
            output_target_plan=output_plan,
            resolved_inputs={"inp": resolved_input},
            helper_plan=ExecutionPlanService().plan_typed_helpers(block),
        )
    )
    assert not shaping.diagnostics
    assert shaping.output_plan is not None
    context = TypedRustExecutionContext(
        source=parsed.program.source,
        block=block,
        request=ExecuteRequest(
            dsl_text=dsl,
            host_function_context=HostFunctionContext(explicit=host_functions or {}),
        ),
        resolved_inputs={
            "inp": resolved_input
        },
        preparation_kind="none",
        output_plan=output_plan,
    )
    native_payload, diagnostics = TypedRustArrowIOBridge().build_payload(context)
    assert diagnostics == []
    assert native_payload is not None
    return {
        "native_program_json": json.dumps(native_payload.native_program),
        "input_streams": dict(native_payload.input_streams),
        "preparation_kind": native_payload.preparation_kind,
        "execution_plan": {
            **_execution_plan_for_typed_input(
                dsl, table, output_targets or output_plan.targets
            ),
            "output_handoff_plan": shaping.output_plan,
        },
        "apply_registry": dict(native_payload.apply_registry),
        "format_catalog_payload": dict(native_payload.format_catalog_payload),
    }


def _output_table(native_result: dict[str, object], target: str = "out") -> pa.Table:
    importer = getattr(
        pa.RecordBatchReader,
        "_import_from_c_capsule",
        pa.RecordBatchReader._import_from_c,
    )
    return importer(native_result["output_streams"][target]).read_all()


def _finalize_typed_native_outputs(
    dsl: str,
    source_table: pa.Table,
    runtime_outputs: dict[str, pa.Table],
) -> dict[str, pa.Table]:
    parsed = parse_program(dsl)
    assert parsed.program is not None
    block = parsed.program.blocks[0]
    service = ExecutionOutputHandoffService()
    target_plan = service.plan_typed(
        TypedOutputPlanRequest(block=block, source=parsed.program.source)
    )
    assert not target_plan.diagnostics
    resolved_input = DataSetRef(
        kind="arrow_table", location="dataset://inp", payload=source_table
    )
    helper_plan = ExecutionPlanService().plan_typed_helpers(block)
    shaping = service.plan_typed_shaping(
        TypedOutputShapingRequest(
            block=block,
            source=parsed.program.source,
            output_target_plan=target_plan,
            resolved_inputs={"inp": resolved_input},
            helper_plan=helper_plan,
        )
    )
    assert not shaping.diagnostics
    assert shaping.output_plan is not None
    finalized = service.finalize_typed(
        TypedOutputFinalizeRequest(
            block=block,
            output_target_plan=target_plan,
            output_plan=shaping.output_plan,
            resolved_inputs={"inp": resolved_input},
            outputs={
                target: DataSetRef(
                    kind="arrow_table",
                    location=f"dataset://{target}",
                    payload=table,
                )
                for target, table in runtime_outputs.items()
            },
            helper_plan=helper_plan,
            source_metadata_lineage=SourceMetadataLineage(
                ordered_sources=(
                    SourceMetadataSnapshot(
                        source_ordinal=0,
                        source_name="inp",
                        schema=source_table.schema,
                    ),
                )
            ),
        )
    )
    assert not finalized.diagnostics
    return {
        target: output.payload for target, output in finalized.outputs.items()
    }


def test_typed_block_executes_structural_assign_and_sum_with_shared_arrow_loop() -> None:
    native, error = load_native_module()
    assert native is not None, error
    cases = (
        (
            "data out; set inp; plus_two = amount + 2; run;",
            "plus_two",
            [4, 6],
        ),
        (
            "data out; set inp; total + amount; run;",
            "total",
            [2, 6],
        ),
    )
    for dsl, column, expected in cases:
        result = native.execute_block(
            _typed_payload(dsl, pa.table({"amount": [2, 4]}))
        )
        assert result["diagnostics"] == []
        assert _output_table(result).column(column).to_pylist() == expected


def test_typed_if_else_executes_inline_assignment_by_reachability() -> None:
    native, error = load_native_module()
    assert native is not None, error
    dsl = "data out; set inp; if flag then value = 1; else value = 0; run;"

    result = native.execute_block(
        _typed_payload(dsl, pa.table({"flag": [True, False]}))
    )

    assert result["diagnostics"] == []
    assert _output_table(result).column("value").to_pylist() == [1, 0]


def test_typed_else_if_preserves_condition_order_and_selected_action() -> None:
    native, error = load_native_module()
    assert native is not None, error
    dsl = (
        "data out; set inp; if first then value = 1; "
        "else if amount @@ 2 then foo; else value = 3; run;"
    )
    selected = native.execute_block(
        _typed_payload(dsl, pa.table({"first": [True], "amount": [0]}))
    )
    condition_fault = native.execute_block(
        _typed_payload(dsl, pa.table({"first": [False], "amount": [0]}))
    )

    assert selected["diagnostics"] == []
    assert _output_table(selected).column("value").to_pylist() == [1]
    assert condition_fault["diagnostics"][0]["code"] == "RUNTIME_EXPRESSION_EVALUATION_ERROR"
    assert "invalid IF statement: else if amount @@ 2 then foo" in condition_fault["diagnostics"][0]["message"]
    assert "unsupported inline action: foo" not in condition_fault["diagnostics"][0]["message"]


def test_typed_if_keeps_power_concat_and_quoted_action_values() -> None:
    native, error = load_native_module()
    assert native is not None, error
    dsl = r"data out; set inp; if amount ** 2 ** 3 = 256 then value = 'C:\tmp' || 'A'; else value = 'x'; run;"

    result = native.execute_block(
        _typed_payload(dsl, pa.table({"amount": [2, 3]}))
    )

    assert result["diagnostics"] == []
    assert _output_table(result).column("value").to_pylist() == [r"C:\tmpA", "x"]


def test_typed_if_defers_unsupported_actions_until_reached_and_skips_after_stop_delete() -> None:
    native, error = load_native_module()
    assert native is not None, error

    def execute(dsl: str, flag: list[bool]):
        return native.execute_block(
            _typed_payload(dsl, pa.table({"flag": flag}))
        )

    unopened = execute("data out; set inp; if flag then foo; run;", [False])
    empty = execute(
        "data out; set inp; if 1 then foo; run;",
        [],
    )
    reached = execute("data out; set inp; if flag then foo; run;", [True])
    stopped = execute(
        "data out; set inp; stop; if 1 then foo; run;",
        [True, True],
    )
    deleted = execute(
        "data out; set inp; delete; if 1 then foo; run;",
        [True, True],
    )
    stopped_in_branch = execute(
        "data out; set inp; if flag then stop; if 1 then foo; run;",
        [True, True],
    )
    deleted_in_branch = execute(
        "data out; set inp; if flag then delete; if 1 then foo; run;",
        [True, True],
    )
    stopped_in_then_do = execute(
        "data out; set inp; if flag then do; stop; end; if 1 then foo; run;",
        [True, True],
    )

    assert unopened["diagnostics"] == []
    assert empty["diagnostics"] == []
    assert reached["diagnostics"][0]["code"] == "RUNTIME_EXPRESSION_EVALUATION_ERROR"
    assert "unsupported inline action: foo" in reached["diagnostics"][0]["message"]
    assert stopped["diagnostics"] == []
    assert _output_table(stopped).num_rows == 1
    assert deleted["diagnostics"] == []
    assert _output_table(deleted).num_rows == 0
    assert stopped_in_branch["diagnostics"] == []
    assert _output_table(stopped_in_branch).num_rows == 1
    assert deleted_in_branch["diagnostics"] == []
    assert _output_table(deleted_in_branch).num_rows == 0
    assert stopped_in_then_do["diagnostics"] == []
    assert _output_table(stopped_in_then_do).num_rows == 1


def test_typed_inline_output_routes_only_selected_rows_and_literal_does_not_suppress() -> None:
    native, error = load_native_module()
    assert native is not None, error

    explicit = native.execute_block(
        _typed_payload(
            "data out; set inp; if flag then output out; run;",
            pa.table({"flag": [True, False, True]}),
            suppress_implicit_output=True,
        )
    )
    literal = native.execute_block(
        _typed_payload(
            "data out; set inp; if flag then value = 'output'; run;",
            pa.table({"flag": [True, False]}),
        )
    )

    assert explicit["diagnostics"] == []
    assert _output_table(explicit).column("flag").to_pylist() == [True, True]
    assert literal["diagnostics"] == []
    literal_table = _output_table(literal)
    assert literal_table.num_rows == 2
    assert literal_table.column("value").to_pylist() == ["output", None]


def test_typed_inline_output_normalizes_work_alias_to_planned_target() -> None:
    native, error = load_native_module()
    assert native is not None, error

    result = native.execute_block(
        _typed_payload(
            "data out; set inp; if flag then output WORK.out; run;",
            pa.table({"flag": [True, False, True]}),
            suppress_implicit_output=True,
        )
    )

    assert result["diagnostics"] == []
    assert _output_table(result).column("flag").to_pylist() == [True, True]


def test_typed_inline_output_uses_plan_unicode_dataset_key_normalization() -> None:
    native, error = load_native_module()
    assert native is not None, error

    result = native.execute_block(
        _typed_payload(
            "data strasse; set inp; if flag then output work.straße; run;",
            pa.table({"flag": [True, False, True]}),
            suppress_implicit_output=True,
            output_targets=("strasse",),
            default_target="strasse",
        )
    )

    assert result["diagnostics"] == []
    assert _output_table(result, "strasse").column("flag").to_pylist() == [True, True]


def test_typed_inline_failure_does_not_partially_evaluate_concat_apply() -> None:
    native, error = load_native_module()
    assert native is not None, error
    invoked: list[object] = []

    def custom(value: object) -> object:
        invoked.append(value)
        return value

    dsl = "data out; set inp; if flag then value = amount || apply('custom', amount) ||; run;"
    result = native.execute_block(
        _typed_payload(
            dsl,
            pa.table({"amount": [3], "flag": [True]}),
            host_functions={"custom": custom},
        )
    )

    assert result["diagnostics"][0]["code"] == "RUNTIME_EXPRESSION_EVALUATION_ERROR"
    assert invoked == []


def test_typed_block_rejects_bad_program_without_legacy_ast_fallback() -> None:
    native, error = load_native_module()
    assert native is not None, error
    payload = _typed_payload(
        "data out; set inp; value = amount + 1; run;",
        pa.table({"amount": [1]}),
    )
    payload["native_program_json"] = json.dumps({"version": 99})
    payload["ast_json"] = '{"statements":[]}'

    result = native.execute_block(payload)

    assert result["diagnostics"][0]["code"] == "PARSE_RUST_AST_DESERIALIZE_FAILED"
    assert "output_streams" not in result


def test_typed_block_uses_saved_default_target_without_reselecting_first_target() -> None:
    native, error = load_native_module()
    assert native is not None, error
    payload = _typed_payload(
        "data out; set inp; value = amount + 1; run;",
        pa.table({"amount": [1]}),
    )
    program = json.loads(payload["native_program_json"])
    program["default_target"] = None
    payload["native_program_json"] = json.dumps(program)

    result = native.execute_block(payload)

    assert result["diagnostics"][0]["code"] == "RUNTIME_EXPRESSION_EVALUATION_ERROR"
    assert "No default output target is declared" in result["diagnostics"][0]["message"]
    assert "output_streams" not in result


def test_real_native_runs_current_typed_payload_with_explicit_registry() -> None:
    native, error = load_native_module()
    assert native is not None, error
    dsl = "data out; set inp; note_len = apply('len', note); output out; run;"
    payload = _typed_payload(
        dsl,
        pa.table({"note": ["abc"]}),
        suppress_implicit_output=True,
        host_functions={"len": len},
    )

    assert payload["apply_registry"] == {"len": len}
    result = native.execute_block(payload)

    assert result["diagnostics"] == []
    assert _output_table(result).column("note_len").to_pylist() == [3]


def test_empty_explicit_registry_rejects_apply_without_builtin_fallback() -> None:
    native, error = load_native_module()
    assert native is not None, error
    dsl = "data out; set inp; note_len = apply('len', note); output out; run;"
    table = pa.table({"note": ["abc"]})
    positive = native.execute_block(
        _typed_payload(
            dsl, table, suppress_implicit_output=True, host_functions={"len": len}
        )
    )
    negative_payload = _typed_payload(
        dsl, table, suppress_implicit_output=True, host_functions={"len": len}
    )
    negative_payload["apply_registry"] = {}
    negative = native.execute_block(negative_payload)

    assert positive["diagnostics"] == []
    assert _output_table(positive).column("note_len").to_pylist() == [3]
    assert len(negative["diagnostics"]) == 1
    assert negative["diagnostics"][0]["code"] == "RUNTIME_EXPRESSION_EVALUATION_ERROR"
    assert "Unknown apply function" in negative["diagnostics"][0]["message"]
    assert "len" in negative["diagnostics"][0]["message"]
    assert "output_streams" not in negative


@pytest.mark.parametrize(
    ("plan_field", "invalid_value"),
    (
        ("strategy", "columnar"),
        ("where_mode", "row_filter"),
        ("rewrite_mode", "planner_owned"),
    ),
    ids=("invalid-strategy", "invalid-where-mode", "invalid-rewrite-mode"),
)
def test_row_loop_plan_single_invalid_mode_is_rejected_with_plan_location(
    plan_field: str,
    invalid_value: str,
) -> None:
    native, error = load_native_module()
    assert native is not None, error
    dsl = "data out; set inp; doubled = amount * 2; output out; run;"
    table = pa.table({"amount": [4]})
    control = native.execute_block(
        _typed_payload(dsl, table, suppress_implicit_output=True)
    )
    payload = cast(
        TypedNativePayload,
        _typed_payload(dsl, table, suppress_implicit_output=True),
    )
    execution_plan = dict(payload["execution_plan"])
    row_loop_plan = dict(execution_plan["row_loop_plan"])
    row_loop_plan[plan_field] = invalid_value
    execution_plan["row_loop_plan"] = row_loop_plan
    payload["execution_plan"] = execution_plan

    result = native.execute_block(payload)

    assert control["diagnostics"] == []
    assert _output_table(control).column("doubled").to_pylist() == [8]
    assert len(result["diagnostics"]) == 1
    assert result["diagnostics"][0]["code"] == "ROW_LOOP_UNSUPPORTED_PLAN"
    assert result["diagnostics"][0]["location"] == "row_loop_plan"
    assert "output_streams" not in result


def test_raw_none_set_with_all_streams_succeeds_on_real_native() -> None:
    native, error = load_native_module()
    assert native is not None, error
    dsl = "data out; set a b; output out; run;"
    inputs = {
        "a": DataSetRef(kind="arrow_table", location="dataset://a", payload=pa.table({"amount": [4]})),
        "b": DataSetRef(kind="arrow_table", location="dataset://b", payload=pa.table({"amount": [6]})),
    }
    payload = _typed_payload_for_inputs(dsl, inputs)
    program = json.loads(payload["native_program_json"])
    assert next(statement for statement in program["statements"] if statement["kind"] == "SET")["payload"]["refs"]
    assert tuple(sorted(payload["input_streams"])) == ("a", "b")

    result = native.execute_block(payload)

    assert result["diagnostics"] == []
    assert _output_table(result).column("amount").to_pylist() == [4, 6]


def test_raw_prepared_set_with_omitted_suffix_stream_succeeds_on_real_native() -> None:
    native, error = load_native_module()
    assert native is not None, error
    dsl = "data out; set a b; output out; run;"
    combined = pa.concat_tables(
        [pa.table({"amount": [4]}), pa.table({"amount": [6]})]
    )
    inputs = {
        "a": DataSetRef(
            kind="arrow_table",
            location="dataset://a",
            payload=combined,
            preparation=PreparationProvenance(kind="prepared_set"),
        )
    }
    payload = _typed_payload_for_inputs(
        dsl, inputs, preparation_kind="prepared_set"
    )
    program = json.loads(payload["native_program_json"])
    source = next(statement for statement in program["statements"] if statement["kind"] == "SET")
    assert [ref["name"] for ref in source["payload"]["refs"]] == ["a"]
    assert tuple(payload["input_streams"]) == ("a",)

    result = native.execute_block(payload)

    assert result["diagnostics"] == []
    assert _output_table(result).column("amount").to_pylist() == [4, 6]


@pytest.mark.parametrize(
    ("fixture_kind", "mutation", "missing_name"),
    (
        ("none", "drop_stream_b", "b"),
        ("prepared_set", "drop_stream_a", "a"),
        ("none", "prepared_merge_drop_b", "b"),
    ),
    ids=("none-missing-b", "prepared-set-missing-a", "prepared-merge-missing-b"),
)
def test_raw_set_missing_stream_is_rejected_with_dataset_location(
    fixture_kind: str,
    mutation: str,
    missing_name: str,
) -> None:
    native, error = load_native_module()
    assert native is not None, error
    dsl = "data out; set a b; output out; run;"
    source_a = pa.table({"amount": [4]})
    source_b = pa.table({"amount": [6]})
    if fixture_kind == "prepared_set":
        combined = pa.concat_tables([source_a, source_b])
        inputs = {
            "a": DataSetRef(
                kind="arrow_table",
                location="dataset://a",
                payload=combined,
                preparation=PreparationProvenance(kind="prepared_set"),
            )
        }
        payload = _typed_payload_for_inputs(
            dsl, inputs, preparation_kind="prepared_set"
        )
    else:
        payload = _typed_payload_for_inputs(
            dsl,
            {
                "a": DataSetRef(kind="arrow_table", location="dataset://a", payload=source_a),
                "b": DataSetRef(kind="arrow_table", location="dataset://b", payload=source_b),
            },
        )
    control_payload = _typed_payload_for_inputs(
        dsl,
        {
            "a": DataSetRef(kind="arrow_table", location="dataset://a", payload=source_a),
            "b": DataSetRef(kind="arrow_table", location="dataset://b", payload=source_b),
        },
    )
    control = native.execute_block(control_payload)
    if mutation == "drop_stream_b":
        payload["input_streams"].pop("b")
    elif mutation == "drop_stream_a":
        payload["input_streams"].pop("a")
    else:
        payload["preparation_kind"] = "prepared_merge"
        payload["input_streams"].pop("b")
    result = native.execute_block(payload)

    assert control["diagnostics"] == []
    assert _output_table(control).column("amount").to_pylist() == [4, 6]
    assert len(result["diagnostics"]) == 1
    assert result["diagnostics"][0]["code"] == "RUNTIME_SET_INPUT_STREAM_MISSING"
    assert result["diagnostics"][0]["location"] == f"dataset:{missing_name}"
    assert "output_streams" not in result


@pytest.mark.parametrize(
    ("mutation", "expected_error", "expected_match"),
    (
        ("missing_kind", ValueError, "payload.preparation_kind is required"),
        ("unknown_kind", ValueError, "payload.preparation_kind is invalid"),
        ("non_string_kind", TypeError, None),
    ),
    ids=("missing-kind", "unknown-kind", "non-string-kind"),
)
def test_raw_preparation_kind_boundary_raises_raw_native_error(
    mutation: str,
    expected_error: type[Exception],
    expected_match: str | None,
) -> None:
    native, error = load_native_module()
    assert native is not None, error
    payload = _typed_payload("data out; set inp; output out; run;", pa.table({"id": [1]}))
    if mutation == "missing_kind":
        payload.pop("preparation_kind")
    elif mutation == "unknown_kind":
        payload["preparation_kind"] = "prepared_unknown"
    else:
        payload["preparation_kind"] = None

    with pytest.raises(expected_error, match=expected_match):
        native.execute_block(payload)


def test_typed_assign_evaluates_only_on_reached_rows_and_keeps_fault_wrapper() -> None:
    native, error = load_native_module()
    assert native is not None, error
    dsl = "data out; set inp; result = amount @@ 2; run;"

    empty = native.execute_block(
        _typed_payload(dsl, pa.table({"amount": pa.array([], type=pa.int64())}))
    )
    assert empty["diagnostics"] == []
    assert _output_table(empty).column("result").to_pylist() == []

    reached = native.execute_block(
        _typed_payload(dsl, pa.table({"amount": [2]}))
    )
    assert reached["diagnostics"][0]["code"] == "RUNTIME_EXPRESSION_EVALUATION_ERROR"
    assert "assignment expression failed:" in reached["diagnostics"][0]["message"]
    assert "amount @@ 2" in reached["diagnostics"][0]["message"]
    assert "output_streams" not in reached


def test_typed_do_executes_the_first_statement_after_its_end() -> None:
    native, error = load_native_module()
    assert native is not None, error
    dsl = "data out; set inp; do i = 1 to 1; inside = 1; end; after = 2; run;"

    result = native.execute_block(_typed_payload(dsl, pa.table({"id": [1]})))

    assert result["diagnostics"] == []
    output = _output_table(result)
    assert output.column("inside").to_pylist() == [1.0]
    assert output.column("after").to_pylist() == [2.0]


def test_typed_bounded_do_preserves_nested_order_power_concat_and_flat_matching() -> None:
    native, error = load_native_module()
    assert native is not None, error

    power = native.execute_block(
        _typed_payload(
            "data out; set inp; do i = 2 ** 3 ** 2 to 2 ** 3 ** 2; output; end; run;",
            pa.table({"id": [1]}),
            suppress_implicit_output=True,
        )
    )
    concatenated_bound = native.execute_block(
        _typed_payload(
            "data out; set inp; do i = 1 || '2' to 12; output; end; run;",
            pa.table({"id": [1]}),
            suppress_implicit_output=True,
        )
    )
    nested = native.execute_block(
        _typed_payload(
            "data out; set inp; do i = 1 to 2; do j = 1 to 2; output; end; end; run;",
            pa.table({"id": [1]}),
            suppress_implicit_output=True,
        )
    )
    descending = native.execute_block(
        _typed_payload(
            "data out; set inp; do i = 3 to 1; output; end; run;",
            pa.table({"id": [1]}),
            suppress_implicit_output=True,
        )
    )
    nested_if = native.execute_block(
        _typed_payload(
            "data out; set inp; do i = 1 to 2; "
            "if flag then do; value = i; end; output; end; run;",
            pa.table({"flag": [True]}),
            suppress_implicit_output=True,
        )
    )
    stop_in_loop = native.execute_block(
        _typed_payload(
            "data out; set inp; do i = 1 to 3; if i = 2 then stop; output; end; run;",
            pa.table({"id": [1, 2]}),
            suppress_implicit_output=True,
        )
    )
    delete_in_loop = native.execute_block(
        _typed_payload(
            "data out; set inp; do i = 1 to 3; if i = 2 then delete; output; end; run;",
            pa.table({"id": [1]}),
            suppress_implicit_output=True,
        )
    )

    assert power["diagnostics"] == []
    assert _output_table(power).column("i").to_pylist() == [512.0]
    assert concatenated_bound["diagnostics"] == []
    assert _output_table(concatenated_bound).column("i").to_pylist() == [12.0]
    assert nested["diagnostics"] == []
    nested_table = _output_table(nested)
    assert nested_table.column("i").to_pylist() == [1, 1, 2, 2]
    assert nested_table.column("j").to_pylist() == [1, 2, 1, 2]
    assert descending["diagnostics"] == []
    assert _output_table(descending).num_rows == 0
    assert nested_if["diagnostics"] == []
    nested_if_table = _output_table(nested_if)
    assert nested_if_table.column("i").to_pylist() == [1.0, 2.0]
    assert nested_if_table.column("value").to_pylist() == [1.0, 2.0]
    assert stop_in_loop["diagnostics"] == []
    assert _output_table(stop_in_loop).column("i").to_pylist() == [1]
    assert delete_in_loop["diagnostics"] == []
    assert _output_table(delete_in_loop).column("i").to_pylist() == [1]


def test_typed_do_defers_invalid_specs_and_boundary_faults_until_reached() -> None:
    native, error = load_native_module()
    assert native is not None, error
    invalid = "data out; set inp; do i = 1 @@ 2 to 3; output; end; run;"

    empty = native.execute_block(
        _typed_payload(invalid, pa.table({"id": pa.array([], type=pa.int64())}))
    )
    reached = native.execute_block(
        _typed_payload(invalid, pa.table({"id": [1]}))
    )
    stopped = native.execute_block(
        _typed_payload(
            "data out; set inp; stop; do i = 1 @@ 2 to 3; output; end; run;",
            pa.table({"id": [1, 2]}),
        )
    )
    unreached_boundary = native.execute_block(
        _typed_payload(
            "data out; set inp; do i = unknown_bound(1) to 3; output; end; run;",
            pa.table({"id": pa.array([], type=pa.int64())}),
            suppress_implicit_output=True,
        )
    )
    reached_boundary = native.execute_block(
        _typed_payload(
            "data out; set inp; do i = unknown_bound(1) to 3; output; end; run;",
            pa.table({"id": [1]}),
            suppress_implicit_output=True,
        )
    )
    bad_concat_number = native.execute_block(
        _typed_payload(
            "data out; set inp; do i = 'x' || 'y' to 2; output; end; run;",
            pa.table({"id": [1]}),
        )
    )
    bad_text_bound = native.execute_block(
        _typed_payload(
            "data out; set inp; do i = 1 to 'not a number'; output; end; run;",
            pa.table({"id": [1]}),
        )
    )

    assert empty["diagnostics"] == []
    assert reached["diagnostics"][0]["code"] == "RUNTIME_EXPRESSION_EVALUATION_ERROR"
    assert "unsupported DO statement:" in reached["diagnostics"][0]["message"]
    assert stopped["diagnostics"] == []
    assert _output_table(stopped).num_rows == 1
    assert unreached_boundary["diagnostics"] == []
    assert reached_boundary["diagnostics"][0]["code"] == "RUNTIME_EXPRESSION_EVALUATION_ERROR"
    assert "unsupported function: unknown_bound" in reached_boundary["diagnostics"][0]["message"]
    assert bad_concat_number["diagnostics"][0]["code"] == "RUNTIME_EXPRESSION_EVALUATION_ERROR"
    assert "numeric value required" in bad_concat_number["diagnostics"][0]["message"]
    assert "xy" in bad_concat_number["diagnostics"][0]["message"]
    assert bad_text_bound["diagnostics"][0]["code"] == "RUNTIME_EXPRESSION_EVALUATION_ERROR"
    assert "numeric value required" in bad_text_bound["diagnostics"][0]["message"]


def test_typed_array_declaration_and_indexed_assignment_use_array_owner() -> None:
    native, error = load_native_module()
    assert native is not None, error
    dsl = (
        "data out; set inp; array vals[3] a b c; vals[2] = 7; "
        "picked = vals[2]; output; run;"
    )

    result = native.execute_block(
        _typed_payload(
            dsl,
            pa.table({"a": [1], "b": [2], "c": [3]}),
            suppress_implicit_output=True,
        )
    )

    assert result["diagnostics"] == []
    output = _output_table(result)
    assert output.column("b").to_pylist() == [7.0]
    assert output.column("picked").to_pylist() == [7.0]


def test_typed_sum_wire_rejects_array_element_target() -> None:
    native, error = load_native_module()
    assert native is not None, error
    dsl = (
        "data out; set inp; array vals[2] a b; "
        "vals[2] = 0; total + amount; output; run;"
    )
    payload = cast(
        TypedNativePayload,
        _typed_payload(
            dsl,
            pa.table({"a": [1], "b": [2], "amount": [3]}),
            suppress_implicit_output=True,
        ),
    )
    program = json.loads(payload["native_program_json"])
    array_target = next(
        statement
        for statement in program["statements"]
        if statement["kind"] == "ASSIGN"
        and statement["payload"]["target"]["tag"] == "array_element"
    )["payload"]["target"]
    array_sum = next(
        statement
        for statement in program["statements"]
        if statement["kind"] == "SUM"
    )
    array_sum["payload"]["target"] = json.loads(json.dumps(array_target))
    payload["native_program_json"] = json.dumps(program)

    result = native.execute_block(payload)

    assert result["diagnostics"][0]["code"] == "PARSE_RUST_AST_DESERIALIZE_FAILED"
    assert "target must use the name variant for SUM" in result["diagnostics"][0]["message"]
    assert "output_streams" not in result


def test_typed_array_star_dimension_and_index_calls_match_corpus() -> None:
    native, error = load_native_module()
    assert native is not None, error
    star_dsl = (
        "data out; set inp; array vars[*] a b c; "
        "dim_vars = dim(vars); x = vars[2]; output; run;"
    )
    star = native.execute_block(
        _typed_payload(
            star_dsl,
            pa.table({"a": [10], "b": [20], "c": [30]}),
            suppress_implicit_output=True,
        )
    )
    delimiter_dsl = (
        "data out; set inp; array chars $ c1 c2; array vals a b c; "
        "vals(1) = vals(1) * 10; vals[2] = vals[2] * 10; vals{3} = vals{3} * 10; "
        "arr_size = dim(vals); name2 = vname(vals[2]); "
        "selected = vals['0' || '2']; selected_plus = vals['0' + '2']; "
        "combined = chars[1] || chars[2]; "
        "added = chars[1] + chars[2]; output; run;"
    )
    delimiters = native.execute_block(
        _typed_payload(
            delimiter_dsl,
            pa.table({"a": [1], "b": [2], "c": [3], "c1": ["x"], "c2": ["y"]}),
            suppress_implicit_output=True,
        )
    )

    assert star["diagnostics"] == []
    star_output = _output_table(star)
    assert star_output.column("dim_vars").to_pylist() == [3.0]
    assert star_output.column("x").to_pylist() == [20]
    assert delimiters["diagnostics"] == []
    output = _output_table(delimiters)
    assert [output.column(name).to_pylist() for name in ("a", "b", "c")] == [
        [10.0],
        [20.0],
        [30.0],
    ]
    assert output.column("arr_size").to_pylist() == [3.0]
    assert output.column("name2").to_pylist() == ["b"]
    assert output.column("selected").to_pylist() == [20]
    assert output.column("selected_plus").to_pylist() == [20]
    assert output.column("combined").to_pylist() == ["xy"]
    assert output.column("added").to_pylist() == ["xy"]


def test_typed_array_faults_keep_reachability_and_index_rules() -> None:
    native, error = load_native_module()
    assert native is not None, error
    invalid_array = "data out; set inp; array malformed[1] 'abc; output; run;"
    empty = native.execute_block(
        _typed_payload(
            invalid_array,
            pa.table({"id": pa.array([], type=pa.int64())}),
            suppress_implicit_output=True,
        )
    )
    reached = native.execute_block(
        _typed_payload(
            invalid_array,
            pa.table({"id": [1]}),
            suppress_implicit_output=True,
        )
    )
    stopped = native.execute_block(
        _typed_payload(
            "data out; set inp; stop; array malformed[1] 'abc; output; run;",
            pa.table({"id": [1, 2]}),
            suppress_implicit_output=True,
        )
    )
    out_of_bounds = native.execute_block(
        _typed_payload(
            "data out; set inp; array vals[3] a b c; picked = vals[4]; output; run;",
            pa.table({"a": [1], "b": [2], "c": [3]}),
            suppress_implicit_output=True,
        )
    )
    nonnumeric_index = native.execute_block(
        _typed_payload(
            "data out; set inp; array vals[3] a b c; vals['x'] = 4; output; run;",
            pa.table({"a": [1], "b": [2], "c": [3]}),
            suppress_implicit_output=True,
        )
    )

    assert empty["diagnostics"] == []
    assert reached["diagnostics"][0]["code"] == "RUNTIME_EXPRESSION_EVALUATION_ERROR"
    assert "invalid ARRAY declaration" in reached["diagnostics"][0]["message"]
    assert stopped["diagnostics"] == []
    assert out_of_bounds["diagnostics"][0]["code"] == "RUNTIME_EXPRESSION_EVALUATION_ERROR"
    assert "array index out of bounds: vals[4]" in out_of_bounds["diagnostics"][0]["message"]
    assert nonnumeric_index["diagnostics"][0]["code"] == "RUNTIME_EXPRESSION_EVALUATION_ERROR"
    assert "numeric value required" in nonnumeric_index["diagnostics"][0]["message"]


def test_typed_name_lists_and_statement_where_use_structured_runtime_policy() -> None:
    native, error = load_native_module()
    assert native is not None, error
    schema = pa.schema(
        [
            pa.field("id", pa.int64(), metadata={b"label": b"Identifier"}),
            pa.field("amount", pa.int64(), metadata={b"label": b"Amount"}),
            pa.field("name", pa.string()),
            pa.field("secret", pa.string()),
        ],
        metadata={b"memlabel": b"Source rows"},
    )
    table = pa.Table.from_arrays(
        [
            pa.array([1, 1, 2, 2]),
            pa.array([2, 3, 2, 2]),
            pa.array(["A", "A", "A", "B"]),
            pa.array(["x", "y", "z", "w"]),
        ],
        schema=schema,
    )
    dsl = (
        "data out; set inp indsname=source_name end=eof; "
        "retain counter; by id; keep id amount source_name eof first.id last.id counter result; "
        "drop secret; rename amount=renamed; "
        "where amount ** 2 ** 3 = 256 and (name || '!') = 'A!'; "
        "counter + 1; result = name || '!'; output out; run;"
    )

    result = native.execute_block(
        _typed_payload_with_planned_outputs(dsl, table)
    )

    assert result["diagnostics"] == []
    output = _output_table(result)
    assert output.column_names == ["id", "renamed", "counter", "result"]
    assert output.column("id").to_pylist() == [1, 2]
    assert output.column("renamed").to_pylist() == [2, 2]
    assert output.column("counter").to_pylist() == [1.0, 2.0]
    assert output.column("result").to_pylist() == ["A!", "A!"]
    finalized = _finalize_typed_native_outputs(dsl, table, {"out": output})["out"]
    assert finalized.schema.metadata == {b"memlabel": b"Source rows"}
    assert finalized.schema.field("renamed").metadata == {b"label": b"Amount"}


def test_typed_output_refs_share_planned_options_routing_and_empty_schemas() -> None:
    native, error = load_native_module()
    assert native is not None, error
    schema = pa.schema(
        [
            pa.field("id", pa.int64(), metadata={b"label": b"Identifier"}),
            pa.field("amount", pa.int32(), metadata={b"label": b"Amount"}),
        ],
        metadata={b"memlabel": b"Per target"},
    )
    dsl = (
        "data alpha(keep=id rename=(id=key) label='Alpha' firstobs=2 obs=1) beta; "
        "set inp; output beta(keep=amount rename=(amount=value) label='Beta'); "
        "output alpha beta alpha; output; output beta; run;"
    )
    populated_source = pa.Table.from_arrays(
        [pa.array([1, 2]), pa.array([10, 20])], schema=schema
    )
    populated = native.execute_block(
        _typed_payload_with_planned_outputs(
            dsl,
            populated_source,
        )
    )
    empty_source = pa.Table.from_arrays(
        [pa.array([], type=pa.int64()), pa.array([], type=pa.int32())],
        schema=schema,
    )
    empty = native.execute_block(
        _typed_payload_with_planned_outputs(dsl, empty_source)
    )

    assert populated["diagnostics"] == []
    alpha = _output_table(populated, "alpha")
    beta = _output_table(populated, "beta")
    finalized = _finalize_typed_native_outputs(
        dsl, populated_source, {"alpha": alpha, "beta": beta}
    )
    assert alpha.column_names == ["key"]
    assert alpha.column("key").to_pylist() == [2]
    assert alpha.column("key").type == pa.int64()
    assert beta.column_names == ["value"]
    assert beta.column("value").to_pylist() == [10, 10, 10, 10, 20, 20, 20, 20]
    assert beta.column("value").type == pa.int32()
    assert finalized["alpha"].schema.field("key").metadata == {b"label": b"Identifier"}
    assert finalized["beta"].schema.field("value").metadata == {b"label": b"Amount"}
    assert finalized["alpha"].schema.metadata == {b"memlabel": b"Alpha"}
    assert finalized["beta"].schema.metadata == {b"memlabel": b"Beta"}

    assert empty["diagnostics"] == []
    empty_alpha = _output_table(empty, "alpha")
    empty_beta = _output_table(empty, "beta")
    finalized_empty = _finalize_typed_native_outputs(
        dsl, empty_source, {"alpha": empty_alpha, "beta": empty_beta}
    )
    assert empty_alpha.column_names == ["key"]
    assert empty_alpha.column("key").type == pa.int64()
    assert empty_beta.column_names == ["value"]
    assert empty_beta.column("value").type == pa.int32()
    assert finalized_empty["alpha"].schema.field("key").metadata == {
        b"label": b"Identifier"
    }
    assert finalized_empty["beta"].schema.field("value").metadata == {
        b"label": b"Amount"
    }


def test_typed_output_where_counter_is_shared_across_inline_and_top_level_events() -> None:
    native, error = load_native_module()
    assert native is not None, error
    dsl = (
        "data out; set inp; if flag then output out(where=(amount > 1) firstobs=2 obs=1); "
        "output out; run;"
    )
    result = native.execute_block(
        _typed_payload_with_planned_outputs(
            dsl,
            pa.table({"id": [1, 2, 3], "amount": [1, 2, 3], "flag": [True, False, True]}),
        )
    )

    assert result["diagnostics"] == []
    output = _output_table(result)
    assert output.column("id").to_pylist() == [3]


def test_typed_dataset_option_where_keeps_concat_and_string_add_rejected() -> None:
    for expression in ("name || '!' = 'A!'", "name + '!' = 'A!'"):
        parsed = parse_program(
            f"data out; set inp(where=({expression})); run;"
        )

        assert parsed.program is None
        assert [diagnostic.code for diagnostic in parsed.diagnostics] == [
            "PARSE_DATASET_WHERE_UNSUPPORTED"
        ]
        assert "dataset option WHERE= is invalid" in parsed.diagnostics[0].message


def test_typed_output_schema_and_metadata_match_legacy_public_execution() -> None:
    native, error = load_native_module()
    assert native is not None, error
    source = pa.Table.from_arrays(
        [pa.array([1, 2]), pa.array([10, 20]), pa.array(["x", "y"])],
        schema=pa.schema(
            [
                pa.field("id", pa.int64(), metadata={b"label": b"Identifier"}),
                pa.field("amount", pa.int64(), metadata={b"label": b"Amount"}),
                pa.field("secret", pa.string()),
            ],
            metadata={b"memlabel": b"Source rows"},
        ),
    )
    dsl = (
        "data out(keep=id amount rename=(amount=renamed) label='Output'); "
        "set inp; output out; run;"
    )

    typed_result = native.execute_block(
        _typed_payload_with_planned_outputs(dsl, source)
    )
    assert typed_result["diagnostics"] == []
    typed_raw = _output_table(typed_result)
    typed_final = _finalize_typed_native_outputs(dsl, source, {"out": typed_raw})["out"]
    legacy_result = DataStepExecutor(runtime_backend="rust").execute(
        ExecuteRequest(
            dsl_text=dsl,
            inputs={
                "inp": DataSetRef(
                    kind="arrow_table", location="dataset://inp", payload=source
                )
            },
        )
    )

    assert not legacy_result.has_errors, legacy_result.diagnostics
    legacy = legacy_result.outputs["out"]
    assert typed_final.to_pydict() == legacy.payload.to_pydict()
    assert typed_final.schema == legacy.payload.schema
    assert typed_final.schema.metadata == legacy.payload.schema.metadata == {
        b"memlabel": b"Output"
    }


def test_typed_runtime_uses_planned_bare_default_not_first_registered_target() -> None:
    native, error = load_native_module()
    assert native is not None, error
    dsl = "data alpha beta; set inp; output beta; output; run;"
    result = native.execute_block(
        _typed_payload_with_planned_outputs(dsl, pa.table({"id": [1, 2]}))
    )

    assert result["diagnostics"] == []
    assert _output_table(result, "alpha").num_rows == 0
    assert _output_table(result, "beta").column("id").to_pylist() == [1, 1, 2, 2]


def test_typed_deferred_output_reports_saved_fault_only_when_reached() -> None:
    native, error = load_native_module()
    assert native is not None, error
    dsl = "data out; set inp; output out(foo=bar); run;"

    empty = native.execute_block(
        _typed_payload(dsl, pa.table({"id": pa.array([], type=pa.int64())}))
    )
    reached = native.execute_block(
        _typed_payload(dsl, pa.table({"id": [1]}))
    )
    stopped = native.execute_block(
        _typed_payload(
            "data out; set inp; stop; output out(foo=bar); run;",
            pa.table({"id": [1, 2]}),
        )
    )

    assert empty["diagnostics"] == []
    assert stopped["diagnostics"] == []
    assert reached["diagnostics"][0]["code"] == "RUNTIME_EXPRESSION_EVALUATION_ERROR"
    assert reached["diagnostics"][0]["message"] == (
        "Rust statement execution failed: Unsupported dataset reference option: foo=bar"
    )
    assert "output_streams" not in reached
