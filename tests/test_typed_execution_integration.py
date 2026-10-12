from __future__ import annotations

import json

import pyarrow as pa
import pytest

from limulus.backend_integration.contracts import (
    TypedRustExecutionPayload,
)
from limulus.backend_integration.rust_executor import RustNativeBlockExecutor
from limulus.execution import DataStepExecutor
from limulus.execution.input_preparation import ExecutionInputPreparationService
from limulus.execution.contracts import TypedInputResolutionRequest
from limulus.host_functions import HostFunctionContext
from limulus.models import DataSetRef, ExecuteRequest
from limulus.syntax import parse_program


def _parsed_block(source_text: str):
    parsed = parse_program(source_text)
    assert parsed.program is not None, parsed.diagnostics
    assert not parsed.has_errors, parsed.diagnostics
    assert len(parsed.program.blocks) == 1
    return parsed.program.source, parsed.program.blocks[0]


def _block_statement_kinds(block) -> set[str]:
    kinds: set[str] = set()

    def collect(statement) -> None:
        kinds.add(statement.kind)
        payload = statement.payload
        for attribute in ("then_action", "action"):
            action = getattr(payload, attribute, None)
            if hasattr(action, "kind") and hasattr(action, "payload"):
                collect(action)

    for statement in block.statements:
        collect(statement)
    return kinds


def _wire_statement_kinds(value) -> set[str]:
    kinds: set[str] = set()

    def visit(item) -> None:
        if isinstance(item, dict):
            kind = item.get("kind")
            if isinstance(kind, str) and kind.upper() == kind:
                kinds.add(kind)
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return kinds


def test_typed_input_resolver_uses_structural_occurrences_and_catalog_priority() -> None:
    _source, block = _parsed_block(
        "data out; set WORK.shared explicit_only available_only registered_only "
        "dictionary.tables WORK.shared; run;"
    )
    explicit_shared = DataSetRef(
        kind="arrow_table", location="dataset://explicit-shared", payload=pa.table({"v": [1]})
    )
    available_shared = DataSetRef(
        kind="arrow_table", location="dataset://available-shared", payload=pa.table({"v": [2]})
    )
    registered_shared = DataSetRef(
        kind="arrow_table", location="dataset://registered-shared", payload=pa.table({"v": [3]})
    )
    request = TypedInputResolutionRequest(
        block=block,
        explicit_inputs={
            "shared": explicit_shared,
            "explicit_only": DataSetRef(
                kind="arrow_table", location="dataset://explicit-only", payload=pa.table({"v": [4]})
            ),
        },
        available_inputs={
            "shared": available_shared,
            "available_only": DataSetRef(
                kind="arrow_table", location="dataset://available-only", payload=pa.table({"v": [5]})
            ),
        },
        registered_tables={
            "shared": registered_shared,
            "available_only": DataSetRef(
                kind="arrow_table", location="dataset://registered-available", payload=pa.table({"v": [6]})
            ),
            "registered_only": DataSetRef(
                kind="arrow_table", location="dataset://registered-only", payload=pa.table({"v": [7]})
            ),
        },
    )

    resolved = ExecutionInputPreparationService().resolve_typed(request)

    assert resolved.diagnostics == ()
    assert resolved.resolved_inputs.inputs["WORK.shared"].location == "dataset://explicit-shared"
    assert resolved.resolved_inputs.inputs["available_only"].location == "dataset://available-only"
    assert resolved.resolved_inputs.inputs["registered_only"].location == "dataset://registered-only"
    assert resolved.resolved_inputs.inputs["dictionary.tables"].location == "dictionary://dictionary.tables"
    assert [item.source_name for item in resolved.source_metadata_lineage.ordered_sources] == [
        "WORK.shared",
        "explicit_only",
        "available_only",
        "registered_only",
        "dictionary.tables",
        "WORK.shared",
    ]
    assert [item.source_ordinal for item in resolved.source_metadata_lineage.ordered_sources] == list(range(6))


def test_typed_input_resolver_preserves_explicit_memory_presentation() -> None:
    _source, block = _parsed_block("data out; set memory_input; run;")
    memory_ref = DataSetRef(
        kind="memory", location="dataset://memory_input", payload=[{"v": 1}]
    )

    resolved = ExecutionInputPreparationService().resolve_typed(
        TypedInputResolutionRequest(
            block=block,
            explicit_inputs={"memory_input": memory_ref},
        )
    )

    assert resolved.diagnostics == ()
    assert resolved.resolved_inputs.preserve_memory_presentation is True
    assert resolved.resolved_inputs.inputs["memory_input"].kind == "arrow_table"


def test_typed_input_resolver_uses_supplied_synthetic_alias_after_catalogs() -> None:
    _source, block = _parsed_block("data out; set only_synthetic; run;")
    synthetic_ref = DataSetRef(
        kind="arrow_table",
        location="dataset://synthetic",
        payload=pa.table({"v": [9]}),
    )

    resolved = ExecutionInputPreparationService().resolve_typed(
        TypedInputResolutionRequest(
            block=block,
            synthetic_inputs={"ONLY_SYNTHETIC": synthetic_ref},
        )
    )

    assert resolved.diagnostics == ()
    resolved_ref = resolved.resolved_inputs.inputs["only_synthetic"]
    assert resolved_ref.location == "dataset://synthetic"
    assert isinstance(resolved_ref.payload, pa.Table)
    assert resolved_ref.payload.column("v").to_pylist() == [9]


def test_typed_input_resolver_returns_empty_carriers_on_missing_or_non_arrow_input() -> None:
    _source, block = _parsed_block("data out; set inp; run;")
    service = ExecutionInputPreparationService()

    missing = service.resolve_typed(TypedInputResolutionRequest(block=block))
    non_arrow = service.resolve_typed(
        TypedInputResolutionRequest(
            block=block,
            explicit_inputs={
                "inp": DataSetRef(
                    kind="unsupported", location="dataset://inp", payload=object()
                )
            },
        )
    )

    assert [diagnostic.code for diagnostic in missing.diagnostics] == [
        "RUNTIME_SET_DATASET_NOT_FOUND"
    ]
    assert missing.resolved_inputs.inputs == {}
    assert missing.source_metadata_lineage.ordered_sources == ()
    assert [diagnostic.code for diagnostic in non_arrow.diagnostics] == [
        "RUNTIME_RUST_BRIDGE_INPUT_NOT_ARROW"
    ]
    assert non_arrow.resolved_inputs.inputs == {}
    assert non_arrow.source_metadata_lineage.ordered_sources == ()


def test_typed_block_coordinator_executes_without_legacy_execution_plan(
    monkeypatch,
) -> None:
    dsl = "data out; set inp; result = amount ** 2 ** 3; output out; run;"
    source, block = _parsed_block(dsl)
    request = ExecuteRequest(
        dsl_text=dsl,
        inputs={
            "inp": DataSetRef(
                kind="arrow_table",
                location="dataset://inp",
                payload=pa.table({"amount": [2]}),
            )
        },
    )

    executor = DataStepExecutor(runtime_backend="rust")
    native_executor = executor._rust_backend._executor
    assert native_executor is not None
    execute_typed_block = native_executor.execute_block
    captured_payloads: list[TypedRustExecutionPayload] = []

    def capture_typed_payload(payload):
        captured_payloads.append(payload)
        return execute_typed_block(payload)

    monkeypatch.setattr(native_executor, "execute_block", capture_typed_payload)
    result = executor.execute(request)

    assert result.diagnostics == ()
    output = result.outputs["out"]
    assert isinstance(output.payload, pa.Table)
    assert output.payload.column("result").to_pylist() == [256]
    assert len(captured_payloads) == 1
    native_payload = captured_payloads[0]
    assert json.loads(native_payload.native_program_json)["version"] == 1
    assert not hasattr(native_payload, "ast_json")
    assert native_payload.execution_plan is not None
    assert native_payload.execution_plan["execution_readiness"]["decision"] == "ready"
    assert native_payload.execution_plan["row_loop_plan"]["mode"] == "arrow_row_loop"
    assert native_payload.execution_plan["output_handoff_plan"].targets == ("out",)


@pytest.mark.parametrize(
    ("expected_kind", "dsl"),
    [
        ("DATA", "data out; set inp; run;"),
        ("SET", "data out; set inp; run;"),
        ("MERGE", "data out; merge inp; run;"),
        ("WHERE", "data out; set inp; where amount > 0; run;"),
        ("ASSIGN", "data out; set inp; result = amount + 1; run;"),
        ("SUM", "data out; set inp; total + 1; run;"),
        ("KEEP", "data out; set inp; keep amount; run;"),
        ("DROP", "data out; set inp; drop other; run;"),
        ("BY", "data out; set inp; by amount; run;"),
        (
            "RETAIN",
            "data out; set inp; retain total 0; total = total + amount; run;",
        ),
        ("LABEL", "data out; set inp; label amount = 'Amount'; run;"),
        ("RENAME", "data out; set inp; rename amount = value; run;"),
        ("OUTPUT", "data out; set inp; output out; run;"),
        ("IF", "data out; set inp; if amount > 0 then result = amount; run;"),
        (
            "ELSE IF",
            "data out; set inp; if amount < 0 then result = 0; "
            "else if amount = 0 then result = 1; "
            "else result = amount; run;",
        ),
        (
            "ELSE",
            "data out; set inp; if amount < 0 then result = 0; "
            "else result = amount; run;",
        ),
        ("DO", "data out; set inp; do i = 1 to 2; result = i; end; run;"),
        (
            "END",
            "data out; set inp; do i = 1 to 1; result = i; end; run;",
        ),
        (
            "ARRAY",
            "data out; set inp; array vals[2] a b; vals[1] = amount; run;",
        ),
        ("DELETE", "data out; set inp; if amount < 0 then delete; run;"),
        ("STOP", "data out; set inp; if amount < 0 then stop; run;"),
        ("RUN", "data out; set inp; run;"),
        ("SKIPPED", "data out; set inp; format amount 8.; run;"),
    ],
)
def test_typed_coordinator_hands_each_statement_variant_to_native(
    expected_kind: str, dsl: str, monkeypatch
) -> None:
    source, block = _parsed_block(dsl)
    executor = DataStepExecutor(runtime_backend="rust")
    native_executor = executor._rust_backend._executor
    assert native_executor is not None
    execute_typed_block = native_executor.execute_block
    captured_payloads: list[TypedRustExecutionPayload] = []

    def capture_typed_payload(payload):
        captured_payloads.append(payload)
        return execute_typed_block(payload)

    monkeypatch.setattr(native_executor, "execute_block", capture_typed_payload)
    result = executor.execute(
        ExecuteRequest(
            dsl_text=dsl,
            inputs={
                "inp": DataSetRef(
                    kind="arrow_table",
                    location="dataset://inp",
                    payload=pa.table(
                        {
                            "amount": [1, 0, -1],
                            "other": [3, 4, 5],
                            "a": [0, 0, 0],
                            "b": [0, 0, 0],
                        }
                    ),
                )
            },
        )
    )

    if expected_kind == "SKIPPED":
        assert result.has_errors
        unsupported = next(
            diagnostic
            for diagnostic in result.diagnostics
            if diagnostic.code == "RUNTIME_RUST_STATEMENT_UNSUPPORTED"
        )
        assert unsupported.stage == "execute"
        assert unsupported.location == "statement:3"
        assert result.outputs.get("out") is None
    else:
        assert result.diagnostics == ()
        assert isinstance(result.outputs["out"].payload, pa.Table)
    assert expected_kind in _block_statement_kinds(block)
    assert len(captured_payloads) == 1
    native_program = json.loads(captured_payloads[0].native_program_json)
    wire_kinds = _wire_statement_kinds(native_program["statements"])
    if expected_kind == "MERGE":
        # Prepared MERGE is the normalized SET carrier at native
        # handoff; provenance travels separately on the typed payload.
        assert captured_payloads[0].preparation_kind == "prepared_merge"
        assert "SET" in wire_kinds
        assert "MERGE" not in wire_kinds
    elif expected_kind == "LABEL":
        # LABEL metadata belongs to Python output finalization, not native
        # calculation statements.
        assert "LABEL" not in wire_kinds
        assert result.outputs["out"].metadata["column_labels"] == {
            "amount": "Amount"
        }
    else:
        assert expected_kind in wire_kinds


def test_typed_block_does_not_invoke_native_or_publish_on_input_failure(
    monkeypatch,
) -> None:
    dsl = "data out; set missing; run;"
    source, block = _parsed_block(dsl)
    executor = DataStepExecutor(runtime_backend="rust")
    calls: list[object] = []

    def forbidden_native_call(payload):
        calls.append(payload)
        raise AssertionError("native entry must not run")

    monkeypatch.setattr(
        executor._rust_backend._executor,
        "execute_block",
        forbidden_native_call,
    )

    result = executor._execute_typed_block(
        source=source,
        block=block,
        request=ExecuteRequest(dsl_text=dsl),
    )

    assert result.outputs == {}
    assert [diagnostic.code for diagnostic in result.diagnostics] == [
        "RUNTIME_SET_DATASET_NOT_FOUND"
    ]
    assert calls == []
    assert executor._generated_tables == {}


def test_typed_block_does_not_publish_partial_outputs_after_native_failure() -> None:
    dsl = "data out; set inp; first = amount + 1; failed = amount @@ 2; run;"
    source, block = _parsed_block(dsl)
    executor = DataStepExecutor(runtime_backend="rust")
    request = ExecuteRequest(
        dsl_text=dsl,
        inputs={
            "inp": DataSetRef(
                kind="arrow_table",
                location="dataset://inp",
                payload=pa.table({"amount": [3]}),
            )
        },
    )

    result = executor._execute_typed_block(
        source=source,
        block=block,
        request=request,
    )

    assert result.outputs == {}
    assert result.outputs_arrow == {}
    assert result.diagnostics
    assert executor._generated_tables == {}


def test_typed_block_resolves_and_invokes_request_scoped_host_function() -> None:
    dsl = 'data out; set inp; result = apply("twice", amount); run;'
    source, block = _parsed_block(dsl)
    invocations: list[int] = []

    def twice(value: int) -> int:
        invocations.append(value)
        return value * 2

    request = ExecuteRequest(
        dsl_text=dsl,
        inputs={
            "inp": DataSetRef(
                kind="arrow_table",
                location="dataset://inp",
                payload=pa.table({"amount": [3]}),
            )
        },
        host_function_context=HostFunctionContext(explicit={"twice": twice}),
    )

    result = DataStepExecutor(runtime_backend="rust")._execute_typed_block(
        source=source,
        block=block,
        request=request,
    )

    assert result.diagnostics == ()
    output = result.outputs["out"]
    assert isinstance(output.payload, pa.Table)
    assert output.payload.column("result").to_pylist() == [6]
    assert invocations == [3]


def test_typed_native_executor_rejects_invalid_output_stream_result(monkeypatch) -> None:
    class NativeWithMissingStreams:
        def __init__(self) -> None:
            self.calls = 0

        def execute_block(self, payload):
            self.calls += 1
            assert "ast_json" not in payload
            assert payload["native_program_json"] == '{"version":1}'
            return {"diagnostics": [], "output_streams": None}

    native = NativeWithMissingStreams()
    monkeypatch.setattr(
        "limulus.backend_integration.rust_executor.load_native_module",
        lambda: (native, None),
    )
    stream = pa.table({"v": [1]}).__arrow_c_stream__()
    payload = TypedRustExecutionPayload(
        native_program={"version": 1, "output_targets": ["out"]},
        native_program_json='{"version":1}',
        input_streams={"inp": stream},
    )

    result = RustNativeBlockExecutor().execute_block(payload)

    assert native.calls == 1
    assert result.outputs == {}
    assert [diagnostic.code for diagnostic in result.diagnostics] == [
        "RUNTIME_RUST_NATIVE_EXECUTION_FAILED"
    ]
