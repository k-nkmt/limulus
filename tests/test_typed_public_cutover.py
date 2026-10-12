from __future__ import annotations

import pyarrow as pa
import pytest

from limulus import Session
from limulus.backend_integration.contracts import RuntimeExecutionResult
from limulus.expression_semantics import ExpressionSemantics
from limulus.execution import DataStepExecutor
from limulus.execution.contracts import OutputHandoffResult
from limulus.io import ExecutorIOService
from limulus.models import DataSetRef, ExecuteRequest, HostFunctionContext
from limulus.models import Diagnostic
from limulus.syntax.program import LarkProgramFrontend


def _arrow_ref(name: str, table: pa.Table) -> DataSetRef:
    return DataSetRef(
        kind="arrow_table",
        location=f"dataset://{name}",
        payload=table,
    )


def test_executor_does_not_compose_the_legacy_split_and_plan_pipeline() -> None:
    executor = DataStepExecutor(runtime_backend="rust")

    assert not hasattr(executor, "_pipeline")


def test_legacy_text_coordinator_and_planning_entrypoints_are_removed() -> None:
    import importlib.util

    from limulus.execution.planning import ExecutionPlanService
    from limulus.execution.rewrites import RewritePlanner

    # The unified-parser pipeline shim module is gone and must stay gone.
    assert importlib.util.find_spec("limulus.execution.pipeline") is None
    assert not hasattr(ExecutionPlanService, "plan_rewrites")
    assert not hasattr(ExecutionPlanService, "build")
    assert not hasattr(RewritePlanner, "build_plan")
    assert not hasattr(RewritePlanner, "collect_lag_lead_specs")


def test_public_typed_submit_uses_expression_and_apply_syntax_without_rescanning(
    monkeypatch,
) -> None:
    import limulus.syntax.program as program_module

    original_parse = LarkProgramFrontend.parse_program
    parsed_sources: list[str] = []
    payloads = []

    def forbidden(*args, **kwargs):
        raise AssertionError("public typed submit reached a legacy text path")

    def parse_then_forbid_rescan(frontend, source: str, *, source_id: str = "<dsl>"):
        parsed = original_parse(frontend, source, source_id=source_id)
        parsed_sources.append(source)
        # Apply occurrences are collected during the one whole-program parse.
        # Any later call would be a source rescan rather than use of the stored
        # ApplyReferenceSyntax inventory.
        monkeypatch.setattr(program_module, "collect_apply_references", forbidden)
        return parsed

    monkeypatch.setattr(LarkProgramFrontend, "parse_program", parse_then_forbid_rescan)
    monkeypatch.setattr(ExpressionSemantics, "parse_expression", forbidden)
    monkeypatch.setattr(ExpressionSemantics, "prepare_expression", forbidden)
    monkeypatch.setattr(ExpressionSemantics, "normalize_expression", forbidden)
    monkeypatch.setattr(ExpressionSemantics, "compile_expression", forbidden)
    monkeypatch.setattr(ExecutorIOService, "extract_output_targets", forbidden)
    monkeypatch.setattr(ExecutorIOService, "extract_data_targets", forbidden)

    executor = DataStepExecutor(runtime_backend="rust")
    native_executor = executor._rust_backend._executor
    execute_block = native_executor.execute_block

    def capture_payload(payload):
        payloads.append(payload)
        return execute_block(payload)

    monkeypatch.setattr(native_executor, "execute_block", capture_payload)

    def twice(value: object) -> object:
        return value * 2  # type: ignore[operator]

    dsl = (
        "data out; set inp; value = apply('twice', amount); "
        "output out(keep=id value); run;"
    )
    response = executor.execute(
        ExecuteRequest(
            dsl_text=dsl,
            inputs={"inp": _arrow_ref("inp", pa.table({"id": [1], "amount": [3]}))},
            host_function_context=HostFunctionContext(explicit={"twice": twice}),
        )
    )

    assert not response.has_errors, response.diagnostics
    assert parsed_sources == [dsl]
    assert len(payloads) == 1
    native_program = payloads[0].native_program
    effective_output_options = native_program["effective_output_options"]
    assert len(effective_output_options) == 1
    assert next(iter(effective_output_options.values()))["keep_vars"] == [
        "id",
        "value",
    ]
    output_statement = next(
        item for item in native_program["statements"] if item["kind"] == "OUTPUT"
    )
    assert output_statement["payload"]["targets"][0]["options"]["keep_vars"] == [
        "id",
        "value",
    ]
    assert response.outputs["out"].payload.column("value").to_pylist() == [6]


def test_public_typed_lag_default_does_not_use_legacy_text_parser() -> None:
    # The legacy default-lexeme text parser was deleted with the rewrites:
    # the symbol must not come back.
    import limulus.execution.rewrites as rewrites_module

    assert not hasattr(rewrites_module, "parse_default_expression")

    cases = (
        ("lag(id, 1, 99)", {"id": [4, 7]}, [99, 4]),
        ("lag(label, 1, 'missing')", {"label": ["a", "b"]}, ["missing", "a"]),
        ("lag(id, 1, .)", {"id": [4, 7]}, [None, 4]),
    )
    for expression, input_columns, expected in cases:
        response = DataStepExecutor(runtime_backend="rust").execute(
            ExecuteRequest(
                dsl_text=(
                    f"data out; set inp; value = {expression}; "
                    "output out(keep=value); run;"
                ),
                inputs={"inp": _arrow_ref("inp", pa.table(input_columns))},
            )
        )

        assert not response.has_errors, response.diagnostics
        assert response.outputs["out"].payload.column("value").to_pylist() == expected


class _RewriteHook:
    def __init__(self, source: str, replacement: str) -> None:
        self.source = source
        self.replacement = replacement
        self.calls: list[str] = []

    def evaluate(self, dsl_text: str) -> str:
        self.calls.append(dsl_text)
        return dsl_text.replace(self.source, self.replacement)


def test_public_executor_applies_hook_then_parses_the_whole_program_once(
    monkeypatch,
) -> None:
    original = (
        "data first; set inp; first_value = amount + 2; run; "
        "data out; set first; result = first_value + 2; run;"
    )
    transformed = original.replace("amount + 2", "amount + 3")
    hook = _RewriteHook("amount + 2", "amount + 3")
    frontend_calls: list[tuple[str, str]] = []
    parse_program = LarkProgramFrontend.parse_program

    def count_parse(frontend, source: str, *, source_id: str = "<dsl>"):
        frontend_calls.append((source, source_id))
        return parse_program(frontend, source, source_id=source_id)

    monkeypatch.setattr(LarkProgramFrontend, "parse_program", count_parse)
    executor = DataStepExecutor(runtime_backend="rust", macro_hook=hook)
    response = executor.execute(
        ExecuteRequest(
            dsl_text=original,
            inputs={"inp": _arrow_ref("inp", pa.table({"amount": [1]}))},
        )
    )

    assert hook.calls == [original]
    assert frontend_calls == [(transformed, "<dsl>")]
    assert not response.has_errors, response.diagnostics
    output = response.outputs["out"].payload
    assert output is not None
    assert output.column("result").to_pylist() == [6]


def test_public_executor_stops_before_native_when_a_later_block_has_stop_error(
    monkeypatch,
) -> None:
    dsl = (
        "data first; set inp; result = amount + 1; run; "
        "data broken; set inp(where=(amount >)); run;"
    )
    executor = DataStepExecutor(runtime_backend="rust")
    calls: list[object] = []

    def record_native(payload):
        calls.append(payload)
        return RuntimeExecutionResult(
            outputs={},
            diagnostics=(),
            actual_engine="rust",
            native_invoked=True,
            phase_metrics={},
        )

    monkeypatch.setattr(executor._rust_backend._executor, "execute_block", record_native)
    monkeypatch.setattr(
        executor._rust_backend._executor, "execute_block", record_native
    )
    response = executor.execute(
        ExecuteRequest(
            dsl_text=dsl,
            inputs={"inp": _arrow_ref("inp", pa.table({"amount": [1]}))},
        )
    )

    assert response.has_errors
    assert response.diagnostics[0].code == "PARSE_DATASET_WHERE_UNSUPPORTED"
    assert calls == []


@pytest.mark.parametrize("entrypoint", ["execute", "submit"])
@pytest.mark.parametrize("failure", ["missing_lark", "missing_grammar"])
def test_public_parser_initialization_failure_returns_diagnostic_without_execution(
    monkeypatch, tmp_path, entrypoint, failure,
) -> None:
    import limulus.syntax.program as program_module

    session = Session(backend="rust")
    session.load("inp", pa.table({"id": [1]}))
    session.load("existing", pa.table({"id": [99]}))
    catalog = dict(session.datasets)
    executor = session._executor

    def forbidden_native(_payload):
        raise AssertionError("parser initialization failure reached native")

    monkeypatch.setattr(
        executor._rust_backend._executor, "execute_block", forbidden_native,
    )
    if failure == "missing_lark":
        monkeypatch.setattr(program_module, "Lark", None)
    else:
        monkeypatch.setattr(
            program_module, "_grammar_path", lambda: tmp_path / "missing.lark",
        )

    dsl = "data existing; set inp; run; data new; set inp; run;"
    if entrypoint == "execute":
        response = executor.execute(ExecuteRequest(dsl_text=dsl))
        assert response.has_errors
        assert not response.outputs
        diagnostics = response.diagnostics
    else:
        result = session.submit(dsl)
        assert not result.success
        assert not result.datasets
        diagnostics = result.log

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.code == "PARSE_BACKEND_CAPABILITY_MISSING"
    assert diagnostic.severity == "error"
    assert diagnostic.stage == "parse"
    assert diagnostic.location == ""
    assert diagnostic.message == "lark parser is not available; parsing cannot proceed."
    assert not executor._generated_tables
    assert set(session.datasets) == set(catalog)
    for name, table in catalog.items():
        assert session.datasets[name] is table


def test_public_parser_does_not_mask_unrelated_frontend_failure(monkeypatch) -> None:
    executor = DataStepExecutor(runtime_backend="rust")

    def broken_frontend(_source):
        raise RuntimeError("unexpected frontend failure")

    monkeypatch.setattr(executor._program_frontend, "parse_program", broken_frontend)
    with pytest.raises(RuntimeError, match="unexpected frontend failure"):
        executor.execute(ExecuteRequest(dsl_text="data out; x=1; run;"))


def test_public_mutable_source_column_uses_its_runtime_output_type() -> None:
    session = Session(backend="rust")
    session.load("inp", pa.table({"value": pa.array([1, 2], type=pa.int64())}))

    result = session.submit("data out; set inp; value = 'text'; run;")

    assert result.success, result.format_log()
    output = result.datasets["out"]
    assert output.schema.field("value").type == pa.string()
    assert output.column("value").to_pylist() == ["text", "text"]


def test_public_output_plan_preserves_tabs_options_routing_and_empty_schema() -> None:
    session = Session(backend="rust")
    session.load("inp", pa.table({"id": pa.array([1, 2], type=pa.int64())}))

    empty = session.submit(
        "data\tout(keep=id rename=(id=key));\tset inp;\twhere id > 100;\trun;"
    )
    assert empty.success, empty.format_log()
    empty_table = empty.datasets["out"]
    assert empty_table.num_rows == 0
    assert empty_table.column_names == ["key"]
    assert empty_table.schema.field("key").type == pa.int64()

    routed = session.submit(
        "data a(keep=id) b(keep=id); set inp; output b(keep=id); output; run;"
    )
    assert routed.success, routed.format_log()
    assert routed.datasets["a"].num_rows == 0
    assert routed.datasets["b"].column("id").to_pylist() == [1, 1, 2, 2]


def test_public_output_contract_8_6_to_8_8_shares_options_and_empty_schema() -> None:
    source_schema = pa.schema(
        [
            pa.field("id", pa.int64(), metadata={b"label": b"Identifier"}),
            pa.field("amount", pa.int32(), metadata={b"label": b"Amount"}),
        ],
        metadata={b"memlabel": b"Input"},
    )
    source = pa.Table.from_arrays(
        [pa.array([1, 2], type=pa.int64()), pa.array([10, 20], type=pa.int32())],
        schema=source_schema,
    )
    dsl = (
        "data literal; set inp; note = 'output'; run; "
        "data out; set inp; "
        "if 0 then output out(keep=id rename=(id=key) label='Shared'); "
        "output out(keep=id rename=(id=key) label='Shared'); "
        "output out; run;"
    )

    session = Session(backend="rust")
    session.load("inp", source)
    result = session.submit(dsl)

    assert result.success, result.format_log()
    assert result.datasets["literal"].column("note").to_pylist() == [
        "output",
        "output",
    ]
    output = result.datasets["out"]
    assert output.column_names == ["key"]
    assert output.column("key").to_pylist() == [1, 1, 2, 2]
    assert output.schema.field("key").type == pa.int64()
    assert output.schema.field("key").metadata == {b"label": b"Identifier"}
    assert output.schema.metadata == {b"memlabel": b"Shared"}

    empty_source = pa.Table.from_arrays(
        [pa.array([], type=pa.int64()), pa.array([], type=pa.int32())],
        schema=source_schema,
    )
    empty_session = Session(backend="rust")
    empty_session.load("inp", empty_source)
    empty_result = empty_session.submit(
        "data out; set inp; "
        "if 0 then output out(keep=id rename=(id=key) label='Shared'); "
        "output out; run;"
    )

    assert empty_result.success, empty_result.format_log()
    empty_output = empty_result.datasets["out"]
    assert empty_output.num_rows == 0
    assert empty_output.column_names == ["key"]
    assert empty_output.schema.field("key").type == pa.int64()
    assert empty_output.schema.field("key").metadata == {b"label": b"Identifier"}
    assert empty_output.schema.metadata == {b"memlabel": b"Shared"}


def test_public_output_option_conflict_is_validate_atomic_across_blocks(monkeypatch) -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    native_executor = executor._rust_backend._executor
    assert native_executor is not None
    original_execute = native_executor.execute_block
    native_calls: list[object] = []
    apply_calls: list[int] = []

    def observe_native(payload):
        native_calls.append(payload)
        return original_execute(payload)

    def observe_apply(value: int) -> int:
        apply_calls.append(value)
        return value

    monkeypatch.setattr(native_executor, "execute_block", observe_native)
    response = executor.execute(
        ExecuteRequest(
            dsl_text=(
                "data prior; set inp; run; "
                "data out; set inp; value = apply('observe', id); "
                "output out(keep=id); output out(keep=amount); run;"
            ),
            inputs={"inp": _arrow_ref("inp", pa.table({"id": [1], "amount": [2]}))},
            host_function_context=HostFunctionContext(explicit={"observe": observe_apply}),
        )
    )

    assert response.has_errors
    assert response.diagnostics[0].code == "RUNTIME_DATASET_OPTION_INVALID"
    assert response.diagnostics[0].stage == "validate"
    assert len(native_calls) == 1
    assert apply_calls == []
    prior = executor._generated_tables["prior"]
    assert prior.payload is not None
    assert prior.payload.to_pylist() == [
        {"id": 1, "amount": 2}
    ]
    assert "out" not in executor._generated_tables


@pytest.mark.parametrize(
    ("dsl", "rows", "output_targets"),
    (
        ("data out out; set inp; run;", {"id": [1]}, ()),
        ("data out out(keep=id); set inp; run;", {"id": [1]}, ()),
        ("data out out; set inp; where id > 99; run;", {"id": [1]}, ()),
        ("data out out; set inp; run;", {"id": [1]}, ("explicit",)),
    ),
    ids=("normal", "optioned", "zero-rows", "explicit-targets"),
)
def test_public_output_contract_8_9_rejects_duplicate_data_targets(
    dsl: str,
    rows: dict[str, list[int]],
    output_targets: tuple[str, ...],
    monkeypatch,
) -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    native_calls: list[object] = []

    def forbidden_native(payload):
        native_calls.append(payload)
        raise AssertionError("duplicate DATA destinations must fail before native")

    monkeypatch.setattr(executor._rust_backend._executor, "execute_block", forbidden_native)
    response = executor.execute(
        ExecuteRequest(
            dsl_text=dsl,
            inputs={"inp": _arrow_ref("inp", pa.table(rows))},
            output_targets=output_targets,
        )
    )

    assert response.has_errors
    assert response.diagnostics[0].code == "VALIDATE_DUPLICATE_DATA_TARGET"
    assert response.diagnostics[0].stage == "validate"
    assert response.outputs == {}
    assert executor._generated_tables == {}
    assert native_calls == []


def test_public_array_and_nested_argument_concat_use_typed_native_path() -> None:
    session = Session(backend="rust")
    session.load("inp", pa.table({"first": [1], "second": [2], "third": [3]}))

    result = session.submit(
        "data out; set inp; array values[3] first second third; "
        "values[2] = 7; picked = values[2]; "
        "joined = cats('pre', 'fix' || '!'); run;"
    )

    assert result.success, result.format_log()
    row = result.datasets["out"].to_pylist()[0]
    assert row["second"] == 7
    assert row["picked"] == 7
    assert row["joined"] == "prefix!"


def test_public_input_failure_does_not_invoke_native(monkeypatch) -> None:
    session = Session(backend="rust")
    calls: list[object] = []

    def record_native(payload):
        calls.append(payload)
        return RuntimeExecutionResult(
            outputs={},
            diagnostics=(),
            actual_engine="rust",
            native_invoked=True,
            phase_metrics={},
        )

    monkeypatch.setattr(session._executor._rust_backend._executor, "execute_block", record_native)
    monkeypatch.setattr(
        session._executor._rust_backend._executor,
        "execute_block",
        record_native,
    )

    result = session.submit("data out; set missing; run;")

    assert not result.success
    assert result.log[0].code == "RUNTIME_SET_DATASET_NOT_FOUND"
    assert result.log[0].stage == "validate"
    assert calls == []


def test_public_non_arrow_input_failure_keeps_preparation_stage(monkeypatch) -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    calls: list[object] = []

    def forbidden_native(payload):
        calls.append(payload)
        raise AssertionError("invalid input carriers must stop before native")

    monkeypatch.setattr(executor._rust_backend._executor, "execute_block", forbidden_native)
    response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set inp; run;",
            inputs={
                "inp": DataSetRef(
                    kind="unsupported",
                    location="dataset://inp",
                    payload=object(),
                )
            },
        )
    )

    assert response.has_errors
    assert response.diagnostics[0].code == "RUNTIME_RUST_BRIDGE_INPUT_NOT_ARROW"
    assert response.diagnostics[0].stage == "pre-processing inputs"
    assert calls == []


def test_public_invalid_dataset_option_keeps_preparation_stage(monkeypatch) -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    calls: list[object] = []

    def forbidden_native(payload):
        calls.append(payload)
        raise AssertionError("invalid dataset options must stop before native")

    monkeypatch.setattr(executor._rust_backend._executor, "execute_block", forbidden_native)
    response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set inp(where=(missing >= 1)); run;",
            inputs={
                "inp": _arrow_ref("inp", pa.table({"amount": [1]})),
            },
        )
    )

    assert response.has_errors
    assert response.diagnostics[0].code == "RUNTIME_DATASET_OPTION_INVALID"
    assert response.diagnostics[0].stage == "pre-processing inputs"
    assert calls == []


def test_public_typed_finalizer_failure_uses_resolve_outputs_stage(monkeypatch) -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    source = _arrow_ref("source", pa.table({"id": [1]}))

    def fail_finalize(request):
        return OutputHandoffResult(
            output_source_metadata_lineage=request.source_metadata_lineage,
            diagnostics=(
                Diagnostic(
                    code="CONVERT_OUTPUT_FAILED",
                    severity="error",
                    location="dataset:out",
                    message="metadata failure",
                ),
            ),
        )

    monkeypatch.setattr(executor._output_handoff_service, "finalize_typed", fail_finalize)
    response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set source; run;",
            inputs={"source": source},
        )
    )

    assert response.has_errors
    assert response.diagnostics[0].code == "CONVERT_OUTPUT_FAILED"
    assert response.diagnostics[0].stage == "resolve outputs"
    assert response.outputs == {}
    assert "out" not in executor._generated_tables


def test_public_parse_error_precedes_request_input_alias_conflict(monkeypatch) -> None:
    session = Session(backend="rust")
    calls: list[object] = []

    def forbidden_native(payload):
        calls.append(payload)
        raise AssertionError("parse errors must stop before native")

    monkeypatch.setattr(
        session._executor._rust_backend._executor,
        "execute_block",
        forbidden_native,
    )
    session.load("inp", pa.table({"amount": [1]}))
    duplicate = _arrow_ref("INP", pa.table({"amount": [2]}))

    result = session._executor.execute(
        ExecuteRequest(
            dsl_text="data out; set inp(where=(amount >)); run;",
            inputs={"inp": _arrow_ref("inp", pa.table({"amount": [1]})), "INP": duplicate},
        )
    )

    assert result.has_errors
    assert result.diagnostics[0].code == "PARSE_DATASET_WHERE_UNSUPPORTED"
    assert result.diagnostics[0].stage == "parse"
    assert calls == []


def test_public_valid_request_input_alias_conflict_still_stops_before_native(
    monkeypatch,
) -> None:
    session = Session(backend="rust")
    calls: list[object] = []

    def forbidden_native(payload):
        calls.append(payload)
        raise AssertionError("ambiguous request inputs must stop before native")

    monkeypatch.setattr(
        session._executor._rust_backend._executor,
        "execute_block",
        forbidden_native,
    )
    result = session._executor.execute(
        ExecuteRequest(
            dsl_text="data out; set inp; run;",
            inputs={
                "inp": _arrow_ref("inp", pa.table({"amount": [1]})),
                "INP": _arrow_ref("INP", pa.table({"amount": [2]})),
            },
        )
    )

    assert result.has_errors
    assert result.diagnostics[0].code == "RUNTIME_REQUEST_INPUT_ALIAS_CONFLICT"
    assert result.diagnostics[0].stage == "resolve inputs"
    assert calls == []


def test_public_resolves_apply_inventory_before_branch_execution(monkeypatch) -> None:
    session = Session(backend="rust")
    session.load("inp", pa.table({"amount": [1]}))
    calls: list[object] = []

    def forbidden_native(payload):
        calls.append(payload)
        raise AssertionError("unresolved apply target must fail before native")

    monkeypatch.setattr(
        session._executor._rust_backend._executor,
        "execute_block",
        forbidden_native,
    )
    result = session.submit(
        "data out; set inp; if 0 then value = apply('not_registered', amount); run;"
    )

    assert not result.success
    assert result.log[0].code == "RUNTIME_APPLY_TARGET_UNRESOLVED"
    assert "not_registered" in result.log[0].message
    assert calls == []
