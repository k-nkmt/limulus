import limulus
import threading
import unittest
from decimal import Decimal
from unittest.mock import patch

import pytest

import pyarrow as pa
import pandas as pd
import polars as pl

from limulus import Session
from limulus.backend_integration.contracts import RuntimeExecutionResult
from limulus.runtime import DataStepExecutor
from limulus.models import DataSetRef, ExecuteRequest, ExecuteResponse, HostFunctionContext


EXECUTOR_SCENARIOS = {
    "explicit_output_targets_precedence": {
        "overview": "Explicit output_targets take precedence over inferred targets",
        "dsl": "data inferred; set in; run;",
        "inputs": {"in": [{"id": 1}]},
        "output_targets": ["explicit"],
        "expected_keys": {"explicit"},
        "expected_output": [{"id": 1}],
    },
    "set_input_case_insensitive_work_prefix": {
        "overview": "Resolves SET input with WORK prefix and case-insensitive name matching",
        "dsl": "data out; set WORK.My_Table; output out; run;",
        "inputs": None,
        "expected_output": [{"id": 1, "amount": 10}],
    },
    "explicit_inputs_precedence": {
        "overview": "Explicit inputs take precedence over registered tables",
        "dsl": "data out; set in; output out; run;",
        "inputs": {"in": [{"id": 2, "amount": 20}]},
        "output_targets": ["out"],
        "expected_output": [{"id": 2, "amount": 20}],
    },
    "submit_backend_override": {
        "overview": (
            "session.submit(backend=...) and limulus.submit(..., backend=...) "
            "temporarily override the runtime backend for a single call"
        ),
        "inputs": {"name": ["hello world", "foo bar"], "x": [10, 20]},
        "dsl": "data out; set inp; where x >= 15; output out; run;",
        "expected_output": [{"name": "foo bar", "x": 20}],
    },
}


def _arrow_ref(name: str, payload: list[dict] | None) -> DataSetRef:
    rows = payload or []
    return DataSetRef(kind="arrow_table", location=f"dataset://{name}", payload=pa.Table.from_pylist(rows))


def _output_rows(response, output_name: str) -> list[dict]:
    return response.outputs[output_name].payload.to_pylist()


def _dataset_rows(dataset) -> list[dict[str, object]]:
    if hasattr(dataset, "to_pylist"):
        return dataset.to_pylist()
    return list(dataset)


class _CountingToPylistTable:
    def __init__(self, rows):
        self._rows = [dict(row) for row in rows]
        self.to_pylist_calls = 0

    def to_pylist(self):
        self.to_pylist_calls += 1
        return [dict(row) for row in self._rows]


EXECUTE_API_SCENARIOS = {
    "valid_request": {
        "dsl": "data out; set in; run;",
        "output_targets": ["out"],
    },
    "compat_notice": {
        "dsl": "data out; set in; if amount = . then output out; run;",
        "inputs": [{"amount": None}],
        "output_targets": ["out"],
        "notice_id": "COMPAT_MISSING_VALUE_SEMANTICS",
    },
    "reject_empty_dsl": {
        "dsl": "   ",
        "output_targets": ["out"],
        "expected_code": "REQ_EMPTY_DSL",
    },
    "reject_invalid_inputs": {
        "dsl": "data out; set in; run;",
        "output_targets": ["out"],
        "expected_code": "REQ_INVALID_INPUTS",
    },
    "reject_invalid_output_targets": {
        "dsl": "data out; set in; run;",
        "output_targets": ["", "out"],
        "expected_code": "REQ_INVALID_OUTPUT_TARGETS",
    },
    "resolve_targets_data_statement": {
        "dsl": "data out; set in; output out; run;",
        "inputs": [{"id": 1}],
        "output_targets": [],
        "expected_output": [{"id": 1}],
    },
    "resolve_targets_data_and_output": {
        "dsl": "data out_a out_b; set in; if id = 1 then output out_a; else output out_b; run;",
        "inputs": [{"id": 1}, {"id": 2}],
        "output_targets": [],
        "expected_out_a": [{"id": 1}],
        "expected_out_b": [{"id": 2}],
    },
    "convert_unsupported_format": {
        "dsl": "data out; set in; output out; run;",
        "inputs": [{"id": 1}],
        "format": "json",
        "expected_code": "CONVERT_OUTPUT_FORMAT_UNSUPPORTED",
    },
    "convert_failed_dataset": {
        "expected_code": "CONVERT_OUTPUT_FAILED",
    },
    "simple_set_reuse_rows": {
        "dsl": "data out; set in; output out; run;",
        "inputs": [{"id": 1, "amount": 10}, {"id": 2, "amount": 20}],
    },
    "pylist_only_explicit": {
        "rows": [{"id": 1}],
    },
    "catalog_resolution": {
        "catalog_payload": [{"id": 1}, {"id": 2}],
        "catalog_dsl": "data out; set in; where id = 2; output out; run;",
        "catalog_expected": [{"id": 2}],
        "explicit_dsl": "data out; set in; output out; run;",
        "explicit_payload": [{"id": 9}],
        "explicit_expected": [{"id": 9}],
    },
    "register_tables_bulk": {
        "dsl": "data out; set in_a in_b; output out; run;",
        "expected_output": [{"id": 1}, {"id": 2}],
    },
    "resolve_registered_set_input": {
        "dsl": "data out; set in; output out; run;",
        "registered_payload": [{"id": 1, "amount": 10}],
        "expected_output": [{"id": 1, "amount": 10}],
    },
    "previous_outputs": {
        "first_dsl": "data tmp; set in; output tmp; run;",
        "first_inputs": [{"id": 1}],
        "second_dsl": "data out; set tmp; output out; run;",
        "expected_output": [{"id": 1}],
    },
    "multi_block_failure": {
        "dsl": "data stage1; set in; output stage1; run; data stage2; set missing_in; output stage2; run;",
        "inputs": [{"id": 1}],
        "expected_code": "RUNTIME_SET_DATASET_NOT_FOUND",
        "expected_location": "block:2",
    },
    "if_do_nesting_limit": {
        "nested_if_count": 11,
        "expected_code": "RUNTIME_LOOP_NESTING_LIMIT_EXCEEDED",
    },
}


def test_submit_api_surface_runtime_requirements_and_format_support() -> None:
    executor = DataStepExecutor()

    requirements = executor.get_runtime_requirements()
    supported_formats = executor.list_supported_formats()
    polars_support = executor.check_format_support("polars")
    support_result = executor.check_format_support("json")

    assert requirements.required_python == ">=3.10"
    assert "macos" in requirements.supported_os
    assert "polars" in supported_formats
    assert polars_support.supported is True
    assert support_result.supported is False
    assert support_result.reason_code == "CAP_UNSUPPORTED_FORMAT"


def test_submit_api_surface_explicit_output_targets_take_precedence() -> None:
    scenario = EXECUTOR_SCENARIOS["explicit_output_targets_precedence"]
    executor = DataStepExecutor()

    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": _arrow_ref("in", scenario["inputs"]["in"])},
            output_targets=scenario["output_targets"],
        )
    )

    assert response.has_errors is False
    assert set(response.outputs.keys()) == scenario["expected_keys"]
    assert _output_rows(response, "explicit") == scenario["expected_output"]


def test_submit_api_surface_parse_error_short_circuits_validate() -> None:
    executor = DataStepExecutor()

    response = executor.execute(
        ExecuteRequest(
            dsl_text="data dictionary.tables; set in; ???; run;",
            inputs={"in": _arrow_ref("in", [{"id": 1}])},
        )
    )

    assert response.has_errors is True
    assert response.diagnostics[0].stage == "parse"
    assert all(diagnostic.stage != "validate" for diagnostic in response.diagnostics)


def test_submit_api_surface_skips_proc_and_macro_blocks_before_data_step_execution() -> None:
    executor = DataStepExecutor()

    response = executor.execute(
        ExecuteRequest(
            dsl_text="""
            %let cutoff = 10;
            %put entering data step;
            %* skip macro comment;
            %macro noop();
            data ignored;
            set in;
            run;
            %mend noop;

            proc sort data=in out=sorted;
            by id;
            run;

            data out;
            set in;
            if amount >= 10 then output out;
            run;
            """,
            inputs={"in": _arrow_ref("in", [{"id": 1, "amount": 5}, {"id": 2, "amount": 12}])},
        )
    )

    assert response.has_errors is False
    assert _output_rows(response, "out") == [{"id": 2, "amount": 12}]


def test_submit_api_surface_skips_proc_run_and_named_macro_blocks() -> None:
    executor = DataStepExecutor()

    response = executor.execute(
        ExecuteRequest(
            dsl_text="""
            %macro noop();
            data inside_macro;
            set in;
            run;
            %mend noop;

            proc print data=in;
            run;

            data out;
            set in;
            output out;
            run;
            """,
            inputs={"in": _arrow_ref("in", [{"id": 1, "amount": 5}])},
        )
    )

    assert response.has_errors is False
    assert _output_rows(response, "out") == [{"id": 1, "amount": 5}]


def test_executor_split_data_step_blocks_uses_typed_projection() -> None:
    from limulus.block_splitter import DataStepBlockSplitter

    splitter = DataStepBlockSplitter()
    dsl_text = "* comment;\ndata out;\n  set in;\nrun;"

    # The split is the typed-block projection — the splitter owns no
    # legacy parser facade or region scanner at all.
    assert not hasattr(splitter, "_parser_service")
    blocks = splitter.split(dsl_text)

    assert blocks == ("data out;\n  set in;\nrun;",)


def test_executor_split_data_step_blocks_falls_back_when_typed_result_stops() -> None:
    from limulus.block_splitter import DataStepBlockSplitter

    splitter = DataStepBlockSplitter()
    dsl_text = "data bad;\n  ???;\nrun;"

    from limulus.syntax.models import ProgramParseResult

    with patch.object(
        splitter._program_frontend,
        "parse_program",
        return_value=ProgramParseResult(program=None, diagnostics=()),
    ):
        blocks = splitter.split(dsl_text)

    assert blocks == ("data bad;\n  ???;\nrun;",)


def test_submit_api_surface_resolves_set_input_case_insensitive_work_prefix() -> None:
    scenario = EXECUTOR_SCENARIOS["set_input_case_insensitive_work_prefix"]
    executor = DataStepExecutor()
    executor.register_table(
        "my_table",
        _arrow_ref("my_table", [{"id": 1, "amount": 10}]),
    )

    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
        )
    )

    assert response.has_errors is False
    assert _output_rows(response, "out") == scenario["expected_output"]


def test_submit_api_surface_prefers_explicit_inputs_over_registered_tables() -> None:
    scenario = EXECUTOR_SCENARIOS["explicit_inputs_precedence"]
    executor = DataStepExecutor()
    executor.register_table(
        "in",
        _arrow_ref("in-registered", [{"id": 1, "amount": 10}]),
    )

    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": _arrow_ref("in-explicit", scenario["inputs"]["in"])},
            output_targets=scenario["output_targets"],
        )
    )

    assert response.has_errors is False
    assert _output_rows(response, "out") == scenario["expected_output"]


def test_session_submit_backend_kwarg_uses_specified_backend() -> None:
    scenario = EXECUTOR_SCENARIOS["submit_backend_override"]
    session = Session(backend="rust")
    session.load("inp", pa.table(scenario["inputs"]))

    result = session.submit(scenario["dsl"], backend="rust")

    assert result.success is True
    assert session["out"].to_pylist() == scenario["expected_output"]


def test_submit_api_surface_apply_can_use_python_builtin_and_module_names_without_registration() -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    response = executor.execute(
        ExecuteRequest(
            dsl_text=(
                "data out; set in; "
                "if apply('len', note) = 0 and apply('math.sqrt', num) = 2 then output out; run;"
            ),
            inputs={
                "in": _arrow_ref("in", [{"id": 1, "note": "", "num": 4}, {"id": 2, "note": "x", "num": 9}])
            },
            output_targets=["out"],
        )
    )

    assert response.has_errors is False
    assert _output_rows(response, "out") == [{"id": 1, "note": "", "num": 4}]


def test_submit_api_surface_apply_looks_up_user_defined_function_without_registration() -> None:
    def myfunc(v):
        return v + 10

    executor = DataStepExecutor(runtime_backend="rust")
    response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set in; if apply('myfunc', amount) = 30 then output out; run;",
            inputs={"in": _arrow_ref("in", [{"id": 1, "amount": 10}, {"id": 2, "amount": 15}, {"id": 3, "amount": 20}])},
            output_targets=["out"],
            host_function_context=HostFunctionContext(caller_globals={"myfunc": myfunc}),
        )
    )

    assert response.has_errors is False
    assert _output_rows(response, "out") == [{"id": 3, "amount": 20}]


def test_submit_api_surface_assignment_function_dispatch_uses_runtime_registry_entrypoint() -> None:
    def double(value):
        return value * 2

    executor = DataStepExecutor(runtime_backend="rust")
    response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set in; doubled = apply('double', amount); output out; keep id doubled; run;",
            inputs={"in": _arrow_ref("in", [{"id": 1, "amount": 4}])},
            output_targets=["out"],
            host_function_context=HostFunctionContext(caller_globals={"double": double}),
        )
    )

    assert response.has_errors is False
    assert _output_rows(response, "out") == [{"id": 1, "doubled": 8.0}]


def test_submit_api_surface_unresolved_apply_target_diagnostic_names_function() -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    response = executor.execute(
        ExecuteRequest(
            dsl_text=(
                "data out; set in; "
                "doubled = apply('missing_apply_target', amount); "
                "output out; keep id doubled; run;"
            ),
            inputs={"in": _arrow_ref("in", [{"id": 1, "amount": 4}])},
            output_targets=["out"],
        )
    )

    assert response.has_errors is True
    assert response.diagnostics
    assert "missing_apply_target" in response.diagnostics[0].message


def test_submit_api_surface_unsupported_function_returns_identifiable_error() -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set in; where custom_not_supported(amount) > 0; output out; run;",
            inputs={"in": _arrow_ref("in", [{"amount": 1}])},
            output_targets=["out"],
        )
    )

    assert response.has_errors is True
    assert response.diagnostics[0].code == "RUNTIME_UNSUPPORTED_FUNCTION"
    assert "custom_not_supported" in response.diagnostics[0].message


def test_submit_api_surface_collision_with_internal_reference_variable_returns_diagnostic() -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set in(in=in_flag); output out; run;",
            inputs={"in": _arrow_ref("in", [{"id": 1, "in_flag": False}])},
            output_targets=["out"],
        )
    )

    assert response.has_errors is True
    assert response.diagnostics[0].code == "RUNTIME_INTERNAL_VAR_NAME_COLLISION"
    assert "in_flag" in response.diagnostics[0].message


def test_submit_api_surface_collision_with_internal_reference_variable_is_case_insensitive() -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set in(in=in_flag); output out; run;",
            inputs={"in": _arrow_ref("in", [{"id": 1, "IN_FLAG": False}])},
            output_targets=["out"],
        )
    )

    assert response.has_errors is True
    assert response.diagnostics[0].code == "RUNTIME_INTERNAL_VAR_NAME_COLLISION"
    assert "IN_FLAG" in response.diagnostics[0].message


def test_submit_api_surface_catalog_resolution_and_explicit_override() -> None:
    scenario = EXECUTE_API_SCENARIOS["catalog_resolution"]
    executor = DataStepExecutor()
    executor.register_table(
        "in",
        _arrow_ref("in", scenario["catalog_payload"]),
    )

    from_catalog = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["catalog_dsl"],
            inputs={},
            output_targets=[],
        )
    )
    assert from_catalog.has_errors is False
    assert _output_rows(from_catalog, "out") == scenario["catalog_expected"]

    explicit = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["explicit_dsl"],
            inputs={"in": _arrow_ref("in-explicit", scenario["explicit_payload"])},
            output_targets=[],
        )
    )
    assert explicit.has_errors is False
    assert _output_rows(explicit, "out") == scenario["explicit_expected"]


def test_submit_api_surface_register_tables_bulk_and_registered_resolution() -> None:
    scenario = EXECUTE_API_SCENARIOS["register_tables_bulk"]
    executor = DataStepExecutor()
    executor.register_tables(
        {
            "in_a": pa.table({"id": [1]}),
            "in_b": _arrow_ref("in_b", [{"id": 2}]),
        }
    )

    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={},
            output_targets=["out"],
        )
    )

    assert response.has_errors is False
    assert _dataset_rows(response.outputs["out"].payload) == scenario["expected_output"]


def test_submit_api_surface_resolves_set_input_from_registered_table() -> None:
    scenario = EXECUTE_API_SCENARIOS["resolve_registered_set_input"]
    executor = DataStepExecutor()
    executor.register_table(
        "in",
        DataSetRef(
            kind="arrow_table",
            location="dataset://in",
            payload=pa.Table.from_pylist(scenario["registered_payload"]),
        ),
    )

    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={},
            output_targets=["out"],
        )
    )

    assert response.has_errors is False
    assert _dataset_rows(response.outputs["out"].payload) == scenario["expected_output"]


def test_submit_api_surface_previous_outputs_are_available_for_next_request() -> None:
    scenario = EXECUTE_API_SCENARIOS["previous_outputs"]
    executor = DataStepExecutor()

    first = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["first_dsl"],
            inputs={"in": _arrow_ref("in", scenario["first_inputs"])},
        )
    )
    assert first.has_errors is False

    second = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["second_dsl"],
        )
    )

    assert second.has_errors is False
    assert _output_rows(second, "out") == scenario["expected_output"]


def test_session_submit_multiple_blocks_overwrite_same_name_dataset() -> None:
    session = Session()
    session.load("source", pa.table({"x": [90]}))

    result = session.submit(
        """
        data test;
            set source;
            x = x - 89;
            output test;
        run;

        data test;
            set test;
            x = 1;
            output;
            x = 2;
            output;
        run;
        """
    )

    assert result.success is True
    assert session["test"].to_pylist() == [{"x": 1}, {"x": 2}]


def test_submit_api_surface_multi_block_failure_preserves_prefix_and_runtime_observations() -> None:
    scenario = EXECUTE_API_SCENARIOS["multi_block_failure"]
    executor = DataStepExecutor()
    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": _arrow_ref("in", scenario["inputs"])},
        )
    )

    assert response.has_errors is True
    assert response.diagnostics[0].code == scenario["expected_code"]
    assert response.diagnostics[0].stage == "validate"
    assert response.diagnostics[0].location == scenario["expected_location"]
    assert response.diagnostics[0].source_text == (
        "data stage2; set missing_in; output stage2; run;"
    )

    strategy = executor.last_strategy_decision
    assert strategy is not None
    assert strategy.strategy == "row"
    assert strategy.engine == "rust"
    assert executor.last_runtime_backend == "rust"
    phase_metrics = executor.last_phase_metrics
    assert {
        "source_access_ms",
        "row_execution_ms",
        "output_export_ms",
    }.issubset(phase_metrics)
    assert all(value >= 0 for value in phase_metrics.values())

    # The successful first block remains in the working catalog even though
    # the second block returns a diagnostic.  Verify that contract through a
    # subsequent public execution rather than reaching into executor state.
    prefix_response = executor.execute(
        ExecuteRequest(
            dsl_text="data after; set stage1; output after; run;",
        )
    )
    assert prefix_response.has_errors is False
    assert _output_rows(prefix_response, "after") == [{"id": 1}]


def test_empty_inferred_output_targets_fail_before_native(monkeypatch) -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    native_executor = executor._rust_backend._executor
    assert native_executor is not None
    native_calls: list[object] = []

    def fail_if_called(payload):
        native_calls.append(payload)
        raise AssertionError("output target resolution must precede native execution")

    monkeypatch.setattr(native_executor, "execute_block", fail_if_called)
    response = executor.execute(
        ExecuteRequest(
            dsl_text="set in; run;",
            inputs={"in": _arrow_ref("in", [{"id": 1}])},
            output_targets=(),
        )
    )

    assert response.has_errors
    assert response.diagnostics[0].code == "REQ_INVALID_OUTPUT_TARGETS"
    assert response.diagnostics[0].stage == "resolve outputs"
    assert native_calls == []


def test_submit_api_surface_multi_block_row_execution_failure_publishes_no_outputs_and_preserves_prefix() -> None:
    # Execute-stage variant of the multi-block prefix contract: block 2 fails
    # mid-row, the failed response carries no output streams at all, the
    # completed successful block stays in the working catalog, and the failed
    # block's target is never published, so a later submit cannot resolve it.
    def snan(value: object) -> object:
        return Decimal("sNaN")

    executor = DataStepExecutor(runtime_backend="rust")
    response = executor.execute(
        ExecuteRequest(
            dsl_text=(
                "data ok1; set in; y = id; run; "
                "data ok2; set in; val = apply('snan', amount); y = val + 1; run;"
            ),
            inputs={
                "in": _arrow_ref(
                    "in",
                    [{"id": 1, "amount": 1.0}, {"id": 2, "amount": 2.0}, {"id": 3, "amount": 3.0}],
                )
            },
            host_function_context=HostFunctionContext(caller_globals={"snan": snan}),
        )
    )

    assert response.has_errors is True
    assert response.outputs == {}
    assert response.diagnostics[0].code == "RUNTIME_EXPRESSION_EVALUATION_ERROR"
    assert response.diagnostics[0].stage == "execute"

    # The successful first block remains in the working catalog even though the
    # second block failed during row execution. Verify that contract through a
    # subsequent public execution rather than reaching into executor state.
    prefix_response = executor.execute(
        ExecuteRequest(
            dsl_text="data after; set ok1; output after; run;",
        )
    )
    assert prefix_response.has_errors is False
    assert _output_rows(prefix_response, "after") == [
        {"id": 1, "amount": 1.0, "y": 1},
        {"id": 2, "amount": 2.0, "y": 2},
        {"id": 3, "amount": 3.0, "y": 3},
    ]

    # The failed block's target never reached the catalog, so a later submit
    # cannot resolve `ok2`.
    failed_block_response = executor.execute(
        ExecuteRequest(
            dsl_text="data z; set ok2; run;",
        )
    )
    assert failed_block_response.has_errors is True
    assert failed_block_response.diagnostics[0].code == "RUNTIME_SET_DATASET_NOT_FOUND"


def test_submit_api_surface_if_then_do_nesting_runs_in_rust() -> None:
    scenario = EXECUTE_API_SCENARIOS["if_do_nesting_limit"]
    executor = DataStepExecutor()
    nested_if_count = scenario["nested_if_count"]
    statements = ["data out", "set in"]
    statements.extend("if id = 1 then do" for _ in range(nested_if_count))
    statements.extend(["output out"])
    statements.extend("end" for _ in range(nested_if_count))
    statements.extend(["run"])
    dsl_text = "; ".join(statements) + ";"

    response = executor.execute(
        ExecuteRequest(
            dsl_text=dsl_text,
            inputs={"in": _arrow_ref("in", [{"id": 1}])},
        )
    )

    assert response.has_errors is False
    assert _output_rows(response, "out") == [{"id": 1}]


class ExecuteApiValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executor = DataStepExecutor()

    def test_accepts_valid_request(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["valid_request"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": _arrow_ref("in", [])},
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(response.diagnostics, ())
        self.assertEqual(response.notices, ())

    def test_rust_evaluates_missing_value_literal_in_if_condition(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["compat_notice"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": _arrow_ref("in", scenario["inputs"])
            },
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        # `.` is a missing literal owned by the native expression lexer;
        # `amount = .` compares the stored scalar to NULL instead of failing
        # with a runtime diagnostic.
        self.assertFalse(response.has_errors)
        self.assertEqual(response.diagnostics, ())
        self.assertEqual(_output_rows(response, "out"), [{"amount": None}])

    def test_rejects_empty_dsl(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["reject_empty_dsl"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": _arrow_ref("in", [])},
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertTrue(response.has_errors)
        self.assertEqual(response.diagnostics[0].code, scenario["expected_code"])

    def test_rejects_invalid_input_mapping(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["reject_invalid_inputs"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"": _arrow_ref("in", [])},
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertTrue(response.has_errors)
        self.assertEqual(response.diagnostics[0].code, scenario["expected_code"])

    def test_rejects_invalid_output_targets(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["reject_invalid_output_targets"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": _arrow_ref("in", [])},
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertTrue(response.has_errors)
        self.assertEqual(response.diagnostics[0].code, scenario["expected_code"])

    def test_resolves_output_targets_from_data_statement_when_request_omits_them(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["resolve_targets_data_statement"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": _arrow_ref("in", scenario["inputs"])},
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(_output_rows(response, "out"), scenario["expected_output"])

    def test_resolves_output_targets_from_data_and_output_statements(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["resolve_targets_data_and_output"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": _arrow_ref("in", scenario["inputs"])},
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(_output_rows(response, "out_a"), scenario["expected_out_a"])
        self.assertEqual(_output_rows(response, "out_b"), scenario["expected_out_b"])

    def test_simple_set_output_path_returns_arrow_rows(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["simple_set_reuse_rows"]
        input_rows = scenario["inputs"]
        response = self.executor.execute(
            ExecuteRequest(
                dsl_text=scenario["dsl"],
                inputs={"in": _arrow_ref("in", input_rows)},
            )
        )

        self.assertFalse(response.has_errors)
        self.assertEqual(_output_rows(response, "out"), input_rows)

    def test_pylist_conversion_is_only_performed_by_explicit_convert_api(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["pylist_only_explicit"]
        table = _CountingToPylistTable(scenario["rows"])
        response = ExecuteResponse(
            outputs={"out": DataSetRef(kind="arrow_table", location="dataset://out", payload=table)},
            outputs_arrow={"out": table},
        )

        arrow_converted = self.executor.convert_outputs(response, "arrow_table")
        self.assertFalse(arrow_converted.has_errors)
        self.assertEqual(table.to_pylist_calls, 0)

        pylist_converted = self.executor.convert_outputs(response, "pylist")
        self.assertFalse(pylist_converted.has_errors)
        self.assertEqual(pylist_converted.outputs["out"], scenario["rows"])
        self.assertEqual(table.to_pylist_calls, 1)

    def test_convert_outputs_reports_diagnostic_for_unsupported_format(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["convert_unsupported_format"]
        response = self.executor.execute(
            ExecuteRequest(
                dsl_text=scenario["dsl"],
                inputs={"in": _arrow_ref("in", scenario["inputs"])},
            )
        )

        converted = self.executor.convert_outputs(response, scenario["format"])

        self.assertTrue(converted.has_errors)
        self.assertEqual(converted.diagnostics[0].code, scenario["expected_code"])

    def test_convert_outputs_reports_diagnostic_when_dataset_cannot_be_converted(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["convert_failed_dataset"]
        response = ExecuteResponse(
            outputs={"out": DataSetRef(kind="memory", location="dataset://out", payload=object())}
        )

        converted = self.executor.convert_outputs(response, "arrow_table")

        self.assertTrue(converted.has_errors)
        self.assertEqual(converted.diagnostics[0].code, scenario["expected_code"])


def test_session_submit_backend_kwarg_restores_session_backend_after_call() -> None:
    scenario = EXECUTOR_SCENARIOS["submit_backend_override"]
    session = Session(backend="rust")
    session.load("inp", pa.table(scenario["inputs"]))

    session.submit(scenario["dsl"], backend="rust")
    second = session.submit("data out2; set inp; where x >= 20; output out2; run;")

    assert second.success is True
    assert session["out2"].to_pylist() == [{"name": "foo bar", "x": 20}]


def test_limulus_submit_backend_kwarg_uses_specified_backend() -> None:
    scenario = EXECUTOR_SCENARIOS["submit_backend_override"]

    result = limulus.submit(
        scenario["dsl"],
        backend="rust",
        inp=pa.table(scenario["inputs"]),
    )

    assert result.success is True
    assert result.datasets["out"].to_pylist() == scenario["expected_output"]


def test_session_submit_hides_result_representation_by_default() -> None:
    session = Session()
    session.load("inp", [{"id": 1}])

    result = session.submit("data out; set inp; output out; run;")

    assert result.success is True
    assert repr(result) == ""


def test_session_submit_accepts_dictionary_tables_as_data_step_input() -> None:
    session = Session()
    session.load("src", [{"id": 1, "amount": 10}, {"id": 2, "amount": 20}])

    result = session.submit(
        """
        data out;
        set dictionary.tables;
        where memname = 'SRC';
        output out;
        run;
        """
    )

    assert result.success is True
    assert result.datasets["out"].to_pylist() == [
        {"LIBNAME": "WORK", "MEMNAME": "SRC", "MEMTYPE": "DATA", "MEMLABEL": "", "NOBS": 2, "NVAR": 2}
    ]


def test_session_submit_accepts_dictionary_columns_in_later_block_after_dataset_creation() -> None:
    session = Session()
    session.load("src", [{"id": 1, "amount": 10}])

    result = session.submit(
        """
        data staged;
        set src;
        output staged;
        run;

        data cols;
        set dictionary.columns;
        where memname = 'STAGED';
        output cols;
        run;
        """
    )

    assert result.success is True
    assert result.datasets["cols"].to_pylist() == [
        {
            "LIBNAME": "WORK",
            "MEMNAME": "STAGED",
            "MEMTYPE": "DATA",
            "NAME": "id",
            "TYPE": "int64",
            "VARNUM": 1,
            "LABEL": "",
            "FORMAT": "",
            "INFORMAT": "",
        },
        {
            "LIBNAME": "WORK",
            "MEMNAME": "STAGED",
            "MEMTYPE": "DATA",
            "NAME": "amount",
            "TYPE": "int64",
            "VARNUM": 2,
            "LABEL": "",
            "FORMAT": "",
            "INFORMAT": "",
        },
    ]


def test_module_submit_dictionary_tables_prefers_latest_generated_dataset_over_initial_input() -> None:
    result = limulus.submit(
        """
        data seed;
        set seed;
        keep id;
        output seed;
        run;

        data out;
        set dictionary.tables;
        where memname = 'SEED';
        output out;
        run;
        """,
        seed=[{"id": 1, "amount": 10}],
    )

    assert result.success is True
    assert result.datasets["out"].to_pylist() == [
        {"LIBNAME": "WORK", "MEMNAME": "SEED", "MEMTYPE": "DATA", "MEMLABEL": "", "NOBS": 1, "NVAR": 1}
    ]


def test_module_submit_keeps_result_representation_visible() -> None:
    result = limulus.submit("data out; set inp; output out; run;", inp=[{"id": 1}])

    assert result.success is True
    assert "SubmitResult(" in repr(result)


def test_advanced_input_normalization_unifies_supported_carriers_and_preserves_provenance() -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    source_metadata = {"memlabel": "source"}
    inputs = {
        "arrow": DataSetRef(
            kind="arrow_table",
            location="source://arrow",
            payload=pa.table({"id": [1]}),
            metadata=source_metadata,
        ),
        "polars": DataSetRef(
            kind="polars",
            location="source://polars",
            payload=pl.DataFrame({"id": [2]}),
            metadata=source_metadata,
        ),
        "pandas": DataSetRef(
            kind="pandas",
            location="source://pandas",
            payload=pd.DataFrame({"id": [3]}),
            metadata=source_metadata,
        ),
        "memory": DataSetRef(
            kind="memory",
            location="source://memory",
            payload=[{"id": 4}],
            metadata=source_metadata,
        ),
    }

    normalized, diagnostic = executor.normalize_execution_inputs(inputs)

    assert diagnostic is None
    assert normalized is not None
    assert normalized.preserve_memory_presentation is False
    assert all(ref.kind == "arrow_table" for ref in normalized.inputs.values())
    assert [normalized.inputs[name].payload.to_pylist() for name in inputs] == [
        [{"id": value}] for value in (1, 2, 3, 4)
    ]
    for name, ref in normalized.inputs.items():
        assert ref.location == inputs[name].location
        assert ref.metadata == source_metadata


def test_registered_memory_input_does_not_request_memory_output_presentation() -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    observed_contexts = []

    def observe(context):
        observed_contexts.append(context)
        return RuntimeExecutionResult(
            outputs={
                "out": DataSetRef(
                    kind="arrow_table",
                    location="dataset://out",
                    payload=pa.table({"id": [1]}),
                )
            },
            diagnostics=(),
            actual_engine="rust",
            native_invoked=True,
            phase_metrics={},
        )

    executor._rust_backend.execute_typed = observe
    executor.register_table(
        "in",
        DataSetRef(
            kind="memory",
            location="registered://in",
            payload=[{"id": 1}],
        ),
    )

    response = executor.execute(
        ExecuteRequest(dsl_text="data out; set in; output out; run;")
    )

    assert response.has_errors is False
    assert len(observed_contexts) == 1
    assert observed_contexts[0].resolved_inputs["in"].kind == "arrow_table"
    assert response.outputs["out"].kind == "arrow_table"


def test_memory_sequence_normalization_uses_first_seen_union_for_later_columns() -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    input_ref = DataSetRef(
        kind="memory",
        location="source://sparse",
        payload=[
            {"first": 1},
            {"second": 2},
            {"first": 3, "third": 4},
        ],
        metadata={"memlabel": "sparse"},
    )

    normalized, diagnostic = executor.normalize_execution_inputs({"in": input_ref})

    assert diagnostic is None
    assert normalized is not None
    normalized_ref = normalized.inputs["in"]
    assert normalized_ref.kind == "arrow_table"
    assert normalized_ref.location == input_ref.location
    assert normalized_ref.metadata == input_ref.metadata
    assert normalized_ref.payload.schema.names == ["first", "second", "third"]
    assert normalized_ref.payload.to_pylist() == [
        {"first": 1, "second": None, "third": None},
        {"first": None, "second": 2, "third": None},
        {"first": 3, "second": None, "third": 4},
    ]


def test_advanced_memory_input_is_arrow_before_rust_and_keeps_memory_presentation() -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    observed_contexts = []

    def observe(context):
        observed_contexts.append(context)
        output = pa.table({"id": [1], "missing": [None]})
        return RuntimeExecutionResult(
            outputs={"out": DataSetRef(kind="arrow_table", location="dataset://out", payload=output)},
            diagnostics=(),
            actual_engine="rust",
            native_invoked=True,
            phase_metrics={},
        )

    executor._rust_backend.execute_typed = observe
    response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set in; output out; run;",
            inputs={
                "in": DataSetRef(
                    kind="memory",
                    location="source://in",
                    payload=[{"id": 1}],
                    metadata={"memlabel": "source"},
                )
            },
        )
    )

    assert response.has_errors is False
    assert len(observed_contexts) == 1
    normalized = observed_contexts[0].resolved_inputs["in"]
    assert normalized.kind == "arrow_table"
    assert normalized.location == "source://in"
    assert normalized.metadata == {"memlabel": "source"}
    assert response.outputs["out"].kind == "memory"
    assert response.outputs["out"].payload == [{"id": 1}]
    assert response.outputs_arrow["out"].to_pylist() == [{"id": 1, "missing": None}]


def test_submit_api_surface_registry_canonicalizes_duplicate_alias_registrations_last_wins() -> None:
    """``a``, ``A``, ``WORK.A`` share one canonical registry entry and the last registration is the only effective input."""
    executor = DataStepExecutor()
    first_ref = _arrow_ref("a", [{"id": 1}])
    second_ref = _arrow_ref("A", [{"id": 2}])
    executor.register_table("a", first_ref)
    executor.register_table("A", second_ref)

    # One canonical entry, keyed by the canonical dataset key.
    registered = executor._registered_tables
    assert len(registered) == 1
    canonical_key = executor._io_service.dataset_name_key("A")
    assert list(registered) == [canonical_key]
    assert registered[canonical_key] is second_ref

    # The last registration's payload is what ``set a;`` resolves.
    response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set a; output out; run;",
            inputs={},
            output_targets=["out"],
        )
    )
    assert response.has_errors is False
    assert _output_rows(response, "out") == [{"id": 2}]


def test_submit_api_surface_registry_re_registration_with_work_prefix_keeps_single_entry() -> None:
    """Re-registering via ``WORK.A`` replaces the same canonical entry and catalog + executor reference the same payload."""
    executor = DataStepExecutor()
    first_ref = _arrow_ref("a", [{"id": 1}])
    last_ref = _arrow_ref("WORK.A", [{"id": 3}])
    executor.register_table("a", first_ref)
    executor.register_table("WORK.A", last_ref)

    registered = executor._registered_tables
    assert len(registered) == 1
    canonical_key = executor._io_service.dataset_name_key("WORK.A")
    assert list(registered) == [canonical_key]
    assert registered[canonical_key] is last_ref

    response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set a; output out; run;",
            inputs={},
            output_targets=["out"],
        )
    )
    assert response.has_errors is False
    assert _output_rows(response, "out") == [{"id": 3}]


def test_submit_api_surface_register_tables_bulk_canonicalizes_last_wins() -> None:
    """Bulk registration in mapping order stores one canonical entry per key, last one wins."""
    executor = DataStepExecutor()
    executor.register_tables(
        {
            "a": [{"id": 1}],
            "A": [{"id": 2}],
        }
    )

    registered = executor._registered_tables
    assert len(registered) == 1
    canonical_key = executor._io_service.dataset_name_key("A")
    assert list(registered) == [canonical_key]
    assert _dataset_rows(registered[canonical_key].payload) == [{"id": 2}]

    response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set a; output out; run;",
            inputs={},
            output_targets=["out"],
        )
    )
    assert response.has_errors is False
    assert _output_rows(response, "out") == [{"id": 2}]


def test_submit_api_surface_unregister_via_any_alias_removes_whole_canonical_entry() -> None:
    """Unregistering through any compatible alias removes the single canonical entry so no stale alias resolves."""
    executor = DataStepExecutor()
    executor.register_table("a", _arrow_ref("a", [{"id": 1}]))
    executor.register_table("A", _arrow_ref("A", [{"id": 1}]))

    # Either spelling removes the one canonical entry.
    assert executor.unregister_table("a") is True

    assert executor._registered_tables == {}
    for alias in ("a", "A", "work.a"):
        response = executor.execute(
            ExecuteRequest(
                dsl_text=f"data out; set {alias}; output out; run;",
                inputs={},
                output_targets=["out"],
            )
        )
        assert response.has_errors is True
        assert response.diagnostics[0].code == "RUNTIME_SET_DATASET_NOT_FOUND"

    assert executor.unregister_table("a") is False
    with pytest.raises(KeyError):
        executor.unregister_table("work.a", missing_ok=False)


def test_submit_api_surface_unregister_work_alias_after_cased_registration() -> None:
    """``work.a`` alias unregister clears an entry registered as ``A``."""
    executor = DataStepExecutor()
    executor.register_table("A", _arrow_ref("A", [{"id": 5}]))

    assert executor.unregister_table("work.a") is True
    assert executor._registered_tables == {}

    response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set A; output out; run;",
            inputs={},
            output_targets=["out"],
        )
    )
    assert response.has_errors is True
    assert response.diagnostics[0].code == "RUNTIME_SET_DATASET_NOT_FOUND"


def test_submit_api_surface_unregister_missing_name_keeps_existing_semantics() -> None:
    """Missing-name unregister keeps the False / KeyError contract."""
    executor = DataStepExecutor()
    assert executor.unregister_table("missing") is False
    with pytest.raises(KeyError):
        executor.unregister_table("missing", missing_ok=False)
    assert executor.unregister_tables(["missing"]) == {"missing": False}


def test_execute_request_rejects_duplicate_canonical_input_aliases_fail_closed() -> None:
    """Duplicate canonical keys in one request's inputs fail closed with a diagnostic naming both raw spellings."""
    executor = DataStepExecutor()
    response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set a; output out; run;",
            inputs={
                "a": _arrow_ref("a", [{"id": 1}]),
                "A": _arrow_ref("A", [{"id": 2}]),
            },
            output_targets=["out"],
        )
    )

    assert response.has_errors is True
    diagnostic = response.diagnostics[0]
    assert diagnostic.code == "RUNTIME_REQUEST_INPUT_ALIAS_CONFLICT"
    assert "'a'" in diagnostic.message
    assert "'A'" in diagnostic.message
    assert response.outputs.get("out") is None


def test_execute_request_rejects_work_alias_conflict_and_identifies_canonical_key() -> None:
    """``WORK.a`` + ``a`` request inputs are ambiguous and the diagnostic names the canonical key."""
    executor = DataStepExecutor()
    response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set a; output out; run;",
            inputs={
                "a": _arrow_ref("a", [{"id": 1}]),
                "WORK.a": _arrow_ref("WORK.a", [{"id": 2}]),
            },
            output_targets=["out"],
        )
    )

    assert response.has_errors is True
    diagnostic = response.diagnostics[0]
    assert diagnostic.code == "RUNTIME_REQUEST_INPUT_ALIAS_CONFLICT"
    assert "'a'" in diagnostic.message
    assert "'WORK.a'" in diagnostic.message
    # The canonical key differs from both raw spellings in this variant.
    assert "canonical dataset key 'A'" in diagnostic.message
    assert response.outputs.get("out") is None


def test_session_load_bulk_load_and_set_alias_paths_resolve_last_registration() -> None:
    """Session load + bulk load on the same canonical key keep one identity and SET resolves the final payload."""
    session = Session()
    session.load("a", [{"id": 1}])
    session.loads({"A": [{"id": 2}]})

    assert len(session._executor._registered_tables) == 1

    result = session.submit("data out; set a; output out; run;")
    assert result.success is True
    assert result.datasets["out"].to_pylist() == [{"id": 2}]


def test_session_unload_work_alias_clears_entry_and_set_no_longer_resolves() -> None:
    """Session unload through ``work.a`` removes the canonical entry so any spelling stops resolving."""
    session = Session()
    session.load("a", [{"id": 1}])

    assert session.unload("work.a") is True

    result = session.submit("data out; set A; output out; run;")
    assert result.success is False
    error_codes = [entry.code for entry in result.log if entry.severity == "error"]
    assert "RUNTIME_SET_DATASET_NOT_FOUND" in error_codes


def test_execute_request_runtime_backend_override_does_not_mutate_executor_default() -> None:
    """A request-scoped runtime backend drives one execution without changing the executor's shared default."""
    executor = DataStepExecutor(runtime_backend="auto")
    assert executor._runtime_backend_preference == "auto"

    override_response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set in; run;",
            inputs={"in": _arrow_ref("in", [{"id": 1}])},
            output_targets=["out"],
            runtime_backend="python",
        )
    )

    assert override_response.has_errors is False
    override_decision = executor.last_strategy_decision
    assert override_decision is not None
    assert override_decision.requested_mode == "python"
    assert override_decision.reason_codes == ("LEGACY_PYTHON_BACKEND_ALIAS",)

    default_response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set in; run;",
            inputs={"in": _arrow_ref("in", [{"id": 2}])},
            output_targets=["out"],
        )
    )

    assert default_response.has_errors is False
    default_decision = executor.last_strategy_decision
    assert default_decision is not None
    assert default_decision.requested_mode == "auto"
    assert default_decision.reason_codes == ("ROW_ONLY_STRATEGY",)

    # The executor-level default backend is unchanged after
    # both the override and the default request ran on the same executor.
    assert executor._runtime_backend_preference == "auto"


def test_execute_request_runtime_backend_invalid_label_preserves_normalization_diagnostic() -> None:
    """An invalid request-scoped backend label fails with the existing request diagnostic and leaves the executor default untouched."""
    executor = DataStepExecutor(runtime_backend="auto")

    response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set in; run;",
            inputs={"in": _arrow_ref("in", [{"id": 1}])},
            output_targets=["out"],
            runtime_backend="bogus",
        )
    )

    assert response.has_errors is True
    assert response.diagnostics
    diagnostic = response.diagnostics[0]
    assert diagnostic.code == "REQ_INVALID_EXECUTION_MODE"
    assert diagnostic.stage == "validate"
    assert executor._runtime_backend_preference == "auto"


def test_execute_request_runtime_backend_override_wins_over_different_executor_default() -> None:
    """A request-level override wins over a different executor-level default, and the None field keeps the default path unchanged."""
    executor = DataStepExecutor(runtime_backend="python")

    override_response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set in; run;",
            inputs={"in": _arrow_ref("in", [{"id": 1}])},
            output_targets=["out"],
            runtime_backend="rust",
        )
    )

    assert override_response.has_errors is False
    override_decision = executor.last_strategy_decision
    assert override_decision is not None
    assert override_decision.requested_mode == "rust"
    assert override_decision.reason_codes == ("EXPLICIT_RUST_ROW",)

    default_response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set in; run;",
            inputs={"in": _arrow_ref("in", [{"id": 2}])},
            output_targets=["out"],
        )
    )

    assert default_response.has_errors is False
    default_decision = executor.last_strategy_decision
    assert default_decision is not None
    assert default_decision.requested_mode == "python"
    assert default_decision.reason_codes == ("LEGACY_PYTHON_BACKEND_ALIAS",)

    assert executor._runtime_backend_preference == "python"


def test_session_submit_backend_override_is_request_scoped_and_keeps_session_default() -> None:
    """Success, default, and failed override submits all leave the session-level default backend unchanged."""
    session = Session(backend="auto")
    session.load("in", [{"id": 1}])

    override_result = session.submit(
        "data out; set in; output out; run;", backend="rust"
    )
    assert override_result.success is True
    override_decision = session._executor.last_strategy_decision
    assert override_decision is not None
    assert override_decision.requested_mode == "rust"
    assert override_decision.reason_codes == ("EXPLICIT_RUST_ROW",)

    default_result = session.submit("data out; set in; output out; run;")
    assert default_result.success is True
    default_decision = session._executor.last_strategy_decision
    assert default_decision is not None
    assert default_decision.requested_mode == "auto"
    assert default_decision.reason_codes == ("ROW_ONLY_STRATEGY",)

    # A failed override submit takes the same request-scoped path: the
    # label still travels with the request even when execution fails.
    failed_result = session.submit(
        "data out; set missing_table; output out; run;", backend="python"
    )
    assert failed_result.success is False

    # No submit outcome rewrites the session-level default.
    assert session._executor._runtime_backend_preference == "auto"


def test_concurrent_submit_backend_overrides_are_request_scoped_and_default_survives_race() -> None:
    """Concurrent submits with different backend overrides each observe their own request label and the executor default survives the race.

    Deterministic post-fix: each thread's backend label travels inside
    ``ExecuteRequest.runtime_backend``, so normalization sees
    exactly that request's own label, and no shared backend preference is
    mutated or restored around ``submit``. The final invariant therefore
    holds on every round. Session-level catalog, ``last_*`` observations,
    and parallel catalog commits remain explicitly NOT thread-safe,
    so this test never asserts on shared ``last_*`` state from
    concurrent threads.
    """
    session = Session(backend="auto")
    session.load("in", [{"id": 1}])
    original_default = session._executor._runtime_backend_preference
    assert original_default == "auto"

    # --- Deterministic race round -------------------------------------------------
    # Thread A is pinned inside its submit (inside the executor, via a slow
    # apply callback) so its save/set/execute window stays open. Thread B
    # starts only after that window opened and is held open until A's submit
    # has fully returned. Pre-fix, B's save therefore captures A's active
    # override and B's restore writes it back LAST, leaving the shared
    # default permanently corrupted. Post-fix there is no save/restore, so
    # the ordering is harmless and the default is untouched.
    window_open = threading.Event()
    release_a = threading.Event()
    release_b = threading.Event()

    def pause_a(value):
        window_open.set()
        assert release_a.wait(timeout=30)
        return value

    def pause_b(value):
        assert release_b.wait(timeout=30)
        return value

    thread_results: dict[str, bool | None] = {"a": None, "b": None}

    def rust_worker_a() -> None:
        result = session.submit(
            "data out_a; set in; x = apply('pause_a', id); output out_a; run;",
            backend="rust",
            functions={"pause_a": pause_a},
        )
        thread_results["a"] = result.success

    def rust_worker_b() -> None:
        result = session.submit(
            "data out_b; set in; output out_b; run;",
            backend="rust",
            functions={"pause_b": pause_b},
        )
        thread_results["b"] = result.success

    thread_a = threading.Thread(target=rust_worker_a)
    thread_a.start()
    assert window_open.wait(timeout=30)
    thread_b = threading.Thread(target=rust_worker_b)
    thread_b.start()
    release_a.set()
    thread_a.join(timeout=30)
    release_b.set()
    thread_b.join(timeout=30)

    # Each thread observed its own request's label: both rust requests
    # succeeded — neither thread's override leaked into the other's result.
    assert thread_results == {"a": True, "b": True}

    # The shared default backend keeps its original value after the
    # interleaved submits (pre-fix this round left it corrupted).
    assert session._executor._runtime_backend_preference == original_default

    # --- Lockstep rounds ------------------------------------------------------------
    # Plain synchronized submits on one session. Post-fix the per-request
    # label makes these invariants hold every round; the shared ``last_*``
    # observation channels are explicitly not asserted.
    iterations = 12
    barrier = threading.Barrier(2)
    failure_codes: list[list[str] | None] = [None] * iterations
    success_flags: list[bool | None] = [None] * iterations

    def invalid_label_worker() -> None:
        for index in range(iterations):
            barrier.wait()
            result = session.submit(
                "data out_a; set in; output out_a; run;",
                backend="bogus",
            )
            failure_codes[index] = [
                entry.code for entry in result.log if entry.severity == "error"
            ]

    def valid_label_worker() -> None:
        for index in range(iterations):
            barrier.wait()
            result = session.submit(
                "data out_b; set in; output out_b; run;",
                backend="rust",
            )
            success_flags[index] = result.success

    thread_a = threading.Thread(target=invalid_label_worker)
    thread_b = threading.Thread(target=valid_label_worker)
    thread_a.start()
    thread_b.start()
    thread_a.join()
    thread_b.join()

    # Each thread observed its own request's label: the invalid label always
    # failed with the existing strategy diagnostic, and the rust label always
    # succeeded — neither thread's override leaked into the other's result.
    assert all(codes == ["REQ_INVALID_EXECUTION_MODE"] for codes in failure_codes)
    assert all(flag is True for flag in success_flags)

    # The executor default is still its original value after all rounds.
    assert session._executor._runtime_backend_preference == original_default


def test_unsupported_advanced_memory_input_fails_before_native_with_source_and_cause() -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    native_calls = []

    def fail_if_called(context):
        native_calls.append(context)
        raise AssertionError("unsupported input must not reach native")

    executor._rust_backend.execute_typed = fail_if_called
    response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set in; output out; run;",
            inputs={
                "in": DataSetRef(
                    kind="memory",
                    location="source://broken",
                    payload=[{"id": object()}],
                )
            },
        )
    )

    assert response.has_errors is True
    assert native_calls == []
    assert executor.last_runtime_backend == "rust"
    diagnostic = response.diagnostics[0]
    assert diagnostic.code == "RUNTIME_RUST_BRIDGE_INPUT_NOT_ARROW"
    assert diagnostic.location == "dataset:in"
    assert "source://broken" in diagnostic.message
    assert "object" in diagnostic.message.lower() or "arrow" in diagnostic.message.lower()
