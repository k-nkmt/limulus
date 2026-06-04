import limulus
import unittest
from unittest.mock import patch

import pyarrow as pa

from limulus import Session
from limulus.runtime import DataStepExecutor
from limulus.models import DataSetRef, ExecuteRequest, ExecuteResponse


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


def test_executor_split_data_step_blocks_uses_parser_before_scanner() -> None:
    executor = DataStepExecutor()
    dsl_text = "* comment;\ndata out;\n  set in;\nrun;"

    with patch.object(executor._block_splitter, "_split_with_scanner", side_effect=AssertionError("scanner fallback should not run")):
        blocks = executor._split_data_step_blocks(dsl_text)

    assert blocks == ("data out;\n  set in;\nrun;",)


def test_executor_split_data_step_blocks_falls_back_when_parser_split_fails() -> None:
    executor = DataStepExecutor()
    dsl_text = "* comment;\ndata out;\n  set in;\nrun;"

    with patch.object(executor._block_splitter, "_split_with_parser", return_value=None):
        blocks = executor._split_data_step_blocks(dsl_text)

    assert blocks == ("data out;\n  set in;\nrun;",)


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
    session = Session(backend="python")
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

    globals()["myfunc"] = myfunc

    executor = DataStepExecutor(runtime_backend="rust")
    response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set in; if apply('myfunc', amount) = 30 then output out; run;",
            inputs={"in": _arrow_ref("in", [{"id": 1, "amount": 10}, {"id": 2, "amount": 15}, {"id": 3, "amount": 20}])},
            output_targets=["out"],
        )
    )

    assert response.has_errors is False
    assert _output_rows(response, "out") == [{"id": 3, "amount": 20}]


def test_submit_api_surface_assignment_function_dispatch_uses_runtime_registry_entrypoint() -> None:
    def double(value):
        return value * 2

    globals()["double"] = double

    executor = DataStepExecutor(runtime_backend="rust")
    response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set in; doubled = apply('double', amount); output out; keep id doubled; run;",
            inputs={"in": _arrow_ref("in", [{"id": 1, "amount": 4}])},
            output_targets=["out"],
        )
    )

    assert response.has_errors is False
    assert _output_rows(response, "out") == [{"id": 1, "doubled": 8.0}]


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


def test_submit_api_surface_python_collision_with_internal_reference_variable_is_case_insensitive() -> None:
    executor = DataStepExecutor(runtime_backend="python")
    response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set in(in=in_flag); output out; run;",
            inputs={
                "in": DataSetRef(
                    kind="memory",
                    location="dataset://in",
                    payload=[{"id": 1, "IN_FLAG": False}],
                )
            },
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
        DataSetRef(kind="memory", location="dataset://in", payload=scenario["catalog_payload"]),
    )

    from_catalog = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["catalog_dsl"],
            inputs={},
            output_targets=[],
        )
    )
    assert from_catalog.has_errors is False
    assert from_catalog.outputs["out"].payload == scenario["catalog_expected"]

    explicit = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["explicit_dsl"],
            inputs={"in": DataSetRef(kind="memory", location="dataset://in-explicit", payload=scenario["explicit_payload"])},
            output_targets=[],
        )
    )
    assert explicit.has_errors is False
    assert explicit.outputs["out"].payload == scenario["explicit_expected"]


def test_submit_api_surface_register_tables_bulk_and_registered_resolution() -> None:
    scenario = EXECUTE_API_SCENARIOS["register_tables_bulk"]
    executor = DataStepExecutor()
    executor.register_tables(
        {
            "in_a": pa.table({"id": [1]}),
            "in_b": DataSetRef(kind="memory", location="dataset://in_b", payload=[{"id": 2}]),
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
            payload=scenario["registered_payload"],
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
            inputs={"in": DataSetRef(kind="memory", location="dataset://in", payload=scenario["first_inputs"])},
        )
    )
    assert first.has_errors is False

    second = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["second_dsl"],
        )
    )

    assert second.has_errors is False
    assert second.outputs["out"].payload == scenario["expected_output"]


def test_session_submit_prefers_prior_block_output_over_existing_same_name_dataset() -> None:
    session = Session()
    session.load("test", pa.table({"x": [90, 91]}))

    result = session.submit(
        """
        data test;
            x = 1;
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


def test_submit_api_surface_multi_block_failure_contains_block_location() -> None:
    scenario = EXECUTE_API_SCENARIOS["multi_block_failure"]
    executor = DataStepExecutor()
    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": DataSetRef(kind="memory", location="dataset://in", payload=scenario["inputs"])},
        )
    )

    assert response.has_errors is True
    assert response.diagnostics[0].code == scenario["expected_code"]
    assert response.diagnostics[0].location == scenario["expected_location"]


def test_submit_api_surface_if_then_do_nesting_limit() -> None:
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
            inputs={"in": DataSetRef(kind="memory", location="dataset://in", payload=[{"id": 1}])},
        )
    )

    assert response.has_errors is True
    assert response.diagnostics[0].code == scenario["expected_code"]


class ExecuteApiValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executor = DataStepExecutor()

    def test_accepts_valid_request(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["valid_request"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": DataSetRef(kind="memory", location="dataset://in")},
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(response.diagnostics, ())
        self.assertEqual(response.notices, ())

    def test_attaches_compatibility_notices_to_execution_response(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["compat_notice"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="memory",
                    location="dataset://in",
                    payload=scenario["inputs"],
                )
            },
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(len(response.notices), 1)
        self.assertEqual(response.notices[0].id, scenario["notice_id"])

    def test_rejects_empty_dsl(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["reject_empty_dsl"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": DataSetRef(kind="memory", location="dataset://in")},
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertTrue(response.has_errors)
        self.assertEqual(response.diagnostics[0].code, scenario["expected_code"])

    def test_rejects_invalid_input_mapping(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["reject_invalid_inputs"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"": DataSetRef(kind="memory", location="dataset://in")},
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertTrue(response.has_errors)
        self.assertEqual(response.diagnostics[0].code, scenario["expected_code"])

    def test_rejects_invalid_output_targets(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["reject_invalid_output_targets"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": DataSetRef(kind="memory", location="dataset://in")},
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertTrue(response.has_errors)
        self.assertEqual(response.diagnostics[0].code, scenario["expected_code"])

    def test_resolves_output_targets_from_data_statement_when_request_omits_them(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["resolve_targets_data_statement"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": DataSetRef(kind="memory", location="dataset://in", payload=scenario["inputs"])},
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].payload, scenario["expected_output"])

    def test_resolves_output_targets_from_data_and_output_statements(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["resolve_targets_data_and_output"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": DataSetRef(kind="memory", location="dataset://in", payload=scenario["inputs"])},
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out_a"].payload, scenario["expected_out_a"])
        self.assertEqual(response.outputs["out_b"].payload, scenario["expected_out_b"])

    def test_simple_set_output_path_reuses_row_objects_without_extra_materialize(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["simple_set_reuse_rows"]
        input_rows = scenario["inputs"]
        response = self.executor.execute(
            ExecuteRequest(
                dsl_text=scenario["dsl"],
                inputs={"in": DataSetRef(kind="memory", location="dataset://in", payload=input_rows)},
            )
        )

        self.assertFalse(response.has_errors)
        self.assertIs(response.outputs["out"].payload[0], input_rows[0])
        self.assertIs(response.outputs["out"].payload[1], input_rows[1])

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
                inputs={"in": DataSetRef(kind="memory", location="dataset://in", payload=scenario["inputs"])},
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
    session = Session(backend="python")
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
