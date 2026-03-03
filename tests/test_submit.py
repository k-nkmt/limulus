import limulus
from limulus import Session
from limulus.runtime import DataStepExecutor
from limulus.models import DataSetRef, ExecuteRequest


SUBMIT_SCENARIOS = {
    "module_submit_basic": {
        "overview": "Module submit registers input datasets and executes successfully",
        "dsl": """
        data out;
        set inp;
        if x >= 0 then output out;
        run;
        """,
        "inputs": {"inp": [{"x": 1}, {"x": -1}]},
        "expected_output": [{"x": 1}],
    },
    "module_run_alias": {
        "overview": "Module run behaves as a submit alias",
        "dsl": """
        data out;
        set inp;
        output out;
        run;
        """,
        "inputs": {"inp": [{"x": 3}]},
        "expected_output": [{"x": 3}],
    },
}


EXECUTOR_SCENARIOS = {
    "convert_outputs_parser_backend_control": {
        "overview": "Output conversion and parser backend switching work correctly",
        "dsl": "data out; set in; output out; run;",
        "inputs": {"in": [{"id": 1}]},
        "output_targets": None,
        "expected_output": [{"id": 1}],
    },
    "invalid_output_targets": {
        "overview": "Returns diagnostics when output targets are unresolved",
        "dsl": "set in; run;",
        "inputs": {"in": [{"id": 1}]},
        "output_targets": [],
        "expected_code": "REQ_INVALID_OUTPUT_TARGETS",
    },
    "explicit_output_targets_precedence": {
        "overview": "Explicit output_targets take precedence over inferred targets",
        "dsl": "data inferred; set in; run;",
        "inputs": {"in": [{"id": 1}]},
        "output_targets": ["explicit"],
        "expected_keys": {"explicit"},
        "expected_output": [{"id": 1}],
    },
    "invalid_syntax_parse_diagnostic": {
        "overview": "Returns PARSE diagnostic for invalid syntax",
        "dsl": "data out; ???; run;",
        "inputs": {"in": None},
        "output_targets": ["out"],
        "expected_code": "PARSE_UNSUPPORTED_STATEMENT",
        "expected_location": "statement:2",
    },
    "set_input_case_insensitive_work_prefix": {
        "overview": "Resolves SET input with WORK prefix and case-insensitive name matching",
        "dsl": "data out; set WORK.My_Table; output out; run;",
        "inputs": None,
        "output_targets": None,
        "expected_output": [{"id": 1, "amount": 10}],
    },
    "work_prefix_in_output_targets": {
        "overview": "Supports WORK prefix in output targets",
        "dsl": "data work.out; set in; output WORK.OUT; run;",
        "inputs": {"in": [{"id": 1}]},
        "output_targets": None,
        "expected_key": "out",
        "expected_output": [{"id": 1}],
    },
    "explicit_inputs_precedence": {
        "overview": "Explicit inputs take precedence over registered tables",
        "dsl": "data out; set in; output out; run;",
        "inputs": {"in": [{"id": 2, "amount": 20}]},
        "output_targets": ["out"],
        "expected_output": [{"id": 2, "amount": 20}],
    },
}


def _memory_ref(name: str, payload: list[dict] | None) -> DataSetRef:
    return DataSetRef(kind="memory", location=f"dataset://{name}", payload=payload)


def test_module_submit_registers_input_dataset_and_executes() -> None:
    scenario = SUBMIT_SCENARIOS["module_submit_basic"]
    result = limulus.submit(scenario["dsl"], **scenario["inputs"])

    assert result.success is True
    assert result.datasets["out"].to_pylist() == scenario["expected_output"]


def test_module_run_is_submit_alias() -> None:
    scenario = SUBMIT_SCENARIOS["module_run_alias"]
    submit_result = limulus.submit(scenario["dsl"], **scenario["inputs"])
    run_result = limulus.run(scenario["dsl"], **scenario["inputs"])

    assert submit_result.success is True
    assert run_result.success is True
    assert run_result.datasets["out"].to_pylist() == scenario["expected_output"]


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


def test_submit_api_surface_convert_outputs_and_parser_backend_control() -> None:
    scenario = EXECUTOR_SCENARIOS["convert_outputs_parser_backend_control"]
    executor = DataStepExecutor()
    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": _memory_ref("in", scenario["inputs"]["in"])},
        )
    )
    converted = executor.convert_outputs(response, "arrow_table")

    assert converted.has_errors is False
    assert converted.outputs["out"].to_pylist() == scenario["expected_output"]

    executor.set_parser_backend("rust")
    rust_result = executor.execute(
        ExecuteRequest(
            dsl_text="data out2; set in; output out2; run;",
            inputs={"in": _memory_ref("in", scenario["inputs"]["in"])},
        )
    )
    assert rust_result.has_errors is False
    assert executor.last_parser_backend == "rust"


def test_submit_api_surface_reports_unresolved_output_targets() -> None:
    scenario = EXECUTOR_SCENARIOS["invalid_output_targets"]
    executor = DataStepExecutor()

    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": _memory_ref("in", scenario["inputs"]["in"])},
            output_targets=scenario["output_targets"],
        )
    )

    assert response.has_errors is True
    assert response.diagnostics[0].code == scenario["expected_code"]


def test_submit_api_surface_explicit_output_targets_take_precedence() -> None:
    scenario = EXECUTOR_SCENARIOS["explicit_output_targets_precedence"]
    executor = DataStepExecutor()

    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": _memory_ref("in", scenario["inputs"]["in"])},
            output_targets=scenario["output_targets"],
        )
    )

    assert response.has_errors is False
    assert set(response.outputs.keys()) == scenario["expected_keys"]
    assert response.outputs["explicit"].payload == scenario["expected_output"]


def test_submit_api_surface_returns_parse_diagnostic_for_invalid_syntax() -> None:
    scenario = EXECUTOR_SCENARIOS["invalid_syntax_parse_diagnostic"]
    executor = DataStepExecutor()

    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": _memory_ref("in", scenario["inputs"]["in"])},
            output_targets=scenario["output_targets"],
        )
    )

    assert response.has_errors is True
    assert response.diagnostics[0].code == scenario["expected_code"]
    assert response.diagnostics[0].location == scenario["expected_location"]
    assert response.diagnostics[0].stage == "parse"


def test_submit_api_surface_tags_input_resolution_stage_in_diagnostics() -> None:
    executor = DataStepExecutor()

    response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set missing; output out; run;",
            output_targets=("out",),
        )
    )

    assert response.has_errors is True
    assert response.diagnostics[0].code == "RUNTIME_SET_DATASET_NOT_FOUND"
    assert response.diagnostics[0].stage == "resolve inputs"


def test_submit_api_surface_resolves_set_input_case_insensitive_work_prefix() -> None:
    scenario = EXECUTOR_SCENARIOS["set_input_case_insensitive_work_prefix"]
    executor = DataStepExecutor()
    executor.register_table(
        "my_table",
        _memory_ref("my_table", [{"id": 1, "amount": 10}]),
    )

    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
        )
    )

    assert response.has_errors is False
    assert response.outputs["out"].payload == scenario["expected_output"]


def test_submit_api_surface_supports_work_prefix_in_output_targets() -> None:
    scenario = EXECUTOR_SCENARIOS["work_prefix_in_output_targets"]
    executor = DataStepExecutor()

    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": _memory_ref("in", scenario["inputs"]["in"])},
        )
    )

    assert response.has_errors is False
    assert scenario["expected_key"] in response.outputs
    assert response.outputs["out"].payload == scenario["expected_output"]


def test_submit_api_surface_prefers_explicit_inputs_over_registered_tables() -> None:
    scenario = EXECUTOR_SCENARIOS["explicit_inputs_precedence"]
    executor = DataStepExecutor()
    executor.register_table(
        "in",
        _memory_ref("in-registered", [{"id": 1, "amount": 10}]),
    )

    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": _memory_ref("in-explicit", scenario["inputs"]["in"])},
            output_targets=scenario["output_targets"],
        )
    )

    assert response.has_errors is False
    assert response.outputs["out"].payload == scenario["expected_output"]


def test_session_submit_hides_result_representation_by_default() -> None:
    session = Session()
    session.load("inp", [{"id": 1}])

    result = session.submit("data out; set inp; output out; run;")

    assert result.success is True
    assert repr(result) == ""


def test_module_submit_keeps_result_representation_visible() -> None:
    result = limulus.submit("data out; set inp; output out; run;", inp=[{"id": 1}])

    assert result.success is True
    assert "SubmitResult(" in repr(result)
