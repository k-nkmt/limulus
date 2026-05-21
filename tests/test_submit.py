import limulus
from unittest.mock import patch

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
    "reserved_dictionary_output_target": {
        "overview": "Rejects DICTIONARY.* output targets during validate stage",
        "dsl": "data dictionary.tables; set in; run;",
        "inputs": {"in": [{"id": 1}]},
        "expected_code": "VALIDATE_RESERVED_OUTPUT_TARGET",
    },
    "validate_keep_unknown_column": {
        "overview": "KEEP with an unknown column is rejected during validate stage",
        "dsl": "data out; set in; keep id missing; run;",
        "inputs": {"in": [{"id": 1, "amount": 10}]},
        "expected_code": "VALIDATE_COLUMN_NOT_FOUND",
    },
    "validate_keep_step_defined_column": {
        "overview": "KEEP accepts variables introduced within the step",
        "dsl": "data out; set in; seq + 1; keep id seq; run;",
        "inputs": {"in": [{"id": 1}]},
        "expected_output": [{"id": 1, "seq": 1.0}],
    },
    "validate_drop_unknown_column": {
        "overview": "DROP with an unknown column is rejected during validate stage",
        "dsl": "data out; set in; drop missing; run;",
        "inputs": {"in": [{"id": 1, "amount": 10}]},
        "expected_code": "VALIDATE_COLUMN_NOT_FOUND",
    },
    "validate_rename_unknown_column": {
        "overview": "RENAME with an unknown source column is rejected during validate stage",
        "dsl": "data out; set in; rename missing=amt; run;",
        "inputs": {"in": [{"id": 1, "amount": 10}]},
        "expected_code": "RUNTIME_RENAME_STATEMENT_INVALID",
    },
    "validate_label_unknown_column": {
        "overview": "LABEL with an unknown column is rejected during validate stage",
        "dsl": 'data out; set in; label missing = "Missing"; run;',
        "inputs": {"in": [{"id": 1, "amount": 10}]},
        "expected_code": "VALIDATE_COLUMN_NOT_FOUND",
    },
    "validate_by_unknown_column": {
        "overview": "BY with a missing source column is rejected during validate stage",
        "dsl": "data out; merge a b; by missing; run;",
        "inputs": {
            "a": [{"id": 1, "x": 10}],
            "b": [{"id": 1, "y": 20}],
        },
        "expected_code": "RUNTIME_BY_PRECONDITION_FAILED",
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


def test_submit_api_surface_parse_error_short_circuits_validate() -> None:
    executor = DataStepExecutor()

    response = executor.execute(
        ExecuteRequest(
            dsl_text="data dictionary.tables; set in; ???; run;",
            inputs={"in": _memory_ref("in", [{"id": 1}])},
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
            inputs={"in": _memory_ref("in", [{"id": 1, "amount": 5}, {"id": 2, "amount": 12}])},
        )
    )

    assert response.has_errors is False
    assert response.outputs["out"].payload == [{"id": 2, "amount": 12}]


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
            inputs={"in": _memory_ref("in", [{"id": 1, "amount": 5}])},
        )
    )

    assert response.has_errors is False
    assert response.outputs["out"].payload == [{"id": 1, "amount": 5}]


def test_submit_api_surface_tags_validate_stage_for_missing_set_input() -> None:
    executor = DataStepExecutor()

    response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set missing; output out; run;",
            output_targets=("out",),
        )
    )

    assert response.has_errors is True
    assert response.diagnostics[0].code == "RUNTIME_SET_DATASET_NOT_FOUND"
    assert response.diagnostics[0].stage == "validate"


def test_submit_api_surface_rejects_reserved_dictionary_output_target_in_validate_stage() -> None:
    scenario = EXECUTOR_SCENARIOS["reserved_dictionary_output_target"]
    executor = DataStepExecutor()

    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": _memory_ref("in", scenario["inputs"]["in"])},
        )
    )

    assert response.has_errors is True
    assert response.diagnostics[0].code == scenario["expected_code"]
    assert response.diagnostics[0].stage == "validate"


def test_submit_api_surface_rejects_keep_unknown_column_in_validate_stage() -> None:
    scenario = EXECUTOR_SCENARIOS["validate_keep_unknown_column"]
    executor = DataStepExecutor()

    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": _memory_ref("in", scenario["inputs"]["in"])},
        )
    )

    assert response.has_errors is True
    assert response.diagnostics[0].code == scenario["expected_code"]
    assert response.diagnostics[0].stage == "validate"


def test_submit_api_surface_allows_keep_for_step_defined_column() -> None:
    scenario = EXECUTOR_SCENARIOS["validate_keep_step_defined_column"]
    executor = DataStepExecutor()

    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": _memory_ref("in", scenario["inputs"]["in"])},
        )
    )

    assert response.has_errors is False
    assert response.outputs["out"].payload == scenario["expected_output"]


def test_submit_api_surface_rejects_drop_unknown_column_in_validate_stage() -> None:
    scenario = EXECUTOR_SCENARIOS["validate_drop_unknown_column"]
    executor = DataStepExecutor()

    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": _memory_ref("in", scenario["inputs"]["in"])},
        )
    )

    assert response.has_errors is True
    assert response.diagnostics[0].code == scenario["expected_code"]
    assert response.diagnostics[0].stage == "validate"


def test_submit_api_surface_rejects_rename_unknown_column_in_validate_stage() -> None:
    scenario = EXECUTOR_SCENARIOS["validate_rename_unknown_column"]
    executor = DataStepExecutor()

    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": _memory_ref("in", scenario["inputs"]["in"])},
        )
    )

    assert response.has_errors is True
    assert response.diagnostics[0].code == scenario["expected_code"]
    assert response.diagnostics[0].stage == "validate"


def test_submit_api_surface_rejects_label_unknown_column_in_validate_stage() -> None:
    scenario = EXECUTOR_SCENARIOS["validate_label_unknown_column"]
    executor = DataStepExecutor()

    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": _memory_ref("in", scenario["inputs"]["in"])},
        )
    )

    assert response.has_errors is True
    assert response.diagnostics[0].code == scenario["expected_code"]
    assert response.diagnostics[0].stage == "validate"


def test_submit_api_surface_rejects_by_unknown_column_in_validate_stage() -> None:
    scenario = EXECUTOR_SCENARIOS["validate_by_unknown_column"]
    executor = DataStepExecutor()

    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "a": _memory_ref("a", scenario["inputs"]["a"]),
                "b": _memory_ref("b", scenario["inputs"]["b"]),
            },
        )
    )

    assert response.has_errors is True
    assert response.diagnostics[0].code == scenario["expected_code"]
    assert response.diagnostics[0].stage == "validate"


def test_submit_api_surface_parse_and_validate_share_structured_diagnostic_fields() -> None:
    executor = DataStepExecutor()

    parse_response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; ???; run;",
            inputs={"in": _memory_ref("in", [{"id": 1}])},
            output_targets=("out",),
        )
    )
    validate_response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set in; keep id missing; run;",
            inputs={"in": _memory_ref("in", [{"id": 1}])},
        )
    )

    parse_diagnostic = parse_response.diagnostics[0]
    validate_diagnostic = validate_response.diagnostics[0]

    assert parse_response.has_errors is True
    assert validate_response.has_errors is True
    assert parse_diagnostic.span is not None
    assert validate_diagnostic.span is not None
    assert parse_diagnostic.labels
    assert validate_diagnostic.labels
    assert parse_diagnostic.source_text == "data out; ???; run;"
    assert validate_diagnostic.source_text == "data out; set in; keep id missing; run;"


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
