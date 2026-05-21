"""Tests for PDV Runtime Service"""
import pytest

from limulus.runtime import PDVRuntimeService, FunctionRegistryService

PDV_RUNTIME_SCENARIOS = {
    "context_initialization": {
        "overview": "PDV context starts with automatic variables and no user-defined fields",
        "expected_automatic_variables": {"_N_": 0, "_ERROR_": 0},
        "expected_missing_user_variables": ("x", "name"),
    },
    "row_iteration_and_error_state": {
        "overview": "Row lifecycle increments _N_ and resets/updates _ERROR_ per runtime rules",
        "row_iterations": 2,
        "expected_n_values": (1, 2),
        "reset_error_after_manual_set": True,
    },
    "variable_and_expression_behavior": {
        "overview": "Variable persistence, IF-chain evaluation, WHERE filtering, and keep/drop transforms are validated",
        "if_chain": {
            "branches": [
                ("amount > 0", {"segment": "positive"}),
                ("amount == 0", {"segment": "zero"}),
            ],
            "else_assignments": {"segment": "negative"},
            "rows": (
                {"amount": 10, "expected": "positive"},
                {"amount": 0, "expected": "zero"},
                {"amount": -1, "expected": "negative"},
            ),
        },
        "where_cases": (
            {"row": {"amount": 10}, "expr": "amount >= 0", "expected": True},
            {"row": {"amount": -2}, "expr": "amount >= 0", "expected": False},
        ),
        "drop_keep_row": {"id": 1, "name": "alice", "amount": 100, "region": "tokyo"},
        "drop_only_expected": {"id": 1, "name": "alice", "amount": 100},
        "keep_only_expected": {"id": 1, "amount": 100},
        "keep_then_drop_expected": {"id": 1, "region": "tokyo"},
    },
    "output_and_diagnostics": {
        "overview": "Output routing and runtime diagnostic metadata are validated for failure and success paths",
        "output_targets": ("out_a", "out_b"),
        "routed_rows": (
            {"target": "out_a", "row": {"id": 1, "amount": 10}},
            {"target": "out_b", "row": {"id": 2, "amount": 20}},
        ),
        "unresolved_target": "missing_out",
        "expected_unresolved_code": "RUNTIME_OUTPUT_TARGET_NOT_FOUND",
        "expected_unresolved_location": "row:1",
        "invalid_reference_expr": "unknown_var > 0",
        "expected_invalid_reference_code": "RUNTIME_INVALID_REFERENCE",
        "uncomputable_expr": "amount > 1",
        "expected_uncomputable_code": "RUNTIME_EXPRESSION_EVALUATION_ERROR",
        "expected_row2_location": "row:2",
    },
}


class TestPDVRuntimeContextCreation:
    """Test creation and initialization of PDV runtime context"""

    def test_create_context_initializes_automatic_variables(self) -> None:
        """Runtime context should initialize _N_ to 0 and _ERROR_ to 0"""
        scenario = PDV_RUNTIME_SCENARIOS["context_initialization"]
        service = PDVRuntimeService()
        context = service.create_context()

        for key, expected in scenario["expected_automatic_variables"].items():
            assert context.get_variable(key) == expected

    def test_context_starts_with_empty_user_variables(self) -> None:
        """New context should have no user-defined variables"""
        scenario = PDV_RUNTIME_SCENARIOS["context_initialization"]
        service = PDVRuntimeService()
        context = service.create_context()

        for key in scenario["expected_automatic_variables"]:
            assert context.has_variable(key)

        for key in scenario["expected_missing_user_variables"]:
            assert not context.has_variable(key)


class TestPDVRuntimeRowIteration:
    """Test row-by-row iteration behavior"""

    def test_begin_row_increments_n_variable(self) -> None:
        """_N_ should increment on each row iteration"""
        scenario = PDV_RUNTIME_SCENARIOS["row_iteration_and_error_state"]
        service = PDVRuntimeService()
        context = service.create_context()

        assert context.get_variable("_N_") == 0

        for expected_n in scenario["expected_n_values"]:
            service.begin_row(context)
            assert context.get_variable("_N_") == expected_n

    def test_begin_row_resets_error_to_zero(self) -> None:
        """_ERROR_ should reset to 0 at the start of each row"""
        scenario = PDV_RUNTIME_SCENARIOS["row_iteration_and_error_state"]
        service = PDVRuntimeService()
        context = service.create_context()

        service.begin_row(context)
        context.set_variable("_ERROR_", 1)
        assert context.get_variable("_ERROR_") == 1

        if scenario["reset_error_after_manual_set"]:
            service.begin_row(context)
            assert context.get_variable("_ERROR_") == 0


class TestPDVRuntimeVariableManagement:
    """Test variable storage and retrieval"""

    def test_set_and_get_variable(self) -> None:
        """Should be able to set and retrieve user variables"""
        service = PDVRuntimeService()
        context = service.create_context()

        context.set_variable("x", 42)
        context.set_variable("name", "Alice")

        assert context.get_variable("x") == 42
        assert context.get_variable("name") == "Alice"

    def test_variable_persists_across_rows(self) -> None:
        """Variables should persist across row iterations (PDV behavior)"""
        service = PDVRuntimeService()
        context = service.create_context()

        service.begin_row(context)
        context.set_variable("x", 100)

        service.begin_row(context)
        # x should still exist from previous row
        assert context.get_variable("x") == 100

    def test_get_nonexistent_variable_returns_none(self) -> None:
        """Getting a variable that doesn't exist should return None"""
        service = PDVRuntimeService()
        context = service.create_context()

        assert context.get_variable("nonexistent") is None

    def test_variable_access_is_case_insensitive(self) -> None:
        service = PDVRuntimeService()
        context = service.create_context()

        context.set_variable("Name", "Alice")

        assert context.get_variable("name") == "Alice"
        assert context.get_variable("NAME") == "Alice"
        assert context.has_variable("name") is True


class TestPDVRuntimeErrorHandling:
    """Test error state management"""

    def test_set_error_updates_error_variable(self) -> None:
        """Setting error flag should update _ERROR_ to 1"""
        service = PDVRuntimeService()
        context = service.create_context()

        service.begin_row(context)
        assert context.get_variable("_ERROR_") == 0

        service.set_error(context, "Test error")
        assert context.get_variable("_ERROR_") == 1

    def test_error_persists_within_row(self) -> None:
        """Error flag should persist within the same row"""
        service = PDVRuntimeService()
        context = service.create_context()

        service.begin_row(context)
        service.set_error(context, "First error")
        assert context.get_variable("_ERROR_") == 1

        # Setting another error should keep _ERROR_ at 1
        service.set_error(context, "Second error")
        assert context.get_variable("_ERROR_") == 1


class TestPDVRuntimeLifecycle:
    """Test execution lifecycle from start to finish"""

    def test_full_lifecycle_with_multiple_rows(self) -> None:
        """Test complete execution cycle through multiple rows"""
        service = PDVRuntimeService()
        context = service.create_context()

        # Initial state
        assert context.get_variable("_N_") == 0
        assert context.get_variable("_ERROR_") == 0

        # Row 1
        service.begin_row(context)
        assert context.get_variable("_N_") == 1
        context.set_variable("sum", 10)

        # Row 2
        service.begin_row(context)
        assert context.get_variable("_N_") == 2
        # sum should persist
        assert context.get_variable("sum") == 10
        context.set_variable("sum", 25)

        # Row 3 with error
        service.begin_row(context)
        assert context.get_variable("_N_") == 3
        service.set_error(context, "Division by zero")
        assert context.get_variable("_ERROR_") == 1

        # Row 4 - error should reset
        service.begin_row(context)
        assert context.get_variable("_N_") == 4
        assert context.get_variable("_ERROR_") == 0


class TestPDVRuntimeStatementEvaluation:
    def test_if_else_if_else_assigns_expected_value(self) -> None:
        scenario = PDV_RUNTIME_SCENARIOS["variable_and_expression_behavior"]["if_chain"]
        service = PDVRuntimeService()
        context = service.create_context()

        for case in scenario["rows"]:
            result = service.evaluate_if_chain(
                row={"amount": case["amount"]},
                context=context,
                branches=scenario["branches"],
                else_assignments=scenario["else_assignments"],
            )
            assert result["segment"] == case["expected"]

    def test_where_filter_controls_output_eligibility(self) -> None:
        scenario = PDV_RUNTIME_SCENARIOS["variable_and_expression_behavior"]
        service = PDVRuntimeService()
        context = service.create_context()

        for case in scenario["where_cases"]:
            assert service.passes_where(case["row"], context, case["expr"]) is case["expected"]

    def test_where_filter_resolves_input_row_keys_case_insensitively(self) -> None:
        service = PDVRuntimeService()
        context = service.create_context()

        assert service.passes_where({"Amount": 10}, context, "amount >= 0") is True
        assert service.passes_where({"Amount": -1}, context, "AMOUNT >= 0") is False

    def test_drop_and_keep_control_output_variables(self) -> None:
        scenario = PDV_RUNTIME_SCENARIOS["variable_and_expression_behavior"]
        service = PDVRuntimeService()

        row = scenario["drop_keep_row"]

        dropped = service.apply_drop_keep(row, drop_vars=("region",), keep_vars=())
        assert dropped == scenario["drop_only_expected"]

        kept = service.apply_drop_keep(row, drop_vars=(), keep_vars=("id", "amount"))
        assert kept == scenario["keep_only_expected"]

        keep_then_drop = service.apply_drop_keep(
            row,
            drop_vars=("amount",),
            keep_vars=("id", "amount", "region"),
        )
        assert keep_then_drop == scenario["keep_then_drop_expected"]

    def test_apply_helper_resolves_callables_and_python_names(self) -> None:
        # direct callable should be invoked
        service = PDVRuntimeService()
        frs: FunctionRegistryService = service._function_registry

        # callable object passed directly
        assert frs._apply(lambda x: x + 1, 5) == 6

        # builtin name without registration
        assert frs._apply("len", "abc") == 3

        # dotted module path without registration
        assert frs._apply("math.sqrt", 16) == 4.0

    def test_shift_lag_lead_support_offsets(self) -> None:
        service = PDVRuntimeService()
        frs: FunctionRegistryService = service._function_registry
        rows = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]

        frs.set_row_view(row_index=3, rows=rows)
        assert frs._lag(10, 2) is None
        assert frs._lag(20, 2) is None
        assert frs._lag(30, 2) == 10

        frs.set_row_view(row_index=0, rows=rows)
        assert frs._lead("id", 2) == 3

        frs.set_row_view(row_index=2, rows=rows)
        assert frs._shift("id", 2) == 1
        assert frs._shift("id", -1) == 4
        assert frs._shift("id", 0) == 3


class TestPDVRuntimeOutputRouting:
    def test_output_statement_routes_rows_to_multiple_datasets(self) -> None:
        scenario = PDV_RUNTIME_SCENARIOS["output_and_diagnostics"]
        service = PDVRuntimeService()
        context = service.create_context()
        service.begin_row(context)

        routed = service.create_output_buffers(scenario["output_targets"])
        diagnostics: list = []

        for route in scenario["routed_rows"]:
            service.route_output_record(
                context=context,
                row=route["row"],
                target=route["target"],
                routed_outputs=routed,
                diagnostics=diagnostics,
            )

        assert diagnostics == []
        assert routed["out_a"] == [{"id": 1, "amount": 10}]
        assert routed["out_b"] == [{"id": 2, "amount": 20}]

    def test_unresolved_output_target_reports_runtime_diagnostic(self) -> None:
        scenario = PDV_RUNTIME_SCENARIOS["output_and_diagnostics"]
        service = PDVRuntimeService()
        context = service.create_context()
        service.begin_row(context)

        routed = service.create_output_buffers(("out_a",))
        diagnostics: list = []

        service.route_output_record(
            context=context,
            row={"id": 1},
            target=scenario["unresolved_target"],
            routed_outputs=routed,
            diagnostics=diagnostics,
        )

        assert routed["out_a"] == []
        assert len(diagnostics) == 1
        assert diagnostics[0].code == scenario["expected_unresolved_code"]
        assert diagnostics[0].location == scenario["expected_unresolved_location"]
        assert scenario["unresolved_target"] in diagnostics[0].message


class TestPDVRuntimeExecutionErrors:
    def test_invalid_reference_raises_identifiable_runtime_error(self) -> None:
        scenario = PDV_RUNTIME_SCENARIOS["output_and_diagnostics"]
        service = PDVRuntimeService()
        context = service.create_context()
        service.begin_row(context)

        with pytest.raises(service.RuntimeExecutionError) as exc:
            service.passes_where({"amount": 10}, context, scenario["invalid_reference_expr"])

        assert context.get_variable("_ERROR_") == 1
        assert exc.value.diagnostic.code == scenario["expected_invalid_reference_code"]
        assert exc.value.diagnostic.location == scenario["expected_unresolved_location"]
        assert exc.value.fatal is True

    def test_uncomputable_expression_raises_identifiable_runtime_error(self) -> None:
        scenario = PDV_RUNTIME_SCENARIOS["output_and_diagnostics"]
        service = PDVRuntimeService()
        context = service.create_context()
        service.begin_row(context)

        with pytest.raises(service.RuntimeExecutionError) as exc:
            service.passes_where({"amount": "abc"}, context, scenario["uncomputable_expr"])

        assert context.get_variable("_ERROR_") == 1
        assert exc.value.diagnostic.code == scenario["expected_uncomputable_code"]
        assert exc.value.diagnostic.location == scenario["expected_unresolved_location"]
        assert scenario["uncomputable_expr"] in exc.value.diagnostic.message

    def test_diagnostic_links_runtime_context_row_number(self) -> None:
        scenario = PDV_RUNTIME_SCENARIOS["output_and_diagnostics"]
        service = PDVRuntimeService()
        context = service.create_context()

        service.begin_row(context)
        service.begin_row(context)

        with pytest.raises(service.RuntimeExecutionError) as exc:
            service.evaluate_if_chain(
                row={"amount": 1},
                context=context,
                branches=[(scenario["invalid_reference_expr"], {"flag": "x"})],
                else_assignments={"flag": "y"},
            )

        assert exc.value.diagnostic.location == scenario["expected_row2_location"]
