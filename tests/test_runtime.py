import unittest
from unittest.mock import patch

import pyarrow as pa

from limulus.runtime import DataStepExecutor
from limulus.models import DataSetRef, ExecuteRequest

RUNTIME_SCENARIOS = {
    "backend_selection_and_fallback": {
        "overview": "Runtime chooses Rust when eligible and falls back to Python on subset/module limitations",
        "native_subset": {
            "dsl": "data out; set in; where amount >= 10; keep id amount; output out; run;",
            "inputs": [{"id": 1, "amount": 5, "tmp": "a"}, {"id": 2, "amount": 12, "tmp": "b"}],
            "expected_output": [{"id": 2, "amount": 12}],
            "expected_backend": "rust",
        },
        "fallback_subset_not_supported": {
            "dsl": "data out; set in; if amount > 0 then output out; run;",
            "inputs": [{"amount": 1}],
            "expected_output": [{"amount": 1}],
            "expected_backend": "python",
        },
        "fallback_native_missing": {
            "dsl": "data out; set in; where amount >= 10; output out; run;",
            "inputs": {"id": [1, 2], "amount": [5, 12]},
            "expected_output": [{"id": 2, "amount": 12}],
            "expected_backend": "python",
        },
    },
    "where_expression_coverage": {
        "overview": "Native runtime evaluates arithmetic and logical WHERE expressions",
        "arithmetic_logical": {
            "dsl": (
                "data out; set in; "
                "where (amount * qty) >= 20 and not (group = 'B'); "
                "output out; run;"
            ),
            "inputs": [
                {"id": 1, "amount": 10, "qty": 2, "group": "A"},
                {"id": 2, "amount": 10, "qty": 1, "group": "A"},
                {"id": 3, "amount": 10, "qty": 3, "group": "B"},
            ],
            "expected_output": [{"id": 1, "amount": 10, "qty": 2, "group": "A"}],
        },
    },
    "control_flow_and_output_routing": {
        "overview": "IF/ELSE, DO/END, sum statement, and multi-output routing preserve expected row semantics",
        "if_else_multi_output": {
            "dsl": "data out_hi out_lo; set in; if amount >= 10 then output out_hi; else output out_lo; run;",
            "inputs": [{"id": 1, "amount": 5}, {"id": 2, "amount": 12}],
            "expected_hi": [{"id": 2, "amount": 12}],
            "expected_lo": [{"id": 1, "amount": 5}],
        },
        "do_assign_sum": {
            "dsl": (
                "data out; set in; "
                "total + amount; do i = 1 to 2; calc = amount * i; output out; end; run;"
            ),
            "inputs": [{"id": 1, "amount": 2}, {"id": 2, "amount": 3}],
            "expected_output": [
                {"id": 1, "amount": 2, "total": 2.0, "i": 1, "calc": 2.0},
                {"id": 1, "amount": 2, "total": 2.0, "i": 2, "calc": 4.0},
                {"id": 2, "amount": 3, "total": 5.0, "i": 1, "calc": 3.0},
                {"id": 2, "amount": 3, "total": 5.0, "i": 2, "calc": 6.0},
            ],
        },
        "if_multi_output_power": {
            "dsl": (
                "data older younger; set in; "
                "height_m = height * 0.0254; weight_kg = weight * 0.454; "
                "bmi = round(weight_kg / (height_m**2), 0.1); "
                "if age > 13 then output older; else output younger; run;"
            ),
            "inputs": [
                {"id": 1, "age": 10, "height": 150.0, "weight": 50.0},
                {"id": 2, "age": 15, "height": 160.0, "weight": 60.0},
            ],
            "expected_older_id": 2,
            "expected_younger_id": 1,
        },
    },
    "dataset_option_compatibility": {
        "overview": "SET/DATA dataset options and merge behaviors remain consistent across runtime preferences",
        "pdv_auto_set_options": {
            "dsl": (
                "data out; set in(in=src) indsname=dsn end=eof; by grp; "
                "if first.grp then flag = 'first'; if last.grp then output out; "
                "keep grp val flag; run;"
            ),
            "inputs": [{"grp": "A", "val": 10}, {"grp": "A", "val": 11}, {"grp": "B", "val": 12}],
            "expected_output": [
                {"grp": "A", "val": 11, "flag": None},
                {"grp": "B", "val": 12, "flag": "first"},
            ],
        },
        "set_options_single_backend": {
            "dsl": "data out; set in(keep=id amount drop=amount rename=(id=key) where=(key > 1)); output out; run;",
            "inputs": [{"id": 1, "amount": 10, "tmp": "a"}, {"id": 2, "amount": 20, "tmp": "b"}],
            "expected_output": [{"key": 2}],
            "expected_backend": "python",
        },
        "set_options_cross_backend": {
            "dsl": "data out; set in(keep=id amount drop=amount rename=(id=key) where=(key > 1)); output out; run;",
            "inputs": [{"id": 1, "amount": 10, "tmp": "a"}, {"id": 2, "amount": 20, "tmp": "b"}],
            "expected_python": [{"key": 2}],
        },
        "output_options_cross_backend": {
            "dsl": "data out(keep=id name drop=name rename=(id=subject_id)); set in; output out; run;",
            "inputs": [{"id": 1, "name": "Alice", "tmp": "a"}, {"id": 2, "name": "Bob", "tmp": "b"}],
            "expected_python": [{"subject_id": 1}, {"subject_id": 2}],
        },
        "merge_options_cross_backend": {
            "dsl": "data out; merge a(keep=id xa rename=(xa=x)) b(keep=id xb rename=(xb=y)); by id; output out; run;",
            "inputs_a": [{"id": 1, "xa": 10, "tmp": "a"}, {"id": 2, "xa": 20, "tmp": "b"}],
            "inputs_b": [{"id": 1, "xb": 100}, {"id": 3, "xb": 300}],
            "expected_python": [{"id": 1, "x": 10, "y": 100}, {"id": 2, "x": 20}, {"id": 3, "y": 300}],
        },
        "where_statement_multi_set_cross_backend": {
            "dsl": "data out; set a b; where id >= 2; output out; run;",
            "inputs_a": [{"id": 1, "x": 10}, {"id": 2, "x": 20}],
            "inputs_b": [{"id": 2, "x": 200}, {"id": 3, "x": 300}],
            "expected_python": [{"id": 2, "x": 20}, {"id": 2, "x": 200}, {"id": 3, "x": 300}],
        },
        "where_statement_merge_cross_backend": {
            "dsl": "data out; merge a b; by id; where id >= 2; output out; run;",
            "inputs_a": [{"id": 1, "x": 10}, {"id": 2, "x": 20}],
            "inputs_b": [{"id": 2, "y": 200}, {"id": 3, "y": 300}],
            "expected_python": [{"id": 2, "x": 20, "y": 200}, {"id": 3, "y": 300}],
        },
        "where_function_multi_set_cross_backend": {
            "dsl": "data out; set a b; where upcase(group) = 'A'; output out; run;",
            "inputs_a": [{"id": 1, "group": "a"}, {"id": 2, "group": "b"}],
            "inputs_b": [{"id": 3, "group": "A"}, {"id": 4, "group": "c"}],
            "expected_python": [{"id": 1, "group": "a"}, {"id": 3, "group": "A"}],
        },
    },
    "stateful_runtime_features": {
        "overview": "Retain state and array operations are supported in native execution paths",
        "retain_state": {
            "dsl": (
                "data out; set in; retain carry; if id = 1 then carry = amount; "
                "output out; keep id carry; run;"
            ),
            "inputs": [{"id": 1, "amount": 10}, {"id": 2, "amount": 99}],
            "expected_output": [{"id": 1, "carry": 10.0}, {"id": 2, "carry": 10.0}],
        },
        "array_assignment_dim_vname": {
            "dsl": (
                "data out; set in; array vals a b c; vals(1) = vals(1) * 10; "
                "vals[2] = vals[2] * 10; vals{3} = vals{3} * 10; "
                "arr_size = dim(vals); name2 = vname(vals[2]); output out; keep a b c arr_size name2; run;"
            ),
            "inputs": [{"a": 1, "b": 2, "c": 3}],
            "expected_output": [{"a": 10.0, "b": 20.0, "c": 30.0, "arr_size": 3.0, "name2": "b"}],
        },
    },
}


def _output_rows(response, target: str):
    dataset = response.outputs[target]
    if dataset.kind == "arrow_table" and hasattr(dataset.payload, "to_pylist"):
        return dataset.payload.to_pylist()
    return dataset.payload


def _arrow_table(rows):
    return pa.Table.from_pylist([dict(row) for row in rows])


class RuntimeNativeIntegrationTests(unittest.TestCase):
    def test_python_runtime_uses_direct_arrow_loop_for_simple_single_set(self) -> None:
        executor = DataStepExecutor(runtime_backend="python")
        request = ExecuteRequest(
            dsl_text="data out; set in; if amount > 0 then output out; run;",
            inputs={
                "in": DataSetRef(
                    kind="arrow_table",
                    location="dataset://in",
                    payload=_arrow_table([
                        {"id": 1, "amount": 10},
                        {"id": 2, "amount": -1},
                    ]),
                )
            },
        )

        with patch("limulus.io_adapters.DataInputAdapterArrow.load", side_effect=AssertionError("full materialization path should not be used")):
            response = executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(_output_rows(response, "out"), [{"id": 1, "amount": 10}])
        self.assertEqual(executor.last_runtime_backend, "python")

    def test_native_runtime_executes_subset_via_rust_backend(self) -> None:
        scenario = RUNTIME_SCENARIOS["backend_selection_and_fallback"]["native_subset"]
        executor = DataStepExecutor(runtime_backend="rust")
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="arrow_table",
                    location="dataset://in",
                    payload=_arrow_table(scenario["inputs"]),
                )
            },
        )

        response = executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(_output_rows(response, "out"), scenario["expected_output"])
        self.assertEqual(executor.last_runtime_backend, scenario["expected_backend"])

    def test_runtime_falls_back_to_python_when_rust_subset_not_supported(self) -> None:
        scenario = RUNTIME_SCENARIOS["backend_selection_and_fallback"]["fallback_subset_not_supported"]
        executor = DataStepExecutor(runtime_backend="rust")
        response = executor.execute(
            ExecuteRequest(
                dsl_text=scenario["dsl"],
                inputs={
                    "in": DataSetRef(
                        kind="memory",
                        location="dataset://in",
                        payload=scenario["inputs"],
                    )
                },
            )
        )

        self.assertFalse(response.has_errors)
        self.assertEqual(_output_rows(response, "out"), scenario["expected_output"])
        self.assertEqual(executor.last_runtime_backend, scenario["expected_backend"])

    def test_runtime_falls_back_to_python_when_native_module_unavailable(self) -> None:
        scenario = RUNTIME_SCENARIOS["backend_selection_and_fallback"]["fallback_native_missing"]
        executor = DataStepExecutor(runtime_backend="rust")
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="arrow_table",
                    location="dataset://in",
                    payload=pa.table(scenario["inputs"]),
                )
            },
        )

        with patch("limulus.backends.load_native_module", return_value=(None, RuntimeError("missing"))):
            response = executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(_output_rows(response, "out"), scenario["expected_output"])
        self.assertEqual(executor.last_runtime_backend, scenario["expected_backend"])

    def test_native_runtime_evaluates_arithmetic_and_logical_where_expression(self) -> None:
        scenario = RUNTIME_SCENARIOS["where_expression_coverage"]["arithmetic_logical"]
        executor = DataStepExecutor(runtime_backend="rust")
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="arrow_table",
                    location="dataset://in",
                    payload=_arrow_table(scenario["inputs"]),
                )
            },
        )

        response = executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(_output_rows(response, "out"), scenario["expected_output"])
        self.assertEqual(executor.last_runtime_backend, "rust")

    def test_native_runtime_supports_if_else_and_output_routing(self) -> None:
        scenario = RUNTIME_SCENARIOS["control_flow_and_output_routing"]["if_else_multi_output"]
        executor = DataStepExecutor(runtime_backend="rust")
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="arrow_table",
                    location="dataset://in",
                    payload=_arrow_table(scenario["inputs"]),
                )
            },
        )

        response = executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(_output_rows(response, "out_hi"), scenario["expected_hi"])
        self.assertEqual(_output_rows(response, "out_lo"), scenario["expected_lo"])
        self.assertEqual(executor.last_runtime_backend, "rust")

    def test_native_runtime_supports_do_assign_and_sum_statement(self) -> None:
        scenario = RUNTIME_SCENARIOS["control_flow_and_output_routing"]["do_assign_sum"]
        executor = DataStepExecutor(runtime_backend="rust")
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="arrow_table",
                    location="dataset://in",
                    payload=_arrow_table(scenario["inputs"]),
                )
            },
        )

        response = executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(_output_rows(response, "out"), scenario["expected_output"])
        self.assertEqual(executor.last_runtime_backend, "rust")

    def test_native_runtime_supports_pdv_automatic_and_set_option_variables(self) -> None:
        scenario = RUNTIME_SCENARIOS["dataset_option_compatibility"]["pdv_auto_set_options"]
        executor = DataStepExecutor(runtime_backend="rust")
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="arrow_table",
                    location="dataset://in",
                    payload=_arrow_table(scenario["inputs"]),
                )
            },
        )

        response = executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(_output_rows(response, "out"), scenario["expected_output"])
        self.assertEqual(executor.last_runtime_backend, "rust")

    def test_runtime_applies_set_dataset_options(self) -> None:
        scenario = RUNTIME_SCENARIOS["dataset_option_compatibility"]["set_options_single_backend"]
        executor = DataStepExecutor(runtime_backend=scenario["expected_backend"])
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="arrow_table",
                    location="dataset://in",
                    payload=_arrow_table(scenario["inputs"]),
                )
            },
        )

        response = executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(_output_rows(response, "out"), scenario["expected_output"])
        self.assertEqual(executor.last_runtime_backend, scenario["expected_backend"])

    def test_native_runtime_supports_if_multi_output_with_power_expression(self) -> None:
        scenario = RUNTIME_SCENARIOS["control_flow_and_output_routing"]["if_multi_output_power"]
        executor = DataStepExecutor(runtime_backend="rust")
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="arrow_table",
                    location="dataset://in",
                    payload=_arrow_table(scenario["inputs"]),
                )
            },
        )

        response = executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(executor.last_runtime_backend, "rust")
        self.assertEqual(_output_rows(response, "older")[0]["id"], scenario["expected_older_id"])
        self.assertEqual(_output_rows(response, "younger")[0]["id"], scenario["expected_younger_id"])

    def test_runtime_applies_set_dataset_options_for_python_and_rust_preferences(self) -> None:
        scenario = RUNTIME_SCENARIOS["dataset_option_compatibility"]["set_options_cross_backend"]
        dsl_text = scenario["dsl"]
        inputs = {
            "in": DataSetRef(
                kind="arrow_table",
                location="dataset://in",
                payload=_arrow_table(scenario["inputs"]),
            )
        }

        outputs_by_backend: dict[str, list[dict[str, object]]] = {}
        for runtime_backend in ("python", "rust"):
            executor = DataStepExecutor(runtime_backend=runtime_backend)
            response = executor.execute(
                ExecuteRequest(
                    dsl_text=dsl_text,
                    inputs=inputs,
                    output_targets=["out"],
                )
            )

            self.assertFalse(response.has_errors)
            outputs_by_backend[runtime_backend] = _output_rows(response, "out")

        self.assertEqual(outputs_by_backend["python"], scenario["expected_python"])
        self.assertEqual(outputs_by_backend["rust"], outputs_by_backend["python"])

    def test_runtime_applies_output_dataset_options_for_python_and_rust_preferences(self) -> None:
        scenario = RUNTIME_SCENARIOS["dataset_option_compatibility"]["output_options_cross_backend"]
        dsl_text = scenario["dsl"]
        inputs = {
            "in": DataSetRef(
                kind="arrow_table",
                location="dataset://in",
                payload=_arrow_table(scenario["inputs"]),
            )
        }

        outputs_by_backend: dict[str, list[dict[str, object]]] = {}
        for runtime_backend in ("python", "rust"):
            executor = DataStepExecutor(runtime_backend=runtime_backend)
            response = executor.execute(
                ExecuteRequest(
                    dsl_text=dsl_text,
                    inputs=inputs,
                    output_targets=["out"],
                )
            )

            self.assertFalse(response.has_errors)
            outputs_by_backend[runtime_backend] = _output_rows(response, "out")

        self.assertEqual(outputs_by_backend["python"], scenario["expected_python"])
        self.assertEqual(outputs_by_backend["rust"], outputs_by_backend["python"])

    def test_merge_dataset_options_are_consistent_between_python_and_rust_preferences(self) -> None:
        scenario = RUNTIME_SCENARIOS["dataset_option_compatibility"]["merge_options_cross_backend"]
        dsl_text = scenario["dsl"]
        inputs = {
            "a": DataSetRef(
                kind="arrow_table",
                location="dataset://a",
                payload=_arrow_table(scenario["inputs_a"]),
            ),
            "b": DataSetRef(
                kind="arrow_table",
                location="dataset://b",
                payload=_arrow_table(scenario["inputs_b"]),
            ),
        }

        outputs_by_backend: dict[str, list[dict[str, object]]] = {}
        last_backend_by_preference: dict[str, str] = {}
        for runtime_backend in ("python", "rust"):
            executor = DataStepExecutor(runtime_backend=runtime_backend)
            response = executor.execute(
                ExecuteRequest(
                    dsl_text=dsl_text,
                    inputs=inputs,
                    output_targets=["out"],
                )
            )
            self.assertFalse(response.has_errors)
            outputs_by_backend[runtime_backend] = _output_rows(response, "out")
            last_backend_by_preference[runtime_backend] = executor.last_runtime_backend

        self.assertEqual(outputs_by_backend["python"], scenario["expected_python"])
        self.assertEqual(outputs_by_backend["rust"], outputs_by_backend["python"])
        self.assertEqual(last_backend_by_preference["rust"], "rust")

    def test_statement_where_on_multi_set_is_consistent_between_python_and_rust_preferences(self) -> None:
        scenario = RUNTIME_SCENARIOS["dataset_option_compatibility"]["where_statement_multi_set_cross_backend"]
        dsl_text = scenario["dsl"]
        inputs = {
            "a": DataSetRef(
                kind="arrow_table",
                location="dataset://a",
                payload=_arrow_table(scenario["inputs_a"]),
            ),
            "b": DataSetRef(
                kind="arrow_table",
                location="dataset://b",
                payload=_arrow_table(scenario["inputs_b"]),
            ),
        }

        outputs_by_backend: dict[str, list[dict[str, object]]] = {}
        for runtime_backend in ("python", "rust"):
            executor = DataStepExecutor(runtime_backend=runtime_backend)
            response = executor.execute(
                ExecuteRequest(
                    dsl_text=dsl_text,
                    inputs=inputs,
                    output_targets=["out"],
                )
            )

            self.assertFalse(response.has_errors)
            outputs_by_backend[runtime_backend] = _output_rows(response, "out")

        self.assertEqual(outputs_by_backend["python"], scenario["expected_python"])
        self.assertEqual(outputs_by_backend["rust"], outputs_by_backend["python"])

    def test_statement_where_on_merge_is_consistent_between_python_and_rust_preferences(self) -> None:
        scenario = RUNTIME_SCENARIOS["dataset_option_compatibility"]["where_statement_merge_cross_backend"]
        dsl_text = scenario["dsl"]
        inputs = {
            "a": DataSetRef(
                kind="arrow_table",
                location="dataset://a",
                payload=_arrow_table(scenario["inputs_a"]),
            ),
            "b": DataSetRef(
                kind="arrow_table",
                location="dataset://b",
                payload=_arrow_table(scenario["inputs_b"]),
            ),
        }

        outputs_by_backend: dict[str, list[dict[str, object]]] = {}
        for runtime_backend in ("python", "rust"):
            executor = DataStepExecutor(runtime_backend=runtime_backend)
            response = executor.execute(
                ExecuteRequest(
                    dsl_text=dsl_text,
                    inputs=inputs,
                    output_targets=["out"],
                )
            )
            self.assertFalse(response.has_errors)
            outputs_by_backend[runtime_backend] = _output_rows(response, "out")

        self.assertEqual(outputs_by_backend["python"], scenario["expected_python"])
        self.assertEqual(outputs_by_backend["rust"], outputs_by_backend["python"])

    def test_statement_where_with_function_on_multi_set_is_consistent_and_keeps_rust_backend(self) -> None:
        scenario = RUNTIME_SCENARIOS["dataset_option_compatibility"]["where_function_multi_set_cross_backend"]
        dsl_text = scenario["dsl"]
        inputs = {
            "a": DataSetRef(
                kind="arrow_table",
                location="dataset://a",
                payload=_arrow_table(scenario["inputs_a"]),
            ),
            "b": DataSetRef(
                kind="arrow_table",
                location="dataset://b",
                payload=_arrow_table(scenario["inputs_b"]),
            ),
        }

        outputs_by_backend: dict[str, list[dict[str, object]]] = {}
        last_backend_by_preference: dict[str, str] = {}
        for runtime_backend in ("python", "rust"):
            executor = DataStepExecutor(runtime_backend=runtime_backend)
            response = executor.execute(
                ExecuteRequest(
                    dsl_text=dsl_text,
                    inputs=inputs,
                    output_targets=["out"],
                )
            )

            self.assertFalse(response.has_errors)
            outputs_by_backend[runtime_backend] = _output_rows(response, "out")
            last_backend_by_preference[runtime_backend] = executor.last_runtime_backend

        self.assertEqual(outputs_by_backend["python"], scenario["expected_python"])
        self.assertEqual(outputs_by_backend["rust"], outputs_by_backend["python"])
        self.assertEqual(last_backend_by_preference["rust"], "rust")

    def test_native_runtime_supports_retain_state_across_rows(self) -> None:
        scenario = RUNTIME_SCENARIOS["stateful_runtime_features"]["retain_state"]
        executor = DataStepExecutor(runtime_backend="rust")
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="arrow_table",
                    location="dataset://in",
                    payload=_arrow_table(scenario["inputs"]),
                )
            },
        )

        response = executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(_output_rows(response, "out"), scenario["expected_output"])
        self.assertEqual(executor.last_runtime_backend, "rust")

    def test_native_runtime_supports_array_assignment_and_dim_vname(self) -> None:
        scenario = RUNTIME_SCENARIOS["stateful_runtime_features"]["array_assignment_dim_vname"]
        executor = DataStepExecutor(runtime_backend="rust")
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="arrow_table",
                    location="dataset://in",
                    payload=_arrow_table(scenario["inputs"]),
                )
            },
        )

        response = executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(_output_rows(response, "out"), scenario["expected_output"])
        self.assertEqual(executor.last_runtime_backend, "rust")


if __name__ == "__main__":
    unittest.main()
