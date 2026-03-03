import unittest

from limulus.runtime import (
    LoopControlService,
    RetainArrayRuntimeService,
    RuntimeExecutionError,
)

ADVANCED_RUNTIME_SCENARIOS = {
    "retain_and_array": {
        "overview": "Retain state carries values across rows and array indexing enforces bounds",
        "retain_vars": ("flag",),
        "row1": {"id": 1, "flag": 1},
        "row2": {"id": 2},
        "expected_row2_flag": 1,
        "array_name": "vals",
        "array_members": ("a", "b"),
        "array_row": {"a": 10, "b": 20},
        "expected_first_value": 10,
        "expected_second_value": 20,
        "invalid_index": 3,
        "expected_error_code": "RUNTIME_ARRAY_INVALID",
    },
    "loop_control": {
        "overview": "DO TO / DO WHILE / DO UNTIL semantics and loop safety limits are enforced",
        "do_to_range": (1, 3),
        "expected_do_to_seen": [1, 2, 3],
        "while_limit": 3,
        "until_limit": 2,
        "max_iterations": 5,
        "expected_error_code": "RUNTIME_LOOP_LIMIT_EXCEEDED",
    },
}


class RetainArrayRuntimeServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = RetainArrayRuntimeService()

    def test_retain_state_keeps_previous_value_when_row_missing_variable(self) -> None:
        scenario = ADVANCED_RUNTIME_SCENARIOS["retain_and_array"]
        retain_state = self.service.create_retain_state(scenario["retain_vars"])

        row1 = self.service.apply_retain_values(scenario["row1"], retain_state)
        self.service.update_retain_state(row1, retain_state)

        row2 = self.service.apply_retain_values(scenario["row2"], retain_state)

        self.assertEqual(row2["flag"], scenario["expected_row2_flag"])

    def test_array_element_access_and_bounds_validation(self) -> None:
        scenario = ADVANCED_RUNTIME_SCENARIOS["retain_and_array"]
        arrays = self.service.define_array(scenario["array_name"], scenario["array_members"])
        row = scenario["array_row"]

        self.assertEqual(
            self.service.get_array_value(arrays, row, scenario["array_name"], 1),
            scenario["expected_first_value"],
        )
        self.assertEqual(
            self.service.get_array_value(arrays, row, scenario["array_name"], 2),
            scenario["expected_second_value"],
        )

        with self.assertRaises(RuntimeExecutionError) as error:
            self.service.get_array_value(arrays, row, scenario["array_name"], scenario["invalid_index"])

        self.assertEqual(error.exception.diagnostic.code, scenario["expected_error_code"])


class LoopControlServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = LoopControlService()

    def test_do_to_runs_with_index_updates(self) -> None:
        scenario = ADVANCED_RUNTIME_SCENARIOS["loop_control"]
        seen: list[int] = []

        self.service.run_do_to(*scenario["do_to_range"], body=lambda i: seen.append(i))

        self.assertEqual(seen, scenario["expected_do_to_seen"])

    def test_do_while_and_do_until_evaluate_conditions(self) -> None:
        scenario = ADVANCED_RUNTIME_SCENARIOS["loop_control"]
        while_counter = {"value": 0}

        def while_condition() -> bool:
            return while_counter["value"] < scenario["while_limit"]

        def while_body() -> None:
            while_counter["value"] += 1

        self.service.run_do_while(while_condition, while_body)
        self.assertEqual(while_counter["value"], scenario["while_limit"])

        until_counter = {"value": 0}

        def until_condition() -> bool:
            return until_counter["value"] >= scenario["until_limit"]

        def until_body() -> None:
            until_counter["value"] += 1

        self.service.run_do_until(until_condition, until_body)
        self.assertEqual(until_counter["value"], scenario["until_limit"])

    def test_loop_safety_limit_prevents_infinite_loop(self) -> None:
        scenario = ADVANCED_RUNTIME_SCENARIOS["loop_control"]
        with self.assertRaises(RuntimeExecutionError) as error:
            self.service.run_do_while(lambda: True, lambda: None, max_iterations=scenario["max_iterations"])

        self.assertEqual(error.exception.diagnostic.code, scenario["expected_error_code"])


if __name__ == "__main__":
    unittest.main()
