import unittest
from unittest.mock import patch

import pyarrow as pa

from limulus.backends import RuntimeExecutionContext, RustArrowIOBridge, RustExecutionPayload, RustNativeBlockExecutor
from limulus.models import DataSetRef, ExecuteRequest
from limulus.parser import ParsedStatement

IO_BRIDGE_SCENARIOS = {
    "rust_arrow_bridge_input": {
        "overview": "Rust bridge prefers Arrow C stream input path and avoids eager to_pylist conversion",
        "dsl": "data out; set in; run;",
        "input_kind": "arrow_table",
        "input_location": "dataset://in",
        "stream_tag": "stream://input",
        "resolved_output_targets": ("out",),
        "expected_diagnostics": [],
        "expected_stream_calls": 1,
        "expected_to_pylist_calls": 0,
    },
    "rust_arrow_bridge_rejection": {
        "overview": "Rust bridge rejects non-arrow input payloads with identifiable diagnostics",
        "dsl": "data out; set in; run;",
        "input_kind": "memory",
        "input_location": "dataset://in",
        "input_payload": [{"id": 1}],
        "resolved_output_targets": ("out",),
        "expected_diagnostic_code": "RUNTIME_RUST_BRIDGE_INPUT_NOT_ARROW",
    },
    "native_block_executor_io": {
        "overview": "Native block executor serializes input streams and deserializes output streams as arrow tables",
        "stream_tag": "stream://input",
        "output_table_rows": [{"id": 1, "amount": 2}],
        "output_target": "out",
        "expected_output_kind": "arrow_table",
        "expected_output_rows": [{"id": 1, "amount": 2}],
    },
}


class _FakeArrowTable:
    def __init__(self, stream_tag: str) -> None:
        self.stream_tag = stream_tag
        self.stream_calls = 0
        self.to_pylist_calls = 0

    def __arrow_c_stream__(self):
        self.stream_calls += 1
        return self.stream_tag

    def to_pylist(self):
        self.to_pylist_calls += 1
        return [{"id": 1}]


class RustArrowIOBridgeTests(unittest.TestCase):
    def test_arrow_c_data_interface_is_used_for_rust_input(self) -> None:
        scenario = IO_BRIDGE_SCENARIOS["rust_arrow_bridge_input"]
        bridge = RustArrowIOBridge()
        table = _FakeArrowTable(scenario["stream_tag"])
        execution_plan = {"kind": "placeholder", "steps": []}
        context = RuntimeExecutionContext(
            request=ExecuteRequest(dsl_text=scenario["dsl"]),
            ast_statements=(),
            resolved_inputs={
                "in": DataSetRef(
                    kind=scenario["input_kind"],
                    location=scenario["input_location"],
                    payload=table,
                ),
            },
            resolved_output_targets=scenario["resolved_output_targets"],
            execution_plan=execution_plan,
        )

        payload, diagnostics = bridge.build_payload(context, function_registry_keys=())

        self.assertEqual(diagnostics, scenario["expected_diagnostics"])
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload.input_streams["in"], scenario["stream_tag"])
        self.assertEqual(payload.execution_plan, execution_plan)
        self.assertEqual(table.stream_calls, scenario["expected_stream_calls"])
        self.assertEqual(table.to_pylist_calls, scenario["expected_to_pylist_calls"])

    def test_non_arrow_input_is_rejected_by_rust_bridge(self) -> None:
        scenario = IO_BRIDGE_SCENARIOS["rust_arrow_bridge_rejection"]
        bridge = RustArrowIOBridge()
        context = RuntimeExecutionContext(
            request=ExecuteRequest(dsl_text=scenario["dsl"]),
            ast_statements=(),
            resolved_inputs={
                "in": DataSetRef(
                    kind=scenario["input_kind"],
                    location=scenario["input_location"],
                    payload=scenario["input_payload"],
                ),
            },
            resolved_output_targets=scenario["resolved_output_targets"],
        )

        payload, diagnostics = bridge.build_payload(context, function_registry_keys=())

        self.assertIsNone(payload)
        self.assertEqual(diagnostics[0].code, scenario["expected_diagnostic_code"])


class RustNativeBlockExecutorTests(unittest.TestCase):
    def test_serialize_payload_uses_streams_without_to_pylist(self) -> None:
        scenario = IO_BRIDGE_SCENARIOS["native_block_executor_io"]
        executor = RustNativeBlockExecutor()
        table = _FakeArrowTable(scenario["stream_tag"])
        execution_plan = {"kind": "placeholder", "steps": ["loop"]}
        payload = RustExecutionPayload(
            ast_statements=[ParsedStatement(kind="SET", text="set in")],
            output_targets=(scenario["output_target"],),
            input_streams={"in": table.__arrow_c_stream__()},
            function_registry_keys=(),
            execution_plan=execution_plan,
        )

        serialized, diagnostic = executor._serialize_payload(payload)

        self.assertIsNone(diagnostic)
        self.assertEqual(serialized["input_streams"]["in"], scenario["stream_tag"])
        self.assertEqual(serialized["execution_plan"], execution_plan)
        self.assertNotIn("inputs", serialized)
        self.assertEqual(table.to_pylist_calls, 0)

    def test_execute_block_deserializes_output_streams_as_arrow_table(self) -> None:
        scenario = IO_BRIDGE_SCENARIOS["native_block_executor_io"]
        executor = RustNativeBlockExecutor()
        output_table = pa.Table.from_pylist(scenario["output_table_rows"])

        class _FakeNativeModule:
            @staticmethod
            def execute_block(_payload):
                return {
                    "diagnostics": [],
                    "output_streams": {scenario["output_target"]: output_table.__arrow_c_stream__()},
                }

        payload = RustExecutionPayload(
            ast_statements=[ParsedStatement(kind="SET", text="set in")],
            output_targets=(scenario["output_target"],),
            input_streams={"in": output_table.__arrow_c_stream__()},
            function_registry_keys=(),
        )

        with patch("limulus.backends.load_native_module", return_value=(_FakeNativeModule(), None)):
            outputs, diagnostics = executor.execute_block(payload)

        self.assertEqual(diagnostics, [])
        self.assertIn(scenario["output_target"], outputs)
        self.assertEqual(outputs[scenario["output_target"]].kind, scenario["expected_output_kind"])
        self.assertEqual(
            outputs[scenario["output_target"]].payload.to_pylist(),
            scenario["expected_output_rows"],
        )


if __name__ == "__main__":
    unittest.main()
