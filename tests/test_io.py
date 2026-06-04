import unittest
from unittest.mock import patch

import pyarrow as pa
import pytest

from limulus.backend_integration import (
    RuntimeExecutionContext,
    RustArrowIOBridge,
    RustExecutionPayload,
    RustNativeBlockExecutor,
)
from limulus.backend_integration.transport import (
    _PREPARED_MERGE_ROWS_MARKER,
    _PREPARED_SET_ROWS_MARKER,
    _REJECTED_TRANSPORT_PATH,
    _SPECIAL_TRANSPORT_PATH,
    _STANDARD_TRANSPORT_KIND,
    _STANDARD_TRANSPORT_PATH,
    _classify_transport_input,
)
from limulus.execution import DataStepExecutor
from limulus.execution.rewrites import RewritePlanner
from limulus.io_adapters import DataFrameAdapterPandas, DataInputAdapterArrow, DataOutputAdapterArrow, InputSpec, OutputSpec
from limulus.models import DataSetRef, ExecuteRequest
from limulus.native_bridge import load_native_module
from limulus.parser import ParsedStatement, ParserService

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
    def test_arrow_table_is_canonical_standard_carrier_across_executor_and_bridge(self) -> None:
        bridge = RustArrowIOBridge()
        executor = DataStepExecutor(runtime_backend="python")
        table = _FakeArrowTable("stream://standard")
        dataset_ref = executor._coerce_dataset_ref(name="in", dataset=table)
        context = RuntimeExecutionContext(
            request=ExecuteRequest(dsl_text="data out; set in; run;"),
            ast_statements=(),
            resolved_inputs={"in": dataset_ref},
            resolved_output_targets=("out",),
        )

        payload, diagnostics = bridge.build_payload(context, function_registry_keys=())

        self.assertEqual(dataset_ref.kind, _STANDARD_TRANSPORT_KIND)
        self.assertEqual(_classify_transport_input(dataset_ref), _STANDARD_TRANSPORT_PATH)
        self.assertEqual(diagnostics, [])
        self.assertIsNotNone(payload)

    def test_memory_rows_without_arrow_transport_are_rejected(self) -> None:
        dataset_ref = DataSetRef(
            kind="memory",
            location="dataset://in",
            payload=[{"id": 1}],
        )

        self.assertEqual(_classify_transport_input(dataset_ref), _REJECTED_TRANSPORT_PATH)

    def test_prepared_set_rows_are_classified_as_explicit_special_transport(self) -> None:
        dataset_ref = DataSetRef(
            kind="memory",
            location=f"dataset://in{_PREPARED_SET_ROWS_MARKER}",
            payload=[{"id": 1}],
        )

        self.assertEqual(_classify_transport_input(dataset_ref), _SPECIAL_TRANSPORT_PATH)

    def test_prepared_merge_rows_are_classified_as_explicit_special_transport(self) -> None:
        dataset_ref = DataSetRef(
            kind="memory",
            location=f"dataset://in{_PREPARED_MERGE_ROWS_MARKER}",
            payload=[{"id": 1}],
        )

        self.assertEqual(_classify_transport_input(dataset_ref), _SPECIAL_TRANSPORT_PATH)

    def test_set_preparation_keeps_supported_arrow_inputs_off_row_planner(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust", parser_backend="python")
        dsl_text = "data out; set a(obs=1) a; keep id; run;"
        parse_result = executor._parser_backend_selector.select("python").parse(
            type("Ctx", (), {"dsl_text": dsl_text})()
        )
        self.assertFalse(parse_result.has_errors)

        with patch.object(
            executor._rewrite_planner,
            "prepare_set_rows",
            return_value=([{"id": 1}], None),
        ) as prepare_set_rows:
            prepared_inputs, diagnostics = executor._prepare_runtime_inputs(
                parse_result.ast.statements,
                {
                    "a": DataSetRef(
                        kind="arrow_table",
                        location="dataset://a",
                        payload=pa.Table.from_pylist([{"id": 1}, {"id": 2}]),
                    )
                },
            )

        self.assertEqual(diagnostics, [])
        prepare_set_rows.assert_not_called()
        self.assertEqual(prepared_inputs["a"].kind, "arrow_table")
        self.assertIn(_PREPARED_SET_ROWS_MARKER, prepared_inputs["a"].location)

    def test_rewrite_planner_prepare_set_rows_orders_stream_before_helper_flags(self) -> None:
        planner = RewritePlanner()

        options_a = type(
            "_OptionsA",
            (),
            {
                "in_var": "in_a",
                "keep_vars": (),
                "drop_vars": (),
                "where_expr": None,
                "rename_map": {},
                "firstobs": None,
                "obs": None,
            },
        )()
        options_b = type(
            "_OptionsB",
            (),
            {
                "in_var": "in_b",
                "keep_vars": (),
                "drop_vars": (),
                "where_expr": None,
                "rename_map": {},
                "firstobs": None,
                "obs": None,
            },
        )()
        source_refs = [
            type("_RefA", (), {"name": "a", "options": options_a})(),
            type("_RefB", (), {"name": "b", "options": options_b})(),
        ]
        resolved_inputs = {
            "a": DataSetRef(kind="memory", location="dataset://a", payload=None),
            "b": DataSetRef(kind="memory", location="dataset://b", payload=None),
        }
        source_rows = {
            "a": [{"grp": "B", "id": 2}, {"grp": "A", "id": 1}],
            "b": [{"grp": "A", "id": 3}],
        }

        def _load_input_rows(input_ref: DataSetRef):
            return list(source_rows[input_ref.location.rsplit("://", 1)[-1]]), None

        rows, diagnostic = planner.prepare_set_rows(
            source_refs=source_refs,
            by_keys=("grp",),
            in_option_vars=("in_a", "in_b"),
            indsname_var="src",
            end_var="last",
            resolved_inputs=resolved_inputs,
            internal_variable_names={"in_a", "in_b", "src", "last", "FIRST.grp", "LAST.grp", "first.grp", "last.grp"},
            load_input_rows=_load_input_rows,
            apply_dataset_reference_options=lambda rows, source_name, option_spec: (list(rows), None),
            resolve_row_key=lambda row, name: name if name in row else None,
            resolve_row_value=lambda row, name: row.get(name),
            prepared_merge_marker=_PREPARED_MERGE_ROWS_MARKER,
        )

        self.assertIsNone(diagnostic)
        self.assertEqual(
            rows,
            [
                {"grp": "A", "id": 1, "in_a": 1, "in_b": 0, "src": "a", "last": 0, "FIRST.grp": 1, "LAST.grp": 0, "first.grp": 1, "last.grp": 0},
                {"grp": "A", "id": 3, "in_a": 0, "in_b": 1, "src": "b", "last": 0, "FIRST.grp": 0, "LAST.grp": 1, "first.grp": 0, "last.grp": 1},
                {"grp": "B", "id": 2, "in_a": 1, "in_b": 0, "src": "a", "last": 1, "FIRST.grp": 1, "LAST.grp": 1, "first.grp": 1, "last.grp": 1},
            ],
        )

    def test_prepared_set_rows_keep_arrow_carrier_for_arrow_inputs(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust", parser_backend="python")
        dsl_text = "data out; set a(obs=1) a; keep id; run;"
        parse_result = executor._parser_backend_selector.select("python").parse(
            type("Ctx", (), {"dsl_text": dsl_text})()
        )
        self.assertFalse(parse_result.has_errors)

        prepared_inputs, diagnostics = executor._prepare_runtime_inputs(
            parse_result.ast.statements,
            {
                "a": DataSetRef(
                    kind="arrow_table",
                    location="dataset://a",
                    payload=pa.Table.from_pylist([{"id": 1}, {"id": 2}]),
                )
            },
        )

        self.assertEqual(diagnostics, [])
        self.assertEqual(prepared_inputs["a"].kind, "arrow_table")
        self.assertIn(_PREPARED_SET_ROWS_MARKER, prepared_inputs["a"].location)
        self.assertEqual(_classify_transport_input(prepared_inputs["a"]), _SPECIAL_TRANSPORT_PATH)

    def test_prepared_set_rows_keep_arrow_carrier_without_row_materialization_for_source_options(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust", parser_backend="python")
        dsl_text = (
            "data out; "
            "set a(keep=id amount where=(id > 1) rename=(amount=amt)) b(obs=1) indsname=src end=last; "
            "output out; run;"
        )
        parse_result = executor._parser_backend_selector.select("python").parse(
            type("Ctx", (), {"dsl_text": dsl_text})()
        )
        self.assertFalse(parse_result.has_errors)

        with patch.object(
            executor._io_service,
            "load_input_rows",
            side_effect=AssertionError("set preparation should not materialize Arrow rows for supported source options"),
        ):
            prepared_inputs, diagnostics = executor._prepare_runtime_inputs(
                parse_result.ast.statements,
                {
                    "a": DataSetRef(
                        kind="arrow_table",
                        location="dataset://a",
                        payload=pa.Table.from_pylist([
                            {"id": 1, "amount": 10, "tmp": "a"},
                            {"id": 2, "amount": 20, "tmp": "b"},
                        ]),
                    ),
                    "b": DataSetRef(
                        kind="arrow_table",
                        location="dataset://b",
                        payload=pa.Table.from_pylist([
                            {"id": 3, "amount": 30},
                            {"id": 4, "amount": 40},
                        ]),
                    ),
                },
            )

        self.assertEqual(diagnostics, [])
        self.assertEqual(prepared_inputs["a"].kind, "arrow_table")
        self.assertIn(_PREPARED_SET_ROWS_MARKER, prepared_inputs["a"].location)
        self.assertEqual(
            prepared_inputs["a"].payload.to_pylist(),
            [
                {"id": 2, "amt": 20, "amount": None, "src": "a", "last": 0},
                {"id": 3, "amt": None, "amount": 30, "src": "b", "last": 1},
            ],
        )

    def test_prepared_merge_rows_keep_arrow_carrier_for_arrow_inputs(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust", parser_backend="python")
        dsl_text = (
            "data out; merge a(firstobs=2 obs=1 keep=id xa rename=(xa=x)) "
            "b(firstobs=2 obs=1 keep=id xb rename=(xb=y)); by id; output out; run;"
        )
        parse_result = executor._parser_backend_selector.select("python").parse(
            type("Ctx", (), {"dsl_text": dsl_text})()
        )
        self.assertFalse(parse_result.has_errors)

        _, prepared_inputs, merge_error = executor._prepare_merge_for_execution(
            ast_statements=parse_result.ast.statements,
            resolved_inputs={
                "a": DataSetRef(
                    kind="arrow_table",
                    location="dataset://a",
                    payload=pa.Table.from_pylist([
                        {"id": 1, "xa": 10},
                        {"id": 2, "xa": 20},
                        {"id": 3, "xa": 30},
                    ]),
                ),
                "b": DataSetRef(
                    kind="arrow_table",
                    location="dataset://b",
                    payload=pa.Table.from_pylist([
                        {"id": 2, "xb": 200},
                        {"id": 3, "xb": 300},
                        {"id": 4, "xb": 400},
                    ]),
                ),
            },
        )

        self.assertIsNone(merge_error)
        self.assertEqual(prepared_inputs["a"].kind, "arrow_table")
        self.assertIn(_PREPARED_MERGE_ROWS_MARKER, prepared_inputs["a"].location)
        self.assertEqual(_classify_transport_input(prepared_inputs["a"]), _SPECIAL_TRANSPORT_PATH)

    def test_prepared_merge_rows_keep_sparse_columns_for_arrow_inputs(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust", parser_backend="python")
        dsl_text = "data out; merge a b; by id; where id >= 2; output out; run;"
        parse_result = executor._parser_backend_selector.select("python").parse(
            type("Ctx", (), {"dsl_text": dsl_text})()
        )
        self.assertFalse(parse_result.has_errors)

        _, prepared_inputs, merge_error = executor._prepare_merge_for_execution(
            ast_statements=parse_result.ast.statements,
            resolved_inputs={
                "a": DataSetRef(
                    kind="arrow_table",
                    location="dataset://a",
                    payload=pa.Table.from_pylist([
                        {"id": 1, "x": 10},
                        {"id": 2, "x": 20},
                    ]),
                ),
                "b": DataSetRef(
                    kind="arrow_table",
                    location="dataset://b",
                    payload=pa.Table.from_pylist([
                        {"id": 2, "y": 200},
                        {"id": 3, "y": 300},
                    ]),
                ),
            },
        )

        self.assertIsNone(merge_error)
        self.assertEqual(
            prepared_inputs["a"].payload.to_pylist(),
            [
                {"id": 1, "x": 10, "y": None, "FIRST.id": 1, "LAST.id": 1, "first.id": 1, "last.id": 1},
                {"id": 2, "x": 20, "y": 200, "FIRST.id": 1, "LAST.id": 1, "first.id": 1, "last.id": 1},
                {"id": 3, "x": None, "y": 300, "FIRST.id": 1, "LAST.id": 1, "first.id": 1, "last.id": 1},
            ],
        )

    def test_prepared_merge_rows_keep_arrow_carrier_without_row_materialization_for_representative_options(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust", parser_backend="python")
        dsl_text = (
            "data out; merge a(firstobs=2 obs=1 keep=id xa rename=(xa=x)) "
            "b(firstobs=2 obs=1 keep=id xb rename=(xb=y)); by id; output out; run;"
        )
        parse_result = executor._parser_backend_selector.select("python").parse(
            type("Ctx", (), {"dsl_text": dsl_text})()
        )
        self.assertFalse(parse_result.has_errors)

        with patch.object(
            executor._io_service,
            "load_input_rows",
            side_effect=AssertionError("merge preparation should not materialize Arrow rows for representative source options"),
        ):
            _, prepared_inputs, merge_error = executor._prepare_merge_for_execution(
                ast_statements=parse_result.ast.statements,
                resolved_inputs={
                    "a": DataSetRef(
                        kind="arrow_table",
                        location="dataset://a",
                        payload=pa.Table.from_pylist([
                            {"id": 1, "xa": 10, "tmp": "a"},
                            {"id": 2, "xa": 20, "tmp": "b"},
                            {"id": 3, "xa": 30, "tmp": "c"},
                        ]),
                    ),
                    "b": DataSetRef(
                        kind="arrow_table",
                        location="dataset://b",
                        payload=pa.Table.from_pylist([
                            {"id": 1, "xb": 100},
                            {"id": 2, "xb": 200},
                            {"id": 3, "xb": 300},
                        ]),
                    ),
                },
            )

        self.assertIsNone(merge_error)
        self.assertEqual(prepared_inputs["a"].kind, "arrow_table")
        self.assertIn(_PREPARED_MERGE_ROWS_MARKER, prepared_inputs["a"].location)
        self.assertEqual(
            prepared_inputs["a"].payload.to_pylist(),
            [{"id": 2, "x": 20, "y": 200, "FIRST.id": 1, "LAST.id": 1, "first.id": 1, "last.id": 1}],
        )

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

    def test_rust_bridge_resolves_string_literal_apply_targets_into_payload(self) -> None:
        bridge = RustArrowIOBridge()

        def double(value):
            return value * 2

        context = RuntimeExecutionContext(
            request=ExecuteRequest(dsl_text="data out; set in; doubled = apply('double', amount); output out; run;"),
            ast_statements=(ParsedStatement(kind="ASSIGN", text="doubled = apply('double', amount)"),),
            resolved_inputs={
                "in": DataSetRef(
                    kind="arrow_table",
                    location="dataset://in",
                    payload=pa.Table.from_pylist([{"amount": 4}]),
                )
            },
            resolved_output_targets=("out",),
        )

        globals()["double"] = double
        payload, diagnostics = bridge.build_payload(context, function_registry_keys=())

        self.assertEqual(diagnostics, [])
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertIn("double", payload.apply_registry)
        self.assertTrue(callable(payload.apply_registry["double"]))

    def test_rust_bridge_preserves_row_loop_plan_contract_in_payload(self) -> None:
        bridge = RustArrowIOBridge()
        executor = DataStepExecutor(runtime_backend="python")
        dsl = "data out; set in; where amount >= 10; score = amount + 1; keep id score; run;"
        parse_result = executor._parser.parse(dsl)
        self.assertFalse(parse_result.has_errors)
        assert parse_result.ast is not None

        input_ref = DataSetRef(
            kind="arrow_table",
            location="dataset://in",
            payload=pa.Table.from_pylist([{"id": 1, "amount": 12}]),
        )
        execution_plan = executor._pipeline._generate_execution_plan(
            request=ExecuteRequest(dsl_text=dsl),
            ast_statements=parse_result.ast.statements,
            resolved_inputs={"in": input_ref},
            resolved_output_targets=("out",),
        )
        self.assertIsNotNone(execution_plan)
        assert execution_plan is not None

        context = RuntimeExecutionContext(
            request=ExecuteRequest(dsl_text=dsl),
            ast_statements=parse_result.ast.statements,
            resolved_inputs={"in": input_ref},
            resolved_output_targets=("out",),
            execution_plan=execution_plan,
        )

        payload, diagnostics = bridge.build_payload(context, function_registry_keys=())

        self.assertEqual(diagnostics, [])
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload.execution_plan, execution_plan)
        self.assertIn("compatibility_path_plan", payload.execution_plan)
        self.assertEqual(
            payload.execution_plan["compatibility_path_plan"],
            {
                "owner": "backend_dispatch_policy",
                "selected_path": "advanced",
                "path_role": "planner_toggle_ready",
                "reason_codes": ("PLANNER_TOGGLE_READY_STATEMENT",),
            },
        )
        self.assertIn("row_loop_plan", payload.execution_plan)
        row_loop_plan = payload.execution_plan["row_loop_plan"]
        self.assertEqual(row_loop_plan["mode"], "arrow_row_loop")
        self.assertEqual(row_loop_plan["engine_mode"], "unified_row_loop")
        self.assertEqual(row_loop_plan["backend_mode"], "rust_first")
        self.assertEqual(row_loop_plan["where_mode"], "shared_pre_row_filter")
        self.assertEqual(row_loop_plan["rewrite_mode"], "planner_owned_pending")
        self.assertEqual(row_loop_plan["cursor_kind"], "arrow_row_cursor")
        self.assertIn("source_slot_order", row_loop_plan)
        self.assertIn("mutable_slot_order", row_loop_plan)
        self.assertIn("automatic_slot_order", row_loop_plan)
        self.assertEqual(row_loop_plan["builder_mode"], "targeted_output_handoff")
        self.assertEqual(row_loop_plan["output_mode"], "targeted_output_handoff")
        self.assertEqual(row_loop_plan["materialization_policy"], "arrow_cursor")
        self.assertEqual(payload.transport_mode, "standard_arrow_stream")
        self.assertEqual(payload.builder_mode, "targeted_output_handoff")
        self.assertEqual(payload.python_limited_mode, False)
        self.assertEqual(payload.rewrite_metadata, execution_plan.get("rewrite_plan"))

    def test_python_standard_path_can_read_arrow_input_without_eager_row_materialization(self) -> None:
        class _NoPyListArrowTable:
            def __init__(self, table: pa.Table) -> None:
                self._table = table
                self.schema = table.schema
                self.num_rows = table.num_rows
                self.to_pylist_calls = 0

            def column(self, index: int):
                return self._table.column(index)

            def to_pylist(self):
                self.to_pylist_calls += 1
                raise AssertionError("table.to_pylist should not be used in standard python path")

        executor = DataStepExecutor(runtime_backend="python")
        table = _NoPyListArrowTable(pa.Table.from_pylist([{"id": 1, "amount": 2}]))

        with patch.object(
            executor._io_service,
            "load_input_rows",
            side_effect=AssertionError("load_input_rows should not be used for arrow carrier path"),
        ):
            response = executor.execute(
                ExecuteRequest(
                    dsl_text="data out; set in; amount2 = amount + 1; output out; run;",
                    inputs={"in": DataSetRef(kind="arrow_table", location="dataset://in", payload=table)},
                )
            )

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].kind, "arrow_table")
        self.assertEqual(response.outputs["out"].payload.to_pylist(), [{"id": 1, "amount": 2, "amount2": 3}])
        self.assertEqual(table.to_pylist_calls, 0)

    def test_python_standard_path_emits_arrow_output_for_arrow_carrier(self) -> None:
        executor = DataStepExecutor(runtime_backend="python")
        response = executor.execute(
            ExecuteRequest(
                dsl_text="data out; set in; amount2 = amount + 1; output out; run;",
                inputs={
                    "in": DataSetRef(
                        kind="arrow_table",
                        location="dataset://in",
                        payload=pa.Table.from_pylist([{"id": 1, "amount": 2}]),
                    )
                },
            )
        )

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].kind, "arrow_table")
        self.assertEqual(response.outputs["out"].payload.to_pylist(), [{"id": 1, "amount": 2, "amount2": 3}])

    def test_python_arrow_output_preserves_planned_schema_for_empty_result(self) -> None:
        executor = DataStepExecutor(runtime_backend="python")
        response = executor.execute(
            ExecuteRequest(
                dsl_text=(
                    "data out; "
                    "set in; "
                    "where amount > 100; "
                    "total = amount + 1; "
                    "keep id total; "
                    "rename total=score; "
                    "run;"
                ),
                inputs={
                    "in": DataSetRef(
                        kind="arrow_table",
                        location="dataset://in",
                        payload=pa.Table.from_pylist([{"id": 1, "amount": 2}]),
                    )
                },
                output_targets=["out"],
            )
        )

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].kind, "arrow_table")
        self.assertEqual(response.outputs["out"].payload.schema.names, ["id", "score"])
        self.assertEqual(response.outputs["out"].payload.to_pylist(), [])

    def test_python_arrow_output_preserves_empty_secondary_target_schema_from_plan(self) -> None:
        executor = DataStepExecutor(runtime_backend="python")
        response = executor.execute(
            ExecuteRequest(
                dsl_text=(
                    "data left(keep=id total rename=(total=left_total)) "
                    "right(keep=id total rename=(total=right_total)); "
                    "set in; "
                    "total = amount + 1; "
                    "if amount > 10 then output left; "
                    "else output right; "
                    "run;"
                ),
                inputs={
                    "in": DataSetRef(
                        kind="arrow_table",
                        location="dataset://in",
                        payload=pa.Table.from_pylist([
                            {"id": 1, "amount": 20},
                            {"id": 2, "amount": 30},
                        ]),
                    )
                },
                output_targets=["left", "right"],
            )
        )

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["left"].kind, "arrow_table")
        self.assertEqual(response.outputs["right"].kind, "arrow_table")
        self.assertEqual(response.outputs["left"].payload.schema.names, ["id", "left_total"])
        self.assertEqual(response.outputs["right"].payload.schema.names, ["id", "right_total"])
        self.assertEqual(response.outputs["left"].payload.to_pylist(), [{"id": 1, "left_total": 21}, {"id": 2, "left_total": 31}])
        self.assertEqual(response.outputs["right"].payload.to_pylist(), [])

    # These comparisons protect shared input/output handoff behavior; they do
    # not preserve an independent Python row-loop ownership contract.
    def test_output_handoff_empty_secondary_target_matches_python_preference_and_rust(self) -> None:
        request = ExecuteRequest(
            dsl_text=(
                "data left(keep=id total rename=(total=left_total)) "
                "right(keep=id total rename=(total=right_total)); "
                "set in; total = amount + 1; if amount > 10 then output left; else output right; run;"
            ),
            inputs={
                "in": DataSetRef(
                    kind="arrow_table",
                    location="dataset://in",
                    payload=pa.Table.from_pylist([
                        {"id": 1, "amount": 20},
                        {"id": 2, "amount": 30},
                    ]),
                )
            },
            output_targets=["left", "right"],
        )

        python_preference_response = DataStepExecutor(runtime_backend="python").execute(request)
        rust_executor = DataStepExecutor(runtime_backend="rust")
        rust_response = rust_executor.execute(request)

        self.assertFalse(python_preference_response.has_errors)
        self.assertFalse(rust_response.has_errors)
        self.assertEqual(
            [(d.code, d.severity, d.stage, d.message) for d in python_preference_response.diagnostics],
            [(d.code, d.severity, d.stage, d.message) for d in rust_response.diagnostics],
        )
        self.assertEqual(python_preference_response.outputs["left"].payload.schema.names, ["id", "left_total"])
        self.assertEqual(rust_response.outputs["left"].payload.schema.names, ["id", "left_total"])
        self.assertEqual(python_preference_response.outputs["right"].payload.schema.names, ["id", "right_total"])
        self.assertEqual(rust_response.outputs["right"].payload.schema.names, ["id", "right_total"])
        self.assertEqual(python_preference_response.outputs["left"].payload.to_pylist(), rust_response.outputs["left"].payload.to_pylist())
        self.assertEqual(python_preference_response.outputs["right"].payload.to_pylist(), rust_response.outputs["right"].payload.to_pylist())

    def test_output_handoff_non_empty_target_preserves_planned_types_between_python_preference_and_rust(self) -> None:
        request = ExecuteRequest(
            dsl_text=(
                "data out; "
                "set in; "
                "where id = 2; "
                "keep id amount; "
                "rename amount=score; "
                "run;"
            ),
            inputs={
                "in": DataSetRef(
                    kind="arrow_table",
                    location="dataset://in",
                    payload=pa.Table.from_pylist([
                        {"id": 1, "amount": 10},
                        {"id": 2, "amount": None},
                    ]),
                )
            },
            output_targets=["out"],
        )

        python_preference_response = DataStepExecutor(runtime_backend="python").execute(request)
        rust_executor = DataStepExecutor(runtime_backend="rust")
        rust_response = rust_executor.execute(request)

        self.assertFalse(python_preference_response.has_errors)
        self.assertFalse(rust_response.has_errors)
        self.assertEqual(
            [(d.code, d.severity, d.stage, d.message) for d in python_preference_response.diagnostics],
            [(d.code, d.severity, d.stage, d.message) for d in rust_response.diagnostics],
        )
        self.assertEqual(python_preference_response.outputs["out"].payload.to_pylist(), [{"id": 2, "score": None}])
        self.assertEqual(rust_response.outputs["out"].payload.to_pylist(), [{"id": 2, "score": None}])
        self.assertEqual(python_preference_response.outputs["out"].payload.schema.names, ["id", "score"])
        self.assertEqual(rust_response.outputs["out"].payload.schema.names, ["id", "score"])
        self.assertEqual(
            python_preference_response.outputs["out"].payload.schema.field("score").type,
            rust_response.outputs["out"].payload.schema.field("score").type,
        )

    def test_helper_column_arrow_path_matches_python_preference_and_rust(self) -> None:
        request = ExecuteRequest(
            dsl_text=(
                "data out; "
                "set in end=eof; "
                "by grp; "
                "seq = _N_; "
                "if last.grp then output out; "
                "keep grp seq eof; "
                "run;"
            ),
            inputs={
                "in": DataSetRef(
                    kind="arrow_table",
                    location="dataset://in",
                    payload=pa.Table.from_pylist(
                        [
                            {"grp": "A", "amount": 1},
                            {"grp": "A", "amount": 2},
                            {"grp": "B", "amount": 3},
                        ]
                    ),
                )
            },
            output_targets=["out"],
        )

        python_preference_response = DataStepExecutor(runtime_backend="python").execute(request)
        rust_executor = DataStepExecutor(runtime_backend="rust")
        rust_response = rust_executor.execute(request)

        self.assertFalse(python_preference_response.has_errors)
        self.assertFalse(rust_response.has_errors)
        self.assertEqual(
            [(d.code, d.severity, d.stage, d.message) for d in python_preference_response.diagnostics],
            [(d.code, d.severity, d.stage, d.message) for d in rust_response.diagnostics],
        )
        python_rows = python_preference_response.outputs["out"].payload
        if hasattr(python_rows, "to_pylist"):
            python_rows = python_rows.to_pylist()
        rust_rows = rust_response.outputs["out"].payload.to_pylist()
        self.assertEqual(python_rows, rust_rows)

    def test_native_standard_path_rejects_unsupported_binary_and_decimal_columns_with_identifiable_diagnostics(self) -> None:
        unsupported_cases = (
            (
                "payload",
                pa.table(
                    {
                        "id": pa.array([1], type=pa.int64()),
                        "payload": pa.array([b"aa"], type=pa.binary()),
                    }
                ),
                "Binary",
            ),
        )

        for field_name, payload, expected_type_name in unsupported_cases:
            with self.subTest(field_name=field_name):
                request = ExecuteRequest(
                    dsl_text=f"data out; set in; keep id {field_name}; run;",
                    inputs={
                        "in": DataSetRef(
                            kind="arrow_table",
                            location="dataset://in",
                            payload=payload,
                        )
                    },
                    output_targets=["out"],
                )

                rust_executor = DataStepExecutor(runtime_backend="rust")
                rust_response = rust_executor.execute(request)

                self.assertTrue(rust_response.has_errors)
                self.assertEqual(len(rust_response.diagnostics), 1)
                self.assertEqual(
                    rust_response.diagnostics[0].code,
                    "RUNTIME_EXPRESSION_EVALUATION_ERROR",
                )
                self.assertIn(
                    f"unsupported Arrow source column type for native cursor: {expected_type_name}",
                    rust_response.diagnostics[0].message,
                )

    def test_output_handoff_preserves_mixed_borrowed_and_owned_columns_between_python_preference_and_rust(self) -> None:
        request = ExecuteRequest(
            dsl_text=(
                "data out; "
                "set in; "
                "score = amount + 1; "
                "keep id tags score; "
                "run;"
            ),
            inputs={
                "in": DataSetRef(
                    kind="arrow_table",
                    location="dataset://in",
                    payload=pa.table(
                        {
                            "id": pa.array([1, 2], type=pa.int64()),
                            "amount": pa.array([10, 20], type=pa.int64()),
                            "tags": pa.array([["a", "b"], ["c"]], type=pa.list_(pa.string())),
                        }
                    ),
                )
            },
            output_targets=["out"],
        )

        python_preference_response = DataStepExecutor(runtime_backend="python").execute(request)
        rust_executor = DataStepExecutor(runtime_backend="rust")
        rust_response = rust_executor.execute(request)

        self.assertFalse(python_preference_response.has_errors)
        self.assertFalse(rust_response.has_errors)
        self.assertEqual(
            python_preference_response.outputs["out"].payload.to_pylist(),
            rust_response.outputs["out"].payload.to_pylist(),
        )
        self.assertEqual(
            python_preference_response.outputs["out"].payload.schema.field("tags").type,
            rust_response.outputs["out"].payload.schema.field("tags").type,
        )
        self.assertEqual(
            python_preference_response.outputs["out"].payload.schema.field("score").type,
            rust_response.outputs["out"].payload.schema.field("score").type,
        )

    def test_rust_standard_path_keeps_arrow_input_for_supported_source_dataset_options(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust", parser_backend="python")
        dsl_text = (
            "data out; "
            "set in(keep=id amount tmp drop=tmp where=(amount >= 20) rename=(amount=amt)); "
            "where amt >= 30; "
            "score = amt + 1; "
            "output out; "
            "keep id score; "
            "rename score=final_score; "
            "run;"
        )
        parse_result = executor._parser_backend_selector.select("python").parse(
            type("Ctx", (), {"dsl_text": dsl_text})()
        )
        self.assertFalse(parse_result.has_errors)

        prepared_inputs, diagnostics = executor._prepare_runtime_inputs(
            parse_result.ast.statements,
            {
                "in": DataSetRef(
                    kind="arrow_table",
                    location="dataset://in",
                    payload=pa.Table.from_pylist(
                        [
                            {"id": 1, "amount": 10, "tmp": "a"},
                            {"id": 2, "amount": 20, "tmp": "b"},
                            {"id": 3, "amount": 30, "tmp": "c"},
                        ]
                    ),
                )
            },
        )

        self.assertEqual(diagnostics, [])
        self.assertEqual(prepared_inputs["in"].kind, "arrow_table")
        self.assertEqual(prepared_inputs["in"].location, "dataset://in")

    def test_output_handoff_preserves_materialized_source_renames_between_python_preference_and_rust(self) -> None:
        request = ExecuteRequest(
            dsl_text=(
                "data out; "
                "set in(keep=id amount rename=(amount=amt)); "
                "score = amt + 1; "
                "keep id amt score; "
                "rename amt=final_amt score=final_score; "
                "run;"
            ),
            inputs={
                "in": DataSetRef(
                    kind="arrow_table",
                    location="dataset://in",
                    payload=pa.Table.from_pylist([
                        {"id": 1, "amount": 10},
                        {"id": 2, "amount": 20},
                    ]),
                )
            },
            output_targets=["out"],
        )

        python_preference_response = DataStepExecutor(runtime_backend="python").execute(request)
        rust_executor = DataStepExecutor(runtime_backend="rust")
        rust_response = rust_executor.execute(request)

        self.assertFalse(python_preference_response.has_errors)
        self.assertFalse(rust_response.has_errors)
        self.assertEqual(
            python_preference_response.outputs["out"].payload.to_pylist(),
            [
                {"id": 1, "final_amt": 10, "final_score": 11},
                {"id": 2, "final_amt": 20, "final_score": 21},
            ],
        )
        self.assertEqual(
            rust_response.outputs["out"].payload.to_pylist(),
            python_preference_response.outputs["out"].payload.to_pylist(),
        )

    def test_rewrite_preparation_keeps_single_source_arrow_carrier_without_row_materialization(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust", parser_backend="python")
        dsl_text = (
            "data out; "
            "set in; "
            "prev_id = lag(id); "
            "next_id = lead(id); "
            "keep id prev_id next_id; "
            "run;"
        )
        parse_result = executor._parser_backend_selector.select("python").parse(
            type("Ctx", (), {"dsl_text": dsl_text})()
        )
        self.assertFalse(parse_result.has_errors)

        resolved_inputs = {
            "in": DataSetRef(
                kind="arrow_table",
                location="dataset://in",
                payload=pa.Table.from_pylist([
                    {"id": 1},
                    {"id": 2},
                    {"id": 3},
                ]),
            )
        }

        with patch.object(
            executor._io_service,
            "load_input_rows",
            side_effect=AssertionError("rewrite preparation should not materialize Arrow rows for single-source carrier"),
        ):
            statements, prepared_inputs, diagnostics = executor._prepare_runtime_evaluations(
                ast_statements=parse_result.ast.statements,
                resolved_inputs=resolved_inputs,
            )

        self.assertEqual(diagnostics, [])
        statement_texts = [statement.text for statement in statements]
        self.assertIn("prev_id = __rewrite_lag_id_1_1__", statement_texts)
        self.assertIn("next_id = __rewrite_lead_id_1_2__", statement_texts)
        self.assertEqual(prepared_inputs["in"].kind, "arrow_table")
        self.assertIn(_PREPARED_SET_ROWS_MARKER, prepared_inputs["in"].location)
        self.assertEqual(
            prepared_inputs["in"].payload.to_pylist(),
            [
                {"id": 1, "__rewrite_lag_id_1_1__": None, "__rewrite_lead_id_1_2__": 2},
                {"id": 2, "__rewrite_lag_id_1_1__": 1, "__rewrite_lead_id_1_2__": 3},
                {"id": 3, "__rewrite_lag_id_1_1__": 2, "__rewrite_lead_id_1_2__": None},
            ],
        )

    def test_rewrite_preparation_keeps_single_source_arrow_carrier_with_source_options_without_row_materialization(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust", parser_backend="python")
        dsl_text = (
            "data out; "
            "set in(obs=2); "
            "prev_id = lag(id); "
            "keep id prev_id; "
            "run;"
        )
        parse_result = executor._parser_backend_selector.select("python").parse(
            type("Ctx", (), {"dsl_text": dsl_text})()
        )
        self.assertFalse(parse_result.has_errors)

        initial_inputs = {
            "in": DataSetRef(
                kind="arrow_table",
                location="dataset://in",
                payload=pa.Table.from_pylist([
                    {"id": 1},
                    {"id": 2},
                    {"id": 3},
                ]),
            )
        }

        with patch.object(
            executor._io_service,
            "load_input_rows",
            side_effect=AssertionError(
                "rewrite preparation should not materialize rows for single-source source-local dataset options"
            ),
        ):
            prepared_inputs, prepare_diagnostics = executor._prepare_runtime_inputs(
                parse_result.ast.statements,
                initial_inputs,
            )
            statements, rewritten_inputs, diagnostics = executor._prepare_runtime_evaluations(
                ast_statements=parse_result.ast.statements,
                resolved_inputs=prepared_inputs,
            )

        self.assertEqual(prepare_diagnostics, [])
        self.assertEqual(diagnostics, [])
        statement_texts = [statement.text for statement in statements]
        self.assertIn("prev_id = __rewrite_lag_id_1_1__", statement_texts)
        self.assertEqual(rewritten_inputs["in"].kind, "arrow_table")
        self.assertIn(_PREPARED_SET_ROWS_MARKER, rewritten_inputs["in"].location)
        self.assertEqual(
            rewritten_inputs["in"].payload.to_pylist(),
            [
                {"id": 1, "__rewrite_lag_id_1_1__": None},
                {"id": 2, "__rewrite_lag_id_1_1__": 1},
            ],
        )

    def test_rewrite_preparation_preserves_by_order_flags_on_prepared_arrow_carrier(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust", parser_backend="python")
        dsl_text = (
            "data out; "
            "set a b; "
            "by grp; "
            "prev_id = lag(id); "
            "keep grp id prev_id first.grp last.grp; "
            "run;"
        )
        parse_result = executor._parser_backend_selector.select("python").parse(
            type("Ctx", (), {"dsl_text": dsl_text})()
        )
        self.assertFalse(parse_result.has_errors)

        initial_inputs = {
            "a": DataSetRef(
                kind="arrow_table",
                location="dataset://a",
                payload=pa.Table.from_pylist([
                    {"grp": "B", "id": 2},
                    {"grp": "A", "id": 1},
                ]),
            ),
            "b": DataSetRef(
                kind="arrow_table",
                location="dataset://b",
                payload=pa.Table.from_pylist([
                    {"grp": "A", "id": 3},
                ]),
            ),
        }

        with patch.object(
            executor._io_service,
            "load_input_rows",
            side_effect=AssertionError(
                "rewrite preparation should preserve BY-order Arrow carriers without row materialization"
            ),
        ):
            prepared_inputs, prepare_diagnostics = executor._prepare_runtime_inputs(
                parse_result.ast.statements,
                initial_inputs,
            )
            statements, rewritten_inputs, diagnostics = executor._prepare_runtime_evaluations(
                ast_statements=parse_result.ast.statements,
                resolved_inputs=prepared_inputs,
            )

        self.assertEqual(prepare_diagnostics, [])
        self.assertEqual(diagnostics, [])
        statement_texts = [statement.text for statement in statements]
        self.assertIn("prev_id = __rewrite_lag_id_1_1__", statement_texts)
        self.assertEqual(rewritten_inputs["a"].kind, "arrow_table")
        self.assertIn(_PREPARED_SET_ROWS_MARKER, rewritten_inputs["a"].location)
        self.assertEqual(
            rewritten_inputs["a"].payload.to_pylist(),
            [
                {
                    "grp": "A",
                    "id": 1,
                    "FIRST.grp": 1,
                    "LAST.grp": 0,
                    "first.grp": 1,
                    "last.grp": 0,
                    "__rewrite_lag_id_1_1__": None,
                },
                {
                    "grp": "A",
                    "id": 3,
                    "FIRST.grp": 0,
                    "LAST.grp": 1,
                    "first.grp": 0,
                    "last.grp": 1,
                    "__rewrite_lag_id_1_1__": 1,
                },
                {
                    "grp": "B",
                    "id": 2,
                    "FIRST.grp": 1,
                    "LAST.grp": 1,
                    "first.grp": 1,
                    "last.grp": 1,
                    "__rewrite_lag_id_1_1__": 3,
                },
            ],
        )

    def test_rewrite_preparation_keeps_prepared_multi_source_arrow_carrier_without_row_materialization(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust", parser_backend="python")
        dsl_text = (
            "data out; "
            "set a(obs=1) b indsname=src end=last; "
            "prev_id = lag(id); "
            "keep id prev_id; "
            "run;"
        )
        parse_result = executor._parser_backend_selector.select("python").parse(
            type("Ctx", (), {"dsl_text": dsl_text})()
        )
        self.assertFalse(parse_result.has_errors)

        initial_inputs = {
            "a": DataSetRef(
                kind="arrow_table",
                location="dataset://a",
                payload=pa.Table.from_pylist([
                    {"id": 1},
                    {"id": 2},
                ]),
            ),
            "b": DataSetRef(
                kind="arrow_table",
                location="dataset://b",
                payload=pa.Table.from_pylist([
                    {"id": 3},
                    {"id": 4},
                ]),
            ),
        }

        with patch.object(
            executor._io_service,
            "load_input_rows",
            side_effect=AssertionError(
                "rewrite preparation should not materialize rows after source preparation has produced a prepared Arrow carrier"
            ),
        ):
            prepared_inputs, prepare_diagnostics = executor._prepare_runtime_inputs(
                parse_result.ast.statements,
                initial_inputs,
            )
            statements, rewritten_inputs, diagnostics = executor._prepare_runtime_evaluations(
                ast_statements=parse_result.ast.statements,
                resolved_inputs=prepared_inputs,
            )

        self.assertEqual(prepare_diagnostics, [])
        self.assertEqual(diagnostics, [])
        statement_texts = [statement.text for statement in statements]
        self.assertIn("prev_id = __rewrite_lag_id_1_1__", statement_texts)
        self.assertEqual(rewritten_inputs["a"].kind, "arrow_table")
        self.assertIn(_PREPARED_SET_ROWS_MARKER, rewritten_inputs["a"].location)
        self.assertEqual(
            rewritten_inputs["a"].payload.to_pylist(),
            [
                {"id": 1, "src": "a", "last": 0, "__rewrite_lag_id_1_1__": None},
                {"id": 3, "src": "b", "last": 0, "__rewrite_lag_id_1_1__": 1},
                {"id": 4, "src": "b", "last": 1, "__rewrite_lag_id_1_1__": 3},
            ],
        )

    def test_rewrite_preparation_marks_explicit_fallback_reason_when_columnar_rewrite_is_not_supported(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust", parser_backend="python")
        dsl_text = (
            "data out; "
            "set in; "
            "prev_id = lag(id); "
            "keep id prev_id; "
            "run;"
        )
        parse_result = executor._parser_backend_selector.select("python").parse(
            type("Ctx", (), {"dsl_text": dsl_text})()
        )
        self.assertFalse(parse_result.has_errors)

        statements, rewritten_inputs, diagnostics = executor._prepare_runtime_evaluations(
            ast_statements=parse_result.ast.statements,
            resolved_inputs={
                "in": DataSetRef(
                    kind="memory",
                    location="dataset://in",
                    payload=[
                        {"id": 1},
                        {"id": 2},
                    ],
                )
            },
        )

        self.assertEqual(diagnostics, [])
        statement_texts = [statement.text for statement in statements]
        self.assertIn("prev_id = __rewrite_lag_id_1_1__", statement_texts)
        self.assertEqual(rewritten_inputs["in"].kind, "memory")
        self.assertIn("|rewrite_fallback=arrow_columnar_not_supported", rewritten_inputs["in"].location)
        self.assertEqual(
            rewritten_inputs["in"].payload,
            [
                {"id": 1, "__rewrite_lag_id_1_1__": None},
                {"id": 2, "__rewrite_lag_id_1_1__": 1},
            ],
        )

    def test_prepared_merge_mode_keeps_arrow_output_on_native_success_path(self) -> None:
        executor = RustNativeBlockExecutor()
        output_table = pa.Table.from_pylist(
            [
                {"id": 2, "x": 20, "y": 200},
                {"id": 3, "x": None, "y": 300},
            ]
        )

        class _FakeNativeModule:
            @staticmethod
            def execute_block(_payload):
                return {
                    "diagnostics": [],
                    "output_streams": {"out": output_table.__arrow_c_stream__()},
                }

        payload = RustExecutionPayload(
            ast_statements=[ParsedStatement(kind="SET", text="set in")],
            output_targets=("out",),
            input_streams={"in": output_table.__arrow_c_stream__()},
            function_registry_keys=(),
            prepared_merge_mode=True,
        )

        with patch("limulus.backend_integration.rust_executor.load_native_module", return_value=(_FakeNativeModule(), None)):
            outputs, diagnostics = executor.execute_block(payload)

        self.assertEqual(diagnostics, [])
        self.assertEqual(outputs["out"].kind, "arrow_table")
        self.assertEqual(
            outputs["out"].payload.to_pylist(),
            [
                {"id": 2, "x": 20, "y": 200},
                {"id": 3, "x": None, "y": 300},
            ],
        )

    def test_rust_standard_path_keeps_arrow_input_for_set_helper_columns(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust", parser_backend="python")
        dsl_text = (
            "data out; "
            "set a(in=in_a) b(in=in_b) indsname=src end=last; "
            "if last then output out; "
            "keep id src last; "
            "run;"
        )
        parse_result = executor._parser_backend_selector.select("python").parse(
            type("Ctx", (), {"dsl_text": dsl_text})()
        )
        self.assertFalse(parse_result.has_errors)

        prepared_inputs, diagnostics = executor._prepare_runtime_inputs(
            parse_result.ast.statements,
            {
                "a": DataSetRef(
                    kind="arrow_table",
                    location="dataset://a",
                    payload=pa.Table.from_pylist([{"id": 1}]),
                ),
                "b": DataSetRef(
                    kind="arrow_table",
                    location="dataset://b",
                    payload=pa.Table.from_pylist([{"id": 2}, {"id": 3}]),
                ),
            },
        )

        self.assertEqual(diagnostics, [])
        self.assertTrue(prepared_inputs)
        for dataset_ref in prepared_inputs.values():
            self.assertEqual(dataset_ref.kind, "arrow_table")
            self.assertNotIn("#prepared_set_rows", dataset_ref.location)

    def test_auto_standard_path_keeps_arrow_input_for_set_helper_columns(self) -> None:
        executor = DataStepExecutor(runtime_backend="auto", parser_backend="python")
        dsl_text = (
            "data out; "
            "set a(in=in_a) b(in=in_b) indsname=src end=last; "
            "if last then output out; "
            "keep id src last; "
            "run;"
        )
        parse_result = executor._parser_backend_selector.select("python").parse(
            type("Ctx", (), {"dsl_text": dsl_text})()
        )
        self.assertFalse(parse_result.has_errors)

        prepared_inputs, diagnostics = executor._prepare_runtime_inputs(
            parse_result.ast.statements,
            {
                "a": DataSetRef(
                    kind="arrow_table",
                    location="dataset://a",
                    payload=pa.Table.from_pylist([{"id": 1}]),
                ),
                "b": DataSetRef(
                    kind="arrow_table",
                    location="dataset://b",
                    payload=pa.Table.from_pylist([{"id": 2}, {"id": 3}]),
                ),
            },
        )

        self.assertEqual(diagnostics, [])
        self.assertTrue(prepared_inputs)
        for dataset_ref in prepared_inputs.values():
            self.assertEqual(dataset_ref.kind, "arrow_table")
            self.assertNotIn("#prepared_set_rows", dataset_ref.location)

    def test_rust_end_option_marks_only_final_concatenated_set_row(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust", parser_backend="python")
        response = executor.execute(
            ExecuteRequest(
                dsl_text=(
                    "data out; "
                    "set a(in=in_a) b(in=in_b) indsname=src end=last; "
                    "if last then output out; "
                    "keep id src last; "
                    "run;"
                ),
                inputs={
                    "a": DataSetRef(
                        kind="arrow_table",
                        location="dataset://a",
                        payload=pa.Table.from_pylist([{"id": 1}]),
                    ),
                    "b": DataSetRef(
                        kind="arrow_table",
                        location="dataset://b",
                        payload=pa.Table.from_pylist([{"id": 2}, {"id": 3}]),
                    ),
                },
                output_targets=["out"],
            )
        )

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].kind, "arrow_table")
        self.assertEqual(response.outputs["out"].payload.to_pylist(), [{"id": 3}])

    def test_auto_end_option_uses_rust_arrow_path_for_final_concatenated_set_row(self) -> None:
        executor = DataStepExecutor(runtime_backend="auto", parser_backend="python")
        response = executor.execute(
            ExecuteRequest(
                dsl_text=(
                    "data out; "
                    "set a(in=in_a) b(in=in_b) indsname=src end=last; "
                    "if last then output out; "
                    "keep id src last; "
                    "run;"
                ),
                inputs={
                    "a": DataSetRef(
                        kind="arrow_table",
                        location="dataset://a",
                        payload=pa.Table.from_pylist([{"id": 1}]),
                    ),
                    "b": DataSetRef(
                        kind="arrow_table",
                        location="dataset://b",
                        payload=pa.Table.from_pylist([{"id": 2}, {"id": 3}]),
                    ),
                },
                output_targets=["out"],
            )
        )

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].kind, "arrow_table")
        self.assertEqual(response.outputs["out"].payload.to_pylist(), [{"id": 3}])

    def test_rust_standard_path_keeps_arrow_input_for_by_helper_columns(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust", parser_backend="python")
        dsl_text = (
            "data out; "
            "set in end=eof; "
            "by grp; "
            "if last.grp then output out; "
            "keep grp amount eof; "
            "run;"
        )
        parse_result = executor._parser_backend_selector.select("python").parse(
            type("Ctx", (), {"dsl_text": dsl_text})()
        )
        self.assertFalse(parse_result.has_errors)

        prepared_inputs, diagnostics = executor._prepare_runtime_inputs(
            parse_result.ast.statements,
            {
                "in": DataSetRef(
                    kind="arrow_table",
                    location="dataset://in",
                    payload=pa.Table.from_pylist(
                        [
                            {"grp": "A", "amount": 1},
                            {"grp": "A", "amount": 2},
                            {"grp": "B", "amount": 3},
                        ]
                    ),
                )
            },
        )

        self.assertEqual(diagnostics, [])
        self.assertEqual(prepared_inputs["in"].kind, "arrow_table")
        self.assertNotIn("#prepared_set_rows", prepared_inputs["in"].location)

    def test_auto_standard_path_keeps_arrow_input_for_by_helper_columns(self) -> None:
        executor = DataStepExecutor(runtime_backend="auto", parser_backend="python")
        dsl_text = (
            "data out; "
            "set in end=eof; "
            "by grp; "
            "if last.grp then output out; "
            "keep grp amount eof; "
            "run;"
        )
        parse_result = executor._parser_backend_selector.select("python").parse(
            type("Ctx", (), {"dsl_text": dsl_text})()
        )
        self.assertFalse(parse_result.has_errors)

        prepared_inputs, diagnostics = executor._prepare_runtime_inputs(
            parse_result.ast.statements,
            {
                "in": DataSetRef(
                    kind="arrow_table",
                    location="dataset://in",
                    payload=pa.Table.from_pylist(
                        [
                            {"grp": "A", "amount": 1},
                            {"grp": "A", "amount": 2},
                            {"grp": "B", "amount": 3},
                        ]
                    ),
                )
            },
        )

        self.assertEqual(diagnostics, [])
        self.assertEqual(prepared_inputs["in"].kind, "arrow_table")
        self.assertNotIn("#prepared_set_rows", prepared_inputs["in"].location)

    def test_rust_by_helper_columns_preserve_n_counter_on_last_group_output(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust", parser_backend="python")
        response = executor.execute(
            ExecuteRequest(
                dsl_text=(
                    "data out; "
                    "set in end=eof; "
                    "by grp; "
                    "seq = _N_; "
                    "if last.grp then output out; "
                    "keep grp seq eof; "
                    "run;"
                ),
                inputs={
                    "in": DataSetRef(
                        kind="arrow_table",
                        location="dataset://in",
                        payload=pa.Table.from_pylist(
                            [
                                {"grp": "A", "amount": 1},
                                {"grp": "A", "amount": 2},
                                {"grp": "B", "amount": 3},
                            ]
                        ),
                    )
                },
                output_targets=["out"],
            )
        )

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].kind, "arrow_table")
        self.assertEqual(
            response.outputs["out"].payload.to_pylist(),
            [{"grp": "A", "seq": 2}, {"grp": "B", "seq": 3}],
        )

    def test_auto_by_helper_columns_preserve_n_counter_on_last_group_output(self) -> None:
        executor = DataStepExecutor(runtime_backend="auto", parser_backend="python")
        response = executor.execute(
            ExecuteRequest(
                dsl_text=(
                    "data out; "
                    "set in end=eof; "
                    "by grp; "
                    "seq = _N_; "
                    "if last.grp then output out; "
                    "keep grp seq eof; "
                    "run;"
                ),
                inputs={
                    "in": DataSetRef(
                        kind="arrow_table",
                        location="dataset://in",
                        payload=pa.Table.from_pylist(
                            [
                                {"grp": "A", "amount": 1},
                                {"grp": "A", "amount": 2},
                                {"grp": "B", "amount": 3},
                            ]
                        ),
                    )
                },
                output_targets=["out"],
            )
        )

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].kind, "arrow_table")
        self.assertEqual(
            response.outputs["out"].payload.to_pylist(),
            [{"grp": "A", "seq": 2}, {"grp": "B", "seq": 3}],
        )

    def test_non_arrow_input_is_rejected_by_rust_bridge(self) -> None:
        scenario = IO_BRIDGE_SCENARIOS["rust_arrow_bridge_rejection"]
        bridge = RustArrowIOBridge()
        dataset_ref = DataSetRef(
            kind=scenario["input_kind"],
            location=scenario["input_location"],
            payload=scenario["input_payload"],
        )
        context = RuntimeExecutionContext(
            request=ExecuteRequest(dsl_text=scenario["dsl"]),
            ast_statements=(),
            resolved_inputs={"in": dataset_ref},
            resolved_output_targets=scenario["resolved_output_targets"],
        )

        payload, diagnostics = bridge.build_payload(context, function_registry_keys=())

        self.assertIsNone(payload)
        self.assertNotEqual(_classify_transport_input(dataset_ref), _STANDARD_TRANSPORT_PATH)
        self.assertEqual(diagnostics[0].code, scenario["expected_diagnostic_code"])


def test_io_interop_arrow_and_pandas_adapters_roundtrip_rows() -> None:
    pyarrow = pytest.importorskip("pyarrow")
    pandas = pytest.importorskip("pandas")

    arrow_input = DataInputAdapterArrow()
    arrow_output = DataOutputAdapterArrow()
    pandas_adapter = DataFrameAdapterPandas()

    arrow_rows = arrow_input.load(
        InputSpec(format="arrow_table", payload=pyarrow.table({"id": [1], "value": [10]}))
    )
    assert arrow_rows == [{"id": 1, "value": 10}]

    stored = arrow_output.store(
        [{"id": 1, "value": 10}],
        OutputSpec(format="arrow_table", location="memory://roundtrip"),
    )
    roundtrip_rows = arrow_input.load(InputSpec(format="arrow_table", payload=stored.payload))
    assert roundtrip_rows == [{"id": 1, "value": 10}]

    frame = pandas.DataFrame([{"id": 11, "value": None}])
    pandas_rows = pandas_adapter.load(InputSpec(format="pandas", payload=frame))
    assert pandas_rows == [{"id": 11, "value": None}]

if __name__ == "__main__":
    unittest.main()
