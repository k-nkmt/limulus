import unittest
from unittest.mock import patch

import pyarrow as pa
import pytest

from limulus.backend_integration.transport import (
    _STANDARD_TRANSPORT_PATH,
    _classify_transport_input,
)
from limulus.execution import DataStepExecutor
from limulus.execution import input_preparation as input_preparation_module
from limulus.execution.contracts import (
    InputPreparationRequest,
    InputResolutionResult,
)
from limulus.execution.rewrites import RewritePlan
from limulus.execution.rewrites import prepare_set_rows
from limulus.io_adapters import DataFrameAdapterPandas, DataInputAdapterArrow, DataOutputAdapterArrow, InputSpec, OutputSpec
from limulus.models import (
    DataSetRef,
    ExecuteRequest,
    NormalizedExecutionInputs,
)
from limulus.parser import ParserService


def _plan_for(executor: DataStepExecutor, statements: object, inputs: object) -> object:
    del executor, statements, inputs
    return RewritePlan(
        merge_mode="none",
        lag_lead_mode="none",
        where_mode="none",
        helper_source_columns=(),
        helper_runtime_columns=(),
    )


def _prepare_runtime_inputs_for_test(executor, ast_statements, resolved_inputs, **kwargs):
    kwargs.setdefault(
        "rewrite_plan",
        _plan_for(executor, ast_statements, resolved_inputs),
    )
    result = executor._input_preparation_service.prepare(
        InputPreparationRequest(
            ast_statements=tuple(ast_statements),
            input_resolution=InputResolutionResult(
                resolved_inputs=NormalizedExecutionInputs(inputs=resolved_inputs),
            ),
            rewrite_plan=kwargs["rewrite_plan"],
        )
    )
    return dict(result.prepared.inputs), list(result.diagnostics)


def _prepare_merge_for_execution_for_test(executor, **kwargs):
    kwargs.setdefault(
        "rewrite_plan",
        _plan_for(executor, kwargs["ast_statements"], kwargs["resolved_inputs"]),
    )
    result = executor._input_preparation_service.prepare(
        InputPreparationRequest(
            ast_statements=tuple(kwargs["ast_statements"]),
            input_resolution=InputResolutionResult(
                resolved_inputs=NormalizedExecutionInputs(
                    inputs=kwargs["resolved_inputs"]
                )
            ),
            rewrite_plan=kwargs["rewrite_plan"],
        )
    )
    return (
        result.prepared_statements,
        dict(result.prepared.inputs),
        result.diagnostics[0] if result.diagnostics else None,
    )


class SharedArrowPreparationTests(unittest.TestCase):
    def test_multi_set_interleave_records_prepared_set_provenance(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust")
        dsl_text = "data out; set a b; output out; run;"
        parse_result = ParserService().parse(dsl_text)
        self.assertFalse(parse_result.has_errors)

        prepared_inputs, diagnostics = _prepare_runtime_inputs_for_test(executor,
            parse_result.ast.statements,
            {
                "a": DataSetRef(
                    kind="arrow_table",
                    location="dataset://a",
                    payload=pa.table({"id": [1]}),
                ),
                "b": DataSetRef(
                    kind="arrow_table",
                    location="dataset://b",
                    payload=pa.table({"id": [2]}),
                ),
            },
        )

        self.assertEqual(diagnostics, [])
        self.assertEqual(tuple(prepared_inputs), ("a",))
        self.assertEqual(prepared_inputs["a"].preparation.kind, "prepared_set")

    def test_set_preparation_keeps_supported_arrow_inputs_off_row_planner(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust")
        dsl_text = "data out; set a(obs=1) a; keep id; run;"
        parse_result = ParserService().parse(dsl_text)
        self.assertFalse(parse_result.has_errors)

        with patch.object(
            input_preparation_module,
            "prepare_set_rows",
            return_value=([{"id": 1}], None),
        ) as prepare_set_rows:
            prepared_inputs, diagnostics = _prepare_runtime_inputs_for_test(executor,
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
        self.assertEqual(prepared_inputs["a"].preparation.kind, "prepared_set")

    def test_rewrite_planner_prepare_set_rows_orders_stream_before_helper_flags(self) -> None:
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

        rows, diagnostic = prepare_set_rows(
            source_refs=source_refs,
            by_keys=("grp",),
            in_option_vars=("in_a", "in_b"),
            indsname_var="src",
            end_var="last",
            resolved_inputs=resolved_inputs,
            internal_variable_names={"in_a", "in_b", "src", "last", "FIRST.grp", "LAST.grp", "first.grp", "last.grp"},
            load_input_rows=_load_input_rows,
            resolve_row_key=lambda row, name: name if name in row else None,
            resolve_row_value=lambda row, name: row.get(name),
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
        executor = DataStepExecutor(runtime_backend="rust")
        dsl_text = "data out; set a(obs=1) a; keep id; run;"
        parse_result = ParserService().parse(dsl_text)
        self.assertFalse(parse_result.has_errors)

        prepared_inputs, diagnostics = _prepare_runtime_inputs_for_test(executor,
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
        self.assertEqual(prepared_inputs["a"].preparation.kind, "prepared_set")
        self.assertEqual(_classify_transport_input(prepared_inputs["a"]), _STANDARD_TRANSPORT_PATH)

    def test_prepared_set_rows_convert_memory_fallback_to_arrow_carrier(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust")
        dsl_text = "data out; set a(in=from_a) a(in=from_a) indsname=source; output out; run;"
        parse_result = ParserService().parse(dsl_text)
        self.assertFalse(parse_result.has_errors)

        prepared_inputs, diagnostics = _prepare_runtime_inputs_for_test(executor,
            parse_result.ast.statements,
            {
                "a": DataSetRef(
                    kind="memory",
                    location="dataset://a",
                    payload=[{"id": 1}],
                    metadata={"origin": "a"},
                ),
            },
        )

        self.assertEqual(diagnostics, [])
        prepared = prepared_inputs["a"]
        self.assertEqual(prepared.kind, "arrow_table")
        self.assertTrue(hasattr(prepared.payload, "__arrow_c_stream__"))
        self.assertEqual(prepared.metadata, {"origin": "a"})


    def test_prepared_set_rows_keep_arrow_carrier_without_row_materialization_for_source_options(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust")
        dsl_text = (
            "data out; "
            "set a(keep=id amount where=(id > 1) rename=(amount=amt)) b(obs=1) indsname=src end=last; "
            "output out; run;"
        )
        parse_result = ParserService().parse(dsl_text)
        self.assertFalse(parse_result.has_errors)

        with patch.object(
            executor._io_service,
            "load_input_rows",
            side_effect=AssertionError("set preparation should not materialize Arrow rows for supported source options"),
        ):
            prepared_inputs, diagnostics = _prepare_runtime_inputs_for_test(executor,
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
        self.assertEqual(prepared_inputs["a"].preparation.kind, "prepared_set")
        self.assertEqual(
            prepared_inputs["a"].payload.to_pylist(),
            [
                {"id": 2, "amt": 20, "amount": None, "src": "a", "last": 0},
                {"id": 3, "amt": None, "amount": 30, "src": "b", "last": 1},
            ],
        )

    def test_prepared_merge_rows_keep_arrow_carrier_for_arrow_inputs(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust")
        dsl_text = (
            "data out; merge a(firstobs=2 obs=1 keep=id xa rename=(xa=x)) "
            "b(firstobs=2 obs=1 keep=id xb rename=(xb=y)); by id; output out; run;"
        )
        parse_result = ParserService().parse(dsl_text)
        self.assertFalse(parse_result.has_errors)

        _, prepared_inputs, merge_error = _prepare_merge_for_execution_for_test(executor,
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
        self.assertEqual(prepared_inputs["a"].preparation.kind, "prepared_merge")
        self.assertEqual(_classify_transport_input(prepared_inputs["a"]), _STANDARD_TRANSPORT_PATH)

    def test_prepared_merge_rows_convert_memory_fallback_to_arrow_carrier(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust")
        dsl_text = "data out; merge a b; output out; run;"
        parse_result = ParserService().parse(dsl_text)
        self.assertFalse(parse_result.has_errors)

        _, prepared_inputs, merge_error = _prepare_merge_for_execution_for_test(executor,
            ast_statements=parse_result.ast.statements,
            resolved_inputs={
                "a": DataSetRef(
                    kind="memory",
                    location="dataset://a",
                    payload=[{"x": 10}],
                    metadata={"origin": "a"},
                ),
                "b": DataSetRef(
                    kind="memory",
                    location="dataset://b",
                    payload=[{"y": 20}],
                ),
            },
        )

        self.assertIsNone(merge_error)
        prepared = prepared_inputs["a"]
        self.assertEqual(prepared.kind, "arrow_table")
        self.assertTrue(hasattr(prepared.payload, "__arrow_c_stream__"))
        self.assertEqual(prepared.metadata, {"origin": "a"})

    def test_prepared_merge_rows_keep_sparse_columns_for_arrow_inputs(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust")
        dsl_text = "data out; merge a b; by id; where id >= 2; output out; run;"
        parse_result = ParserService().parse(dsl_text)
        self.assertFalse(parse_result.has_errors)

        _, prepared_inputs, merge_error = _prepare_merge_for_execution_for_test(executor,
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
        executor = DataStepExecutor(runtime_backend="rust")
        dsl_text = (
            "data out; merge a(firstobs=2 obs=1 keep=id xa rename=(xa=x)) "
            "b(firstobs=2 obs=1 keep=id xb rename=(xb=y)); by id; output out; run;"
        )
        parse_result = ParserService().parse(dsl_text)
        self.assertFalse(parse_result.has_errors)

        with patch.object(
            executor._io_service,
            "load_input_rows",
            side_effect=AssertionError("merge preparation should not materialize Arrow rows for representative source options"),
        ):
            _, prepared_inputs, merge_error = _prepare_merge_for_execution_for_test(executor,
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
        self.assertEqual(prepared_inputs["a"].preparation.kind, "prepared_merge")
        self.assertEqual(
            prepared_inputs["a"].payload.to_pylist(),
            [{"id": 2, "x": 20, "y": 200, "FIRST.id": 1, "LAST.id": 1, "first.id": 1, "last.id": 1}],
        )

    def test_rust_row_path_emits_arrow_output_for_arrow_carrier(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust")
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

    def test_rust_arrow_output_preserves_planned_schema_for_empty_result(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust")
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

    def test_rust_arrow_output_preserves_empty_secondary_target_schema_from_plan(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust")
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

    def test_output_handoff_empty_secondary_target_uses_rust_row_contract(self) -> None:
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

        rust_response = DataStepExecutor(runtime_backend="rust").execute(request)

        self.assertFalse(rust_response.has_errors)
        self.assertEqual(rust_response.outputs["left"].payload.schema.names, ["id", "left_total"])
        self.assertEqual(rust_response.outputs["right"].payload.schema.names, ["id", "right_total"])
        self.assertEqual(
            rust_response.outputs["left"].payload.to_pylist(),
            [{"id": 1, "left_total": 21}, {"id": 2, "left_total": 31}],
        )
        self.assertEqual(rust_response.outputs["right"].payload.to_pylist(), [])

    def test_output_handoff_non_empty_target_preserves_planned_types_in_rust_row(self) -> None:
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

        rust_response = DataStepExecutor(runtime_backend="rust").execute(request)

        self.assertFalse(rust_response.has_errors)
        self.assertEqual(rust_response.outputs["out"].payload.to_pylist(), [{"id": 2, "score": None}])
        self.assertEqual(rust_response.outputs["out"].payload.schema.names, ["id", "score"])

    def test_helper_column_arrow_path_uses_rust_row_contract(self) -> None:
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

        rust_response = DataStepExecutor(runtime_backend="rust").execute(request)

        self.assertFalse(rust_response.has_errors)
        self.assertEqual(
            rust_response.outputs["out"].payload.to_pylist(),
            [{"grp": "A", "seq": 2.0}, {"grp": "B", "seq": 3.0}],
        )

    def test_output_handoff_preserves_mixed_borrowed_and_owned_columns_in_rust_row(self) -> None:
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

        rust_response = DataStepExecutor(runtime_backend="rust").execute(request)

        self.assertFalse(rust_response.has_errors)
        self.assertEqual(
            rust_response.outputs["out"].payload.to_pylist(),
            [
                {"id": 1, "tags": ["a", "b"], "score": 11.0},
                {"id": 2, "tags": ["c"], "score": 21.0},
            ],
        )
        self.assertEqual(
            rust_response.outputs["out"].payload.schema.field("tags").type,
            pa.list_(pa.string()),
        )
        self.assertEqual(
            rust_response.outputs["out"].payload.schema.field("score").type,
            pa.float64(),
        )

    def test_rust_standard_path_keeps_arrow_input_for_supported_source_dataset_options(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust")
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
        parse_result = ParserService().parse(dsl_text)
        self.assertFalse(parse_result.has_errors)

        prepared_inputs, diagnostics = _prepare_runtime_inputs_for_test(executor,
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
        self.assertEqual(prepared_inputs["in"].preparation.kind, "prepared_set")
        self.assertEqual(
            prepared_inputs["in"].payload.to_pylist(),
            [{"id": 2, "amt": 20}, {"id": 3, "amt": 30}],
        )

    def test_output_handoff_preserves_materialized_source_renames_in_rust_row(self) -> None:
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

        rust_response = DataStepExecutor(runtime_backend="rust").execute(request)

        self.assertFalse(rust_response.has_errors)
        self.assertEqual(
            rust_response.outputs["out"].payload.to_pylist(),
            [
                {"id": 1, "final_amt": 10, "final_score": 11},
                {"id": 2, "final_amt": 20, "final_score": 21},
            ],
        )

    def test_rewrite_preparation_keeps_single_source_arrow_carrier_without_row_materialization(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust")
        dsl_text = (
            "data out; "
            "set in; "
            "prev_id = lag(id); "
            "next_id = lead(id); "
            "keep id prev_id next_id; "
            "run;"
        )

        with patch.object(
            executor._io_service,
            "load_input_rows",
            side_effect=AssertionError("rewrite preparation should not materialize Arrow rows for single-source carrier"),
        ):
            response = executor.execute(
                ExecuteRequest(
                    dsl_text=dsl_text,
                    inputs={
                        "in": DataSetRef(
                            kind="arrow_table",
                            location="dataset://in",
                            payload=pa.Table.from_pylist([
                                {"id": 1},
                                {"id": 2},
                                {"id": 3},
                            ]),
                        )
                    },
                    output_targets=["out"],
                )
            )

        self.assertFalse(response.has_errors)
        self.assertEqual(
            response.outputs["out"].payload.to_pylist(),
            [
                {"id": 1, "prev_id": None, "next_id": 2},
                {"id": 2, "prev_id": 1, "next_id": 3},
                {"id": 3, "prev_id": 2, "next_id": None},
            ],
        )
        prepared_inputs = executor.last_prepared_execution_inputs.inputs
        self.assertEqual(prepared_inputs["in"].kind, "arrow_table")
        self.assertEqual(prepared_inputs["in"].preparation.kind, "prepared_set")
        self.assertEqual(
            prepared_inputs["in"].payload.to_pylist(),
            [
                {"id": 1, "__rewrite_lag_id_1_1__": None, "__rewrite_lead_id_1_2__": 2},
                {"id": 2, "__rewrite_lag_id_1_1__": 1, "__rewrite_lead_id_1_2__": 3},
                {"id": 3, "__rewrite_lag_id_1_1__": 2, "__rewrite_lead_id_1_2__": None},
            ],
        )

    def test_rewrite_preparation_keeps_single_source_arrow_carrier_with_source_options_without_row_materialization(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust")
        dsl_text = (
            "data out; "
            "set in(obs=2); "
            "prev_id = lag(id); "
            "keep id prev_id; "
            "run;"
        )

        with patch.object(
            executor._io_service,
            "load_input_rows",
            side_effect=AssertionError(
                "rewrite preparation should not materialize rows for single-source source-local dataset options"
            ),
        ):
            response = executor.execute(
                ExecuteRequest(
                    dsl_text=dsl_text,
                    inputs={
                        "in": DataSetRef(
                            kind="arrow_table",
                            location="dataset://in",
                            payload=pa.Table.from_pylist([
                                {"id": 1},
                                {"id": 2},
                                {"id": 3},
                            ]),
                        )
                    },
                    output_targets=["out"],
                )
            )

        self.assertFalse(response.has_errors)
        self.assertEqual(
            response.outputs["out"].payload.to_pylist(),
            [
                {"id": 1, "prev_id": None},
                {"id": 2, "prev_id": 1},
            ],
        )
        prepared_inputs = executor.last_prepared_execution_inputs.inputs
        self.assertEqual(prepared_inputs["in"].kind, "arrow_table")
        self.assertEqual(prepared_inputs["in"].preparation.kind, "prepared_set")
        self.assertEqual(
            prepared_inputs["in"].payload.to_pylist(),
            [
                {"id": 1, "__rewrite_lag_id_1_1__": None},
                {"id": 2, "__rewrite_lag_id_1_1__": 1},
            ],
        )

    def test_rewrite_preparation_preserves_by_order_flags_on_prepared_arrow_carrier(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust")
        dsl_text = (
            "data out; "
            "set a b; "
            "by grp; "
            "prev_id = lag(id); "
            "keep grp id prev_id first.grp last.grp; "
            "run;"
        )

        with patch.object(
            executor._io_service,
            "load_input_rows",
            side_effect=AssertionError(
                "rewrite preparation should preserve BY-order Arrow carriers without row materialization"
            ),
        ):
            response = executor.execute(
                ExecuteRequest(
                    dsl_text=dsl_text,
                    inputs={
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
                    },
                    output_targets=["out"],
                )
            )

        self.assertFalse(response.has_errors)
        self.assertEqual(
            response.outputs["out"].payload.to_pylist(),
            [
                {"grp": "A", "id": 1, "prev_id": None},
                {"grp": "A", "id": 3, "prev_id": 1},
                {"grp": "B", "id": 2, "prev_id": 3},
            ],
        )
        prepared_inputs = executor.last_prepared_execution_inputs.inputs
        self.assertEqual(prepared_inputs["a"].kind, "arrow_table")
        self.assertEqual(prepared_inputs["a"].preparation.kind, "prepared_set")
        self.assertEqual(
            prepared_inputs["a"].payload.to_pylist(),
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

    def test_lag_by_multi_output_handoff_keeps_arrow_and_removes_internal_helpers(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust")
        response_request = ExecuteRequest(
            dsl_text=(
                "data left right; "
                "set in; "
                "by grp; "
                "prev = lag(id); "
                "if first.grp then output left; "
                "else output right; "
                "keep grp id prev; "
                "run;"
            ),
            inputs={
                "in": DataSetRef(
                    kind="arrow_table",
                    location="dataset://in",
                    payload=pa.Table.from_pylist([
                        {"grp": "A", "id": 1},
                        {"grp": "A", "id": 2},
                        {"grp": "B", "id": 3},
                    ]),
                )
            },
            output_targets=("left", "right"),
        )

        with patch.object(executor._rust_backend, "execute_typed", wraps=executor._rust_backend.execute_typed) as native_execute:
            response = executor.execute(response_request)

        self.assertFalse(response.has_errors)
        native_context = native_execute.call_args.args[0]
        self.assertTrue(native_context.resolved_inputs)
        for dataset_ref in native_context.resolved_inputs.values():
            self.assertEqual(dataset_ref.kind, "arrow_table")
            self.assertTrue(hasattr(dataset_ref.payload, "__arrow_c_stream__"))

        self.assertEqual(response.outputs["left"].kind, "arrow_table")
        self.assertEqual(response.outputs["right"].kind, "arrow_table")
        self.assertEqual(response.outputs["left"].payload.schema.names, ["grp", "id", "prev"])
        self.assertEqual(response.outputs["right"].payload.schema.names, ["grp", "id", "prev"])
        for output in response.outputs.values():
            for helper_name in ("FIRST.grp", "LAST.grp", "first.grp", "last.grp", "__rewrite_lag_id_1_1__"):
                self.assertNotIn(helper_name, output.payload.schema.names)
        self.assertEqual(
            response.outputs["left"].payload.to_pylist(),
            [{"grp": "A", "id": 1, "prev": None}, {"grp": "B", "id": 3, "prev": 2.0}],
        )
        self.assertEqual(
            response.outputs["right"].payload.to_pylist(),
            [{"grp": "A", "id": 2, "prev": 1.0}],
        )

    def test_lag_by_multi_output_handoff_preserves_empty_arrow_target_schema(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust")
        response = executor.execute(
            ExecuteRequest(
                dsl_text=(
                    "data left right; "
                    "set in; "
                    "by grp; "
                    "prev = lag(id); "
                    "if first.grp then output left; "
                    "keep grp id prev; "
                    "run;"
                ),
                inputs={
                    "in": DataSetRef(
                        kind="arrow_table",
                        location="dataset://in",
                        payload=pa.Table.from_pylist([
                            {"grp": "A", "id": 1},
                            {"grp": "A", "id": 2},
                            {"grp": "B", "id": 3},
                        ]),
                    )
                },
                output_targets=("left", "right"),
            )
        )

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["right"].kind, "arrow_table")
        self.assertEqual(response.outputs["right"].payload.schema.names, ["grp", "id", "prev"])
        self.assertEqual(response.outputs["right"].payload.to_pylist(), [])

    def test_rewrite_preparation_keeps_prepared_multi_source_arrow_carrier_without_row_materialization(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust")
        dsl_text = (
            "data out; "
            "set a(obs=1) b indsname=src end=last; "
            "prev_id = lag(id); "
            "keep id prev_id; "
            "run;"
        )

        with patch.object(
            executor._io_service,
            "load_input_rows",
            side_effect=AssertionError(
                "rewrite preparation should not materialize rows after source preparation has produced a prepared Arrow carrier"
            ),
        ):
            response = executor.execute(
                ExecuteRequest(
                    dsl_text=dsl_text,
                    inputs={
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
                    },
                    output_targets=["out"],
                )
            )

        self.assertFalse(response.has_errors)
        self.assertEqual(
            response.outputs["out"].payload.to_pylist(),
            [
                {"id": 1, "prev_id": None},
                {"id": 3, "prev_id": 1},
                {"id": 4, "prev_id": 3},
            ],
        )
        prepared_inputs = executor.last_prepared_execution_inputs.inputs
        self.assertEqual(prepared_inputs["a"].kind, "arrow_table")
        self.assertEqual(prepared_inputs["a"].preparation.kind, "prepared_set")
        self.assertEqual(
            prepared_inputs["a"].payload.to_pylist(),
            [
                {"id": 1, "src": "a", "last": 0, "__rewrite_lag_id_1_1__": None},
                {"id": 3, "src": "b", "last": 0, "__rewrite_lag_id_1_1__": 1},
                {"id": 4, "src": "b", "last": 1, "__rewrite_lag_id_1_1__": 3},
            ],
        )

    def test_rust_standard_path_prepares_arrow_input_for_set_helper_columns(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust")
        dsl_text = (
            "data out; "
            "set a(in=in_a) b(in=in_b) indsname=src end=last; "
            "if last then output out; "
            "keep id src last; "
            "run;"
        )
        parse_result = ParserService().parse(dsl_text)
        self.assertFalse(parse_result.has_errors)

        prepared_inputs, diagnostics = _prepare_runtime_inputs_for_test(executor,
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
            self.assertEqual(dataset_ref.preparation.kind, "prepared_set")
            self.assertTrue(
                {"in_a", "in_b", "src", "last"}.issubset(
                    set(dataset_ref.payload.column_names)
                )
            )

    def test_auto_standard_path_prepares_arrow_input_for_set_helper_columns(self) -> None:
        executor = DataStepExecutor(runtime_backend="auto")
        dsl_text = (
            "data out; "
            "set a(in=in_a) b(in=in_b) indsname=src end=last; "
            "if last then output out; "
            "keep id src last; "
            "run;"
        )
        parse_result = ParserService().parse(dsl_text)
        self.assertFalse(parse_result.has_errors)

        prepared_inputs, diagnostics = _prepare_runtime_inputs_for_test(executor,
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
            self.assertEqual(dataset_ref.preparation.kind, "prepared_set")
            self.assertTrue(
                {"in_a", "in_b", "src", "last"}.issubset(
                    set(dataset_ref.payload.column_names)
                )
            )

    def test_rust_end_option_marks_only_final_concatenated_set_row(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust")
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
        executor = DataStepExecutor(runtime_backend="auto")
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

    def test_rust_standard_path_prepares_arrow_input_for_by_helper_columns(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust")
        dsl_text = (
            "data out; "
            "set in end=eof; "
            "by grp; "
            "if last.grp then output out; "
            "keep grp amount eof; "
            "run;"
        )
        parse_result = ParserService().parse(dsl_text)
        self.assertFalse(parse_result.has_errors)

        prepared_inputs, diagnostics = _prepare_runtime_inputs_for_test(executor,
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
        self.assertEqual(prepared_inputs["in"].preparation.kind, "prepared_set")
        self.assertTrue(
            {"eof", "FIRST.grp", "LAST.grp", "first.grp", "last.grp"}.issubset(
                set(prepared_inputs["in"].payload.column_names)
            )
        )

    def test_auto_standard_path_prepares_arrow_input_for_by_helper_columns(self) -> None:
        executor = DataStepExecutor(runtime_backend="auto")
        dsl_text = (
            "data out; "
            "set in end=eof; "
            "by grp; "
            "if last.grp then output out; "
            "keep grp amount eof; "
            "run;"
        )
        parse_result = ParserService().parse(dsl_text)
        self.assertFalse(parse_result.has_errors)

        prepared_inputs, diagnostics = _prepare_runtime_inputs_for_test(executor,
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
        self.assertEqual(prepared_inputs["in"].preparation.kind, "prepared_set")
        self.assertTrue(
            {"eof", "FIRST.grp", "LAST.grp", "first.grp", "last.grp"}.issubset(
                set(prepared_inputs["in"].payload.column_names)
            )
        )

    def test_rust_by_helper_columns_preserve_n_counter_on_last_group_output(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust")
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
        executor = DataStepExecutor(runtime_backend="auto")
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
