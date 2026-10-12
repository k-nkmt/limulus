from __future__ import annotations

import inspect
from unittest.mock import patch

import pyarrow as pa

from limulus.execution import DataStepExecutor
from limulus.execution import coordinator as coordinator_module
from limulus.execution import input_preparation as input_preparation_module
from limulus.execution.contracts import (
    InputPreparationRequest,
    InputResolutionResult,
)
from limulus.execution.rewrites import RewritePlan
from limulus.execution.rewrites import RewritePlanner
from limulus.models import DataSetRef, NormalizedExecutionInputs, SourceMetadataLineage
from limulus.models import ExecuteRequest
from limulus.parser import DatasetReferenceOptionSpec
from limulus.parser import ParserService


def _parse(executor: DataStepExecutor, dsl: str):
    result = ParserService().parse(dsl)
    assert not result.has_errors
    return result.ast.statements


def _plan_for(executor: DataStepExecutor, statements, inputs):
    del executor, statements, inputs
    return RewritePlan(
        merge_mode="none",
        lag_lead_mode="none",
        where_mode="none",
        helper_source_columns=(),
        helper_runtime_columns=(),
    )


def _prepare_merge_for_test(executor, statements, inputs, rewrite_plan):
    result = executor._input_preparation_service.prepare(
        InputPreparationRequest(
            ast_statements=tuple(statements),
            input_resolution=InputResolutionResult(
                resolved_inputs=NormalizedExecutionInputs(inputs=inputs),
            ),
            rewrite_plan=rewrite_plan,
        )
    )
    return (
        result.prepared_statements,
        result.prepared.inputs,
        result.diagnostics[0] if result.diagnostics else None,
    )


def _prepare_inputs_for_test(
    executor,
    statements,
    inputs,
    rewrite_plan,
    *,
    source_metadata_lineage=None,
):
    result = executor._input_preparation_service.prepare(
        InputPreparationRequest(
            ast_statements=tuple(statements),
            input_resolution=InputResolutionResult(
                resolved_inputs=NormalizedExecutionInputs(inputs=inputs),
                source_metadata_lineage=source_metadata_lineage
                if source_metadata_lineage is not None
                else SourceMetadataLineage(),
            ),
            rewrite_plan=rewrite_plan,
        )
    )
    return dict(result.prepared.inputs), list(result.diagnostics)


def test_dataset_option_orchestration_has_no_row_option_callback() -> None:
    coordinator_source = inspect.getsource(coordinator_module.DataStepExecutor)
    rewrite_source = inspect.getsource(RewritePlanner)

    assert "_apply_dataset_reference_options_for_prepare" not in coordinator_source
    assert "apply_dataset_reference_options_to_rows" not in coordinator_source
    assert "apply_dataset_reference_options" not in rewrite_source


def test_set_where_uses_shared_arrow_preparation_once() -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    statements = _parse(executor, "data out; set source(where=(id >= 2)); run;")
    source = DataSetRef(
        kind="arrow_table",
        location="dataset://source",
        payload=pa.table({"id": [1, 2, 3]}),
    )
    rewrite_plan = _plan_for(executor, statements, {"source": source})
    source_metadata_lineage, lineage_error = (
        input_preparation_module.capture_source_metadata_lineage(
            source_refs=statements[1].dataset_refs,
            resolved_inputs={"source": source},
        )
    )
    assert lineage_error is None

    with patch.object(
        input_preparation_module,
        "try_apply_dataset_reference_options_to_arrow_table",
        wraps=input_preparation_module.try_apply_dataset_reference_options_to_arrow_table,
    ) as shared_prepare:
        prepared, diagnostics = _prepare_inputs_for_test(
            executor,
            statements,
            {"source": source},
            rewrite_plan,
            source_metadata_lineage=source_metadata_lineage,
        )

    assert diagnostics == []
    assert shared_prepare.call_count == 1
    assert prepared["source"].payload.to_pylist() == [{"id": 2}, {"id": 3}]


def test_preparation_without_resolution_lineage_applies_source_option_once() -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    statements = _parse(executor, "data out; set source(where=(id >= 2)); run;")
    source = DataSetRef(
        kind="arrow_table",
        location="dataset://source",
        payload=pa.table({"id": [1, 2, 3]}),
    )
    rewrite_plan = _plan_for(executor, statements, {"source": source})

    with patch.object(
        input_preparation_module,
        "try_apply_dataset_reference_options_to_arrow_table",
        wraps=input_preparation_module.try_apply_dataset_reference_options_to_arrow_table,
    ) as shared_prepare:
        result = executor._input_preparation_service.prepare(
            InputPreparationRequest(
                ast_statements=tuple(statements),
                input_resolution=InputResolutionResult(
                    resolved_inputs=NormalizedExecutionInputs(inputs={"source": source}),
                ),
                rewrite_plan=rewrite_plan,
            )
        )

    assert result.diagnostics == ()
    assert shared_prepare.call_count == 1
    assert result.prepared.inputs["source"].payload.to_pylist() == [
        {"id": 2},
        {"id": 3},
    ]
    prepared_ref = result.prepared_statements[0].dataset_refs[0]
    assert prepared_ref.options == DatasetReferenceOptionSpec()


def test_set_where_with_internal_flag_uses_shared_arrow_preparation_once() -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    statements = _parse(executor, "data out; set source(where=(id >= 2) in=source_seen); run;")
    source = DataSetRef(
        kind="arrow_table",
        location="dataset://source",
        payload=pa.table({"id": [1, 2, 3]}),
    )
    rewrite_plan = _plan_for(executor, statements, {"source": source})
    source_metadata_lineage, lineage_error = (
        input_preparation_module.capture_source_metadata_lineage(
            source_refs=statements[1].dataset_refs,
            resolved_inputs={"source": source},
        )
    )
    assert lineage_error is None

    with patch.object(
        input_preparation_module,
        "try_apply_dataset_reference_options_to_arrow_table",
        wraps=input_preparation_module.try_apply_dataset_reference_options_to_arrow_table,
    ) as shared_prepare:
        prepared, diagnostics = _prepare_inputs_for_test(
            executor,
            statements,
            {"source": source},
            rewrite_plan,
            source_metadata_lineage=source_metadata_lineage,
        )

    assert diagnostics == []
    assert shared_prepare.call_count == 1
    assert prepared["source"].payload.to_pylist() == [
        {"id": 2, "source_seen": 1},
        {"id": 3, "source_seen": 1},
    ]


def test_set_where_non_arrow_fails_before_row_materialization() -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    statements = _parse(executor, "data out; set source(where=(id >= 2)); run;")
    source = DataSetRef(kind="memory", location="dataset://source", payload=[{"id": 1}])
    rewrite_plan = _plan_for(executor, statements, {"source": source})

    with patch.object(
        executor._io_service,
        "load_input_rows",
        side_effect=AssertionError("WHERE must not reach row materialization"),
    ) as load_rows:
        _, diagnostics = _prepare_inputs_for_test(
            executor,
            statements,
            {"source": source},
            rewrite_plan,
        )

    assert load_rows.call_count == 0
    assert len(diagnostics) == 1
    assert diagnostics[0].code == "RUNTIME_DATASET_OPTION_INVALID"
    assert "WHERE" in diagnostics[0].message


def test_merge_where_non_arrow_fails_before_row_materialization() -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    statements = _parse(executor, "data out; merge left(where=(id >= 2)) right; by id; run;")
    inputs = {
        "left": DataSetRef(kind="memory", location="dataset://left", payload=[{"id": 1}]),
        "right": DataSetRef(kind="memory", location="dataset://right", payload=[{"id": 1}]),
    }
    rewrite_plan = _plan_for(executor, statements, inputs)

    with patch.object(
        executor._io_service,
        "load_input_rows",
        side_effect=AssertionError("WHERE must not reach row materialization"),
    ) as load_rows:
        _, _, diagnostic = _prepare_merge_for_test(
            executor, statements, inputs, rewrite_plan
        )

    assert load_rows.call_count == 0
    assert diagnostic is not None
    assert diagnostic.code == "RUNTIME_DATASET_OPTION_INVALID"
    assert "WHERE" in diagnostic.message


def test_merge_where_uses_shared_arrow_preparation_for_each_source() -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    statements = _parse(
        executor,
        "data out; merge left(where=(id >= 2)) right(where=(id >= 2)); by id; run;",
    )
    inputs = {
        "left": DataSetRef(
            kind="arrow_table",
            location="dataset://left",
            payload=pa.table({"id": [1, 2], "left_value": [10, 20]}),
        ),
        "right": DataSetRef(
            kind="arrow_table",
            location="dataset://right",
            payload=pa.table({"id": [2, 3], "right_value": [200, 300]}),
        ),
    }
    rewrite_plan = _plan_for(executor, statements, inputs)

    with patch.object(
        input_preparation_module,
        "try_apply_dataset_reference_options_to_arrow_table",
        wraps=input_preparation_module.try_apply_dataset_reference_options_to_arrow_table,
    ) as shared_prepare:
        _, prepared, diagnostic = _prepare_merge_for_test(
            executor, statements, inputs, rewrite_plan
        )

    assert diagnostic is None
    # Lineage projection is schema-only; the input option is consumed exactly
    # once per source while building the prepared carrier.
    assert shared_prepare.call_count == 2
    assert prepared["left"].payload.to_pylist() == [
        {
            "id": 2,
            "left_value": 20,
            "right_value": 200,
            "FIRST.id": 1,
            "LAST.id": 1,
            "first.id": 1,
            "last.id": 1,
        },
        {
            "id": 3,
            "left_value": None,
            "right_value": 300,
            "FIRST.id": 1,
            "LAST.id": 1,
            "first.id": 1,
            "last.id": 1,
        },
    ]


def test_set_where_evaluation_failure_returns_shared_diagnostic_before_native() -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    request = ExecuteRequest(
        dsl_text="data out; set source(where=(name > 10)); output out; run;",
        inputs={
            "source": DataSetRef(
                kind="arrow_table",
                location="dataset://source",
                payload=pa.table({"id": [1, 2], "name": ["a", "b"]}),
            )
        },
        output_targets=("out",),
    )

    with patch.object(executor._rust_backend, "execute_typed", wraps=executor._rust_backend.execute_typed) as native_execute:
        response = executor.execute(request)

    native_execute.assert_not_called()
    assert response.has_errors
    assert len(response.diagnostics) == 1
    diagnostic = response.diagnostics[0]
    assert diagnostic.code == "RUNTIME_DATASET_OPTION_INVALID"
    assert "WHERE" in diagnostic.message
    assert "source" in diagnostic.message


def test_merge_where_evaluation_failure_returns_shared_diagnostic_before_native() -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    request = ExecuteRequest(
        dsl_text="data out; merge left(where=(name > 10)) right; by id; output out; run;",
        inputs={
            "left": DataSetRef(
                kind="arrow_table",
                location="dataset://left",
                payload=pa.table({"id": [1, 2], "name": ["a", "b"]}),
            ),
            "right": DataSetRef(
                kind="arrow_table",
                location="dataset://right",
                payload=pa.table({"id": [1, 2], "value": [3, 4]}),
            ),
        },
        output_targets=("out",),
    )

    with patch.object(executor._rust_backend, "execute_typed", wraps=executor._rust_backend.execute_typed) as native_execute:
        response = executor.execute(request)

    native_execute.assert_not_called()
    assert response.has_errors
    assert len(response.diagnostics) == 1
    diagnostic = response.diagnostics[0]
    assert diagnostic.code == "RUNTIME_DATASET_OPTION_INVALID"
    assert "WHERE" in diagnostic.message
    assert "left" in diagnostic.message
