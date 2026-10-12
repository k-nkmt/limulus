from __future__ import annotations

import inspect

import pyarrow as pa

from limulus.execution import DataStepExecutor
from limulus.execution import output_handoff as output_handoff_module
from limulus.execution.contracts import (
    TypedOutputFinalizeRequest,
)
from limulus.execution.output_handoff import (
    OutputHandoffResult,
)
from limulus.models import (
    DataSetRef,
    Diagnostic,
    ExecuteRequest,
)


def test_output_service_keeps_host_apply_boundary_out_of_finalization() -> None:
    source = inspect.getsource(output_handoff_module)

    assert "apply_registry" not in source
    assert "_execute_program" not in source


def test_pipeline_does_not_commit_catalog_when_finalization_fails(monkeypatch) -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    source = DataSetRef(
        kind="arrow_table",
        location="dataset://source",
        payload=pa.table({"id": [1]}),
    )

    def fail_finalize(request: TypedOutputFinalizeRequest) -> OutputHandoffResult:
        return OutputHandoffResult(
            output_source_metadata_lineage=request.source_metadata_lineage,
            diagnostics=(
                Diagnostic(
                    code="CONVERT_OUTPUT_FAILED",
                    severity="error",
                    location="dataset:out",
                    message="metadata failure",
                ),
            ),
        )

    monkeypatch.setattr(executor._output_handoff_service, "finalize_typed", fail_finalize)
    response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set source; run;",
            inputs={"source": source},
        )
    )

    assert response.has_errors is True
    assert response.outputs == {}
    assert "out" not in executor._generated_tables


def test_pipeline_reuses_each_block_lazy_mapping_without_rebuilding(monkeypatch) -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    source = DataSetRef(
        kind="arrow_table",
        location="dataset://source",
        payload=pa.table({"id": [1]}),
    )
    build_calls: list[tuple[str, ...]] = []
    original_builder = executor._output_handoff_service.build_lazy_outputs

    def count_builds(outputs):
        build_calls.append(tuple(outputs))
        return original_builder(outputs)

    monkeypatch.setattr(
        executor._output_handoff_service,
        "build_lazy_outputs",
        count_builds,
    )
    conversion_calls = 0
    original_converter = executor._io_service.dataset_ref_to_arrow_table

    def count_conversions(dataset_name, dataset_ref):
        nonlocal conversion_calls
        conversion_calls += 1
        return original_converter(dataset_name, dataset_ref)

    monkeypatch.setattr(
        executor._io_service,
        "dataset_ref_to_arrow_table",
        count_conversions,
    )

    response = executor.execute(
        ExecuteRequest(
            dsl_text=(
                "data left; set source; run; "
                "data right; set source; run;"
            ),
            inputs={"source": source},
        )
    )

    assert response.has_errors is False
    assert len(build_calls) == 2
    assert conversion_calls == 0
    assert set(response.outputs_arrow) == {"left", "right"}
    assert response.outputs_arrow["left"].to_pylist() == [{"id": 1}]
    assert conversion_calls == 1
    assert response.outputs_arrow["left"] is response.outputs_arrow["left"]
    assert conversion_calls == 1
    assert response.outputs_arrow["right"].to_pylist() == [{"id": 1}]
    assert conversion_calls == 2


def test_pipeline_aggregates_distinct_and_same_target_public_lazy_outputs() -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    source = DataSetRef(
        kind="arrow_table",
        location="dataset://source",
        payload=pa.table({"id": [1]}),
    )
    response = executor.execute(
        ExecuteRequest(
            dsl_text=(
                "data left(label='Left'); set source; run; "
                "data out(label='First'); set source; run; "
                "data out(label='Second'); set source; run;"
            ),
            inputs={"source": source},
        )
    )

    assert response.has_errors is False
    assert set(response.outputs) == {"left", "out"}
    assert response.outputs["left"].metadata["memlabel"] == "Left"
    assert response.outputs["out"].metadata["memlabel"] == "Second"
    assert set(response.outputs_arrow) == {"left", "out"}
    assert response.outputs_arrow["left"].schema.metadata[b"memlabel"] == b"Left"
    assert response.outputs_arrow["out"].schema.metadata[b"memlabel"] == b"Second"


def test_second_block_finalization_failure_preserves_first_catalog_update(monkeypatch) -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    source = DataSetRef(
        kind="arrow_table",
        location="dataset://source",
        payload=pa.table({"id": [1]}),
    )
    original_finalize = executor._output_handoff_service.finalize_typed
    finalize_calls = 0

    def fail_second(request: TypedOutputFinalizeRequest) -> OutputHandoffResult:
        nonlocal finalize_calls
        finalize_calls += 1
        if finalize_calls == 2:
            return OutputHandoffResult(
                output_source_metadata_lineage=request.source_metadata_lineage,
                diagnostics=(
                    Diagnostic(
                        code="CONVERT_OUTPUT_FAILED",
                        severity="error",
                        location="dataset:second",
                        message="metadata failure",
                    ),
                ),
            )
        return original_finalize(request)

    monkeypatch.setattr(executor._output_handoff_service, "finalize_typed", fail_second)
    response = executor.execute(
        ExecuteRequest(
            dsl_text=(
                "data first; set source; run; "
                "data second; set source; run;"
            ),
            inputs={"source": source},
        )
    )

    assert response.has_errors is True
    assert finalize_calls == 2
    assert "first" in executor._generated_tables
    assert "second" not in executor._generated_tables
