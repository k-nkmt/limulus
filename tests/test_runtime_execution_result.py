from __future__ import annotations

from dataclasses import fields
import inspect
import json

import pyarrow as pa
import pytest

from limulus.backend_integration.contracts import (
    RuntimeExecutionResult,
    TypedRustExecutionPayload,
)
from limulus.backend_integration.rust_executor import RustNativeBlockExecutor
from limulus.backend_integration.rust_runtime import RustRuntimeBackend
from limulus.execution import DataStepExecutor
from limulus.execution.contracts import (
    ExecutionObservations,
    PipelineExecutionRequest,
    PipelineExecutionResult,
)
from limulus.models import (
    DataSetRef,
    ExecuteRequest,
    ExecuteResponse,
    PreparedExecutionInputs,
    SourceMetadataLineage,
    SourceMetadataSnapshot,
)


def test_parser_state_is_absent_from_advanced_execution_contracts() -> None:
    assert "parser_backend_preference" not in {
        field.name for field in fields(PipelineExecutionRequest)
    }
    assert "parser_backend" not in {
        field.name for field in fields(ExecutionObservations)
    }
    assert "parser_backend" not in inspect.signature(DataStepExecutor).parameters
    assert not hasattr(DataStepExecutor, "set_parser_backend")
    assert not hasattr(DataStepExecutor, "last_parser_backend")

    executor = DataStepExecutor(runtime_backend="rust")
    assert not hasattr(executor, "_parser_backend_preference")
    assert not hasattr(executor, "_last_parser_backend")


def test_runtime_execution_result_is_typed_and_immutable() -> None:
    result = RuntimeExecutionResult(
        outputs={"out": DataSetRef(kind="arrow_table", location="dataset://out")},
        diagnostics=(),
        actual_engine="rust",
        native_invoked=True,
        phase_metrics={"row_execution_ms": 1.25},
    )

    assert result.outputs["out"].location == "dataset://out"
    assert result.diagnostics == ()
    assert result.actual_engine == "rust"
    assert result.native_invoked is True
    assert result.phase_metrics == {"row_execution_ms": 1.25}


def test_runtime_execution_result_rejects_non_rust_actual_engine() -> None:
    with pytest.raises(ValueError, match="actual_engine must be 'rust'"):
        RuntimeExecutionResult(
            outputs={},
            diagnostics=(),
            actual_engine="python",  # type: ignore[arg-type]
            native_invoked=False,
            phase_metrics={},
        )


def test_pipeline_execution_request_snapshots_catalog_mappings() -> None:
    registered = {"registered": DataSetRef(kind="memory", location="dataset://registered")}
    generated = {"generated": DataSetRef(kind="memory", location="dataset://generated")}

    request = PipelineExecutionRequest(
        request=ExecuteRequest(dsl_text="data out; run;"),
        registered_tables=registered,
        generated_tables=generated,
        runtime_backend_preference="rust",
    )

    registered["mutated"] = DataSetRef(kind="memory", location="dataset://mutated")
    generated.clear()

    assert tuple(request.registered_tables) == ("registered",)
    assert tuple(request.generated_tables) == ("generated",)
    with pytest.raises(TypeError):
        request.registered_tables["new"] = DataSetRef(  # type: ignore[index]
            kind="memory", location="dataset://new"
        )
    with pytest.raises(TypeError):
        request.generated_tables["new"] = DataSetRef(  # type: ignore[index]
            kind="memory", location="dataset://new"
        )


def test_pipeline_request_freezes_nested_inputs_metadata_and_options_without_payload_copy() -> None:
    payload = pa.table({"id": [1]})
    input_metadata = {"lineage": {"source": "before"}}
    options = {"dispatch_hints": {"backend": {"name": "rust"}}}
    input_ref = DataSetRef(
        kind="arrow_table",
        location="dataset://input",
        payload=payload,
        metadata=input_metadata,
    )
    request = PipelineExecutionRequest(
        request=ExecuteRequest(
            dsl_text="data out; set input; run;",
            inputs={"input": input_ref},
            options=options,
        ),
        registered_tables={"input": input_ref},
        generated_tables={},
        runtime_backend_preference="rust",
    )

    input_metadata["lineage"]["source"] = "after"
    options["dispatch_hints"]["backend"]["name"] = "python"

    snap_ref = request.request.inputs["input"]
    assert snap_ref.payload is payload
    assert snap_ref.metadata["lineage"]["source"] == "before"
    assert request.request.options["dispatch_hints"]["backend"]["name"] == "rust"
    assert request.registered_tables["input"].metadata["lineage"]["source"] == "before"
    with pytest.raises(TypeError):
        snap_ref.metadata["lineage"]["source"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        request.request.options["dispatch_hints"]["backend"]["name"] = "changed"  # type: ignore[index]


def test_pipeline_result_freezes_prepared_and_catalog_metadata_without_payload_copy() -> None:
    prepared_payload = pa.table({"id": [1]})
    output_payload = pa.table({"id": [1]})
    prepared_metadata = {"lineage": {"source": "prepared"}}
    output_metadata = {"lineage": {"source": "output"}}
    prepared_ref = DataSetRef(
        kind="arrow_table",
        location="dataset://prepared",
        payload=prepared_payload,
        metadata=prepared_metadata,
    )
    output_ref = DataSetRef(
        kind="arrow_table",
        location="dataset://output",
        payload=output_payload,
        metadata=output_metadata,
    )
    result = PipelineExecutionResult(
        response=ExecuteResponse(),
        catalog_updates={"output": output_ref},
        observations=ExecutionObservations(
            prepared_inputs=PreparedExecutionInputs(inputs={"prepared": prepared_ref})
        ),
    )

    prepared_metadata["lineage"]["source"] = "mutated"
    output_metadata["lineage"]["source"] = "mutated"

    result_prepared = result.observations.prepared_inputs.inputs["prepared"]
    result_output = result.catalog_updates["output"]
    assert result_prepared.payload is prepared_payload
    assert result_output.payload is output_payload
    assert result_prepared.metadata["lineage"]["source"] == "prepared"
    assert result_output.metadata["lineage"]["source"] == "output"
    with pytest.raises(TypeError):
        result_prepared.metadata["lineage"]["source"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        result_output.metadata["lineage"]["source"] = "changed"  # type: ignore[index]


def test_pipeline_observations_snapshot_nested_lineage_metadata_and_preserve_order_identity() -> None:
    first_schema = pa.schema([pa.field("id", pa.int64())])
    second_schema = pa.schema([pa.field("name", pa.string())])
    lineage_metadata = {
        "first": {"details": {"source": "before"}},
        "second": {"details": {"source": "second"}},
    }
    lineage = SourceMetadataLineage(
        ordered_sources=(
            SourceMetadataSnapshot(
                source_ordinal=4,
                source_name="first",
                schema=first_schema,
                dataset_ref_metadata=lineage_metadata["first"],
            ),
            SourceMetadataSnapshot(
                source_ordinal=9,
                source_name="second",
                schema=second_schema,
                dataset_ref_metadata=lineage_metadata["second"],
            ),
        )
    )
    payload = pa.table({"id": [1]})
    prepared = PreparedExecutionInputs(
        inputs={
            "first": DataSetRef(
                kind="arrow_table",
                location="dataset://first",
                payload=payload,
            )
        },
        source_metadata_lineage=lineage,
    )
    observations = ExecutionObservations(
        prepared_inputs=prepared,
        output_source_metadata_lineage=lineage,
    )

    lineage_metadata["first"]["details"]["source"] = "mutated"

    prepared_lineage = observations.prepared_inputs.source_metadata_lineage
    output_lineage = observations.output_source_metadata_lineage
    assert [source.source_name for source in prepared_lineage.ordered_sources] == [
        "first",
        "second",
    ]
    assert [source.source_ordinal for source in output_lineage.ordered_sources] == [4, 9]
    assert prepared_lineage.ordered_sources[0].schema is first_schema
    assert output_lineage.ordered_sources[1].schema is second_schema
    assert observations.prepared_inputs.inputs["first"].payload is payload
    assert (
        prepared_lineage.ordered_sources[0].dataset_ref_metadata["details"]["source"]
        == "before"
    )
    assert (
        output_lineage.ordered_sources[0].dataset_ref_metadata["details"]["source"]
        == "before"
    )
    with pytest.raises(TypeError):
        prepared_lineage.ordered_sources[0].dataset_ref_metadata["details"]["source"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        output_lineage.ordered_sources[0].dataset_ref_metadata["details"]["source"] = "changed"  # type: ignore[index]


def test_pipeline_execution_observations_and_result_freeze_mappings() -> None:
    phase_metrics = {"row_execution_ms": 1.25}
    observations = ExecutionObservations(phase_metrics=phase_metrics)
    catalog_updates = {
        "out": DataSetRef(kind="memory", location="dataset://out"),
    }
    result = PipelineExecutionResult(
        response=ExecuteResponse(),
        catalog_updates=catalog_updates,
        observations=observations,
    )

    phase_metrics["source_access_ms"] = 2.0
    catalog_updates["later"] = DataSetRef(kind="memory", location="dataset://later")

    assert result.observations.phase_metrics == {"row_execution_ms": 1.25}
    assert tuple(result.catalog_updates) == ("out",)
    with pytest.raises(TypeError):
        result.observations.phase_metrics["new"] = 1.0  # type: ignore[index]
    with pytest.raises(TypeError):
        result.catalog_updates["new"] = DataSetRef(  # type: ignore[index]
            kind="memory", location="dataset://new"
        )


def test_pipeline_returns_contract_with_successful_prefix_updates_on_failure() -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    pipeline_request = PipelineExecutionRequest(
        request=ExecuteRequest(
            dsl_text=(
                "data stage1; set inp; output stage1; run; "
                "data stage2; set missing_in; output stage2; run;"
            ),
            inputs={
                "inp": DataSetRef(
                    kind="arrow_table",
                    location="dataset://inp",
                    payload=pa.table({"id": [1]}),
                )
            },
        ),
        registered_tables={},
        generated_tables={},
        runtime_backend_preference="rust",
    )

    response = executor.execute(pipeline_request.request)

    assert response.has_errors is True
    assert set(executor._generated_tables) == {"stage1"}
    assert not hasattr(response, "parser_backend")
    assert executor.last_strategy_decision is not None
    assert executor.last_runtime_backend == "rust"


def test_pipeline_early_failure_returns_default_observations() -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    response = executor.execute(ExecuteRequest(dsl_text=""))

    assert response.has_errors is True
    assert not hasattr(response, "parser_backend")
    assert executor.last_strategy_decision is None
    assert executor.last_runtime_backend is None
    assert executor.last_phase_metrics == {}
    assert executor.last_prepared_execution_inputs.inputs == {}
    assert executor.last_output_source_metadata_lineage.ordered_sources == ()


def test_rust_native_executor_has_no_observation_side_channel_fields() -> None:
    executor = RustNativeBlockExecutor()

    assert not hasattr(executor, "last_phase_metrics")
    assert not hasattr(executor, "last_source_owner_metadata")
    assert not hasattr(executor, "last_native_invoked")


def _typed_native_payload(*, output_targets: tuple[str, ...] = ("out",)) -> TypedRustExecutionPayload:
    native_program = {"version": 1, "output_targets": list(output_targets)}
    return TypedRustExecutionPayload(
        native_program=native_program,
        native_program_json=json.dumps(native_program),
        input_streams={"inp": object()},
    )


def test_rust_runtime_backend_accepts_typed_adapter_result_without_side_channels() -> None:
    payload = _typed_native_payload()
    output = DataSetRef(
        kind="arrow_table",
        location="dataset://out",
        payload=pa.table({"id": [1]}),
    )

    class StubBridge:
        def build_payload(self, _context):
            return payload, []

    class StubExecutor:
        def is_available(self):
            return True

        def execute_block(self, _payload):
            return RuntimeExecutionResult(
                outputs={"out": output},
                diagnostics=(),
                actual_engine="rust",
                native_invoked=True,
                phase_metrics={"source_access_ms": 0.5, "row_execution_ms": 1.25},
            )

    result = RustRuntimeBackend(
        executor=StubExecutor(), typed_bridge=StubBridge()
    ).execute_typed(object())

    assert isinstance(result, RuntimeExecutionResult)
    assert result.outputs == {"out": output}
    assert result.diagnostics == ()
    assert result.actual_engine == "rust"
    assert result.native_invoked is True
    assert result.phase_metrics == {"source_access_ms": 0.5, "row_execution_ms": 1.25}


def test_rust_runtime_backend_capability_failure_is_typed_without_native_invocation() -> None:
    result = RustRuntimeBackend().execute_typed(object())

    assert isinstance(result, RuntimeExecutionResult)
    assert result.outputs == {}
    assert result.actual_engine == "rust"
    assert result.native_invoked is False
    assert result.phase_metrics == {}
    assert result.diagnostics[0].code == "RUNTIME_BACKEND_CAPABILITY_MISSING"


@pytest.mark.parametrize(
    "native_result",
    [
        {"diagnostics": []},
        {"diagnostics": [], "output_streams": ["not-a-mapping"]},
    ],
    ids=("missing-output-streams", "non-mapping-output-streams"),
)
def test_native_output_streams_are_required_and_mapping(
    native_result: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeNativeModule:
        @staticmethod
        def execute_block(_payload: dict[str, object]) -> dict[str, object]:
            return native_result

    monkeypatch.setattr(
        "limulus.backend_integration.rust_executor.load_native_module",
        lambda: (FakeNativeModule(), None),
    )

    result = RustNativeBlockExecutor().execute_block(_typed_native_payload())

    assert result.outputs == {}
    assert result.native_invoked is True
    assert result.diagnostics[0].code == "RUNTIME_RUST_NATIVE_EXECUTION_FAILED"


def test_native_missing_expected_output_target_is_failure_without_partial_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeNativeModule:
        @staticmethod
        def execute_block(_payload: dict[str, object]) -> dict[str, object]:
            return {"diagnostics": [], "output_streams": {"other": object()}}

    monkeypatch.setattr(
        "limulus.backend_integration.rust_executor.load_native_module",
        lambda: (FakeNativeModule(), None),
    )

    result = RustNativeBlockExecutor().execute_block(_typed_native_payload())

    assert result.outputs == {}
    assert result.native_invoked is True
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "RUNTIME_RUST_NATIVE_EXECUTION_FAILED"
    assert diagnostic.location == "dataset:out"
    assert "out" in diagnostic.message


def test_native_unexpected_output_target_is_failure_without_partial_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeNativeModule:
        @staticmethod
        def execute_block(_payload: dict[str, object]) -> dict[str, object]:
            return {
                "diagnostics": [],
                "output_streams": {"out": object(), "extra": object()},
            }

    monkeypatch.setattr(
        "limulus.backend_integration.rust_executor.load_native_module",
        lambda: (FakeNativeModule(), None),
    )

    result = RustNativeBlockExecutor().execute_block(_typed_native_payload())

    assert result.outputs == {}
    assert result.native_invoked is True
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "RUNTIME_RUST_NATIVE_EXECUTION_FAILED"
    assert diagnostic.location == "dataset:EXTRA"
    assert "unexpected" in diagnostic.message
