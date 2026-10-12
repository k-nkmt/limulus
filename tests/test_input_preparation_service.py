from __future__ import annotations

import inspect
from unittest.mock import patch

import pyarrow as pa
import pytest

import limulus.execution.input_preparation as input_preparation_module
from limulus.backend_integration.transport import derive_preparation_kind
import limulus.execution as execution_package
from limulus.execution import DataStepExecutor
from limulus.execution.coordinator import DataStepExecutor as DataStepExecutorType
from limulus.execution.contracts import (
    InputPreparationRequest,
    InputPreparationResult,
    InputResolutionResult,
)
from limulus.execution.input_preparation import ExecutionInputPreparationService
from limulus.execution.planning import ExecutionPlanService
from limulus.models import (
    DataSetRef,
    ExecuteRequest,
    NormalizedExecutionInputs,
    PreparedExecutionInputs,
    PreparationProvenance,
    SourceMetadataLineage,
    SourceMetadataSnapshot,
)
from limulus.execution.contracts import _snapshot_dataset_ref, _snapshot_prepared_inputs
from limulus.parser import (
    DatasetReference,
    DatasetReferenceOptionSpec,
    ParsedStatement,
    SetStatementOptionSpec,
)
from limulus.execution.rewrites import RewritePlan


def _set_statement(*names: str) -> ParsedStatement:
    return ParsedStatement(
        kind="SET",
        text=f"set {' '.join(names)}",
        dataset_refs=tuple(DatasetReference(name=name) for name in names),
    )


def _ref(location: str, *, kind: str = "arrow_table", value: int = 1) -> DataSetRef:
    payload = pa.table({"id": [value]}) if kind == "arrow_table" else [{"id": value}]
    return DataSetRef(kind=kind, location=location, payload=payload)


def test_mixed_string_promotion_reuses_values_and_unaffected_columns() -> None:
    field = pa.field("key", pa.string(), nullable=False, metadata={b"label": b"Key"})
    left = pa.Table.from_arrays(
        [pa.array(["日本", "é"]), pa.array([1, 2], type=pa.int16())],
        schema=pa.schema([field, pa.field("value", pa.int16())], metadata={b"memlabel": b"Left"}),
    )
    right = pa.table({"key": pa.array(["日本"], type=pa.large_string())})
    promoted, unchanged = input_preparation_module._promote_mixed_string_tables([left, right])

    assert unchanged is right
    assert promoted.schema.field("key") == field.with_type(pa.large_string())
    assert promoted.schema.metadata == left.schema.metadata
    assert promoted.column("key").to_pylist() == left.column("key").to_pylist()
    assert promoted.column("key").chunk(0).buffers()[2].address == left.column("key").chunk(0).buffers()[2].address
    assert promoted.column("value").chunk(0).buffers()[1].address == left.column("value").chunk(0).buffers()[1].address
    for tables in ([left], [left, left], [right, right]):
        assert input_preparation_module._promote_mixed_string_tables(tables) is tables


def test_pipeline_no_longer_owns_resolution_or_normalization_calls() -> None:
    source = inspect.getsource(DataStepExecutorType._execute_typed_block)

    assert "resolve_typed" in source
    assert "prepare_typed" in source
    assert "_resolve_inputs" not in source
    assert "normalize_execution_inputs" not in source


def test_preparation_owner_has_no_rewrite_planner_construction_or_policy_call() -> None:
    source = inspect.getsource(ExecutionInputPreparationService)

    assert "RewritePlanner" not in source
    assert "plan_rewrites" not in source


def test_set_preprocessing_island_is_owned_only_by_input_service() -> None:
    from limulus.execution import coordinator as coordinator_module

    coordinator_source = inspect.getsource(coordinator_module.DataStepExecutor)
    assert not hasattr(coordinator_module.DataStepExecutor, "_prepare_runtime_inputs")
    forbidden = (
        "_try_interleave_set_inputs",
        "_prepare_set_rows_with_temporary_variables",
        "_prepare_set_arrow_inputs_if_supported",
        "_sort_inputs_by_by_keys",
        "_arrow_set_schema_tables",
        "_set_schema_incompatibility_diagnostic",
        "_detect_set_input_internal_variable_collisions",
        "prepare_set_rows",
    )

    assert [symbol for symbol in forbidden if symbol in coordinator_source] == []


def test_input_resolution_contract_stays_module_local() -> None:
    assert not hasattr(execution_package, "ExecutionInputPreparationService")
    assert not hasattr(execution_package, "InputResolutionRequest")
    assert ExecutionInputPreparationService is not None


def test_prepare_materializes_options_from_supplied_immutable_rewrite_plan() -> None:
    service = ExecutionInputPreparationService()
    source = ParsedStatement(
        kind="SET",
        text="set source",
        dataset_refs=(
            DatasetReference(
                name="source",
                options=DatasetReferenceOptionSpec(
                    keep_vars=("id", "amount", "drop_me"),
                    drop_vars=("drop_me",),
                    rename_map={"amount": "score"},
                    firstobs=2,
                    obs=1,
                ),
            ),
        ),
    )
    source_ref = _ref("dataset://source", value=1)
    source_ref = DataSetRef(
        kind="arrow_table",
        location=source_ref.location,
        payload=pa.table(
            {
                "id": [1, 2, 3],
                "amount": [10, 20, 30],
                "drop_me": ["a", "b", "c"],
            }
        ),
    )
    rewrite_plan = RewritePlan(
        merge_mode="none",
        lag_lead_mode="none",
        where_mode="none",
        helper_source_columns=(),
        helper_runtime_columns=(),
    )

    result = service.prepare(
        InputPreparationRequest(
            ast_statements=(source,),
            input_resolution=InputResolutionResult(
                resolved_inputs=NormalizedExecutionInputs(inputs={"source": source_ref})
            ),
            rewrite_plan=rewrite_plan,
        )
    )

    assert result.diagnostics == ()
    assert result.prepared.inputs["source"].payload.to_pylist() == [
        {"id": 2, "score": 20}
    ]
    assert result.prepared.preparation_kind == "prepared_set"
    assert len(result.prepared_statements) == 1
    prepared_source = result.prepared_statements[0]
    assert prepared_source.kind == "SET"
    assert tuple(ref.name for ref in prepared_source.dataset_refs) == ("source",)
    assert prepared_source.dataset_refs[0].options == DatasetReferenceOptionSpec()


def test_prepare_failure_is_fail_closed_without_partial_carrier() -> None:
    service = ExecutionInputPreparationService()
    source = ParsedStatement(
        kind="SET",
        text="set source",
        dataset_refs=(
            DatasetReference(
                name="source",
                options=DatasetReferenceOptionSpec(rename_map={"missing": "renamed"}),
            ),
        ),
    )
    rewrite_plan = RewritePlan(
        merge_mode="none",
        lag_lead_mode="none",
        where_mode="none",
        helper_source_columns=(),
        helper_runtime_columns=(),
    )

    result = service.prepare(
        InputPreparationRequest(
            ast_statements=(source,),
            input_resolution=InputResolutionResult(
                resolved_inputs=NormalizedExecutionInputs(
                    inputs={"source": _ref("dataset://source")}
                )
            ),
            rewrite_plan=rewrite_plan,
        )
    )

    assert result.diagnostics
    assert result.diagnostics[0].code == "RUNTIME_DATASET_OPTION_INVALID"
    assert result.prepared.inputs == {}
    assert result.prepared_statements == ()


def test_prepare_does_not_construct_or_replan_rewrite_policy() -> None:
    source = _set_statement("source")
    rewrite_plan = RewritePlan(
        merge_mode="none",
        lag_lead_mode="none",
        where_mode="none",
        helper_source_columns=(),
        helper_runtime_columns=(),
    )
    service = ExecutionInputPreparationService()

    with patch.object(
        ExecutionPlanService,
        "plan_typed_helpers",
        side_effect=AssertionError("input preparation must consume settled policy"),
    ):
        result = service.prepare(
            InputPreparationRequest(
                ast_statements=(source,),
                input_resolution=InputResolutionResult(
                    resolved_inputs=NormalizedExecutionInputs(
                        inputs={"source": _ref("dataset://source")}
                    )
                ),
                rewrite_plan=rewrite_plan,
            )
        )

    assert result.diagnostics == ()


def test_prepare_returns_collapsed_set_carrier_and_transformed_statement() -> None:
    service = ExecutionInputPreparationService()
    source = ParsedStatement(
        kind="SET",
        text="set a a b indsname=source end=last",
        dataset_refs=(
            DatasetReference(
                name="a",
                options=DatasetReferenceOptionSpec(in_var="from_a"),
            ),
            DatasetReference(
                name="a",
                options=DatasetReferenceOptionSpec(in_var="from_a"),
            ),
            DatasetReference(
                name="b",
                options=DatasetReferenceOptionSpec(in_var="from_b"),
            ),
        ),
        statement_options=SetStatementOptionSpec(
            indsname_var="source",
            end_var="last",
        ),
    )
    by_statement = ParsedStatement(kind="BY", text="by grp")
    inputs = {
        "a": DataSetRef(
            kind="arrow_table",
            location="dataset://a",
            payload=pa.table({"id": [1, 2], "grp": ["B", "A"]}),
            metadata={"origin": "a"},
        ),
        "b": DataSetRef(
            kind="arrow_table",
            location="dataset://b",
            payload=pa.table({"id": [3], "grp": ["A"]}),
            metadata={"origin": "b"},
        ),
    }

    result = service.prepare(
        InputPreparationRequest(
            ast_statements=(source, by_statement),
            input_resolution=InputResolutionResult(
                resolved_inputs=NormalizedExecutionInputs(inputs=inputs)
            ),
            rewrite_plan=RewritePlan(
                merge_mode="none",
                lag_lead_mode="none",
                where_mode="none",
                helper_source_columns=(),
                helper_runtime_columns=(),
            ),
        )
    )

    assert result.diagnostics == ()
    assert tuple(result.prepared.inputs) == ("a",)
    prepared_input = result.prepared.inputs["a"]
    assert prepared_input.preparation.kind == "prepared_set"
    assert set(prepared_input.preparation.internal_variable_names) == {
        "from_a",
        "from_b",
        "source",
        "last",
        "FIRST.grp",
        "LAST.grp",
        "first.grp",
        "last.grp",
    }
    assert prepared_input.metadata == {"origin": "a"}
    assert prepared_input.payload.to_pylist() == [
        {
            "id": 2,
            "grp": "A",
            "from_a": 1,
            "from_b": 0,
            "source": "a",
            "last": 0,
            "FIRST.grp": 1,
            "LAST.grp": 0,
            "first.grp": 1,
            "last.grp": 0,
        },
        {
            "id": 2,
            "grp": "A",
            "from_a": 1,
            "from_b": 0,
            "source": "a",
            "last": 0,
            "FIRST.grp": 0,
            "LAST.grp": 0,
            "first.grp": 0,
            "last.grp": 0,
        },
        {
            "id": 3,
            "grp": "A",
            "from_a": 0,
            "from_b": 1,
            "source": "b",
            "last": 0,
            "FIRST.grp": 0,
            "LAST.grp": 1,
            "first.grp": 0,
            "last.grp": 1,
        },
        {
            "id": 1,
            "grp": "B",
            "from_a": 1,
            "from_b": 0,
            "source": "a",
            "last": 0,
            "FIRST.grp": 1,
            "LAST.grp": 0,
            "first.grp": 1,
            "last.grp": 0,
        },
        {
            "id": 1,
            "grp": "B",
            "from_a": 1,
            "from_b": 0,
            "source": "a",
            "last": 1,
            "FIRST.grp": 0,
            "LAST.grp": 1,
            "first.grp": 0,
            "last.grp": 1,
        },
    ]
    transformed = result.prepared_statements[0]
    assert tuple(ref.name for ref in transformed.dataset_refs) == ("a",)
    assert transformed.statement_options == source.statement_options
    assert transformed.span == source.span
    assert result.prepared_statements[1] == by_statement
    assert [
        (item.source_ordinal, item.source_name)
        for item in result.prepared.source_metadata_lineage.ordered_sources
    ] == [(0, "a"), (1, "a"), (2, "b")]


def test_prepare_simple_set_interleave_returns_collapsed_statement() -> None:
    service = ExecutionInputPreparationService()
    source = _set_statement("a", "b")
    result = service.prepare(
        InputPreparationRequest(
            ast_statements=(source,),
            input_resolution=InputResolutionResult(
                resolved_inputs=NormalizedExecutionInputs(
                    inputs={
                        "a": _ref("dataset://a", value=1),
                        "b": _ref("dataset://b", value=2),
                    }
                )
            ),
            rewrite_plan=RewritePlan(
                merge_mode="none",
                lag_lead_mode="none",
                where_mode="none",
                helper_source_columns=(),
                helper_runtime_columns=(),
            ),
        )
    )

    assert result.diagnostics == ()
    assert tuple(result.prepared.inputs) == ("a",)
    assert result.prepared.inputs["a"].payload.to_pylist() == [
        {"id": 1},
        {"id": 2},
    ]
    assert tuple(ref.name for ref in result.prepared_statements[0].dataset_refs) == ("a",)


def test_arrow_merge_by_selection_plan_keeps_duplicate_groups_and_null_last_order() -> None:
    service = ExecutionInputPreparationService()
    left = pa.table(
        {
            "id": pa.array([2, 2, None], type=pa.int64()),
            "value": pa.array([20, 21, 99], type=pa.int64()),
        }
    )
    right = pa.table(
        {
            "id": pa.array([1, 2, 2, None], type=pa.int64()),
            "value": pa.array([10, 200, 201, 199], type=pa.int64()),
        }
    )

    plan, diagnostic = service._build_arrow_merge_by_selection_plan(
        source_specs=(
            {"source_name": "left", "table": left, "resolved_by_names": ("id",)},
            {"source_name": "right", "table": right, "resolved_by_names": ("id",)},
        ),
        by_keys=("id",),
    )

    assert diagnostic is not None
    assert diagnostic.code == "RUNTIME_MERGE_REPEATED_BY_VALUES"
    assert diagnostic.severity == "warning"
    assert plan is not None
    assert plan.source_names == ("left", "right")
    assert [key.parts[0].value for key in plan.output_keys] == [1, 2, 2, None]
    assert plan.row_indices_by_source == (
        (None, 0, 1, 2),
        (0, 1, 2, 3),
    )
    assert plan.present_by_source == (
        (False, True, True, True),
        (True, True, True, True),
    )


def test_arrow_merge_by_selection_plan_carries_short_group_final_row() -> None:
    service = ExecutionInputPreparationService()
    plan, diagnostic = service._build_arrow_merge_by_selection_plan(
        source_specs=(
            {
                "source_name": "short",
                "table": pa.table({"id": [7], "value": [70]}),
                "resolved_by_names": ("id",),
            },
            {
                "source_name": "long",
                "table": pa.table({"id": [7, 7, 7], "value": [71, 72, 73]}),
                "resolved_by_names": ("id",),
            },
        ),
        by_keys=("id",),
    )

    assert diagnostic is None
    assert plan is not None
    assert plan.output_keys[0].parts[0].value == 7
    assert plan.row_indices_by_source == ((0, 0, 0), (0, 1, 2))
    assert plan.present_by_source == ((True, True, True), (True, True, True))


def test_arrow_merge_by_selection_plan_warns_for_repeated_groups_in_multiple_sources() -> None:
    service = ExecutionInputPreparationService()
    plan, diagnostic = service._build_arrow_merge_by_selection_plan(
        source_specs=(
            {
                "source_name": "left",
                "table": pa.table({"id": [7, 7], "value": [70, 71]}),
                "resolved_by_names": ("id",),
            },
            {
                "source_name": "right",
                "table": pa.table({"id": [7, 7, 7], "value": [80, 81, 82]}),
                "resolved_by_names": ("id",),
            },
        ),
        by_keys=("id",),
    )

    assert plan is not None
    assert diagnostic is not None
    assert diagnostic.code == "RUNTIME_MERGE_REPEATED_BY_VALUES"
    assert diagnostic.severity == "warning"
    assert "id" in diagnostic.message
    assert "left" in diagnostic.message
    assert "right" in diagnostic.message
    assert plan.row_indices_by_source == ((0, 1, 1), (0, 1, 2))


def test_prepare_merge_by_uses_selection_plan_for_duplicate_groups() -> None:
    service = ExecutionInputPreparationService()
    merge = ParsedStatement(
        kind="MERGE",
        text="merge short long",
        dataset_refs=(DatasetReference(name="short"), DatasetReference(name="long")),
    )
    by = ParsedStatement(kind="BY", text="by id")
    result = service.prepare(
        InputPreparationRequest(
            ast_statements=(merge, by),
            input_resolution=InputResolutionResult(
                resolved_inputs=NormalizedExecutionInputs(
                    inputs={
                        "short": DataSetRef(
                            kind="arrow_table",
                            location="dataset://short",
                            payload=pa.table({"id": [7], "left": [70]}),
                        ),
                        "long": DataSetRef(
                            kind="arrow_table",
                            location="dataset://long",
                            payload=pa.table({
                                "id": [7, 7, 7],
                                "right": [71, 72, 73],
                            }),
                        ),
                    }
                )
            ),
            rewrite_plan=_merge_plan(),
        )
    )

    assert result.diagnostics == ()
    assert result.prepared.inputs["short"].payload.to_pylist() == [
        {"id": 7, "left": 70, "right": 71, "FIRST.id": 1, "LAST.id": 0, "first.id": 1, "last.id": 0},
        {"id": 7, "left": 70, "right": 72, "FIRST.id": 0, "LAST.id": 0, "first.id": 0, "last.id": 0},
        {"id": 7, "left": 70, "right": 73, "FIRST.id": 0, "LAST.id": 1, "first.id": 0, "last.id": 1},
    ]


def test_prepare_merge_by_returns_warning_without_changing_many_to_many_alignment() -> None:
    service = ExecutionInputPreparationService()
    merge = ParsedStatement(
        kind="MERGE",
        text="merge left right",
        dataset_refs=(DatasetReference(name="left"), DatasetReference(name="right")),
    )
    by = ParsedStatement(kind="BY", text="by id")
    result = service.prepare(
        InputPreparationRequest(
            ast_statements=(merge, by),
            input_resolution=InputResolutionResult(
                resolved_inputs=NormalizedExecutionInputs(
                    inputs={
                        "left": DataSetRef(
                            kind="arrow_table",
                            location="dataset://left",
                            payload=pa.table({"id": [7, 7], "left_value": [70, 71]}),
                        ),
                        "right": DataSetRef(
                            kind="arrow_table",
                            location="dataset://right",
                            payload=pa.table({"id": [7, 7, 7], "right_value": [80, 81, 82]}),
                        ),
                    }
                )
            ),
            rewrite_plan=_merge_plan(),
        )
    )

    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].code == "RUNTIME_MERGE_REPEATED_BY_VALUES"
    assert result.diagnostics[0].severity == "warning"
    assert result.prepared.inputs["left"].payload.to_pylist() == [
        {"id": 7, "left_value": 70, "right_value": 80, "FIRST.id": 1, "LAST.id": 0, "first.id": 1, "last.id": 0},
        {"id": 7, "left_value": 71, "right_value": 81, "FIRST.id": 0, "LAST.id": 0, "first.id": 0, "last.id": 0},
        {"id": 7, "left_value": 71, "right_value": 82, "FIRST.id": 0, "LAST.id": 1, "first.id": 0, "last.id": 1},
    ]


def test_prepare_merge_by_gathers_nested_fields_without_row_round_trip() -> None:
    service = ExecutionInputPreparationService()
    merge = ParsedStatement(
        kind="MERGE",
        text="merge left right",
        dataset_refs=(DatasetReference(name="left"), DatasetReference(name="right")),
    )
    by = ParsedStatement(kind="BY", text="by id")
    left_nested_type = pa.struct(
        [
            pa.field("label", pa.string(), metadata={b"child": b"left-label"}),
            pa.field("values", pa.list_(pa.int64()), metadata={b"child": b"left-values"}),
        ]
    )
    right_nested_type = pa.list_(
        pa.field("item", pa.int64(), metadata={b"child": b"right-item"})
    )
    left = pa.table(
        {
            "id": pa.array([2, 2], type=pa.int64()),
            "left_nested": pa.array(
                [{"label": "a", "values": [1]}, None],
                type=left_nested_type,
            ),
        }
    )
    right = pa.table(
        {
            "id": pa.array([2, 2, 3], type=pa.int64()),
            "right_nested": pa.array([[10], [20], None], type=right_nested_type),
        }
    )
    left = pa.Table.from_batches(left.to_batches(max_chunksize=1))
    right = pa.Table.from_batches(right.to_batches(max_chunksize=1))

    result = service.prepare(
        InputPreparationRequest(
            ast_statements=(merge, by),
            input_resolution=InputResolutionResult(
                resolved_inputs=NormalizedExecutionInputs(
                    inputs={
                        "left": DataSetRef(
                            kind="arrow_table",
                            location="dataset://left",
                            payload=left,
                        ),
                        "right": DataSetRef(
                            kind="arrow_table",
                            location="dataset://right",
                            payload=right,
                        ),
                    }
                ),
            ),
            rewrite_plan=_merge_plan(),
        )
    )

    assert tuple(diagnostic.code for diagnostic in result.diagnostics) == (
        "RUNTIME_MERGE_REPEATED_BY_VALUES",
    )
    assert result.diagnostics[0].severity == "warning"
    output = result.prepared.inputs["left"].payload
    assert output.schema.field("left_nested").type == left_nested_type
    assert output.schema.field("left_nested").type[0].metadata == {b"child": b"left-label"}
    assert output.schema.field("right_nested").type == right_nested_type
    assert output.schema.field("right_nested").type.value_field.metadata == {
        b"child": b"right-item"
    }
    assert output.to_pylist() == [
        {
            "id": 2,
            "left_nested": {"label": "a", "values": [1]},
            "right_nested": [10],
            "FIRST.id": 1,
            "LAST.id": 0,
            "first.id": 1,
            "last.id": 0,
        },
        {
            "id": 2,
            "left_nested": None,
            "right_nested": [20],
            "FIRST.id": 0,
            "LAST.id": 1,
            "first.id": 0,
            "last.id": 1,
        },
        {
            "id": 3,
            "left_nested": None,
            "right_nested": None,
            "FIRST.id": 1,
            "LAST.id": 1,
            "first.id": 1,
            "last.id": 1,
        },
    ]


def test_arrow_merge_by_selection_plan_uses_raw_temporal_key_and_rejects_nested_key() -> None:
    service = ExecutionInputPreparationService()
    temporal = pa.table(
        {
            "when": pa.array([1_234_567_890_123], type=pa.timestamp("ns")),
            "value": [1],
        }
    )
    plan, diagnostic = service._build_arrow_merge_by_selection_plan(
        source_specs=(
            {"source_name": "temporal", "table": temporal, "resolved_by_names": ("when",)},
        ),
        by_keys=("when",),
    )

    assert diagnostic is None
    assert plan is not None
    assert plan.output_keys[0].parts[0].value == 1_234_567_890_123

    nested, nested_diagnostic = service._build_arrow_merge_by_selection_plan(
        source_specs=(
            {
                "source_name": "nested",
                "table": pa.table({"id": pa.array([[1]], type=pa.list_(pa.int64()))}),
                "resolved_by_names": ("id",),
            },
        ),
        by_keys=("id",),
    )

    assert nested is None
    assert nested_diagnostic is not None
    assert nested_diagnostic.code == "RUNTIME_BY_PRECONDITION_FAILED"
    assert "nested" in nested_diagnostic.message.lower()


@pytest.mark.parametrize(
    "nested_type",
    [
        pa.list_(pa.int64()),
        pa.large_list(pa.int64()),
        pa.list_(pa.int64(), 2),
        pa.struct([("label", pa.string())]),
        pa.map_(pa.string(), pa.int64()),
    ],
)
def test_sort_inputs_by_by_keys_converts_nested_by_key_to_stage_diagnostic(
    nested_type: pa.DataType,
) -> None:
    service = ExecutionInputPreparationService()
    inputs = {
        "src": DataSetRef(
            kind="arrow_table",
            location="dataset://src",
            payload=pa.table({"nested": pa.array([], type=nested_type)}),
        ),
    }

    sorted_inputs, diagnostic = service._sort_inputs_by_by_keys(
        inputs, by_keys=("nested",)
    )

    assert sorted_inputs is None
    assert diagnostic is not None
    assert diagnostic.code == "RUNTIME_BY_PRECONDITION_FAILED"
    assert diagnostic.severity == "error"
    assert diagnostic.location == "dataset:src/column:nested"
    assert "src" in diagnostic.message
    assert "nested" in diagnostic.message
    assert str(nested_type) in diagnostic.message


def test_sort_inputs_by_by_keys_preserves_non_referenced_nested_column_payload() -> None:
    service = ExecutionInputPreparationService()
    nested_type = pa.list_(pa.int64())
    preparation = PreparationProvenance(
        kind="prepared_set", internal_variable_names=frozenset({"from_src"})
    )
    inputs = {
        "src": DataSetRef(
            kind="arrow_table",
            location="dataset://src",
            payload=pa.table(
                {
                    "grp": ["B", "A", "B"],
                    "vals": pa.array([[1], [2], [3]], type=nested_type),
                }
            ),
            preparation=preparation,
        ),
    }

    sorted_inputs, diagnostic = service._sort_inputs_by_by_keys(inputs, by_keys=("grp",))

    assert diagnostic is None
    assert sorted_inputs is not None
    ref = sorted_inputs["src"]
    assert ref.kind == "arrow_table"
    assert isinstance(ref.payload, pa.Table)
    assert ref.payload.schema.field("vals").type == nested_type
    assert ref.payload.column("vals").to_pylist() == [[2], [1], [3]]
    assert ref.preparation == preparation


def test_prepare_set_by_nested_key_fails_closed_with_diagnostic() -> None:
    service = ExecutionInputPreparationService()
    source = _set_statement("src")
    by_statement = ParsedStatement(kind="BY", text="by vals")
    inputs = {
        "src": DataSetRef(
            kind="arrow_table",
            location="dataset://src",
            payload=pa.table(
                {
                    "vals": pa.array([[1], [2]], type=pa.list_(pa.int64())),
                }
            ),
        ),
    }

    result = service.prepare(
        InputPreparationRequest(
            ast_statements=(source, by_statement),
            input_resolution=InputResolutionResult(
                resolved_inputs=NormalizedExecutionInputs(inputs=inputs)
            ),
            rewrite_plan=RewritePlan(
                merge_mode="none",
                lag_lead_mode="none",
                where_mode="none",
                helper_source_columns=(),
                helper_runtime_columns=(),
            ),
        )
    )

    codes = [diagnostic.code for diagnostic in result.diagnostics]
    assert "RUNTIME_BY_PRECONDITION_FAILED" in codes
    assert result.prepared.inputs == {}


def test_sort_inputs_by_by_keys_diagnostic_names_resolved_column_when_case_differs() -> None:
    service = ExecutionInputPreparationService()
    nested_type = pa.list_(pa.int64())
    inputs = {
        "src": DataSetRef(
            kind="arrow_table",
            location="dataset://src",
            # Stored column is "Vals"; the BY key is written lowercase. The
            # diagnostic must name the RESOLVED column, not just the written key.
            payload=pa.table({"Vals": pa.array([[1], [2]], type=nested_type)}),
        ),
    }

    sorted_inputs, diagnostic = service._sort_inputs_by_by_keys(
        inputs, by_keys=("vals",)
    )

    assert sorted_inputs is None
    assert diagnostic is not None
    assert diagnostic.code == "RUNTIME_BY_PRECONDITION_FAILED"
    assert diagnostic.location == "dataset:src/column:Vals"
    assert "src" in diagnostic.message
    assert "vals" in diagnostic.message
    assert "Vals" in diagnostic.message
    assert str(nested_type) in diagnostic.message


def test_sort_inputs_by_by_keys_converts_unexpected_sort_failure_to_diagnostic() -> None:
    # Covers the unexpected-sort-failure conversion boundary around
    # ref.payload.sort_by (not the deterministic nested pre-check): a
    # dictionary-encoded key is a non-nested type that pyarrow itself refuses
    # to sort, so the raise comes from Arrow and the conversion must turn it
    # into a fail-closed diagnostic instead of letting it cross the public
    # boundary. (pa.Table is immutable, so patching sort_by is not possible.)
    service = ExecutionInputPreparationService()
    inputs = {
        "src": DataSetRef(
            kind="arrow_table",
            location="dataset://src",
            payload=pa.table(
                {
                    "grp": pa.array(["B", "A"]).dictionary_encode(),
                    "value": [2, 1],
                }
            ),
        ),
    }

    sorted_inputs, diagnostic = service._sort_inputs_by_by_keys(
        inputs, by_keys=("grp",)
    )

    assert sorted_inputs is None
    assert diagnostic is not None
    assert diagnostic.code == "RUNTIME_BY_PRECONDITION_FAILED"
    assert diagnostic.severity == "error"
    assert "src" in diagnostic.message
    underlying_text = str(pa.array(["a"]).dictionary_encode().type)
    assert underlying_text in diagnostic.message


def _nested_public_source_table() -> pa.Table:
    return pa.table(
        {
            "grp": ["B", "A", "B"],
            "vals": pa.array(
                [["a"], ["b", "c"], None], type=pa.list_(pa.string())
            ),
            "info": pa.array(
                [{"n": 1}, {"n": 2}, {"n": 3}],
                type=pa.struct([pa.field("n", pa.int64())]),
            ),
        }
    )


def _run_public_nested_step(dsl_text: str):
    executor = DataStepExecutor(runtime_backend="rust")
    response = executor.execute(
        ExecuteRequest(
            dsl_text=dsl_text,
            inputs={
                "src": DataSetRef(
                    kind="arrow_table",
                    location="dataset://src",
                    payload=_nested_public_source_table(),
                ),
            },
            output_targets=("out",),
        )
    )
    return executor, response


def test_public_set_by_keeps_non_referenced_nested_columns_in_arrow_carrier() -> None:
    executor, response = _run_public_nested_step(
        "data out; set src; by grp; output out; run;"
    )

    assert response.has_errors is False
    assert response.diagnostics == ()
    out = response.outputs["out"]
    assert isinstance(out, DataSetRef)
    assert out.kind == "arrow_table"
    table = out.payload
    assert isinstance(table, pa.Table)
    # BY-referenced scalar sort plus untouched nested payload, in stable
    # ascending BY order ("A" row first, "B" rows in original tie order).
    assert table.column_names == ["grp", "vals", "info"]
    assert table.schema.field("vals").type == pa.list_(pa.string())
    assert table.schema.field("info").type == pa.struct([pa.field("n", pa.int64())])
    assert table.to_pylist() == [
        {"grp": "A", "vals": ["b", "c"], "info": {"n": 2}},
        {"grp": "B", "vals": ["a"], "info": {"n": 1}},
        {"grp": "B", "vals": None, "info": {"n": 3}},
    ]

    # The prepared carrier is still an Arrow table: BY sorting must not fall
    # back to Python row materialization, and typed provenance survives.
    prepared = executor.last_prepared_execution_inputs
    assert prepared.preparation_kind == "prepared_set"
    carrier = prepared.inputs["src"]
    assert isinstance(carrier.payload, pa.Table)
    assert carrier.payload.schema.field("vals").type == pa.list_(pa.string())
    assert carrier.payload.schema.field("info").type == pa.struct(
        [pa.field("n", pa.int64())]
    )
    assert carrier.preparation.kind == "prepared_set"


def test_public_set_by_list_key_fails_closed_at_input_stage() -> None:
    executor, response = _run_public_nested_step(
        "data out; set src; by vals; output out; run;"
    )

    assert response.has_errors is True
    by_diagnostic = next(
        diagnostic
        for diagnostic in response.diagnostics
        if diagnostic.code == "RUNTIME_BY_PRECONDITION_FAILED"
    )
    # Pipeline tagging places input preparation failures at the input stage,
    # before any evaluation or row materialization could run.
    assert by_diagnostic.stage == "pre-processing inputs"
    # Fail closed: no partial output target is published.
    assert response.outputs.get("out") is None
    assert "src" in by_diagnostic.message
    assert "vals" in by_diagnostic.message
    assert str(pa.list_(pa.string())) in by_diagnostic.message


def test_public_set_by_struct_key_fails_closed_at_input_stage() -> None:
    executor, response = _run_public_nested_step(
        "data out; set src; by info; output out; run;"
    )

    assert response.has_errors is True
    by_diagnostic = next(
        diagnostic
        for diagnostic in response.diagnostics
        if diagnostic.code == "RUNTIME_BY_PRECONDITION_FAILED"
    )
    # Pins the earlier execute-stage struct failure into the earlier
    # input-stage diagnostic; nested BY must not surface as a public exception.
    assert by_diagnostic.stage == "pre-processing inputs"
    assert response.outputs.get("out") is None
    assert "src" in by_diagnostic.message
    assert "info" in by_diagnostic.message
    assert str(pa.struct([pa.field("n", pa.int64())])) in by_diagnostic.message


def test_arrow_merge_ordinal_selection_plan_preserves_occurrences_and_missing_rows() -> None:
    service = ExecutionInputPreparationService()
    plan = service._build_arrow_merge_ordinal_selection_plan(
        source_specs=(
            {"source_name": "first", "table": pa.table({"value": [1, 2]})},
            {"source_name": "empty", "table": pa.table({"value": []})},
            {"source_name": "repeated", "table": pa.table({"value": [3]})},
        ),
    )

    assert plan.source_names == ("first", "empty", "repeated")
    assert plan.row_count == 2
    assert plan.row_indices_by_source == (
        (0, 1),
        (None, None),
        (0, None),
    )
    assert plan.present_by_source == (
        (True, True),
        (False, False),
        (True, False),
    )


def test_arrow_merge_ordinal_selection_plan_keeps_repeated_dataset_occurrences_distinct() -> None:
    service = ExecutionInputPreparationService()
    repeated = pa.table({"value": [10, 20]})
    shortened_occurrence = pa.table({"value": [30]})

    plan = service._build_arrow_merge_ordinal_selection_plan(
        source_specs=(
            {"source_name": "same", "table": repeated},
            {"source_name": "same", "table": shortened_occurrence},
        ),
    )

    assert plan.source_names == ("same", "same")
    assert plan.row_indices_by_source == ((0, 1), (0, None))
    assert plan.present_by_source == ((True, True), (True, False))


def test_prepare_merge_without_by_uses_ordinal_selection_plan_for_empty_occurrence() -> None:
    service = ExecutionInputPreparationService()
    merge = ParsedStatement(
        kind="MERGE",
        text="merge first empty repeated",
        dataset_refs=(
            DatasetReference(name="first"),
            DatasetReference(name="empty"),
            DatasetReference(name="repeated"),
        ),
    )
    result = service.prepare(
        InputPreparationRequest(
            ast_statements=(merge,),
            input_resolution=InputResolutionResult(
                resolved_inputs=NormalizedExecutionInputs(
                    inputs={
                        "first": DataSetRef(
                            kind="arrow_table",
                            location="dataset://first",
                            payload=pa.table({"left": [1, 2]}),
                        ),
                        "empty": DataSetRef(
                            kind="arrow_table",
                            location="dataset://empty",
                            payload=pa.table({"middle": []}, schema=pa.schema([("middle", pa.int64())])),
                        ),
                        "repeated": DataSetRef(
                            kind="arrow_table",
                            location="dataset://repeated",
                            payload=pa.table({"right": [3]}),
                        ),
                    }
                ),
            ),
            rewrite_plan=_merge_plan(),
        )
    )

    assert result.diagnostics == ()
    assert result.prepared.inputs["first"].payload.to_pylist() == [
        {"left": 1, "middle": None, "right": 3},
        {"left": 2, "middle": None, "right": None},
    ]


def test_prepare_merge_without_by_gathers_nested_fields_and_typed_missing_rows() -> None:
    service = ExecutionInputPreparationService()
    merge = ParsedStatement(
        kind="MERGE",
        text="merge left right",
        dataset_refs=(DatasetReference(name="left"), DatasetReference(name="right")),
    )
    left_type = pa.list_(
        pa.field("item", pa.int64(), metadata={b"child": b"left-item"})
    )
    right_type = pa.struct(
        [pa.field("name", pa.string(), metadata={b"child": b"right-name"})]
    )
    left = pa.table(
        {
            "left_nested": pa.array([[1], None], type=left_type),
        }
    )
    right = pa.table(
        {
            "right_nested": pa.array([{"name": "r"}], type=right_type),
        }
    )
    left = pa.Table.from_batches(left.to_batches(max_chunksize=1))
    right = pa.Table.from_batches(right.to_batches(max_chunksize=1))
    result = service.prepare(
        InputPreparationRequest(
            ast_statements=(merge,),
            input_resolution=InputResolutionResult(
                resolved_inputs=NormalizedExecutionInputs(
                    inputs={
                        "left": DataSetRef(
                            kind="arrow_table",
                            location="dataset://left",
                            payload=left,
                        ),
                        "right": DataSetRef(
                            kind="arrow_table",
                            location="dataset://right",
                            payload=right,
                        ),
                    }
                )
            ),
            rewrite_plan=_merge_plan(),
        )
    )

    assert result.diagnostics == ()
    output = result.prepared.inputs["left"].payload
    assert output.schema.field("left_nested").type == left_type
    assert output.schema.field("left_nested").type.value_field.metadata == {
        b"child": b"left-item"
    }
    assert output.schema.field("right_nested").type == right_type
    assert output.schema.field("right_nested").type[0].metadata == {
        b"child": b"right-name"
    }
    assert output.schema.field("right_nested").nullable is True
    assert output.to_pylist() == [
        {"left_nested": [1], "right_nested": {"name": "r"}},
        {"left_nested": None, "right_nested": None},
    ]
    source = inspect.getsource(
        ExecutionInputPreparationService._prepare_merge_arrow_input_without_by
    )
    assert "to_pydict(" not in source
    assert "from_pylist(" not in source


def test_prepare_merge_without_by_null_field_uses_concrete_type_for_missing_rows() -> None:
    service = ExecutionInputPreparationService()
    merge = _merge_statement(
        DatasetReference(name="null_source"),
        DatasetReference(name="typed_source"),
    )
    result = service.prepare(
        InputPreparationRequest(
            ast_statements=(merge,),
            input_resolution=InputResolutionResult(
                resolved_inputs=NormalizedExecutionInputs(
                    inputs={
                        "null_source": DataSetRef(
                            kind="arrow_table",
                            location="dataset://null_source",
                            payload=pa.table(
                                {
                                    "score": pa.array([None, None], type=pa.null()),
                                }
                            ),
                        ),
                        "typed_source": DataSetRef(
                            kind="arrow_table",
                            location="dataset://typed_source",
                            payload=pa.table(
                                {"score": pa.array([5], type=pa.int64())}
                            ),
                        ),
                    }
                ),
                # Preserve the direct carrier path for this defensive null/
                # concrete-field compatibility case; the normal complete
                # lineage path reports duplicate non-BY columns first.
                source_metadata_lineage=SourceMetadataLineage(
                    (
                        SourceMetadataSnapshot(
                            source_ordinal=0,
                            source_name="null_source",
                            schema=pa.schema(
                                [pa.field("score", pa.null(), nullable=True)]
                            ),
                        ),
                    )
                ),
            ),
            rewrite_plan=_merge_plan(),
        )
    )

    assert result.diagnostics == ()
    output = result.prepared.inputs["null_source"].payload
    assert output.schema.field("score").type == pa.int64()
    assert output.to_pylist() == [{"score": 5}, {"score": None}]


def test_prepare_merge_without_by_rejects_incompatible_shared_field_type() -> None:
    service = ExecutionInputPreparationService()
    merge = ParsedStatement(
        kind="MERGE",
        text="merge left right",
        dataset_refs=(DatasetReference(name="left"), DatasetReference(name="right")),
    )
    result = service.prepare(
        InputPreparationRequest(
            ast_statements=(merge,),
            input_resolution=InputResolutionResult(
                resolved_inputs=NormalizedExecutionInputs(
                    inputs={
                        "left": DataSetRef(
                            kind="arrow_table",
                            location="dataset://left",
                            payload=pa.table({"score": pa.array([1], type=pa.int64())}),
                        ),
                        "right": DataSetRef(
                            kind="arrow_table",
                            location="dataset://right",
                            payload=pa.table({"score": pa.array([1.0], type=pa.float64())}),
                        ),
                    }
                ),
                # Keep the lineage intentionally partial so this test reaches
                # the Arrow carrier's shared-field compatibility guard rather
                # than the pre-existing duplicate-column policy.
                source_metadata_lineage=SourceMetadataLineage(
                    (
                        SourceMetadataSnapshot(
                            source_ordinal=0,
                            source_name="left",
                            schema=pa.schema(
                                [pa.field("score", pa.int64(), nullable=True)]
                            ),
                        ),
                    )
                ),
            ),
            rewrite_plan=_merge_plan(),
        )
    )

    assert result.prepared.inputs == {}
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].code == "RUNTIME_MERGE_FIELD_TYPE_INCOMPATIBLE"
    assert "score" in result.diagnostics[0].location
    assert "ordinal" in result.diagnostics[0].location


def test_prepare_merge_by_rejects_incompatible_key_type_before_gather() -> None:
    service = ExecutionInputPreparationService()
    merge = ParsedStatement(
        kind="MERGE",
        text="merge left right",
        dataset_refs=(DatasetReference(name="left"), DatasetReference(name="right")),
    )
    by = ParsedStatement(kind="BY", text="by id")
    result = service.prepare(
        InputPreparationRequest(
            ast_statements=(merge, by),
            input_resolution=InputResolutionResult(
                resolved_inputs=NormalizedExecutionInputs(
                    inputs={
                        "left": DataSetRef(
                            kind="arrow_table",
                            location="dataset://left",
                            payload=pa.table({"id": pa.array([1], type=pa.int64())}),
                        ),
                        "right": DataSetRef(
                            kind="arrow_table",
                            location="dataset://right",
                            payload=pa.table({"id": pa.array([1], type=pa.uint64())}),
                        ),
                    }
                )
            ),
            rewrite_plan=_merge_plan(),
        )
    )

    assert result.prepared.inputs == {}
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].code == "RUNTIME_BY_PRECONDITION_FAILED"
    assert result.diagnostics[0].location == "dataset:left/column:id/selection:by"
    assert "id" in result.diagnostics[0].message


def test_prepare_merge_gather_failure_is_fail_closed_without_python_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ExecutionInputPreparationService()
    merge = ParsedStatement(
        kind="MERGE",
        text="merge left right",
        dataset_refs=(DatasetReference(name="left"), DatasetReference(name="right")),
    )

    def fail_take(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected Arrow take failure")

    monkeypatch.setattr(input_preparation_module.pc, "take", fail_take)
    result = service.prepare(
        InputPreparationRequest(
            ast_statements=(merge,),
            input_resolution=InputResolutionResult(
                resolved_inputs=NormalizedExecutionInputs(
                    inputs={
                        "left": DataSetRef(
                            kind="arrow_table",
                            location="dataset://left",
                            payload=pa.table({"left": [1]}),
                        ),
                        "right": DataSetRef(
                            kind="arrow_table",
                            location="dataset://right",
                            payload=pa.table({"right": [2]}),
                        ),
                    }
                )
            ),
            rewrite_plan=_merge_plan(),
        )
    )

    assert result.prepared.inputs == {}
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].code == "RUNTIME_MERGE_ARROW_GATHER_FAILED"
    assert "selection" in result.diagnostics[0].location


def test_prepare_merge_combine_failure_is_fail_closed_with_source_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ExecutionInputPreparationService()
    merge = ParsedStatement(
        kind="MERGE",
        text="merge left right",
        dataset_refs=(DatasetReference(name="left"), DatasetReference(name="right")),
    )

    def fail_combine(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected Arrow combine failure")

    monkeypatch.setattr(
        input_preparation_module,
        "_arrow_combine_taken_chunks",
        fail_combine,
    )
    result = service.prepare(
        InputPreparationRequest(
            ast_statements=(merge,),
            input_resolution=InputResolutionResult(
                resolved_inputs=NormalizedExecutionInputs(
                    inputs={
                        "left": DataSetRef(
                            kind="arrow_table",
                            location="dataset://left",
                            payload=pa.table({"left": [1]}),
                        ),
                        "right": DataSetRef(
                            kind="arrow_table",
                            location="dataset://right",
                            payload=pa.table({"right": [2]}),
                        ),
                    }
                )
            ),
            rewrite_plan=_merge_plan(),
        )
    )

    assert result.prepared.inputs == {}
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "RUNTIME_MERGE_ARROW_GATHER_FAILED"
    assert diagnostic.location == "dataset:left/column:left/selection:combine_chunks"
    assert "injected Arrow combine failure" in diagnostic.message


def test_prepare_merge_by_empty_carrier_preserves_schema_order_and_automatic_columns() -> None:
    service = ExecutionInputPreparationService()
    merge = _merge_statement(
        DatasetReference(
            name="left",
            options=DatasetReferenceOptionSpec(in_var="in_left"),
        ),
        DatasetReference(
            name="right",
            options=DatasetReferenceOptionSpec(in_var="in_right"),
        ),
    )
    by = ParsedStatement(kind="BY", text="by id")
    left_value_type = pa.list_(pa.int64())
    right_value_type = pa.struct([pa.field("label", pa.string())])
    result = service.prepare(
        InputPreparationRequest(
            ast_statements=(merge, by),
            input_resolution=InputResolutionResult(
                resolved_inputs=NormalizedExecutionInputs(
                    inputs={
                        "left": DataSetRef(
                            kind="arrow_table",
                            location="dataset://left",
                            payload=pa.table(
                                {
                                    "id": pa.array([], type=pa.int64()),
                                    "left_value": pa.array([], type=left_value_type),
                                }
                            ),
                        ),
                        "right": DataSetRef(
                            kind="arrow_table",
                            location="dataset://right",
                            payload=pa.table(
                                {
                                    "id": pa.array([], type=pa.int64()),
                                    "right_value": pa.array([], type=right_value_type),
                                }
                            ),
                        ),
                    }
                )
            ),
            rewrite_plan=_merge_plan(),
        )
    )

    assert result.diagnostics == ()
    output = result.prepared.inputs["left"].payload
    assert output.num_rows == 0
    assert output.column_names == [
        "id",
        "left_value",
        "right_value",
        "in_left",
        "in_right",
        "FIRST.id",
        "LAST.id",
        "first.id",
        "last.id",
    ]
    assert output.schema.field("id").type == pa.int64()
    assert output.schema.field("left_value").type == left_value_type
    assert output.schema.field("right_value").type == right_value_type
    assert output.schema.field("in_left").type == pa.int8()
    assert output.schema.field("in_right").type == pa.int8()
    for name in ("FIRST.id", "LAST.id", "first.id", "last.id"):
        assert output.schema.field(name).type == pa.int8()


def test_prepare_merge_by_null_field_uses_concrete_type_for_gather() -> None:
    service = ExecutionInputPreparationService()
    merge = _merge_statement(
        DatasetReference(name="null_key"),
        DatasetReference(name="typed_key"),
    )
    by = ParsedStatement(kind="BY", text="by id")
    result = service.prepare(
        InputPreparationRequest(
            ast_statements=(merge, by),
            input_resolution=InputResolutionResult(
                resolved_inputs=NormalizedExecutionInputs(
                    inputs={
                        "null_key": DataSetRef(
                            kind="arrow_table",
                            location="dataset://null_key",
                            payload=pa.table(
                                {
                                    "id": pa.array([None], type=pa.null()),
                                    "left_value": pa.array([10], type=pa.int64()),
                                }
                            ),
                        ),
                        "typed_key": DataSetRef(
                            kind="arrow_table",
                            location="dataset://typed_key",
                            payload=pa.table(
                                {
                                    "id": pa.array([None], type=pa.int64()),
                                    "right_value": pa.array([20], type=pa.int64()),
                                }
                            ),
                        ),
                    }
                )
            ),
            rewrite_plan=_merge_plan(),
        )
    )

    assert result.diagnostics == ()
    output = result.prepared.inputs["null_key"].payload
    assert output.schema.field("id").type == pa.int64()
    assert output.to_pylist() == [
        {
            "id": None,
            "left_value": 10,
            "right_value": 20,
            "FIRST.id": 1,
            "LAST.id": 1,
            "first.id": 1,
            "last.id": 1,
        }
    ]


def test_prepare_set_statement_options_and_by_use_full_carrier_preparation() -> None:
    service = ExecutionInputPreparationService()
    source = ParsedStatement(
        kind="SET",
        text="set a b indsname=src end=eof",
        dataset_refs=(
            DatasetReference(name="a"),
            DatasetReference(name="b"),
        ),
        statement_options=SetStatementOptionSpec(
            indsname_var="src",
            end_var="eof",
        ),
    )
    by_statement = ParsedStatement(kind="BY", text="by grp")
    result = service.prepare(
        InputPreparationRequest(
            ast_statements=(source, by_statement),
            input_resolution=InputResolutionResult(
                resolved_inputs=NormalizedExecutionInputs(
                    inputs={
                        "a": DataSetRef(
                            kind="arrow_table",
                            location="dataset://a",
                            payload=pa.table({"id": [1], "grp": ["A"]}),
                        ),
                        "b": DataSetRef(
                            kind="arrow_table",
                            location="dataset://b",
                            payload=pa.table({"id": [2], "grp": ["A"]}),
                        ),
                    }
                )
            ),
            rewrite_plan=RewritePlan(
                merge_mode="none",
                lag_lead_mode="none",
                where_mode="none",
                helper_source_columns=(),
                helper_runtime_columns=(),
            ),
        )
    )

    assert result.diagnostics == ()
    rows = result.prepared.inputs["a"].payload.to_pylist()
    assert len(rows) == 2
    assert [row["src"] for row in rows] == ["a", "b"]
    assert [row["eof"] for row in rows] == [0, 1]
    assert [row["FIRST.grp"] for row in rows] == [1, 0]
    assert [row["LAST.grp"] for row in rows] == [0, 1]
    assert tuple(ref.name for ref in result.prepared_statements[0].dataset_refs) == ("a",)


def test_prepare_repeated_set_source_with_indsname_does_not_reprepare_collapsed_rows() -> None:
    service = ExecutionInputPreparationService()
    source = ParsedStatement(
        kind="SET",
        text="set a a indsname=src",
        dataset_refs=(
            DatasetReference(name="a"),
            DatasetReference(name="a"),
        ),
        statement_options=SetStatementOptionSpec(indsname_var="src"),
    )
    result = service.prepare(
        InputPreparationRequest(
            ast_statements=(source,),
            input_resolution=InputResolutionResult(
                resolved_inputs=NormalizedExecutionInputs(
                    inputs={
                        "a": DataSetRef(
                            kind="arrow_table",
                            location="dataset://a",
                            payload=pa.table({"id": [1, 2]}),
                        )
                    }
                )
            ),
            rewrite_plan=RewritePlan(
                merge_mode="none",
                lag_lead_mode="none",
                where_mode="none",
                helper_source_columns=(),
                helper_runtime_columns=(),
            ),
        )
    )

    assert result.diagnostics == ()
    rows = result.prepared.inputs["a"].payload.to_pylist()
    assert len(rows) == 4
    assert [row["id"] for row in rows] == [1, 2, 1, 2]
    assert [row["src"] for row in rows] == ["a", "a", "a", "a"]
    assert tuple(ref.name for ref in result.prepared_statements[0].dataset_refs) == ("a",)


def test_prepare_repeated_set_source_without_options_preserves_each_occurrence() -> None:
    service = ExecutionInputPreparationService()
    source = _set_statement("a", "a")
    result = service.prepare(
        InputPreparationRequest(
            ast_statements=(source,),
            input_resolution=InputResolutionResult(
                resolved_inputs=NormalizedExecutionInputs(
                    inputs={
                        "a": DataSetRef(
                            kind="arrow_table",
                            location="dataset://a",
                            payload=pa.table({"id": [1, 2]}),
                        )
                    }
                )
            ),
            rewrite_plan=RewritePlan(
                merge_mode="none",
                lag_lead_mode="none",
                where_mode="none",
                helper_source_columns=(),
                helper_runtime_columns=(),
            ),
        )
    )

    assert result.diagnostics == ()
    assert len(result.prepared.inputs["a"].payload.to_pylist()) == 4
    assert tuple(ref.name for ref in result.prepared_statements[0].dataset_refs) == ("a",)


def test_prepare_missing_set_stream_is_fail_closed_without_transformed_statement() -> None:
    service = ExecutionInputPreparationService()
    source = _set_statement("a", "missing")
    result = service.prepare(
        InputPreparationRequest(
            ast_statements=(source,),
            input_resolution=InputResolutionResult(
                resolved_inputs=NormalizedExecutionInputs(inputs={"a": _ref("dataset://a")})
            ),
            rewrite_plan=RewritePlan(
                merge_mode="none",
                lag_lead_mode="none",
                where_mode="none",
                helper_source_columns=(),
                helper_runtime_columns=(),
            ),
        )
    )

    assert result.diagnostics
    assert result.diagnostics[0].code == "RUNTIME_SET_DATASET_NOT_FOUND"
    assert result.prepared.inputs == {}
    assert result.prepared_statements == ()


def test_coordinator_has_no_set_statement_collapse_production_branch() -> None:
    from limulus.execution import coordinator as coordinator_module

    coordinator_source = inspect.getsource(coordinator_module.DataStepExecutor)

    assert "_collapse_rewrite_prepared_set_statement" not in coordinator_source


def _merge_statement(*refs: DatasetReference) -> ParsedStatement:
    return ParsedStatement(
        kind="MERGE",
        text="merge " + " ".join(ref.name for ref in refs),
        dataset_refs=tuple(refs),
    )


def _merge_plan() -> RewritePlan:
    return RewritePlan(
        merge_mode="merge",
        lag_lead_mode="none",
        where_mode="none",
        helper_source_columns=(),
        helper_runtime_columns=(),
    )


def test_prepare_merge_by_aligns_carrier_and_transforms_statement_with_in_vars() -> None:
    service = ExecutionInputPreparationService()
    merge = _merge_statement(
        DatasetReference(
            name="LEFT",
            options=DatasetReferenceOptionSpec(in_var="in_left"),
        ),
        DatasetReference(
            name="right",
            options=DatasetReferenceOptionSpec(in_var="in_right"),
        ),
    )
    by = ParsedStatement(kind="BY", text="by id")
    result = service.prepare(
        InputPreparationRequest(
            ast_statements=(merge, by),
            input_resolution=InputResolutionResult(
                resolved_inputs=NormalizedExecutionInputs(
                    inputs={
                        "left": DataSetRef(
                            kind="arrow_table",
                            location="dataset://left",
                            payload=pa.table(
                                {"id": [2, 1], "left_value": [20, 10]}
                            ),
                            metadata={"origin": "left"},
                        ),
                        "RIGHT": DataSetRef(
                            kind="arrow_table",
                            location="dataset://right",
                            payload=pa.table(
                                {"id": [1, 3], "right_value": [100, 300]}
                            ),
                        ),
                    }
                )
            ),
            rewrite_plan=_merge_plan(),
        )
    )

    assert result.diagnostics == ()
    assert tuple(result.prepared.inputs) == ("LEFT",)
    prepared = result.prepared.inputs["LEFT"]
    assert prepared.preparation.kind == "prepared_merge"
    assert set(prepared.preparation.internal_variable_names) == {
        "in_left",
        "in_right",
        "FIRST.id",
        "LAST.id",
        "first.id",
        "last.id",
    }
    assert prepared.metadata == {"origin": "left"}
    assert prepared.payload.to_pylist() == [
        {
            "id": 1,
            "left_value": 10,
            "right_value": 100,
            "in_left": 1,
            "in_right": 1,
            "FIRST.id": 1,
            "LAST.id": 1,
            "first.id": 1,
            "last.id": 1,
        },
        {
            "id": 2,
            "left_value": 20,
            "right_value": None,
            "in_left": 1,
            "in_right": 0,
            "FIRST.id": 1,
            "LAST.id": 1,
            "first.id": 1,
            "last.id": 1,
        },
        {
            "id": 3,
            "left_value": None,
            "right_value": 300,
            "in_left": 0,
            "in_right": 1,
            "FIRST.id": 1,
            "LAST.id": 1,
            "first.id": 1,
            "last.id": 1,
        },
    ]
    transformed = result.prepared_statements[0]
    assert transformed.kind == "SET"
    assert tuple(ref.name for ref in transformed.dataset_refs) == ("LEFT",)
    assert result.prepared_statements[1] == by


def test_prepare_merge_without_by_keeps_ordinal_alignment_and_in_vars() -> None:
    service = ExecutionInputPreparationService()
    merge = _merge_statement(
        DatasetReference(
            name="a",
            options=DatasetReferenceOptionSpec(in_var="from_a"),
        ),
        DatasetReference(
            name="b",
            options=DatasetReferenceOptionSpec(in_var="from_b"),
        ),
    )
    result = service.prepare(
        InputPreparationRequest(
            ast_statements=(merge,),
            input_resolution=InputResolutionResult(
                resolved_inputs=NormalizedExecutionInputs(
                    inputs={
                        "a": DataSetRef(
                            kind="arrow_table",
                            location="dataset://a",
                            payload=pa.table({"a_value": [1, 2]}),
                        ),
                        "b": DataSetRef(
                            kind="arrow_table",
                            location="dataset://b",
                            payload=pa.table({"b_value": [10]}),
                        ),
                    }
                )
            ),
            rewrite_plan=_merge_plan(),
        )
    )

    assert result.diagnostics == ()
    rows = result.prepared.inputs["a"].payload.to_pylist()
    assert rows == [
        {"a_value": 1, "b_value": 10, "from_a": 1, "from_b": 1},
        {"a_value": 2, "b_value": None, "from_a": 1, "from_b": 0},
    ]
    assert result.prepared_statements[0].kind == "SET"


def test_prepare_merge_duplicate_non_by_column_is_fail_closed() -> None:
    service = ExecutionInputPreparationService()
    merge = _merge_statement(DatasetReference(name="a"), DatasetReference(name="b"))
    by = ParsedStatement(kind="BY", text="by id")
    result = service.prepare(
        InputPreparationRequest(
            ast_statements=(merge, by),
            input_resolution=InputResolutionResult(
                resolved_inputs=NormalizedExecutionInputs(
                    inputs={
                        "a": DataSetRef(
                            kind="arrow_table",
                            location="dataset://a",
                            payload=pa.table({"id": [1], "value": [10]}),
                        ),
                        "b": DataSetRef(
                            kind="arrow_table",
                            location="dataset://b",
                            payload=pa.table({"id": [1], "value": [20]}),
                        ),
                    }
                )
            ),
            rewrite_plan=_merge_plan(),
        )
    )

    assert result.diagnostics
    assert result.diagnostics[0].code == "RUNTIME_MERGE_DUPLICATE_COLUMN"
    assert result.prepared.inputs == {}
    assert result.prepared_statements == ()


def test_prepare_merge_missing_stream_is_fail_closed_without_partial_result() -> None:
    service = ExecutionInputPreparationService()
    merge = _merge_statement(
        DatasetReference(name="a"), DatasetReference(name="missing")
    )
    result = service.prepare(
        InputPreparationRequest(
            ast_statements=(merge,),
            input_resolution=InputResolutionResult(
                resolved_inputs=NormalizedExecutionInputs(
                    inputs={
                        "a": DataSetRef(
                            kind="arrow_table",
                            location="dataset://a",
                            payload=pa.table({"id": [1]}),
                        )
                    }
                )
            ),
            rewrite_plan=_merge_plan(),
        )
    )

    assert result.diagnostics
    assert result.diagnostics[0].code == "RUNTIME_SET_DATASET_NOT_FOUND"
    assert result.prepared.inputs == {}
    assert result.prepared_statements == ()


def test_coordinator_has_no_merge_preparation_production_path() -> None:
    from limulus.execution import coordinator as coordinator_module

    coordinator_source = inspect.getsource(coordinator_module.DataStepExecutor)

    forbidden = (
        "_prepare_merge_for_execution",
        "_prepare_merge_arrow_input_if_supported",
        "_prepare_merge_arrow_input_without_by",
        "_coerce_prepared_merge_input",
        "_merge_by_schema_incompatibility_diagnostic",
        "_annotate_arrow_by_group_flags",
        "_detect_internal_variable_collision_in_schema",
        "prepare_merge_rows",
    )
    assert [symbol for symbol in forbidden if symbol in coordinator_source] == []


class TestTypedPreparationProvenance:
    """Immutable carrier provenance contract and aggregate validation.

    Prepared kind and case-preserving internal variable names travel in a typed
    contract that is separate from ``location``, ``payload``, and ``metadata``.
    The aggregate rule is validated from typed provenance only: no marker
    inference, no MERGE preference, no silent overwrite.
    """

    @staticmethod
    def _prepared_ref(
        *,
        preparation: PreparationProvenance | None = None,
        location: str = "dataset://a",
    ) -> DataSetRef:
        if preparation is None:
            preparation = PreparationProvenance()
        return DataSetRef(
            kind="arrow_table",
            location=location,
            payload=pa.table({"id": [1]}),
            preparation=preparation,
        )

    def test_dataset_ref_default_preparation_is_none_provenance(self) -> None:
        default = PreparationProvenance()
        assert default.kind == "none"
        assert default.internal_variable_names == ()

        ref = DataSetRef(kind="arrow_table", location="dataset://plain")
        assert ref.preparation == default

        # Existing positional/keyword constructions stay default-compatible.
        legacy = DataSetRef(
            kind="arrow_table",
            location="dataset://legacy",
            payload=pa.table({"id": [1]}),
            metadata={"source": "unit"},
        )
        assert legacy.preparation.kind == "none"
        assert legacy.location == "dataset://legacy"

    def test_preparation_provenance_is_immutable_and_case_preserving(self) -> None:
        provenance = PreparationProvenance(
            kind="prepared_merge",
            internal_variable_names=("Retention", "first_Rate"),
        )
        assert isinstance(provenance.internal_variable_names, tuple)
        assert provenance.internal_variable_names == ("Retention", "first_Rate")

        with pytest.raises(Exception):
            provenance.kind = "none"  # type: ignore[misc]

        with pytest.raises(ValueError, match="unsupported preparation kind"):
            PreparationProvenance(kind="prepared_unknown")

        with pytest.raises(ValueError, match="unsupported preparation kind"):
            DataSetRef(
                kind="arrow_table",
                location="dataset://bad",
                preparation=PreparationProvenance(kind="nope"),
            )

    def test_prepared_carrier_round_trips_through_snapshot_without_reset(self) -> None:
        ref = self._prepared_ref(
            preparation=PreparationProvenance(
                kind="prepared_set",
                internal_variable_names=("NewTotal", "Flag_Internal"),
            ),
            location="dataset://a#prepared_set_rows",
        )

        ref_snapshot = _snapshot_dataset_ref(ref)
        assert isinstance(ref_snapshot, DataSetRef)
        assert ref_snapshot.preparation.kind == "prepared_set"
        assert ref_snapshot.preparation.internal_variable_names == ("NewTotal", "Flag_Internal")
        assert ref_snapshot.location == "dataset://a#prepared_set_rows"
        # Payload is referenced, not duplicated.
        assert ref_snapshot.payload is ref.payload

        result = InputPreparationResult(
            prepared=PreparedExecutionInputs(
                inputs={"a": ref},
                preparation_kind="prepared_set",
            )
        )
        round_tripped = result.prepared.inputs["a"]
        assert round_tripped.preparation.kind == "prepared_set"
        assert round_tripped.preparation.internal_variable_names == ("NewTotal", "Flag_Internal")
        # The aggregate preparation kind survives the prepared-inputs snapshot.
        assert result.prepared.preparation_kind == "prepared_set"

    def test_prepared_execution_inputs_snapshot_round_trips_provenance(self) -> None:
        ref = self._prepared_ref(
            preparation=PreparationProvenance(
                kind="prepared_merge",
                internal_variable_names=("In_Var", "first.Rate"),
            ),
            location="dataset://m",
        )
        prepared = PreparedExecutionInputs(
            inputs={"m": ref, "plain": self._prepared_ref()},
            preparation_kind="prepared_merge",
        )

        snapshot = _snapshot_prepared_inputs(prepared)
        assert snapshot is not prepared
        assert snapshot.preparation_kind == "prepared_merge"
        round_tripped = snapshot.inputs["m"]
        assert round_tripped.preparation.kind == "prepared_merge"
        assert round_tripped.preparation.internal_variable_names == ("In_Var", "first.Rate")
        assert round_tripped.location == "dataset://m"
        # Payload is referenced, not duplicated.
        assert round_tripped.payload is ref.payload
        # The snapshot mapping stays immutable.
        with pytest.raises(TypeError):
            snapshot.inputs["new"] = round_tripped  # type: ignore[index]

    def test_aggregate_preparation_provenance_accepts_consistent_inputs(self) -> None:
        aggregate = ExecutionInputPreparationService._aggregate_preparation_provenance

        assert aggregate({}, "none") is None
        assert aggregate(
            {"a": self._prepared_ref(preparation=PreparationProvenance())}, "none"
        ) is None
        assert aggregate(
            {
                "a": self._prepared_ref(
                    preparation=PreparationProvenance(kind="prepared_set", internal_variable_names=("X",))
                )
            },
            "prepared_set",
        ) is None
        assert aggregate(
            {
                "a": self._prepared_ref(
                    preparation=PreparationProvenance(kind="prepared_merge", internal_variable_names=("Y",))
                )
            },
            "prepared_merge",
        ) is None
        assert aggregate(
            {
                "prepared": self._prepared_ref(
                    preparation=PreparationProvenance(kind="prepared_set", internal_variable_names=("X",))
                ),
                "plain": self._prepared_ref(),
            },
            "prepared_set",
        ) is None

    def test_aggregate_preparation_provenance_rejects_mixed_kinds(self) -> None:
        conflict = ExecutionInputPreparationService._aggregate_preparation_provenance(
            {
                "a": self._prepared_ref(
                    preparation=PreparationProvenance(kind="prepared_set", internal_variable_names=("X",))
                ),
                "b": self._prepared_ref(
                    preparation=PreparationProvenance(kind="prepared_merge", internal_variable_names=("Y",))
                ),
            },
            "prepared_set",
        )

        assert conflict is not None
        assert conflict.code == "RUNTIME_PREPARATION_PROVENANCE_CONFLICT"
        assert conflict.severity == "error"
        assert "prepared_set" in conflict.message
        assert "prepared_merge" in conflict.message
        # Stage naming stays orchestrator-owned, matching service diagnostics.
        assert conflict.stage == ""

    def test_aggregate_preparation_provenance_rejects_mismatched_declared_kind(self) -> None:
        aggregate = ExecutionInputPreparationService._aggregate_preparation_provenance
        prepared = {
            "a": self._prepared_ref(
                preparation=PreparationProvenance(kind="prepared_set", internal_variable_names=("X",))
            )
        }

        for declared in ("none", "prepared_merge"):
            diagnostic = aggregate(prepared, declared)
            assert diagnostic is not None
            assert diagnostic.code == "RUNTIME_PREPARATION_PROVENANCE_CONFLICT"
            assert diagnostic.severity == "error"
            assert "prepared_set" in diagnostic.message
            assert declared in diagnostic.message

    def test_aggregate_ignores_markers_and_uses_typed_provenance_only(self) -> None:
        # A merge-flavored location marker must not influence the aggregate:
        # only typed provenance is authoritative.
        ref = self._prepared_ref(
            preparation=PreparationProvenance(kind="prepared_set", internal_variable_names=("X",)),
            location="dataset://a#prepared_merge_rows",
        )
        assert (
            ExecutionInputPreparationService._aggregate_preparation_provenance(
                {"a": ref}, "prepared_set"
            )
            is None
        )


class TestTypedProvenanceMigration:
    """Prepared carrier provenance is typed end to end.

    Producers write preparation kind and case-preserving internal names into
    the typed contract while user locations stay byte-for-byte opaque;
    planning, handoff, transport, and the native bridge read the typed
    contract only, so marker-like user data can never change behavior.
    """

    @staticmethod
    def _set_plan() -> RewritePlan:
        return RewritePlan(
            merge_mode="none",
            lag_lead_mode="none",
            where_mode="none",
            helper_source_columns=(),
            helper_runtime_columns=(),
        )

    @staticmethod
    def _set_input(
        location: str,
        payload: pa.Table,
        *,
        preparation: PreparationProvenance | None = None,
    ) -> DataSetRef:
        return DataSetRef(
            kind="arrow_table",
            location=location,
            payload=payload,
            preparation=preparation or PreparationProvenance(),
        )

    # --- producers: write typed provenance --------------------------------

    def test_set_arrow_carrier_sets_typed_provenance_and_preserves_marker_like_location(self) -> None:
        service = ExecutionInputPreparationService()
        user_location = "dataset://UserBase|internal=x#prepared_set_rows"
        source = ParsedStatement(
            kind="SET",
            text="set a b in=in_a",
            dataset_refs=(
                DatasetReference(
                    name="a", options=DatasetReferenceOptionSpec(in_var="in_a")
                ),
                DatasetReference(name="b"),
            ),
        )
        by_statement = ParsedStatement(kind="BY", text="by grp")
        result = service.prepare(
            InputPreparationRequest(
                ast_statements=(source, by_statement),
                input_resolution=InputResolutionResult(
                    resolved_inputs=NormalizedExecutionInputs(
                        inputs={
                            "a": self._set_input(
                                user_location,
                                pa.table({"id": [1], "grp": ["A"]}),
                            ),
                            "b": self._set_input(
                                "dataset://b", pa.table({"id": [2], "grp": ["A"]})
                            ),
                        }
                    )
                ),
                rewrite_plan=self._set_plan(),
            )
        )

        assert result.diagnostics == ()
        carrier = result.prepared.inputs["a"]
        # The user location is opaque: no suffix is concatenated onto it.
        assert carrier.location == user_location
        provenance = carrier.preparation
        assert provenance.kind == "prepared_set"
        assert provenance.internal_variable_names == tuple(
            sorted(
                {
                    "in_a",
                    "FIRST.grp",
                    "LAST.grp",
                    "first.grp",
                    "last.grp",
                }
            )
        )
        assert result.prepared.preparation_kind == "prepared_set"

    def test_set_rows_carrier_sets_typed_provenance(self) -> None:
        service = ExecutionInputPreparationService()
        source = ParsedStatement(
            kind="SET",
            text="set a(in=in_a) b",
            dataset_refs=(
                DatasetReference(
                    name="a", options=DatasetReferenceOptionSpec(in_var="in_a")
                ),
                DatasetReference(name="b"),
            ),
        )
        with patch.object(
            input_preparation_module,
            "try_apply_dataset_reference_options_to_arrow_table",
            return_value=(None, None),
        ):
            result = service.prepare(
                InputPreparationRequest(
                    ast_statements=(source,),
                    input_resolution=InputResolutionResult(
                        resolved_inputs=NormalizedExecutionInputs(
                            inputs={
                                "a": self._set_input(
                                    "dataset://a", pa.table({"id": [1]})
                                ),
                                "b": self._set_input(
                                    "dataset://b", pa.table({"id": [2]})
                                ),
                            }
                        )
                    ),
                    rewrite_plan=self._set_plan(),
                )
            )

        assert result.diagnostics == ()
        carrier = result.prepared.inputs["a"]
        assert carrier.preparation.kind == "prepared_set"
        assert set(carrier.preparation.internal_variable_names) == {"in_a"}
        assert carrier.location == "dataset://a"
        assert result.prepared.preparation_kind == "prepared_set"

    def test_merge_by_arrow_carrier_sets_typed_provenance(self) -> None:
        service = ExecutionInputPreparationService()
        merge = ParsedStatement(
            kind="MERGE",
            text="merge left right in=in_left in=in_right",
            dataset_refs=(
                DatasetReference(
                    name="LEFT", options=DatasetReferenceOptionSpec(in_var="in_left")
                ),
                DatasetReference(
                    name="right",
                    options=DatasetReferenceOptionSpec(in_var="in_right"),
                ),
            ),
        )
        by = ParsedStatement(kind="BY", text="by id")
        result = service.prepare(
            InputPreparationRequest(
                ast_statements=(merge, by),
                input_resolution=InputResolutionResult(
                    resolved_inputs=NormalizedExecutionInputs(
                        inputs={
                            "left": self._set_input(
                                "dataset://left",
                                pa.table({"id": [2, 1], "left_value": [20, 10]}),
                            ),
                            "RIGHT": self._set_input(
                                "dataset://right",
                                pa.table({"id": [1, 3], "right_value": [100, 300]}),
                            ),
                        }
                    )
                ),
                rewrite_plan=RewritePlan(
                    merge_mode="merge",
                    lag_lead_mode="none",
                    where_mode="none",
                    helper_source_columns=(),
                    helper_runtime_columns=(),
                ),
            )
        )

        assert result.diagnostics == ()
        carrier = result.prepared.inputs["LEFT"]
        assert carrier.location == "dataset://left"
        assert carrier.preparation.kind == "prepared_merge"
        assert set(carrier.preparation.internal_variable_names) == {
            "in_left",
            "in_right",
            "FIRST.id",
            "LAST.id",
            "first.id",
            "last.id",
        }
        assert result.prepared.preparation_kind == "prepared_merge"

    def test_merge_rows_carrier_sets_typed_provenance(self) -> None:
        service = ExecutionInputPreparationService()
        merge = ParsedStatement(
            kind="MERGE",
            text="merge left right in=in_left in=in_right",
            dataset_refs=(
                DatasetReference(
                    name="left", options=DatasetReferenceOptionSpec(in_var="in_left")
                ),
                DatasetReference(
                    name="right",
                    options=DatasetReferenceOptionSpec(in_var="in_right"),
                ),
            ),
        )
        by = ParsedStatement(kind="BY", text="by id")
        with patch.object(
            input_preparation_module,
            "try_apply_dataset_reference_options_to_arrow_table",
            return_value=(None, None),
        ):
            result = service.prepare(
                InputPreparationRequest(
                    ast_statements=(merge, by),
                    input_resolution=InputResolutionResult(
                        resolved_inputs=NormalizedExecutionInputs(
                            inputs={
                                "left": self._set_input(
                                    "dataset://left",
                                    pa.table({"id": [1], "left_value": [10]}),
                                ),
                                "right": self._set_input(
                                    "dataset://right",
                                    pa.table({"id": [1], "right_value": [100]}),
                                ),
                            }
                        )
                    ),
                    rewrite_plan=RewritePlan(
                        merge_mode="merge",
                        lag_lead_mode="none",
                        where_mode="none",
                        helper_source_columns=(),
                        helper_runtime_columns=(),
                    ),
                )
            )

        assert result.diagnostics == ()
        carrier = result.prepared.inputs["left"]
        assert carrier.preparation.kind == "prepared_merge"
        assert set(carrier.preparation.internal_variable_names) == {
            "in_left",
            "in_right",
            "FIRST.id",
            "LAST.id",
            "first.id",
            "last.id",
        }
        assert carrier.location == "dataset://left"

    def test_interleave_carrier_sets_typed_provenance_and_preserves_location(self) -> None:
        service = ExecutionInputPreparationService()
        source = ParsedStatement(
            kind="SET",
            text="set a b",
            dataset_refs=(DatasetReference(name="a"), DatasetReference(name="b")),
        )
        result = service.prepare(
            InputPreparationRequest(
                ast_statements=(source,),
                input_resolution=InputResolutionResult(
                    resolved_inputs=NormalizedExecutionInputs(
                        inputs={
                            "a": self._set_input(
                                "dataset://a", pa.table({"id": [1]})
                            ),
                            "b": self._set_input(
                                "dataset://b", pa.table({"id": [2]})
                            ),
                        }
                    )
                ),
                rewrite_plan=self._set_plan(),
            )
        )

        assert result.diagnostics == ()
        assert tuple(result.prepared.inputs) == ("a",)
        carrier = result.prepared.inputs["a"]
        assert carrier.location == "dataset://a"
        assert carrier.preparation.kind == "prepared_set"
        assert carrier.preparation.internal_variable_names == ()

    # --- consumers: read typed provenance only -----------------------------

    def test_projection_dedupes_internal_aliases_from_typed_provenance(self) -> None:
        service = ExecutionInputPreparationService()
        ref = self._set_input(
            "dataset://a",
            pa.table({"in_a": [1], "IN_A": [2], "keep": [3]}),
            preparation=PreparationProvenance(
                kind="prepared_set", internal_variable_names=("in_a",)
            ),
        )

        projected = service.project_prepared_internal_aliases_for_handoff({"a": ref})

        # First physical alias retained, user columns untouched.
        assert projected["a"].payload.column_names == ["in_a", "keep"]
        assert projected["a"].location == "dataset://a"
        assert projected["a"].preparation.kind == "prepared_set"

    def test_projection_ignores_marker_like_location_without_typed_provenance(self) -> None:
        service = ExecutionInputPreparationService()
        ref = DataSetRef(
            kind="arrow_table",
            location="dataset://user#prepared_set_rows|internal=secret",
            payload=pa.table({"secret": [1], "SECRET": [2], "keep": [3]}),
        )

        projected = service.project_prepared_internal_aliases_for_handoff({"a": ref})

        # A marker-like user location is opaque user data: both case variants
        # are user columns and must survive unchanged.
        assert projected["a"].payload.column_names == ["secret", "SECRET", "keep"]

    def test_conflicting_user_carrier_kind_is_fail_closed_at_preparation(self) -> None:
        service = ExecutionInputPreparationService()
        source = ParsedStatement(
            kind="SET",
            text="set a(in=in_a)",
            dataset_refs=(
                DatasetReference(
                    name="a", options=DatasetReferenceOptionSpec(in_var="in_a")
                ),
            ),
        )
        result = service.prepare(
            InputPreparationRequest(
                ast_statements=(source,),
                input_resolution=InputResolutionResult(
                    resolved_inputs=NormalizedExecutionInputs(
                        inputs={
                            "a": self._set_input(
                                "dataset://a", pa.table({"id": [1]})
                            ),
                            # User-supplied extra carrier declares the merge kind:
                            # the prepared SET carrier then conflicts with it.
                            "c": DataSetRef(
                                kind="arrow_table",
                                location="dataset://c",
                                payload=pa.table({"id": [2]}),
                                preparation=PreparationProvenance(
                                    kind="prepared_merge"
                                ),
                            ),
                        }
                    )
                ),
                rewrite_plan=self._set_plan(),
            )
        )

        assert [d.code for d in result.diagnostics] == [
            "RUNTIME_PREPARATION_PROVENANCE_CONFLICT"
        ]
        assert result.prepared.inputs == {}
        assert result.prepared.preparation_kind == "none"

    def test_conflicting_user_carrier_kind_aborts_public_execution(self) -> None:
        executor = DataStepExecutor(runtime_backend="rust")
        response = executor.execute(
            ExecuteRequest(
                dsl_text="data out; set a(in=in_a); output out; run;",
                inputs={
                    "a": self._set_input("dataset://a", pa.table({"id": [1]})),
                    # The conflicting carrier's location is marker-flavored on
                    # purpose: if any layer re-inferred the kind from location
                    # text, the conflict would silently disappear and the
                    # request would succeed.
                    "c": DataSetRef(
                        kind="arrow_table",
                        location="dataset://c#prepared_set_rows",
                        payload=pa.table({"id": [2]}),
                        preparation=PreparationProvenance(kind="prepared_merge"),
                    ),
                },
                output_targets=("out",),
            )
        )

        assert response.has_errors is True
        assert response.diagnostics[0].code == "RUNTIME_PREPARATION_PROVENANCE_CONFLICT"
        # Fail closed: no partial output target is published.
        assert response.outputs.get("out") is None

    # --- transport: carry typed provenance opaquely ------------------------

    def test_transport_derives_typed_kind_without_marker_preference(self) -> None:
        prepared_merge = self._set_input(
            "dataset://a",
            pa.table({"id": [1]}),
            preparation=PreparationProvenance(kind="prepared_merge"),
        )
        prepared_set = self._set_input(
            "dataset://b",
            pa.table({"id": [2]}),
            preparation=PreparationProvenance(kind="prepared_set"),
        )
        plain = self._set_input(
            "dataset://c#prepared_set_rows|internal=x", pa.table({"id": [3]})
        )

        assert derive_preparation_kind([plain]) == "none"
        assert derive_preparation_kind([plain, prepared_set]) == "prepared_set"
        assert derive_preparation_kind([plain, prepared_merge]) == "prepared_merge"

    def test_transport_rejects_mixed_typed_kinds_defensively(self) -> None:
        prepared_merge = self._set_input(
            "dataset://a",
            pa.table({"id": [1]}),
            preparation=PreparationProvenance(kind="prepared_merge"),
        )
        prepared_set = self._set_input(
            "dataset://b",
            pa.table({"id": [2]}),
            preparation=PreparationProvenance(kind="prepared_set"),
        )

        with pytest.raises(ValueError):
            derive_preparation_kind([prepared_set, prepared_merge])
