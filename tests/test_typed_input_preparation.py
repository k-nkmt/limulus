from __future__ import annotations

from unittest.mock import patch

import pyarrow as pa

from limulus.expression_semantics import ExpressionSemantics
from limulus.execution.contracts import (
    TypedInputResolutionRequest,
    TypedInputPreparationRequest,
)
from limulus.execution.input_preparation import ExecutionInputPreparationService
from limulus.execution.planning import ExecutionPlanService
from limulus.models import DataSetRef
from limulus.syntax import parse_program
from limulus.syntax.models import (
    DatasetOptionSyntax,
    SourceStatementSyntax,
)


def _typed_block(source_text: str):
    parsed = parse_program(source_text)
    assert parsed.program is not None
    assert parsed.has_errors is False
    assert len(parsed.program.blocks) == 1
    return parsed.program.blocks[0]


def test_prepare_typed_applies_source_options_by_and_helper_carrier() -> None:
    source_text = (
        "data out; set source(keep=id amount group drop=drop_me "
        "where=(id >= 2) rename=(amount=score) firstobs=2 obs=1); "
        "by group; value=lag(score, 1, 99); "
        "runtime=lag(score, 1, score + 1); run;"
    )
    block = _typed_block(source_text)
    helper_plan = ExecutionPlanService().plan_typed_helpers(block)
    assert tuple(spec.is_source_owned for spec in helper_plan.lag_lead_specs) == (
        True,
        False,
    )
    source_ref = DataSetRef(
        kind="arrow_table",
        location="dataset://source",
        payload=pa.table(
            {
                "id": [1, 2, 3, 4],
                "amount": [10, 20, 30, 40],
                "group": ["z", "b", "a", "a"],
                "drop_me": [100, 200, 300, 400],
            }
        ).replace_schema_metadata({b"origin": b"source"}),
        metadata={"origin": "typed-input"},
    )
    service = ExecutionInputPreparationService()
    resolution = service.resolve_typed(
        TypedInputResolutionRequest(
            block=block,
            explicit_inputs={"source": source_ref},
        )
    )
    assert resolution.diagnostics == ()

    with (
        patch.object(
            ExpressionSemantics,
            "parse_expression",
            side_effect=AssertionError("typed input preparation parsed source text"),
        ),
        patch.object(
            ExpressionSemantics,
            "compile_expression",
            side_effect=AssertionError("typed input preparation compiled source text"),
        ),
        patch.object(
            service,
            "_extract_variable_list",
            side_effect=AssertionError("typed input preparation read BY display text"),
        ),
        patch("ast.parse", side_effect=AssertionError("typed input preparation reparsed source")),
        patch("builtins.compile", side_effect=AssertionError("typed input preparation compiled source")),
    ):
        typed_result = service.prepare_typed(
            TypedInputPreparationRequest(
                block=block,
                helper_plan=helper_plan,
                input_resolution=resolution,
            )
        )

    assert typed_result.diagnostics == ()
    assert typed_result.pre_evaluation_targets == helper_plan.pre_evaluation_targets
    assert typed_result.prepared.source_metadata_lineage == resolution.source_metadata_lineage
    typed_ref = typed_result.prepared.inputs["source"]
    assert isinstance(typed_ref.payload, pa.Table)
    assert typed_ref.location == "dataset://source"
    assert typed_ref.metadata == {"origin": "typed-input"}
    assert typed_ref.preparation.kind == "prepared_set"
    assert helper_plan.lag_lead_specs[0].helper_column in typed_ref.preparation.internal_variable_names
    assert helper_plan.lag_lead_specs[1].helper_column not in typed_ref.preparation.internal_variable_names
    assert typed_ref.payload.column("id").to_pylist() == [4]
    assert typed_ref.payload.column("score").to_pylist() == [40]
    assert typed_ref.payload.column("group").to_pylist() == ["a"]
    assert typed_ref.payload.column_names == [
        "id",
        "score",
        "group",
        "FIRST.group",
        "LAST.group",
        "first.group",
        "last.group",
        helper_plan.lag_lead_specs[0].helper_column,
    ]
    assert typed_ref.payload.column(helper_plan.lag_lead_specs[0].helper_column).to_pylist() == [
        99
    ]
    assert helper_plan.lag_lead_specs[1].helper_column not in typed_ref.payload.column_names

    prepared_source = next(
        statement
        for statement in typed_result.block.statements
        if statement.kind in {"SET", "MERGE"}
    )
    assert isinstance(prepared_source.payload, SourceStatementSyntax)
    original_source = next(
        statement for statement in block.statements if statement.kind == "SET"
    )
    assert prepared_source.span == original_source.span
    assert prepared_source.origin == original_source.origin
    assert prepared_source.apply_references == original_source.apply_references
    assert prepared_source.payload.refs[0].span == original_source.payload.refs[0].span
    assert isinstance(prepared_source.payload, SourceStatementSyntax)
    assert tuple(reference.name for reference in prepared_source.payload.refs) == ("source",)
    assert prepared_source.payload.refs[0].options == DatasetOptionSyntax()
    assert prepared_source.legacy_display_span is None


def test_prepare_typed_merge_occurrences_and_metadata() -> None:
    source_text = (
        "data merged; merge left(in=left_seen) right(in=right_seen) "
        "end=last; by id; run;"
    )
    block = _typed_block(source_text)
    helper_plan = ExecutionPlanService().plan_typed_helpers(block)
    inputs = {
        "left": DataSetRef(
            kind="arrow_table",
            location="dataset://left",
            payload=pa.table({"id": [1, 2], "left_value": ["l1", "l2"]}),
            metadata={"origin": "left"},
        ),
        "right": DataSetRef(
            kind="arrow_table",
            location="dataset://right",
            payload=pa.table({"id": [1, 3], "right_value": ["r1", "r3"]}),
            metadata={"origin": "right"},
        ),
    }
    service = ExecutionInputPreparationService()
    resolution = service.resolve_typed(
        TypedInputResolutionRequest(
            block=block,
            explicit_inputs=inputs,
        )
    )
    typed_result = service.prepare_typed(
        TypedInputPreparationRequest(
            block=block,
            helper_plan=helper_plan,
            input_resolution=resolution,
        )
    )

    assert typed_result.diagnostics == ()
    ordered_sources = typed_result.prepared.source_metadata_lineage.ordered_sources
    assert [
        (snapshot.source_name, dict(snapshot.dataset_ref_metadata))
        for snapshot in ordered_sources
    ] == [
        ("left", {"origin": "left"}),
        ("right", {"origin": "right"}),
    ]
    typed_ref = typed_result.prepared.inputs["left"]
    assert isinstance(typed_ref.payload, pa.Table)
    assert typed_ref.preparation.internal_variable_names == (
        "FIRST.id",
        "LAST.id",
        "first.id",
        "last",
        "last.id",
        "left_seen",
        "right_seen",
    )
    assert typed_ref.metadata == {"origin": "left"}
    assert typed_ref.payload.column_names == [
        "id",
        "left_value",
        "right_value",
        "left_seen",
        "right_seen",
        "FIRST.id",
        "LAST.id",
        "first.id",
        "last.id",
    ]
    assert typed_ref.payload.to_pylist() == [
        {
            "id": 1,
            "left_value": "l1",
            "right_value": "r1",
            "left_seen": 1,
            "right_seen": 1,
            "FIRST.id": 1,
            "LAST.id": 1,
            "first.id": 1,
            "last.id": 1,
        },
        {
            "id": 2,
            "left_value": "l2",
            "right_value": None,
            "left_seen": 1,
            "right_seen": 0,
            "FIRST.id": 1,
            "LAST.id": 1,
            "first.id": 1,
            "last.id": 1,
        },
        {
            "id": 3,
            "left_value": None,
            "right_value": "r3",
            "left_seen": 0,
            "right_seen": 1,
            "FIRST.id": 1,
            "LAST.id": 1,
            "first.id": 1,
            "last.id": 1,
        },
    ]
    prepared_source = typed_result.block.statements[1]
    assert prepared_source.kind == "SET"
    assert isinstance(prepared_source.payload, SourceStatementSyntax)
    assert prepared_source.payload.refs[0].name == "left"
    original_source = block.statements[1]
    assert isinstance(original_source.payload, SourceStatementSyntax)
    assert prepared_source.span == original_source.span
    assert prepared_source.origin == original_source.origin
    assert prepared_source.apply_references == original_source.apply_references
    assert prepared_source.payload.refs[0].span == original_source.payload.refs[0].span
    assert prepared_source.payload.options.end_var == "last"


def test_typed_concat_default_remains_runtime_owned() -> None:
    block = _typed_block(
        'data out; set source; value=lag(score, 1, "a" || "b"); run;'
    )
    helper_plan = ExecutionPlanService().plan_typed_helpers(block)
    assert len(helper_plan.lag_lead_specs) == 1
    spec = helper_plan.lag_lead_specs[0]
    assert spec.is_source_owned is False

    service = ExecutionInputPreparationService()
    source_ref = DataSetRef(
        kind="arrow_table",
        location="dataset://source",
        payload=pa.table({"score": [3, 5]}),
    )
    resolution = service.resolve_typed(
        TypedInputResolutionRequest(
            block=block,
            explicit_inputs={"source": source_ref},
        )
    )
    result = service.prepare_typed(
        TypedInputPreparationRequest(
            block=block,
            helper_plan=helper_plan,
            input_resolution=resolution,
        )
    )

    assert result.diagnostics == ()
    prepared_payload = result.prepared.inputs["source"].payload
    assert isinstance(prepared_payload, pa.Table)
    assert spec.helper_column not in prepared_payload.column_names


def test_prepare_typed_preserves_empty_filtered_arrow_schema_and_metadata() -> None:
    block = _typed_block(
        "data out; set source(where=(amount > 99)); run;"
    )
    source_ref = DataSetRef(
        kind="arrow_table",
        location="dataset://source",
        payload=pa.table({"amount": [1, 2]}).replace_schema_metadata(
            {b"source": b"preserved"}
        ),
        metadata={"origin": "empty-result"},
    )
    service = ExecutionInputPreparationService()
    resolution = service.resolve_typed(
        TypedInputResolutionRequest(
            block=block,
            explicit_inputs={"source": source_ref},
        )
    )
    typed_result = service.prepare_typed(
        TypedInputPreparationRequest(
            block=block,
            helper_plan=ExecutionPlanService().plan_typed_helpers(block),
            input_resolution=resolution,
        )
    )

    assert typed_result.diagnostics == ()
    typed_payload = typed_result.prepared.inputs["source"].payload
    assert isinstance(typed_payload, pa.Table)
    assert typed_payload.num_rows == 0
    assert typed_payload.schema.metadata == {b"source": b"preserved"}
    assert typed_result.prepared.inputs["source"].metadata == {"origin": "empty-result"}


def test_prepare_typed_failure_returns_no_partial_carrier() -> None:
    block = _typed_block(
        "data out; set source(where=(missing >= 1)); value=lag(amount); run;"
    )
    service = ExecutionInputPreparationService()
    source_ref = DataSetRef(
        kind="arrow_table",
        location="dataset://source",
        payload=pa.table({"amount": [1, 2]}),
    )
    resolution = service.resolve_typed(
        TypedInputResolutionRequest(
            block=block,
            explicit_inputs={"source": source_ref},
        )
    )
    result = service.prepare_typed(
        TypedInputPreparationRequest(
            block=block,
            helper_plan=ExecutionPlanService().plan_typed_helpers(block),
            input_resolution=resolution,
        )
    )

    assert result.diagnostics
    assert result.diagnostics[0].code == "RUNTIME_DATASET_OPTION_INVALID"
    assert result.prepared.inputs == {}
    assert result.prepared.preparation_kind == "none"
