"""Typed output shaping and finalization."""

from __future__ import annotations

import pyarrow as pa
import pytest
from unittest.mock import patch

from limulus.execution.contracts import (
    TypedOutputFinalizeRequest,
    TypedOutputPlanRequest,
    TypedOutputShapingRequest,
)
from limulus.execution.output_handoff import ExecutionOutputHandoffService
from limulus.execution.rewrites import TypedHelperPlan
from limulus.models import (
    DataSetRef,
    SourceMetadataLineage,
    SourceMetadataSnapshot,
)
from limulus.syntax import parse_program


def _helper_plan(*, runtime_columns: tuple[str, ...] = ()) -> TypedHelperPlan:
    return TypedHelperPlan(
        merge_mode="none",
        lag_lead_mode="none",
        where_mode="none",
        helper_source_columns=(),
        helper_runtime_columns=runtime_columns,
        lag_lead_specs=(),
        pre_evaluation_targets=(),
    )


def _context(
    dsl: str,
    table: pa.Table,
    *,
    runtime_columns: tuple[str, ...] = (),
):
    parsed = parse_program(dsl)
    assert parsed.program is not None, parsed.diagnostics
    assert not parsed.has_errors, parsed.diagnostics
    block = parsed.program.blocks[0]
    service = ExecutionOutputHandoffService()
    target_plan = service.plan_typed(
        TypedOutputPlanRequest(block=block, source=parsed.program.source)
    )
    source = DataSetRef(
        kind="arrow_table",
        location="dataset://source",
        payload=table,
    )
    inputs = {"source": source}
    helper_plan = _helper_plan(runtime_columns=runtime_columns)
    shaped = service.plan_typed_shaping(
        TypedOutputShapingRequest(
            block=block,
            source=parsed.program.source,
            output_target_plan=target_plan,
            resolved_inputs=inputs,
            helper_plan=helper_plan,
        )
    )
    lineage = SourceMetadataLineage(
        ordered_sources=(
            SourceMetadataSnapshot(
                source_ordinal=0,
                source_name="source",
                schema=table.schema,
                dataset_ref_metadata={"origin": "source"},
            ),
        )
    )
    return service, block, target_plan, shaped, helper_plan, inputs, lineage


def _finalize(
    context,
    outputs: dict[str, DataSetRef],
):
    service, block, target_plan, shaped, helper_plan, inputs, lineage = context
    assert shaped.diagnostics == ()
    assert shaped.output_plan is not None
    return service.finalize_typed(
        TypedOutputFinalizeRequest(
            block=block,
            output_target_plan=target_plan,
            output_plan=shaped.output_plan,
            resolved_inputs=inputs,
            outputs=outputs,
            helper_plan=helper_plan,
            source_metadata_lineage=lineage,
        )
    )


def _ref(payload: pa.Table, target: str) -> DataSetRef:
    return DataSetRef(
        kind="arrow_table",
        location=f"dataset://{target}",
        payload=payload,
    )


def _source_table(*, rows: int = 2) -> pa.Table:
    return pa.table(
        [
            pa.array(list(range(1, rows + 1)), type=pa.int64()),
            pa.array([10 * value for value in range(1, rows + 1)], type=pa.int64()),
            pa.array([True] * rows, type=pa.bool_()),
            pa.array(["src"] * rows, type=pa.string()),
            pa.array([False] * rows, type=pa.bool_()),
            pa.array([99] * rows, type=pa.int64()),
        ],
        schema=pa.schema(
            [
                pa.field("id", pa.int64(), metadata={b"label": b"Source ID"}),
                pa.field("amount", pa.int64(), metadata={b"label": b"Source amount"}),
                pa.field("present", pa.bool_()),
                pa.field("source_name", pa.string()),
                pa.field("done", pa.bool_()),
                pa.field("_lag_helper", pa.int64()),
            ],
            metadata={b"memlabel": b"Source"},
        ),
    )


def test_typed_shaping_and_finalize_use_shared_data_and_output_policy() -> None:
    table = _source_table()
    common = (
        "set source(in=present) indsname=source_name end=done; by id; "
        "keep id amount; rename id=key amount=base_amount; "
        "label id='Identifier' amount='Base amount'; run;"
    )
    data_context = _context(
        "data out(keep=key base_amount rename=(base_amount=final_amount) "
        "label='shared'); " + common,
        table,
        runtime_columns=("_lag_helper",),
    )
    output_context = _context(
        "data out; "
        "set source(in=present) indsname=source_name end=done; by id; "
        "keep id amount; rename id=key amount=base_amount; "
        "label id='Identifier' amount='Base amount'; "
        "output out(keep=key base_amount rename=(base_amount=final_amount) "
        "label='shared'); run;",
        table,
        runtime_columns=("_lag_helper",),
    )

    data_plan = data_context[3].output_plan
    output_plan = output_context[3].output_plan
    assert data_plan is not None and output_plan is not None
    assert data_plan.projected_columns_by_target == output_plan.projected_columns_by_target
    assert data_plan.type_expectations_by_target == output_plan.type_expectations_by_target
    assert data_plan.projected_columns_by_target["out"] == ("key", "final_amount")
    assert data_plan.type_expectations_by_target["out"] == {
        "key": "int64",
        "final_amount": "int64",
    }

    runtime_output = pa.table(
        [pa.array([1, 2]), pa.array([10, 20]), pa.array([True, True]),
         pa.array(["src", "src"]), pa.array([False, False]), pa.array([99, 99]),
         pa.array([True, False]), pa.array([False, True])],
        names=[
            "key", "final_amount", "present", "source_name", "done",
            "_lag_helper", "FIRST.id", "LAST.id",
        ],
    )
    data_result = _finalize(data_context, {"out": _ref(runtime_output, "out")})
    output_result = _finalize(output_context, {"out": _ref(runtime_output, "out")})
    for result in (data_result, output_result):
        assert result.diagnostics == ()
        output = result.outputs["out"]
        assert output.payload.column_names == ["key", "final_amount"]
        assert output.metadata["memlabel"] == "shared"
        assert output.payload.schema.metadata[b"memlabel"] == b"shared"
        assert output.payload.schema.field("key").metadata[b"label"] == b"Identifier"
        assert output.payload.schema.field("final_amount").metadata[b"label"] == b"Base amount"
    assert data_result.outputs["out"].payload.schema == output_result.outputs["out"].payload.schema
    assert data_result.outputs["out"].metadata == output_result.outputs["out"].metadata


def test_typed_finalize_projects_each_target_schema_even_when_empty() -> None:
    table = _source_table(rows=0)
    context = _context(
        "data left(keep=id label='Left') right(keep=amount label='Right'); "
        "set source; run;",
        table,
    )
    result = _finalize(
        context,
        {
            "left": _ref(pa.table({"id": pa.array([], type=pa.int64())}), "left"),
            "right": _ref(pa.table({"amount": pa.array([], type=pa.int64())}), "right"),
        },
    )

    assert result.diagnostics == ()
    assert result.outputs["left"].payload.column_names == ["id"]
    assert result.outputs["right"].payload.column_names == ["amount"]
    assert result.outputs["left"].payload.schema.field("id").type == pa.int64()
    assert result.outputs["right"].payload.schema.field("amount").type == pa.int64()
    assert result.outputs["left"].metadata["memlabel"] == "Left"
    assert result.outputs["right"].metadata["memlabel"] == "Right"


@pytest.mark.parametrize(
    ("retain", "expected"),
    (
        ("retain a .", ("id", "a")),
        ("retain a 1e3", ("id", "a")),
        ("retain a 'literal'", ("id", "a")),
        ("retain a b", ("id", "a", "b")),
    ),
    ids=("missing-initial-value", "exponent-initial-value", "quoted-initial-value", "variables"),
)
def test_typed_step_columns_keep_only_retain_variables(
    retain: str, expected: tuple[str, ...]
) -> None:
    context = _context(
        f"data out; set source; {retain}; run;",
        pa.table({"id": [1]}),
    )
    assert context[3].output_plan is not None
    assert context[3].output_plan.projected_columns_by_target["out"] == expected


def test_unreachable_inline_options_are_shared_without_python_row_filtering() -> None:
    table = _source_table()
    context = _context(
        "data out; set source; "
        "if 0 then output out(keep=id where=(id > 1) firstobs=2 obs=3 label='Inline'); "
        "run;",
        table,
    )
    target_plan = context[2]
    options = target_plan.effective_output_options["OUT"]
    assert options.keep_vars == ("id",)
    assert options.firstobs == 2
    assert options.obs == 3
    assert options.label == "Inline"
    assert options.where_expr is not None

    result = _finalize(
        context,
        {"out": _ref(pa.table({"id": [1, 2]}), "out")},
    )
    assert result.diagnostics == ()
    assert result.outputs["out"].payload.to_pylist() == [{"id": 1}, {"id": 2}]
    assert result.outputs["out"].metadata["memlabel"] == "Inline"


def test_invalid_confirmed_policy_does_not_create_a_successful_shape_plan() -> None:
    parsed = parse_program(
        "data out(keep=id); set source; output out(keep=amount); run;"
    )
    assert parsed.program is not None
    block = parsed.program.blocks[0]
    service = ExecutionOutputHandoffService()
    policy = service.plan_typed(
        TypedOutputPlanRequest(block=block, source=parsed.program.source)
    )
    result = service.plan_typed_shaping(
        TypedOutputShapingRequest(
            block=block,
            source=parsed.program.source,
            output_target_plan=policy,
            resolved_inputs={"source": DataSetRef("memory", "source", [])},
            helper_plan=_helper_plan(),
        )
    )

    assert [diagnostic.code for diagnostic in result.diagnostics] == [
        "RUNTIME_DATASET_OPTION_INVALID"
    ]
    assert result.output_plan is None


def test_duplicate_data_targets_are_rejected_before_typed_shaping() -> None:
    parsed = parse_program("data out OUT; set source; run;")
    assert parsed.program is not None
    block = parsed.program.blocks[0]
    service = ExecutionOutputHandoffService()
    policy = service.plan_typed(
        TypedOutputPlanRequest(block=block, source=parsed.program.source)
    )
    result = service.plan_typed_shaping(
        TypedOutputShapingRequest(
            block=block,
            source=parsed.program.source,
            output_target_plan=policy,
            resolved_inputs={"source": DataSetRef("memory", "source", [])},
            helper_plan=_helper_plan(),
        )
    )

    assert [diagnostic.code for diagnostic in result.diagnostics] == [
        "VALIDATE_DUPLICATE_DATA_TARGET"
    ]
    assert result.output_plan is None


@pytest.mark.parametrize(
    "dsl",
    (
        "data out(keep=a b rename=(a=x b=x)); set source; run;",
        "data out; set source; output out(keep=a b rename=(a=x b=x)); run;",
    ),
    ids=("data-option", "direct-output-option"),
)
def test_data_and_output_rename_collisions_share_typed_validation(
    dsl: str,
) -> None:
    parsed = parse_program(dsl)
    assert parsed.program is not None
    block = parsed.program.blocks[0]
    service = ExecutionOutputHandoffService()
    policy = service.plan_typed(
        TypedOutputPlanRequest(block=block, source=parsed.program.source)
    )
    source = DataSetRef(
        kind="arrow_table",
        location="dataset://source",
        payload=pa.table({"a": [1], "b": [2]}),
    )
    result = service.plan_typed_shaping(
        TypedOutputShapingRequest(
            block=block,
            source=parsed.program.source,
            output_target_plan=policy,
            resolved_inputs={"source": source},
            helper_plan=_helper_plan(),
        )
    )

    assert [diagnostic.code for diagnostic in result.diagnostics] == [
        "RUNTIME_DATASET_OPTION_INVALID"
    ]
    assert result.diagnostics[0].stage == "validate"
    assert result.diagnostics[0].span is not None
    assert result.output_plan is None


def test_typed_shape_and_finalize_do_not_read_legacy_display_or_collectors() -> None:
    table = pa.table({"id": [1]})
    parsed = parse_program("data out; set source; keep id; output out(keep=id); run;")
    assert parsed.program is not None
    block = parsed.program.blocks[0]
    service = ExecutionOutputHandoffService()
    target_plan = service.plan_typed(
        TypedOutputPlanRequest(block=block, source=parsed.program.source)
    )
    source = DataSetRef("arrow_table", "dataset://source", table)
    inputs = {"source": source}
    helper = _helper_plan()
    # The legacy text collectors were deleted; the typed path must not
    # reintroduce them, and the surviving internal-name collector stays
    # unused for this keep-only block.
    assert not hasattr(service, "_extract_variable_list")
    assert not hasattr(service, "_collect_output_target_option_specs")
    assert not hasattr(service, "_collect_step_defined_columns_ordered")
    with patch.object(
        service,
        "_collect_internal_output_variable_names",
        side_effect=AssertionError("legacy internal names"),
    ):
        shaped = service.plan_typed_shaping(
            TypedOutputShapingRequest(
                block=block,
                source=parsed.program.source,
                output_target_plan=target_plan,
                resolved_inputs=inputs,
                helper_plan=helper,
            )
        )
        assert shaped.output_plan is not None
        result = service.finalize_typed(
            TypedOutputFinalizeRequest(
                block=block,
                output_target_plan=target_plan,
                output_plan=shaped.output_plan,
                resolved_inputs=inputs,
                outputs={"out": _ref(table, "out")},
                helper_plan=helper,
            )
        )

    assert result.diagnostics == ()
    payload = result.outputs["out"].payload
    assert payload is not None
    assert payload.to_pylist() == [{"id": 1}]


def test_typed_shaping_fail_closes_on_duplicate_rename_target() -> None:
    """A rename-created duplicate column is rejected at the shaping stage."""

    table = pa.table({"a": [1], "b": [2]})
    parsed = parse_program("data out; set source; rename a=b; run;")
    assert parsed.program is not None
    block = parsed.program.blocks[0]
    service = ExecutionOutputHandoffService()
    target_plan = service.plan_typed(
        TypedOutputPlanRequest(block=block, source=parsed.program.source)
    )
    shaped = service.plan_typed_shaping(
        TypedOutputShapingRequest(
            block=block,
            source=parsed.program.source,
            output_target_plan=target_plan,
            resolved_inputs={"source": DataSetRef("arrow_table", "dataset://source", table)},
            helper_plan=_helper_plan(),
        )
    )

    assert shaped.output_plan is None
    assert [diagnostic.code for diagnostic in shaped.diagnostics] == [
        "RUNTIME_RENAME_STATEMENT_INVALID"
    ]
