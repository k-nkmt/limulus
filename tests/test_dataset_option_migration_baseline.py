"""SET dataset-reference option regressions with self-contained Arrow fixtures."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

import pyarrow as pa
import pytest

from limulus.arrow_bridge import analyze_merge_schema_ownership, materialize_merge_carrier
from limulus.execution import DataStepExecutor
from limulus.execution.contracts import (
    InputPreparationRequest,
    InputResolutionResult,
)
from limulus.execution.rewrites import RewritePlan
from limulus.models import (
    DataSetRef,
    ExecuteRequest,
    NormalizedExecutionInputs,
    SourceMetadataLineage,
    SourceMetadataSnapshot,
)
from limulus.parser import ParserService


@dataclass(frozen=True)
class DatasetOptionBaselineCase:
    dsl: str
    input_table: pa.Table
    expected_rows: tuple[dict[str, Any], ...]
    expected_schema: pa.Schema
    expected_dataset_metadata: Mapping[bytes, bytes]
    expected_field_metadata: Mapping[str, Mapping[bytes, bytes]]


def _plan_for(executor: DataStepExecutor, statements, inputs):
    del executor, statements, inputs
    return RewritePlan(
        merge_mode="none",
        lag_lead_mode="none",
        where_mode="none",
        helper_source_columns=(),
        helper_runtime_columns=(),
    )


def _metadata_table(
    *,
    dataset_label: bytes,
    fields: tuple[tuple[str, pa.DataType, bytes], ...],
    rows: list[Mapping[str, Any]],
) -> pa.Table:
    schema = pa.schema(
        [pa.field(name, data_type, metadata={b"label": field_label}) for name, data_type, field_label in fields],
        metadata={b"memlabel": dataset_label},
    )
    return pa.Table.from_pylist(rows, schema=schema)


def _execute_arrow_case(
    dsl: str,
    inputs: Mapping[str, DataSetRef],
) -> tuple[DataStepExecutor, Any]:
    executor = DataStepExecutor(runtime_backend="rust")
    response = executor.execute(ExecuteRequest(dsl_text=dsl, inputs=inputs))
    return executor, response


def _serialize_typed_block(dsl: str) -> dict[str, object]:
    from limulus.syntax import parse_program
    from limulus.syntax.models import DatasetOptionSyntax
    from limulus.syntax.native_payload import serialize_block

    parsed = parse_program(dsl)
    assert parsed.program is not None
    return serialize_block(
        parsed.program.blocks[0],
        source=parsed.program.source,
        output_targets=("out",),
        default_target="out",
        suppress_implicit_output=False,
        effective_output_options={"out": DatasetOptionSyntax()},
    )


def test_merge_schema_ownership_analysis_is_deterministic_and_metadata_free() -> None:
    lineage = SourceMetadataLineage(
        ordered_sources=(
            SourceMetadataSnapshot(
                source_ordinal=0,
                source_name="left",
                schema=pa.schema(
                    [
                        pa.field("ID", pa.int64(), metadata={b"label": b"left id"}),
                        pa.field("left_only", pa.int32(), metadata={b"label": b"left"}),
                        pa.field("shared", pa.string(), metadata={b"label": b"left shared"}),
                        pa.field("FIRST.id", pa.int8(), metadata={b"label": b"runtime"}),
                    ],
                    metadata={b"memlabel": b"left"},
                ),
            ),
            SourceMetadataSnapshot(
                source_ordinal=1,
                source_name="right",
                schema=pa.schema(
                    [
                        pa.field("id", pa.int32(), metadata={b"label": b"right id"}),
                        pa.field("right_only", pa.timestamp("us"), metadata={b"label": b"right"}),
                        pa.field("SHARED", pa.string(), metadata={b"label": b"right shared"}),
                        pa.field("LAST.id", pa.int8(), metadata={b"label": b"runtime"}),
                    ],
                    metadata={b"memlabel": b"right"},
                ),
            ),
            SourceMetadataSnapshot(
                source_ordinal=2,
                source_name="third",
                schema=pa.schema(
                    [pa.field("third_only", pa.float64(), metadata={b"label": b"third"})],
                    metadata={b"memlabel": b"third"},
                ),
            ),
        )
    )

    ownership = analyze_merge_schema_ownership(lineage, by_keys=("id",))

    assert ownership.type_hints == {
        "ID": pa.int64(),
        "left_only": pa.int32(),
        "right_only": pa.timestamp("us"),
        "third_only": pa.float64(),
    }
    assert ownership.column_order == (
        "ID",
        "left_only",
        "shared",
        "right_only",
        "third_only",
    )
    assert ownership.ambiguous_non_by_fields == ("SHARED",)
    assert all(isinstance(dtype, pa.DataType) for dtype in ownership.type_hints.values())
    assert all(not hasattr(dtype, "metadata") for dtype in ownership.type_hints.values())
    with pytest.raises(TypeError):
        ownership.type_hints["new_field"] = pa.bool_()


def test_merge_carrier_preserves_value_bearing_column_type_and_buffer() -> None:
    source = pa.table({"value": pa.array([10, 20], type=pa.int64())})

    materialized = materialize_merge_carrier(
        source,
        type_hints={"value": pa.int32()},
        column_order=("value",),
    )

    assert materialized.schema.field("value").type == pa.int64()
    assert materialized.column("value").to_pylist() == [10, 20]
    assert materialized.column("value").chunk(0).buffers() == source.column("value").chunk(0).buffers()


@pytest.fixture(scope="module")
def dataset_option_baseline_case() -> DatasetOptionBaselineCase:
    """Return the one durable option-order/schema/metadata characterization case."""

    input_schema = pa.schema(
        [
            pa.field("id", pa.int64(), metadata={b"label": b"Identifier"}),
            pa.field("Amount", pa.int64(), metadata={b"label": b"Amount"}),
            pa.field("temp", pa.string(), metadata={b"label": b"Temporary"}),
        ],
        metadata={b"memlabel": b"Dataset Option Input"},
    )
    input_table = pa.Table.from_arrays(
        [
            pa.array([1, 2, 3, 4], type=pa.int64()),
            pa.array([10, 20, 30, 40], type=pa.int64()),
            pa.array(["discard-1", "discard-2", "discard-3", "discard-4"]),
        ],
        schema=input_schema,
    )

    # Keep the option spelling and order stable: KEEP -> DROP -> WHERE ->
    # RENAME -> FIRSTOBS/OBS.  WHERE and RENAME deliberately use a different
    # case from the input column to characterize case-insensitive resolution.
    dsl = (
        'data out(label="Dataset Option Output"); '
        "set inp(keep=id Amount temp drop=temp where=(amount >= 20) "
        "rename=(Amount=score) firstobs=2 obs=1); "
        'label id="Identifier" score="Amount"; '
        "output out; run;"
    )

    expected_schema = pa.schema(
        [
            pa.field("id", pa.int64(), metadata={b"label": b"Identifier"}),
            pa.field("score", pa.int64(), metadata={b"label": b"Amount"}),
        ],
        metadata={b"memlabel": b"Dataset Option Output"},
    )
    return DatasetOptionBaselineCase(
        dsl=dsl,
        input_table=input_table,
        expected_rows=(
            {"id": 3, "score": 30},
        ),
        expected_schema=expected_schema,
        expected_dataset_metadata={b"memlabel": b"Dataset Option Output"},
        expected_field_metadata={
            "id": {b"label": b"Identifier"},
            "score": {b"label": b"Amount"},
        },
    )


def test_dataset_option_order_arrow_result_and_metadata_baseline(
    dataset_option_baseline_case: DatasetOptionBaselineCase,
) -> None:
    case = dataset_option_baseline_case
    executor = DataStepExecutor(runtime_backend="rust")
    response = executor.execute(
        ExecuteRequest(
            dsl_text=case.dsl,
            inputs={
                "inp": DataSetRef(
                    kind="arrow_table",
                    location="dataset://inp",
                    payload=case.input_table,
                )
            },
        )
    )

    assert response.has_errors is False
    assert executor.last_runtime_backend == "rust"

    output = response.outputs_arrow["out"]
    assert isinstance(output, pa.Table)
    expected_output = pa.Table.from_pylist(
        list(case.expected_rows),
        schema=case.expected_schema,
    )
    assert output.equals(expected_output, check_metadata=True)
    assert output.schema.names == list(case.expected_schema.names)
    assert output.schema.types == case.expected_schema.types
    assert output.schema.metadata == dict(case.expected_dataset_metadata)
    for field_name, expected_metadata in case.expected_field_metadata.items():
        assert output.schema.field(field_name).metadata == dict(expected_metadata)


def test_multiple_set_uses_ordered_source_schema_and_metadata() -> None:
    source_a = _metadata_table(
        dataset_label=b"SET source A",
        fields=(
            ("id", pa.int64(), b"A id"),
            ("shared", pa.int64(), b"A shared"),
            ("a_only", pa.int32(), b"A only"),
        ),
        rows=[{"id": 1, "shared": 10, "a_only": 100}],
    )
    source_b = _metadata_table(
        dataset_label=b"SET source B",
        fields=(
            ("id", pa.int64(), b"B id"),
            ("shared", pa.int64(), b"B shared"),
            ("b_only", pa.string(), b"B only"),
        ),
        rows=[{"id": 2, "shared": 20, "b_only": "later"}],
    )

    executor, response = _execute_arrow_case(
        "data out; set a b; output out; run;",
        {
            "b": DataSetRef(kind="arrow_table", location="dataset://b", payload=source_b),
            "a": DataSetRef(kind="arrow_table", location="dataset://a", payload=source_a),
        },
    )

    assert response.has_errors is False
    assert executor.last_runtime_backend == "rust"
    output = response.outputs_arrow["out"]
    assert output.to_pylist() == [
        {"id": 1, "shared": 10, "a_only": 100, "b_only": None},
        {"id": 2, "shared": 20, "a_only": None, "b_only": "later"},
    ]
    assert output.schema.names == ["id", "shared", "a_only", "b_only"]
    assert output.schema.metadata == {b"memlabel": b"SET source A"}
    assert output.schema.field("id").metadata == {b"label": b"A id"}
    assert output.schema.field("shared").metadata == {b"label": b"A shared"}
    assert output.schema.field("a_only").metadata == {b"label": b"A only"}
    assert output.schema.field("b_only").metadata == {b"label": b"B only"}


def test_multiple_set_lineage_applies_ordered_metadata_after_source_options() -> None:
    source_a = _metadata_table(
        dataset_label=b"SET source A",
        fields=(
            ("id", pa.int64(), b"A id"),
            ("shared", pa.int64(), b"A shared"),
            ("a_only", pa.string(), b"A only"),
            ("drop_a", pa.string(), b"A drop"),
        ),
        rows=[{"id": 1, "shared": 10, "a_only": "a", "drop_a": "discard"}],
    )
    source_b = _metadata_table(
        dataset_label=b"SET source B",
        fields=(
            ("id", pa.int64(), b"B id"),
            ("shared", pa.int64(), b"B shared"),
            ("b_only", pa.string(), b"B only"),
            ("drop_b", pa.string(), b"B drop"),
        ),
        rows=[{"id": 2, "shared": 20, "b_only": "b", "drop_b": "discard"}],
    )
    executor, response = _execute_arrow_case(
        "data out; "
        "set b(keep=id shared b_only drop_b drop=drop_b) "
        "a(keep=id shared a_only drop_a drop=drop_a) "
        "b(keep=id shared b_only drop_b drop=drop_b rename=(b_only=renamed_b)); "
        "output out; run;",
        # Deliberately reverse the mapping order.  DSL source order is b, a, b.
        {
            "a": DataSetRef(kind="arrow_table", location="dataset://a", payload=source_a),
            "b": DataSetRef(kind="arrow_table", location="dataset://b", payload=source_b),
        },
    )

    assert response.has_errors is False
    output = response.outputs_arrow["out"]
    assert output.schema.metadata == {b"memlabel": b"SET source B"}
    assert output.schema.names == ["id", "shared", "b_only", "a_only", "renamed_b"]
    assert output.schema.field("id").metadata == {b"label": b"B id"}
    assert output.schema.field("shared").metadata == {b"label": b"B shared"}
    assert output.schema.field("b_only").metadata == {b"label": b"B only"}
    assert output.schema.field("a_only").metadata == {b"label": b"A only"}
    assert output.schema.field("renamed_b").metadata == {b"label": b"B only"}


def test_source_metadata_lineage_is_ordered_option_shaped_and_row_free() -> None:
    source_a = _metadata_table(
        dataset_label=b"lineage A",
        fields=(
            ("id", pa.int64(), b"A id"),
            ("score", pa.int64(), b"A score"),
            ("drop_me", pa.string(), b"A drop"),
        ),
        rows=[{"id": 1, "score": 10, "drop_me": "a"}],
    )
    source_b = _metadata_table(
        dataset_label=b"lineage B",
        fields=(
            ("id", pa.int64(), b"B id"),
            ("score", pa.int64(), b"B score"),
            ("drop_me", pa.string(), b"B drop"),
        ),
        rows=[{"id": 2, "score": 20, "drop_me": "b"}],
    )
    executor = DataStepExecutor(runtime_backend="rust")
    dsl = (
        "data out; "
        "set b(keep=id score rename=(score=first_score)) "
        "a(drop=drop_me) "
        "b(keep=id score rename=(score=second_score)); "
        "output out; run;"
    )
    parse_result = ParserService().parse(dsl)
    assert parse_result.has_errors is False

    inputs = {
            "a": DataSetRef(
                kind="arrow_table",
                location="dataset://a",
                payload=source_a,
                metadata={"origin": "a"},
            ),
            "b": DataSetRef(
                kind="arrow_table",
                location="dataset://b",
                payload=source_b,
                metadata={"origin": "b"},
            ),
        }
    rewrite_plan = _plan_for(executor, parse_result.ast.statements, inputs)
    preparation_result = executor._input_preparation_service.prepare(
        InputPreparationRequest(
            ast_statements=tuple(parse_result.ast.statements),
            input_resolution=InputResolutionResult(
                resolved_inputs=NormalizedExecutionInputs(inputs=inputs),
            ),
            rewrite_plan=rewrite_plan,
        )
    )
    prepared = dict(preparation_result.prepared.inputs)
    diagnostics = list(preparation_result.diagnostics)

    assert diagnostics == []
    assert prepared["b"].kind == "arrow_table"
    lineage = preparation_result.prepared.source_metadata_lineage
    assert [(item.source_ordinal, item.source_name) for item in lineage.ordered_sources] == [
        (0, "b"),
        (1, "a"),
        (2, "b"),
    ]
    assert [tuple(item.schema.names) for item in lineage.ordered_sources] == [
        ("id", "first_score"),
        ("id", "score"),
        ("id", "second_score"),
    ]
    assert [dict(item.dataset_ref_metadata) for item in lineage.ordered_sources] == [
        {"origin": "b"},
        {"origin": "a"},
        {"origin": "b"},
    ]
    for item in lineage.ordered_sources:
        assert not hasattr(item, "table")
        assert not hasattr(item, "rows")
        assert not hasattr(item, "payload")


def test_source_metadata_lineage_reaches_output_handoff_but_not_typed_payload() -> None:
    source = _metadata_table(
        dataset_label=b"handoff source",
        fields=(("id", pa.int64(), b"id"),),
        rows=[{"id": 1}],
    )
    executor, response = _execute_arrow_case(
        "data out; set source; output out; run;",
        {
            "source": DataSetRef(
                kind="arrow_table",
                location="dataset://source",
                payload=source,
                metadata={"origin": "source"},
            )
        },
    )
    assert response.has_errors is False
    assert [item.source_name for item in executor.last_output_source_metadata_lineage.ordered_sources] == [
        "source"
    ]

    serialized = _serialize_typed_block("data out; set source; output out; run;")
    assert "source_metadata_lineage" not in serialized


def test_merge_source_metadata_lineage_preserves_dsl_order_through_output_handoff() -> None:
    source_a = _metadata_table(
        dataset_label=b"MERGE lineage A",
        fields=(
            ("id", pa.int64(), b"A id"),
            ("left", pa.int64(), b"A left"),
            ("drop_a", pa.string(), b"A drop"),
        ),
        rows=[{"id": 1, "left": 10, "drop_a": "a"}],
    )
    source_b = _metadata_table(
        dataset_label=b"MERGE lineage B",
        fields=(
            ("id", pa.int64(), b"B id"),
            ("right", pa.string(), b"B right"),
            ("drop_b", pa.string(), b"B drop"),
        ),
        rows=[{"id": 1, "right": "one", "drop_b": "b"}],
    )
    dsl = (
        "data out; "
        "merge a(keep=id left drop_a drop=drop_a rename=(left=left_value)) "
        "b(keep=id right drop_b drop=drop_b rename=(right=right_value)); "
        "by id; output out; run;"
    )
    executor, response = _execute_arrow_case(
        dsl,
        {
            "b": DataSetRef(
                kind="arrow_table",
                location="dataset://b",
                payload=source_b,
                metadata={"origin": "b"},
            ),
            "a": DataSetRef(
                kind="arrow_table",
                location="dataset://a",
                payload=source_a,
                metadata={"origin": "a"},
            ),
        },
    )

    assert response.has_errors is False
    lineage = executor.last_output_source_metadata_lineage
    assert [(item.source_ordinal, item.source_name) for item in lineage.ordered_sources] == [
        (0, "a"),
        (1, "b"),
    ]
    assert [tuple(item.schema.names) for item in lineage.ordered_sources] == [
        ("id", "left_value"),
        ("id", "right_value"),
    ]
    assert [dict(item.dataset_ref_metadata) for item in lineage.ordered_sources] == [
        {"origin": "a"},
        {"origin": "b"},
    ]

    prepared_inputs = executor.last_prepared_execution_inputs.inputs
    assert tuple(prepared_inputs) == ("a",)
    prepared_a = prepared_inputs["a"]
    assert prepared_a.preparation.kind == "prepared_merge"
    assert prepared_a.location == "dataset://a"
    assert prepared_a.payload is not None
    assert prepared_a.payload.schema.names == [
        "id",
        "left_value",
        "right_value",
        "FIRST.id",
        "LAST.id",
        "first.id",
        "last.id",
    ]
    assert prepared_a.payload.to_pylist() == [
        {
            "id": 1,
            "left_value": 10,
            "right_value": "one",
            "FIRST.id": 1,
            "LAST.id": 1,
            "first.id": 1,
            "last.id": 1,
        }
    ]
    output = response.outputs_arrow["out"]
    assert output.to_pylist() == [
        {"id": 1, "left_value": 10, "right_value": "one"}
    ]
    assert output.schema.field("left_value").metadata == {b"label": b"A left"}
    assert output.schema.field("right_value").metadata == {b"label": b"B right"}
    serialized = _serialize_typed_block(dsl)
    assert "source_metadata_lineage" not in serialized


def test_merge_without_by_prepared_handoff_uses_single_carrier_and_lineage() -> None:
    source_a = _metadata_table(
        dataset_label=b"MERGE ordinal A",
        fields=(("left", pa.int64(), b"A left"),),
        rows=[{"left": 10}, {"left": 20}],
    )
    source_b = _metadata_table(
        dataset_label=b"MERGE ordinal B",
        fields=(("right", pa.int64(), b"B right"),),
        rows=[{"right": 100}],
    )
    executor, response = _execute_arrow_case(
        "data out; merge a b; output out; run;",
        {
            "b": DataSetRef(
                kind="arrow_table",
                location="dataset://b",
                payload=source_b,
                metadata={"origin": "b"},
            ),
            "a": DataSetRef(
                kind="arrow_table",
                location="dataset://a",
                payload=source_a,
                metadata={"origin": "a"},
            ),
        },
    )

    assert response.has_errors is False
    assert executor.last_runtime_backend == "rust"
    output = response.outputs_arrow["out"]
    assert output.to_pylist() == [
        {"left": 10, "right": 100},
        {"left": 20, "right": None},
    ]
    assert output.schema.names == ["left", "right"]
    assert output.schema.metadata == {b"memlabel": b"MERGE ordinal A"}
    assert output.schema.field("left").metadata == {b"label": b"A left"}
    assert output.schema.field("right").metadata == {b"label": b"B right"}

    prepared = executor.last_prepared_execution_inputs
    assert prepared.preparation_kind == "prepared_merge"
    assert tuple(prepared.inputs) == ("a",)
    assert prepared.inputs["a"].preparation.kind == "prepared_merge"
    assert prepared.inputs["a"].location == "dataset://a"
    assert prepared.inputs["a"].payload.schema.names == ["left", "right"]
    assert [item.source_name for item in prepared.source_metadata_lineage.ordered_sources] == [
        "a",
        "b",
    ]
    assert [dict(item.dataset_ref_metadata) for item in prepared.source_metadata_lineage.ordered_sources] == [
        {"origin": "a"},
        {"origin": "b"},
    ]

    serialized = _serialize_typed_block("data out; set a; output out; run;")
    assert "source_metadata_lineage" not in serialized


def test_multiple_set_rejects_incompatible_schema_type_without_fallback() -> None:
    source_a = _metadata_table(
        dataset_label=b"SET numeric",
        fields=(("id", pa.int64(), b"A id"), ("value", pa.int64(), b"A value")),
        rows=[{"id": 1, "value": 10}],
    )
    source_b = _metadata_table(
        dataset_label=b"SET text",
        fields=(("id", pa.int64(), b"B id"), ("value", pa.string(), b"B value")),
        rows=[{"id": 2, "value": "ten"}],
    )

    executor, response = _execute_arrow_case(
        "data out; set a b; output out; run;",
        {
            "b": DataSetRef(kind="arrow_table", location="dataset://b", payload=source_b),
            "a": DataSetRef(kind="arrow_table", location="dataset://a", payload=source_a),
        },
    )

    assert response.has_errors is True
    diagnostic = next(diagnostic for diagnostic in response.diagnostics if diagnostic.severity == "error")
    assert diagnostic.code == "RUNTIME_SET_SCHEMA_INCOMPATIBLE"
    diagnostic_text = f"{diagnostic.location} {diagnostic.message}".lower()
    assert re.search(r"(?<![a-z0-9_])a(?![a-z0-9_])", diagnostic_text)
    assert re.search(r"(?<![a-z0-9_])b(?![a-z0-9_])", diagnostic_text)
    assert "value" in diagnostic_text
    assert "int64" in diagnostic_text and "string" in diagnostic_text
    assert response.outputs_arrow == {}
    assert executor.last_runtime_backend is None


def test_multiple_set_rejects_incompatible_schema_with_null_only_source() -> None:
    source_a = _metadata_table(
        dataset_label=b"SET numeric",
        fields=(("id", pa.int64(), b"A id"), ("value", pa.int64(), b"A value")),
        rows=[{"id": 1, "value": 10}],
    )
    source_b = _metadata_table(
        dataset_label=b"SET text",
        fields=(("id", pa.int64(), b"B id"), ("value", pa.string(), b"B value")),
        rows=[{"id": 2, "value": "ten"}],
    )
    source_c = _metadata_table(
        dataset_label=b"SET null",
        fields=(("id", pa.int64(), b"C id"), ("value", pa.null(), b"C value")),
        rows=[{"id": 3, "value": None}],
    )

    executor, response = _execute_arrow_case(
        "data out; set a b c; output out; run;",
        {
            "a": DataSetRef(kind="arrow_table", location="dataset://a", payload=source_a),
            "b": DataSetRef(kind="arrow_table", location="dataset://b", payload=source_b),
            "c": DataSetRef(kind="arrow_table", location="dataset://c", payload=source_c),
        },
    )

    assert response.has_errors is True
    diagnostic = next(diagnostic for diagnostic in response.diagnostics if diagnostic.severity == "error")
    assert diagnostic.code == "RUNTIME_SET_SCHEMA_INCOMPATIBLE"
    diagnostic_text = f"{diagnostic.location} {diagnostic.message}".lower()
    for source_name in ("a", "b", "c"):
        assert re.search(rf"(?<![a-z0-9_]){source_name}(?![a-z0-9_])", diagnostic_text)
    assert "value" in diagnostic_text
    assert "int64" in diagnostic_text and "string" in diagnostic_text
    assert response.outputs_arrow == {}
    assert executor.last_runtime_backend is None


def test_merge_uses_ordered_source_and_column_owner_metadata() -> None:
    source_a = _metadata_table(
        dataset_label=b"MERGE source A",
        fields=(
            ("id", pa.int64(), b"A id"),
            ("left_value", pa.int64(), b"A left"),
        ),
        rows=[{"id": 1, "left_value": 10}],
    )
    source_b = _metadata_table(
        dataset_label=b"MERGE source B",
        fields=(
            ("id", pa.int64(), b"B id"),
            ("right_value", pa.string(), b"B right"),
        ),
        rows=[{"id": 2, "right_value": "later"}],
    )

    executor, response = _execute_arrow_case(
        "data out; merge a b; by id; output out; run;",
        {
            "b": DataSetRef(kind="arrow_table", location="dataset://b", payload=source_b),
            "a": DataSetRef(kind="arrow_table", location="dataset://a", payload=source_a),
        },
    )

    assert response.has_errors is False
    assert executor.last_runtime_backend == "rust"
    output = response.outputs_arrow["out"]
    assert output.to_pylist() == [
        {"id": 1, "left_value": 10, "right_value": None},
        {"id": 2, "left_value": None, "right_value": "later"},
    ]
    assert output.schema.names == ["id", "left_value", "right_value"]
    assert output.schema.metadata == {b"memlabel": b"MERGE source A"}
    assert output.schema.field("id").metadata == {b"label": b"A id"}
    assert output.schema.field("left_value").metadata == {b"label": b"A left"}
    assert output.schema.field("right_value").metadata == {b"label": b"B right"}


@pytest.mark.parametrize("has_by", [True, False], ids=["by", "without-by"])
@pytest.mark.parametrize("empty", [True, False], ids=["zero-rows", "all-null"])
def test_merge_preserves_source_types_for_empty_and_all_null_fields(
    has_by: bool,
    empty: bool,
) -> None:
    """Characterize source-owned types before metadata restoration is involved."""

    row_count = 0 if empty else 1
    source_a_columns: dict[str, pa.Array] = {
        "amount": pa.array([None] * row_count, type=pa.int32()),
    }
    source_b_columns: dict[str, pa.Array] = {
        "name": pa.array([None] * row_count, type=pa.string()),
        "event_time": pa.array([None] * row_count, type=pa.timestamp("us")),
    }
    if has_by:
        key_values = pa.array([] if empty else [1], type=pa.int64())
        source_a_columns = {"id": key_values, **source_a_columns}
        source_b_columns = {"id": key_values, **source_b_columns}

    source_a = pa.table(source_a_columns)
    source_b = pa.table(source_b_columns)
    by_clause = " by id;" if has_by else ""
    executor, response = _execute_arrow_case(
        f"data out; merge a b;{by_clause} output out; run;",
        {
            "a": DataSetRef(kind="arrow_table", location="dataset://a", payload=source_a),
            "b": DataSetRef(kind="arrow_table", location="dataset://b", payload=source_b),
        },
    )

    assert response.has_errors is False
    assert executor.last_runtime_backend == "rust"
    output = response.outputs_arrow["out"]
    expected_names = ["id", "amount", "name", "event_time"] if has_by else [
        "amount",
        "name",
        "event_time",
    ]
    expected_types = [pa.int64(), pa.int32(), pa.string(), pa.timestamp("us")] if has_by else [
        pa.int32(),
        pa.string(),
        pa.timestamp("us"),
    ]
    assert output.schema.names == expected_names
    assert output.schema.types == expected_types
    if empty:
        assert output.to_pylist() == []
    elif has_by:
        assert output.to_pylist() == [
            {"id": 1, "amount": None, "name": None, "event_time": None}
        ]
    else:
        assert output.to_pylist() == [
            {"amount": None, "name": None, "event_time": None}
        ]


def test_merge_without_by_applies_options_per_occurrence_and_aligns_ordinals() -> None:
    """Keep the no-BY option contract independent from source metadata tests."""

    source = pa.table(
        {
            "amount": pa.array([10, 20, 30, 40], type=pa.int64()),
            "discard": pa.array(["drop-1", "drop-2", "drop-3", "drop-4"]),
            "text": pa.array(["a", "b", "c", "d"]),
        }
    )
    executor, response = _execute_arrow_case(
        "data out; "
        "merge "
        "source(keep=amount discard drop=discard where=(amount >= 20) "
        "rename=(amount=left_amount) firstobs=1 obs=2 in=left_seen) "
        "source(keep=text rename=(text=right_text) firstobs=2 obs=1 in=right_seen); "
        "source_presence = left_seen + right_seen; "
        "keep left_amount right_text source_presence; "
        "output out; run;",
        {"source": DataSetRef(kind="arrow_table", location="dataset://source", payload=source)},
    )

    assert response.has_errors is False
    assert executor.last_runtime_backend == "rust"
    output = response.outputs_arrow["out"]
    assert output.schema.names == ["left_amount", "right_text", "source_presence"]
    assert output.schema.types == [pa.int64(), pa.string(), pa.float64()]
    assert output.to_pylist() == [
        {"left_amount": 20, "right_text": "b", "source_presence": 2.0},
        {"left_amount": 30, "right_text": None, "source_presence": 1.0},
    ]


def test_merge_by_preserves_by_first_column_order_when_source_key_is_not_first() -> None:
    source_left = pa.table(
        {
            "value": pa.array([10], type=pa.int32()),
            "id": pa.array([1], type=pa.int64()),
        }
    )
    source_right = pa.table(
        {
            "other": pa.array([20], type=pa.int32()),
            "id": pa.array([1], type=pa.int64()),
        }
    )

    executor, response = _execute_arrow_case(
        "data out; merge left right; by id; output out; run;",
        {
            "left": DataSetRef(kind="arrow_table", location="dataset://left", payload=source_left),
            "right": DataSetRef(kind="arrow_table", location="dataset://right", payload=source_right),
        },
    )

    assert response.has_errors is False
    assert executor.last_runtime_backend == "rust"
    output = response.outputs_arrow["out"]
    assert output.schema.names == ["id", "value", "other"]
    assert output.to_pylist() == [{"id": 1, "value": 10, "other": 20}]


def test_merge_without_by_preserves_schema_for_an_empty_option_occurrence() -> None:
    source_empty = pa.table(
        {"empty_amount": pa.array([10, 20], type=pa.int32())}
    )
    source_values = pa.table(
        {"right_value": pa.array(["first", "second"], type=pa.string())}
    )

    executor, response = _execute_arrow_case(
        "data out; "
        "merge empty_source(keep=empty_amount where=(empty_amount > 100)) "
        "values(keep=right_value); "
        "keep empty_amount right_value; output out; run;",
        {
            "empty_source": DataSetRef(
                kind="arrow_table",
                location="dataset://empty_source",
                payload=source_empty,
            ),
            "values": DataSetRef(
                kind="arrow_table",
                location="dataset://values",
                payload=source_values,
            ),
        },
    )

    assert response.has_errors is False
    assert executor.last_runtime_backend == "rust"
    output = response.outputs_arrow["out"]
    assert output.schema.names == ["empty_amount", "right_value"]
    assert output.schema.types == [pa.int32(), pa.string()]
    assert output.to_pylist() == [
        {"empty_amount": None, "right_value": "first"},
        {"empty_amount": None, "right_value": "second"},
    ]


def test_merge_output_metadata_priority_overrides_first_source_base() -> None:
    source_a = _metadata_table(
        dataset_label=b"MERGE source A",
        fields=(
            ("id", pa.int64(), b"A id"),
            ("left_value", pa.int64(), b"A left"),
        ),
        rows=[{"id": 1, "left_value": 10}],
    )
    source_b = _metadata_table(
        dataset_label=b"MERGE source B",
        fields=(
            ("id", pa.int64(), b"B id"),
            ("right_value", pa.string(), b"B right"),
        ),
        rows=[{"id": 2, "right_value": "later"}],
    )

    executor, response = _execute_arrow_case(
        'data out(label="merge explicit"); '
        'merge a b; by id; label id="merge id"; output out; run;',
        {
            "b": DataSetRef(
                kind="arrow_table",
                location="dataset://b",
                payload=source_b,
                metadata={"origin": "merge-b", "base": "b"},
            ),
            "a": DataSetRef(
                kind="arrow_table",
                location="dataset://a",
                payload=source_a,
                metadata={"origin": "merge-a", "base": "a"},
            ),
        },
    )

    assert response.has_errors is False
    assert response.outputs["out"].metadata == {
        "origin": "merge-a",
        "base": "a",
        "memlabel": "merge explicit",
        "column_labels": {"id": "merge id"},
    }
    output = response.outputs_arrow["out"]
    assert output.schema.metadata == {b"memlabel": b"merge explicit"}
    assert output.schema.field("id").metadata == {b"label": b"merge id"}
    assert output.schema.field("left_value").metadata == {b"label": b"A left"}
    assert output.schema.field("right_value").metadata == {b"label": b"B right"}


def test_merge_by_output_rename_preserves_first_source_metadata_owner() -> None:
    source_a = _metadata_table(
        dataset_label=b"MERGE rename source A",
        fields=(
            ("id", pa.int64(), b"A id"),
            ("left_value", pa.int64(), b"A left"),
        ),
        rows=[{"id": 1, "left_value": 10}],
    )
    source_b = _metadata_table(
        dataset_label=b"MERGE rename source B",
        fields=(
            ("id", pa.int64(), b"B id"),
            ("right_value", pa.string(), b"B right"),
        ),
        rows=[{"id": 1, "right_value": "later"}],
    )

    executor, response = _execute_arrow_case(
        "data out; merge a b; by id; rename id=key; output out; run;",
        {
            "a": DataSetRef(
                kind="arrow_table",
                location="dataset://a",
                payload=source_a,
            ),
            "b": DataSetRef(
                kind="arrow_table",
                location="dataset://b",
                payload=source_b,
            ),
        },
    )

    assert response.has_errors is False
    assert executor.last_runtime_backend == "rust"
    output = response.outputs_arrow["out"]
    assert output.schema.names == ["key", "left_value", "right_value"]
    assert output.schema.field("key").metadata == {b"label": b"A id"}
    assert output.schema.field("left_value").metadata == {b"label": b"A left"}
    assert output.schema.field("right_value").metadata == {b"label": b"B right"}


def test_merge_lineage_applies_options_per_repeated_source_occurrence() -> None:
    source = _metadata_table(
        dataset_label=b"MERGE repeated source",
        fields=(
            ("id", pa.int64(), b"source id"),
            ("left_value", pa.int64(), b"source left"),
            ("right_value", pa.string(), b"source right"),
            ("discard", pa.string(), b"source discard"),
        ),
        rows=[{"id": 1, "left_value": 10, "right_value": "one", "discard": "drop"}],
    )

    executor, response = _execute_arrow_case(
        "data out; "
        "merge source(keep=id left_value rename=(left_value=left)) "
        "source(keep=id right_value rename=(right_value=right)); "
        "by id; output out; run;",
        {
            "source": DataSetRef(
                kind="arrow_table",
                location="dataset://source",
                payload=source,
            )
        },
    )

    assert response.has_errors is False
    output = response.outputs_arrow["out"]
    assert output.schema.names == ["id", "left", "right"]
    assert output.schema.metadata == {b"memlabel": b"MERGE repeated source"}
    assert output.schema.field("id").metadata == {b"label": b"source id"}
    assert output.schema.field("left").metadata == {b"label": b"source left"}
    assert output.schema.field("right").metadata == {b"label": b"source right"}


def test_merge_rejects_duplicate_non_by_column_with_public_diagnostic() -> None:
    source_a = _metadata_table(
        dataset_label=b"MERGE source A",
        fields=(("id", pa.int64(), b"A id"), ("shared", pa.int64(), b"A shared")),
        rows=[{"id": 1, "shared": 10}],
    )
    source_b = _metadata_table(
        dataset_label=b"MERGE source B",
        fields=(("id", pa.int64(), b"B id"), ("shared", pa.int64(), b"B shared")),
        rows=[{"id": 2, "shared": 20}],
    )

    executor, response = _execute_arrow_case(
        "data out; merge a b; by id; output out; run;",
        {
            "b": DataSetRef(kind="arrow_table", location="dataset://b", payload=source_b),
            "a": DataSetRef(kind="arrow_table", location="dataset://a", payload=source_a),
        },
    )

    assert response.has_errors is True
    assert response.outputs_arrow == {}
    diagnostic = next(diagnostic for diagnostic in response.diagnostics if diagnostic.severity == "error")
    assert diagnostic.code == "RUNTIME_MERGE_DUPLICATE_COLUMN"
    assert "shared" in diagnostic.message.lower()
    assert executor.last_runtime_backend is None


def test_merge_rejects_duplicate_non_by_column_for_zero_row_sources() -> None:
    source_a = pa.table({"shared": pa.array([], type=pa.int64())})
    source_b = pa.table({"shared": pa.array([], type=pa.int64())})

    executor, response = _execute_arrow_case(
        "data out; merge a b; output out; run;",
        {
            "a": DataSetRef(kind="arrow_table", location="dataset://a", payload=source_a),
            "b": DataSetRef(kind="arrow_table", location="dataset://b", payload=source_b),
        },
    )

    assert response.has_errors is True
    assert response.outputs_arrow == {}
    diagnostic = next(diagnostic for diagnostic in response.diagnostics if diagnostic.severity == "error")
    assert diagnostic.code == "RUNTIME_MERGE_DUPLICATE_COLUMN"
    assert diagnostic.stage == "pre-evaluations"
    assert "shared" in diagnostic.message.lower()
    assert executor.last_runtime_backend is None


def test_merge_rejects_incompatible_by_type_before_native_execution() -> None:
    source_a = _metadata_table(
        dataset_label=b"MERGE source A",
        fields=(
            ("id", pa.int64(), b"A id"),
            ("left_value", pa.int64(), b"A left"),
        ),
        rows=[{"id": 1, "left_value": 10}],
    )
    source_b = _metadata_table(
        dataset_label=b"MERGE source B",
        fields=(
            ("id", pa.string(), b"B id"),
            ("right_value", pa.int64(), b"B right"),
        ),
        rows=[{"id": "2", "right_value": 20}],
    )

    executor, response = _execute_arrow_case(
        "data out; merge a b; by id; output out; run;",
        {
            "b": DataSetRef(kind="arrow_table", location="dataset://b", payload=source_b),
            "a": DataSetRef(kind="arrow_table", location="dataset://a", payload=source_a),
        },
    )

    assert response.has_errors is True
    assert response.outputs_arrow == {}
    diagnostic = next(diagnostic for diagnostic in response.diagnostics if diagnostic.severity == "error")
    assert diagnostic.code == "RUNTIME_BY_PRECONDITION_FAILED"
    diagnostic_text = f"{diagnostic.location} {diagnostic.message}".lower()
    assert "id" in diagnostic_text
    assert "int64" in diagnostic_text and "string" in diagnostic_text
    assert "a" in diagnostic_text and "b" in diagnostic_text
    assert executor.last_runtime_backend is None


def test_merge_rejects_incompatible_by_type_for_zero_row_sources() -> None:
    source_a = pa.table({"id": pa.array([], type=pa.int64())})
    source_b = pa.table({"id": pa.array([], type=pa.string())})

    executor, response = _execute_arrow_case(
        "data out; merge a b; by id; output out; run;",
        {
            "a": DataSetRef(kind="arrow_table", location="dataset://a", payload=source_a),
            "b": DataSetRef(kind="arrow_table", location="dataset://b", payload=source_b),
        },
    )

    assert response.has_errors is True
    assert response.outputs_arrow == {}
    diagnostic = next(diagnostic for diagnostic in response.diagnostics if diagnostic.severity == "error")
    assert diagnostic.code == "RUNTIME_BY_PRECONDITION_FAILED"
    assert diagnostic.stage == "pre-evaluations"
    diagnostic_text = f"{diagnostic.location} {diagnostic.message}".lower()
    assert "id" in diagnostic_text
    assert "int64" in diagnostic_text and "string" in diagnostic_text
    assert "a" in diagnostic_text and "b" in diagnostic_text
    assert executor.last_runtime_backend is None


def test_output_metadata_priority_preserves_first_source_base_and_explicit_overrides() -> None:
    source_a = _metadata_table(
        dataset_label=b"SET source A",
        fields=(("id", pa.int64(), b"A id"),),
        rows=[{"id": 1}],
    )
    source_b = _metadata_table(
        dataset_label=b"SET source B",
        fields=(("id", pa.int64(), b"B id"),),
        rows=[{"id": 2}],
    )

    executor, response = _execute_arrow_case(
        'data out(label="explicit output"); '
        'set a b; label id="Explicit id"; output out; run;',
        {
            "b": DataSetRef(
                kind="arrow_table",
                location="dataset://b",
                payload=source_b,
                metadata={"origin": "source-b", "base": "b"},
            ),
            "a": DataSetRef(
                kind="arrow_table",
                location="dataset://a",
                payload=source_a,
                metadata={
                    "origin": "source-a",
                    "base": "a",
                    "memlabel": "source-ref-a",
                },
            ),
        },
    )

    assert response.has_errors is False
    output_ref = response.outputs["out"]
    assert output_ref.metadata == {
        "origin": "source-a",
        "base": "a",
        "memlabel": "explicit output",
        "column_labels": {"id": "Explicit id"},
    }
    output = response.outputs_arrow["out"]
    assert output.schema.metadata == {b"memlabel": b"explicit output"}
    assert output.schema.field("id").metadata == {b"label": b"Explicit id"}


def test_multi_output_metadata_priority_keeps_source_base_for_empty_target() -> None:
    source = _metadata_table(
        dataset_label=b"multi source",
        fields=(("id", pa.int64(), b"source id"),),
        rows=[{"id": 1}],
    )

    executor, response = _execute_arrow_case(
        'data left(label="left output") right(label="right output"); '
        "set source; "
        "if id = 1 then output left; else output right; "
        "run;",
        {
            "source": DataSetRef(
                kind="arrow_table",
                location="dataset://source",
                payload=source,
                metadata={"origin": "source", "base": "shared"},
            )
        },
    )

    assert response.has_errors is False
    assert response.outputs["left"].metadata == {
        "origin": "source",
        "base": "shared",
        "memlabel": "left output",
    }
    assert response.outputs["right"].metadata == {
        "origin": "source",
        "base": "shared",
        "memlabel": "right output",
    }
    left = response.outputs_arrow["left"]
    right = response.outputs_arrow["right"]
    assert left.schema.metadata == {b"memlabel": b"left output"}
    assert left.schema.field("id").metadata == {b"label": b"source id"}
    assert right.to_pylist() == []
    assert right.schema.names == ["id"]
    assert right.schema.metadata == {b"memlabel": b"right output"}
    assert right.schema.field("id").metadata == {b"label": b"source id"}


def test_statement_output_rename_preserves_source_field_metadata() -> None:
    source = _metadata_table(
        dataset_label=b"statement rename source",
        fields=(
            ("id", pa.int64(), b"Identifier"),
            ("amount", pa.int64(), b"Amount"),
            ("status", pa.string(), b"Status"),
        ),
        rows=[{"id": 1, "amount": 10, "status": "ready"}],
    )

    executor, response = _execute_arrow_case(
        "data out; set source; rename amount=score; output out; run;",
        {
            "source": DataSetRef(
                kind="arrow_table",
                location="dataset://source",
                payload=source,
            )
        },
    )

    assert response.has_errors is False
    assert executor.last_runtime_backend == "rust"
    output = response.outputs_arrow["out"]
    assert output.to_pylist() == [{"id": 1, "score": 10, "status": "ready"}]
    assert output.schema.names == ["id", "score", "status"]
    assert output.schema.types == [pa.int64(), pa.int64(), pa.string()]
    assert output.schema.metadata == {b"memlabel": b"statement rename source"}
    assert output.schema.field("id").metadata == {b"label": b"Identifier"}
    assert output.schema.field("score").metadata == {b"label": b"Amount"}
    assert output.schema.field("status").metadata == {b"label": b"Status"}


def test_data_target_rename_preserves_source_field_metadata() -> None:
    source = _metadata_table(
        dataset_label=b"target rename source",
        fields=(
            ("id", pa.int64(), b"Identifier"),
            ("amount", pa.int64(), b"Amount"),
            ("status", pa.string(), b"Status"),
        ),
        rows=[{"id": 1, "amount": 10, "status": "ready"}],
    )

    executor, response = _execute_arrow_case(
        "data out(keep=id amount status rename=(amount=score)); "
        "set source; output out; run;",
        {
            "source": DataSetRef(
                kind="arrow_table",
                location="dataset://source",
                payload=source,
            )
        },
    )

    assert response.has_errors is False
    assert executor.last_runtime_backend == "rust"
    output = response.outputs_arrow["out"]
    assert output.to_pylist() == [{"id": 1, "score": 10, "status": "ready"}]
    assert output.schema.names == ["id", "score", "status"]
    assert output.schema.types == [pa.int64(), pa.int64(), pa.string()]
    assert output.schema.metadata == {b"memlabel": b"target rename source"}
    assert output.schema.field("id").metadata == {b"label": b"Identifier"}
    assert output.schema.field("score").metadata == {b"label": b"Amount"}
    assert output.schema.field("status").metadata == {b"label": b"Status"}


def test_chained_output_rename_is_isolated_per_target_and_preserves_unrenamed_fields() -> None:
    source = _metadata_table(
        dataset_label=b"chained rename source",
        fields=(
            ("id", pa.int64(), b"Identifier"),
            ("amount", pa.int64(), b"Amount"),
            ("status", pa.string(), b"Status"),
        ),
        rows=[
            {"id": 1, "amount": 10, "status": "left"},
            {"id": 2, "amount": 20, "status": "right"},
        ],
    )

    executor, response = _execute_arrow_case(
        "data renamed(keep=id score status rename=(score=renamed_score)) "
        "plain(keep=id score status); "
        "set source; rename amount=score; "
        "if id = 1 then output renamed; else output plain; run;",
        {
            "source": DataSetRef(
                kind="arrow_table",
                location="dataset://source",
                payload=source,
            )
        },
    )

    assert response.has_errors is False
    assert executor.last_runtime_backend == "rust"
    renamed = response.outputs_arrow["renamed"]
    plain = response.outputs_arrow["plain"]

    assert renamed.to_pylist() == [{"id": 1, "renamed_score": 10, "status": "left"}]
    assert renamed.schema.names == ["id", "renamed_score", "status"]
    assert renamed.schema.metadata == {b"memlabel": b"chained rename source"}
    assert renamed.schema.field("id").metadata == {b"label": b"Identifier"}
    assert renamed.schema.field("renamed_score").metadata == {b"label": b"Amount"}
    assert renamed.schema.field("status").metadata == {b"label": b"Status"}

    assert plain.to_pylist() == [{"id": 2, "score": 20, "status": "right"}]
    assert plain.schema.names == ["id", "score", "status"]
    assert plain.schema.metadata == {b"memlabel": b"chained rename source"}
    assert plain.schema.field("id").metadata == {b"label": b"Identifier"}
    assert plain.schema.field("score").metadata == {b"label": b"Amount"}
    assert plain.schema.field("status").metadata == {b"label": b"Status"}


def test_output_rename_after_keep_does_not_collide_with_dropped_field() -> None:
    source = _metadata_table(
        dataset_label=b"false collision source",
        fields=(
            ("amount", pa.int64(), b"Amount"),
            ("other", pa.int64(), b"Other"),
        ),
        rows=[{"amount": 10, "other": 99}],
    )

    executor, response = _execute_arrow_case(
        "data out(keep=amount rename=(amount=other)); set source; output out; run;",
        {
            "source": DataSetRef(
                kind="arrow_table",
                location="dataset://source",
                payload=source,
            )
        },
    )

    assert response.has_errors is False
    assert executor.last_runtime_backend == "rust"
    output = response.outputs_arrow["out"]
    assert output.to_pylist() == [{"other": 10}]
    assert output.schema.names == ["other"]
    assert output.schema.types == [pa.int64()]
    assert output.schema.field("other").metadata == {b"label": b"Amount"}
