import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest

from limulus import DataSetRef, DataStepExecutor, ExecuteRequest, Session
from limulus.execution import input_preparation as input_preparation_module
from limulus import native_bridge as native_bridge_module
from limulus.native_bridge import load_native_module
from limulus.models import HostFunctionContext


def _arrow_ref(name: str, table: pa.Table) -> DataSetRef:
    return DataSetRef(kind="arrow_table", location=f"dataset://{name}", payload=table)


@pytest.mark.parametrize("reverse_types", [False, True])
@pytest.mark.parametrize("empty_source", [None, "left", "right", "both"])
def test_merge_promotes_mixed_string_keys_preserving_metadata(
    reverse_types: bool, empty_source: str | None
) -> None:
    left_type, right_type = (
        (pa.large_string(), pa.string()) if reverse_types else (pa.string(), pa.large_string())
    )
    left_keys = [] if empty_source in {"left", "both"} else ["日本", "é", None]
    right_keys = [] if empty_source in {"right", "both"} else ["日本", "z", None]
    left_values = list(range(1, len(left_keys) + 1))
    right_values = list(range(10, 10 + len(right_keys)))
    left = pa.Table.from_arrays(
        [pa.array(left_keys, type=left_type), pa.array(left_values, type=pa.int16())],
        schema=pa.schema(
            [
                pa.field("KEY", left_type, metadata={b"label": b"Left key"}),
                pa.field("a", pa.int16(), metadata={b"label": b"Left value"}),
            ],
            metadata={b"memlabel": b"Left dataset"},
        ),
    )
    right = pa.Table.from_arrays(
        [pa.array(right_keys, type=right_type), pa.array(right_values, type=pa.int32())],
        schema=pa.schema(
            [
                pa.field("species", right_type, metadata={b"label": b"Right key"}),
                pa.field("b", pa.int32(), metadata={b"label": b"Right value"}),
            ]
        ),
    )
    session = Session()
    session.loads({"left": left, "right": right})
    result = session.submit(
        "data out; merge left(in=l) right(rename=(species=key) where=(b >= 10)); "
        "by key; run;"
    )

    assert result.success, result.format_log()
    output = session["out"]
    left_rows = dict(zip(left_keys, left_values))
    right_rows = dict(zip(right_keys, right_values))
    keys = sorted(left_rows.keys() | right_rows.keys(), key=lambda key: (key is None, key or ""))
    assert output.to_pylist() == [
        {"KEY": key, "a": left_rows.get(key), "b": right_rows.get(key)} for key in keys
    ]
    assert output.schema.field("KEY").type == pa.large_string()
    assert output.schema.field("KEY").metadata == {b"label": b"Left key"}
    assert output.schema.field("a") == left.schema.field("a")
    assert output.schema.field("b") == right.schema.field("b")
    assert output.schema.metadata == left.schema.metadata
    assert session["left"].schema == left.schema
    assert session["right"].schema == right.schema


@pytest.mark.parametrize("reverse_types", [False, True])
@pytest.mark.parametrize(
    "source_statement",
    [
        "set left right;",
        "set left(in=l) right(in=r);",
        "set left right; by key;",
        "set left(rename=(key=shared)) right(rename=(key=shared));",
    ],
)
def test_set_promotes_mixed_string_columns_after_source_options(
    reverse_types: bool, source_statement: str
) -> None:
    types = (pa.large_string(), pa.string()) if reverse_types else (pa.string(), pa.large_string())
    sources = {}
    for name, data_type, values in zip(("left", "right"), types, (["a", None], ["é", "日本"])):
        sources[name] = pa.Table.from_arrays(
            [pa.chunked_array([pa.array(values[:1], type=data_type), pa.array(values[1:], type=data_type)])],
            schema=pa.schema(
                [pa.field("key", data_type, metadata={b"label": name.encode()})],
                metadata={b"memlabel": name.encode()},
            ),
        )
    session = Session()
    session.loads(sources)
    result = session.submit(f"data out; {source_statement} run;")

    assert result.success, result.format_log()
    output = session["out"]
    column_name = "shared" if "rename=" in source_statement else "key"
    expected = ["a", "é", "日本", None] if "by key" in source_statement else ["a", None, "é", "日本"]
    assert output.column(column_name).to_pylist() == expected
    assert output.schema.field(column_name).type == pa.large_string()
    assert output.schema.field(column_name).metadata == {b"label": b"left"}
    assert output.schema.metadata == {b"memlabel": b"left"}
    assert session["left"].schema == sources["left"].schema
    assert session["right"].schema == sources["right"].schema


@pytest.mark.parametrize("source_statement", ["set left right;", "merge left right; by key;"])
@pytest.mark.parametrize("types", [(pa.float32(), pa.float64()), (pa.string(), pa.int64())])
def test_string_promotion_keeps_other_type_conflicts_strict(source_statement, types) -> None:
    session = Session()
    for name, data_type in zip(("left", "right"), types):
        value = "1" if pa.types.is_string(data_type) else 1
        session.load(name, pa.table({"key": pa.array([value], type=data_type)}))
    result = session.submit(f"data out; {source_statement} run;")

    assert not result.success
    assert "out" not in session.datasets


def test_session_rust_preserves_int64_boundaries_2pow53_neighbors_and_nulls() -> None:
    signed_values = [
        -(2**63),
        -(2**53) - 1,
        2**53 - 1,
        2**53,
        2**53 + 1,
        2**63 - 1,
        None,
    ]
    unsigned_values = [
        0,
        2**53 - 1,
        2**53,
        2**53 + 1,
        2**64 - 1,
        1,
        None,
    ]
    source = pa.table(
        {
            "signed_value": pa.array(signed_values, type=pa.int64()),
            "unsigned_value": pa.array(unsigned_values, type=pa.uint64()),
        }
    )

    session = Session(backend="rust")
    session.load("source", source)
    result = session.submit("data out; set source; output out; run;", backend="rust")

    assert result.success is True
    output = session.to_arrow("out")
    assert output.schema.field("signed_value").type == pa.int64()
    assert output.schema.field("unsigned_value").type == pa.uint64()
    assert output.column("signed_value").to_pylist() == signed_values
    assert output.column("unsigned_value").to_pylist() == unsigned_values


def test_data_step_executor_rust_distinguishes_adjacent_2pow53_integers() -> None:
    values = [2**53, 2**53 + 1, 2**53 + 2]
    source = pa.table({"value": pa.array(values, type=pa.int64())})
    executor = DataStepExecutor(runtime_backend="rust")

    response = executor.execute(
        ExecuteRequest(
            dsl_text=(
                "data out; set source; "
                "if value = 9007199254740993 then output out; run;"
            ),
            inputs={"source": _arrow_ref("source", source)},
            output_targets=("out",),
        )
    )

    assert response.has_errors is False
    assert executor.last_runtime_backend == "rust"
    output = response.outputs_arrow["out"]
    assert output.schema.field("value").type == pa.int64()
    assert output.column("value").to_pylist() == [2**53 + 1]


def test_session_rust_preserves_timestamp_ns_and_time64_ns_raw_values() -> None:
    timestamp_values = [-1, -1001, 1_000_001, None]
    time_values = [1, 1001, 86_399_999_999_999, None]
    source = pa.table(
        {
            "timestamp_value": pa.array(timestamp_values, type=pa.timestamp("ns")),
            "time_value": pa.array(time_values, type=pa.time64("ns")),
        }
    )

    session = Session(backend="rust")
    session.load("source", source)
    result = session.submit("data out; set source; output out; run;", backend="rust")

    assert result.success is True
    output = session.to_arrow("out")
    assert output.schema.field("timestamp_value").type == pa.timestamp("ns")
    assert output.schema.field("time_value").type == pa.time64("ns")
    assert output.column("timestamp_value").cast(pa.int64()).to_pylist() == timestamp_values
    assert output.column("time_value").cast(pa.int64()).to_pylist() == time_values


def test_python_empty_output_temporal_type_resolver_matches_native_contract() -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    native_executor = executor._rust_backend._executor
    assert native_executor is not None
    valid = {
        "timestamp[ns]": pa.timestamp("ns"),
        "timestamp[ns, tz=UTC]": pa.timestamp("ns", tz="UTC"),
        "timestamp[us, tz=America/New_York]": pa.timestamp(
            "us", tz="America/New_York"
        ),
        "time64[us]": pa.time64("us"),
        "time64[ns]": pa.time64("ns"),
    }
    for type_name, expected in valid.items():
        assert native_executor._resolve_arrow_type_name(type_name) == expected

    invalid = (
        "timestamp[ps]",
        "timestamp[ns, tz=]",
        "timestamp[ns",
        "time64[s]",
        "time64[ns",
    )
    for type_name in invalid:
        with pytest.raises(ValueError):
            native_executor._resolve_arrow_type_name(type_name)


def test_data_step_executor_rust_preserves_timestamp_timezone_identity_and_raw_values() -> None:
    raw_values = [-1, 1_000_001, None]
    source = pa.table(
        {
            "utc_value": pa.array(raw_values, type=pa.timestamp("ns", tz="UTC")),
            "new_york_value": pa.array(
                raw_values, type=pa.timestamp("ns", tz="America/New_York")
            ),
        }
    )
    executor = DataStepExecutor(runtime_backend="rust")

    response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set source; output out; run;",
            inputs={"source": _arrow_ref("source", source)},
            output_targets=("out",),
        )
    )

    assert response.has_errors is False
    assert executor.last_runtime_backend == "rust"
    output = response.outputs_arrow["out"]
    assert output.schema.field("utc_value").type == pa.timestamp("ns", tz="UTC")
    assert output.schema.field("new_york_value").type == pa.timestamp(
        "ns", tz="America/New_York"
    )
    assert output.column("utc_value").cast(pa.int64()).to_pylist() == raw_values
    assert output.column("new_york_value").cast(pa.int64()).to_pylist() == raw_values


def test_data_step_executor_rust_orders_signed_and_unsigned_values_above_2pow53() -> None:
    values = [2**53, 2**53 + 1, 2**53 + 2]
    source = pa.table(
        {
            "signed_value": pa.array(values, type=pa.int64()),
            "unsigned_value": pa.array(values, type=pa.uint64()),
        }
    )
    executor = DataStepExecutor(runtime_backend="rust")

    response = executor.execute(
        ExecuteRequest(
            dsl_text=(
                "data out; set source; "
                "if signed_value > 9007199254740992 "
                "and unsigned_value > 9007199254740992 then output out; run;"
            ),
            inputs={"source": _arrow_ref("source", source)},
            output_targets=("out",),
        )
    )

    assert response.has_errors is False
    assert executor.last_runtime_backend == "rust"
    output = response.outputs_arrow["out"]
    assert output.schema.field("signed_value").type == pa.int64()
    assert output.schema.field("unsigned_value").type == pa.uint64()
    assert output.column("signed_value").to_pylist() == [2**53 + 1, 2**53 + 2]
    assert output.column("unsigned_value").to_pylist() == [2**53 + 1, 2**53 + 2]


def test_session_rust_applies_all_source_local_options_once_before_native_set() -> None:
    source = pa.table(
        {
            "id": pa.array([1, 2, 3, 4], type=pa.int64()),
            "amount": pa.array([10, 20, 30, 40], type=pa.int64()),
            "drop_me": pa.array(["a", "b", "c", "d"]),
        }
    )
    session = Session(backend="rust")
    session.load("source", source)

    result = session.submit(
        "data out; "
        "set source(keep=id amount drop_me drop=drop_me "
        "where=(amount >= 20) rename=(amount=score) firstobs=2 obs=1); "
        "output out; run;",
        backend="rust",
    )

    assert result.success is True
    output = session.to_arrow("out")
    assert output.column_names == ["id", "score"]
    assert output.schema.types == [pa.int64(), pa.int64()]
    assert output.to_pylist() == [{"id": 3, "score": 30}]


def test_data_step_executor_rust_preserves_exact_and_nested_fields_for_by_merge() -> None:
    nested_type = pa.struct(
        [
            pa.field("label", pa.string(), metadata={b"role": b"label"}),
            pa.field("score", pa.int64(), metadata={b"role": b"exact-score"}),
        ]
    )
    left = pa.table(
        {
            "id": pa.array([2**53, 2**53 + 2, 2**53 + 1], type=pa.int64()),
            "left_exact": pa.array(
                [2**53 + 10, 2**53 + 12, 2**53 + 11], type=pa.int64()
            ),
            "left_nested": pa.array(
                [
                    {"label": "left-base", "score": 2**53 + 20},
                    {"label": "left-two", "score": 2**53 + 22},
                    {"label": "left-one", "score": 2**53 + 21},
                ],
                type=nested_type,
            ),
            "ignored": pa.array(["drop-base", "drop-two", "drop-one"]),
        }
    )
    right = pa.table(
        {
            "id": pa.array([2**53, 2**53 + 1, 2**53 + 3], type=pa.int64()),
            "right_exact": pa.array(
                [2**53 + 30, 2**53 + 31, 2**53 + 33], type=pa.uint64()
            ),
            "right_nested": pa.array(
                [
                    {"label": "right-base", "score": 2**53 + 40},
                    {"label": "right-one", "score": 2**53 + 41},
                    {"label": "right-three", "score": 2**53 + 43},
                ],
                type=nested_type,
            ),
            "ignored": pa.array(["drop-base", "drop-one", "drop-three"]),
        }
    )
    executor = DataStepExecutor(runtime_backend="rust")

    response = executor.execute(
        ExecuteRequest(
            dsl_text=(
                "data out; merge "
                "left(keep=id left_exact left_nested ignored drop=ignored "
                "where=(id > 9007199254740992) "
                "rename=(left_exact=left_value) firstobs=2 obs=1) "
                "right(keep=id right_exact right_nested ignored drop=ignored "
                "where=(id > 9007199254740992) "
                "rename=(right_exact=right_value) firstobs=1 obs=2); "
                "by id; output out; run;"
            ),
            inputs={
                "left": _arrow_ref("left", left),
                "right": _arrow_ref("right", right),
            },
            output_targets=("out",),
        )
    )

    assert response.has_errors is False
    assert executor.last_runtime_backend == "rust"
    output = response.outputs_arrow["out"]
    assert output.schema.names == [
        "id",
        "left_value",
        "left_nested",
        "right_value",
        "right_nested",
    ]
    assert output.schema.field("id").type == pa.int64()
    assert output.schema.field("left_value").type == pa.int64()
    assert output.schema.field("right_value").type == pa.uint64()
    assert output.schema.field("left_nested").type == nested_type
    assert output.schema.field("right_nested").type == nested_type
    assert output.to_pylist() == [
        {
            "id": 2**53 + 1,
            "left_value": 2**53 + 11,
            "left_nested": {"label": "left-one", "score": 2**53 + 21},
            "right_value": 2**53 + 31,
            "right_nested": {"label": "right-one", "score": 2**53 + 41},
        },
        {
            "id": 2**53 + 3,
            "left_value": None,
            "left_nested": None,
            "right_value": 2**53 + 33,
            "right_nested": {"label": "right-three", "score": 2**53 + 43},
        },
    ]


def test_data_step_executor_rust_preserves_exact_and_nested_fields_for_ordinal_merge() -> None:
    nested_type = pa.struct(
        [
            pa.field("label", pa.string(), metadata={b"role": b"label"}),
            pa.field("score", pa.int64(), metadata={b"role": b"exact-score"}),
        ]
    )
    left = pa.table(
        {
            "left_exact": pa.array(
                [2**53 + 101, 2**53 + 102, 2**53 + 103], type=pa.int64()
            ),
            "left_nested": pa.array(
                [
                    {"label": "left-zero", "score": 2**53 + 201},
                    {"label": "left-one", "score": 2**53 + 202},
                    {"label": "left-two", "score": 2**53 + 203},
                ],
                type=nested_type,
            ),
            "ignored": pa.array(["drop-zero", "drop-one", "drop-two"]),
        }
    )
    right = pa.table(
        {
            "right_exact": pa.array(
                [2**53 + 301, 2**53 + 302], type=pa.uint64()
            ),
            "right_nested": pa.array(
                [
                    {"label": "right-zero", "score": 2**53 + 401},
                    None,
                ],
                type=nested_type,
            ),
            "ignored": pa.array(["drop-zero", "drop-one"]),
        }
    )
    executor = DataStepExecutor(runtime_backend="rust")

    response = executor.execute(
        ExecuteRequest(
            dsl_text=(
                "data out; merge "
                "left(keep=left_exact left_nested ignored drop=ignored "
                "rename=(left_exact=left_value) firstobs=2 obs=1) "
                "right(keep=right_exact right_nested ignored drop=ignored "
                "rename=(right_exact=right_value) firstobs=1 obs=2); "
                "output out; run;"
            ),
            inputs={
                "left": _arrow_ref("left", left),
                "right": _arrow_ref("right", right),
            },
            output_targets=("out",),
        )
    )

    assert response.has_errors is False
    assert executor.last_runtime_backend == "rust"
    output = response.outputs_arrow["out"]
    assert output.schema.names == [
        "left_value",
        "left_nested",
        "right_value",
        "right_nested",
    ]
    assert output.schema.field("left_value").type == pa.int64()
    assert output.schema.field("right_value").type == pa.uint64()
    assert output.schema.field("left_nested").type == nested_type
    assert output.schema.field("right_nested").type == nested_type
    assert output.to_pylist() == [
        {
            "left_value": 2**53 + 102,
            "left_nested": {"label": "left-one", "score": 2**53 + 202},
            "right_value": 2**53 + 301,
            "right_nested": {"label": "right-zero", "score": 2**53 + 401},
        },
        {
            "left_value": None,
            "left_nested": None,
            "right_value": 2**53 + 302,
            "right_nested": None,
        },
    ]


def test_public_prepared_merge_path_does_not_materialize_rows() -> None:
    left = pa.table(
        {
            "id": pa.array([1, 2], type=pa.int64()),
            "left_value": pa.array([10, 20], type=pa.int64()),
        }
    )
    right = pa.table(
        {
            "id": pa.array([1, 2], type=pa.int64()),
            "right_value": pa.array([100, 200], type=pa.int64()),
        }
    )
    ordinal_left = pa.table(
        {"left_value": pa.array([10, 20], type=pa.int64())}
    )
    ordinal_right = pa.table(
        {"right_value": pa.array([100, 200], type=pa.int64())}
    )
    executor = DataStepExecutor(runtime_backend="rust")

    by_response = executor.execute(
        ExecuteRequest(
            dsl_text="data by_out; merge left right; by id; output by_out; run;",
            inputs={
                "left": _arrow_ref("left", left),
                "right": _arrow_ref("right", right),
            },
            output_targets=("by_out",),
        )
    )
    ordinal_response = executor.execute(
        ExecuteRequest(
            dsl_text="data ordinal_out; merge left right; output ordinal_out; run;",
            inputs={
                "left": _arrow_ref("ordinal_left", ordinal_left),
                "right": _arrow_ref("ordinal_right", ordinal_right),
            },
            output_targets=("ordinal_out",),
        )
    )

    assert by_response.has_errors is False
    assert ordinal_response.has_errors is False
    assert executor.last_runtime_backend == "rust"
    assert "by_out" in by_response.outputs_arrow
    assert "ordinal_out" in ordinal_response.outputs_arrow

    helper_sources = (
        inspect.getsource(
            input_preparation_module.ExecutionInputPreparationService._prepare_merge_arrow_input_if_supported
        ),
        inspect.getsource(
            input_preparation_module.ExecutionInputPreparationService._prepare_merge_arrow_input_without_by
        ),
    )
    forbidden_calls = ("to_pydict(", "from_pylist(")
    assert all(
        not any(call in source for call in forbidden_calls)
        for source in helper_sources
    ), "prepared merge must keep Arrow tables out of row-wide Python materialization"


def test_native_logical_row_consumers_do_not_materialize_whole_rows() -> None:
    native_root = Path(__file__).parents[1] / "native" / "limulus_native" / "src"
    runtime_text = (native_root / "runtime" / "mod.rs").read_text()
    expressions_text = (native_root / "expressions" / "mod.rs").read_text()
    functions_text = (native_root / "expressions" / "functions.rs").read_text()
    output_text = (native_root / "output" / "mod.rs").read_text()
    assert "row_dict_at(" not in runtime_text
    assert "source_value_by_name(" not in functions_text
    assert "LogicalRowAccess" in functions_text
    assert "row.iter()" not in output_text
    assert "get_row_item(" not in output_text
    assert "LogicalRowAccess" in output_text
    resolver_start = expressions_text.index("pub(crate) fn resolve_value")
    resolver_end = expressions_text.index("pub(crate) fn scalar_to_pyobject", resolver_start)
    resolver_text = expressions_text[resolver_start:resolver_end]
    assert "get_row_item(" not in resolver_text
    assert "logical_row" in resolver_text


def test_native_logical_row_owned_assignment_overrides_source_value() -> None:
    source = pa.table(
        {
            "id": pa.array([10, 20], type=pa.int64()),
            "source_value": pa.array([1, 2], type=pa.int64()),
        }
    )
    executor = DataStepExecutor(runtime_backend="rust")

    response = executor.execute(
        ExecuteRequest(
            dsl_text=(
                "data out; set source; "
                "source_value = id; "
                "output out; run;"
            ),
            inputs={"source": _arrow_ref("source", source)},
            output_targets=("out",),
        )
    )

    assert response.has_errors is False
    output = response.outputs_arrow["out"]
    assert output.schema.field("source_value").type == pa.int64()
    assert output.column("source_value").to_pylist() == [10, 20]


def test_native_output_snapshots_and_next_row_reset_source_overrides() -> None:
    source = pa.table({"id": [1, 2], "value": [10, 20]})
    executor = DataStepExecutor(runtime_backend="rust")
    response = executor.execute(
        ExecuteRequest(
            dsl_text=(
                "data out; set source; output; "
                "if id = 1 then value = 99; output; run;"
            ),
            inputs={"source": _arrow_ref("source", source)},
        )
    )

    assert response.has_errors is False
    assert response.outputs_arrow["out"].column("value").to_pylist() == [10, 99, 20, 20]


def test_native_reused_row_view_supports_zero_columns_with_source_rows() -> None:
    source = pa.table({"discard": [1, 2]}).drop(["discard"])
    assert source.num_rows == 2
    session = Session(backend="rust")
    session.load("source", source)

    result = session.submit("data out; set source; row_number = _N_; run;")

    assert result.success, result.log
    assert session["out"].column("row_number").to_pylist() == [1, 2]


def test_native_output_rename_preserves_owned_assignment_override() -> None:
    source = pa.table(
        {
            "id": pa.array([10, 20], type=pa.int64()),
            "value": pa.array([1, 2], type=pa.int64()),
        }
    )
    executor = DataStepExecutor(runtime_backend="rust")

    response = executor.execute(
        ExecuteRequest(
            dsl_text=(
                "data out; set source; value = id; "
                "rename value=renamed; output out; run;"
            ),
            inputs={"source": _arrow_ref("source", source)},
            output_targets=("out",),
        )
    )

    assert response.has_errors is False
    assert response.outputs_arrow["out"].to_pylist() == [
        {"id": 10, "renamed": 10},
        {"id": 20, "renamed": 20},
    ]


def test_native_output_projection_uses_independent_target_logical_views() -> None:
    source = pa.table(
        {
            "id": pa.array([1, 0], type=pa.int64()),
            "value": pa.array([11, 22], type=pa.int64()),
        }
    )
    executor = DataStepExecutor(runtime_backend="rust")

    response = executor.execute(
        ExecuteRequest(
            dsl_text=(
                "data left(keep=id value rename=(value=left_value)) "
                "right(keep=id value rename=(value=right_value)); "
                "set source; "
                "if id then output left; else output right; run;"
            ),
            inputs={"source": _arrow_ref("source", source)},
            output_targets=("left", "right"),
        )
    )

    assert response.has_errors is False
    assert response.outputs_arrow["left"].to_pylist() == [
        {"id": 1, "left_value": 11}
    ]
    assert response.outputs_arrow["right"].to_pylist() == [
        {"id": 0, "right_value": 22}
    ]


def test_native_output_where_uses_structured_ast_and_target_local_survivor_counter() -> None:
    source = pa.table(
        {
            "id": pa.array([1, 2, 3, 4], type=pa.int64()),
            "status": pa.array(["drop", "keep", "keep", "keep"]),
        }
    )
    executor = DataStepExecutor(runtime_backend="rust")

    response = executor.execute(
        ExecuteRequest(
            dsl_text=(
                "data out(where=((status = 'keep') and not (status is null)) "
                "firstobs=2 obs=1); "
                "set source; output out; run;"
            ),
            inputs={"source": _arrow_ref("source", source)},
            output_targets=("out",),
        )
    )

    assert response.has_errors is False
    assert response.outputs_arrow["out"].to_pylist() == [
        {"id": 3, "status": "keep"}
    ]


def test_native_output_dataset_options_register_each_target_independently() -> None:
    source = pa.table(
        {
            "id": pa.array([1, 2, 3, 4], type=pa.int64()),
            "status": pa.array(["drop", "keep", "keep", "keep"]),
        }
    )
    executor = DataStepExecutor(runtime_backend="rust")

    response = executor.execute(
        ExecuteRequest(
            dsl_text=(
                "data where_only(where=(status = 'keep')) "
                "firstobs_only(firstobs=2) "
                "obs_only(obs=2) "
                "compound(where=(status = 'keep') firstobs=2 obs=1); "
                "set source; "
                "output where_only; output firstobs_only; "
                "output obs_only; output compound; run;"
            ),
            inputs={"source": _arrow_ref("source", source)},
            output_targets=("where_only", "firstobs_only", "obs_only", "compound"),
        )
    )

    assert response.has_errors is False
    assert response.outputs_arrow["where_only"].column("id").to_pylist() == [2, 3, 4]
    assert response.outputs_arrow["firstobs_only"].column("id").to_pylist() == [2, 3, 4]
    assert response.outputs_arrow["obs_only"].column("id").to_pylist() == [1, 2]
    assert response.outputs_arrow["compound"].column("id").to_pylist() == [3]


def test_native_output_where_respects_target_drop_visibility() -> None:
    source = pa.table(
        {
            "id": pa.array([1, 2], type=pa.int64()),
            "status": pa.array(["keep", "drop"]),
        }
    )
    executor = DataStepExecutor(runtime_backend="rust")

    response = executor.execute(
        ExecuteRequest(
            dsl_text=(
                "data out(drop=status where=(status = 'keep')); "
                "set source; output out; run;"
            ),
            inputs={"source": _arrow_ref("source", source)},
            output_targets=("out",),
        )
    )

    assert response.has_errors is True
    diagnostic_text = " ".join(
        f"{item.code} {item.location} {item.message}" for item in response.diagnostics
    ).lower()
    assert "status" in diagnostic_text
    assert "where" in diagnostic_text


def test_native_empty_output_preserves_source_schema_after_target_where() -> None:
    source = pa.table(
        {
            "id": pa.array([1, 2], type=pa.int64()),
            "status": pa.array(["drop", "drop"]),
        }
    )
    executor = DataStepExecutor(runtime_backend="rust")

    response = executor.execute(
        ExecuteRequest(
            dsl_text=(
                "data out(where=(status = 'keep')); "
                "set source; output out; run;"
            ),
            inputs={"source": _arrow_ref("source", source)},
            output_targets=("out",),
        )
    )

    assert response.has_errors is False
    output = response.outputs_arrow["out"]
    assert output.column_names == ["id", "status"]
    assert output.schema.field("id").type == pa.int64()
    assert output.schema.field("status").type == pa.string()
    assert output.to_pylist() == []


@pytest.mark.parametrize(
    ("column_name", "nested_type", "values"),
    (
        ("list_value", pa.list_(pa.int64()), [[1], None]),
        ("large_list_value", pa.large_list(pa.int64()), [[1], None]),
        ("fixed_size_list_value", pa.list_(pa.int64(), 1), [[1], None]),
        (
            "struct_value",
            pa.struct([pa.field("item", pa.int64())]),
            [{"item": 1}, None],
        ),
    ),
)
@pytest.mark.parametrize("predicate", ("is null", "is not null"))
def test_native_output_null_predicate_rejects_nested_operands(
    column_name: str,
    nested_type: pa.DataType,
    values: list[object],
    predicate: str,
) -> None:
    source = pa.table(
        {
            "id": pa.array([1, 2], type=pa.int64()),
            column_name: pa.array(values, type=nested_type),
        }
    )
    executor = DataStepExecutor(runtime_backend="rust")

    response = executor.execute(
        ExecuteRequest(
            dsl_text=(
                f"data out(where=({column_name} {predicate})); "
                "set source; output out; run;"
            ),
            inputs={"source": _arrow_ref("source", source)},
            output_targets=("out",),
        )
    )

    assert response.has_errors is True
    assert response.outputs_arrow == {}
    diagnostic_text = " ".join(
        f"{item.code} {item.location} {item.message}" for item in response.diagnostics
    ).lower()
    assert column_name in diagnostic_text
    assert "where" in diagnostic_text
    assert "complex" in diagnostic_text or "nested" in diagnostic_text


def test_native_output_null_predicate_preserves_scalar_behavior() -> None:
    source = pa.table(
        {
            "id": pa.array([1, 2], type=pa.int64()),
            "status": pa.array([None, "keep"], type=pa.string()),
        }
    )
    executor = DataStepExecutor(runtime_backend="rust")

    response = executor.execute(
        ExecuteRequest(
            dsl_text="data out(where=(status is null)); set source; output out; run;",
            inputs={"source": _arrow_ref("source", source)},
            output_targets=("out",),
        )
    )

    assert response.has_errors is False
    assert response.outputs_arrow["out"].to_pylist() == [
        {"id": 1, "status": None}
    ]


@pytest.mark.parametrize(
    ("column_name", "nested_type", "values"),
    (
        ("list_value", pa.list_(pa.int64()), [[1], None]),
        ("large_list_value", pa.large_list(pa.int64()), [[1], None]),
        ("fixed_size_list_value", pa.list_(pa.int64(), 1), [[1], None]),
        (
            "struct_value",
            pa.struct([pa.field("item", pa.int64())]),
            [{"item": 1}, None],
        ),
    ),
)
@pytest.mark.parametrize("predicate", ("is null", "is not null"))
def test_native_input_null_predicate_rejects_nested_operands(
    column_name: str,
    nested_type: pa.DataType,
    values: list[object],
    predicate: str,
) -> None:
    source = pa.table(
        {
            "id": pa.array([1, 2], type=pa.int64()),
            column_name: pa.array(values, type=nested_type),
        }
    )
    executor = DataStepExecutor(runtime_backend="rust")

    response = executor.execute(
        ExecuteRequest(
            dsl_text=(
                f"data out; set source(where=({column_name} {predicate})); "
                "output out; run;"
            ),
            inputs={"source": _arrow_ref("source", source)},
            output_targets=("out",),
        )
    )

    assert response.has_errors is True
    assert response.outputs_arrow == {}
    diagnostic_text = " ".join(
        f"{item.code} {item.location} {item.message}" for item in response.diagnostics
    ).lower()
    assert column_name in diagnostic_text
    assert "where" in diagnostic_text
    assert "complex" in diagnostic_text or "nested" in diagnostic_text


@pytest.mark.parametrize(
    ("values", "dsl_text", "output_name"),
    (
        (
            [],
            "data out; set source; output out; run;",
            "nested",
        ),
        (
            [None, None],
            "data out(keep=nested rename=(nested=renamed)); "
            "set source; output out; run;",
            "renamed",
        ),
    ),
)
def test_native_empty_and_all_null_nested_inputs_preserve_field_lineage(
    values: list[object],
    dsl_text: str,
    output_name: str,
) -> None:
    nested_type = pa.list_(
        pa.field("item", pa.int64(), metadata={b"child-role": b"item"})
    )
    nested_field = pa.field(
        "nested",
        nested_type,
        nullable=True,
        metadata={b"column-role": b"nested"},
    )
    source = pa.Table.from_arrays(
        [pa.array(values, type=nested_type)],
        schema=pa.schema(
            [nested_field],
            metadata={b"dataset-role": b"source-backed"},
        ),
    )
    executor = DataStepExecutor(runtime_backend="rust")

    response = executor.execute(
        ExecuteRequest(
            dsl_text=dsl_text,
            inputs={"source": _arrow_ref("source", source)},
            output_targets=("out",),
        )
    )

    assert response.has_errors is False
    output = response.outputs_arrow["out"]
    assert output.schema.metadata == {b"dataset-role": b"source-backed"}
    assert output.schema.names == [output_name]
    assert output.schema.field(output_name).metadata == {b"column-role": b"nested"}
    assert output.schema.field(output_name).type.value_field.metadata == {
        b"child-role": b"item"
    }
    if values:
        assert output.to_pylist() == [{output_name: None}, {output_name: None}]
    else:
        assert output.to_pylist() == []


def test_native_empty_input_preserves_source_schema() -> None:
    source = pa.table(
        {
            "id": pa.array([], type=pa.int64()),
            "timestamp_value": pa.array([], type=pa.timestamp("ns", tz="UTC")),
        }
    )
    executor = DataStepExecutor(runtime_backend="rust")

    response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set source; output out; run;",
            inputs={"source": _arrow_ref("source", source)},
            output_targets=("out",),
        )
    )

    assert response.has_errors is False
    output = response.outputs_arrow["out"]
    assert output.column_names == ["id", "timestamp_value"]
    assert output.schema.field("id").type == pa.int64()
    assert output.schema.field("timestamp_value").type == pa.timestamp(
        "ns", tz="UTC"
    )
    assert output.to_pylist() == []


def test_native_datastep_where_selects_rows_through_logical_row() -> None:
    source = pa.table(
        {
            "id": pa.array([1, 2, 3], type=pa.int64()),
            "status": pa.array(["drop", "keep", "keep"]),
        }
    )
    executor = DataStepExecutor(runtime_backend="rust")

    response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set source; where status = 'keep'; output out; run;",
            inputs={"source": _arrow_ref("source", source)},
            output_targets=("out",),
        )
    )

    assert response.has_errors is False
    assert response.outputs_arrow["out"].to_pylist() == [
        {"id": 2, "status": "keep"},
        {"id": 3, "status": "keep"},
    ]


def test_native_structured_where_rejects_malformed_payload_before_row_execution() -> None:
    native_module, load_error = load_native_module()
    assert native_module is not None, load_error
    fixture_path = (
        Path(__file__).parents[1]
        / "native"
        / "limulus_native"
        / "tests"
        / "fixtures"
        / "typed_payload_v1.json"
    )
    valid_program = json.loads(fixture_path.read_text(encoding="utf-8"))["valid_program"]

    malformed_expressions = (
        {"kind": "unknown", "column": "status"},
        {
            "kind": "comparison",
            "column": "status",
            "operator": "contains",
            "value": "keep",
        },
        {"kind": "logical", "operator": "and", "operands": []},
        {"kind": "not"},
    )
    for expression in malformed_expressions:
        program = json.loads(json.dumps(valid_program))
        program["effective_output_options"]["work.out"]["where_expr"] = expression
        native_result = native_module.execute_block(
            {
                "native_program_json": json.dumps(program),
                "preparation_kind": "none",
                "input_streams": {},
                "execution_plan": {},
                "apply_registry": {},
                "format_catalog_payload": {},
            }
        )
        diagnostics = native_result["diagnostics"]
        assert diagnostics[0]["code"] == "PARSE_RUST_AST_DESERIALIZE_FAILED", (
            expression,
            diagnostics,
        )


def test_native_apply_converts_only_requested_raw_temporal_argument_and_fails_closed() -> None:
    source = pa.table(
        {"id": pa.array([1], type=pa.int64()), "timestamp_value": pa.array([1], type=pa.timestamp("ns"))}
    )
    executor = DataStepExecutor(runtime_backend="rust")

    def identity(value: object) -> object:
        return value

    response = executor.execute(
        ExecuteRequest(
            dsl_text=(
                "data out; set source; "
                "value = apply('identity', timestamp_value); output out; run;"
            ),
            inputs={"source": _arrow_ref("source", source)},
            output_targets=("out",),
            host_function_context=HostFunctionContext(caller_globals={"identity": identity}),
        )
    )

    assert response.has_errors is True
    assert response.outputs_arrow == {}
    diagnostic_text = " ".join(
        f"{item.code} {item.location} {item.message}" for item in response.diagnostics
    ).lower()
    assert "timestamp_value" in diagnostic_text
    assert "apply" in diagnostic_text
    assert "python boundary" in diagnostic_text


def test_data_step_executor_rust_preserves_source_backed_nested_types_across_batches() -> None:
    list_type = pa.list_(
        pa.field("item", pa.int64(), metadata={b"child-role": b"list-item"})
    )
    large_list_type = pa.large_list(
        pa.field("item", pa.string(), metadata={b"child-role": b"large-item"})
    )
    fixed_size_list_type = pa.list_(
        pa.field("item", pa.int32(), metadata={b"child-role": b"fixed-item"}),
        2,
    )
    struct_type = pa.struct(
        [
            pa.field("label", pa.string(), metadata={b"child-role": b"label"}),
            pa.field("score", pa.int64(), metadata={b"child-role": b"score"}),
        ]
    )
    schema = pa.schema(
        [
            pa.field("id", pa.int64(), metadata={b"column-role": b"filter-key"}),
            pa.field("list_value", list_type, metadata={b"column-role": b"list"}),
            pa.field(
                "large_list_value",
                large_list_type,
                metadata={b"column-role": b"large-list"},
            ),
            pa.field(
                "fixed_size_list_value",
                fixed_size_list_type,
                metadata={b"column-role": b"fixed-size-list"},
            ),
            pa.field("struct_value", struct_type, metadata={b"column-role": b"struct"}),
        ],
        metadata={b"dataset-role": b"source-backed"},
    )
    source = pa.Table.from_batches(
        [
            pa.RecordBatch.from_arrays(
                [
                    pa.array([1, 2], type=pa.int64()),
                    pa.array([[10, 11], None], type=list_type),
                    pa.array([["one"], None], type=large_list_type),
                    pa.array([[101, 102], None], type=fixed_size_list_type),
                    pa.array(
                        [
                            {"label": "one", "score": 101},
                            None,
                        ],
                        type=struct_type,
                    ),
                ],
                schema=schema,
            ),
            pa.RecordBatch.from_arrays(
                [
                    pa.array([3, 4], type=pa.int64()),
                    pa.array([[], [40, None]], type=list_type),
                    pa.array([[], ["four", None]], type=large_list_type),
                    pa.array([[303, None], [404, 405]], type=fixed_size_list_type),
                    pa.array(
                        [
                            {"label": "three", "score": 303},
                            {"label": "four", "score": None},
                        ],
                        type=struct_type,
                    ),
                ],
                schema=schema,
            ),
        ],
        schema=schema,
    )
    assert source.column("id").num_chunks == 2
    executor = DataStepExecutor(runtime_backend="rust")

    response = executor.execute(
        ExecuteRequest(
            dsl_text=(
                "data out; "
                "set source(keep=id list_value large_list_value fixed_size_list_value "
                "struct_value where=(id > 1) "
                "rename=(struct_value=renamed_struct)); "
                "output out; run;"
            ),
            inputs={"source": _arrow_ref("source", source)},
            output_targets=("out",),
        )
    )

    assert response.has_errors is False
    assert executor.last_runtime_backend == "rust"
    output = response.outputs_arrow["out"]
    assert output.schema.names == [
        "id",
        "list_value",
        "large_list_value",
        "fixed_size_list_value",
        "renamed_struct",
    ]
    assert output.schema.metadata == {b"dataset-role": b"source-backed"}
    assert output.schema.field("id").type == pa.int64()
    assert output.schema.field("list_value").type == list_type
    assert output.schema.field("large_list_value").type == large_list_type
    assert output.schema.field("fixed_size_list_value").type == fixed_size_list_type
    assert output.schema.field("renamed_struct").type == struct_type
    assert output.to_pylist() == [
        {
            "id": 2,
            "list_value": None,
            "large_list_value": None,
            "fixed_size_list_value": None,
            "renamed_struct": None,
        },
        {
            "id": 3,
            "list_value": [],
            "large_list_value": [],
            "fixed_size_list_value": [303, None],
            "renamed_struct": {"label": "three", "score": 303},
        },
        {
            "id": 4,
            "list_value": [40, None],
            "large_list_value": ["four", None],
            "fixed_size_list_value": [404, 405],
            "renamed_struct": {"label": "four", "score": None},
        },
    ]
    expected_field_metadata = {
        "id": {b"column-role": b"filter-key"},
        "list_value": {b"column-role": b"list"},
        "large_list_value": {b"column-role": b"large-list"},
        "fixed_size_list_value": {b"column-role": b"fixed-size-list"},
        "renamed_struct": {b"column-role": b"struct"},
    }
    for name, metadata in expected_field_metadata.items():
        assert output.schema.field(name).metadata == metadata
    assert output.schema.field("list_value").type.value_field.metadata == {
        b"child-role": b"list-item"
    }
    assert output.schema.field("large_list_value").type.value_field.metadata == {
        b"child-role": b"large-item"
    }
    assert output.schema.field("fixed_size_list_value").type.value_field.metadata == {
        b"child-role": b"fixed-item"
    }
    actual_struct_type = output.schema.field("renamed_struct").type
    assert actual_struct_type.field("label").metadata == {b"child-role": b"label"}
    assert actual_struct_type.field("score").metadata == {b"child-role": b"score"}


def test_data_step_executor_rust_rejects_nested_evaluation_without_partial_publish() -> None:
    nested_type = pa.list_(pa.int64())
    source = pa.table(
        {
            "id": pa.array([1, 2], type=pa.int64()),
            "nested_value": pa.array([[10], [20]], type=nested_type),
        }
    )
    executor = DataStepExecutor(runtime_backend="rust")

    response = executor.execute(
        ExecuteRequest(
            dsl_text=(
                "data out; set source; "
                "if nested_value > 0 then output out; run;"
            ),
            inputs={"source": _arrow_ref("source", source)},
            output_targets=("out",),
        )
    )

    assert response.has_errors is True
    assert response.outputs == {}
    assert response.outputs_arrow == {}
    diagnostic = next(
        diagnostic for diagnostic in response.diagnostics if diagnostic.severity == "error"
    )
    diagnostic_text = " ".join(
        (diagnostic.code, diagnostic.location, diagnostic.message)
    ).lower()
    assert "nested_value" in diagnostic_text or "nested arrow source column" in diagnostic_text
    assert any(
        context in diagnostic_text for context in ("expression", "where", "unsupported")
    )


def test_native_source_backed_nested_set_gathers_compatible_multi_source_fields() -> None:
    nested_type = pa.list_(
        pa.field("item", pa.int64(), metadata={b"child-role": b"item"})
    )
    left = pa.table(
        {
            "id": pa.array([1, 2], type=pa.int64()),
            "nested": pa.array([[10], None], type=nested_type),
        }
    )
    right = pa.table(
        {
            "id": pa.array([3], type=pa.int64()),
            "nested": pa.array([[30, None]], type=nested_type),
        }
    )
    executor = DataStepExecutor(runtime_backend="rust")

    response = executor.execute(
        ExecuteRequest(
            dsl_text=(
                "data out; set left(keep=nested) right(keep=nested); "
                "output out; run;"
            ),
            inputs={
                "left": _arrow_ref("left", left),
                "right": _arrow_ref("right", right),
            },
            output_targets=("out",),
        )
    )

    assert response.has_errors is False
    output = response.outputs_arrow["out"]
    assert output.schema.field("nested").type == nested_type
    assert output.column("nested").to_pylist() == [[10], None, [30, None]]


def test_native_multi_source_set_preserves_matching_nested_field_metadata() -> None:
    child = pa.field("item", pa.int64(), metadata={b"child-role": b"item"})
    nested_type = pa.list_(child)
    left_field = pa.field(
        "nested",
        nested_type,
        nullable=True,
        metadata={b"column-role": b"nested"},
    )
    right_field = pa.field(
        "nested",
        nested_type,
        nullable=True,
        metadata={b"column-role": b"nested"},
    )
    left = pa.Table.from_arrays(
        [pa.array([[10], None], type=nested_type)],
        schema=pa.schema([left_field], metadata={b"dataset-role": b"source"}),
    )
    right = pa.Table.from_arrays(
        [pa.array([[20]], type=nested_type)],
        schema=pa.schema([right_field], metadata={b"dataset-role": b"source"}),
    )
    executor = DataStepExecutor(runtime_backend="rust")

    response = executor.execute(
        ExecuteRequest(
            dsl_text=(
                "data out; set left(keep=nested) right(keep=nested); "
                "output out; run;"
            ),
            inputs={
                "left": _arrow_ref("left", left),
                "right": _arrow_ref("right", right),
            },
            output_targets=("out",),
        )
    )

    assert response.has_errors is False
    output = response.outputs_arrow["out"]
    assert output.to_pylist() == [{"nested": [10]}, {"nested": None}, {"nested": [20]}]
    assert output.schema.metadata == {b"dataset-role": b"source"}
    assert output.schema.field("nested").metadata == {b"column-role": b"nested"}
    assert output.schema.field("nested").type.value_field.metadata == {
        b"child-role": b"item"
    }


@pytest.mark.parametrize("mismatch", ("field_metadata", "child_metadata", "nullable"))
def test_native_multi_source_set_rejects_nested_field_metadata_mismatch(
    mismatch: str,
) -> None:
    left_child = pa.field("item", pa.int64(), metadata={b"child-role": b"left"})
    right_child = pa.field(
        "item",
        pa.int64(),
        metadata={b"child-role": b"right" if mismatch == "child_metadata" else b"left"},
    )
    nested_type = pa.list_(left_child)
    right_type = pa.list_(right_child)
    left_field = pa.field(
        "nested",
        nested_type,
        nullable=True,
        metadata={b"column-role": b"left"},
    )
    right_field = pa.field(
        "nested",
        right_type,
        nullable=mismatch != "nullable",
        metadata={
            b"column-role": b"right" if mismatch == "field_metadata" else b"left"
        },
    )
    left = pa.Table.from_arrays(
        [pa.array([[10]], type=nested_type)],
        schema=pa.schema([left_field], metadata={b"dataset-role": b"source"}),
    )
    right = pa.Table.from_arrays(
        [pa.array([[20]], type=right_type)],
        schema=pa.schema([right_field], metadata={b"dataset-role": b"source"}),
    )
    executor = DataStepExecutor(runtime_backend="rust")

    response = executor.execute(
        ExecuteRequest(
            dsl_text=(
                "data out; set left(keep=nested) right(keep=nested); "
                "output out; run;"
            ),
            inputs={
                "left": _arrow_ref("left", left),
                "right": _arrow_ref("right", right),
            },
            output_targets=("out",),
        )
    )

    assert response.has_errors is True
    assert response.outputs_arrow == {}
    diagnostic_text = " ".join(
        f"{item.code} {item.location} {item.message}" for item in response.diagnostics
    ).lower()
    assert "incompatible" in diagnostic_text
    assert "nested" in diagnostic_text or "field" in diagnostic_text


def test_native_source_backed_nested_set_rejects_incompatible_field_ownership() -> None:
    left_type = pa.list_(pa.field("item", pa.int64()))
    right_type = pa.list_(pa.field("item", pa.string()))
    left = pa.table({"nested": pa.array([[1]], type=left_type)})
    right = pa.table({"nested": pa.array([["one"]], type=right_type)})
    executor = DataStepExecutor(runtime_backend="rust")

    response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set left right; output out; run;",
            inputs={
                "left": _arrow_ref("left", left),
                "right": _arrow_ref("right", right),
            },
            output_targets=("out",),
        )
    )

    assert response.has_errors is True
    assert response.outputs_arrow == {}
    diagnostic_text = " ".join(
        f"{item.code} {item.location} {item.message}" for item in response.diagnostics
    ).lower()
    assert "incompatible" in diagnostic_text
    assert "nested" in diagnostic_text


def test_native_freshness_inventory_covers_all_rust_sources_and_build_inputs(tmp_path: Path) -> None:
    repo_root = tmp_path
    native_root = repo_root / "native" / "limulus_native"
    source_root = native_root / "src"
    (source_root / "expressions" / "functions").mkdir(parents=True)
    (source_root / "lib.rs").write_text("", encoding="utf-8")
    (source_root / "source_registry.rs").write_text("", encoding="utf-8")
    (source_root / "expressions" / "mod.rs").write_text("", encoding="utf-8")
    (source_root / "expressions" / "functions" / "nested.rs").write_text(
        "", encoding="utf-8"
    )
    (native_root / "build.rs").write_text("fn main() {}\n", encoding="utf-8")
    (native_root / "Cargo.lock").write_text("# lock\n", encoding="utf-8")
    (native_root / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
    cargo_config = repo_root / ".cargo" / "config.toml"
    cargo_config.parent.mkdir()
    cargo_config.write_text("[env]\n", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text("[tool.maturin]\n", encoding="utf-8")

    inventory = native_bridge_module._native_freshness_inputs(repo_root)
    relative_inventory = tuple(
        path.relative_to(repo_root).as_posix() for path in inventory
    )

    assert relative_inventory == (
        ".cargo/config.toml",
        "native/limulus_native/Cargo.lock",
        "native/limulus_native/Cargo.toml",
        "native/limulus_native/build.rs",
        "native/limulus_native/src/expressions/functions/nested.rs",
        "native/limulus_native/src/expressions/mod.rs",
        "native/limulus_native/src/lib.rs",
        "native/limulus_native/src/source_registry.rs",
        "pyproject.toml",
    )
    assert native_bridge_module._newest_native_source(repo_root) is not None


@pytest.mark.parametrize(
    "freshness_input",
    (
        "source",
        "build.rs",
        "Cargo.lock",
        "Cargo.toml",
        ".cargo/config.toml",
        "pyproject.toml",
    ),
)
def test_native_freshness_rejects_artifact_when_any_build_input_is_newer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    freshness_input: str,
) -> None:
    repo_root = tmp_path
    native_root = repo_root / "native" / "limulus_native"
    source_root = native_root / "src"
    source_root.mkdir(parents=True)
    rust_source = source_root / "source_registry.rs"
    rust_source.write_text("", encoding="utf-8")
    build_script = native_root / "build.rs"
    build_script.write_text("fn main() {}\n", encoding="utf-8")
    cargo_lock = native_root / "Cargo.lock"
    cargo_lock.write_text("# lock\n", encoding="utf-8")
    cargo_toml = native_root / "Cargo.toml"
    cargo_toml.write_text("[package]\n", encoding="utf-8")
    cargo_config = repo_root / ".cargo" / "config.toml"
    cargo_config.parent.mkdir()
    cargo_config.write_text("[env]\n", encoding="utf-8")
    pyproject_toml = repo_root / "pyproject.toml"
    pyproject_toml.write_text("[tool.maturin]\n", encoding="utf-8")
    artifact = native_root / "target" / "release" / "liblimulus_native.dylib"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"artifact")

    freshness_inputs = native_bridge_module._native_freshness_inputs(repo_root)
    for path in freshness_inputs:
        os.utime(path, (100, 100))
    os.utime(artifact, (200, 200))
    newer_path = {
        "source": rust_source,
        "build.rs": build_script,
        "Cargo.lock": native_root / "Cargo.lock",
        "Cargo.toml": cargo_toml,
        ".cargo/config.toml": cargo_config,
        "pyproject.toml": pyproject_toml,
    }[freshness_input]
    os.utime(newer_path, (300, 300))
    monkeypatch.setattr(native_bridge_module, "_repo_root", lambda: repo_root)

    diagnostic = native_bridge_module._detect_native_source_drift(
        SimpleNamespace(__file__=str(artifact))
    )

    assert diagnostic is not None
    assert newer_path.relative_to(repo_root).as_posix() in diagnostic


def test_native_current_artifact_resolves_to_a_concrete_extension_path() -> None:
    native_module, load_error = load_native_module()
    assert native_module is not None, load_error
    module_path = Path(native_module.__file__)
    assert module_path.is_file()
    assert module_path.name.startswith("limulus_native")
