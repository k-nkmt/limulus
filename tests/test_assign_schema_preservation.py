"""Assign result assembly preserves unchanged source columns."""

from __future__ import annotations

import limulus.session
import pyarrow as pa
import pytest

from limulus import Session


def test_assign_preserves_unchanged_list_column_and_uses_computed_changed_new_columns() -> None:
    items_field = pa.field(
        "items",
        pa.list_(pa.field("element", pa.string(), metadata={b"child": b"keep"})),
        metadata={b"label": b"Items"},
    )
    name_field = pa.field("name", pa.string(), metadata={b"label": b"Name"})
    score_field = pa.field("score", pa.float64(), nullable=False, metadata={b"label": b"Score"})
    source_schema = pa.schema(
        [items_field, name_field, score_field],
        metadata={b"memlabel": b"Source"},
    )
    source = pa.Table.from_arrays(
        [
            pa.array([["a", "b"], [None], ["c"]], type=items_field.type),
            pa.array(["Alice", "Bob", "Cara"], type=name_field.type),
            pa.array([10.0, 3.0, 0.5], type=score_field.type),
        ],
        schema=source_schema,
    )
    session = Session()
    session.load("src", source)

    session.assign("src", out="out", score="score * 2", tag=3.5)

    out = session.to_arrow("out")

    # Column order and row alignment are preserved.
    assert out.column_names == ["items", "name", "score", "tag"]
    assert out.num_rows == source.num_rows

    # Unchanged columns keep the original field wholesale: outer field
    # metadata, nested child field metadata, nullability, type, and values.
    assert out.schema.field("items") == items_field
    assert out.schema.field("name") == name_field
    assert out.column("items").type == items_field.type
    assert out.column("items").to_pylist() == [["a", "b"], [None], ["c"]]
    assert out.column("name").to_pylist() == ["Alice", "Bob", "Cara"]

    # Changed columns use the computed result, not the source values.
    assert out.column("score").type == pa.float64()
    assert out.column("score").to_pylist() == [20.0, 6.0, 1.0]
    assert out.schema.field("score").metadata == score_field.metadata

    # New columns use the computed result too.
    assert out.column("tag").to_pylist() == [3.5, 3.5, 3.5]

    # Table-level metadata follows the existing finalization contract.
    assert out.schema.metadata == source_schema.metadata


def test_assign_replaces_existing_column_case_insensitively_and_keeps_output_name() -> None:
    """Case-insensitive targets replace the existing column in place.

    The resolved actual target name owns the changed set: the output keeps the
    original column name and position, adopts the computed values, and keeps
    the changed column's outer field metadata per the existing contract.
    """

    weight_field = pa.field("Weight", pa.float64(), metadata={b"label": b"Weight"})
    name_field = pa.field("Name", pa.string(), metadata={b"label": b"Name"})
    source_schema = pa.schema([weight_field, name_field], metadata={b"memlabel": b"Source"})
    session = Session()
    session.load(
        "src",
        pa.Table.from_arrays(
            [pa.array([1.5, 2.5], type=pa.float64()), pa.array(["A", "B"], type=pa.string())],
            schema=source_schema,
        ),
    )
    source = session.to_arrow("src")

    session.assign("src", out="out", weight="weight * 2")
    out = session.to_arrow("out")

    assert out.column_names == ["Weight", "Name"]
    assert out.schema.field("Weight").metadata == weight_field.metadata
    assert out.column("Weight").to_pylist() == [3.0, 5.0]
    assert out.schema.field("Name").equals(name_field, check_metadata=True)
    assert out.column("Name").to_pylist() == source.column("Name").to_pylist()
    assert out.schema.metadata == source_schema.metadata


def test_assign_sequential_references_resolve_created_and_mixed_case_columns() -> None:
    """Sequential assignment and created-column references keep current behavior."""

    value_field = pa.field("value", pa.int64())
    session = Session()
    session.load("src", pa.table({"value": pa.array([3, 5], type=pa.int64())}))

    session.assign("src", out="seq", doubled="value * 2", quad="doubled * 2", total="DOUBLED + quad")
    out = session.to_arrow("seq")

    assert out.column_names == ["value", "doubled", "quad", "total"]
    assert out.column("doubled").to_pylist() == [6, 10]
    assert out.column("quad").to_pylist() == [12, 20]
    assert out.column("total").to_pylist() == [18, 30]
    assert out.column("value").to_pylist() == [3, 5]


def test_assign_replaces_list_column_with_computed_result() -> None:
    """A replaced List column adopts the computed result, not the source type."""

    items_field = pa.field(
        "items",
        pa.list_(pa.field("element", pa.string(), metadata={b"child": b"keep"})),
        metadata={b"label": b"Items"},
    )
    score_field = pa.field("score", pa.int64())
    session = Session()
    session.load(
        "src",
        pa.Table.from_arrays(
            [
                pa.array([["a"], ["b"]], type=items_field.type),
                pa.array([1, 2], type=pa.int64()),
            ],
            schema=pa.schema([items_field, score_field]),
        ),
    )

    session.assign("src", out="out", items=["x"])
    out = session.to_arrow("out")

    assert out.column_names == ["items", "score"]
    # The replacement uses the computed result and its result type; the source
    # list type and child metadata are not restored for a changed column.
    assert out.column("items").type == pa.large_list(pa.field("item", pa.large_string()))
    assert out.column("items").to_pylist() == [["x"], ["x"]]
    assert out.schema.field("items").metadata == items_field.metadata
    assert out.schema.field("score").equals(score_field, check_metadata=True)
    assert out.column("score").to_pylist() == [1, 2]


def test_assign_preserves_source_and_catalog_when_evaluation_fails() -> None:
    """Failed evaluation publishes nothing and keeps the existing source."""

    items_field = pa.field(
        "items",
        pa.list_(pa.field("element", pa.string(), metadata={b"child": b"keep"})),
        metadata={b"label": b"Items"},
    )
    session = Session()
    session.load(
        "src",
        pa.Table.from_arrays(
            [pa.array([["a"], ["b"]], type=items_field.type), pa.array([1, 2], type=pa.int64())],
            schema=pa.schema([items_field, pa.field("score", pa.int64())]),
        ),
    )
    session.load("existing", pa.table({"k": [1]}))
    source = session.to_arrow("src")

    with pytest.raises(ValueError, match="Unsupported function"):
        session.assign("src", out="out", bad="unknown_func(score)")

    with pytest.raises(KeyError):
        session.to_arrow("out")

    failed = session.to_arrow("src")
    assert failed.schema.field("items").equals(items_field, check_metadata=True)
    assert failed.column("items").to_pylist() == source.column("items").to_pylist()
    assert failed.column("score").to_pylist() == source.column("score").to_pylist()
    assert session.to_arrow("existing").column("k").to_pylist() == [1]


def test_assign_rejects_dictionary_reserved_out_and_preserves_catalog() -> None:
    """A refused publication target leaves the source and catalog unchanged."""

    session = Session()
    session.load("src", pa.table({"score": [1, 2]}))
    source = session.to_arrow("src")

    with pytest.raises(ValueError, match="Cannot overwrite dictionary table"):
        session.assign("src", out="DICTIONARY.TABLES", score="score * 2")

    with pytest.raises(KeyError):
        session.to_arrow("DICTIONARY.TABLES")

    failed = session.to_arrow("src")
    assert failed.equals(source)
    assert failed.column("score").to_pylist() == [1, 2]


def test_assign_overwrites_existing_out_and_updates_source_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing out names are overwritten and out=None updates the source."""

    session = Session()
    session.load("src", pa.table({"score": [1, 2]}))
    session.load("other", pa.table({"k": [9]}))

    session.assign("src", out="other", score="score * 10")
    overwritten = session.to_arrow("other")
    assert overwritten.column_names == ["score"]
    assert overwritten.column("score").to_pylist() == [10, 20]
    assert session.to_arrow("src").column("score").to_pylist() == [1, 2]

    session.assign("src", score="score + 1")
    updated = session.to_arrow("src")
    assert updated.column("score").to_pylist() == [2, 3]
    assert "score" in updated.column_names

    # Construction failure: the owning finalize boundary wraps the real
    # implementation and returns a row-mismatched table; the row-count guard
    # raises before Session.load, so the catalog and source stay unchanged.
    real_finalize = limulus.session.preserve_arrow_metadata

    def finalize_with_extra_row(source: pa.Table, target: pa.Table) -> pa.Table:
        finalized = real_finalize(source, target)
        return pa.concat_tables([finalized, finalized.slice(0, 1)])

    monkeypatch.setattr(limulus.session, "preserve_arrow_metadata", finalize_with_extra_row)

    with pytest.raises(ValueError, match="row count mismatch"):
        session.assign("src", out="failed", score="score * 2")

    with pytest.raises(KeyError):
        session.to_arrow("failed")

    assert session.to_arrow("src").column("score").to_pylist() == [2, 3]


def test_assign_assembly_maps_case_distinct_names_exactly() -> None:
    list_field = pa.field(
        "X",
        pa.list_(pa.field("element", pa.string(), metadata={b"child": b"keep"})),
        metadata={b"label": b"Items"},
    )
    numeric_field = pa.field("x", pa.int64(), metadata={b"label": b"Count"})
    name_field = pa.field("name", pa.string())
    source = pa.Table.from_arrays(
        [
            pa.array([["a"], ["b"]], type=list_field.type),
            pa.array([1, 2], type=numeric_field.type),
            pa.array(["Alice", "Bob"], type=name_field.type),
        ],
        schema=pa.schema([list_field, numeric_field, name_field]),
    )
    session = Session()
    session.load("src", source)

    # An unrelated new column must not disturb either case-distinct column.
    session.assign("src", out="with_y", y=3.5)
    out_y = session.to_arrow("with_y")
    assert out_y.column_names == ["X", "x", "name", "y"]
    assert out_y.schema.field("X") == list_field
    assert out_y.schema.field("x") == numeric_field
    assert out_y.column("X").to_pylist() == [["a"], ["b"]]
    assert out_y.column("x").to_pylist() == [1, 2]

    # Replacing exact x keeps X untouched and adopts the x computation.
    session.assign("src", out="with_x", x="x * 10")
    out_x = session.to_arrow("with_x")
    assert out_x.column_names == ["X", "x", "name"]
    assert out_x.schema.field("X") == list_field
    assert out_x.column("X").to_pylist() == [["a"], ["b"]]
    assert out_x.column("x").to_pylist() == [10, 20]
    assert out_x.column("x").type == pa.int64()


def _assert_column_reuses_source_buffers(
    out_column: pa.ChunkedArray,
    source_column: pa.ChunkedArray,
) -> None:
    """Asserts chunk boundaries and original buffer addresses are reused."""

    assert out_column.num_chunks == source_column.num_chunks
    for out_chunk, source_chunk in zip(out_column.chunks, source_column.chunks):
        assert out_chunk.equals(source_chunk)
        got_buffers = out_chunk.buffers()
        want_buffers = source_chunk.buffers()
        assert len(got_buffers) == len(want_buffers)
        for got_buffer, want_buffer in zip(got_buffers, want_buffers):
            if want_buffer is None:
                assert got_buffer is None
            else:
                assert got_buffer is not None
                assert got_buffer.address == want_buffer.address
                assert got_buffer.size == want_buffer.size


def test_assign_preserves_untouched_null_empty_list_and_chunk_boundaries() -> None:
    tags_field = pa.field(
        "tags",
        pa.list_(pa.field("element", pa.string(), metadata={b"child": b"keep"})),
        metadata={b"label": b"Tags"},
    )
    name_field = pa.field("name", pa.string(), nullable=False, metadata={b"label": b"Name"})
    fields = [tags_field, name_field]

    parts = [
        pa.Table.from_arrays(
            [
                pa.array([["a"], [], None], type=tags_field.type),
                pa.array(["Alice", "Bob", "Cara"], type=name_field.type),
            ],
            schema=pa.schema(fields),
        ),
        pa.Table.from_arrays(
            [
                pa.array([[None], ["d"], []], type=tags_field.type),
                pa.array(["Dave", "Eve", "Fay"], type=name_field.type),
            ],
            schema=pa.schema(fields),
        ),
    ]
    session = Session()
    session.load("src", pa.concat_tables(parts))
    source = session.to_arrow("src")
    assert source.column("tags").num_chunks == 2

    session.assign("src", out="out", flag=1)
    out = session.to_arrow("out")

    assert out.column_names == ["tags", "name", "flag"]
    assert out.num_rows == source.num_rows
    assert out.schema.field("tags").equals(tags_field, check_metadata=True)
    assert out.schema.field("name").equals(name_field, check_metadata=True)
    assert out.column("tags").to_pylist() == source.column("tags").to_pylist()
    assert out.column("name").to_pylist() == source.column("name").to_pylist()
    _assert_column_reuses_source_buffers(out.column("tags"), source.column("tags"))
    _assert_column_reuses_source_buffers(out.column("name"), source.column("name"))


def test_assign_preserves_untouched_dictionary_struct_and_string_types() -> None:
    dict_field = pa.field(
        "grade",
        pa.dictionary(pa.int32(), pa.string()),
        metadata={b"label": b"Grade"},
    )
    struct_field = pa.field(
        "contact",
        pa.struct([pa.field("code", pa.int32(), metadata={b"child": b"code"})]),
        metadata={b"label": b"Contact"},
    )
    note_field = pa.field("note", pa.string(), metadata={b"label": b"Note"})
    source_schema = pa.schema([dict_field, struct_field, note_field])
    session = Session()
    session.load(
        "src",
        pa.Table.from_arrays(
            [
                pa.array(["a", None, "b"], type=dict_field.type),
                pa.array([{"code": 1}, None, {"code": None}], type=struct_field.type),
                pa.array(["x", None, "z"], type=note_field.type),
            ],
            schema=source_schema,
        ),
    )
    source = session.to_arrow("src")

    session.assign("src", out="out", flag=2)
    out = session.to_arrow("out")

    assert out.column_names == ["grade", "contact", "note", "flag"]
    assert out.num_rows == source.num_rows
    assert out.schema.field("grade").equals(dict_field, check_metadata=True)
    assert out.schema.field("contact").equals(struct_field, check_metadata=True)
    assert out.schema.field("note").equals(note_field, check_metadata=True)
    for column_name in ("grade", "contact", "note"):
        assert out.column(column_name).to_pylist() == source.column(column_name).to_pylist()
        _assert_column_reuses_source_buffers(out.column(column_name), source.column(column_name))


def test_assign_preserves_zero_row_schema_of_untouched_columns() -> None:
    list_field = pa.field(
        "tags",
        pa.list_(pa.field("element", pa.string(), metadata={b"child": b"keep"})),
        metadata={b"label": b"Tags"},
    )
    struct_field = pa.field(
        "contact",
        pa.struct([pa.field("code", pa.int32(), metadata={b"child": b"code"})]),
    )
    dict_field = pa.field("grade", pa.dictionary(pa.int32(), pa.string()), metadata={b"label": b"Grade"})
    string_field = pa.field("note", pa.string(), nullable=False)
    source_schema = pa.schema(
        [list_field, struct_field, dict_field, string_field],
        metadata={b"memlabel": b"Empty"},
    )
    session = Session()
    session.load(
        "src",
        pa.Table.from_arrays(
            [pa.array([], type=field.type) for field in source_schema],
            schema=source_schema,
        ),
    )
    source = session.to_arrow("src")

    session.assign("src", out="out", flag=3)
    out = session.to_arrow("out")

    assert out.num_rows == 0
    assert out.column_names == ["tags", "contact", "grade", "note", "flag"]
    assert out.schema.field("tags").equals(list_field, check_metadata=True)
    assert out.schema.field("contact").equals(struct_field, check_metadata=True)
    assert out.schema.field("grade").equals(dict_field, check_metadata=True)
    assert out.schema.field("note").equals(string_field, check_metadata=True)
    assert out.schema.metadata == source_schema.metadata
