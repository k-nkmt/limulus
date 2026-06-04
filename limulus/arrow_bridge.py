from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pyarrow as pa

from .naming import _column_key


def materialize_rebuilt_table(
    rows: Sequence[Mapping[str, Any]],
    source_table: pa.Table,
    *,
    type_hints: Mapping[str, pa.DataType] | None = None,
) -> pa.Table:
    rebuilt = pa.Table.from_pylist([dict(row) for row in rows])
    rebuilt = cast_rebuilt_table_to_hints(rebuilt, type_hints=type_hints)
    return restore_arrow_schema_from_sources(rebuilt, source_tables=(source_table,))


def preserve_arrow_metadata(source: pa.Table, target: pa.Table) -> pa.Table:
    return restore_arrow_schema_from_sources(target, source_tables=(source,))


def polars_result_to_arrow(result: Any, *source_tables: pa.Table) -> pa.Table:
    if hasattr(result, "collect"):
        result = result.collect()
    return restore_arrow_schema_from_sources(result.to_arrow(), source_tables=source_tables)


def restore_arrow_schema_from_sources(target: pa.Table, *, source_tables: Sequence[pa.Table]) -> pa.Table:
    if not source_tables:
        return target

    target_schema = target.schema
    source_fields: dict[str, pa.Field] = {}
    ambiguous_fields: set[str] = set()

    for source in source_tables:
        for field in source.schema:
            field_key = _column_key(field.name)
            existing_field = source_fields.get(field_key)
            if existing_field is None:
                source_fields[field_key] = field
                continue
            if existing_field.metadata != field.metadata:
                ambiguous_fields.add(field_key)

    fields = []
    for field in target_schema:
        field_key = _column_key(field.name)
        source_field = None if field_key in ambiguous_fields else source_fields.get(field_key)
        if source_field is None or source_field.metadata is None:
            fields.append(field)
            continue
        fields.append(field.with_metadata(source_field.metadata))

    schema_metadata = target_schema.metadata
    if len(source_tables) == 1 and source_tables[0].schema.metadata is not None:
        schema_metadata = source_tables[0].schema.metadata

    schema = pa.schema(fields, metadata=schema_metadata)
    return pa.Table.from_arrays([target.column(index) for index in range(target.num_columns)], schema=schema)


def cast_rebuilt_table_to_hints(
    target: pa.Table,
    *,
    type_hints: Mapping[str, pa.DataType] | None,
) -> pa.Table:
    if not type_hints:
        return target

    normalized_type_hints = {_column_key(name): dtype for name, dtype in type_hints.items()}
    fields: list[pa.Field] = []
    arrays: list[pa.Array] = []

    for index, field in enumerate(target.schema):
        hint = normalized_type_hints.get(_column_key(field.name))
        array = target.column(index)
        if hint is not None and field.type != hint:
            try:
                array = array.cast(hint)
                field = pa.field(field.name, hint, nullable=field.nullable, metadata=field.metadata)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError, TypeError, ValueError):
                pass
        fields.append(field)
        arrays.append(array)

    return pa.Table.from_arrays(arrays, schema=pa.schema(fields, metadata=target.schema.metadata))