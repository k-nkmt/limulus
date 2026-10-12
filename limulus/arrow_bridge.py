from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import pyarrow as pa

from .naming import _column_key
from .models import SourceMetadataLineage, SourceMetadataSnapshot


@dataclass(frozen=True)
class MergeSchemaOwnership:
    """Immutable type ownership derived from option-shaped MERGE schemas."""

    type_hints: Mapping[str, pa.DataType] = field(
        default_factory=lambda: MappingProxyType({})
    )
    column_order: tuple[str, ...] = ()
    ambiguous_non_by_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "type_hints", MappingProxyType(dict(self.type_hints)))
        object.__setattr__(self, "column_order", tuple(self.column_order))
        object.__setattr__(
            self,
            "ambiguous_non_by_fields",
            tuple(self.ambiguous_non_by_fields),
        )


@dataclass(frozen=True)
class OutputLineageProjectionResult:
    """Schema-only output lineage projection result.

    ``lineage`` contains copied snapshots only; it never owns target arrays or
    row data.  ``invariant_error`` is reserved for a duplicate that reaches
    this projector despite the pre-native projection validator.
    """

    lineage: SourceMetadataLineage | None = None
    invariant_error: str | None = None


def _project_output_column_names(
    names: Sequence[str],
    *,
    stages: Sequence[Any],
) -> tuple[str, ...]:
    """Apply output stages to logical names without touching target arrays."""

    projected = [str(name) for name in names]
    for stage in stages:
        keep = tuple(getattr(stage, "keep", ()) or ())
        drop = tuple(getattr(stage, "drop", ()) or ())
        rename = dict(getattr(stage, "rename", {}) or {})
        if keep:
            keep_keys = {_column_key(name) for name in keep}
            projected = [name for name in projected if _column_key(name) in keep_keys]
        if drop:
            drop_keys = {_column_key(name) for name in drop}
            projected = [name for name in projected if _column_key(name) not in drop_keys]
        if rename:
            normalized_rename = {
                _column_key(source): str(target)
                for source, target in rename.items()
            }
            projected = [
                normalized_rename.get(_column_key(name), name)
                for name in projected
            ]
    return tuple(projected)


def project_lineage_for_output(
    lineage: SourceMetadataLineage,
    *,
    stages: Sequence[Any],
    final_columns: Sequence[str],
) -> OutputLineageProjectionResult:
    """Project schema-only source lineage onto one final output target.

    Each stage is applied in the fixed KEEP -> DROP -> RENAME order.  The
    implementation transforms Arrow fields only; field type, nullability,
    field metadata, schema metadata, source order, and dataset-ref metadata
    are copied without touching output arrays or buffers.
    """

    if not lineage.ordered_sources:
        return OutputLineageProjectionResult()

    final_keys = {_column_key(name) for name in final_columns}
    projected_snapshots: list[SourceMetadataSnapshot] = []

    for snapshot in lineage.ordered_sources:
        fields = list(snapshot.schema)
        for stage in stages:
            keep = tuple(getattr(stage, "keep", ()) or ())
            drop = tuple(getattr(stage, "drop", ()) or ())
            rename = dict(getattr(stage, "rename", {}) or {})

            if keep:
                keep_keys = {_column_key(name) for name in keep}
                fields = [field for field in fields if _column_key(field.name) in keep_keys]
            if drop:
                drop_keys = {_column_key(name) for name in drop}
                fields = [field for field in fields if _column_key(field.name) not in drop_keys]
            if rename:
                normalized_rename = {
                    _column_key(source): str(target)
                    for source, target in rename.items()
                }
                fields = [
                    pa.field(
                        normalized_rename.get(_column_key(field.name), field.name),
                        field.type,
                        nullable=field.nullable,
                        metadata=field.metadata,
                    )
                    for field in fields
                ]

        fields = [field for field in fields if _column_key(field.name) in final_keys]
        seen: set[str] = set()
        for field in fields:
            field_key = _column_key(field.name)
            if field_key in seen:
                return OutputLineageProjectionResult(
                    invariant_error=(
                        "output lineage projection produced a duplicate canonical field "
                        f"{field.name!r} in source {snapshot.source_name!r}"
                    )
                )
            seen.add(field_key)

        projected_snapshots.append(
            SourceMetadataSnapshot(
                source_ordinal=snapshot.source_ordinal,
                source_name=snapshot.source_name,
                schema=pa.schema(fields, metadata=snapshot.schema.metadata),
                dataset_ref_metadata=snapshot.dataset_ref_metadata,
            )
        )

    return OutputLineageProjectionResult(
        lineage=SourceMetadataLineage(tuple(projected_snapshots))
    )


def analyze_merge_schema_ownership(
    lineage: SourceMetadataLineage,
    *,
    by_keys: Sequence[str],
) -> MergeSchemaOwnership:
    """Return deterministic MERGE type owners from schema-only lineage.

    Field comparison is case-insensitive, while the first provider's spelling
    is retained for the returned type-hint keys and column order.  Metadata is
    deliberately not copied: only ``pa.DataType`` values cross this boundary.
    Mixed string BY keys use large_string with the first provider's ownership.
    ``FIRST.`` and ``LAST.`` fields are runtime-owned helpers and are excluded
    even if a defensive direct caller supplies them in a lineage snapshot.
    """

    normalized_by_keys = {_column_key(name) for name in by_keys}
    owner_fields: dict[str, pa.Field] = {}
    provider_counts: dict[str, int] = {}
    column_order: list[str] = []
    first_source_fields: dict[str, pa.Field] = {}
    by_field_types: dict[str, set[pa.DataType]] = {}

    for source_index, snapshot in enumerate(lineage.ordered_sources):
        seen_in_source: set[str] = set()
        for source_field in snapshot.schema:
            field_key = _column_key(source_field.name)
            if field_key.startswith("FIRST.") or field_key.startswith("LAST."):
                continue
            if field_key in normalized_by_keys and not pa.types.is_null(source_field.type):
                by_field_types.setdefault(field_key, set()).add(source_field.type)

            if field_key not in owner_fields:
                owner_fields[field_key] = source_field
                column_order.append(source_field.name)
            if field_key not in seen_in_source:
                provider_counts[field_key] = provider_counts.get(field_key, 0) + 1
                seen_in_source.add(field_key)
            if source_index == 0:
                first_source_fields.setdefault(field_key, source_field)

    type_hints: dict[str, pa.DataType] = {}
    ambiguous_non_by_fields: list[str] = []
    for field_key in owner_fields:
        if field_key in normalized_by_keys:
            first_source_field = first_source_fields.get(field_key)
            if first_source_field is not None:
                type_hints[first_source_field.name] = (
                    pa.large_string()
                    if by_field_types.get(field_key) == {pa.string(), pa.large_string()}
                    else first_source_field.type
                )
            continue

        if provider_counts[field_key] == 1:
            owner_field = owner_fields[field_key]
            type_hints[owner_field.name] = owner_field.type
        else:
            ambiguous_non_by_fields.append(field_key)

    return MergeSchemaOwnership(
        type_hints=type_hints,
        column_order=tuple(column_order),
        ambiguous_non_by_fields=tuple(ambiguous_non_by_fields),
    )


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


def restore_arrow_schema_from_sources(
    target: pa.Table,
    *,
    source_tables: Sequence[pa.Table] = (),
    lineage: SourceMetadataLineage | None = None,
    preparation_kind: str | None = None,
    by_keys: Sequence[str] = (),
) -> pa.Table:
    """Restore Arrow schema metadata without changing buffers or data types.

    A prepared SET or MERGE may collapse several option-shaped source tables
    into one carrier before the native handoff.  When its source lineage is
    available, metadata ownership follows the preparation-specific policy
    while target arrays and data types remain untouched.
    """

    if lineage is not None and lineage.ordered_sources:
        if preparation_kind == "prepared_merge":
            return restore_arrow_schema_from_merge_lineage(
                target,
                lineage=lineage,
                by_keys=by_keys,
            )
        return restore_arrow_schema_from_lineage(target, lineage=lineage, by_keys=by_keys)

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


def restore_arrow_schema_from_lineage(
    target: pa.Table,
    *,
    lineage: SourceMetadataLineage,
    by_keys: Sequence[str] = (),
) -> pa.Table:
    """Apply ordered SET metadata ownership from schema-only snapshots.

    ``by_keys`` is accepted as part of the output-handoff contract for the
    subsequent MERGE policy; SET restoration deliberately treats all fields
    uniformly.  Only Arrow schema objects are inspected, and target arrays
    are reused directly in the rebuilt table.
    """

    del by_keys
    ordered_sources = lineage.ordered_sources
    if not ordered_sources:
        return target

    missing = object()
    first_provider_by_key: dict[str, pa.Field | None] = {}
    for snapshot in ordered_sources:
        for source_field in snapshot.schema:
            field_key = _column_key(source_field.name)
            if field_key not in first_provider_by_key:
                first_provider_by_key[field_key] = source_field

    fields: list[pa.Field] = []
    for target_field in target.schema:
        source_field = first_provider_by_key.get(_column_key(target_field.name), missing)
        if source_field is missing:
            fields.append(target_field)
            continue
        # The first provider owns the field even when it has no metadata;
        # later source occurrences must not fill or overwrite that decision.
        fields.append(target_field.with_metadata(source_field.metadata))

    first_schema_metadata = ordered_sources[0].schema.metadata
    schema = pa.schema(fields, metadata=first_schema_metadata)
    return pa.Table.from_arrays(
        [target.column(index) for index in range(target.num_columns)],
        schema=schema,
    )


def restore_arrow_schema_from_merge_lineage(
    target: pa.Table,
    *,
    lineage: SourceMetadataLineage,
    by_keys: Sequence[str] = (),
) -> pa.Table:
    """Restore MERGE metadata using first-source and unique-owner rules.

    The first DSL source owns the output schema and every BY field.  A
    non-BY field is restored only when exactly one lineage source provides it;
    duplicate non-BY columns are rejected during MERGE preparation and are
    therefore intentionally left untouched if a caller reaches this handoff.
    Target arrays are borrowed directly so this rebuild cannot materialize
    rows or alter Arrow types.
    """

    ordered_sources = lineage.ordered_sources
    if not ordered_sources:
        return target

    by_keys_normalized = {_column_key(name) for name in by_keys}
    first_source_fields = {
        _column_key(source_field.name): source_field
        for source_field in ordered_sources[0].schema
    }
    providers_by_key: dict[str, list[pa.Field]] = {}
    for snapshot in ordered_sources:
        for source_field in snapshot.schema:
            providers_by_key.setdefault(_column_key(source_field.name), []).append(source_field)

    fields: list[pa.Field] = []
    for target_field in target.schema:
        field_key = _column_key(target_field.name)
        source_field: pa.Field | None
        if field_key in by_keys_normalized:
            source_field = first_source_fields.get(field_key)
        else:
            providers = providers_by_key.get(field_key, ())
            source_field = providers[0] if len(providers) == 1 else None
        if source_field is None:
            fields.append(target_field)
            continue
        fields.append(target_field.with_metadata(source_field.metadata))

    schema = pa.schema(fields, metadata=ordered_sources[0].schema.metadata)
    return pa.Table.from_arrays(
        [target.column(index) for index in range(target.num_columns)],
        schema=schema,
    )


def cast_rebuilt_table_to_hints(
    target: pa.Table,
    *,
    type_hints: Mapping[str, pa.DataType] | None,
) -> pa.Table:
    return _materialize_typed_table(
        target,
        type_hints=type_hints,
        column_order=(),
        preserve_value_types=False,
    )


def materialize_merge_carrier(
    rows_or_table: pa.Table | Sequence[Mapping[str, Any]],
    *,
    type_hints: Mapping[str, pa.DataType] | None = None,
    column_order: Sequence[str] = (),
) -> pa.Table:
    """Build a MERGE carrier while retaining owned source field types.

    ``rows_or_table`` may already be an Arrow table produced by an alignment
    stage or a row sequence from a defensive materializer.  Only data types
    and deterministic field order cross this helper boundary; field and
    schema metadata remain owned by the later restoration stage.
    """

    if isinstance(rows_or_table, pa.Table):
        target = rows_or_table
    else:
        target = pa.Table.from_pylist([dict(row) for row in rows_or_table])
    return _materialize_typed_table(
        target,
        type_hints=type_hints,
        column_order=column_order,
        preserve_value_types=True,
    )


def _materialize_typed_table(
    target: pa.Table,
    *,
    type_hints: Mapping[str, pa.DataType] | None,
    column_order: Sequence[str],
    preserve_value_types: bool,
) -> pa.Table:
    if not type_hints:
        return target

    normalized_type_hints = {_column_key(name): dtype for name, dtype in type_hints.items()}
    target_fields = tuple(target.schema)

    fields: list[pa.Field] = []
    arrays: list[Any] = []

    def append_field(*, field: pa.Field, array: Any) -> None:
        hint = normalized_type_hints.get(_column_key(field.name))
        can_complete_type = (
            not preserve_value_types
            or field.type == pa.null()
            or _is_null_only_array(array)
        )
        if hint is not None and field.type != hint and can_complete_type:
            try:
                array = array.cast(hint)
                field = pa.field(
                    field.name,
                    hint,
                    nullable=field.nullable,
                    metadata=field.metadata,
                )
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError, TypeError, ValueError):
                # Preserve existing values/types when a non-compatible value
                # cannot be coerced; this helper is not a general cast policy.
                pass
        fields.append(field)
        arrays.append(array)

    # Preserve the pre-materializer order.  In particular, the BY path emits
    # BY keys before non-key fields and runtime helpers; ownership order must
    # not reorder those already-present columns.
    for index, field in enumerate(target_fields):
        append_field(field=field, array=target.column(index))

    existing_keys = {_column_key(field.name) for field in target_fields}
    emitted_missing_keys: set[str] = set()
    for owner_name in column_order:
        field_key = _column_key(owner_name)
        if field_key in existing_keys or field_key in emitted_missing_keys:
            continue
        hint = normalized_type_hints.get(field_key)
        if hint is None:
            continue
        emitted_missing_keys.add(field_key)
        fields.append(pa.field(owner_name, hint, nullable=True))
        arrays.append(pa.array([None] * int(target.num_rows), type=hint))

    return pa.Table.from_arrays(
        arrays,
        schema=pa.schema(fields, metadata=target.schema.metadata),
    )


def _is_null_only_array(array: Any) -> bool:
    """Return whether an existing Arrow array has no value-bearing entries."""

    try:
        length = len(array)
    except TypeError:
        return False
    if length == 0:
        return True

    null_count = getattr(array, "null_count", None)
    if null_count is not None:
        return int(null_count) == length
    try:
        return all(value is None for value in array.to_pylist())
    except (AttributeError, TypeError, ValueError):
        return False
