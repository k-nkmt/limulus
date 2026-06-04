from __future__ import annotations

from typing import Any


def _decode_metadata_value(value: bytes | None) -> str | None:
    if value is None:
        return None
    return value.decode("utf-8", errors="replace")


def _case_context(*, case_id: str, backend: str | None = None, runtime_backend: str | None = None) -> str:
    parts = [f"case={case_id}"]
    if backend is not None:
        parts.append(f"expected_backend={backend}")
    if runtime_backend is not None:
        parts.append(f"runtime_backend={runtime_backend}")
    return ", ".join(parts)


def assert_rows_match(
    actual: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    *,
    case_id: str,
    backend: str | None = None,
) -> None:
    assert actual == expected, (
        f"{_case_context(case_id=case_id, backend=backend)}: "
        f"expected rows {expected!r}, got {actual!r}"
    )


def assert_backend_choice(
    actual: str,
    expected: str,
    *,
    case_id: str,
    runtime_backend: str,
) -> None:
    assert actual == expected, (
        f"{_case_context(case_id=case_id, backend=expected, runtime_backend=runtime_backend)}: "
        f"got backend {actual!r}"
    )


def assert_table_metadata(
    table: Any,
    *,
    memlabel: str | None,
    column_labels: dict[str, str],
    case_id: str,
    backend: str | None = None,
) -> None:
    schema_metadata = getattr(getattr(table, "schema", None), "metadata", None) or {}
    actual_dataset_label = _decode_metadata_value(schema_metadata.get(b"memlabel"))
    assert actual_dataset_label == memlabel, (
        f"{_case_context(case_id=case_id, backend=backend)}: "
        f"expected memlabel {memlabel!r}, got {actual_dataset_label!r}"
    )

    for column_name, expected_label in column_labels.items():
        field = table.schema.field(column_name)
        actual_label = _decode_metadata_value((field.metadata or {}).get(b"label"))
        assert actual_label == expected_label, (
            f"{_case_context(case_id=case_id, backend=backend)}: "
            f"expected column label for {column_name!r} = {expected_label!r}, got {actual_label!r}"
        )
