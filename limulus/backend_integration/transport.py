from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..models import DataSetRef


_STANDARD_TRANSPORT_KIND = "arrow_table"
_STANDARD_TRANSPORT_PATH = "standard"
_SPECIAL_TRANSPORT_PATH = "special"
_REJECTED_TRANSPORT_PATH = "rejected"
_PREPARED_SET_ROWS_MARKER = "#prepared_set_rows"
_PREPARED_MERGE_ROWS_MARKER = "#prepared_merge_rows"


def _normalize_transport_kind(kind: str | None) -> str:
    if not isinstance(kind, str):
        return ""
    return kind.strip().lower()


def _has_prepared_transport_marker(location: str | None) -> bool:
    if not isinstance(location, str):
        return False
    return (
        _PREPARED_SET_ROWS_MARKER in location
        or _PREPARED_MERGE_ROWS_MARKER in location
    )


def _has_prepared_transport_rows(payload: Any) -> bool:
    return (
        isinstance(payload, Sequence)
        and not isinstance(payload, (str, bytes, bytearray))
        and all(isinstance(item, Mapping) for item in payload)
    )


def _has_prepared_transport_arrow(payload: Any) -> bool:
    return hasattr(payload, "__arrow_c_stream__")


def _classify_transport_input(dataset_ref: DataSetRef) -> str:
    normalized_kind = _normalize_transport_kind(dataset_ref.kind)
    if _has_prepared_transport_marker(dataset_ref.location):
        if normalized_kind == _STANDARD_TRANSPORT_KIND and _has_prepared_transport_arrow(dataset_ref.payload):
            return _SPECIAL_TRANSPORT_PATH
        if normalized_kind == "memory" and _has_prepared_transport_rows(dataset_ref.payload):
            return _SPECIAL_TRANSPORT_PATH

    if normalized_kind == _STANDARD_TRANSPORT_KIND:
        return _STANDARD_TRANSPORT_PATH

    return _REJECTED_TRANSPORT_PATH


def _uses_prepared_merge_transport(dataset_ref: DataSetRef) -> bool:
    return _classify_transport_input(dataset_ref) == _SPECIAL_TRANSPORT_PATH and (
        _PREPARED_MERGE_ROWS_MARKER in (dataset_ref.location or "")
    )


def _uses_prepared_set_transport(dataset_ref: DataSetRef) -> bool:
    return _classify_transport_input(dataset_ref) == _SPECIAL_TRANSPORT_PATH and (
        _PREPARED_SET_ROWS_MARKER in (dataset_ref.location or "")
    )


__all__ = [
    "_classify_transport_input",
    "_has_prepared_transport_arrow",
    "_has_prepared_transport_marker",
    "_has_prepared_transport_rows",
    "_normalize_transport_kind",
    "_PREPARED_MERGE_ROWS_MARKER",
    "_PREPARED_SET_ROWS_MARKER",
    "_REJECTED_TRANSPORT_PATH",
    "_SPECIAL_TRANSPORT_PATH",
    "_STANDARD_TRANSPORT_KIND",
    "_STANDARD_TRANSPORT_PATH",
    "_uses_prepared_merge_transport",
    "_uses_prepared_set_transport",
]