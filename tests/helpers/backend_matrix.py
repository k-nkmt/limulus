from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pyarrow as pa

from limulus.models import DataSetRef


def _arrow_payload(payload: object) -> pa.Table:
    if isinstance(payload, pa.Table):
        return payload
    if isinstance(payload, Mapping):
        return pa.table(payload)
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return pa.Table.from_pylist([dict(row) for row in payload])
    raise TypeError(f"unsupported arrow payload type: {type(payload)!r}")


def build_session_payload(input_kind: str, payload: object) -> object:
    if input_kind == "arrow_table":
        return _arrow_payload(payload)
    return payload


def build_dataset_ref(name: str, input_kind: str, payload: object) -> DataSetRef:
    return DataSetRef(
        kind=input_kind,
        location=f"dataset://{name}",
        payload=build_session_payload(input_kind, payload),
    )