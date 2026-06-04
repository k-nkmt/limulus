from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class ArrowRowCursorProtocol(Protocol):
    row_count: int

    def bind(self, row_index: int) -> None:
        ...

    def value_at_slot(self, slot_index: int) -> object:
        ...

    def snapshot(self) -> dict[str, object]:
        ...


class ArrowTableRowCursor:
    """Arrow table backed row cursor for slot-index reads."""

    def __init__(self, table: Any) -> None:
        self._table = table
        self._names = tuple(getattr(getattr(table, "schema", None), "names", ()) or ())
        self._columns = tuple(table.column(index) for index in range(len(self._names)))
        self.row_count = int(getattr(table, "num_rows", 0) or 0)
        self._bound_row_index = 0

    def bind(self, row_index: int) -> None:
        if row_index < 0 or row_index >= self.row_count:
            raise IndexError(row_index)
        self._bound_row_index = row_index

    def value_at_slot(self, slot_index: int) -> object:
        if slot_index < 0 or slot_index >= len(self._columns):
            raise IndexError(slot_index)
        return self._columns[slot_index][self._bound_row_index].as_py()

    def snapshot(self) -> dict[str, object]:
        return {
            name: self.value_at_slot(slot_index)
            for slot_index, name in enumerate(self._names)
        }


def create_arrow_row_cursor(table: Any) -> ArrowRowCursorProtocol:
    return ArrowTableRowCursor(table)


__all__ = [
    "ArrowRowCursorProtocol",
    "ArrowTableRowCursor",
    "create_arrow_row_cursor",
]
