from __future__ import annotations


def _column_key(name: str) -> str:
    return name.strip().upper()


__all__ = ["_column_key"]