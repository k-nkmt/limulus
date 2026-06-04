from __future__ import annotations


def _column_key(name: str) -> str:
    return name.strip().upper()


def _dataset_key(name: str) -> str:
    normalized = name.strip().upper()
    if normalized.startswith("WORK."):
        return normalized[5:]
    return normalized


__all__ = ["_column_key", "_dataset_key"]