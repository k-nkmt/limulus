from __future__ import annotations

from collections.abc import Iterable

from ..models import DataSetRef, PreparationKind


_STANDARD_TRANSPORT_KIND = "arrow_table"
_STANDARD_TRANSPORT_PATH = "standard"
_REJECTED_TRANSPORT_PATH = "rejected"


def _normalize_transport_kind(kind: str | None) -> str:
    if not isinstance(kind, str):
        return ""
    return kind.strip().lower()


def _classify_transport_input(dataset_ref: DataSetRef) -> str:
    normalized_kind = _normalize_transport_kind(dataset_ref.kind)
    if normalized_kind == _STANDARD_TRANSPORT_KIND:
        return _STANDARD_TRANSPORT_PATH

    return _REJECTED_TRANSPORT_PATH


def derive_preparation_kind(dataset_refs: Iterable[DataSetRef]) -> PreparationKind:
    """Aggregate typed carrier provenance into one transport preparation kind.

    Provenance is read from the typed ``DataSetRef.preparation`` contract only:
    ``location`` is opaque user data and is never scanned.  ``none`` carriers
    do not participate; a single distinct prepared kind becomes the aggregate.
    Mixed kinds cannot be produced by the validated preparation flow, so this
    layer treats them as an upstream contract bug and fails closed instead of
    preferring MERGE over SET.
    """

    observed_kinds: set[str] = set()
    for dataset_ref in dataset_refs:
        provenance = getattr(dataset_ref, "preparation", None)
        kind = getattr(provenance, "kind", "none")
        if kind != "none":
            observed_kinds.add(str(kind))

    if not observed_kinds:
        return "none"
    if len(observed_kinds) == 1:
        return next(iter(observed_kinds))  # type: ignore[return-value]
    raise ValueError(
        "prepared carriers mix preparation kinds "
        f"{sorted(observed_kinds)}; the validated preparation flow cannot "
        "hand off an ambiguous aggregate kind"
    )


__all__ = [
    "_classify_transport_input",
    "_normalize_transport_kind",
    "_REJECTED_TRANSPORT_PATH",
    "_STANDARD_TRANSPORT_KIND",
    "_STANDARD_TRANSPORT_PATH",
    "derive_preparation_kind",
]
