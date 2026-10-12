"""Shared prepared-input and Arrow-export primitives for native bridges."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..models import DataSetRef, Diagnostic, PreparationKind
from .transport import (
    _STANDARD_TRANSPORT_KIND,
    _STANDARD_TRANSPORT_PATH,
    _classify_transport_input,
    derive_preparation_kind,
)


def dataset_options_are_passthrough(options: object) -> bool:
    """Return whether native input-option ownership has consumed all fields."""

    if options is None:
        return True
    return (
        getattr(options, "in_var", None) is None
        and not getattr(options, "keep_vars", ())
        and not getattr(options, "drop_vars", ())
        and getattr(options, "where_expr", None) is None
        and not getattr(options, "rename_map", ())
        and getattr(options, "firstobs", None) is None
        and getattr(options, "obs", None) is None
    )


def validate_prepared_input_contract(
    *,
    resolved_inputs: Mapping[str, DataSetRef],
    preparation_kind: PreparationKind,
    source_statement_kind: str | None,
    source_refs: Sequence[Any],
    validate_collapsed_shape: bool = True,
) -> list[Diagnostic]:
    """Validate carrier provenance and consumed source options identically."""

    carrier_kind = derive_preparation_kind(resolved_inputs.values())
    context_kind = preparation_kind
    if context_kind != "none" and context_kind != carrier_kind:
        return [
            Diagnostic(
                code="RUNTIME_PREPARED_INPUT_CONTRACT_INVALID",
                severity="error",
                location="statement:source",
                message=(
                    "Native ingress preparation kind does not match the input "
                    f"carrier provenance: context={context_kind!r}, "
                    f"carrier={carrier_kind!r}."
                ),
            )
        ]

    for source_ref in source_refs:
        options = getattr(source_ref, "options", None)
        if dataset_options_are_passthrough(options):
            continue
        source_name = str(getattr(source_ref, "name", "<unknown>"))
        if carrier_kind == "none":
            return [
                Diagnostic(
                    code="RUNTIME_INPUT_OPTION_NOT_PREPARED",
                    severity="error",
                    location=f"dataset:{source_name}",
                    message=(
                        f"Input dataset options for '{source_name}' must be consumed "
                        "by Arrow input preparation before native row execution."
                    ),
                )
            ]
        return [
            Diagnostic(
                code="RUNTIME_PREPARED_INPUT_OPTION_REMAINING",
                severity="error",
                location=f"dataset:{source_name}",
                message=(
                    f"Prepared input '{source_name}' still carries source-local "
                    "dataset options at native ingress."
                ),
            )
        ]

    if carrier_kind == "none" or not validate_collapsed_shape:
        return []

    if source_statement_kind is None:
        return [
            Diagnostic(
                code="RUNTIME_PREPARED_INPUT_CONTRACT_INVALID",
                severity="error",
                location="statement:source",
                message=(
                    "A prepared input carrier requires a collapsed SET "
                    "statement at native ingress."
                ),
            )
        ]

    if source_statement_kind != "SET" or len(source_refs) != 1:
        return [
            Diagnostic(
                code="RUNTIME_PREPARED_INPUT_CONTRACT_INVALID",
                severity="error",
                location="statement:source",
                message=(
                    "Prepared input requires exactly one collapsed SET source "
                    "reference."
                ),
            )
        ]

    carrier_name = str(getattr(source_refs[0], "name", ""))
    carrier = next(
        (
            dataset_ref
            for dataset_name, dataset_ref in resolved_inputs.items()
            if str(dataset_name).casefold() == carrier_name.casefold()
        ),
        None,
    )
    carrier_kind_on_carrier = getattr(
        getattr(carrier, "preparation", None), "kind", "none"
    )
    if carrier is None or carrier_kind_on_carrier != carrier_kind:
        return [
            Diagnostic(
                code="RUNTIME_PREPARED_INPUT_CONTRACT_INVALID",
                severity="error",
                location=f"dataset:{carrier_name or '<unknown>'}",
                message=(
                    f"Collapsed SET source '{carrier_name}' does not carry the "
                    f"expected {carrier_kind} preparation kind."
                ),
            )
        ]

    return []


def export_arrow_inputs(
    resolved_inputs: Mapping[str, DataSetRef],
) -> tuple[dict[str, Any], list[Diagnostic]]:
    """Export ordered resolved inputs using the established bridge diagnostics."""

    input_streams: dict[str, Any] = {}
    diagnostics: list[Diagnostic] = []
    for dataset_name, dataset_ref in resolved_inputs.items():
        transport_path = _classify_transport_input(dataset_ref)
        if transport_path != _STANDARD_TRANSPORT_PATH:
            diagnostics.append(
                Diagnostic(
                    code="RUNTIME_RUST_BRIDGE_INPUT_NOT_ARROW",
                    severity="error",
                    location=f"dataset:{dataset_name}",
                    message=(
                        f"Rust backend requires {_STANDARD_TRANSPORT_KIND} input: "
                        f"{dataset_name}"
                    ),
                )
            )
            return input_streams, diagnostics

        payload = dataset_ref.payload
        if payload is None or not hasattr(payload, "__arrow_c_stream__"):
            diagnostics.append(
                Diagnostic(
                    code="RUNTIME_RUST_BRIDGE_ARROW_EXPORT_MISSING",
                    severity="error",
                    location=f"dataset:{dataset_name}",
                    message=(
                        f"{_STANDARD_TRANSPORT_KIND} payload does not expose "
                        f"__arrow_c_stream__: {dataset_name}"
                    ),
                )
            )
            return input_streams, diagnostics
        try:
            input_streams[dataset_name] = payload.__arrow_c_stream__()
        except Exception as error:
            diagnostics.append(
                Diagnostic(
                    code="RUNTIME_RUST_BRIDGE_ARROW_EXPORT_FAILED",
                    severity="error",
                    location=f"dataset:{dataset_name}",
                    message=(
                        f"Failed to export Arrow C Data stream for '{dataset_name}': "
                        f"{error}"
                    ),
                )
            )
            return input_streams, diagnostics

    return input_streams, diagnostics


__all__ = [
    "dataset_options_are_passthrough",
    "export_arrow_inputs",
    "validate_prepared_input_contract",
]
