"""Block validation owner for the execution pipeline.

The validation stage is request-scoped: it receives immutable typed
block/catalog snapshots and returns diagnostics; stage tags and block
locations belong to the execution coordinator, not this module.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..io import ExecutorIOService
from ..io_adapters import (
    DataFrameAdapterPandas,
    DataInputAdapterArrow,
    DataOutputAdapterArrow,
)
from ..models import DataSetRef, Diagnostic, DiagnosticLabel, DiagnosticSpan
from ..naming import _column_key
from ..parser import ParsedStatement
from ..syntax.models import (
    AssignStatementSyntax,
    ArrayStatementSyntax,
    BlockSyntax,
    DataStatementSyntax,
    DeferredStatementError,
    DoStatementSyntax,
    IfStatementSyntax,
    LabelStatementSyntax,
    NameListStatementSyntax,
    NameSyntax,
    OutputStatementSyntax,
    RenameStatementSyntax,
    SourceDocument,
    SourceSpan,
    SourceStatementSyntax,
    StatementSyntax,
    SumStatementSyntax,
)
from .contracts import TypedValidationRequest, ValidationResult


_SUPPORTED_FORMATS: tuple[str, ...] = ("polars", "pandas", "arrow_table")


def has_output_projection_collision(columns: Sequence[str]) -> bool:
    """Return whether projected output names collide after column-key folding."""

    seen: set[str] = set()
    for column in columns:
        key = _column_key(column)
        if key in seen:
            return True
        seen.add(key)
    return False


def _validation_diagnostic(
    *,
    code: str,
    message: str,
    span: DiagnosticSpan | None,
    source_text: str | None,
    label_message: str | None = None,
    stage: str = "",
) -> Diagnostic:
    """Build one validation diagnostic from an already-resolved span."""

    resolved_label = label_message or _validate_label_message(code)
    return Diagnostic(
        code=code,
        severity="error",
        message=message,
        stage=stage,
        span=span,
        labels=(
            (DiagnosticLabel(span=span, message=resolved_label),)
            if span is not None
            else ()
        ),
        source_text=source_text if span is not None else None,
    )


def _diagnostic_span_from_source_span(span: SourceSpan) -> DiagnosticSpan:
    """Convert a pure syntax ``SourceSpan`` to the shared diagnostic span DTO.

    The field vocabulary is identical; this conversion keeps the typed syntax
    span type out of the diagnostic transport contract.
    """

    return DiagnosticSpan(
        start=span.start,
        end=span.end,
        line=span.line,
        column=span.column,
        end_line=span.end_line,
        end_column=span.end_column,
        source_id=span.source_id,
    )


def _is_numeric_column_token(token: str) -> bool:
    """Match the RETAIN numeric-word filter without regex.

    Equivalent to the ``[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)`` full match
    over already-typed name values (``5``, ``+5``, ``.5``, ``5.``); ``.`` and
    non-numeric names are not numeric tokens.
    """

    body = token[1:] if token[:1] in ("+", "-") else token
    whole, dot, fraction = body.partition(".")
    if not dot:
        return whole != "" and whole.isdigit()
    if whole == "" and fraction == "":
        return False
    return (whole == "" or whole.isdigit()) and (fraction == "" or fraction.isdigit())


def build_typed_validation_diagnostic(
    *,
    code: str,
    message: str,
    span: SourceSpan | None,
    source: SourceDocument,
    label_message: str | None = None,
    stage: str = "",
) -> Diagnostic:
    """Canonical typed-entry validation diagnostic.

    The typed entry point passes pure ``SourceSpan`` values anchored to the
    request-scoped source document; they are converted to the existing
    ``DiagnosticSpan`` DTO here, and ``source_text`` stays the full document
    text (display only — structure is never recovered from it).
    """

    return _validation_diagnostic(
        code=code,
        message=message,
        span=None if span is None else _diagnostic_span_from_source_span(span),
        source_text=source.text,
        label_message=label_message,
        stage=stage,
    )


def _validate_label_message(code: str) -> str:
    categories = {
        "RUNTIME_SET_DATASET_NOT_FOUND": "missing dataset",
        "VALIDATE_RESERVED_OUTPUT_TARGET": "reserved name",
        "VALIDATE_COLUMN_NOT_FOUND": "unknown variable",
        "RUNTIME_BY_PRECONDITION_FAILED": "missing BY key",
        "RUNTIME_RENAME_STATEMENT_INVALID": "invalid rename",
        "RUNTIME_DATASET_OPTION_INVALID": "invalid dataset option",
        "VALIDATE_DUPLICATE_DATA_TARGET": "duplicate target",
    }
    return categories.get(code, "validation issue")


class ExecutionValidationService:
    """Authoritative owner of block-level validation before input resolution."""

    def __init__(self, io_service: ExecutorIOService | None = None) -> None:
        self._io_service = io_service or ExecutorIOService(
            arrow_input=DataInputAdapterArrow(),
            arrow_output=DataOutputAdapterArrow(),
            pandas_adapter=DataFrameAdapterPandas(),
            supported_formats=_SUPPORTED_FORMATS,
        )

    def validate_typed(self, request: TypedValidationRequest) -> ValidationResult:
        """Validate one pure typed block through the same diagnostic owner.

        Explicit internal typed entry point.  Target names,
        dataset references, and name lists are read from the ``BlockSyntax``
        structure and passed to the existing schema binding and diagnostic
        owner; no source text is re-scanned or re-parsed.  The rejection
        order is fixed: reserved output targets, the per-DATA-statement
        duplicate-target check (before the registration dedup), then input
        dataset resolution, then the static column references.
        """

        diagnostics: list[Diagnostic] = []
        diagnostics.extend(self._validate_typed_reserved_output_targets(request))
        diagnostics.extend(
            self.validate_typed_data_target_duplicates(request).diagnostics
        )
        diagnostics.extend(self._validate_typed_input_dataset_refs(request))
        if diagnostics:
            return ValidationResult(diagnostics=tuple(diagnostics))
        diagnostics.extend(self._validate_typed_static_column_refs(request))
        return ValidationResult(diagnostics=tuple(diagnostics))

    def validate_typed_data_target_duplicates(
        self,
        request: TypedValidationRequest,
    ) -> ValidationResult:
        """Run the shared duplicate-DATA-target guard without row/schema input.

        Output planning calls this same guard before registering or
        deduplicating typed targets, so explicit output-target overrides
        cannot bypass it.
        """

        if not isinstance(request, TypedValidationRequest):
            raise TypeError("request must be TypedValidationRequest")
        return ValidationResult(
            diagnostics=tuple(self._validate_typed_data_duplicate_targets(request))
        )

    def resolve_typed_output_targets(
        self,
        block: BlockSyntax,
        explicit_output_targets: Sequence[str] = (),
    ) -> tuple[str, ...]:
        """Resolve typed registrations with the existing dataset-name owner."""

        if not isinstance(block, BlockSyntax):
            raise TypeError("block must be a BlockSyntax")
        if explicit_output_targets:
            return tuple(
                self._normalize_dataset_name(target)
                for target in explicit_output_targets
            )

        data_target_names: list[str] = []
        output_target_names: list[str] = []
        for statement in block.statements:
            if statement.kind == "DATA" and isinstance(
                statement.payload, DataStatementSyntax
            ):
                data_target_names.extend(
                    reference.name for reference in statement.payload.outputs
                )
            elif statement.kind == "OUTPUT" and isinstance(
                statement.payload, OutputStatementSyntax
            ):
                output_target_names.extend(
                    reference.name for reference in statement.payload.targets
                )
        return self._merge_unique_targets(data_target_names, output_target_names)

    def _resolved_candidate_inputs(
        self,
        *,
        requested_names: Sequence[str],
        explicit_inputs: Mapping[str, DataSetRef],
        available_inputs: Mapping[str, DataSetRef],
        registered_tables: Mapping[str, DataSetRef],
    ) -> dict[str, DataSetRef]:
        """Candidate catalog view shared by both validation request paths."""

        synthetic_inputs = self._io_service.resolve_synthetic_inputs_for_names(
            requested_names=requested_names,
            explicit_inputs=explicit_inputs,
            available_inputs=available_inputs,
            registered_tables=registered_tables,
            cache=None,
        )
        candidates: dict[str, DataSetRef] = {}
        candidates.update(registered_tables)
        candidates.update(available_inputs)
        candidates.update(explicit_inputs)
        candidates.update(synthetic_inputs)
        return candidates

    # -- explicit typed entry point -------------------------------------------

    # -- explicit typed entry point -------------------------------------------

    def _typed_source_statement(
        self,
        request: TypedValidationRequest,
    ) -> tuple[StatementSyntax, SourceStatementSyntax] | None:
        """First SET/MERGE statement whose typed payload is structured.

        A statement whose payload stayed a ``DeferredStatementError`` is a
        structural fault judged at initial parse; the typed validation entry
        has no structure to read there and adds no new early rejection for it.
        """

        for item in request.block.statements:
            if item.kind in {"SET", "MERGE"} and isinstance(
                item.payload, SourceStatementSyntax
            ):
                return item, item.payload
        return None

    @staticmethod
    def _typed_statement_span(statement: StatementSyntax) -> DiagnosticSpan:
        return _diagnostic_span_from_source_span(statement.span)

    def _validate_typed_reserved_output_targets(
        self,
        request: TypedValidationRequest,
    ) -> list[Diagnostic]:
        targets = self.resolve_typed_output_targets(
            request.block,
            request.explicit_output_targets,
        )
        owner_span = next(
            (
                statement.span
                for statement in request.block.statements
                if statement.kind in {"DATA", "OUTPUT"}
            ),
            None,
        )
        span = (
            None
            if request.explicit_output_targets or owner_span is None
            else _diagnostic_span_from_source_span(owner_span)
        )

        diagnostics: list[Diagnostic] = []
        for target in targets:
            key = self._dataset_name_key(target)
            if key != "DICTIONARY" and not key.startswith("DICTIONARY."):
                continue
            diagnostics.append(
                _validation_diagnostic(
                    code="VALIDATE_RESERVED_OUTPUT_TARGET",
                    message=f"Reserved output target is not allowed: {target}",
                    span=span,
                    source_text=request.source.text,
                )
            )
        return diagnostics

    def _validate_typed_data_duplicate_targets(
        self,
        request: TypedValidationRequest,
    ) -> list[Diagnostic]:
        """Reject duplicate output targets inside one DATA statement.

        The original DATA reference column is inspected before the target
        registration dedup, normalized with the existing dataset key (case
        and ``work.``), and rejected unconditionally — independent of the
        options carried by either occurrence and independent of any explicit
        ``output_targets``.  Duplicate occurrences point at the later
        reference span and name the target.  Inline OUTPUT statements are not
        part of this check, so multiple OUTPUT writes to one declared target
        and same-named targets in different blocks stay allowed.
        """

        diagnostics: list[Diagnostic] = []
        for statement in request.block.statements:
            if statement.kind != "DATA" or not isinstance(
                statement.payload, DataStatementSyntax
            ):
                continue
            seen_keys: set[str] = set()
            for reference in statement.payload.outputs:
                key = self._dataset_name_key(reference.name)
                if key in seen_keys:
                    diagnostics.append(
                        build_typed_validation_diagnostic(
                            code="VALIDATE_DUPLICATE_DATA_TARGET",
                            message=(
                                "DATA statement has duplicate output target: "
                                f"{reference.name}"
                            ),
                            span=reference.span,
                            source=request.source,
                            stage="validate",
                        )
                    )
                    return diagnostics
                seen_keys.add(key)
        return diagnostics

    def _validate_typed_input_dataset_refs(
        self,
        request: TypedValidationRequest,
    ) -> list[Diagnostic]:
        typed_source = self._typed_source_statement(request)
        if typed_source is None:
            return []
        source_statement, source_payload = typed_source

        source_refs = source_payload.refs
        span = self._typed_statement_span(source_statement)
        source_text = request.source.text
        input_names = [ref.name for ref in source_refs]
        if not input_names:
            return [
                _validation_diagnostic(
                    code="RUNTIME_SET_DATASET_NOT_FOUND",
                    message="SET statement requires an input dataset name.",
                    span=span,
                    source_text=source_text,
                )
            ]

        candidates = self._resolved_candidate_inputs(
            requested_names=input_names,
            explicit_inputs=request.explicit_inputs,
            available_inputs=request.available_inputs,
            registered_tables=request.registered_tables,
        )

        diagnostics: list[Diagnostic] = []
        for input_name in input_names:
            if self._resolve_dataset_alias(candidates, input_name) is not None:
                continue
            diagnostics.append(
                _validation_diagnostic(
                    code="RUNTIME_SET_DATASET_NOT_FOUND",
                    message=f"Input dataset is not provided: {input_name}",
                    span=span,
                    source_text=source_text,
                )
            )
        return diagnostics

    def _validate_typed_static_column_refs(
        self,
        request: TypedValidationRequest,
    ) -> list[Diagnostic]:
        typed_source = self._typed_source_statement(request)
        if typed_source is None:
            return []
        source_statement, source_payload = typed_source

        source_refs = source_payload.refs
        source_text = request.source.text
        span = self._typed_statement_span(source_statement)
        candidates = self._resolved_candidate_inputs(
            requested_names=[ref.name for ref in source_refs],
            explicit_inputs=request.explicit_inputs,
            available_inputs=request.available_inputs,
            registered_tables=request.registered_tables,
        )

        source_columns: dict[str, set[str]] = {}
        available_columns: set[str] = set()
        diagnostics: list[Diagnostic] = []

        for source_ref in source_refs:
            input_ref = self._resolve_dataset_alias(candidates, source_ref.name)
            if input_ref is None:
                continue
            columns = self._infer_dataset_columns(input_ref)
            if columns is None:
                continue
            transformed_columns, option_diagnostics = (
                self._apply_source_option_columns_for_validate(
                    columns=columns,
                    source_name=source_ref.name,
                    option_spec=source_ref.options,
                    span=span,
                    source_text=source_text,
                )
            )
            if option_diagnostics:
                diagnostics.extend(option_diagnostics)
                return diagnostics
            source_key = self._dataset_name_key(source_ref.name)
            source_columns[source_key] = transformed_columns
            available_columns.update(transformed_columns)

        statement_options = source_payload.options
        by_keys = self._typed_variable_list(request.block.statements, "BY")
        in_option_vars = [
            ref.options.in_var
            for ref in source_refs
            if ref.options.in_var is not None
        ]
        available_columns.update(
            self._collect_internal_variable_names(
                in_option_vars=in_option_vars,
                indsname_var=statement_options.indsname_var,
                end_var=statement_options.end_var,
                by_keys=by_keys,
            )
        )
        available_columns.update(
            self._collect_typed_step_defined_columns(request.block.statements)
        )

        for statement in request.block.statements:
            if statement.kind == "BY" and isinstance(
                statement.payload, NameListStatementSyntax
            ):
                for by_key in (name.name for name in statement.payload.names):
                    normalized_by_key = _column_key(by_key)
                    missing_in_sources = [
                        source_name
                        for source_name, columns in source_columns.items()
                        if normalized_by_key
                        not in {_column_key(column) for column in columns}
                    ]
                    if missing_in_sources:
                        diagnostics.append(
                            build_typed_validation_diagnostic(
                                code="RUNTIME_BY_PRECONDITION_FAILED",
                                message=(
                                    f"BY key '{by_key}' is missing in source rows."
                                ),
                                span=statement.span,
                                source=request.source,
                                label_message="missing BY key",
                            )
                        )
                        return diagnostics
                continue

            column_diagnostic = self._validate_typed_statement_column_refs(
                statement=statement,
                available_columns=available_columns,
                source=request.source,
            )
            if column_diagnostic is not None:
                diagnostics.append(column_diagnostic)
                return diagnostics

        return diagnostics

    def _validate_typed_statement_column_refs(
        self,
        *,
        statement: StatementSyntax,
        available_columns: set[str],
        source: SourceDocument,
    ) -> Diagnostic | None:
        payload = statement.payload
        if isinstance(payload, DeferredStatementError):
            # Structural fault saved at initial parse: no typed structure to
            # check and no new early rejection.
            return None
        if statement.kind == "RENAME" and isinstance(payload, RenameStatementSyntax):
            rename_map = dict(payload.pairs)
            rename_validation = self._validate_rename_statement_contract(rename_map)
            if rename_validation is not None:
                return build_typed_validation_diagnostic(
                    code=rename_validation.code,
                    message=rename_validation.message,
                    span=statement.span,
                    source=source,
                    label_message="invalid rename",
                )
            referenced_columns = tuple(rename_map)
            diagnostic_code = "RUNTIME_RENAME_STATEMENT_INVALID"
        elif statement.kind in {"KEEP", "DROP"} and isinstance(
            payload, NameListStatementSyntax
        ):
            referenced_columns = tuple(name.name for name in payload.names)
            diagnostic_code = "VALIDATE_COLUMN_NOT_FOUND"
        elif statement.kind == "LABEL" and isinstance(payload, LabelStatementSyntax):
            referenced_columns = tuple(attribute.name for attribute in payload.labels)
            diagnostic_code = "VALIDATE_COLUMN_NOT_FOUND"
        else:
            # Only the static KEEP/DROP/BY/RENAME name references are checked
            # by the current validation contract; general-expression name
            # classification stays with the later structured-expression owner.
            return None

        return self._unknown_column_diagnostic(
            statement_kind=statement.kind,
            referenced_columns=referenced_columns,
            available_columns=available_columns,
            code=diagnostic_code,
            span=self._typed_statement_span(statement),
            source_text=source.text,
        )

    def _typed_variable_list(
        self,
        statements: Sequence[StatementSyntax],
        kind: str,
    ) -> tuple[str, ...]:
        for item in statements:
            if item.kind == kind and isinstance(item.payload, NameListStatementSyntax):
                return tuple(name.name for name in item.payload.names)
        return ()

    def _collect_typed_step_defined_columns(
        self,
        statements: Sequence[StatementSyntax],
    ) -> set[str]:
        """Defined columns read from the typed targets/bindings only.

        Targets and name lists come from the typed payloads; expression
        bodies (including inline IF action conditions and values) are never
        classified here.  Deferred actions keep the existing contract: no
        reachability judgment and no invented defined name.
        """

        defined: set[str] = set()
        for statement in statements:
            payload = statement.payload
            if isinstance(payload, AssignStatementSyntax):
                defined.add(payload.target.name)
            elif isinstance(payload, SumStatementSyntax):
                defined.add(payload.target.name)
            elif isinstance(payload, IfStatementSyntax):
                action = payload.then_action
                if isinstance(action, StatementSyntax):
                    if isinstance(action.payload, AssignStatementSyntax):
                        defined.add(action.payload.target.name)
                    elif isinstance(action.payload, SumStatementSyntax):
                        defined.add(action.payload.target.name)
            elif isinstance(payload, DoStatementSyntax):
                if payload.loop_var:
                    defined.add(payload.loop_var)
            elif isinstance(payload, ArrayStatementSyntax):
                defined.update(name for name in payload.variables if name)
            elif isinstance(payload, NameListStatementSyntax) and (
                statement.kind == "RETAIN"
            ):
                defined.update(self._retained_defined_names(payload.names))
        return defined

    @staticmethod
    def _retained_defined_names(names: Sequence[NameSyntax]) -> tuple[str, ...]:
        """RETAIN name-list entries that define columns.

        Numeric lexemes and quoted tokens are not column names; the filter
        reads only the typed name values and never re-tokenizes source text.
        """

        retained: list[str] = []
        for name in names:
            token = name.name
            if token.startswith(("'", '"')):
                continue
            if _is_numeric_column_token(token):
                continue
            retained.append(token)
        return tuple(retained)

    def _unknown_column_diagnostic(
        self,
        *,
        statement_kind: str,
        referenced_columns: Sequence[str],
        available_columns: set[str],
        code: str,
        span: DiagnosticSpan | None,
        source_text: str | None,
        label_message: str = "unknown variable",
    ) -> Diagnostic | None:
        """Shared missing-variable comparison for both request paths."""

        if not referenced_columns:
            return None

        normalized_available = {_column_key(name) for name in available_columns}
        missing = [
            name
            for name in referenced_columns
            if _column_key(name) not in normalized_available
        ]
        if not missing:
            return None

        return _validation_diagnostic(
            code=code,
            message=(
                f"{statement_kind} statement references unknown variable: {missing[0]}"
            ),
            span=span,
            source_text=source_text,
            label_message=label_message,
        )

    def _infer_dataset_columns(self, input_ref: DataSetRef) -> set[str] | None:
        normalized_kind = input_ref.kind.strip().lower()
        payload = input_ref.payload
        if normalized_kind == "arrow_table" and hasattr(payload, "schema"):
            schema = getattr(payload, "schema", None)
            names = getattr(schema, "names", None)
            if names is not None:
                return {str(name) for name in names}

        rows, load_error = self._io_service.load_input_rows(input_ref)
        if load_error is not None:
            return None
        columns: set[str] = set()
        for row in rows:
            if isinstance(row, Mapping):
                columns.update(str(name) for name in row.keys())
        return columns

    def _apply_source_option_columns_for_validate(
        self,
        *,
        columns: set[str],
        source_name: str,
        option_spec: Any,
        span: DiagnosticSpan | None,
        source_text: str | None,
    ) -> tuple[set[str], list[Diagnostic]]:
        working = set(columns)
        keep_vars = tuple(getattr(option_spec, "keep_vars", ()))
        drop_vars = tuple(getattr(option_spec, "drop_vars", ()))
        rename_map = dict(getattr(option_spec, "rename_map", {}))

        def resolve_column(name: str) -> str | None:
            normalized_name = _column_key(name)
            return next(
                (
                    candidate
                    for candidate in working
                    if _column_key(candidate) == normalized_name
                ),
                None,
            )

        if keep_vars:
            missing = [name for name in keep_vars if resolve_column(name) is None]
            if missing:
                return set(), [
                    _validation_diagnostic(
                        code="VALIDATE_COLUMN_NOT_FOUND",
                        message=(
                            "Dataset option KEEP= references unknown variable "
                            f"'{missing[0]}' for source '{source_name}'."
                        ),
                        span=span,
                        source_text=source_text,
                        label_message="unknown variable",
                    )
                ]
            working = {
                resolved
                for name in keep_vars
                if (resolved := resolve_column(name)) is not None
            }

        if drop_vars:
            missing = [name for name in drop_vars if resolve_column(name) is None]
            if missing:
                return set(), [
                    _validation_diagnostic(
                        code="VALIDATE_COLUMN_NOT_FOUND",
                        message=(
                            "Dataset option DROP= references unknown variable "
                            f"'{missing[0]}' for source '{source_name}'."
                        ),
                        span=span,
                        source_text=source_text,
                        label_message="unknown variable",
                    )
                ]
            drop_columns = {
                resolved
                for name in drop_vars
                if (resolved := resolve_column(name)) is not None
            }
            working.difference_update(drop_columns)

        if rename_map:
            if len({_column_key(value) for value in rename_map.values()}) != len(
                rename_map
            ):
                return set(), [
                    _validation_diagnostic(
                        code="RUNTIME_DATASET_OPTION_INVALID",
                        message=(
                            "Dataset option RENAME= has duplicate target names "
                            f"for source '{source_name}'."
                        ),
                        span=span,
                        source_text=source_text,
                        label_message="invalid rename",
                    )
                ]
            missing = [name for name in rename_map if resolve_column(name) is None]
            if missing:
                return set(), [
                    _validation_diagnostic(
                        code="RUNTIME_DATASET_OPTION_INVALID",
                        message=(
                            "Dataset option RENAME= references unknown variable "
                            f"'{missing[0]}' for source '{source_name}'."
                        ),
                        span=span,
                        source_text=source_text,
                        label_message="unknown variable",
                    )
                ]
            resolved_rename_map = {
                resolved: new_name
                for old_name, new_name in rename_map.items()
                if (resolved := resolve_column(old_name)) is not None
            }
            working = {resolved_rename_map.get(name, name) for name in working}

        return working, []

    @staticmethod
    def _validate_rename_statement_contract(
        rename_map: Mapping[str, str],
    ) -> Diagnostic | None:
        if len(set(rename_map.values())) != len(rename_map):
            return Diagnostic(
                code="RUNTIME_RENAME_STATEMENT_INVALID",
                severity="error",
                message="RENAME statement has duplicate target variable names.",
            )

        for source, target in rename_map.items():
            if target in rename_map and rename_map.get(target) == source:
                return Diagnostic(
                    code="RUNTIME_RENAME_STATEMENT_INVALID",
                    severity="error",
                    message="RENAME statement contains circular reference.",
                )
        return None

    def _collect_internal_variable_names(
        self,
        *,
        in_option_vars: Sequence[str],
        indsname_var: str | None,
        end_var: str | None,
        by_keys: Sequence[str],
    ) -> set[str]:
        names = {name for name in in_option_vars if name}
        if indsname_var:
            names.add(indsname_var)
        if end_var:
            names.add(end_var)
        for key in by_keys:
            names.update({f"FIRST.{key}", f"LAST.{key}", f"first.{key}", f"last.{key}"})
        return names

    def _merge_unique_targets(self, *target_groups: Sequence[str]) -> tuple[str, ...]:
        merged: list[str] = []
        seen: set[str] = set()
        for group in target_groups:
            for target in group:
                normalized_target = self._normalize_dataset_name(target)
                target_key = self._dataset_name_key(normalized_target)
                if not normalized_target or target_key in seen:
                    continue
                merged.append(normalized_target)
                seen.add(target_key)
        return tuple(merged)

    def _normalize_dataset_name(self, name: str) -> str:
        return self._io_service.normalize_dataset_name(name)

    def _dataset_name_key(self, name: str) -> str:
        return self._io_service.dataset_name_key(name)

    def _resolve_dataset_alias(
        self,
        datasets: Mapping[str, DataSetRef],
        requested_name: str,
    ) -> DataSetRef | None:
        return self._io_service.resolve_dataset_alias(datasets, requested_name)


__all__ = [
    "ExecutionValidationService",
    "build_typed_validation_diagnostic",
    "has_output_projection_collision",
]
