"""Pure typed output target and shared option policy planning."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ..models import Diagnostic
from ..naming import _column_key, _dataset_key
from ..syntax.models import (
    BlockSyntax,
    DataStatementSyntax,
    DatasetOptionSyntax,
    DatasetReferenceSyntax,
    DeferredStatementError,
    ElseStatementSyntax,
    IfStatementSyntax,
    OutputStatementSyntax,
    SourceDocument,
    StatementSyntax,
    WhereComparison,
    WhereLogicalExpression,
    WhereNotExpression,
    WhereNullPredicate,
)
from .validation import build_typed_validation_diagnostic


@dataclass(frozen=True)
class TypedOutputTargetPlan:
    """Structural target policy consumed by later output shaping and wire units."""

    targets: tuple[str, ...]
    default_target: str | None
    suppress_implicit_output: bool
    effective_output_options: Mapping[str, DatasetOptionSyntax]
    diagnostics: tuple[Diagnostic, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(
            self,
            "effective_output_options",
            MappingProxyType(dict(self.effective_output_options)),
        )
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))


def plan_typed_output_targets(
    *,
    block: BlockSyntax,
    source: SourceDocument,
    resolved_output_targets: Sequence[str],
    normalize_target: Callable[[str], str],
) -> TypedOutputTargetPlan:
    """Resolve typed registrations, default routing, suppression, and options.

    DATA and top-level OUTPUT references define registrations unless explicit
    output targets were supplied.  Inline OUTPUT references declare options
    for existing targets but never register a new target.  No display text is
    consulted.
    """

    data_references = _first_data_references(block)
    top_level_references: list[DatasetReferenceSyntax] = []
    inline_references: list[DatasetReferenceSyntax] = []
    suppress_implicit_output = False

    for statement in block.statements:
        if statement.kind == "OUTPUT" and isinstance(
            statement.payload, OutputStatementSyntax
        ):
            suppress_implicit_output = True
            top_level_references.extend(statement.payload.targets)
        elif isinstance(statement.payload, IfStatementSyntax):
            found, has_output = _inline_output_references(
                statement.payload.then_action
            )
            inline_references.extend(found)
            suppress_implicit_output = suppress_implicit_output or has_output
        elif isinstance(statement.payload, ElseStatementSyntax):
            found, has_output = _inline_output_references(statement.payload.action)
            inline_references.extend(found)
            suppress_implicit_output = suppress_implicit_output or has_output

    targets = tuple(normalize_target(target) for target in resolved_output_targets)

    default_target = (
        normalize_target(top_level_references[0].name)
        if top_level_references
        else (targets[0] if targets else None)
    )

    registered_keys = {_dataset_key(target) for target in targets}
    options_by_target: dict[str, DatasetOptionSyntax] = {}
    for target in targets:
        options_by_target.setdefault(_dataset_key(target), DatasetOptionSyntax())

    diagnostics: list[Diagnostic] = []
    first_nonempty: dict[str, tuple[tuple[object, ...], DatasetReferenceSyntax]] = {}
    references = sorted(
        (*data_references, *top_level_references, *inline_references),
        key=lambda reference: reference.span.start,
    )
    for reference in references:
        key = _dataset_key(reference.name)
        if key not in registered_keys:
            # An unregistered inline/OUTPUT reference stays a runtime routing
            # error and cannot introduce an option-map entry.
            continue
        options = reference.options
        if options.in_var is not None:
            diagnostics.append(
                build_typed_validation_diagnostic(
                    code="RUNTIME_DATASET_OPTION_INVALID",
                    message=(
                        f"Input-only IN option is not allowed for output target: "
                        f"{reference.name}"
                    ),
                    span=reference.span,
                    source=source,
                    stage="validate",
                )
            )
            continue
        if not _has_nonempty_option(options):
            continue
        normalized = _normalized_options(options)
        previous = first_nonempty.get(key)
        if previous is None:
            first_nonempty[key] = (normalized, reference)
            options_by_target[key] = options
        elif previous[0] != normalized:
            diagnostics.append(
                build_typed_validation_diagnostic(
                    code="RUNTIME_DATASET_OPTION_INVALID",
                    message=(
                        f"Conflicting output options for target: {reference.name}"
                    ),
                    span=reference.span,
                    source=source,
                    stage="validate",
                )
            )

    return TypedOutputTargetPlan(
        targets=targets,
        default_target=default_target,
        suppress_implicit_output=suppress_implicit_output,
        effective_output_options=options_by_target,
        diagnostics=tuple(diagnostics),
    )


def _first_data_references(block: BlockSyntax) -> tuple[DatasetReferenceSyntax, ...]:
    for statement in block.statements:
        if statement.kind == "DATA" and isinstance(
            statement.payload, DataStatementSyntax
        ):
            return statement.payload.outputs
    return ()


def _inline_output_references(
    action: StatementSyntax | DeferredStatementError | None,
) -> tuple[tuple[DatasetReferenceSyntax, ...], bool]:
    if not isinstance(action, StatementSyntax):
        return (), False
    if action.kind == "OUTPUT" and isinstance(action.payload, OutputStatementSyntax):
        return action.payload.targets, True
    if isinstance(action.payload, IfStatementSyntax):
        return _inline_output_references(action.payload.then_action)
    if isinstance(action.payload, ElseStatementSyntax):
        return _inline_output_references(action.payload.action)
    return (), False


def _has_nonempty_option(options: DatasetOptionSyntax) -> bool:
    return bool(
        options.keep_vars
        or options.drop_vars
        or options.rename_map
        or options.where_expr is not None
        or options.firstobs is not None
        or options.obs is not None
        or options.label is not None
    )


def _normalized_options(options: DatasetOptionSyntax) -> tuple[object, ...]:
    return (
        tuple(_column_key(name) for name in options.keep_vars),
        tuple(_column_key(name) for name in options.drop_vars),
        tuple(
            (_column_key(source), _column_key(target))
            for source, target in options.rename_map
        ),
        _normalized_where(options.where_expr),
        options.firstobs,
        options.obs,
        options.label,
    )


def _normalized_where(expression: object | None) -> object:
    if expression is None:
        return None
    if isinstance(expression, WhereComparison):
        return (
            "comparison",
            _column_key(expression.column),
            expression.operator,
            _typed_scalar(expression.value),
        )
    if isinstance(expression, WhereNullPredicate):
        return ("null", _column_key(expression.column), expression.negated)
    if isinstance(expression, WhereLogicalExpression):
        return (
            "logical",
            expression.operator,
            tuple(_normalized_where(item) for item in expression.operands),
        )
    if isinstance(expression, WhereNotExpression):
        return ("not", _normalized_where(expression.operand))
    raise TypeError(f"unsupported typed WHERE expression: {type(expression).__name__}")


def _typed_scalar(value: object) -> tuple[str, object]:
    # bool is an int subclass, so the explicit type tag avoids equating True
    # and 1 when comparing source-preserved WHERE literal values.
    return (type(value).__name__, value)


__all__ = ["TypedOutputTargetPlan", "plan_typed_output_targets"]
