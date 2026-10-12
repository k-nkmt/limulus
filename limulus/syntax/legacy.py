"""Flat parser-result models and serializers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models import Diagnostic, DiagnosticSpan
from .models import (
    WhereComparison,
    WhereExpression,
    WhereLogicalExpression,
    WhereNotExpression,
    WhereNullPredicate,
)


@dataclass(frozen=True)
class DatasetReferenceOptionSpec:
    in_var: str | None = None
    keep_vars: tuple[str, ...] = field(default_factory=tuple)
    drop_vars: tuple[str, ...] = field(default_factory=tuple)
    where_expr: WhereExpression | None = None
    rename_map: dict[str, str] = field(default_factory=dict)
    firstobs: int | None = None
    obs: int | None = None
    label: str | None = None


@dataclass(frozen=True)
class SetStatementOptionSpec:
    indsname_var: str | None = None
    end_var: str | None = None


@dataclass(frozen=True)
class DatasetReference:
    name: str
    options: DatasetReferenceOptionSpec = field(default_factory=DatasetReferenceOptionSpec)


@dataclass(frozen=True)
class ParsedStatement:
    kind: str
    text: str
    span: DiagnosticSpan | None = None
    dataset_refs: tuple[DatasetReference, ...] = field(default_factory=tuple)
    output_refs: tuple[DatasetReference, ...] = field(default_factory=tuple)
    statement_options: SetStatementOptionSpec = field(default_factory=SetStatementOptionSpec)
    rename_map: dict[str, str] = field(default_factory=dict)
    label_map: dict[str, str] = field(default_factory=dict)
    if_spec: "IfStatementSpec | None" = None
    do_spec: "DoStatementSpec | None" = None
    array_spec: "ArrayStatementSpec | None" = None


@dataclass(frozen=True)
class IfStatementSpec:
    condition: str
    then_action: str | None = None
    is_subset: bool = False
    is_then_do: bool = False


@dataclass(frozen=True)
class DoStatementSpec:
    loop_var: str
    start_expr: str
    end_expr: str


@dataclass(frozen=True)
class ArrayStatementSpec:
    array_name: str
    variables: tuple[str, ...]
    declared_size: int | None = None
    wildcard_size: bool = False
    character_array: bool = False


@dataclass(frozen=True)
class DataStepAst:
    statements: tuple[ParsedStatement, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class StatementRegion:
    kind: str
    start: int
    end: int


@dataclass(frozen=True)
class ParseResult:
    ast: DataStepAst = field(default_factory=DataStepAst)
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    @property
    def has_errors(self) -> bool:
        return any(diag.severity == "error" for diag in self.diagnostics)


def where_expression_to_dict(expression: WhereExpression | None) -> dict[str, object] | None:
    """Serialize a typed WHERE AST using the stable plan/golden shape."""

    if expression is None:
        return None
    if isinstance(expression, WhereComparison):
        return {
            "kind": "comparison",
            "column": expression.column,
            "operator": expression.operator,
            "value": expression.value,
        }
    if isinstance(expression, WhereNullPredicate):
        return {
            "kind": "null",
            "column": expression.column,
            "negated": expression.negated,
        }
    if isinstance(expression, WhereLogicalExpression):
        return {
            "kind": "logical",
            "operator": expression.operator,
            "operands": [where_expression_to_dict(operand) for operand in expression.operands],
        }
    if isinstance(expression, WhereNotExpression):
        return {
            "kind": "not",
            "operand": where_expression_to_dict(expression.operand),
        }
    raise TypeError(f"unsupported dataset WHERE expression: {type(expression).__name__}")


def statement_to_dict(statement: ParsedStatement) -> dict[str, Any]:
    span = statement.span
    if_spec = statement.if_spec
    do_spec = statement.do_spec
    array_spec = statement.array_spec
    return {
        "kind": statement.kind,
        "text": statement.text,
        "span": (
            {
                "start": span.start,
                "end": span.end,
                "line": span.line,
                "column": span.column,
                "end_line": span.end_line,
                "end_column": span.end_column,
                "source_id": span.source_id,
            }
            if span is not None
            else None
        ),
        "dataset_refs": [
            {
                "name": dataset_ref.name,
                "options": {
                    "in_var": dataset_ref.options.in_var,
                    "keep_vars": list(dataset_ref.options.keep_vars),
                    "drop_vars": list(dataset_ref.options.drop_vars),
                    "where_expr": where_expression_to_dict(dataset_ref.options.where_expr),
                    "rename_map": dict(dataset_ref.options.rename_map),
                    "firstobs": dataset_ref.options.firstobs,
                    "obs": dataset_ref.options.obs,
                    "label": dataset_ref.options.label,
                },
            }
            for dataset_ref in statement.dataset_refs
        ],
        "output_refs": [
            {
                "name": dataset_ref.name,
                "options": {
                    "in_var": dataset_ref.options.in_var,
                    "keep_vars": list(dataset_ref.options.keep_vars),
                    "drop_vars": list(dataset_ref.options.drop_vars),
                    "where_expr": where_expression_to_dict(dataset_ref.options.where_expr),
                    "rename_map": dict(dataset_ref.options.rename_map),
                    "firstobs": dataset_ref.options.firstobs,
                    "obs": dataset_ref.options.obs,
                    "label": dataset_ref.options.label,
                },
            }
            for dataset_ref in statement.output_refs
        ],
        "statement_options": {
            "indsname_var": statement.statement_options.indsname_var,
            "end_var": statement.statement_options.end_var,
        },
        "rename_map": dict(statement.rename_map),
        "label_map": dict(statement.label_map),
        "if_spec": (
            {
                "condition": if_spec.condition,
                "then_action": if_spec.then_action,
                "is_subset": if_spec.is_subset,
                "is_then_do": if_spec.is_then_do,
            }
            if if_spec is not None
            else None
        ),
        "do_spec": (
            {
                "loop_var": do_spec.loop_var,
                "start_expr": do_spec.start_expr,
                "end_expr": do_spec.end_expr,
            }
            if do_spec is not None
            else None
        ),
        "array_spec": (
            {
                "array_name": array_spec.array_name,
                "variables": list(array_spec.variables),
                "declared_size": array_spec.declared_size,
                "wildcard_size": array_spec.wildcard_size,
                "character_array": array_spec.character_array,
            }
            if array_spec is not None
            else None
        ),
    }


def ast_to_dict(ast: DataStepAst) -> dict[str, Any]:
    return {"statements": [statement_to_dict(statement) for statement in ast.statements]}


def ast_statements_to_dict(statements: list[ParsedStatement] | tuple[ParsedStatement, ...]) -> dict[str, Any]:
    return {"statements": [statement_to_dict(statement) for statement in statements]}


__all__ = [
    "ArrayStatementSpec",
    "DataStepAst",
    "DatasetReference",
    "DatasetReferenceOptionSpec",
    "DoStatementSpec",
    "IfStatementSpec",
    "ParseResult",
    "ParsedStatement",
    "SetStatementOptionSpec",
    "StatementRegion",
    "ast_statements_to_dict",
    "ast_to_dict",
    "statement_to_dict",
    "where_expression_to_dict",
]
