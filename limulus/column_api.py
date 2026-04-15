from __future__ import annotations

from dataclasses import dataclass
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from lark import Lark, Tree
import pyarrow as pa

from ._naming import _column_key
from .evaluator import ExpressionEvaluator
from .runtime import PDVRuntimeService


_FUNCTION_CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_CASE_WHEN_GRAMMAR = r"""
?start: case_expr

case_expr: CASE when_clause+ else_clause? END
when_clause: WHEN expression THEN expression
else_clause: ELSE expression

expression: expression_atom+

?expression_atom: STRING
                | NAME
                | NUMBER
                | OP
                | COMMA
                | DOT
                | COLON
                | LPAR expression? RPAR
                | LBRACK expression? RBRACK
                | LBRACE expression? RBRACE

CASE.5: /case/i
WHEN.5: /when/i
THEN.5: /then/i
ELSE.5: /else/i
END.5: /end/i
NAME.1: /[A-Za-z_][\w\.]*/
NUMBER: /(?:\d+\.\d*|\d+|\.\d+)(?:[eE][+-]?\d+)?/
STRING: /'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"/
OP: />=|<=|!=|==|\^=|~=|¬=|\|\||\*\*|[-+*\/%^<>=]/
COMMA: ","
DOT: "."
COLON: ":"
LPAR: "("
RPAR: ")"
LBRACK: "["
RBRACK: "]"
LBRACE: "{"
RBRACE: "}"

%import common.WS_INLINE
%ignore WS_INLINE
"""


@dataclass(frozen=True)
class _CaseWhenClause:
    condition: str
    value_expression: str


@dataclass(frozen=True)
class _AssignmentSpec:
    target: str
    kind: str
    value: Any = None
    expression: str | None = None
    clauses: tuple[_CaseWhenClause, ...] = ()
    else_expression: str | None = None


class _CaseWhenParser:
    def __init__(self) -> None:
        self._parser = Lark(_CASE_WHEN_GRAMMAR, parser="lalr", propagate_positions=True)

    def parse(self, expression: str) -> tuple[tuple[_CaseWhenClause, ...], str | None] | None:
        stripped = expression.strip()
        if not stripped.lower().startswith("case"):
            return None

        try:
            parsed = self._parser.parse(stripped)
        except Exception as error:
            raise ValueError(f"Invalid case expression: {expression}") from error

        root = parsed if parsed.data == "case_expr" else parsed.children[0]
        clauses: list[_CaseWhenClause] = []
        else_expression: str | None = None

        for child in root.children:
            if not isinstance(child, Tree):
                continue

            if child.data == "when_clause":
                expression_nodes = [node for node in child.children if isinstance(node, Tree) and node.data == "expression"]
                if len(expression_nodes) != 2:
                    raise ValueError(f"Invalid case expression: {expression}")
                clauses.append(
                    _CaseWhenClause(
                        condition=self._slice_text(stripped, expression_nodes[0]),
                        value_expression=self._slice_text(stripped, expression_nodes[1]),
                    )
                )
                continue

            if child.data == "else_clause":
                expression_nodes = [node for node in child.children if isinstance(node, Tree) and node.data == "expression"]
                if len(expression_nodes) != 1:
                    raise ValueError(f"Invalid case expression: {expression}")
                else_expression = self._slice_text(stripped, expression_nodes[0])

        if not clauses:
            raise ValueError(f"Invalid case expression: {expression}")
        return tuple(clauses), else_expression

    @staticmethod
    def _slice_text(source: str, node: Tree) -> str:
        return source[node.meta.start_pos:node.meta.end_pos].strip()


_CASE_WHEN_PARSER = _CaseWhenParser()


def normalize_column_sequence(columns: str | Sequence[str] | None) -> tuple[str, ...]:
    if columns is None:
        return ()
    if isinstance(columns, str):
        return (columns,)
    return tuple(str(column) for column in columns)


def resolve_column_name(table: pa.Table, column: str, *, parameter_name: str) -> str:
    if column in table.column_names:
        return column

    column_key = _column_key(column)
    matches = [name for name in table.column_names if _column_key(name) == column_key]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise KeyError(f"Ambiguous column reference for Session.{parameter_name}: {column}")
    raise KeyError(f"Column not found for Session.{parameter_name}: {column}")


def resolve_existing_columns(
    table: pa.Table,
    columns: str | Sequence[str] | None,
    *,
    parameter_name: str,
) -> tuple[str, ...]:
    normalized_columns = normalize_column_sequence(columns)
    return tuple(resolve_column_name(table, column, parameter_name=parameter_name) for column in normalized_columns)


def resolve_transpose_var_columns(
    table: pa.Table,
    *,
    by_columns: Sequence[str],
    id_columns: Sequence[str],
    var: str | Sequence[str] | None,
) -> tuple[str, ...]:
    if var is not None:
        resolved = resolve_existing_columns(table, var, parameter_name="var")
        if resolved:
            return resolved

    excluded = {_column_key(name) for name in (*by_columns, *id_columns)}
    resolved = tuple(name for name in table.column_names if _column_key(name) not in excluded)
    if not resolved:
        raise ValueError("Session.transpose requires at least one VAR column")
    return resolved


def transpose_table(
    table: pa.Table,
    *,
    by_columns: tuple[str, ...],
    id_columns: tuple[str, ...],
    var_columns: tuple[str, ...],
    materialize_table: Callable[[Sequence[Mapping[str, Any]], pa.Table], pa.Table],
) -> pa.Table:
    use_id_layout = bool(id_columns)
    if use_id_layout and len(id_columns) != 1:
        raise ValueError("Session.transpose currently supports exactly one ID column")
    if use_id_layout and len(var_columns) != 1:
        raise ValueError("Session.transpose with id currently supports exactly one VAR column")

    source_rows = table.to_pylist()
    ordered_output_columns: list[str] = []
    output_rows: list[dict[str, Any]] = []

    for group_key, group_rows in _group_rows(source_rows, by_columns=by_columns):
        base_row = {column: group_key[index] for index, column in enumerate(by_columns)}

        if not use_id_layout:
            for variable_name in var_columns:
                transposed_row = dict(base_row)
                transposed_row["_NAME_"] = variable_name
                for row_index, row in enumerate(group_rows, start=1):
                    transposed_row[f"COL{row_index}"] = row.get(variable_name)
                output_rows.append(transposed_row)
            continue

        id_column = id_columns[0]
        value_column = var_columns[0]
        transposed_row = dict(base_row)
        seen_ids: set[str] = set()

        for row in group_rows:
            raw_id_value = row.get(id_column)
            if raw_id_value is None or raw_id_value == "":
                raise ValueError(f"Missing ID value for Session.transpose: {id_column}")

            output_column = str(raw_id_value)
            if output_column in seen_ids:
                raise ValueError(f"Duplicate ID value for Session.transpose: {output_column}")

            seen_ids.add(output_column)
            if output_column not in ordered_output_columns:
                ordered_output_columns.append(output_column)
            transposed_row[output_column] = row.get(value_column)

        for output_column in ordered_output_columns:
            transposed_row.setdefault(output_column, None)
        output_rows.append(transposed_row)

    if use_id_layout:
        for output_row in output_rows:
            for output_column in ordered_output_columns:
                output_row.setdefault(output_column, None)

    return materialize_table(output_rows, table)


def assign_columns(
    table: pa.Table,
    assignments: Mapping[str, Any],
    *,
    materialize_table: Callable[[Sequence[Mapping[str, Any]], pa.Table], pa.Table],
) -> pa.Table:
    specs = _normalize_assignments(assignments)
    if not specs:
        return table

    runtime = PDVRuntimeService()
    context = runtime.create_context()
    evaluator = ExpressionEvaluator(runtime.get_eval_scope)
    source_rows = table.to_pylist()
    output_rows: list[dict[str, Any]] = []

    for row_index, source_row in enumerate(source_rows):
        runtime.set_row_view(row_index=row_index, rows=source_rows)
        runtime.begin_row(context)
        working_row = dict(source_row)
        for spec in specs:
            working_row[spec.target] = _evaluate_assignment_spec(
                spec,
                row=working_row,
                evaluator=evaluator,
                runtime=runtime,
                context=context,
            )
        output_rows.append(working_row)

    return materialize_table(output_rows, table)


def _evaluate_assignment_spec(
    spec: _AssignmentSpec,
    *,
    row: Mapping[str, Any],
    evaluator: ExpressionEvaluator,
    runtime: PDVRuntimeService,
    context: Any,
) -> Any:
    if spec.kind == "literal":
        return spec.value

    if spec.kind in {"expr", "func"}:
        return _evaluate_expression_value(
            spec.expression or "",
            row=row,
            evaluator=evaluator,
            runtime=runtime,
            context=context,
        )

    if spec.kind == "case_when":
        for clause in spec.clauses:
            matched = _evaluate_expression_value(
                clause.condition,
                row=row,
                evaluator=evaluator,
                runtime=runtime,
                context=context,
            )
            if bool(matched):
                return _evaluate_expression_value(
                    clause.value_expression,
                    row=row,
                    evaluator=evaluator,
                    runtime=runtime,
                    context=context,
                )

        if spec.else_expression is None:
            return None

        return _evaluate_expression_value(
            spec.else_expression,
            row=row,
            evaluator=evaluator,
            runtime=runtime,
            context=context,
        )

    raise ValueError(f"Unsupported assignment kind: {spec.kind}")


def _evaluate_expression_value(
    expression: str,
    *,
    row: Mapping[str, Any],
    evaluator: ExpressionEvaluator,
    runtime: PDVRuntimeService,
    context: Any,
) -> Any:
    unsupported_function = _find_unsupported_function(
        expression,
        supported_functions=runtime.get_eval_scope().keys(),
    )
    if unsupported_function is not None:
        raise ValueError(f"Unsupported function: {unsupported_function}")

    value, diagnostic = evaluator.evaluate_scalar(expression, row=row, context=context, array_defs={})
    if diagnostic is not None:
        raise ValueError(diagnostic.message)
    return value


def _normalize_assignments(assignments: Mapping[str, Any]) -> tuple[_AssignmentSpec, ...]:
    normalized: list[_AssignmentSpec] = []
    for target, value in assignments.items():
        if not target or not target.strip():
            raise ValueError("Assignment target must be a non-empty column name")

        if isinstance(value, str):
            case_when = _CASE_WHEN_PARSER.parse(value)
            if case_when is not None:
                normalized.append(
                    _AssignmentSpec(
                        target=target,
                        kind="case_when",
                        clauses=case_when[0],
                        else_expression=case_when[1],
                    )
                )
                continue

            kind = "func" if _contains_function_call(value) else "expr"
            normalized.append(_AssignmentSpec(target=target, kind=kind, expression=value))
            continue

        normalized.append(_AssignmentSpec(target=target, kind="literal", value=value))

    return tuple(normalized)


def _contains_function_call(expression: str) -> bool:
    return bool(_FUNCTION_CALL.search(expression))


def _find_unsupported_function(
    expression: str,
    *,
    supported_functions: Iterable[str],
) -> str | None:
    supported = {name.lower() for name in supported_functions}
    for function_name in _FUNCTION_CALL.findall(expression):
        lowered = function_name.lower()
        if lowered in {"and", "or", "not", "in"}:
            continue
        if lowered not in supported:
            return function_name
    return None


def _group_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    by_columns: Sequence[str],
) -> list[tuple[tuple[Any, ...], list[dict[str, Any]]]]:
    grouped: list[tuple[tuple[Any, ...], list[dict[str, Any]]]] = []
    grouped_index: dict[tuple[Any, ...], int] = {}

    for row in rows:
        group_key = tuple(row.get(column) for column in by_columns)
        group_position = grouped_index.get(group_key)
        if group_position is None:
            grouped_index[group_key] = len(grouped)
            grouped.append((group_key, [dict(row)]))
            continue
        grouped[group_position][1].append(dict(row))

    if grouped:
        return grouped
    return [(tuple(), [])]