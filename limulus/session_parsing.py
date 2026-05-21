from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

try:
    from lark import Token, Tree
    from lark.exceptions import UnexpectedInput
except Exception:  # pragma: no cover
    Token = Any  # type: ignore[assignment]
    Tree = Any  # type: ignore[assignment]
    UnexpectedInput = Exception  # type: ignore[assignment]

from .lark_support import build_lark_parser_from_file
from .models import Diagnostic, DiagnosticLabel, DiagnosticSpan, RenderRequest
from .renderer import render_diagnostics


_SIMPLE_FILTER_PARSER = build_lark_parser_from_file("session_filter.lark")
_CREATE_TABLE_SQL_PARSER = build_lark_parser_from_file("session_sql_create.lark")
_DROP_TABLE_SQL_PARSER = build_lark_parser_from_file("session_sql_drop.lark")


@dataclass(frozen=True)
class SimpleFilterSpec:
    variable_name: str
    operator: str
    scalar_value: Any
    source_text: str


@dataclass(frozen=True)
class SqlClassification:
    kind: str
    target: str | None
    query: str


def parse_simple_filter(source: str, expression: str) -> SimpleFilterSpec:
    source_text = expression.strip()
    if _SIMPLE_FILTER_PARSER is None:
        raise RuntimeError("Session.filter parser is unavailable")

    try:
        parsed = _SIMPLE_FILTER_PARSER.parse(source_text)
    except UnexpectedInput as error:
        _raise_session_parse_error(
            code="SESSION_FILTER_PARSE_ERROR",
            message="Unsupported filter expression for Session.filter.",
            source_id=f"filter:{source}",
            source_text=source_text,
            span=_span_from_unexpected_input(source_text, error, source_id=f"filter:{source}"),
            label_message="simple comparison expression",
            notes=("Expected format: <column> <operator> <value> with operators >= <= != = > <.",),
        )

    root = _root_tree(parsed, "filter_expr")
    variable_name = _require_token(root.children[0], "NAME").value
    operator = _require_token(root.children[1], "FILTER_OP").value
    scalar_node = _require_tree(root.children[2])

    if scalar_node.data == "string_scalar":
        raw_value = _require_token(scalar_node.children[0], "STRING").value
        scalar_value = _unquote_string_literal(raw_value)
    elif scalar_node.data == "number_scalar":
        raw_value = _require_token(scalar_node.children[0], "NUMBER").value
        scalar_value = float(raw_value) if any(marker in raw_value for marker in (".", "e", "E")) else int(raw_value)
    else:
        invalid_token = _require_token(scalar_node.children[0], "INVALID_VALUE")
        _raise_session_parse_error(
            code="SESSION_FILTER_PARSE_ERROR",
            message="Unsupported filter literal for Session.filter.",
            source_id=f"filter:{source}",
            source_text=source_text,
            span=_span_from_token(source_text, invalid_token, source_id=f"filter:{source}"),
            label_message="filter literal",
            notes=("Use a quoted string literal or a numeric literal.",),
        )

    return SimpleFilterSpec(
        variable_name=variable_name,
        operator=operator,
        scalar_value=scalar_value,
        source_text=source_text,
    )


def classify_sql(query: str) -> SqlClassification:
    source_text = query.strip().rstrip(";").strip()

    if source_text.lower().startswith("create table"):
        return _classify_create_table_sql(query, source_text)
    if source_text.lower().startswith("drop table"):
        return _classify_drop_table_sql(query, source_text)

    return SqlClassification(kind="select", target=None, query=_rewrite_dictionary_table_references(source_text))


def extract_sql_target(query: str) -> tuple[str | None, str]:
    classification = classify_sql(query)
    if classification.kind != "create_table":
        return None, classification.query
    return classification.target, classification.query


def _rewrite_dictionary_table_references(query: str) -> str:
    rewritten: list[str] = []
    index = 0
    while index < len(query):
        character = query[index]
        if character in {'"', "'"}:
            end = _consume_quoted_segment(query, index)
            rewritten.append(query[index:end])
            index = end
            continue

        dictionary_reference = _match_dictionary_reference(query, index)
        if dictionary_reference is not None:
            replacement, end = dictionary_reference
            rewritten.append(replacement)
            index = end
            continue

        rewritten.append(character)
        index += 1

    return "".join(rewritten)


def _classify_create_table_sql(query: str, source_text: str) -> SqlClassification:
    if _CREATE_TABLE_SQL_PARSER is None:
        raise RuntimeError("CREATE TABLE parser is unavailable")

    try:
        parsed = _CREATE_TABLE_SQL_PARSER.parse(source_text)
    except UnexpectedInput as error:
        _raise_sql_classification_error(
            query,
            message="Unsupported CREATE TABLE form for Session.sql.",
            label_message="CREATE TABLE classification",
            notes=("Supported classified form: CREATE TABLE name AS <query>.",),
            span=_span_from_unexpected_input(query, error, source_id="sql"),
        )

    root = _root_tree(parsed, "create_table_stmt")
    target = _slice_tree_text(source_text, _require_tree(root.children[2]))
    executable_query = _require_token(root.children[4], "QUERY_TEXT").value.strip()
    return SqlClassification(
        kind="create_table",
        target=target,
        query=_rewrite_dictionary_table_references(executable_query),
    )


def _classify_drop_table_sql(query: str, source_text: str) -> SqlClassification:
    if _DROP_TABLE_SQL_PARSER is None:
        raise RuntimeError("DROP TABLE parser is unavailable")

    try:
        parsed = _DROP_TABLE_SQL_PARSER.parse(source_text)
    except UnexpectedInput as error:
        _raise_sql_classification_error(
            query,
            message="Unsupported DROP TABLE form for Session.sql.",
            label_message="DROP TABLE classification",
            notes=("Supported classified form: DROP TABLE name.",),
            span=_span_from_unexpected_input(query, error, source_id="sql"),
        )

    root = _root_tree(parsed, "drop_table_stmt")
    target = _slice_tree_text(source_text, _require_tree(root.children[2]))
    return SqlClassification(kind="drop_table", target=target, query=source_text)


def _raise_sql_classification_error(
    query: str,
    *,
    message: str,
    label_message: str,
    notes: Sequence[str],
    span: DiagnosticSpan | None,
) -> None:
    _raise_session_parse_error(
        code="SESSION_SQL_CLASSIFICATION_ERROR",
        message=message,
        source_id="sql",
        source_text=query,
        label_message=label_message,
        notes=notes,
        span=span,
    )


def _root_tree(parsed: Any, expected: str) -> Any:
    if parsed.data == expected:
        return parsed
    if parsed.children and isinstance(parsed.children[0], Tree) and parsed.children[0].data == expected:
        return parsed.children[0]
    raise ValueError(f"Unexpected parse tree root: {parsed.data}")


def _require_tree(node: Any) -> Any:
    if not isinstance(node, Tree):
        raise ValueError(f"Expected tree node, got: {type(node)!r}")
    return node


def _require_token(node: Any, token_type: str) -> Any:
    if not isinstance(node, Token) or node.type != token_type:
        raise ValueError(f"Expected token {token_type}, got: {node!r}")
    return node


def _slice_tree_text(source_text: str, node: Any) -> str:
    return source_text[node.meta.start_pos:node.meta.end_pos].strip()


def _span_from_token(source_text: str, token: Any, *, source_id: str) -> DiagnosticSpan:
    return _source_span(source_text, token.start_pos, token.end_pos, source_id=source_id)


def _span_from_unexpected_input(source_text: str, error: Exception, *, source_id: str) -> DiagnosticSpan:
    start = max(getattr(error, "pos_in_stream", 0), 0)
    line = max(getattr(error, "line", 1), 1)
    column = max(getattr(error, "column", 1), 1)
    end = _unexpected_span_end(source_text, start)
    return DiagnosticSpan(
        start=start,
        end=end,
        line=line,
        column=column,
        end_line=line,
        end_column=column + max(end - start, 1),
        source_id=source_id,
    )


def _unexpected_span_end(source_text: str, start: int) -> int:
    if start >= len(source_text):
        return start + 1
    end = start
    while end < len(source_text) and not source_text[end].isspace():
        end += 1
    return max(end, start + 1)


def _unquote_string_literal(value: str) -> str:
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1].replace("''", "'")
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1].replace('""', '"')
    return value


def _consume_quoted_segment(text: str, start: int) -> int:
    quote = text[start]
    index = start + 1
    while index < len(text):
        if text[index] != quote:
            index += 1
            continue
        if index + 1 < len(text) and text[index + 1] == quote:
            index += 2
            continue
        return index + 1
    return len(text)


def _match_dictionary_reference(text: str, start: int) -> tuple[str, int] | None:
    prefix = "dictionary"
    if start > 0 and _is_identifier_char(text[start - 1]):
        return None
    if text[start : start + len(prefix)].lower() != prefix:
        return None

    index = start + len(prefix)
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text) or text[index] != ".":
        return None

    index += 1
    while index < len(text) and text[index].isspace():
        index += 1

    for name in ("tables", "columns"):
        end = index + len(name)
        if text[index:end].lower() != name:
            continue
        if end < len(text) and _is_identifier_char(text[end]):
            return None
        return (f'"dictionary.{name}"', end)
    return None


def _is_identifier_char(character: str) -> bool:
    return character == "_" or character.isalnum()


def _raise_session_parse_error(
    *,
    code: str,
    message: str,
    source_id: str,
    source_text: str,
    label_message: str,
    notes: Sequence[str] = (),
    span: DiagnosticSpan | None = None,
) -> None:
    actual_span = span or _whole_source_span(source_text, source_id=source_id)
    labels = (DiagnosticLabel(span=actual_span, message=label_message),) if actual_span is not None else ()
    diagnostic = Diagnostic(
        code=code,
        severity="error",
        message=message,
        stage="parse",
        span=actual_span,
        labels=labels,
        notes=tuple(notes),
        source_text=source_text,
    )
    raise ValueError(_render_session_diagnostic(diagnostic, source_id=source_id))


def raise_session_sql_execution_error(query: str, error: Exception) -> None:
    actual_span = _whole_source_span(query, source_id="sql")
    labels = (DiagnosticLabel(span=actual_span, message="sql execution"),) if actual_span is not None else ()
    diagnostic = Diagnostic(
        code="SESSION_SQL_EXECUTION_ERROR",
        severity="error",
        message="SQL execution failed for Session.sql.",
        stage="execute",
        span=actual_span,
        labels=labels,
        notes=(str(error),),
        source_text=query,
    )
    raise ValueError(_render_session_diagnostic(diagnostic, source_id="sql"))


def _render_session_diagnostic(diagnostic: Diagnostic, *, source_id: str) -> str:
    updated = _with_source_id(diagnostic, source_id=source_id)
    return render_diagnostics(
        RenderRequest(
            source_id=source_id,
            source_text=updated.source_text,
            diagnostics=(updated,),
        )
    )


def _with_source_id(diagnostic: Diagnostic, *, source_id: str) -> Diagnostic:
    span = None
    if diagnostic.span is not None:
        span = _source_span(
            diagnostic.source_text or "",
            diagnostic.span.start,
            diagnostic.span.end,
            source_id=source_id,
        )

    labels = tuple(
        DiagnosticLabel(
            span=_source_span(
                diagnostic.source_text or "",
                label.span.start,
                label.span.end,
                source_id=source_id,
            ),
            message=label.message,
            kind=label.kind,
        )
        for label in diagnostic.labels
    )

    return Diagnostic(
        code=diagnostic.code,
        severity=diagnostic.severity,
        message=diagnostic.message,
        location=diagnostic.location,
        stage=diagnostic.stage,
        span=span,
        labels=labels,
        notes=diagnostic.notes,
        source_text=diagnostic.source_text,
    )


def _whole_source_span(source_text: str, *, source_id: str) -> DiagnosticSpan | None:
    stripped = source_text.strip()
    if not stripped:
        return None
    start = source_text.find(stripped)
    end = start + len(stripped)
    return _source_span(source_text, start, end, source_id=source_id)


def _source_span(source_text: str, start: int, end: int, *, source_id: str) -> DiagnosticSpan:
    safe_start = max(min(start, len(source_text)), 0)
    safe_end = max(min(end, len(source_text)), safe_start + 1)
    prefix = source_text[:safe_start]
    line = prefix.count("\n") + 1
    last_newline = prefix.rfind("\n")
    column = safe_start + 1 if last_newline < 0 else safe_start - last_newline
    segment = source_text[safe_start:safe_end]
    end_line = line + segment.count("\n")
    end_column = column + max(safe_end - safe_start, 1) if "\n" not in segment else None
    return DiagnosticSpan(
        start=safe_start,
        end=safe_end,
        line=line,
        column=column,
        end_line=end_line,
        end_column=end_column,
        source_id=source_id,
    )


__all__ = [
    "SimpleFilterSpec",
    "SqlClassification",
    "classify_sql",
    "extract_sql_target",
    "parse_simple_filter",
    "raise_session_sql_execution_error",
]