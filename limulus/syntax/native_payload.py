"""Serialization and structural validation of typed native payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Literal, NoReturn, cast

from ..naming import _dataset_key
from .models import (
    ApplyReferenceSyntax,
    ArrayElementTargetSyntax,
    ArrayStatementSyntax,
    AssignStatementSyntax,
    BinarySyntax,
    BlockSyntax,
    CallSyntax,
    ConcatSyntax,
    DataStatementSyntax,
    DatasetOptionSyntax,
    DatasetReferenceSyntax,
    DeferredStatementError,
    DeleteStatementSyntax,
    DoStatementSyntax,
    ElseStatementSyntax,
    EndStatementSyntax,
    DeferredExpressionError,
    ExpressionSyntax,
    IfStatementSyntax,
    LabelStatementSyntax,
    LiteralSyntax,
    NameListStatementSyntax,
    NameSyntax,
    NameTargetSyntax,
    OutputStatementSyntax,
    RenameStatementSyntax,
    RunStatementSyntax,
    SkippedStatementSyntax,
    SourceDocument,
    SourceStatementSyntax,
    SourceSpan,
    StatementSyntax,
    StopStatementSyntax,
    SubscriptSyntax,
    SumStatementSyntax,
    UnarySyntax,
    WhereComparison,
    WhereExpression,
    WhereLogicalExpression,
    WhereNotExpression,
    WhereNullPredicate,
    WhereStatementSyntax,
)


NATIVE_PAYLOAD_VERSION = 1
# This is carried by the internal typed native program envelope.


class NativePayloadValidationError(ValueError):
    """A structural wire-contract rejection, before native deserialization.

    ``reason`` is deliberately a small transport classification so the native
    boundary can map version, tag, and field failures to its established
    invalid-payload diagnostic owner without this pure module owning a
    diagnostic policy.
    """

    def __init__(
        self,
        reason: Literal["version", "tag", "field"],
        path: str,
        message: str,
    ) -> None:
        super().__init__(f"{path}: {message}")
        self.reason = reason
        self.path = path


def _wire_error(
    reason: Literal["version", "tag", "field"], path: str, message: str
) -> NoReturn:
    raise NativePayloadValidationError(reason, path, message)


def _wire_object(
    value: object, path: str, fields: set[str]
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _wire_error("field", path, "expected an object")
    if any(not isinstance(key, str) for key in value):
        _wire_error("field", path, "object keys must be strings")
    actual = set(value)
    missing = fields - actual
    if missing:
        field = sorted(missing)[0]
        _wire_error(
            "field", f"{path}.{field}", f"missing required field {field!r}"
        )
    extra = actual - fields
    if extra:
        field = sorted(extra)[0]
        _wire_error("field", f"{path}.{field}", f"unknown field {field!r}")
    return cast(Mapping[str, object], value)


def _wire_string(value: object, path: str) -> str:
    if not isinstance(value, str):
        _wire_error("field", path, "expected a string")
    return value


def _wire_optional_string(value: object, path: str) -> None:
    if value is not None:
        _wire_string(value, path)


def _wire_bool(value: object, path: str) -> None:
    if not isinstance(value, bool):
        _wire_error("field", path, "expected a boolean")


def _wire_int(value: object, path: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if type(value) is not int:
        _wire_error("field", path, "expected an integer" + (" or null" if optional else ""))


def _wire_list(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        _wire_error("field", path, "expected an array")
    return cast(list[object], value)


def _wire_string_list(value: object, path: str) -> None:
    for index, item in enumerate(_wire_list(value, path)):
        _wire_string(item, f"{path}[{index}]")


_SPAN_FIELDS = {
    "start",
    "end",
    "line",
    "column",
    "end_line",
    "end_column",
    "source_id",
}


def _validate_span(value: object, path: str) -> None:
    span = _wire_object(value, path, _SPAN_FIELDS)
    for name in ("start", "end", "line", "column"):
        _wire_int(span[name], f"{path}.{name}")
    for name in ("end_line", "end_column"):
        _wire_int(span[name], f"{path}.{name}", optional=True)
    _wire_string(span["source_id"], f"{path}.source_id")


def _validate_optional_span(value: object, path: str) -> None:
    if value is not None:
        _validate_span(value, path)


def _validate_apply_reference(value: object, path: str) -> None:
    reference = _wire_object(value, path, {"target", "call_span", "owner_span"})
    _wire_string(reference["target"], f"{path}.target")
    _validate_span(reference["call_span"], f"{path}.call_span")
    _validate_span(reference["owner_span"], f"{path}.owner_span")


def _validate_apply_references(value: object, path: str) -> None:
    for index, reference in enumerate(_wire_list(value, path)):
        _validate_apply_reference(reference, f"{path}[{index}]")


def _validate_deferred(value: object, path: str) -> None:
    deferred = _wire_object(
        value,
        path,
        {
            "tag",
            "error_kind",
            "native_cause",
            "where_syntax_summary",
            "span",
            "display_text",
        },
    )
    if deferred["tag"] != "deferred":
        _wire_error("tag", f"{path}.tag", "expected 'deferred'")
    _wire_string(deferred["error_kind"], f"{path}.error_kind")
    _wire_string(deferred["native_cause"], f"{path}.native_cause")
    _wire_string(deferred["where_syntax_summary"], f"{path}.where_syntax_summary")
    _validate_span(deferred["span"], f"{path}.span")
    _wire_string(deferred["display_text"], f"{path}.display_text")


def _validate_expression(value: object, path: str) -> str:
    if not isinstance(value, Mapping):
        _wire_error("field", path, "expected an expression object")
    tag = value.get("tag")
    if not isinstance(tag, str):
        _wire_error("tag", f"{path}.tag", "expected an expression tag")

    common = {"tag", "span", "origin"}
    variants: dict[str, set[str]] = {
        "literal": {"kind", "lexeme"},
        "name": {"name"},
        "unary": {"operator", "operand"},
        "binary": {"operator", "left", "right"},
        "call": {"name", "arguments"},
        "subscript": {"base", "index"},
        "concat": {"parts"},
        "deferred": {
            "error_kind",
            "native_cause",
            "where_syntax_summary",
            "display_text",
            "apply_references",
        },
    }
    if tag not in variants:
        _wire_error("tag", f"{path}.tag", f"unknown expression tag {tag!r}")
    expression = _wire_object(value, path, common | variants[tag])
    _validate_span(expression["span"], f"{path}.span")
    _validate_optional_span(expression["origin"], f"{path}.origin")

    if tag == "literal":
        kind = _wire_string(expression["kind"], f"{path}.kind")
        if kind not in {"missing", "string", "numeric", "format"}:
            _wire_error("tag", f"{path}.kind", "unknown literal kind")
        _wire_string(expression["lexeme"], f"{path}.lexeme")
    elif tag == "name":
        _wire_string(expression["name"], f"{path}.name")
    elif tag == "unary":
        operator = _wire_string(expression["operator"], f"{path}.operator")
        if operator not in {"not", "-"}:
            _wire_error("field", f"{path}.operator", "unsupported unary operator")
        _validate_expression(expression["operand"], f"{path}.operand")
    elif tag == "binary":
        operator = _wire_string(expression["operator"], f"{path}.operator")
        if operator not in {
            "or", "and", "==", "!=", ">", ">=", "<", "<=", "+", "-", "*", "/", "**", "mod"
        }:
            _wire_error("field", f"{path}.operator", "unsupported binary operator")
        _validate_expression(expression["left"], f"{path}.left")
        _validate_expression(expression["right"], f"{path}.right")
    elif tag == "call":
        _wire_string(expression["name"], f"{path}.name")
        for index, argument in enumerate(_wire_list(expression["arguments"], f"{path}.arguments")):
            _validate_expression(argument, f"{path}.arguments[{index}]")
    elif tag == "subscript":
        _validate_expression(expression["base"], f"{path}.base")
        _validate_expression(expression["index"], f"{path}.index")
    elif tag == "concat":
        parts = _wire_list(expression["parts"], f"{path}.parts")
        if len(parts) < 2:
            _wire_error("field", f"{path}.parts", "concat requires at least two parts")
        for index, part in enumerate(parts):
            _validate_expression(part, f"{path}.parts[{index}]")
    else:
        _wire_string(expression["error_kind"], f"{path}.error_kind")
        _wire_string(expression["native_cause"], f"{path}.native_cause")
        _wire_string(expression["where_syntax_summary"], f"{path}.where_syntax_summary")
        _wire_string(expression["display_text"], f"{path}.display_text")
        _validate_apply_references(expression["apply_references"], f"{path}.apply_references")
    return tag


def _validate_where_expression(value: object, path: str) -> None:
    if not isinstance(value, Mapping):
        _wire_error("field", path, "expected a restrictive WHERE object")
    kind = value.get("kind")
    if not isinstance(kind, str):
        _wire_error("tag", f"{path}.kind", "expected a WHERE variant kind")
    variants: dict[str, set[str]] = {
        "comparison": {"kind", "column", "operator", "value"},
        "null": {"kind", "column", "negated"},
        "logical": {"kind", "operator", "operands"},
        "not": {"kind", "operand"},
    }
    if kind not in variants:
        _wire_error("tag", f"{path}.kind", f"unknown WHERE kind {kind!r}")
    expression = _wire_object(value, path, variants[kind])
    if kind == "comparison":
        _wire_string(expression["column"], f"{path}.column")
        operator = _wire_string(expression["operator"], f"{path}.operator")
        if operator not in {"=", "==", "!=", ">", ">=", "<", "<="}:
            _wire_error("field", f"{path}.operator", "unsupported WHERE comparison operator")
        scalar = expression["value"]
        if not (
            isinstance(scalar, (str, bool, int, float))
            and (not isinstance(scalar, float) or isfinite(scalar))
        ):
            _wire_error("field", f"{path}.value", "expected a JSON scalar")
    elif kind == "null":
        _wire_string(expression["column"], f"{path}.column")
        _wire_bool(expression["negated"], f"{path}.negated")
    elif kind == "logical":
        operator = _wire_string(expression["operator"], f"{path}.operator")
        if operator not in {"and", "or"}:
            _wire_error("field", f"{path}.operator", "unsupported WHERE logical operator")
        for index, operand in enumerate(_wire_list(expression["operands"], f"{path}.operands")):
            _validate_where_expression(operand, f"{path}.operands[{index}]")
    else:
        _validate_where_expression(expression["operand"], f"{path}.operand")


_OPTION_FIELDS = {
    "in_var",
    "keep_vars",
    "drop_vars",
    "where_expr",
    "rename_map",
    "firstobs",
    "obs",
}


def _validate_options(value: object, path: str) -> None:
    options = _wire_object(value, path, _OPTION_FIELDS)
    _wire_optional_string(options["in_var"], f"{path}.in_var")
    _wire_string_list(options["keep_vars"], f"{path}.keep_vars")
    _wire_string_list(options["drop_vars"], f"{path}.drop_vars")
    if options["where_expr"] is not None:
        _validate_where_expression(options["where_expr"], f"{path}.where_expr")
    rename_map = options["rename_map"]
    if not isinstance(rename_map, Mapping):
        _wire_error("field", f"{path}.rename_map", "expected an object")
    for key, target in rename_map.items():
        _wire_string(key, f"{path}.rename_map key")
        _wire_string(target, f"{path}.rename_map[{key!r}]")
    _wire_int(options["firstobs"], f"{path}.firstobs", optional=True)
    _wire_int(options["obs"], f"{path}.obs", optional=True)


def _validate_dataset_reference(value: object, path: str) -> None:
    reference = _wire_object(value, path, {"name", "options", "span"})
    _wire_string(reference["name"], f"{path}.name")
    _validate_options(reference["options"], f"{path}.options")
    _validate_span(reference["span"], f"{path}.span")


def _validate_dataset_references(value: object, path: str) -> None:
    for index, reference in enumerate(_wire_list(value, path)):
        _validate_dataset_reference(reference, f"{path}[{index}]")


def _validate_assignment_target(value: object, path: str) -> str:
    if not isinstance(value, Mapping):
        _wire_error("field", path, "expected an assignment target object")
    tag = value.get("tag")
    if tag == "name":
        target = _wire_object(value, path, {"tag", "name", "span"})
        _wire_string(target["name"], f"{path}.name")
        _validate_span(target["span"], f"{path}.span")
        return "name"
    if tag == "array_element":
        target = _wire_object(value, path, {"tag", "name", "delimiter", "index", "span"})
        _wire_string(target["name"], f"{path}.name")
        delimiter = _wire_string(target["delimiter"], f"{path}.delimiter")
        if delimiter not in {"(", "[", "{"}:
            _wire_error("field", f"{path}.delimiter", "unsupported array delimiter")
        _validate_expression(target["index"], f"{path}.index")
        _validate_span(target["span"], f"{path}.span")
        return "array_element"
    _wire_error("tag", f"{path}.tag", f"unknown assignment target tag {tag!r}")


def _validate_inline_action(value: object, path: str) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        _wire_error("field", path, "expected an inline action object or null")
    tag = value.get("tag")
    if tag == "statement":
        action = _wire_object(value, path, {"tag", "statement"})
        _validate_statement(action["statement"], f"{path}.statement")
    elif tag == "deferred":
        _validate_deferred(value, path)
    else:
        _wire_error("tag", f"{path}.tag", f"unknown inline action tag {tag!r}")


_STATEMENT_TAGS = {
    "DATA": "data",
    "SET": "source",
    "MERGE": "source",
    "WHERE": "where",
    "ASSIGN": "assign",
    "SUM": "sum",
    "KEEP": "name_list",
    "DROP": "name_list",
    "BY": "name_list",
    "RETAIN": "name_list",
    "LABEL": "label",
    "RENAME": "rename",
    "OUTPUT": "output",
    "IF": "if",
    "ELSE IF": "if",
    "ELSE": "else",
    "DO": "do",
    "END": "end",
    "ARRAY": "array",
    "DELETE": "delete",
    "STOP": "stop",
    "RUN": "run",
    "SKIPPED": "skipped",
}


def _validate_statement(value: object, path: str) -> None:
    statement = _wire_object(
        value,
        path,
        {"kind", "payload", "display_text", "span", "origin", "apply_references"},
    )
    kind = _wire_string(statement["kind"], f"{path}.kind")
    if kind not in _STATEMENT_TAGS:
        _wire_error("tag", f"{path}.kind", f"unknown statement kind {kind!r}")
    _wire_string(statement["display_text"], f"{path}.display_text")
    _validate_span(statement["span"], f"{path}.span")
    _validate_optional_span(statement["origin"], f"{path}.origin")
    _validate_apply_references(statement["apply_references"], f"{path}.apply_references")

    payload_value = statement["payload"]
    if not isinstance(payload_value, Mapping):
        _wire_error("field", f"{path}.payload", "expected a statement payload object")
    payload_tag = payload_value.get("tag")
    if payload_tag == "deferred":
        _validate_deferred(payload_value, f"{path}.payload")
        return
    expected_tag = _STATEMENT_TAGS[kind]
    if payload_tag != expected_tag:
        _wire_error("tag", f"{path}.payload.tag", f"{kind} requires tag {expected_tag!r}")

    payload_fields: dict[str, set[str]] = {
        "data": {"tag", "outputs"},
        "source": {"tag", "refs", "options"},
        "where": {"tag", "condition"},
        "assign": {"tag", "target", "value"},
        "sum": {"tag", "target", "increment"},
        "name_list": {"tag", "names"},
        "label": {"tag"},
        "rename": {"tag", "rename_map"},
        "output": {"tag", "targets"},
        "if": {"tag", "condition", "then_action", "is_subset", "is_then_do"},
        "else": {"tag", "action"},
        "do": {"tag", "loop_var", "start", "end"},
        "array": {
            "tag", "array_name", "variables", "span", "declared_size",
            "wildcard_size", "character_array",
        },
        "delete": {"tag"},
        "stop": {"tag"},
        "end": {"tag"},
        "run": {"tag"},
        "skipped": {"tag"},
    }
    statement_payload = _wire_object(
        payload_value, f"{path}.payload", payload_fields[expected_tag]
    )

    if expected_tag == "data":
        _validate_dataset_references(
            statement_payload["outputs"], f"{path}.payload.outputs"
        )
    elif expected_tag == "source":
        _validate_dataset_references(
            statement_payload["refs"], f"{path}.payload.refs"
        )
        options = _wire_object(
            statement_payload["options"],
            f"{path}.payload.options",
            {"indsname_var", "end_var"},
        )
        _wire_optional_string(options["indsname_var"], f"{path}.payload.options.indsname_var")
        _wire_optional_string(options["end_var"], f"{path}.payload.options.end_var")
    elif expected_tag == "where":
        _validate_expression(
            statement_payload["condition"], f"{path}.payload.condition"
        )
    elif expected_tag == "assign":
        _validate_assignment_target(
            statement_payload["target"], f"{path}.payload.target"
        )
        _validate_expression(statement_payload["value"], f"{path}.payload.value")
    elif expected_tag == "sum":
        if (
            _validate_assignment_target(
                statement_payload["target"], f"{path}.payload.target"
            )
            != "name"
        ):
            _wire_error("tag", f"{path}.payload.target.tag", "SUM target must be a name")
        _validate_expression(
            statement_payload["increment"], f"{path}.payload.increment"
        )
    elif expected_tag == "name_list":
        for index, name in enumerate(
            _wire_list(statement_payload["names"], f"{path}.payload.names")
        ):
            if _validate_expression(name, f"{path}.payload.names[{index}]") != "name":
                _wire_error("tag", f"{path}.payload.names[{index}].tag", "name list entries must use the name variant")
    elif expected_tag == "rename":
        rename_map = statement_payload["rename_map"]
        if not isinstance(rename_map, Mapping):
            _wire_error("field", f"{path}.payload.rename_map", "expected an object")
        for old_name, new_name in rename_map.items():
            _wire_string(old_name, f"{path}.payload.rename_map key")
            _wire_string(new_name, f"{path}.payload.rename_map[{old_name!r}]")
    elif expected_tag == "output":
        _validate_dataset_references(
            statement_payload["targets"], f"{path}.payload.targets"
        )
    elif expected_tag == "if":
        _validate_expression(
            statement_payload["condition"], f"{path}.payload.condition"
        )
        _validate_inline_action(
            statement_payload["then_action"], f"{path}.payload.then_action"
        )
        _wire_bool(statement_payload["is_subset"], f"{path}.payload.is_subset")
        _wire_bool(
            statement_payload["is_then_do"], f"{path}.payload.is_then_do"
        )
    elif expected_tag == "else":
        _validate_inline_action(
            statement_payload["action"], f"{path}.payload.action"
        )
    elif expected_tag == "do":
        _wire_optional_string(statement_payload["loop_var"], f"{path}.payload.loop_var")
        start = statement_payload["start"]
        end = statement_payload["end"]
        if (statement_payload["loop_var"] is None) != (start is None) or (start is None) != (end is None):
            _wire_error("field", f"{path}.payload", "DO loop variable and bounds must be all present or all null")
        if start is not None:
            _validate_expression(start, f"{path}.payload.start")
            _validate_expression(end, f"{path}.payload.end")
    elif expected_tag == "array":
        _wire_string(statement_payload["array_name"], f"{path}.payload.array_name")
        _wire_string_list(statement_payload["variables"], f"{path}.payload.variables")
        _validate_span(statement_payload["span"], f"{path}.payload.span")
        _wire_int(statement_payload["declared_size"], f"{path}.payload.declared_size", optional=True)
        _wire_bool(statement_payload["wildcard_size"], f"{path}.payload.wildcard_size")
        _wire_bool(statement_payload["character_array"], f"{path}.payload.character_array")


def validate_native_program(payload: object) -> None:
    """Validate the version-1 typed wire shape without changing its values.

    This validates transport fields and tagged unions only. It does not infer
    source, replan outputs, interpret option semantics, or normalize any wire
    values. The producer's final output option map must cover the normalized
    final target keys, while duplicate final targets and an unregistered
    default target retain their existing downstream routing behavior.
    """

    if not isinstance(payload, Mapping):
        _wire_error("field", "program", "expected an object")
    raw_version = payload.get("version")
    if type(raw_version) is not int:
        _wire_error("field", "version", "required version must be an integer")
    if raw_version != NATIVE_PAYLOAD_VERSION:
        _wire_error(
            "version",
            "version",
            f"unsupported native payload version {raw_version}; expected {NATIVE_PAYLOAD_VERSION}",
        )
    program = _wire_object(
        payload,
        "program",
        {
            "version",
            "output_targets",
            "default_target",
            "suppress_implicit_output",
            "effective_output_options",
            "statements",
        },
    )

    target_values = _wire_list(program["output_targets"], "output_targets")
    normalized_targets: set[str] = set()
    for index, target in enumerate(target_values):
        target_name = _wire_string(target, f"output_targets[{index}]")
        normalized_targets.add(_dataset_key(target_name))
    _wire_optional_string(program["default_target"], "default_target")
    _wire_bool(program["suppress_implicit_output"], "suppress_implicit_output")

    option_map = program["effective_output_options"]
    if not isinstance(option_map, Mapping):
        _wire_error("field", "effective_output_options", "expected an object")
    normalized_options: set[str] = set()
    for target_name, options in option_map.items():
        key = _wire_string(target_name, "effective_output_options key")
        normalized_key = _dataset_key(key)
        if normalized_key in normalized_options:
            _wire_error(
                "field",
                f"effective_output_options[{key!r}]",
                "duplicate normalized output target key",
            )
        normalized_options.add(normalized_key)
        _validate_options(options, f"effective_output_options[{key!r}]")
    if normalized_options != normalized_targets:
        _wire_error(
            "field",
            "effective_output_options",
            "normalized target keys must match output_targets",
        )

    for index, statement in enumerate(_wire_list(program["statements"], "statements")):
        _validate_statement(statement, f"statements[{index}]")


def serialize_span(span: SourceSpan) -> dict[str, int | str | None]:
    """Return a source-aware primitive representation of a syntax span."""

    return {
        "start": span.start,
        "end": span.end,
        "line": span.line,
        "column": span.column,
        "end_line": span.end_line,
        "end_column": span.end_column,
        "source_id": span.source_id,
    }


def _serialize_apply_reference(
    reference: ApplyReferenceSyntax,
) -> dict[str, object]:
    return {
        "target": reference.target,
        "call_span": serialize_span(reference.call_span),
        "owner_span": serialize_span(reference.owner_span),
    }


def _common_fields(
    tag: str,
    span: SourceSpan,
    origin: SourceSpan | None,
) -> dict[str, object]:
    return {
        "tag": tag,
        "span": serialize_span(span),
        "origin": serialize_span(origin) if origin is not None else None,
    }


def serialize_expression(expression: ExpressionSyntax) -> dict[str, object]:
    """Serialize an expression node recursively without reading display text."""

    if isinstance(expression, LiteralSyntax):
        return {
            **_common_fields("literal", expression.span, expression.origin),
            "kind": expression.kind,
            "lexeme": expression.lexeme,
        }
    if isinstance(expression, NameSyntax):
        return {
            **_common_fields("name", expression.span, expression.origin),
            "name": expression.name,
        }
    if isinstance(expression, UnarySyntax):
        return {
            **_common_fields("unary", expression.span, expression.origin),
            "operator": expression.operator,
            "operand": serialize_expression(expression.operand),
        }
    if isinstance(expression, BinarySyntax):
        return {
            **_common_fields("binary", expression.span, expression.origin),
            "operator": expression.operator,
            "left": serialize_expression(expression.left),
            "right": serialize_expression(expression.right),
        }
    if isinstance(expression, CallSyntax):
        return {
            **_common_fields("call", expression.span, expression.origin),
            "name": expression.name,
            "arguments": [
                serialize_expression(argument) for argument in expression.arguments
            ],
        }
    if isinstance(expression, SubscriptSyntax):
        return {
            **_common_fields("subscript", expression.span, expression.origin),
            "base": serialize_expression(expression.base),
            "index": serialize_expression(expression.index),
        }
    if isinstance(expression, ConcatSyntax):
        return {
            **_common_fields("concat", expression.span, expression.origin),
            "parts": [serialize_expression(part) for part in expression.parts],
        }
    if isinstance(expression, DeferredExpressionError):
        return {
            **_common_fields("deferred", expression.span, None),
            "error_kind": expression.error_kind,
            "native_cause": expression.native_cause,
            "where_syntax_summary": expression.where_syntax_summary,
            # These are saved diagnostic / resolution facts, kept outside any
            # partial calculation fields because a Deferred node is whole.
            "display_text": expression.display_text,
            "apply_references": [
                _serialize_apply_reference(reference)
                for reference in expression.apply_references
            ],
        }
    raise TypeError(f"unsupported expression syntax: {type(expression).__name__}")


def _source_slice(source: SourceDocument, span: SourceSpan) -> str:
    """Slice display text from its saved span without deriving syntax from it."""

    if span.source_id != source.source_id:
        raise ValueError(
            f"span source_id {span.source_id!r} does not match source "
            f"{source.source_id!r}"
        )
    if span.start < 0 or span.end > len(source.text):
        raise ValueError(
            f"span [{span.start}, {span.end}) is outside source length "
            f"{len(source.text)}"
        )
    return source.text[span.start : span.end]


def _serialize_where_expression(expression: WhereExpression) -> dict[str, object]:
    """Serialize the restrictive dataset-option WHERE union as its own dialect."""

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
            "operands": [
                _serialize_where_expression(operand)
                for operand in expression.operands
            ],
        }
    if isinstance(expression, WhereNotExpression):
        return {
            "kind": "not",
            "operand": _serialize_where_expression(expression.operand),
        }
    raise TypeError(
        f"unsupported restrictive WHERE syntax: {type(expression).__name__}"
    )


def _serialize_dataset_options(options: DatasetOptionSyntax) -> dict[str, object]:
    """Serialize native option fields, leaving LABEL with its Python owner."""

    return {
        "in_var": options.in_var,
        "keep_vars": list(options.keep_vars),
        "drop_vars": list(options.drop_vars),
        "where_expr": (
            _serialize_where_expression(options.where_expr)
            if options.where_expr is not None
            else None
        ),
        "rename_map": {source: target for source, target in options.rename_map},
        "firstobs": options.firstobs,
        "obs": options.obs,
    }


def _serialize_dataset_reference(
    reference: DatasetReferenceSyntax,
) -> dict[str, object]:
    return {
        "name": reference.name,
        "options": _serialize_dataset_options(reference.options),
        "span": serialize_span(reference.span),
    }


def _serialize_assignment_target(
    target: NameTargetSyntax | ArrayElementTargetSyntax,
) -> dict[str, object]:
    if isinstance(target, NameTargetSyntax):
        return {
            "tag": "name",
            "name": target.name,
            "span": serialize_span(target.span),
        }
    if isinstance(target, ArrayElementTargetSyntax):
        return {
            "tag": "array_element",
            "name": target.name,
            "delimiter": target.delimiter,
            "index": serialize_expression(target.index),
            "span": serialize_span(target.span),
        }
    raise TypeError(f"unsupported assignment target: {type(target).__name__}")


def _serialize_deferred_statement(
    error: DeferredStatementError,
) -> dict[str, object]:
    return {
        "tag": "deferred",
        "error_kind": error.error_kind,
        "native_cause": error.native_cause,
        "where_syntax_summary": error.where_syntax_summary,
        "span": serialize_span(error.span),
        "display_text": error.display_text,
    }


def _serialize_inline_action(
    action: StatementSyntax | DeferredStatementError | None,
    source: SourceDocument,
) -> dict[str, object] | None:
    if action is None:
        return None
    if isinstance(action, StatementSyntax):
        return {
            "tag": "statement",
            "statement": serialize_statement(action, source),
        }
    if isinstance(action, DeferredStatementError):
        return _serialize_deferred_statement(action)
    raise TypeError(f"unsupported inline action: {type(action).__name__}")


def _serialize_statement_payload(
    statement: StatementSyntax,
    source: SourceDocument,
) -> dict[str, object]:
    payload = statement.payload
    if isinstance(payload, DeferredStatementError):
        return _serialize_deferred_statement(payload)
    if isinstance(payload, DataStatementSyntax):
        return {
            "tag": "data",
            "outputs": [
                _serialize_dataset_reference(reference)
                for reference in payload.outputs
            ],
        }
    if isinstance(payload, SourceStatementSyntax):
        return {
            "tag": "source",
            "refs": [
                _serialize_dataset_reference(reference) for reference in payload.refs
            ],
            "options": {
                "indsname_var": payload.options.indsname_var,
                "end_var": payload.options.end_var,
            },
        }
    if isinstance(payload, WhereStatementSyntax):
        return {
            "tag": "where",
            "condition": serialize_expression(payload.condition),
        }
    if isinstance(payload, AssignStatementSyntax):
        return {
            "tag": "assign",
            "target": _serialize_assignment_target(payload.target),
            "value": serialize_expression(payload.value),
        }
    if isinstance(payload, SumStatementSyntax):
        return {
            "tag": "sum",
            "target": _serialize_assignment_target(payload.target),
            "increment": serialize_expression(payload.increment),
        }
    if isinstance(payload, NameListStatementSyntax):
        return {
            "tag": "name_list",
            "names": [serialize_expression(name) for name in payload.names],
        }
    if isinstance(payload, LabelStatementSyntax):
        # LABEL values belong to Python's metadata owner, not native row calculation.
        return {"tag": "label"}
    if isinstance(payload, RenameStatementSyntax):
        return {
            "tag": "rename",
            "rename_map": {source_name: target for source_name, target in payload.pairs},
        }
    if isinstance(payload, OutputStatementSyntax):
        return {
            "tag": "output",
            "targets": [
                _serialize_dataset_reference(reference)
                for reference in payload.targets
            ],
        }
    if isinstance(payload, IfStatementSyntax):
        return {
            "tag": "if",
            "condition": serialize_expression(payload.condition),
            "then_action": _serialize_inline_action(payload.then_action, source),
            "is_subset": payload.is_subset,
            "is_then_do": payload.is_then_do,
        }
    if isinstance(payload, ElseStatementSyntax):
        return {
            "tag": "else",
            "action": _serialize_inline_action(payload.action, source),
        }
    if isinstance(payload, DoStatementSyntax):
        return {
            "tag": "do",
            "loop_var": payload.loop_var,
            "start": (
                serialize_expression(payload.start)
                if payload.start is not None
                else None
            ),
            "end": (
                serialize_expression(payload.end) if payload.end is not None else None
            ),
        }
    if isinstance(payload, ArrayStatementSyntax):
        return {
            "tag": "array",
            "array_name": payload.array_name,
            "variables": list(payload.variables),
            "span": serialize_span(payload.span),
            "declared_size": payload.declared_size,
            "wildcard_size": payload.wildcard_size,
            "character_array": payload.character_array,
        }
    if isinstance(payload, DeleteStatementSyntax):
        return {"tag": "delete"}
    if isinstance(payload, StopStatementSyntax):
        return {"tag": "stop"}
    if isinstance(payload, EndStatementSyntax):
        return {"tag": "end"}
    if isinstance(payload, RunStatementSyntax):
        return {"tag": "run"}
    if isinstance(payload, SkippedStatementSyntax):
        return {"tag": "skipped"}
    raise TypeError(f"unsupported statement payload: {type(payload).__name__}")


def serialize_statement(
    statement: StatementSyntax,
    source: SourceDocument,
) -> dict[str, object]:
    """Serialize one statement from its typed payload and saved source span."""

    return {
        "kind": statement.kind,
        "payload": _serialize_statement_payload(statement, source),
        "display_text": _source_slice(source, statement.span),
        "span": serialize_span(statement.span),
        "origin": (
            serialize_span(statement.origin)
            if statement.origin is not None
            else None
        ),
        "apply_references": [
            _serialize_apply_reference(reference)
            for reference in statement.apply_references
        ],
    }


def serialize_block(
    block: BlockSyntax,
    *,
    source: SourceDocument,
    output_targets: Sequence[str],
    default_target: str | None,
    suppress_implicit_output: bool,
    effective_output_options: Mapping[str, DatasetOptionSyntax],
) -> dict[str, object]:
    """Serialize a block with output policy explicitly supplied by its owner."""

    return {
        "version": NATIVE_PAYLOAD_VERSION,
        "output_targets": list(output_targets),
        "default_target": default_target,
        "suppress_implicit_output": suppress_implicit_output,
        "effective_output_options": {
            target: _serialize_dataset_options(options)
            for target, options in effective_output_options.items()
        },
        "statements": [
            serialize_statement(statement, source) for statement in block.statements
        ],
    }


__all__ = [
    "NATIVE_PAYLOAD_VERSION",
    "NativePayloadValidationError",
    "serialize_block",
    "serialize_expression",
    "serialize_span",
    "serialize_statement",
    "validate_native_program",
]
