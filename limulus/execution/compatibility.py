"""Missing-value semantics notices attached to execution results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..expression_semantics import _typed_expression_children
from ..models import CompatibilityNotice, DataSetRef
from ..syntax.models import BlockSyntax, LiteralSyntax
from .statement_expressions import block_expression_slots


_MISSING_VALUE_NOTICE = CompatibilityNotice(
    id="COMPAT_MISSING_VALUE_SEMANTICS",
    category="missing-values",
    summary="The missing-value literal '.' is treated as null.",
    link="compat://missing-value-semantics",
)

_EMPTY_STRING_NOTICE = CompatibilityNotice(
    id="COMPAT_EMPTY_STRING_SEMANTICS",
    category="missing-values",
    summary=(
        "Empty strings are distinct from null. "
        "missing() and cmiss() treat both as missing."
    ),
    link="compat://empty-string-semantics",
)


def _collect_compatibility_notices(
    block: BlockSyntax, inputs: Mapping[str, DataSetRef]
) -> tuple[CompatibilityNotice, ...]:
    """Detect missing-value and empty-string usage and return semantics notices."""

    has_missing = False
    has_empty_string = False
    pending = [slot.expression for slot in block_expression_slots(block)]
    while pending:
        expression = pending.pop()
        if isinstance(expression, LiteralSyntax):
            if expression.kind == "missing":
                has_missing = True
            elif expression.kind == "string" and expression.lexeme in {"''", '""'}:
                has_empty_string = True
        if has_missing and has_empty_string:
            break
        pending.extend(_typed_expression_children(expression))
    if not has_missing:
        for dataset in inputs.values():
            payload = dataset.payload
            if payload is None:
                continue
            if isinstance(payload, Mapping) and any(
                value is None for value in payload.values()
            ):
                has_missing = True
                break
            if isinstance(payload, Sequence) and not isinstance(
                payload, (str, bytes, bytearray)
            ) and any(
                item is None
                or (
                    isinstance(item, Mapping)
                    and any(value is None for value in item.values())
                )
                for item in payload
            ):
                has_missing = True
                break
    notices: list[CompatibilityNotice] = []
    if has_missing:
        notices.append(_MISSING_VALUE_NOTICE)
    if has_empty_string:
        notices.append(_EMPTY_STRING_NOTICE)
    return tuple(notices)
