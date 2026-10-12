"""Typed execution-tree rewrites.

The helper plan owns which original calls can be replaced.  This module only
walks immutable syntax nodes, preserves deferred descriptors, and asks the
existing output-handoff owner to replan the resulting block.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
import unicodedata

from ..syntax.models import (
    ApplyReferenceSyntax,
    ArrayElementTargetSyntax,
    AssignStatementSyntax,
    BinarySyntax,
    BlockSyntax,
    CallSyntax,
    ConcatSyntax,
    DeferredExpressionError,
    DeferredStatementError,
    DoStatementSyntax,
    ElseStatementSyntax,
    ExpressionSyntax,
    IfStatementSyntax,
    LiteralSyntax,
    NameSyntax,
    SourceDocument,
    SourceSpan,
    StatementSyntax,
    SubscriptSyntax,
    SumStatementSyntax,
    UnarySyntax,
    WhereStatementSyntax,
)
from .contracts import TypedOutputPlanRequest
from .output_handoff import ExecutionOutputHandoffService
from .rewrites import TypedHelperPlan, TypedLagLeadSpec
from .typed_output_planning import TypedOutputTargetPlan

ApplyOccurrenceKey = tuple[str, int, int]


@dataclass(frozen=True)
class _ApplyOccurrence:
    target: str | None


@dataclass(frozen=True)
class TypedSyntaxRewriteResult:
    """One transformed syntax block and the plans derived from that structure."""

    block: BlockSyntax
    apply_references: tuple[ApplyReferenceSyntax, ...]
    output_plan: TypedOutputTargetPlan

    def __post_init__(self) -> None:
        object.__setattr__(self, "apply_references", tuple(self.apply_references))


def rewrite_typed_block(
    block: BlockSyntax,
    helper_plan: TypedHelperPlan,
    *,
    source: SourceDocument,
    output_handoff_service: ExecutionOutputHandoffService,
    explicit_output_targets: Sequence[str] = (),
    generated_apply_references: Sequence[ApplyReferenceSyntax] = (),
) -> TypedSyntaxRewriteResult:
    """Apply source-owned helper decisions and replan outputs from the result.

    A planned helper is matched by the identity of its original ``CallSyntax``
    object.  Span equality alone is insufficient because a generated node can
    reuse the same source span.  The traversal never reads display text or
    reparses source; whole-expression and statement deferred nodes stay opaque.
    """

    if not isinstance(block, BlockSyntax):
        raise TypeError("block must be a BlockSyntax")
    if not isinstance(helper_plan, TypedHelperPlan):
        raise TypeError("helper_plan must be a TypedHelperPlan")
    if not isinstance(source, SourceDocument):
        raise TypeError("source must be a SourceDocument")
    if not isinstance(output_handoff_service, ExecutionOutputHandoffService):
        raise TypeError(
            "output_handoff_service must be an ExecutionOutputHandoffService"
        )

    source_owned_specs = {
        id(spec.call): spec
        for spec in helper_plan.lag_lead_specs
        if spec.is_source_owned
    }
    rewritten_statements: list[StatementSyntax] = []
    for statement in block.statements:
        rewritten, _ = _rewrite_statement(statement, source_owned_specs)
        rewritten_statements.append(rewritten)

    rewritten_block = (
        block
        if all(
            original is rewritten
            for original, rewritten in zip(block.statements, rewritten_statements)
        )
        else replace(block, statements=tuple(rewritten_statements))
    )
    generated_references = tuple(generated_apply_references)
    for reference in generated_references:
        if not isinstance(reference, ApplyReferenceSyntax):
            raise TypeError(
                "generated_apply_references must contain ApplyReferenceSyntax values"
            )
        if (
            reference.call_span.source_id != source.source_id
            or reference.call_span.end > len(source.text)
            or reference.owner_span.source_id != source.source_id
            or reference.owner_span.end > len(source.text)
        ):
            raise ValueError(
                "generated apply metadata spans must belong to the supplied source"
            )

    rewritten_block = _reconcile_apply_inventories(
        rewritten_block,
        generated_apply_references=generated_references,
    )
    apply_references = tuple(
        reference
        for statement in rewritten_block.statements
        for reference in statement.apply_references
    )
    output_plan = output_handoff_service.plan_typed(
        TypedOutputPlanRequest(
            block=rewritten_block,
            source=source,
            explicit_output_targets=tuple(explicit_output_targets),
        )
    )
    return TypedSyntaxRewriteResult(
        block=rewritten_block,
        apply_references=apply_references,
        output_plan=output_plan,
    )


def _reconcile_apply_inventories(
    block: BlockSyntax,
    *,
    generated_apply_references: tuple[ApplyReferenceSyntax, ...],
) -> BlockSyntax:
    """Keep inventories aligned with retained and explicitly generated calls.

    Live apply calls are read from typed expression nodes. Whole deferred
    expressions contribute their saved inventory, and deferred statements
    preserve references within their saved span because they have no partial
    syntax tree. New apply calls need explicit metadata; source text is never
    scanned to infer a target.
    """

    live_apply_occurrences: dict[ApplyOccurrenceKey, _ApplyOccurrence] = {}
    deferred_expression_references: dict[
        ApplyOccurrenceKey, list[ApplyReferenceSyntax]
    ] = {}
    deferred_statement_spans: list[SourceSpan] = []
    for statement in block.statements:
        _collect_statement_apply_occurrences(
            statement,
            live_apply_occurrences,
            deferred_expression_references,
            deferred_statement_spans,
        )

    generated_by_key: dict[ApplyOccurrenceKey, ApplyReferenceSyntax] = {}
    for reference in generated_apply_references:
        key = _apply_reference_key(reference)
        occurrence = live_apply_occurrences.get(key)
        if occurrence is None:
            raise ValueError(
                "generated apply metadata must match a retained typed apply CallSyntax"
            )
        if occurrence.target is not None and reference.target != occurrence.target:
            raise ValueError(
                "generated apply metadata target must match the typed apply CallSyntax"
            )
        if key in generated_by_key:
            raise ValueError("generated apply metadata must be unique by occurrence")
        generated_by_key[key] = reference

    existing_by_key: dict[ApplyOccurrenceKey, list[ApplyReferenceSyntax]] = {}
    for statement in _walk_statements(block.statements):
        for reference in statement.apply_references:
            existing_by_key.setdefault(_apply_reference_key(reference), []).append(
                reference
            )
    for key, occurrence in live_apply_occurrences.items():
        existing = existing_by_key.get(key, [])
        generated = generated_by_key.get(key)
        if not existing and generated is None:
            raise ValueError(
                "new typed apply CallSyntax requires explicit apply metadata"
            )
        if generated is None and occurrence.target is not None and any(
            reference.target != occurrence.target for reference in existing
        ):
            raise ValueError("changed apply target requires explicit apply metadata")

    rewritten_statements = tuple(
        _reconcile_statement_apply_references(
            statement,
            live_apply_occurrences=live_apply_occurrences,
            deferred_expression_references=deferred_expression_references,
            deferred_statement_spans=deferred_statement_spans,
            generated_references=tuple(generated_by_key.values()),
        )
        for statement in block.statements
    )
    if all(
        original is rewritten
        for original, rewritten in zip(block.statements, rewritten_statements)
    ):
        return block
    return replace(block, statements=rewritten_statements)


def _reconcile_statement_apply_references(
    statement: StatementSyntax,
    *,
    live_apply_occurrences: dict[ApplyOccurrenceKey, _ApplyOccurrence],
    deferred_expression_references: dict[
        ApplyOccurrenceKey, list[ApplyReferenceSyntax]
    ],
    deferred_statement_spans: list[SourceSpan],
    generated_references: tuple[ApplyReferenceSyntax, ...],
) -> StatementSyntax:
    payload = statement.payload
    if isinstance(payload, IfStatementSyntax) and isinstance(
        payload.then_action, StatementSyntax
    ):
        action = _reconcile_statement_apply_references(
            payload.then_action,
            live_apply_occurrences=live_apply_occurrences,
            deferred_expression_references=deferred_expression_references,
            deferred_statement_spans=deferred_statement_spans,
            generated_references=generated_references,
        )
        if action is not payload.then_action:
            payload = replace(payload, then_action=action)
    elif isinstance(payload, ElseStatementSyntax) and isinstance(
        payload.action, StatementSyntax
    ):
        action = _reconcile_statement_apply_references(
            payload.action,
            live_apply_occurrences=live_apply_occurrences,
            deferred_expression_references=deferred_expression_references,
            deferred_statement_spans=deferred_statement_spans,
            generated_references=generated_references,
        )
        if action is not payload.action:
            payload = replace(payload, action=action)

    retained = [
        reference
        for reference in statement.apply_references
        if _apply_reference_key(reference)
        not in {_apply_reference_key(generated) for generated in generated_references}
        and _apply_reference_is_live(
            reference,
            live_apply_occurrences=live_apply_occurrences,
            deferred_expression_references=deferred_expression_references,
            deferred_statement_spans=deferred_statement_spans,
        )
    ]
    retained.extend(
        reference
        for reference in generated_references
        if _span_contains(statement.span, reference.call_span)
    )
    retained.sort(key=lambda reference: (reference.call_span.start, reference.call_span.end))
    references = tuple(retained)
    if payload is statement.payload and references == statement.apply_references:
        return statement
    return replace(statement, payload=payload, apply_references=references)


def _apply_reference_is_live(
    reference: ApplyReferenceSyntax,
    *,
    live_apply_occurrences: dict[ApplyOccurrenceKey, _ApplyOccurrence],
    deferred_expression_references: dict[
        ApplyOccurrenceKey, list[ApplyReferenceSyntax]
    ],
    deferred_statement_spans: list[SourceSpan],
) -> bool:
    key = _apply_reference_key(reference)
    occurrence = live_apply_occurrences.get(key)
    if occurrence is not None:
        return occurrence.target is None or occurrence.target == reference.target
    if any(
        deferred_reference.target == reference.target
        for deferred_reference in deferred_expression_references.get(key, [])
    ):
        return True
    return any(
        _span_contains(span, reference.call_span)
        for span in deferred_statement_spans
    )


def _apply_reference_key(reference: ApplyReferenceSyntax) -> ApplyOccurrenceKey:
    span = reference.call_span
    return (span.source_id, span.start, span.end)


def _walk_statements(statements: tuple[StatementSyntax, ...]):
    for statement in statements:
        yield statement
        payload = statement.payload
        if isinstance(payload, IfStatementSyntax) and isinstance(
            payload.then_action, StatementSyntax
        ):
            yield from _walk_statements((payload.then_action,))
        elif isinstance(payload, ElseStatementSyntax) and isinstance(
            payload.action, StatementSyntax
        ):
            yield from _walk_statements((payload.action,))


def _span_contains(outer: SourceSpan, inner: SourceSpan) -> bool:
    return (
        outer.source_id == inner.source_id
        and outer.start <= inner.start
        and inner.end <= outer.end
    )


def _collect_statement_apply_occurrences(
    statement: StatementSyntax,
    live_apply_occurrences: dict[ApplyOccurrenceKey, _ApplyOccurrence],
    deferred_expression_references: dict[
        ApplyOccurrenceKey, list[ApplyReferenceSyntax]
    ],
    deferred_statement_spans: list[SourceSpan],
) -> None:
    payload = statement.payload
    if isinstance(payload, DeferredStatementError):
        deferred_statement_spans.append(payload.span)
    elif isinstance(payload, AssignStatementSyntax):
        if isinstance(payload.target, ArrayElementTargetSyntax):
            _collect_expression_apply_occurrences(
                payload.target.index,
                live_apply_occurrences,
                deferred_expression_references,
            )
        _collect_expression_apply_occurrences(
            payload.value, live_apply_occurrences, deferred_expression_references
        )
    elif isinstance(payload, SumStatementSyntax):
        _collect_expression_apply_occurrences(
            payload.increment, live_apply_occurrences, deferred_expression_references
        )
    elif isinstance(payload, WhereStatementSyntax):
        _collect_expression_apply_occurrences(
            payload.condition, live_apply_occurrences, deferred_expression_references
        )
    elif isinstance(payload, IfStatementSyntax):
        _collect_expression_apply_occurrences(
            payload.condition, live_apply_occurrences, deferred_expression_references
        )
        _collect_action_apply_occurrences(
            payload.then_action,
            live_apply_occurrences,
            deferred_expression_references,
            deferred_statement_spans,
        )
    elif isinstance(payload, ElseStatementSyntax):
        _collect_action_apply_occurrences(
            payload.action,
            live_apply_occurrences,
            deferred_expression_references,
            deferred_statement_spans,
        )
    elif isinstance(payload, DoStatementSyntax):
        if payload.start is not None:
            _collect_expression_apply_occurrences(
                payload.start,
                live_apply_occurrences,
                deferred_expression_references,
            )
        if payload.end is not None:
            _collect_expression_apply_occurrences(
                payload.end,
                live_apply_occurrences,
                deferred_expression_references,
            )


def _collect_action_apply_occurrences(
    action: StatementSyntax | DeferredStatementError | None,
    live_apply_occurrences: dict[ApplyOccurrenceKey, _ApplyOccurrence],
    deferred_expression_references: dict[
        ApplyOccurrenceKey, list[ApplyReferenceSyntax]
    ],
    deferred_statement_spans: list[SourceSpan],
) -> None:
    if isinstance(action, StatementSyntax):
        _collect_statement_apply_occurrences(
            action,
            live_apply_occurrences,
            deferred_expression_references,
            deferred_statement_spans,
        )
    elif isinstance(action, DeferredStatementError):
        deferred_statement_spans.append(action.span)


def _collect_expression_apply_occurrences(
    expression: ExpressionSyntax,
    live_apply_occurrences: dict[ApplyOccurrenceKey, _ApplyOccurrence],
    deferred_expression_references: dict[
        ApplyOccurrenceKey, list[ApplyReferenceSyntax]
    ],
) -> None:
    if isinstance(expression, DeferredExpressionError):
        for reference in expression.apply_references:
            deferred_expression_references.setdefault(
                _apply_reference_key(reference), []
            ).append(reference)
    elif isinstance(expression, CallSyntax):
        occurrence = _typed_apply_occurrence(expression)
        if occurrence is not None:
            key, typed_occurrence = occurrence
            existing = live_apply_occurrences.get(key)
            if (
                existing is not None
                and existing.target is not None
                and typed_occurrence.target is not None
                and existing.target != typed_occurrence.target
            ):
                raise ValueError(
                    "typed apply Calls with one occurrence need distinct spans"
                )
            live_apply_occurrences[key] = typed_occurrence
        for argument in expression.arguments:
            _collect_expression_apply_occurrences(
                argument, live_apply_occurrences, deferred_expression_references
            )
    elif isinstance(expression, UnarySyntax):
        _collect_expression_apply_occurrences(
            expression.operand, live_apply_occurrences, deferred_expression_references
        )
    elif isinstance(expression, BinarySyntax):
        _collect_expression_apply_occurrences(
            expression.left, live_apply_occurrences, deferred_expression_references
        )
        _collect_expression_apply_occurrences(
            expression.right, live_apply_occurrences, deferred_expression_references
        )
    elif isinstance(expression, SubscriptSyntax):
        _collect_expression_apply_occurrences(
            expression.base, live_apply_occurrences, deferred_expression_references
        )
        _collect_expression_apply_occurrences(
            expression.index, live_apply_occurrences, deferred_expression_references
        )
    elif isinstance(expression, ConcatSyntax):
        for part in expression.parts:
            _collect_expression_apply_occurrences(
                part, live_apply_occurrences, deferred_expression_references
            )


def _typed_apply_occurrence(
    expression: CallSyntax,
) -> tuple[ApplyOccurrenceKey, _ApplyOccurrence] | None:
    if expression.name.casefold() != "apply" or not expression.arguments:
        return None
    target = expression.arguments[0]
    if not isinstance(target, LiteralSyntax) or target.kind != "string":
        return None
    if (
        target.span.source_id != expression.span.source_id
        or target.span.start < expression.span.start
        or target.span.end > expression.span.end
    ):
        return None
    decoded_target = _decode_typed_string_literal(target.lexeme)
    if decoded_target is None:
        return None
    key = (expression.span.source_id, expression.span.start, target.span.end)
    return key, _ApplyOccurrence(target=decoded_target)


def _decode_typed_string_literal(lexeme: str) -> str | None:
    """Decode a string token without parsing source or evaluating Python."""

    if len(lexeme) < 2 or lexeme[0] not in ("'", '"') or lexeme[-1] != lexeme[0]:
        return None
    quote = lexeme[0]
    content = lexeme[1:-1]
    decoded: list[str] = []
    index = 0
    simple_escapes = {
        "a": "\a",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
        "v": "\v",
    }
    while index < len(content):
        character = content[index]
        if character == quote and index + 1 < len(content) and content[index + 1] == quote:
            # The initial scanner passes the full span to Python's literal
            # evaluator, where adjacent literals concatenate without a quote.
            index += 2
            continue
        if character != "\\":
            decoded.append(character)
            index += 1
            continue
        index += 1
        if index >= len(content):
            return None
        escaped = content[index]
        if escaped in ("\\", "'", '"'):
            decoded.append(escaped)
            index += 1
            continue
        if escaped in simple_escapes:
            decoded.append(simple_escapes[escaped])
            index += 1
            continue
        if escaped == "N":
            if index + 1 >= len(content) or content[index + 1] != "{":
                return None
            name_end = content.find("}", index + 2)
            if name_end < 0:
                return None
            try:
                decoded.append(unicodedata.lookup(content[index + 2 : name_end]))
            except KeyError:
                return None
            index = name_end + 1
            continue
        if escaped in "xuU":
            width = {"x": 2, "u": 4, "U": 8}[escaped]
            digits = content[index + 1 : index + 1 + width]
            if len(digits) != width or any(
                digit not in "0123456789abcdefABCDEF" for digit in digits
            ):
                return None
            codepoint = int(digits, 16)
            if codepoint > 0x10FFFF:
                return None
            decoded.append(chr(codepoint))
            index += width + 1
            continue
        if escaped in "01234567":
            digits = escaped
            while (
                len(digits) < 3
                and index + len(digits) < len(content)
                and content[index + len(digits)] in "01234567"
            ):
                digits += content[index + len(digits)]
            decoded.append(chr(int(digits, 8)))
            index += len(digits)
            continue
        if escaped == "\n":
            index += 1
            continue
        if escaped == "\r" and index + 1 < len(content) and content[index + 1] == "\n":
            index += 2
            continue
        decoded.extend(("\\", escaped))
        index += 1
    return "".join(decoded)


def _rewrite_statement(
    statement: StatementSyntax,
    source_owned_specs: dict[int, TypedLagLeadSpec],
) -> tuple[StatementSyntax, frozenset[SourceSpan]]:
    payload = statement.payload
    removed_apply_spans: set[SourceSpan] = set()
    payload_changed = False

    if isinstance(payload, AssignStatementSyntax):
        target = payload.target
        if isinstance(target, ArrayElementTargetSyntax):
            index, removed = _rewrite_expression(target.index, source_owned_specs)
            removed_apply_spans.update(removed)
            if index is not target.index:
                target = replace(target, index=index)
        value, removed = _rewrite_expression(payload.value, source_owned_specs)
        removed_apply_spans.update(removed)
        if target is not payload.target or value is not payload.value:
            payload = replace(payload, target=target, value=value)
            payload_changed = True

    elif isinstance(payload, SumStatementSyntax):
        increment, removed = _rewrite_expression(
            payload.increment, source_owned_specs
        )
        removed_apply_spans.update(removed)
        if increment is not payload.increment:
            payload = replace(payload, increment=increment)
            payload_changed = True

    elif isinstance(payload, WhereStatementSyntax):
        condition, removed = _rewrite_expression(
            payload.condition, source_owned_specs
        )
        removed_apply_spans.update(removed)
        if condition is not payload.condition:
            payload = replace(payload, condition=condition)
            payload_changed = True

    elif isinstance(payload, IfStatementSyntax):
        original_condition = payload.condition
        condition, removed = _rewrite_expression(
            payload.condition, source_owned_specs
        )
        removed_apply_spans.update(removed)
        action, removed = _rewrite_action(payload.then_action, source_owned_specs)
        removed_apply_spans.update(removed)
        if condition is not payload.condition or action is not payload.then_action:
            payload = replace(
                payload,
                condition=condition,
                then_action=action,
                condition_display_edits=(
                    () if condition is not original_condition else payload.condition_display_edits
                ),
                condition_display_span=(
                    None
                    if condition is not original_condition
                    else payload.condition_display_span
                ),
            )
            payload_changed = True

    elif isinstance(payload, ElseStatementSyntax):
        action, removed = _rewrite_action(payload.action, source_owned_specs)
        removed_apply_spans.update(removed)
        if action is not payload.action:
            payload = replace(payload, action=action)
            payload_changed = True

    elif isinstance(payload, DoStatementSyntax):
        start = payload.start
        end = payload.end
        if start is not None:
            start, removed = _rewrite_expression(start, source_owned_specs)
            removed_apply_spans.update(removed)
        if end is not None:
            end, removed = _rewrite_expression(end, source_owned_specs)
            removed_apply_spans.update(removed)
        if start is not payload.start or end is not payload.end:
            payload = replace(
                payload,
                start=start,
                end=end,
                boundary_display_spans=None,
            )
            payload_changed = True

    if not payload_changed and not removed_apply_spans:
        return statement, frozenset()

    retained_references = tuple(
        reference
        for reference in statement.apply_references
        if reference.call_span not in removed_apply_spans
    )
    rewritten_statement = replace(
        statement,
        payload=payload,
        apply_references=retained_references,
        legacy_display_span=(
            None if payload_changed else statement.legacy_display_span
        ),
    )
    return rewritten_statement, frozenset(removed_apply_spans)


def _rewrite_action(
    action: StatementSyntax | DeferredStatementError | None,
    source_owned_specs: dict[int, TypedLagLeadSpec],
) -> tuple[
    StatementSyntax | DeferredStatementError | None,
    frozenset[SourceSpan],
]:
    if not isinstance(action, StatementSyntax):
        return action, frozenset()
    return _rewrite_statement(action, source_owned_specs)


def _rewrite_expression(
    expression: ExpressionSyntax,
    source_owned_specs: dict[int, TypedLagLeadSpec],
) -> tuple[ExpressionSyntax, frozenset[SourceSpan]]:
    if isinstance(expression, DeferredExpressionError):
        return expression, frozenset()

    if isinstance(expression, CallSyntax):
        spec = source_owned_specs.get(id(expression))
        if spec is not None and spec.call is expression:
            replacement = NameSyntax(
                name=spec.helper_column,
                span=expression.span,
                origin=expression.origin or expression.span,
            )
            return replacement, frozenset(_apply_call_spans(expression))
        arguments: list[ExpressionSyntax] = []
        removed_apply_spans: set[SourceSpan] = set()
        for argument in expression.arguments:
            rewritten, removed = _rewrite_expression(argument, source_owned_specs)
            arguments.append(rewritten)
            removed_apply_spans.update(removed)
        if all(
            original is rewritten
            for original, rewritten in zip(expression.arguments, arguments)
        ):
            return expression, frozenset(removed_apply_spans)
        return (
            replace(expression, arguments=tuple(arguments)),
            frozenset(removed_apply_spans),
        )

    if isinstance(expression, UnarySyntax):
        operand, removed = _rewrite_expression(expression.operand, source_owned_specs)
        return (
            expression if operand is expression.operand else replace(expression, operand=operand),
            removed,
        )

    if isinstance(expression, BinarySyntax):
        left, removed_left = _rewrite_expression(expression.left, source_owned_specs)
        right, removed_right = _rewrite_expression(expression.right, source_owned_specs)
        if left is expression.left and right is expression.right:
            return expression, removed_left | removed_right
        return (
            replace(expression, left=left, right=right),
            removed_left | removed_right,
        )

    if isinstance(expression, SubscriptSyntax):
        base, removed_base = _rewrite_expression(expression.base, source_owned_specs)
        index, removed_index = _rewrite_expression(expression.index, source_owned_specs)
        if base is expression.base and index is expression.index:
            return expression, removed_base | removed_index
        return (
            replace(expression, base=base, index=index),
            removed_base | removed_index,
        )

    if isinstance(expression, ConcatSyntax):
        parts: list[ExpressionSyntax] = []
        removed_apply_spans: set[SourceSpan] = set()
        for part in expression.parts:
            rewritten, removed = _rewrite_expression(part, source_owned_specs)
            parts.append(rewritten)
            removed_apply_spans.update(removed)
        if all(
            original is rewritten
            for original, rewritten in zip(expression.parts, parts)
        ):
            return expression, frozenset(removed_apply_spans)
        return (
            replace(expression, parts=tuple(parts)),
            frozenset(removed_apply_spans),
        )

    # LiteralSyntax and NameSyntax have no children; the expression union is
    # deliberately closed, so an unknown node is a programming error.
    if isinstance(expression, (LiteralSyntax, NameSyntax)):
        return expression, frozenset()
    raise TypeError(f"unsupported expression node: {type(expression).__name__}")


def _apply_call_spans(expression: ExpressionSyntax) -> set[SourceSpan]:
    """Collect typed apply calls removed with a replaced subtree.

    Deferred errors deliberately expose no partial expression tree; their
    saved apply inventory is retained on the owning typed statement.
    """

    if isinstance(expression, DeferredExpressionError):
        return set()
    if isinstance(expression, CallSyntax):
        spans = {expression.span} if expression.name.casefold() == "apply" else set()
        for argument in expression.arguments:
            spans.update(_apply_call_spans(argument))
        return spans
    if isinstance(expression, UnarySyntax):
        return _apply_call_spans(expression.operand)
    if isinstance(expression, BinarySyntax):
        return _apply_call_spans(expression.left) | _apply_call_spans(expression.right)
    if isinstance(expression, SubscriptSyntax):
        return _apply_call_spans(expression.base) | _apply_call_spans(expression.index)
    if isinstance(expression, ConcatSyntax):
        spans: set[SourceSpan] = set()
        for part in expression.parts:
            spans.update(_apply_call_spans(part))
        return spans
    return set()


__all__ = ["TypedSyntaxRewriteResult", "rewrite_typed_block"]
