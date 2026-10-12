"""Focused tests for the IF / ELSE condition frontend entry."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from limulus.syntax.models import (
    ApplyReferenceSyntax,
    AssignStatementSyntax,
    ArrayElementTargetSyntax,
    BinarySyntax,
    DatasetOptionSyntax,
    DatasetReferenceSyntax,
    DeferredExpressionError,
    DeferredStatementError,
    DeleteStatementSyntax,
    DoStatementSyntax,
    ElseStatementSyntax,
    IfStatementSyntax,
    LiteralSyntax,
    NameSyntax,
    NameTargetSyntax,
    OutputStatementSyntax,
    SourceDocument,
    StopStatementSyntax,
    StatementSyntax,
    SumStatementSyntax,
)
from limulus.syntax.program import LarkProgramFrontend

REPO_ROOT = Path(__file__).resolve().parents[1]

_DENY_HOOK = textwrap.dedent(
    """
    import sys

    _DENIED = ("pyarrow", "polars", "limulus_native", "limulus.session")


    class _DeniedDependencyGate:
        def find_spec(self, fullname, path=None, target=None):
            for root in _DENIED:
                if fullname == root or fullname.startswith(root + "."):
                    raise ImportError(
                        f"module {fullname!r} is denied by the conditional frontend isolation hook"
                    )
            return None


    sys.meta_path.insert(0, _DeniedDependencyGate())
    """
)


def _run_isolated(snippet: str) -> None:
    script = _DENY_HOOK + "\n" + textwrap.dedent(snippet)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"isolated import failed:\n{result.stdout}\n{result.stderr}"


def _program(text: str, document_id: str = "conditionals"):
    frontend = LarkProgramFrontend()
    document = SourceDocument(text, document_id)
    regions = frontend.extract_program_regions(document)
    assert regions is not None, f"region identification failed for {text!r}"
    return document, frontend, regions


def _region(document, regions, needle: str):
    """The one statement region whose raw (trimmed) text contains ``needle``."""

    matches = [
        region
        for region in regions.statement_regions
        if needle.lower() in document.text[region.span.start : region.span.end].lower()
    ]
    assert len(matches) == 1, f"needle {needle!r} matched {len(matches)} regions in {document.text!r}"
    return matches[0]


def _parse_if_with_entry(text: str, needle: str) -> StatementSyntax:
    document, frontend, regions = _program(text)
    return frontend.parse_if_statement(document, _region(document, regions, needle))


def _parse_else_with_entry(text: str, needle: str) -> StatementSyntax:
    document, frontend, regions = _program(text)
    return frontend.parse_else_statement(document, _region(document, regions, needle))


# --- subset IF: typed condition, no action --------------------------------------


def test_subset_if_condition_is_typed_with_exact_span() -> None:
    text = "data out; if amount < 0; run;"
    statement = _parse_if_with_entry(text, "if amount")
    payload = statement.payload
    assert isinstance(payload, IfStatementSyntax)
    assert payload.is_subset is True
    assert payload.is_then_do is False
    assert payload.then_action is None
    condition = payload.condition
    assert isinstance(condition, BinarySyntax)
    assert condition.operator == "<"
    assert isinstance(condition.left, NameSyntax)
    assert condition.left.name == "amount"
    assert text[condition.left.span.start : condition.left.span.end] == "amount"
    assert isinstance(condition.right, LiteralSyntax)
    assert condition.right.lexeme == "0"
    assert text[condition.right.span.start : condition.right.span.end] == "0"


def test_unicode_condition_offsets_match_old_facade_truth() -> None:
    # Old facade truth: a non-ASCII condition token never reached the
    # if_structured split (spec None) and stayed lazily invalid; the typed
    # path keeps exactly that failure shape, lazily and with raw offsets.
    text = "data out; if 対象=1; run;"
    statement = _parse_if_with_entry(text, "if 対象")
    payload = statement.payload
    assert isinstance(payload, DeferredStatementError)
    assert "invalid IF statement" in payload.native_cause
    assert payload.display_text == "if 対象=1"
    # A non-ASCII quoted literal stays within the legacy split and is typed.
    text = "data out; if x = '日本語' then y = 2; run;"
    statement = _parse_if_with_entry(text, "日本語")
    payload = statement.payload
    assert isinstance(payload, IfStatementSyntax)
    condition = payload.condition
    assert isinstance(condition, BinarySyntax)
    assert isinstance(condition.right, LiteralSyntax)
    assert condition.right.lexeme == "'日本語'"
    assert text[condition.right.span.start : condition.right.span.end] == "'日本語'"


# --- open-quote lazy families ------------------------------------------------------


def test_open_quote_if_statement_keeps_whole_fault_family() -> None:
    from limulus.syntax.models import DeferredStatementError
    from limulus.syntax.program import parse_program

    raw = "data out; if 0 then x = 'abc; run;"
    result = parse_program(raw, source_id="<conditionals>")
    assert result.program is not None
    assert result.diagnostics == ()
    if_statement = result.program.blocks[0].statements[1]
    assert if_statement.kind == "IF"
    payload = if_statement.payload
    assert isinstance(payload, DeferredStatementError)
    assert payload.native_cause == "invalid IF statement: if 0 then x = 'abc"
    assert (
        raw[if_statement.span.start : if_statement.span.end] == "if 0 then x = 'abc"
    )


# --- IF THEN: structured inline actions at exact spans ---------------------------


def test_if_then_assign_action_typed_with_exact_spans() -> None:
    text = "data out; if amount < 0 then a = 1; run;"
    statement = _parse_if_with_entry(text, "then a = 1")
    payload = statement.payload
    assert isinstance(payload, IfStatementSyntax)
    assert payload.is_subset is False
    assert payload.is_then_do is False
    action = payload.then_action
    assert isinstance(action, StatementSyntax)
    assert action.kind == "ASSIGN"
    assign = action.payload
    assert isinstance(assign, AssignStatementSyntax)
    assert isinstance(assign.target, NameTargetSyntax)
    assert assign.target.name == "a"
    assert text[assign.target.span.start : assign.target.span.end] == "a"
    assert isinstance(assign.value, LiteralSyntax)
    assert text[assign.value.span.start : assign.value.span.end] == "1"
    condition = payload.condition
    assert isinstance(condition, BinarySyntax)
    assert condition.operator == "<"
    assert statement.span.source_id == "conditionals"

def test_if_then_array_target_action_is_structured() -> None:
    text = "data out; if p then x(i) = v; run;"
    statement = _parse_if_with_entry(text, "x(i)")
    payload = statement.payload
    assert isinstance(payload, IfStatementSyntax)
    action = payload.then_action
    assert isinstance(action, StatementSyntax)
    assert action.kind == "ASSIGN"
    assign = action.payload
    assert isinstance(assign, AssignStatementSyntax)
    target = assign.target
    assert isinstance(target, ArrayElementTargetSyntax)
    assert target.name == "x"
    assert target.delimiter == "("
    assert isinstance(target.index, NameSyntax)
    assert target.index.name == "i"


def test_if_then_sum_action_is_structured() -> None:
    text = "data out; if p then seq + 1; run;"
    statement = _parse_if_with_entry(text, "seq +")
    payload = statement.payload
    assert isinstance(payload, IfStatementSyntax)
    action = payload.then_action
    assert isinstance(action, StatementSyntax)
    assert action.kind == "SUM"
    sum_payload = action.payload
    assert isinstance(sum_payload, SumStatementSyntax)
    assert isinstance(sum_payload.target, NameTargetSyntax)
    assert sum_payload.target.name == "seq"
    assert text[sum_payload.target.span.start : sum_payload.target.span.end] == "seq"
    assert isinstance(sum_payload.increment, LiteralSyntax)
    assert sum_payload.increment.lexeme == "1"


def test_if_then_stop_and_delete_actions_are_simple_payloads() -> None:
    for action_text, payload_type in (("stop", StopStatementSyntax), ("delete", DeleteStatementSyntax)):
        text = f"data out; if p then {action_text}; run;"
        statement = _parse_if_with_entry(text, f"then {action_text}")
        payload = statement.payload
        assert isinstance(payload, IfStatementSyntax)
        assert isinstance(payload.condition, NameSyntax)
        action = payload.then_action
        assert isinstance(action, StatementSyntax)
        assert isinstance(action.payload, payload_type)
        # A bare inline action keeps its exact trimmed span.
        assert text[action.span.start : action.span.end] == action_text


# --- THEN DO: flags with a bare DO payload, flat order untouched -----------------


def test_inline_sum_shape_follows_the_native_inline_executor() -> None:
    # Old native inline truth (execute_inline_action): the SUM shape runs
    # BEFORE the assignment shape and its parse has NO bare-name gate, so
    # `x = a + b` inside THEN was a SUM on the opaque name `x = a` — a real
    # legacy execution quirk that this structure preserves verbatim instead
    # of "improving" it into an assignment.
    text = "data out; if c then x = a + b; run;"
    statement = _parse_if_with_entry(text, "then x")
    payload = statement.payload
    assert isinstance(payload, IfStatementSyntax)
    action = payload.then_action
    assert isinstance(action, StatementSyntax)
    assert action.kind == "SUM"
    sum_payload = action.payload
    assert isinstance(sum_payload, SumStatementSyntax)
    assert sum_payload.target.name == "x = a"
    assert text[sum_payload.target.span.start : sum_payload.target.span.end] == "x = a"
    assert isinstance(sum_payload.increment, NameSyntax)
    assert sum_payload.increment.name == "b"


def test_then_do_marks_flag_with_bare_do_and_flat_order() -> None:
    text = "data out; if c then do; x = 1; end; run;"
    document, frontend, regions = _program(text)
    assert [text[r.span.start : r.span.end] for r in regions.statement_regions] == [
        "data out",
        "if c then do",
        "x = 1",
        "end",
        "run",
    ]
    statement = frontend.parse_if_statement(
        document, _region(document, regions, "if c")
    )
    payload = statement.payload
    assert isinstance(payload, IfStatementSyntax)
    assert payload.is_then_do is True
    assert payload.is_subset is False
    action = payload.then_action
    assert isinstance(action, StatementSyntax)
    assert action.kind == "DO"
    bare = action.payload
    assert isinstance(bare, DoStatementSyntax)
    assert bare.loop_var is None and bare.start is None and bare.end is None
    # The bare inline DO keeps its exact "do" span; the body stays flat.
    assert text[action.span.start : action.span.end] == "do"


def test_then_do_with_extra_tokens_stays_bare_do_flag() -> None:
    text = "data out; if c then do while(a); x = 1; end; run;"
    statement = _parse_if_with_entry(text, "if c then do")
    payload = statement.payload
    assert isinstance(payload, IfStatementSyntax)
    assert payload.is_then_do is True
    action = payload.then_action
    assert isinstance(action, StatementSyntax)
    assert isinstance(action.payload, DoStatementSyntax)
    # No loop bounds are parsed here (the DO boundary entry owns bounds).
    assert action.payload.start is None    # The action span still covers the full inline do... slice.
    assert text[action.span.start : action.span.end] == "do while(a)"


# --- ELSE IF / ELSE --------------------------------------------------------------


def test_else_if_condition_is_parsed_with_kind_else_if() -> None:
    text = "data out; if a then x=1; else if b then y=2; run;"
    statement = _parse_if_with_entry(text, "else if")
    assert statement.kind == "ELSE IF"
    payload = statement.payload
    assert isinstance(payload, IfStatementSyntax)
    assert isinstance(payload.condition, NameSyntax)
    assert payload.condition.name == "b"
    action = payload.then_action
    assert isinstance(action, StatementSyntax)
    assert action.kind == "ASSIGN"
    assert isinstance(action.payload, AssignStatementSyntax)
    assert action.payload.target.name == "y"


def test_else_action_is_structured() -> None:
    text = "data out; if a then x=1; else b=2; run;"
    statement = _parse_else_with_entry(text, "else b")
    assert statement.kind == "ELSE"
    payload = statement.payload
    assert isinstance(payload, ElseStatementSyntax)
    action = payload.action
    assert isinstance(action, StatementSyntax)
    assert action.kind == "ASSIGN"
    assert isinstance(action.payload, AssignStatementSyntax)
    assert action.payload.target.name == "b"


def test_else_do_action_is_a_bare_do_statement() -> None:
    text = "data out; if a then x=1; else do; c=3; end; run;"
    statement = _parse_else_with_entry(text, "else do")
    assert statement.kind == "ELSE"
    payload = statement.payload
    assert isinstance(payload, ElseStatementSyntax)
    action = payload.action
    assert action is not None
    assert isinstance(action, StatementSyntax)
    assert action.kind == "DO"
    assert isinstance(action.payload, DoStatementSyntax)
    assert action.payload.loop_var is None


def test_bare_else_keeps_no_action() -> None:
    text = "data out; if a then b=1; else; run;"
    statement = _parse_else_with_entry(text, "else")
    assert statement.kind == "ELSE"
    payload = statement.payload
    assert isinstance(payload, ElseStatementSyntax)
    assert payload.action is None


def test_else_without_then_fails_lazily_on_the_inner_action() -> None:
    # ``else if x;`` keeps the old ELSE kind; the inline body "if x" is not a
    # supported action, so it stays a deferred descriptor INSIDE the ELSE
    # payload (the ELSE has no condition to keep) with the runtime family.
    text = "data out; if a then x=1; else if x; run;"
    statement = _parse_else_with_entry(text, "else if")
    assert statement.kind == "ELSE"
    payload = statement.payload
    assert isinstance(payload, ElseStatementSyntax)
    action = payload.action
    assert isinstance(action, DeferredStatementError)
    assert "unsupported inline action: if x" in action.native_cause
    assert text[action.span.start : action.span.end] == "if x"


def test_else_glued_prefix_rejected_like_the_old_facade() -> None:
    # The old facade classified glued ``elseX`` prefixes as assignments /
    # sums, not ELSE statements; these restricted entries must not claim them.
    document, frontend, regions = _program("data out; elsefoo x = 1; run;")
    with pytest.raises(ValueError, match="else"):
        frontend.parse_else_statement(document, regions.statement_regions[1])


def test_glued_elseif_prefix_is_not_an_if_entry() -> None:
    document, frontend, regions = _program("data out; elseif x then y=1; run;")
    with pytest.raises(ValueError, match="if"):
        frontend.parse_if_statement(document, regions.statement_regions[1])


# --- inline OUTPUT references and options (8.x parity) ----------------------------


def test_if_then_output_action_with_options_is_reference_typed() -> None:
    text = "data out; if 0 then output out(keep=id); run;"
    statement = _parse_if_with_entry(text, "then output")
    payload = statement.payload
    assert isinstance(payload, IfStatementSyntax)
    action = payload.then_action
    assert isinstance(action, StatementSyntax)
    assert action.kind == "OUTPUT"
    assert isinstance(action.payload, OutputStatementSyntax)
    targets = action.payload.targets
    assert len(targets) == 1
    target = targets[0]
    assert isinstance(target, DatasetReferenceSyntax)
    assert target.name == "out"
    assert isinstance(target.options, DatasetOptionSyntax)
    assert target.options.keep_vars == ("id",)
    # The reference span covers the exact original slice including the option
    # group; the inline declaration is kept for the later target policy
    # even though the branch is never reached.
    assert text[target.span.start : target.span.end] == "out(keep=id)"


def test_if_then_output_multiple_targets_in_source_order() -> None:
    text = "data out; if f then output a b; run;"
    statement = _parse_if_with_entry(text, "then output")
    payload = statement.payload
    assert isinstance(payload, IfStatementSyntax)
    action = payload.then_action
    assert isinstance(action, StatementSyntax)
    assert isinstance(action.payload, OutputStatementSyntax)
    targets = action.payload.targets
    assert [t.name for t in targets] == ["a", "b"]
    assert text[targets[0].span.start : targets[0].span.end] == "a"
    assert text[targets[1].span.start : targets[1].span.end] == "b"


def test_bare_inline_output_keeps_empty_reference_list() -> None:
    text = "data out; if f then output; run;"
    statement = _parse_if_with_entry(text, "then output")
    payload = statement.payload
    assert isinstance(payload, IfStatementSyntax)
    action = payload.then_action
    assert isinstance(action, StatementSyntax)
    assert isinstance(action.payload, OutputStatementSyntax)
    assert action.payload.targets == ()


def test_real_output_action_differs_from_quoted_literal_condition() -> None:
    # The suppression decision reads the OUTPUT structure only; an
    # " output" literal inside the condition is a string literal with no
    # suppression meaning, and no substring-derived boolean field exists on
    # any syntax payload of the new typed path.
    text = "data out; if msg = ' output' then y=1; run;"
    statement = _parse_if_with_entry(text, "if msg")
    payload = statement.payload
    assert isinstance(payload, IfStatementSyntax)
    condition = payload.condition
    assert isinstance(condition, BinarySyntax)
    assert isinstance(condition.right, LiteralSyntax)
    assert condition.right.lexeme == "' output'"
    assert payload.then_action is not None
    # And a real inline OUTPUT is a typed reference statement, structurally
    # distinct from the quoted literal condition.
    text = "data out; if f then output out; run;"
    statement = _parse_if_with_entry(text, "then output")
    reanalyzed = statement.payload
    assert isinstance(reanalyzed, IfStatementSyntax)
    action = reanalyzed.then_action
    assert isinstance(action, StatementSyntax)
    assert action.kind == "OUTPUT"
    assert isinstance(action.payload, OutputStatementSyntax)


def test_glued_inline_output_prefix_routes_as_a_named_ref() -> None:
    # Old inline truth: ``outputfoo`` is an OUTPUT to the dataset "outputfoo".
    text = "data out; if p then outputfoo; run;"
    statement = _parse_if_with_entry(text, "outputfoo")
    payload = statement.payload
    assert isinstance(payload, IfStatementSyntax)
    action = payload.then_action
    assert isinstance(action, StatementSyntax)
    assert action.kind == "OUTPUT"
    assert isinstance(action.payload, OutputStatementSyntax)
    assert [t.name for t in action.payload.targets] == ["outputfoo"]


# --- lazy faults: unsupported inline -> INNER descriptor; invalid split -> outer -


def test_unsupported_inline_action_descriptor_keeps_condition() -> None:
    text = "data out; if p then foo; run;"
    statement = _parse_if_with_entry(text, "then foo")
    payload = statement.payload
    # The condition and branch structure survive; the unknown action is held
    # as the DeferredStatementError descriptor INSIDE the IF payload.
    assert isinstance(payload, IfStatementSyntax)
    assert isinstance(payload.condition, NameSyntax)
    assert payload.condition.name == "p"
    assert payload.is_subset is False
    assert payload.is_then_do is False
    action = payload.then_action
    assert isinstance(action, DeferredStatementError)
    assert action.error_kind == "parse_fault"
    assert "unsupported inline action: foo" in action.native_cause
    # The descriptor span/display address the ACTION's own raw trimmed slice.
    assert text[action.span.start : action.span.end] == "foo"
    assert action.display_text == "foo"
    # Not a valid SKIPPED (or any wrapper statement kind).
    assert not isinstance(action, StatementSyntax)
    # The outer statement keeps the whole IF span; never program-stopping.
    assert statement.kind == "IF"
    assert statement.span != action.span
    assert text[statement.span.start : statement.span.end] == "if p then foo"


def test_nested_fault_hierarchy_split_failure_vs_action_descriptor() -> None:
    # ONLY a failed condition/action split replaces the whole statement.
    text = "data out; if c then; run;"
    statement = _parse_if_with_entry(text, "if c")
    payload = statement.payload
    assert isinstance(payload, DeferredStatementError)
    assert "invalid IF statement" in payload.native_cause
    assert payload.span == statement.span


def test_invalid_condition_is_whole_expression_deferred() -> None:
    text = "data out; if x = then y=1; run;"
    statement = _parse_if_with_entry(text, "if x =")
    payload = statement.payload
    assert isinstance(payload, IfStatementSyntax)
    assert payload.is_subset is False
    condition = payload.condition
    assert isinstance(condition, DeferredExpressionError)
    assert condition.error_kind == "parse_fault"
    # The deferred descriptor keeps the whole condition span; no compile or
    # evaluation happened.
    assert text[condition.span.start : condition.span.end] == "x ="


def test_sum_shape_fault_is_an_inner_action_descriptor() -> None:
    text = "data out; if p then x +; run;"
    statement = _parse_if_with_entry(text, "then x +")
    payload = statement.payload
    assert isinstance(payload, IfStatementSyntax)
    assert isinstance(payload.condition, NameSyntax)
    action = payload.then_action
    assert isinstance(action, DeferredStatementError)
    assert "unsupported inline action: x +" in action.native_cause
    assert text[action.span.start : action.span.end] == "x +"
    assert action.display_text == "x +"


def test_if_zero_or_then_condition_values_keep_structure() -> None:
    for literal in ("0", "1"):
        text = f"data out; if {literal} then foo; run;"
        statement = _parse_if_with_entry(text, "then foo")
        payload = statement.payload
        assert isinstance(payload, IfStatementSyntax)
        condition = payload.condition
        assert isinstance(condition, LiteralSyntax)
        assert condition.lexeme == literal
        action = payload.then_action
        assert isinstance(action, DeferredStatementError)
        assert "unsupported inline action: foo" in action.native_cause
        # Structure only: the parser neither evaluated nor executed `foo`,
        # and no reach-time condition decision exists on any payload.
        assert payload.is_subset is False


def test_else_if_unknown_action_is_an_inner_descriptor() -> None:
    text = "data out; if a then x=1; else if b then foo; run;"
    statement = _parse_if_with_entry(text, "else if")
    assert statement.kind == "ELSE IF"
    payload = statement.payload
    assert isinstance(payload, IfStatementSyntax)
    assert isinstance(payload.condition, NameSyntax)
    assert payload.condition.name == "b"
    action = payload.then_action
    assert isinstance(action, DeferredStatementError)
    assert "unsupported inline action: foo" in action.native_cause
    assert text[action.span.start : action.span.end] == "foo"


def test_invalid_condition_and_invalid_action_coexist() -> None:
    # Both faults are kept: the condition descriptor records its whole slice
    # and the action fault is held on the action itself so later owners can
    # fire the condition fault first; neither replaces the other.
    text = "data out; if x = then foo; run;"
    statement = _parse_if_with_entry(text, "if x =")
    payload = statement.payload
    assert isinstance(payload, IfStatementSyntax)
    condition = payload.condition
    assert isinstance(condition, DeferredExpressionError)
    assert text[condition.span.start : condition.span.end] == "x ="
    action = payload.then_action
    assert isinstance(action, DeferredStatementError)
    assert "unsupported inline action: foo" in action.native_cause
    assert text[action.span.start : action.span.end] == "foo"


def test_unicode_action_descriptor_keeps_raw_display_and_offsets() -> None:
    # A quoted unicode literal action splits (STRING token) but matches no
    # supported inline shape; the descriptor keeps the raw trimmed slice.
    text = "data out; if c then '日本語メモ'; run;"
    statement = _parse_if_with_entry(text, "日本語")
    payload = statement.payload
    assert isinstance(payload, IfStatementSyntax)
    assert isinstance(payload.condition, NameSyntax)
    action = payload.then_action
    assert isinstance(action, DeferredStatementError)
    assert action.display_text == "'日本語メモ'"
    assert text[action.span.start : action.span.end] == "'日本語メモ'"


def test_tab_action_descriptor_span_addresses_the_raw_document() -> None:
    text = "data out;\tif c then\tfoo\t; run;"
    document, frontend, regions = _program(text)
    statement = frontend.parse_if_statement(document, _region(document, regions, "then"))
    payload = statement.payload
    assert isinstance(payload, IfStatementSyntax)
    action = payload.then_action
    assert isinstance(action, DeferredStatementError)
    # The descriptor slice is trimmed out of the RAW text (tabs preserved).
    assert text[action.span.start : action.span.end] == "foo"
    assert action.display_text == "foo"


# --- apply inventory: whole statement slice, source order, no duplicates ---------


def test_apply_inventory_covers_condition_and_untaken_action_once() -> None:
    text = 'data out; if c then x = apply("f", a) + apply("g", b); run;'
    statement = _parse_if_with_entry(text, "apply")
    references = statement.apply_references
    assert [r.target for r in references] == ["f", "g"]
    for reference in references:
        assert isinstance(reference, ApplyReferenceSyntax)
        assert reference.owner_span == statement.span
        assert text[reference.call_span.start : reference.call_span.end].startswith("apply")


def test_apply_inventory_inside_quoted_condition_is_excluded() -> None:
    text = "data out; if msg = 'apply(\"fake\")' then y=1; run;"
    statement = _parse_if_with_entry(text, "if msg")
    assert statement.apply_references == ()


def test_faulted_action_statement_keeps_whole_statement_apply_inventory() -> None:
    text = 'data out; if c then foo apply("h", a); run;'
    statement = _parse_if_with_entry(text, "apply")
    payload = statement.payload
    # The unknown action descriptor is inner; the WHOLE statement slice scan
    # (including the untaken faulted action) stays on the outer statement.
    assert isinstance(payload, IfStatementSyntax)
    assert isinstance(payload.then_action, DeferredStatementError)
    assert [r.target for r in statement.apply_references] == ["h"]


def test_inline_output_scan_fault_keeps_the_output_kind() -> None:
    # Per the deferred-action contract: an OUTPUT action was already
    # RECOGNIZED as a known statement kind, so a references/scan failure
    # keeps the typed OUTPUT statement (no filler kind, no bare descriptor
    # fall-through, no fabricated or partial target adoption) whose payload
    # is the DeferredStatementError descriptor carrying the action's own
    # raw display slice and cause.  This also lets the output planning owner's election
    # suppression off the real OUTPUT structure without display rescanning.
    for document_id, text, entry, needle, statement_kind in (
        ("conditionals", "data out; if c then output 1; run;", "if", "output 1", "IF"),
        ("conditionals_b", "data out; if 0 then output 1; run;", "if", "output 1", "IF"),
        ("conditionals_c", "data out; if 1 then output 1; run;", "if", "output 1", "IF"),
        ("conditionals_d", "data out; if a then x=1; else if b then output 2; run;", "if", "output 2", "ELSE IF"),
        ("conditionals_e", "data out; if a then x=1; else output 2; run;", "else", "output 2", "ELSE"),
    ):
        statement = _statement_dispatch(document_id, text, entry, needle)
        assert statement.kind == statement_kind, text
        payload = statement.payload
        if isinstance(payload, IfStatementSyntax):
            action = payload.then_action
        else:
            assert isinstance(payload, ElseStatementSyntax)
            action = payload.action
        # The recognized kind survives as a typed statement.
        assert isinstance(action, StatementSyntax), text
        assert action.kind == "OUTPUT", text
        # ...with the fault as the statement's own payload.
        descriptor = action.payload
        assert isinstance(descriptor, DeferredStatementError), text
        assert "Unsupported statement syntax" in descriptor.native_cause, text
        # The payload descriptor addresses the ACTION's own raw slice; the
        # typed OUTPUT statement span equals that exact action span.
        assert text[action.span.start : action.span.end] == "output 1" or (
            text[action.span.start : action.span.end] == "output 2"
        ), text
        assert descriptor.span == action.span, text
        assert descriptor.display_text == text[action.span.start : action.span.end], text
        # Contrast: the outer whole-slice inventory is still the single normal
        # lexical scan of the branch statement (no apply refs in these texts).
        assert statement.apply_references == (), text
        # No fabricated or partially filled-in target exists anywhere.
        for ref in getattr(descriptor, "apply_references", ()):  # pragma: no cover
            assert not ref, text


# --- dispatch helper for the parameterized OUTPUT-kind protection ----------------


def _statement_dispatch(document_id: str, text: str, entry: str, needle: str) -> StatementSyntax:
    document, frontend, regions = _program(text, document_id)
    region = _region(document, regions, needle)
    if entry == "else":
        return frontend.parse_else_statement(document, region)
    return frontend.parse_if_statement(document, region)


# --- restricted-entry reject and tab normalization ---------------------------------


def test_non_if_statement_is_rejected_by_the_limited_entries() -> None:
    text = "data out; set inp; run;"
    document, frontend, regions = _program(text)
    with pytest.raises(ValueError, match="if"):
        frontend.parse_if_statement(document, regions.statement_regions[1])
    with pytest.raises(ValueError, match="else"):
        frontend.parse_else_statement(document, regions.statement_regions[1])


def test_tab_separators_split_like_spaces_with_raw_offsets() -> None:
    text = "data out; if\tx = 1\tthen\ta = 2; run;"
    document, frontend, regions = _program(text)
    region = _region(document, regions, "then")
    statement = frontend.parse_if_statement(document, region)
    payload = statement.payload
    assert isinstance(payload, IfStatementSyntax)
    condition = payload.condition
    assert isinstance(condition, BinarySyntax)
    assert condition.operator == "=="
    # Spans always address the raw request text.
    assert text[statement.span.start : statement.span.end] == "if\tx = 1\tthen\ta = 2"
    action = payload.then_action
    assert isinstance(action, StatementSyntax)
    assert isinstance(action.payload, AssignStatementSyntax)
    assert action.payload.target.name == "a"
    assert statement.span.source_id == document.source_id


# --- isolation and data-library-independence --------------------------------------


def test_conditionals_entry_imports_without_data_libraries() -> None:
    _run_isolated(
        """
        from limulus.syntax.models import (
            AssignStatementSyntax,
            IfStatementSyntax,
            SourceDocument,
        )
        from limulus.syntax.program import LarkProgramFrontend

        frontend = LarkProgramFrontend()
        document = SourceDocument("data out; if c then x = 1; run;", "isolated")
        regions = frontend.extract_program_regions(document)
        assert regions is not None
        statement = frontend.parse_if_statement(document, regions.statement_regions[1])
        payload = statement.payload
        assert isinstance(payload, IfStatementSyntax), payload
        assert isinstance(payload.then_action.payload, AssignStatementSyntax), payload
        """
    )


# --- old public path parity (original contracts only) ------------------------------


def test_legacy_public_parse_of_if_statements_is_untouched() -> None:
    """The legacy facade keeps the original public IF contract."""

    from limulus.parser import LarkParserFrontend

    expected_kinds = (
        ("data out; if amount < 0 then a=1; run;", ("DATA", "IF", "RUN")),
        ("data out; if c then do; x=1; end; run;", ("DATA", "IF", "ASSIGN", "END", "RUN")),
        ("data out; if c; run;", ("DATA", "IF", "RUN")),
        (
            "data out; if a then x=1; else if b then y=2; run;",
            ("DATA", "IF", "ELSE IF", "RUN"),
        ),
        ("data out; if a then x=1; else c=3; run;", ("DATA", "IF", "ELSE", "RUN")),
    )
    frontend = LarkParserFrontend()
    for text, kinds in expected_kinds:
        result = frontend.parse(text)
        assert result.has_errors is False, text
        assert tuple(s.kind for s in result.ast.statements) == kinds, text


# --- initial IF condition display edits ---------------------------------------------


def test_initial_if_parse_stores_symbol_and_bool_word_display_edits() -> None:
    """The initial if_structured split stores the display edits for valid
    and deferred conditions alike: symbol equality aliases and logic words
    only, ordered, non-overlapping, raw-anchored."""

    text = "data out; if x = 1 AND y ^= 2 then foo; run;"
    statement = _parse_if_with_entry(text, "if x")
    payload = statement.payload
    assert isinstance(payload, IfStatementSyntax)
    edits = payload.condition_display_edits
    assert [text[span.start : span.end] for span, _ in edits] == ["=", "AND", "^="]
    assert [replacement for _, replacement in edits] == ["==", "and", "!="]
    # Ordered ascending and non-overlapping inside the condition bounds.
    assert edits[0][0].end <= edits[1][0].start
    assert edits[1][0].end <= edits[2][0].start
    assert edits[0][0].source_id == edits[1][0].source_id == statement.span.source_id

    # The condition lives between the ``if`` keyword and ``then``; every
    # edit lies inside that raw window (the typed condition node span is an
    # operator-token span and is deliberately not used as the bound).
    assert isinstance(payload.condition, BinarySyntax)
    condition_window_start = text.index("if") + len("if")
    condition_window_end = text.index("then")
    assert edits[0][0].start >= condition_window_start
    assert edits[-1][0].end <= condition_window_end


def test_initial_if_deferred_condition_still_stores_display_edits() -> None:
    text = "data out; if [x=1] then foo; run;"
    statement = _parse_if_with_entry(text, "if [x")
    payload = statement.payload
    assert isinstance(payload, IfStatementSyntax)
    edits = payload.condition_display_edits
    assert [text[span.start : span.end] for span, _ in edits] == ["="]
    assert [replacement for _, replacement in edits] == ["=="]


def test_initial_if_subset_condition_stores_display_edits() -> None:
    text = "data out; if x = 1 and y = 2; run;"
    statement = _parse_if_with_entry(text, "if x")
    payload = statement.payload
    assert isinstance(payload, IfStatementSyntax)
    assert payload.is_subset is True
    edits = payload.condition_display_edits
    assert [text[span.start : span.end] for span, _ in edits] == ["=", "="]
    assert [replacement for _, replacement in edits] == ["==", "=="]


def test_initial_if_display_edits_leave_quotes_case_and_mod_raw() -> None:
    text = "data out; if x = 'a' and mod(x, 2) = 0 NOT then foo; run;"
    statement = _parse_if_with_entry(text, "if x")
    payload = statement.payload
    assert isinstance(payload, IfStatementSyntax)
    edits = payload.condition_display_edits
    # Only the symbol equality and the logic word get edits; quoted content,
    # the function call, and case are untouched (NOT lowers, mod stays raw).
    assert [text[span.start : span.end] for span, _ in edits] == ["=", "=", "NOT"]
    assert [replacement for _, replacement in edits] == ["==", "==", "not"]


def test_initial_if_fault_keeps_no_display_edits() -> None:
    """A whole-statement IF fault (split failure) has no condition payload
    and therefore no stored display edits."""

    from limulus.syntax.models import DeferredStatementError

    text = "data out; if x = 1 then; run;"
    statement = _parse_if_with_entry(text, "if x")
    payload = statement.payload
    assert isinstance(payload, DeferredStatementError)
    assert "invalid IF statement" in payload.native_cause
