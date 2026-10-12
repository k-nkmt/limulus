"""Focused tests for the assignment statement frontend entry."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from limulus.syntax.models import (
    ArrayElementTargetSyntax,
    AssignStatementSyntax,
    BinarySyntax,
    CallSyntax,
    ConcatSyntax,
    DeferredExpressionError,
    DeferredStatementError,
    LiteralSyntax,
    NameSyntax,
    NameTargetSyntax,
    SourceDocument,
    SourceSpan,
    StatementSyntax,
    SumStatementSyntax,
)

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
                        f"module {fullname!r} is denied by the assignment frontend isolation hook"
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


def _text_of(document: SourceDocument, span: SourceSpan) -> str:
    return document.text[span.start : span.end]


def _assign(statement_text: str, *, result: list[SourceDocument] | None = None):
    """Parse one assignment statement from a single-statement fixture."""

    from limulus.syntax.program import LarkProgramFrontend

    document = SourceDocument(statement_text, "assignments")
    if result is not None:
        result.append(document)
    frontend = LarkProgramFrontend()
    return document, frontend.parse_assignment_statement(document, document.span(0, len(statement_text)))


def _sum(statement_text: str, *, result: list[SourceDocument] | None = None):
    from limulus.syntax.program import LarkProgramFrontend

    document = SourceDocument(statement_text, "assignments")
    if result is not None:
        result.append(document)
    frontend = LarkProgramFrontend()
    return document, frontend.parse_sum_statement(document, document.span(0, len(statement_text)))


def _in_text(text: str, statement: str, entry: str, *, document_out: list[SourceDocument] | None = None):
    from limulus.syntax.program import LarkProgramFrontend

    document = SourceDocument(text, "assignments")
    if document_out is not None:
        document_out.append(document)
    frontend = LarkProgramFrontend()
    start = text.index(statement)
    span = document.span(start, start + len(statement))
    if entry == "assign":
        return frontend.parse_assignment_statement(document, span)
    return frontend.parse_sum_statement(document, span)


# --- ordinary assignment structure -------------------------------------------------


def test_ordinary_assign_target_rhs_types_and_exact_spans() -> None:
    documents: list[SourceDocument] = []
    document, statement = _assign("x = 1", result=documents)
    assert statement.kind == "ASSIGN"
    payload = statement.payload
    assert isinstance(payload, AssignStatementSyntax)
    target = payload.target
    assert isinstance(target, NameTargetSyntax)
    assert target.name == "x"
    assert _text_of(document, target.span) == "x"
    value = payload.value
    assert isinstance(value, LiteralSyntax)
    assert value.kind == "numeric"
    assert value.lexeme == "1"
    assert _text_of(document, value.span) == "1"


def test_assign_rhs_bare_binary_expression() -> None:
    document, statement = _assign("x = grid + 2")
    payload = statement.payload
    value = payload.value
    assert isinstance(value, BinarySyntax)
    assert value.operator == "+"
    assert isinstance(value.left, NameSyntax)
    assert value.left.name == "grid"
    assert isinstance(value.right, LiteralSyntax)
    assert value.right.lexeme == "2"


def test_assign_rhs_quoted_value_keeps_semicolons_and_exact_slice() -> None:
    document, statement = _assign("x = 'a;b'")
    payload = statement.payload
    assert isinstance(payload.value, LiteralSyntax)
    assert payload.value.lexeme == "'a;b'"
    assert _text_of(document, payload.value.span) == "'a;b'"


# --- shared general-expression entry reuse (power / concat) -----------------------


def test_assign_power_right_associative_and_quoted_operator_not_rewritten() -> None:
    document, statement = _assign("q = a ** 2 ** 3")
    value = statement.payload.value
    assert isinstance(value, BinarySyntax)
    assert value.operator == "**"
    assert isinstance(value.left, NameSyntax)
    assert isinstance(value.right, BinarySyntax)
    assert value.right.operator == "**"
    document, quoted = _assign("q = 'a ** 2'")
    assert isinstance(quoted.payload.value, LiteralSyntax)
    assert quoted.payload.value.lexeme == "'a ** 2'"


def test_assign_concat_structure_and_string_operand_not_binary() -> None:
    document, statement = _assign("s = a || b")
    value = statement.payload.value
    assert isinstance(value, ConcatSyntax)
    assert len(value.parts) == 2
    document, statement = _assign("s = a + 1 || b")
    value = statement.payload.value
    assert isinstance(value, ConcatSyntax)
    assert isinstance(value.parts[0], BinarySyntax)


# --- power statements flow through the region pass ---------------------------------


def test_power_assignment_survives_the_region_pass_with_exact_spans() -> None:
    from limulus.syntax.program import parse_program

    result = parse_program("data out; set in; x = 2 ** 3; run;")
    assert result.program is not None
    assert result.diagnostics == (), result.diagnostics
    statements = result.program.blocks[0].statements
    assert [statement.kind for statement in statements] == [
        "DATA", "SET", "ASSIGN", "RUN"
    ]
    assignment = statements[2]
    assert assignment.kind == "ASSIGN"
    assert (
        result.program.source.text[assignment.span.start : assignment.span.end]
        == "x = 2 ** 3"
    )
    payload = assignment.payload
    assert isinstance(payload, AssignStatementSyntax)
    value = payload.value
    assert isinstance(value, BinarySyntax)
    assert value.operator == "**"


def test_tab_arithmetic_assignment_keeps_raw_rhs_display() -> None:
    from limulus.syntax.program import parse_program

    raw = "data out; x = 2 *\t3; run;"
    result = parse_program(raw)
    assert result.program is not None
    assert result.diagnostics == (), result.diagnostics
    assignment = result.program.blocks[0].statements[1]
    assert assignment.kind == "ASSIGN"
    payload_text = result.program.source.text[
        assignment.span.start : assignment.span.end
    ]
    assert payload_text == "x = 2 *\t3"


# --- array targets (legacy binding contracts) -------------------------------------


@pytest.mark.parametrize(
    ("statement_text", "delimiter", "rhs_lexeme"),
    [
        ("a[i] = b", "[", "b"),
        ("m{2} = 'txt'", "{", "'txt'"),
        ("c(2) = d", "(", "d"),
    ],
)
def test_array_targets_keep_delimiter_index_and_spans(
    statement_text: str, delimiter: str, rhs_lexeme: str
) -> None:
    documents: list[SourceDocument] = []
    document, statement = _assign(statement_text, result=documents)
    payload = statement.payload
    assert isinstance(payload, AssignStatementSyntax)
    target = payload.target
    assert isinstance(target, ArrayElementTargetSyntax)
    assert target.name == statement_text.split(delimiter, 1)[0]
    assert target.delimiter == delimiter
    # The target span covers the whole legacy binding token (name through
    # bracket close), matching the native array-reference token shape.
    assert _text_of(document, target.span) == statement_text.split(" ", 1)[0]
    assert isinstance(target.index, (NameSyntax, LiteralSyntax))
    rhs = payload.value
    assert isinstance(rhs, (NameSyntax, LiteralSyntax))
    assert _text_of(document, rhs.span) == rhs_lexeme


# --- separator identification ------------------------------------------------------


def test_raw_first_equals_is_the_separator_with_opaque_comparison_target() -> None:
    # Native `parse_assignment` (mod.rs L209) does a raw split_once('='):
    # quotes, brackets, and comparison `==` contexts are all ignored, and the
    # left side does not pass any identifier gate.  The typed entry keeps the
    # same binding: `a[i == j] = x` splits at the first raw `=` giving the
    # opaque scalar name target `a[i` and an RHS that fails closed as a
    # whole-expression descriptor (legacy ledger).
    document, statement = _assign("a[i == j] = x")
    payload = statement.payload
    target = payload.target
    assert isinstance(target, NameTargetSyntax)
    assert target.name == "a[i"
    assert _text_of(document, target.span) == "a[i"
    assert isinstance(payload.value, DeferredExpressionError)
    assert payload.value.native_cause == "expression expected"


def test_concat_index_without_equals_keeps_array_target_structure() -> None:
    document, statement = _assign("a[i||j] = x")
    payload = statement.payload
    target = payload.target
    assert isinstance(target, ArrayElementTargetSyntax)
    assert target.name == "a"
    assert target.delimiter == "["
    assert isinstance(target.index, ConcatSyntax)
    assert isinstance(payload.value, NameSyntax)
    assert payload.value.name == "x"


def test_inline_then_action_span_is_owned_by_the_caller() -> None:
    text = "data out; if p then a = 1; run;\n"
    statement = _in_text(text, "a = 1", "assign")
    payload = statement.payload
    assert isinstance(payload, AssignStatementSyntax)
    assert payload.target.name == "a"
    assert isinstance(payload.value, LiteralSyntax)
    # The inline span is anchored to the whole program document; the span
    # equals the caller-provided slice.
    assert statement.span.start == text.index("a = 1")
    assert statement.span.end == text.index("a = 1") + len("a = 1")


# --- open-quote assignment inside block --------------------------------------------


def test_open_quote_assignment_inside_block_keeps_deferred_rhs() -> None:
    raw = "data a; set in; x = 'abc; run;"
    from limulus.syntax.program import parse_program

    result = parse_program(raw, source_id="<assignments>")
    assert result.program is not None
    assert result.diagnostics == ()
    assignment = result.program.blocks[0].statements[2]
    assert assignment.kind == "ASSIGN"
    payload = assignment.payload
    assert isinstance(payload, AssignStatementSyntax)
    assert payload.target.name == "x"
    value = payload.value
    assert isinstance(value, DeferredExpressionError)
    assert value.error_kind == "parse_fault"
    assert value.native_cause == "unterminated string literal"


def test_open_quote_sum_inside_block_keeps_deferred_increment() -> None:
    from limulus.syntax.models import SumStatementSyntax
    from limulus.syntax.program import parse_program

    raw = "data a; x + 'abc; run;"
    result = parse_program(raw, source_id="<names>")
    assert result.program is not None
    assert result.diagnostics == ()
    sum_statement = result.program.blocks[0].statements[1]
    assert sum_statement.kind == "SUM"
    payload = sum_statement.payload
    assert isinstance(payload, SumStatementSyntax)
    increment = payload.increment
    assert isinstance(increment, DeferredExpressionError)
    assert increment.error_kind == "parse_fault"
    assert increment.native_cause == "unterminated string literal"


# --- deferred RHS, target faults, and apply inventory ------------------------------


def test_rhs_fault_is_whole_expression_deferred_without_partial_evaluation() -> None:
    document, statement = _assign("x = a or *2")
    value = statement.payload.value
    assert isinstance(value, DeferredExpressionError)
    assert value.error_kind == "parse_fault"
    assert statement.payload.target.name == "x"


def test_apply_inventory_on_valid_rhs_source_order_and_duplicates() -> None:
    document, statement = _assign(
        'x = apply("fn", a) + apply("fn", b) + apply("other", c)'
    )
    references = statement.apply_references
    assert [ref.target for ref in references] == ["fn", "fn", "other"]
    assert references[0].owner_span == statement.span


def test_opaque_scalar_targets_are_accepted_with_inventory_like_native() -> None:
    # Native binding accepts any nonempty trimmed left side; `x y` is an
    # opaque scalar target (no identifier gate) and the apply inventory
    # stays the whole-statement lexical scan.
    document, statement = _assign("x y = apply('fn', a)")
    payload = statement.payload
    assert isinstance(payload, AssignStatementSyntax)
    assert isinstance(payload.target, NameTargetSyntax)
    assert payload.target.name == "x y"
    assert _text_of(document, payload.target.span) == "x y"
    assert isinstance(payload.value, CallSyntax)
    references = statement.apply_references
    assert [ref.target for ref in references] == ["fn"]


def test_unterminated_target_quote_keeps_array_target_with_deferred_index() -> None:
    corpus = "x['bad] = apply(\"fn\", a) +"
    document, statement = _assign(corpus)
    payload = statement.payload
    assert isinstance(payload, AssignStatementSyntax)
    target = payload.target
    assert isinstance(target, ArrayElementTargetSyntax)
    assert target.name == "x"
    assert target.delimiter == "["
    assert _text_of(document, target.index.span) == "'bad"
    assert isinstance(target.index, DeferredExpressionError)
    assert target.index.native_cause == "unterminated string literal"
    # An unterminated quote region consumes the remainder, so later apply
    # calls contribute nothing to the saved typed inventory.
    assert statement.apply_references == ()


def test_rhs_fault_descriptor_keeps_its_own_inventory() -> None:
    document, statement = _assign("x = apply('fn', a) * || 2")
    value = statement.payload.value
    assert isinstance(value, DeferredExpressionError)
    assert [ref.target for ref in value.apply_references] == ["fn"]


def test_unsupported_unary_plus_and_list_literal_stay_deferred() -> None:
    document, statement = _assign("x = +5")
    assert isinstance(statement.payload.value, DeferredExpressionError)
    assert statement.payload.target.name == "x"
    document, statement = _assign("x = [1, 2]")
    assert isinstance(statement.payload.value, DeferredExpressionError)


# --- sum statements ----------------------------------------------------------------


def test_sum_structures_single_name_target_and_increment() -> None:
    documents: list[SourceDocument] = []
    document, statement = _sum("total + 1", result=documents)
    assert statement.kind == "SUM"
    payload = statement.payload
    assert isinstance(payload, SumStatementSyntax)
    assert payload.target.name == "total"
    assert _text_of(document, payload.target.span) == "total"
    increment = payload.increment
    assert isinstance(increment, LiteralSyntax)
    assert increment.lexeme == "1"
    assert _text_of(document, increment.span) == "1"

    document, statement = _sum("total + a + 2")
    payload = statement.payload
    increment = payload.increment
    assert isinstance(increment, BinarySyntax)
    assert increment.operator == "+"
    # Expression node spans anchor to their own occurrence (operator token
    # span per the general-expression contract); operands keep exact operand slices.
    assert _text_of(document, increment.left.span) == "a"
    assert _text_of(document, increment.right.span) == "2"


def test_sum_increment_with_apply_keeps_inventory() -> None:
    document, statement = _sum("total + apply('fn', a)")
    references = statement.apply_references
    assert [ref.target for ref in references] == ["fn"]


def test_sum_non_identifier_name_is_never_a_fabricated_sum() -> None:
    # The SUM statement shape check is the legacy grammar's own
    # `[A-Za-z_][\w\.]*\s*\+` gate: `x[1] + 2` and `invalid a b+1` were never
    # SUM statements, so the limited entry refuses them (the structural
    # structural classification owner handles the unsupported statement)
    # instead of fabricating an accepted SUM payload.
    with pytest.raises(ValueError):
        _sum("x[1] + 2")
    with pytest.raises(ValueError):
        _sum("invalid a b+1")


def test_sum_blank_parts_keep_native_statement_fault_family() -> None:
    document, statement = _sum("+2")
    payload = statement.payload
    assert isinstance(payload, DeferredStatementError)
    assert payload.native_cause.startswith("Invalid SUM statement")
    assert payload.display_text == "+2"
    assert statement.kind == "SUM"
    document, statement = _sum("total +")
    payload = statement.payload
    assert isinstance(payload, DeferredStatementError)
    assert payload.native_cause.startswith("Invalid SUM statement")


def test_sum_entry_classification_is_limited() -> None:
    with pytest.raises(ValueError):
        _sum("x == y")


def test_sum_increment_fault_is_deferred_expression() -> None:
    document, statement = _sum("total + apply('fn', a) +")
    payload = statement.payload
    assert not isinstance(payload, DeferredStatementError)
    assert isinstance(payload.increment, DeferredExpressionError)


# --- target failures keep typed statement payload ----------------------------------


@pytest.mark.parametrize(
    ("statement_text", "expected_target"),
    [
        ("x[1 = 2", "x[1"),  # unbalanced target bracket: opaque scalar target
        ("x[] = 2", "x[]"),  # empty array index candidate skipped: opaque
        ("x{ } = 2", "x{ }"),  # blank array index candidate skipped: opaque
        ("x 'stray' = 2", "x 'stray'"),  # no bracket: opaque scalar target
    ],
)
def test_opaque_targets_follow_native_raw_binding(
    statement_text: str, expected_target: str
) -> None:
    statement = _in_text(statement_text, statement_text, "assign")
    payload = statement.payload
    assert isinstance(payload, AssignStatementSyntax)
    assert isinstance(payload.target, NameTargetSyntax)
    assert payload.target.name == expected_target
    assert isinstance(payload.value, LiteralSyntax)
    assert statement.kind == "ASSIGN"
    assert isinstance(statement.span, SourceSpan)


@pytest.mark.parametrize(
    "statement_text",
    ["= 2", "x ="],
)
def test_assignment_shape_faults_use_native_cause_family(statement_text: str) -> None:
    statement = _in_text(statement_text, statement_text, "assign")
    payload = statement.payload
    assert isinstance(payload, DeferredStatementError)
    assert "invalid assignment statement" in payload.native_cause


def test_multi_dimension_index_fault_stays_deferred_inside_target() -> None:
    document, statement = _assign("x{2,3} = 1")
    payload = statement.payload
    assert isinstance(payload, AssignStatementSyntax)
    target = payload.target
    assert isinstance(target, ArrayElementTargetSyntax)
    index = target.index
    assert isinstance(index, DeferredExpressionError)
    assert _text_of(document, index.span) == "2,3"
    assert "unexpected token at end of expression" in index.native_cause


# --- entry boundary ----------------------------------------------------------------


def test_non_assignment_rejected_without_filler() -> None:
    document = SourceDocument("run;")
    from limulus.syntax.program import LarkProgramFrontend

    frontend = LarkProgramFrontend()
    with pytest.raises(ValueError):
        frontend.parse_assignment_statement(document, document.span(0, 4))
    with pytest.raises(ValueError):
        frontend.parse_sum_statement(document, document.span(0, 4))


def test_span_must_belong_to_document() -> None:
    document = SourceDocument("x = 1")
    from limulus.syntax.program import LarkProgramFrontend

    frontend = LarkProgramFrontend()
    foreign = SourceSpan(start=0, end=5, line=1, column=1, source_id="other")
    with pytest.raises(ValueError):
        frontend.parse_assignment_statement(document, foreign)


def test_region_object_entry_uses_statement_span() -> None:
    document = SourceDocument("x = 1")
    from limulus.syntax.program import LarkProgramFrontend, ProgramStatementRegion

    frontend = LarkProgramFrontend()
    region = ProgramStatementRegion(kind="OTHER", span=document.span(0, 5))
    statement = frontend.parse_assignment_statement(document, region)
    assert statement.kind == "ASSIGN"
    assert statement.span == region.span
    assert isinstance(statement.payload, AssignStatementSyntax)


def test_assignment_frontend_imports_without_data_environment() -> None:
    _run_isolated(
        """
        import sys

        from limulus.syntax.program import LarkProgramFrontend
        from limulus.syntax.models import SourceDocument

        frontend = LarkProgramFrontend()
        document = SourceDocument("x = a ** 2 ** 3")
        statement = frontend.parse_assignment_statement(
            document, document.span(0, len(document.text))
        )
        payload = statement.payload
        assert payload.value.operator == "**"
        for denied in ("pyarrow", "polars", "limulus_native", "limulus.session"):
            assert denied not in sys.modules, denied
        """
    )


# --- raw index spans, unicode offsets, and no beyond-span scans ---------------------


def test_raw_index_carries_all_chars_from_first_opener_to_terminal_closer() -> None:
    # Native parse_array_reference_token (mod.rs L311): the first opener in
    # the candidate order wins and the index keeps every source character up
    # to the terminal closer BEFORE any trim; no first-balanced-close and no
    # full-source walk.  `x[1]trash[2]` therefore binds name `x` with the raw
    # index `1]trash[2`, which the general-expression entry fails closed (never index 1).
    document, statement = _assign("x[1]trash[2] = 1")
    payload = statement.payload
    target = payload.target
    assert isinstance(target, ArrayElementTargetSyntax)
    assert target.name == "x"
    assert target.delimiter == "["
    assert _text_of(document, target.index.span) == "1]trash[2"
    assert isinstance(target.index, DeferredExpressionError)
    assert target.index.native_cause == "unexpected token at end of expression"


def test_raw_index_span_keeps_surrounding_whitespace_before_expression_trim() -> None:
    document, statement = _assign("x[ i ] = v")
    payload = statement.payload
    target = payload.target
    assert isinstance(target, ArrayElementTargetSyntax)
    assert target.name == "x"
    assert isinstance(target.index, NameSyntax)
    assert _text_of(document, target.index.span) == "i"
    # The parse-input index range is the raw bounded slice (open+1 .. the
    # terminal closer, whitespace included) and never walks beyond the
    # caller's span; the emitted node span follows the general-expression occurrence
    # convention (the name node anchors to its own slice `i`).  The raw-range
    # behavior is proven by the whole-slice deferred spans in the corpora
    # above (`2,3`, `1]trash[2`, `'bad`) and by the no-beyond-span test.
    assert isinstance(payload.value, NameSyntax)
    assert _text_of(document, payload.value.span) == "v"


def test_index_span_stays_inside_the_caller_span_with_unicode_and_inline() -> None:
    text = "値 = 前文; left = m{添字} = right; z = 4"
    statements: list[SourceDocument] = []
    statement = _in_text(text, "m{添字} = right", "assign", document_out=statements)
    document = statements[-1]
    payload = statement.payload
    assert isinstance(payload, AssignStatementSyntax)
    # Progress marched past the earlier statement and stops inside the span.
    assert statement.span.start == text.index("m{添字}")
    assert statement.span.end == text.index("m{添字}") + len("m{添字} = right")
    target = payload.target
    assert isinstance(target, ArrayElementTargetSyntax)
    assert target.name == "m"
    assert target.delimiter == "{"
    assert _text_of(document, target.index.span) == "添字"
    assert _text_of(document, payload.value.span) == "right"
    # The raw index span never reaches past the statement span.
    assert statement.span.start < target.index.span.start
    assert target.index.span.end <= statement.span.end


def test_unbalanced_target_does_not_scan_beyond_the_span() -> None:
    # A closing bracket appearing AFTER the statement span must not be pulled
    # into the index span: `x[1 = v` inside a larger document binds the
    # opaque scalar target `x[1` with RHS `v` exactly like the native split.
    text = "head; x[1 = v; ] = other\n"
    statement = _in_text(text, "x[1 = v", "assign")
    payload = statement.payload
    assert isinstance(payload, AssignStatementSyntax)
    assert isinstance(payload.target, NameTargetSyntax)
    assert payload.target.name == "x[1"
    assert isinstance(payload.value, NameSyntax)
    assert payload.value.name == "v"


def test_case_s_with_double_equals_splits_at_first_raw_equals() -> None:
    statement = _in_text("s=='flag'=1", "s=='flag'=1", "assign")
    payload = statement.payload
    assert isinstance(payload, AssignStatementSyntax)
    assert payload.target.name == "s"
    assert isinstance(payload.value, DeferredExpressionError)
    assert payload.value.native_cause == "expression expected"
    assert payload.value.span.start == 2
    assert payload.value.span.end == len("s=='flag'=1")
