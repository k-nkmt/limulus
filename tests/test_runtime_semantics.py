"""Public runtime-semantics regressions exercised through ``Session.submit``."""

from decimal import Decimal

import pyarrow as pa
import pytest

from limulus.execution import DataStepExecutor
from limulus.execution.contracts import PipelineExecutionRequest
from limulus import Session
from limulus.models import DataSetRef, ExecuteRequest, PreparationProvenance


def _new_session() -> Session:
    session = Session()
    session.load("inp", pa.table({"flag": [True], "amount": [10]}))
    return session


def _submit_rows(dsl: str) -> list[dict[str, object]]:
    result = _new_session().submit(dsl)
    assert result.success is True, result.format_log()
    return result.datasets["out"].to_pylist()


def _submit_result(dsl: str):
    return _new_session().submit(dsl)


@pytest.mark.parametrize(
    ("body", "expect_missing_notice", "expect_empty_notice"),
    [
        ('text = ".";', False, False),
        ("text = '.';", False, False),
        ('text = "a" || ".";', False, False),
        ('label id = ".";', False, False),
        ("/* . */ value = 1;", False, False),
        ("* .; value = 1;", False, False),
        ("%put .; value = 1;", False, False),
        ("value = .5;", False, False),
        ("value = None;", False, False),
        ("value = .;", True, False),
        ("value = -.;", True, False),
        ("value = sum(id, .);", True, False),
        ("where id != .;", True, False),
        ("if 0 then value = .;", True, False),
        ('text = "a" || .;', True, False),
        ('text = "";', False, True),
        ("text = '';", False, True),
        ('text = " ";', False, False),
        ('value = length("");', False, True),
        ('if 0 then text = "";', False, True),
        ('text = "x"; if text = "";', False, True),
        ('text = ""; value = .;', True, True),
        ('/* "" */ value = 1;', False, False),
        ('* ""; value = 1;', False, False),
        ('%put ""; value = 1;', False, False),
        ('label id = "";', False, False),
    ],
)
def test_missing_value_notices_use_parsed_literals(
    body: str, expect_missing_notice: bool, expect_empty_notice: bool
) -> None:
    source = pa.table({"id": [1]})
    response = DataStepExecutor().execute(
        ExecuteRequest(
            dsl_text=f"data out; set inp; {body} run;",
            inputs={"inp": DataSetRef(kind="arrow_table", location="dataset://inp", payload=source)},
        )
    )

    assert not response.has_errors, response.diagnostics
    expected = []
    if expect_missing_notice:
        expected.append("COMPAT_MISSING_VALUE_SEMANTICS")
    if expect_empty_notice:
        expected.append("COMPAT_EMPTY_STRING_SEMANTICS")
    assert [notice.id for notice in response.notices] == expected


def test_empty_string_notice_does_not_scan_input_values() -> None:
    response = DataStepExecutor().execute(
        ExecuteRequest(
            dsl_text="data out; set inp; run;",
            inputs={
                "inp": DataSetRef(
                    kind="arrow_table", location="dataset://inp", payload=pa.table({"text": [""]})
                )
            },
        )
    )

    assert not response.has_errors, response.diagnostics
    assert response.outputs_arrow["out"].to_pylist() == [{"text": ""}]
    assert response.notices == ()


class TestMissingLiteralStatementContexts:
    """A standalone dot is the same NULL literal in every statement context.

    Quoted dots stay text, numeric dots such as ``.5`` stay numeric, and a
    misplaced dot fails closed with a stage-aware diagnostic instead of
    publishing a partial output.
    """

    def test_top_level_assign_dot_is_null(self) -> None:
        rows = _submit_rows("data out; set inp; x = .; run;")
        assert rows == [{"flag": True, "amount": 10, "x": None}]

    def test_null_keyword_supports_assignment_and_equality(self) -> None:
        rows = _submit_rows(
            "data out; set inp; x = null; equal_null = x = null; if x = null; run;"
        )
        assert rows == [{"flag": True, "amount": 10, "x": None, "equal_null": True}]

    def test_if_inline_assign_dot_is_null(self) -> None:
        rows = _submit_rows("data out; set inp; if flag then x = .; run;")
        assert rows == [{"flag": True, "amount": 10, "x": None}]

    def test_else_inline_assign_dot_is_null(self) -> None:
        rows = _submit_rows("data out; set inp; if amount < 0 then x = 1; else x = .; run;")
        assert rows == [{"flag": True, "amount": 10, "x": None}]

    def test_else_inline_unique_assignment_target_is_collected(self) -> None:
        rows = _submit_rows("data out; set inp; if amount < 0 then x = 1; else y = .; run;")
        assert rows == [{"flag": True, "amount": 10, "x": None, "y": None}]

    def test_do_block_interior_assign_dot_is_null(self) -> None:
        rows = _submit_rows("data out; set inp; if flag then do; x = .; end; run;")
        assert rows == [{"flag": True, "amount": 10, "x": None}]

        rows = _submit_rows("data out; set inp; if amount < 0 then do; x = 1; end; else do; x = .; end; run;")
        assert rows == [{"flag": True, "amount": 10, "x": None}]

    def test_quoted_dot_stays_text_in_both_quote_styles(self) -> None:
        rows = _submit_rows('data out; set inp; x = "."; run;')
        assert rows == [{"flag": True, "amount": 10, "x": "."}]

        rows = _submit_rows("data out; set inp; x = '.'; run;")
        assert rows == [{"flag": True, "amount": 10, "x": "."}]

    def test_decimal_dot_stays_numeric(self) -> None:
        rows = _submit_rows("data out; set inp; x = .5; run;")
        assert rows == [{"flag": True, "amount": 10, "x": 0.5}]

    def test_misplaced_dot_fails_stage_aware_without_partial_publish(self) -> None:
        result = _submit_result("data out; set inp; x = 1 . 2; run;")

        assert result.success is False
        assert "out" not in result.datasets
        rendered = result.format_log()
        assert "unexpected token at end of expression" in rendered.lower()
        assert any(
            entry.severity == "error" and entry.code == "RUNTIME_EXPRESSION_EVALUATION_ERROR"
            for entry in result.log
        )

    def test_bare_do_block_assignment_is_stage_aware_parse_rejected(self) -> None:
        # A bare top-level ``do; ... end;`` is not part of the statement grammar
        # (DO blocks are owned by IF/ELSE actions), so the missing-literal body
        # must surface as the stage-aware parse diagnostic, not a runtime crash
        # or a partial publish.
        result = _submit_result("data out; set inp; do; x = .; end; run;")

        assert result.success is False
        assert "out" not in result.datasets
        assert any(
            entry.severity == "error"
            and entry.code == "PARSE_UNSUPPORTED_STATEMENT"
            and entry.stage == "parse"
            for entry in result.log
        )

    def test_none_and_null_keywords_are_the_same_null_as_dot(self) -> None:
        # ``None``/``null`` spell the same missing literal as ``.``; the public
        # assignment path must not turn either keyword into text or zero.
        rows = _submit_rows("data out; set inp; x = None; run;")
        assert rows == [{"flag": True, "amount": 10, "x": None}]

        rows = _submit_rows("data out; set inp; x = null; run;")
        assert rows == [{"flag": True, "amount": 10, "x": None}]

    def test_identifier_trailing_dot_resolves_per_native_contract(self) -> None:
        # Native token contract (native/limulus_native/src/expressions/mod.rs
        # resolve_simple_expression_value): an identifier token ending in ``.``
        # is kept whole; with ``a`` undefined it resolves to Text("a.") instead
        # of the bare-variable NULL. The dot here is part of the name token,
        # never a missing literal.
        rows = _submit_rows("data out; set inp; x = a.; run;")
        assert rows == [{"flag": True, "amount": 10, "x": "a."}]

    def test_dot_plus_dot_propagates_null(self) -> None:
        # Missing literal in arithmetic context propagates NULL through the
        # native arithmetic guard instead of coercing to zero.
        rows = _submit_rows("data out; set inp; x = . + .; run;")
        assert rows == [{"flag": True, "amount": 10, "x": None}]


class TestUnicodeScalarStringFunctions:
    """The six string functions expose Unicode scalar semantics via Session.submit.

    Positions and lengths are 1-based Unicode scalar counts (never UTF-8 bytes or
    grapheme clusters), search matches map back to original-scalar ordinals, no
    normalization is applied, and malformed input fails closed through the
    result envelope instead of panicking.
    """

    ZWJ_FAMILY = "👨‍👩‍👧‍👦"  # 7 scalars: 👨 ZWJ 👩 ZWJ 👧 ZWJ 👦
    JP_FLAG = "🇯🇵"  # 2 scalars: 🇯 U+1F1EF, 🇵 U+1F1F5
    DECOMPOSED_CAFE = "cafe\u0301"  # 'e' + combining acute = 5 scalars

    @staticmethod
    def _text_session(texts: list[str]) -> Session:
        session = Session()
        session.load("inp", pa.table({"txt": texts}))
        return session

    @classmethod
    def _rows(cls, texts: list[str], assignments: str) -> list[dict[str, object]]:
        result = cls._text_session(texts).submit(f"data out; set inp; {assignments} run;")
        assert result.success is True, result.format_log()
        return result.datasets["out"].to_pylist()

    def test_substr_scalar_positions_on_cjk(self) -> None:
        rows = self._rows(["日本語"], "tail = substr(txt, 2); single = substr(txt, 2, 1);")

        assert rows == [{"txt": "日本語", "tail": "本語", "single": "本"}]

    def test_substr_picks_scalar_inside_multi_scalar_emoji_and_flag(self) -> None:
        rows = self._rows(
            [self.ZWJ_FAMILY],
            "n = length(txt); third = substr(txt, 3, 1);",
        )
        assert rows == [{"txt": self.ZWJ_FAMILY, "n": 7, "third": "👩"}]

        rows = self._rows(
            [self.JP_FLAG],
            "n = length(txt); second = substr(txt, 2, 1);",
        )
        assert rows == [{"txt": self.JP_FLAG, "n": 2, "second": "\U0001f1f5"}]

    def test_length_and_lengthn_count_scalars_not_bytes(self) -> None:
        texts = ["日本語", "café", self.DECOMPOSED_CAFE, self.ZWJ_FAMILY, self.JP_FLAG, "abc", ""]
        rows = self._rows(texts, "n = length(txt); nn = lengthn(txt);")

        assert rows == [
            {"txt": "日本語", "n": 3, "nn": 3},
            {"txt": "café", "n": 4, "nn": 4},
            {"txt": self.DECOMPOSED_CAFE, "n": 5, "nn": 5},
            {"txt": self.ZWJ_FAMILY, "n": 7, "nn": 7},
            {"txt": self.JP_FLAG, "n": 2, "nn": 2},
            {"txt": "abc", "n": 3, "nn": 3},
            {"txt": "", "n": 0, "nn": 0},
        ]

    def test_substr_and_length_treat_precomposed_and_decomposed_accents_as_written(self) -> None:
        rows = self._rows(["café"], "accent = substr(txt, 4); n = length(txt);")
        assert rows == [{"txt": "café", "accent": "é", "n": 4}]

        rows = self._rows([self.DECOMPOSED_CAFE], "accent = substr(txt, 5); n = length(txt);")
        assert rows == [{"txt": self.DECOMPOSED_CAFE, "accent": "\u0301", "n": 5}]

    def test_find_and_index_report_one_based_scalar_positions(self) -> None:
        rows = self._rows(
            ["日本語のテキスト"],
            "idx_no = index(txt, 'の'); find_text = find(txt, 'テキスト');"
            " miss_find = find(txt, 'XYZ'); miss_index = index(txt, 'XYZ');",
        )

        assert rows == [
            {"txt": "日本語のテキスト", "idx_no": 4, "find_text": 5, "miss_find": 0, "miss_index": 0}
        ]

    def test_find_start_position_is_a_scalar_ordinal_and_stays_bounded(self) -> None:
        rows = self._rows(
            ["日本語"],
            "from1 = find(txt, '本', 1); from2 = find(txt, '本', 2);"
            " skipped = find(txt, '日', 2); beyond = find(txt, '本', 99);",
        )

        assert rows == [
            {"txt": "日本語", "from1": 2, "from2": 2, "skipped": 0, "beyond": 0}
        ]

    def test_find_case_insensitive_reports_original_scalar_positions(self) -> None:
        rows = self._rows(
            ["日本ABC語"],
            "from1 = find(txt, 'abc', 1, 'i'); from2 = find(txt, 'abc', 2, 'i');",
        )
        assert rows == [{"txt": "日本ABC語", "from1": 3, "from2": 3}]

        # 'İ' (U+0130) lowers to 'i' + U+0307; the reported position must stay the
        # matched İ's scalar ordinal in the original string.
        rows = self._rows(["İXİy"], "ci = find(txt, 'İy', 1, 'i');")
        assert rows == [{"txt": "İXİy", "ci": 3}]

        rows = self._rows(["İXy"], "ci = find(txt, 'xy', 1, 'i');")
        assert rows == [{"txt": "İXy", "ci": 2}]

    def test_search_compares_stored_scalars_without_normalization(self) -> None:
        rows = self._rows([self.DECOMPOSED_CAFE], "miss = find(txt, 'café');")
        assert rows == [{"txt": self.DECOMPOSED_CAFE, "miss": 0}]

        rows = self._rows(["café"], "miss = index(txt, 'e\u0301');")
        assert rows == [{"txt": "café", "miss": 0}]

    def test_prxmatch_reports_scalar_positions(self) -> None:
        rows = self._rows(
            ["日本語のテキスト"],
            "pos = prxmatch('/テキスト/', txt); miss = prxmatch('/ZZZ/', txt);",
        )

        assert rows == [{"txt": "日本語のテキスト", "pos": 5, "miss": 0}]

    def test_substr_boundary_positions_stay_bounded_and_panic_free(self) -> None:
        rows = self._rows(
            ["日本語", "abc"],
            "beyond = substr(txt, 99); oversized = substr(txt, 2, 999); negative = substr(txt, 1, -5);",
        )

        assert rows == [
            {"txt": "日本語", "beyond": "", "oversized": "本語", "negative": ""},
            {"txt": "abc", "beyond": "", "oversized": "bc", "negative": ""},
        ]

    def test_invalid_arity_fails_closed_with_diagnostic_and_no_partial_publish(self) -> None:
        result = self._text_session(["x"]).submit("data out; set inp; x = substr(txt); run;")

        assert result.success is False
        assert "out" not in result.datasets
        rendered = result.format_log()
        assert "substr" in rendered.lower()
        assert "expects" in rendered.lower()
        assert any(
            entry.severity == "error" and entry.code == "RUNTIME_EXPRESSION_EVALUATION_ERROR"
            for entry in result.log
        )

        result = self._text_session(["x"]).submit("data out; set inp; x = length(txt, txt); run;")

        assert result.success is False
        assert "out" not in result.datasets
        assert any(
            entry.severity == "error" and entry.code == "RUNTIME_EXPRESSION_EVALUATION_ERROR"
            for entry in result.log
        )


class TestNullArithmeticAndDecimalConversion:
    """NULL arithmetic and Decimal conversion via Session.submit.

    NULL propagation and the non-NULL guard are verified per operator with a
    nullable source column, so the NULL row and the non-NULL row are evaluated
    in the same data step: a NULL operand propagates while the non-NULL row
    keeps its existing numeric result. Comparison, logical, and ordering guards
    with NULL keep the native contract values unchanged.

    Decimal conversion is verified on the public path. Invalid Decimal
    fail-closed evidence is produced through an apply callback returning
    ``Decimal('sNaN')`` — a valid Python Decimal whose scalar text form is not
    f64-parseable (every other finite Decimal instance always parses, so this
    checked coercion error is otherwise only reachable at native level, where
    the native expressions suite covers all invalid payload shapes). Valid
    Decimal coercion is covered through Arrow decimal128 input columns and
    apply callbacks.
    """

    ARITHMETIC_CASES = [
        ("s = a + 1;", 2.0),
        ("s = a - 1;", 0.0),
        ("s = a * 2;", 2.0),
        ("s = a / 2;", 0.5),
        ("s = a mod 3;", 1.0),
    ]

    @staticmethod
    def _null_arithmetic_session() -> Session:
        session = Session()
        session.load("inp", pa.table({"a": [1, None], "e": [3, None]}))
        return session

    @staticmethod
    def _apply_session() -> Session:
        session = Session()
        session.load("inp", pa.table({"amount": [10]}))
        return session

    @classmethod
    def _rows(cls, assignment: str) -> list[dict[str, object]]:
        result = cls._null_arithmetic_session().submit(f"data out; set inp; {assignment} run;")
        assert result.success is True, result.format_log()
        return result.datasets["out"].to_pylist()

    def test_binary_null_arithmetic_propagates_and_non_null_results_are_unchanged(self) -> None:
        for assignment, non_null_result in self.ARITHMETIC_CASES:
            rows = self._rows(assignment)
            assert rows == [
                {"a": 1, "e": 3, "s": non_null_result},
                {"a": None, "e": None, "s": None},
            ], assignment

    def test_unary_negative_propagates_null_and_keeps_non_null_results(self) -> None:
        rows = self._rows("s = -a;")
        assert rows == [
            {"a": 1, "e": 3, "s": -1.0},
            {"a": None, "e": None, "s": None},
        ]

    def test_power_propagates_null_with_variable_exponent(self) -> None:
        # With a variable exponent the `**` assignment text crosses the bridge
        # unchanged and evaluates through the native Pow operator, which must
        # propagate NULL exactly like the other binary arithmetic operators.
        rows = self._rows("x = 2 ** e; y = 2 ** a;")
        assert rows == [
            {"a": 1, "e": 3, "x": 8.0, "y": 2.0},
            {"a": None, "e": None, "x": None, "y": None},
        ]

    def test_power_propagates_null_with_numeric_exponent(self) -> None:
        # `a ** 2` in an assignment is normalized to
        # `pow(a, 2)` before the native handoff, so the Pow NULL guard must
        # also hold through the pow() function form; a NULL operand must never
        # silently fold to 0.0.
        rows = self._rows("x = a ** 2;")
        assert rows == [
            {"a": 1, "e": 3, "x": 1.0},
            {"a": None, "e": None, "x": None},
        ]

    def test_explicit_pow_function_propagates_null_and_keeps_non_null_results(self) -> None:
        # The bridge normalizes `var ** lit` assignments to pow(var, lit), so
        # the explicit function form must keep the same binary-arithmetic NULL
        # guard as the operator form: NULL propagates, non-NULL stays 1.0
        # (pow(1, 2)).
        rows = self._rows("x = pow(a, 2);")
        assert rows == [
            {"a": 1, "e": 3, "x": 1.0},
            {"a": None, "e": None, "x": None},
        ]

    def test_explicit_mod_function_matches_operator_null_propagation(self) -> None:
        rows = self._rows("x = mod(a, 3); y = mod(3, a); z = mod(a, a);")
        assert rows == [
            {"a": 1, "e": 3, "x": 1.0, "y": 0.0, "z": 0.0},
            {"a": None, "e": None, "x": None, "y": None, "z": None},
        ]

    def test_comparison_logical_and_ordering_guards_are_unchanged(self) -> None:
        # Native contract values (expressions/mod.rs
        # comparison_and_logical_null_semantics_are_unchanged and
        # null_ordering_semantics_are_unchanged) with variable operands across
        # NULL and non-NULL rows: equality with the missing literal, logical
        # coercion (NULL -> false), and ordering comparisons keep their
        # existing rules on the public path.
        session = self._null_arithmetic_session()
        result = session.submit(
            "data out; set inp; eq_missing = a = .; ne_missing = a != .; negated = not a;"
            " gt_zero = a > 0; lt_one = a < 1; or_false = a or false; and_true = a and true; run;"
        )
        assert result.success is True, result.format_log()
        assert result.datasets["out"].to_pylist() == [
            {
                "a": 1,
                "e": 3,
                "eq_missing": False,
                "ne_missing": True,
                "negated": False,
                "gt_zero": True,
                "lt_one": False,
                "or_false": True,
                "and_true": True,
            },
            {
                "a": None,
                "e": None,
                "eq_missing": True,
                "ne_missing": False,
                "negated": True,
                "gt_zero": False,
                "lt_one": True,
                "or_false": False,
                "and_true": False,
            },
        ]

    def test_decimal128_column_checked_numeric_and_boolean_coercion(self) -> None:
        # Arrow decimal128 input columns decode to ScalarValue::Decimal on the
        # standard source path; checked numeric coercion keeps existing numeric
        # results and boolean consumption of a valid Decimal stays unchanged.
        # The NULL decimal row also proves the arithmetic NULL guard precedes
        # Decimal coercion (Decimal NULL never folds to 0).
        session = Session()
        session.load(
            "inp",
            pa.table({"d": pa.array([Decimal("1.5"), None], type=pa.decimal128(10, 2))}),
        )
        result = session.submit("data out; set inp; x = d + 1; y = not (d > 0); run;")
        assert result.success is True, result.format_log()
        assert result.datasets["out"].to_pylist() == [
            {"d": Decimal("1.50"), "x": 2.5, "y": False},
            {"d": None, "x": None, "y": True},
        ]

    def test_apply_decimal_checked_numeric_and_boolean_coercion(self) -> None:
        # Host callbacks return Python decimal.Decimal values; they cross the
        # bridge as ScalarValue::Decimal and the checked numeric/boolean
        # coercions keep the native results on the public path.
        def decimal_target(value: object) -> object:
            return Decimal("1.5")

        session = self._apply_session()
        result = session.submit(
            "data out; set inp; val = apply('decimal_target', amount); x = val + 1; y = not val; run;",
            functions={"decimal_target": decimal_target},
        )
        assert result.success is True, result.format_log()
        assert result.datasets["out"].to_pylist() == [
            {"amount": 10, "val": Decimal("1.5"), "x": 2.5, "y": False}
        ]

    def test_apply_decimal_zero_boolean_coercion(self) -> None:
        # Decimal("0") is a valid Decimal: it converts to false in a boolean
        # context (so ``not`` yields true) instead of failing closed, matching
        # the native valid-Decimal contract.
        def zero_decimal_target(value: object) -> object:
            return Decimal("0")

        session = self._apply_session()
        result = session.submit(
            "data out; set inp; val = apply('zero_decimal_target', amount); x = not val; y = val > 0; run;",
            functions={"zero_decimal_target": zero_decimal_target},
        )
        assert result.success is True, result.format_log()
        assert result.datasets["out"].to_pylist() == [
            {"amount": 10, "val": Decimal("0"), "x": True, "y": False}
        ]

    def test_invalid_decimal_numeric_conversion_fails_closed_with_value_and_type(self) -> None:
        # Decimal('sNaN') is a valid Python Decimal whose scalar text form is
        # not f64-parseable. The numeric consumer must fail closed carrying the
        # value and the requested numeric type — never convert it to 0, and the
        # failed block must not publish a target.
        def snan_target(value: object) -> object:
            return Decimal("sNaN")

        session = self._apply_session()
        result = session.submit(
            "data out; set inp; val = apply('snan_target', amount); x = val + 1; run;",
            functions={"snan_target": snan_target},
        )

        assert result.success is False
        assert "out" not in result.datasets
        assert any(
            entry.severity == "error"
            and entry.code == "RUNTIME_EXPRESSION_EVALUATION_ERROR"
            and "numeric value required" in entry.message
            and "sNaN" in entry.message
            for entry in result.log
        )

    def test_invalid_decimal_boolean_conversion_fails_closed_with_value_and_type(self) -> None:
        # The boolean consumer must fail closed carrying the value and the
        # requested boolean type — never silently fold to false.
        def snan_target(value: object) -> object:
            return Decimal("sNaN")

        session = self._apply_session()
        result = session.submit(
            "data out; set inp; val = apply('snan_target', amount); x = not val; run;",
            functions={"snan_target": snan_target},
        )

        assert result.success is False
        assert "out" not in result.datasets
        assert any(
            entry.severity == "error"
            and entry.code == "RUNTIME_EXPRESSION_EVALUATION_ERROR"
            and "boolean value required" in entry.message
            and "sNaN" in entry.message
            for entry in result.log
        )


class TestSumStatementStateContract:
    """The SUM statement state contract via Session.submit.

    Mirrors the native statement-path tests in
    ``native/limulus_native/src/runtime/mod.rs`` on the public path, multi-row:

    - an absent target starts from the 0 base and later rows reuse the prior
      accumulator, so ``seq + 1;`` produces 1, 2, 3;
    - an explicit NULL target resets the base to 0 without reviving the prior
      accumulator;
    - a NULL increment (the standalone ``.`` missing literal) adds 0 and keeps
      the base unchanged;
    - ``SEQ``/``seq`` spellings share one canonical case-insensitive
      accumulator;
    - top-level, IF／ELSE inline, IF-then-DO interior, and DO loop contexts
      apply the same update rule;
    - a non-numeric target value or increment fails closed identifying the
      target variable, without a partial publish;
    - SUM keeps its own 0-initialization/NULL rules while normal assignment
      arithmetic ``seq = seq + 1;`` propagates NULL for the same variable.

    The NULL-rule cases use a source ``seq`` column because public rows always
    carry every input column; the absent-target cases use an input without
    ``seq`` so the Absent classification is exercised.
    """

    @staticmethod
    def _submit(table: pa.Table, statements: str):
        session = Session()
        session.load("inp", table)
        return session.submit(f"data out; set inp; {statements} run;")

    @classmethod
    def _rows(cls, table: pa.Table, statements: str) -> list[dict[str, object]]:
        result = cls._submit(table, statements)
        assert result.success is True, result.format_log()
        return result.datasets["out"].to_pylist()

    def test_absent_target_starts_from_zero_and_uses_prior_total(self) -> None:
        # Mirrors first_sum_value_starts_from_zero_and_accumulates_absent_rows:
        # the first SUM value starts from the 0 base (1), and every later row
        # reuses the prior accumulator when the target is absent from the row.
        rows = self._rows(pa.table({"v": [1, 1, 1]}), "seq + 1;")
        assert rows == [
            {"v": 1, "seq": 1.0},
            {"v": 1, "seq": 2.0},
            {"v": 1, "seq": 3.0},
        ]

    def test_explicit_null_target_resets_mid_sequence_without_revival(self) -> None:
        # Mirrors sum_explicit_null_row_resets_sequence_without_revival_via_statement_path:
        # the explicit NULL row resets the base to 0 (value 1) instead of
        # reviving the accumulated total (which would give 4.0 on that row),
        # and the following row keeps its own base.
        rows = self._rows(
            pa.table({"seq": pa.array([1.0, 2.0, None, 2.0], type=pa.float64())}),
            "seq + 1;",
        )
        assert rows == [{"seq": 2.0}, {"seq": 3.0}, {"seq": 1.0}, {"seq": 3.0}]

    def test_null_increment_adds_zero_without_changing_base(self) -> None:
        # Mirrors sum_null_increment_keeps_base_via_statement_path: the
        # standalone ``.`` missing literal classifies as a NULL increment that
        # adds 0, so the base stays 3.0 on every row.
        rows = self._rows(
            pa.table({"seq": pa.array([3.0, 3.0], type=pa.float64())}),
            "seq + .;",
        )
        assert rows == [{"seq": 3.0}, {"seq": 3.0}]

    def test_case_variations_share_one_canonical_accumulator(self) -> None:
        # Mirrors sum_case_variations_share_one_accumulator_via_statement_paths:
        # an upper-case statement spelling resolves the same canonical state as
        # the lowercase source column ([3, 4, 5], not a fresh accumulator per
        # row), and two spellings inside one data step accumulate onto the value
        # the first spelling stored.
        rows = self._rows(
            pa.table({"seq": pa.array([2.0, 3.0, 4.0], type=pa.float64())}),
            "SEQ + 1;",
        )
        assert rows == [{"seq": 3.0}, {"seq": 4.0}, {"seq": 5.0}]

        rows = self._rows(
            pa.table({"seq": pa.array([1.0, 1.0, 1.0], type=pa.float64())}),
            "SEQ + 1; seq + 1;",
        )
        assert rows == [{"seq": 3.0}, {"seq": 3.0}, {"seq": 3.0}]

    def test_statement_contexts_share_one_accumulation_sequence(self) -> None:
        # Mirrors the native context tests (top-level, IF/ELSE inline,
        # IF-then-DO interior, DO loop): each context executes the SUM once per
        # row and applies the same update rule, so all contexts produce the
        # same sequence. The single-iteration DO loop keeps one execution per
        # row; multi-iteration DO behavior is pinned by the
        # runtime_do_assign_sum case harness.
        contexts = [
            ("top_level", pa.table({"v": [1, 1, 1]}), "seq + 1;"),
            ("if_inline", pa.table({"flag": [True, True, True]}), "if flag then seq + 1;"),
            (
                "else_inline",
                pa.table({"flag": [False, False, False]}),
                "if flag then other = 1; else seq + 1;",
            ),
            (
                "if_then_do",
                pa.table({"flag": [True, True, True]}),
                "if flag then do; seq + 1; end;",
            ),
            (
                "do_loop",
                pa.table({"v": [1, 1, 1]}),
                "do i = 1 to 1; seq + 1; end;",
            ),
        ]
        for label, table, statements in contexts:
            rows = self._rows(table, statements)
            values = [row["seq"] for row in rows]
            assert values == [1.0, 2.0, 3.0], f"{label}: {rows}"

    def test_non_numeric_target_value_fails_closed_with_variable_and_cause(self) -> None:
        # Mirrors sum_non_numeric_target_value_fails_closed_without_silent_zero:
        # a text target value is never silently converted to 0; the diagnostic
        # identifies the SUM target variable, and the block publishes nothing.
        result = self._submit(pa.table({"seq": ["abc"]}), "seq + 1;")

        assert result.success is False
        assert "out" not in result.datasets
        assert any(
            entry.severity == "error"
            and entry.code == "RUNTIME_EXPRESSION_EVALUATION_ERROR"
            and "seq" in entry.message
            and "numeric value required" in entry.message
            for entry in result.log
        )

    def test_non_numeric_increment_fails_closed_with_variable_and_cause(self) -> None:
        # Mirrors sum_non_numeric_increment_fails_closed_with_target_identified:
        # a non-numeric increment from a text variable fails closed naming the
        # SUM target, and the block publishes nothing.
        result = self._submit(pa.table({"txt": ["xyz"]}), "seq + txt;")

        assert result.success is False
        assert "out" not in result.datasets
        assert any(
            entry.severity == "error"
            and entry.code == "RUNTIME_EXPRESSION_EVALUATION_ERROR"
            and "seq" in entry.message
            and "numeric increment" in entry.message
            for entry in result.log
        )

    def test_sum_and_assignment_rules_stay_separate_for_the_same_variable(self) -> None:
        # Mirrors sum_rules_stay_separate_from_assignment_null_propagation: for
        # the SAME variable ``seq``, the SUM statement owns the 0-initialization
        # rule (absent or explicit NULL base plus increment -> 1 on the first
        # row) while normal assignment arithmetic propagates NULL instead.
        seq_null = pa.table({"seq": pa.array([None], type=pa.float64())})
        assert self._rows(seq_null, "seq + 1;") == [{"seq": 1.0}]
        assert self._rows(seq_null, "seq = seq + 1;") == [{"seq": None}]

        seq_absent = pa.table({"v": [1]})
        assert self._rows(seq_absent, "seq + 1;") == [{"v": 1, "seq": 1.0}]
        assert self._rows(seq_absent, "seq = seq + 1;") == [{"v": 1, "seq": None}]


class TestTypedPreparationProvenanceOpacity:
    """Marker-like user locations change nothing.

    Prepared kind and internal variable names travel in the typed
    ``DataSetRef.preparation`` contract, so a user ``location`` that happens to
    contain ``#prepared_set_rows``, ``#prepared_merge_rows``, or ``|internal=``
    is opaque origin data and can never alter column projection, preparation
    kind derivation, or native handoff behavior.
    """

    def test_marker_like_location_preserves_columns_on_public_submit(self) -> None:
        session = Session()
        session._executor.register_tables(
            {
                "weird": DataSetRef(
                    kind="arrow_table",
                    location="dataset://weird#prepared_set_rows|internal=secret",
                    payload=pa.table(
                        {"secret": [1], "internal": [2], "keep": [3]},
                    ),
                )
            }
        )
        result = session.submit("data out; set weird; output out; run;")

        assert result.success is True, result.format_log()
        table = result.datasets["out"]
        assert table.column_names == ["secret", "internal", "keep"]
        assert table.to_pylist() == [{"secret": 1, "internal": 2, "keep": 3}]

    def test_marker_like_location_does_not_derive_prepared_kind(self) -> None:
        session = Session()
        session._executor.register_tables(
            {
                "weird": DataSetRef(
                    kind="arrow_table",
                    location="dataset://weird#prepared_merge_rows|internal=x",
                    payload=pa.table({"v": [1]}),
                )
            }
        )
        result = session.submit("data out; set weird(in=in_a); output out; run;")

        assert result.success is True, result.format_log()
        table = result.datasets["out"]
        # The in= internal variable is preparation-internal and is excluded
        # from the output; its name comes from the typed contract, never from
        # the marker-like location.
        assert table.column_names == ["v"]
        prepared = session._executor.last_prepared_execution_inputs
        assert prepared.preparation_kind == "prepared_set"
        assert prepared.inputs["weird"].preparation.kind == "prepared_set"
        assert prepared.inputs["weird"].preparation.internal_variable_names == ("in_a",)
        # The user location stays byte-for-byte opaque.
        assert prepared.inputs["weird"].location == "dataset://weird#prepared_merge_rows|internal=x"


class TestPreparationProvenanceHandoffRegression:
    """Typed preparation provenance handoff regressions.

    These are the positive end-to-end checks for the silent data loss that the
    marker-encoded provenance era reproduced:

    - a marker-like user location plus a same-named user column keeps every
      column through the public MERGE path (the SET case lives in
      ``TestTypedPreparationProvenanceOpacity``);
    - a prepared SET/MERGE excludes only internal helper columns while the
      user location, ``DataSetRef.metadata``, and non-evaluated nested Arrow
      payload survive to the final public output;
    - a user carrier whose typed preparation kind conflicts with the prepared
      carriers fails closed with ``RUNTIME_PREPARATION_PROVENANCE_CONFLICT``
      and is never re-inferred from its location text.
    """

    @staticmethod
    def _register(session: Session, datasets: dict[str, DataSetRef]) -> None:
        session._executor.register_tables(datasets)

    # --- (a) marker-like location through the public MERGE path ------------

    def test_marker_like_location_keeps_columns_on_public_merge(self) -> None:
        session = Session()
        self._register(
            session,
            {
                "l": DataSetRef(
                    kind="arrow_table",
                    location="dataset://l#prepared_merge_rows|internal=secret",
                    payload=pa.table(
                        {"id": [1], "secret": [10], "keep_l": [100]},
                    ),
                ),
                "r": DataSetRef(
                    kind="arrow_table",
                    location="dataset://r",
                    payload=pa.table({"id": [1], "value_r": [7]}),
                ),
            },
        )
        result = session.submit("data out; merge l r; by id; output out; run;")

        assert result.success is True, result.format_log()
        table = result.datasets["out"]
        # Only the internal FIRST./LAST. helpers are excluded; the user column
        # whose name is encoded in the marker-like location survives.
        assert table.column_names == ["id", "secret", "keep_l", "value_r"]
        assert table.to_pylist() == [{"id": 1, "secret": 10, "keep_l": 100, "value_r": 7}]

        prepared = session._executor.last_prepared_execution_inputs
        assert prepared.preparation_kind == "prepared_merge"
        carrier = prepared.inputs["l"]
        assert carrier.location == "dataset://l#prepared_merge_rows|internal=secret"
        assert carrier.preparation.kind == "prepared_merge"
        assert carrier.preparation.internal_variable_names == (
            "FIRST.id",
            "LAST.id",
            "first.id",
            "last.id",
        )

    # --- (b) prepared public paths preserve location/metadata/nested payload

    def test_prepared_set_public_path_preserves_metadata_and_nested_payload(self) -> None:
        session = Session()
        self._register(
            session,
            {
                "in": DataSetRef(
                    kind="arrow_table",
                    location="dataset://prepared/in",
                    payload=pa.table(
                        {
                            "id": [2, 1],
                            "amount": [10, 20],
                            "tags": pa.array(
                                [["a"], ["b", "c"]], type=pa.list_(pa.string())
                            ),
                            "info": pa.array(
                                [{"n": 1}, {"n": 2}],
                                type=pa.struct([pa.field("n", pa.int64())]),
                            ),
                        }
                    ),
                    metadata={"origin": "unit"},
                )
            },
        )
        result = session.submit("data out; set in(in=in_a); by id; output out; run;")

        assert result.success is True, result.format_log()
        table = result.datasets["out"]
        # Only internal helper columns are excluded from the final output.
        assert table.column_names == ["id", "amount", "tags", "info"]
        assert table.to_pylist() == [
            {"id": 1, "amount": 20, "tags": ["b", "c"], "info": {"n": 2}},
            {"id": 2, "amount": 10, "tags": ["a"], "info": {"n": 1}},
        ]
        # Non-evaluated nested payload keeps value, type, and order end-to-end.
        assert table.schema.field("tags").type == pa.list_(pa.string())
        assert table.schema.field("info").type == pa.struct(
            [pa.field("n", pa.int64())]
        )

        prepared = session._executor.last_prepared_execution_inputs
        assert prepared.preparation_kind == "prepared_set"
        carrier = prepared.inputs["in"]
        assert carrier.location == "dataset://prepared/in"
        assert carrier.metadata == {"origin": "unit"}
        assert carrier.preparation.kind == "prepared_set"
        assert "in_a" in carrier.preparation.internal_variable_names

    def test_prepared_merge_public_path_preserves_metadata_and_nested_payload(self) -> None:
        session = Session()
        self._register(
            session,
            {
                "l": DataSetRef(
                    kind="arrow_table",
                    location="dataset://l",
                    payload=pa.table(
                        {
                            "id": [1],
                            "left_nested": pa.array(
                                [[1, 2]], type=pa.list_(pa.int64())
                            ),
                        }
                    ),
                    metadata={"origin": "left"},
                ),
                "r": DataSetRef(
                    kind="arrow_table",
                    location="dataset://r",
                    payload=pa.table(
                        {
                            "id": [1],
                            "right_struct": pa.array(
                                [{"label": "x"}],
                                type=pa.struct([pa.field("label", pa.string())]),
                            ),
                        }
                    ),
                ),
            },
        )
        result = session.submit("data out; merge l r; by id; output out; run;")

        assert result.success is True, result.format_log()
        table = result.datasets["out"]
        # Only the internal FIRST./LAST. helpers are excluded.
        assert table.column_names == ["id", "left_nested", "right_struct"]
        assert table.to_pylist() == [
            {"id": 1, "left_nested": [1, 2], "right_struct": {"label": "x"}}
        ]
        assert table.schema.field("left_nested").type == pa.list_(pa.int64())
        assert table.schema.field("right_struct").type == pa.struct(
            [pa.field("label", pa.string())]
        )

        prepared = session._executor.last_prepared_execution_inputs
        assert prepared.preparation_kind == "prepared_merge"
        carrier = prepared.inputs["l"]
        assert carrier.location == "dataset://l"
        assert carrier.metadata == {"origin": "left"}
        assert carrier.preparation.kind == "prepared_merge"

    # --- (c) public conflict rejection without location re-inference -------
    #
    # Session-level input resolution only resolves statement-referenced
    # sources, so a conflicting provenance-carrying carrier can only reach
    # the prepared-input aggregate through the public request path
    # (``ExecuteRequest.inputs``).  That public-path rejection is pinned in
    # ``TestTypedProvenanceMigration`` (test_input_preparation_service.py).


class TestByTieOrderContracts:
    """SET/MERGE BY ordering and FIRST./LAST. contracts.

    Within one BY key group a single SET keeps the source's original
    occurrence order, multi SET keeps the source declaration order plus each
    source's local occurrence order, and MERGE matches rows per occurrence
    ordinal (a short source keeps its group's last row). No new tie-break
    syntax is introduced. ``FIRST.<var>``/``LAST.<var>`` mark the first/last
    row of each prepared key group, judged on the combined prepared sequence
    for SET and the MERGE output sequence for MERGE. The Arrow preparation
    path and the row-carrier execution path must observe the same sequence.
    """

    TIED_SOURCE = pa.table({"k": [2, 1, 2, 1, 2], "v": ["a", "b", "c", "d", "e"]})
    MULTI_SOURCE_1 = pa.table({"k": [2, 1, 2], "v": ["a1", "b1", "c1"]})
    MULTI_SOURCE_2 = pa.table({"k": [2, 1, 2], "v": ["a2", "b2", "c2"]})
    MERGE_SOURCE_LEFT = pa.table({"k": [1, 1, 1, 2], "left": [10, 11, 12, 20]})
    MERGE_SOURCE_RIGHT = pa.table({"k": [1, 2, 2], "right": [100, 200, 201]})

    @staticmethod
    def _tied_session() -> Session:
        session = Session()
        session.load("src", TestByTieOrderContracts.TIED_SOURCE)
        return session

    @classmethod
    def _multi_session(cls) -> Session:
        session = Session()
        session.load("s1", cls.MULTI_SOURCE_1)
        session.load("s2", cls.MULTI_SOURCE_2)
        return session

    @classmethod
    def _merge_session(cls) -> Session:
        session = Session()
        session.load("l", cls.MERGE_SOURCE_LEFT)
        session.load("rr", cls.MERGE_SOURCE_RIGHT)
        return session

    @classmethod
    def _rows(cls, statements: str) -> list[dict[str, object]]:
        result = cls._tied_session().submit(f"data out; set src; {statements} run;")
        assert result.success is True, result.format_log()
        return result.datasets["out"].to_pylist()

    @classmethod
    def _multi_rows(cls, statements: str):
        result = cls._multi_session().submit(f"data out; set s1 s2; {statements} run;")
        assert result.success is True, result.format_log()
        return result.datasets["out"]

    @classmethod
    def _merge_rows(cls, statements: str):
        result = cls._merge_session().submit(f"data out; merge l rr; {statements} run;")
        assert result.success is True, result.format_log()
        return result.datasets["out"]

    def test_single_set_tied_by_keys_keep_source_local_occurrence_order(self) -> None:
        # Ascending key groups with the source's occurrence order preserved
        # inside each group; the k=2 tie values stay a, c, e (never re-sorted
        # alphabetically) because no tie-break syntax is introduced.
        rows = self._rows("by k;")

        assert rows == [
            {"k": 1, "v": "b"},
            {"k": 1, "v": "d"},
            {"k": 2, "v": "a"},
            {"k": 2, "v": "c"},
            {"k": 2, "v": "e"},
        ]
        assert [row["v"] for row in rows if row["k"] == 2] == ["a", "c", "e"]

    def test_first_last_follow_prepared_sequence(self) -> None:
        # FIRST./LAST. are judged on the prepared sequence: FIRST.k marks the
        # first row of each key group, LAST.k the last row, in the same
        # source-local tie order as above.
        rows = self._rows("by k; f = first.k; l = last.k;")

        assert rows == [
            {"k": 1, "v": "b", "f": True, "l": False},
            {"k": 1, "v": "d", "f": False, "l": True},
            {"k": 2, "v": "a", "f": True, "l": False},
            {"k": 2, "v": "c", "f": False, "l": False},
            {"k": 2, "v": "e", "f": False, "l": True},
        ]

    def test_row_carrier_by_path_matches_arrow_preparation_order(self) -> None:
        # The corresponding input preparation path (row memory carrier through
        # DataStepExecutor) must produce the same prepared sequence as the
        # Arrow path above.
        executor = DataStepExecutor(runtime_backend="rust")
        response = executor.execute(
            ExecuteRequest(
                dsl_text="data out; set src; by k; run;",
                inputs={
                    "src": DataSetRef(
                        kind="memory",
                        location="dataset://rows",
                        payload=[
                            {"k": 2, "v": "a"},
                            {"k": 1, "v": "b"},
                            {"k": 2, "v": "c"},
                            {"k": 1, "v": "d"},
                            {"k": 2, "v": "e"},
                        ],
                    )
                },
                output_targets=("out",),
            )
        )

        assert response.has_errors is False, response.diagnostics
        out = response.outputs["out"]
        # The memory row carrier keeps the row-list presentation end-to-end;
        # the BY preparation path must still deliver the pinned sequence.
        assert out.payload == [
            {"k": 1, "v": "b"},
            {"k": 1, "v": "d"},
            {"k": 2, "v": "a"},
            {"k": 2, "v": "c"},
            {"k": 2, "v": "e"},
        ]

    def test_multi_set_source_declaration_order_and_source_local_occurrences(self) -> None:
        # Within each key group s1's local occurrences come first in their
        # original order (declaration order), then s2's — not a concatenation
        # of individually sorted sources and not alphabetical within the
        # group.
        table = self._multi_rows("by k;")

        assert table.column_names == ["k", "v"]
        assert table.to_pylist() == [
            {"k": 1, "v": "b1"},
            {"k": 1, "v": "b2"},
            {"k": 2, "v": "a1"},
            {"k": 2, "v": "c1"},
            {"k": 2, "v": "a2"},
            {"k": 2, "v": "c2"},
        ]

    def test_multi_set_first_last_follow_combined_sequence_and_no_ordinal_leak(self) -> None:
        # FIRST./LAST. are judged on the combined prepared sequence (across
        # sources inside one key group), and the preparation-internal ordinal
        # helpers never leak into the public output schema.
        table = self._multi_rows("by k; f = first.k; l = last.k;")

        assert table.column_names == ["k", "v", "f", "l"]
        assert not any(
            name.startswith("__prepare") or "ordinal" in name.lower()
            for name in table.column_names
        )
        assert table.to_pylist() == [
            {"k": 1, "v": "b1", "f": True, "l": False},
            {"k": 1, "v": "b2", "f": False, "l": True},
            {"k": 2, "v": "a1", "f": True, "l": False},
            {"k": 2, "v": "c1", "f": False, "l": False},
            {"k": 2, "v": "a2", "f": False, "l": False},
            {"k": 2, "v": "c2", "f": False, "l": True},
        ]

    def test_merge_occurrence_alignment_keeps_short_source_last_row(self) -> None:
        # Rows are matched per BY key occurrence ordinal: within group k=1 the
        # three left rows (10, 11, 12) each pair with rr's single row 100, and
        # within group k=2 the two right rows (200, 201) each pair with l's
        # single row 20. A short source keeps its group's LAST row; this is
        # NOT a SET-style concatenation order.
        table = self._merge_rows("by k;")

        assert table.column_names == ["k", "left", "right"]
        assert table.to_pylist() == [
            {"k": 1, "left": 10, "right": 100},
            {"k": 1, "left": 11, "right": 100},
            {"k": 1, "left": 12, "right": 100},
            {"k": 2, "left": 20, "right": 200},
            {"k": 2, "left": 20, "right": 201},
        ]

    def test_merge_first_last_follow_merge_output_sequence(self) -> None:
        # FIRST./LAST. are judged on the MERGE output sequence (fl/ll avoid
        # colliding with the source name ``l``), and no preparation-internal
        # ordinal helpers leak into the public output schema.
        table = self._merge_rows("by k; fl = first.k; ll = last.k;")

        assert table.column_names == ["k", "left", "right", "fl", "ll"]
        assert not any(
            name.startswith("__prepare") or "ordinal" in name.lower()
            for name in table.column_names
        )
        assert table.to_pylist() == [
            {"k": 1, "left": 10, "right": 100, "fl": True, "ll": False},
            {"k": 1, "left": 11, "right": 100, "fl": False, "ll": False},
            {"k": 1, "left": 12, "right": 100, "fl": False, "ll": True},
            {"k": 2, "left": 20, "right": 200, "fl": True, "ll": False},
            {"k": 2, "left": 20, "right": 201, "fl": False, "ll": True},
        ]


class TestUnsupportedExpressionFailsClosedWithNativeDiagnostic:
    """Unsupported expression sources fail closed through the native owner.

    A native-unsupported source must reach the public envelope as the
    stage-aware native expression diagnostic and never materialize a Python
    object.  The success-contrast guards observe the same native-owned
    expression path within this module: standalone ``.`` in
    ``TestMissingLiteralStatementContexts``, ``**`` and builtin dispatch in
    ``TestNullArithmeticAndDecimalConversion`` and
    ``TestUnicodeScalarStringFunctions``, and the explicit ``apply(...)``
    host gateway in ``test_apply_decimal_checked_numeric_and_boolean_coercion``
    plus ``tests/test_apply_public_regression.py``.
    """

    def test_unsupported_list_literal_fails_closed_without_python_object(self) -> None:
        # The native lexer has no list-literal form: "[" at expression
        # position parses to "expression expected".  The failed native parse
        # must never be re-run through Python eval: the native parse cause
        # survives to the public log and no output target is published.
        result = _submit_result("data out; set inp; x = [1, 2, 3]; run;")

        assert result.success is False
        assert "out" not in result.datasets
        rendered = result.format_log()
        assert "expression expected" in rendered.lower()
        assert "python pow fallback" not in rendered.lower()
        assert any(
            entry.severity == "error"
            and entry.code == "RUNTIME_EXPRESSION_EVALUATION_ERROR"
            and entry.stage == "execute"
            for entry in result.log
        )


class TestRepeatedByWarningPresentation:
    @staticmethod
    def _merge_session(*, repeated: bool) -> Session:
        session = Session()
        session.load(
            "left",
            pa.table(
                {
                    "id": [1, 1] if repeated else [1],
                    "left_value": [10, 11] if repeated else [10],
                }
            ),
        )
        session.load(
            "right",
            pa.table(
                {
                    "id": [1, 1, 1] if repeated else [1],
                    "right_value": [20, 21, 22] if repeated else [20],
                }
            ),
        )
        return session

    def test_warning_keeps_success_output_and_is_printed(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = self._merge_session(repeated=True).submit(
            "data out; merge left right; by id; run;"
        )
        captured = capsys.readouterr()

        assert result.success is True
        assert result.datasets["out"].to_pylist() == [
            {"id": 1, "left_value": 10, "right_value": 20},
            {"id": 1, "left_value": 11, "right_value": 21},
            {"id": 1, "left_value": 11, "right_value": 22},
        ]
        warnings = [
            entry
            for entry in result.log
            if entry.code == "RUNTIME_MERGE_REPEATED_BY_VALUES"
        ]
        assert len(warnings) == 1
        assert warnings[0].severity == "warning"
        assert warnings[0].stage == "pre-evaluations"
        assert "Error entries:" not in result.format_log()
        assert "RUNTIME_MERGE_REPEATED_BY_VALUES" in captured.out

    def test_one_to_many_has_no_warning_and_keeps_occurrence_alignment(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        session = Session()
        session.load("left", pa.table({"id": [1], "left_value": [10]}))
        session.load(
            "right",
            pa.table({"id": [1, 1, 1], "right_value": [20, 21, 22]}),
        )

        result = session.submit("data out; merge left right; by id; run;")
        captured = capsys.readouterr()

        assert result.success is True
        assert result.datasets["out"].to_pylist() == [
            {"id": 1, "left_value": 10, "right_value": 20},
            {"id": 1, "left_value": 10, "right_value": 21},
            {"id": 1, "left_value": 10, "right_value": 22},
        ]
        assert not any(
            entry.code == "RUNTIME_MERGE_REPEATED_BY_VALUES"
            for entry in result.log
        )
        assert captured.out == ""

    def test_success_without_warning_stays_quiet(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = self._merge_session(repeated=False).submit(
            "data out; merge left right; by id; run;"
        )
        captured = capsys.readouterr()

        assert result.success is True
        assert result.datasets["out"].to_pylist() == [
            {"id": 1, "left_value": 10, "right_value": 20}
        ]
        assert result.log == ()
        assert captured.out == ""


class TestDefaultRuntimeBackendSelection:
    """An unspecified runtime backend defaults to ``auto`` and selects the Rust row engine.

    A defensive pipeline request without an explicit preference also defaults
    to ``auto``, and the explicit legacy ``python`` label keeps choosing the
    same Rust row strategy as a compatibility alias.
    """
    @staticmethod
    def _request() -> ExecuteRequest:
        return ExecuteRequest(
            dsl_text="data out; set inp; run;",
            inputs={
                "inp": DataSetRef(
                    kind="arrow_table",
                    location="dataset://inp",
                    payload=pa.table({"id": [1]}),
                )
            },
            output_targets=("out",),
        )

    def test_public_executor_default_is_auto_and_selects_rust_row(self) -> None:
        executor = DataStepExecutor()

        response = executor.execute(self._request())

        assert response.has_errors is False
        assert executor._runtime_backend_preference == "auto"
        assert executor.last_runtime_backend == "rust"
        assert executor.last_strategy_decision is not None
        assert executor.last_strategy_decision.requested_mode == "auto"
        assert executor.last_strategy_decision.strategy == "row"
        assert executor.last_strategy_decision.engine == "rust"
        assert executor.last_strategy_decision.reason_codes == ("ROW_ONLY_STRATEGY",)

    def test_defensive_pipeline_request_default_is_auto(self) -> None:
        request = PipelineExecutionRequest(request=self._request())

        assert request.runtime_backend_preference == "auto"

    def test_explicit_legacy_python_label_still_selects_rust_row(self) -> None:
        executor = DataStepExecutor(runtime_backend="python")

        response = executor.execute(self._request())

        assert response.has_errors is False
        assert executor.last_runtime_backend == "rust"
        assert executor.last_strategy_decision is not None
        assert executor.last_strategy_decision.requested_mode == "python"
        assert executor.last_strategy_decision.strategy == "row"
        assert executor.last_strategy_decision.engine == "rust"
        assert executor.last_strategy_decision.reason_codes == (
            "LEGACY_PYTHON_BACKEND_ALIAS",
        )
