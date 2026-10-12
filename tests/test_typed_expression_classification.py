"""Typed expression classification contracts."""

from __future__ import annotations

import ast as ast_module
import builtins
import importlib
import inspect
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

from limulus.execution.statement_expressions import (
    block_expression_slots,
    statement_expression_slots,
)
from limulus.expression_semantics import ExpressionSemantics
from limulus.syntax import compat as syntax_compat
from limulus.syntax import parse_program
from limulus.syntax.expressions import parse_general_expression
from limulus.syntax.models import (
    CallSyntax,
    DeferredExpressionError,
    DeferredStatementError,
    DoStatementSyntax,
    ExpressionSyntax,
    IfStatementSyntax,
    LiteralSyntax,
    NameSyntax,
    SourceDocument,
    SourceSpan,
    StatementSyntax,
)

_SEMANTICS = ExpressionSemantics()
REPO_ROOT = Path(__file__).resolve().parents[1]


def _typed_expression(text: str) -> ExpressionSyntax:
    source = SourceDocument(text)
    result = parse_general_expression(source, SourceSpan(0, len(text), 1, 1))
    return result.expression


def _legacy_function_names(text: str) -> tuple[str, ...]:
    return _SEMANTICS.function_names(text)


def _typed_function_names(text: str) -> tuple[str, ...]:
    return _SEMANTICS.function_names_typed(_typed_expression(text))


def _statements(dsl_text: str) -> tuple[StatementSyntax, ...]:
    parsed = parse_program(dsl_text)
    assert parsed.program is not None
    assert parsed.has_errors is False
    return parsed.program.blocks[0].statements


# --------------------------------------------------------------------------
# central contract: typed classification mirrors the legacy AST classification
# --------------------------------------------------------------------------


def test_typed_function_classification_matches_legacy_ast_classification() -> None:
    parity = {
        "f(a, g(b))": ("f", "g"),
        "x + y": (),
        "catx('-', upcase(name), 'find(value)')": ("catx", "upcase"),
        # ast.walk is breadth-first: outer-first, siblings before grandchildren
        "f(g(1)) + h(2)": ("f", "h", "g"),
        "put(x, yymmdd10.)": ("put",),
        "cats('a', x)": ("cats",),
        # Power classification is identical; evaluation rules stay native
        "a ** 2": (),
        "-x": (),
        "not (a and b)": (),
        "arr(i) + x[1]": ("arr",),
        # one structural concat node lowers to the helper call
        "upper(a) || b": ("__dsl_concat__", "upper"),
    }
    for text, expected in parity.items():
        assert _legacy_function_names(text) == expected, text
        assert _typed_function_names(text) == expected, text


def test_typed_call_order_stays_breadth_first_not_depth_first() -> None:
    text = "f(g(1)) + h(2)"
    assert _legacy_function_names(text) == ("f", "h", "g")
    assert _typed_function_names(text) == _legacy_function_names(text)
    # guards against silently switching the walk to depth-first (f, g, h)
    assert _typed_function_names(text) != ("f", "g", "h")


def test_typed_column_names_exclude_call_and_format_names() -> None:
    assert _SEMANTICS.column_names_typed(
        _typed_expression("arr(i) + x[1]")
    ) == ("i", "x")
    assert _SEMANTICS.column_names_typed(_typed_expression("x + y")) == ("x", "y")
    # the call name is not counted as a column name
    assert "arr" not in _SEMANTICS.column_names_typed(_typed_expression("arr(i)"))
    # the put/input format literal is not a column, only its other arguments are
    assert _SEMANTICS.column_names_typed(
        _typed_expression("put(x, comma8.)")
    ) == ("x",)


def test_typed_find_unsupported_function_matches_legacy() -> None:
    supported = ("upcase",)
    text = "upcase(x) + unknown_fn(y)"

    legacy = _SEMANTICS.find_unsupported_function(text, supported_functions=supported)
    typed = _SEMANTICS.find_unsupported_function_typed(
        _typed_expression(text), supported_functions=("UPCASE",)
    )
    assert legacy == typed == "unknown_fn"

    legacy_none = _SEMANTICS.find_unsupported_function(
        "upcase(x) + unknown_fn(y)", supported_functions=("UPCASE", "UNKNOWN_FN")
    )
    typed_none = _SEMANTICS.find_unsupported_function_typed(
        _typed_expression("upcase(x) + unknown_fn(y)"),
        supported_functions=("upcase", "unknown_fn"),
    )
    assert legacy_none is None and typed_none is None

    legacy_concat = _SEMANTICS.find_unsupported_function("a||b", supported_functions=())
    typed_concat = _SEMANTICS.find_unsupported_function_typed(
        _typed_expression("a||b"), supported_functions=()
    )
    assert legacy_concat == typed_concat == "__dsl_concat__"


def test_typed_concat_matches_lowering_left_fold_helper_count_and_order() -> None:
    """Concat classifies as the same left-folded helper-call chain lowering
    builds (one ``__dsl_concat__`` per fold); parts are walked without
    re-parsing and the original tree stays untouched."""

    assert _typed_function_names("a || b") == ("__dsl_concat__",)
    assert _typed_function_names("a || b || c") == ("__dsl_concat__", "__dsl_concat__")
    assert _legacy_function_names("a || b || c") == ("__dsl_concat__", "__dsl_concat__")

    chain_text = "outer(inner(a)) || m1(x) || m2(y)"
    folded_order = (
        "__dsl_concat__",
        "__dsl_concat__",
        "m2",
        "outer",
        "m1",
        "inner",
    )
    assert _legacy_function_names(chain_text) == folded_order
    assert _typed_function_names(chain_text) == folded_order
    supported = ("__dsl_concat__", "outer", "m1", "inner")
    assert (
        _SEMANTICS.find_unsupported_function(chain_text, supported_functions=supported)
        == "m2"
    )
    assert (
        _SEMANTICS.find_unsupported_function_typed(
            _typed_expression(chain_text), supported_functions=supported
        )
        == "m2"
    )

    nested_text = "a || (b || c)"
    assert _legacy_function_names(nested_text) == (
        "__dsl_concat__",
        "__dsl_concat__",
    )
    assert _typed_function_names(nested_text) == (
        "__dsl_concat__",
        "__dsl_concat__",
    )
    assert _SEMANTICS.column_names_typed(_typed_expression(nested_text)) == (
        "a",
        "b",
        "c",
    )


def test_whole_deferred_classification_is_canonical_and_empty() -> None:
    """Canonical policy: the new frontend structure is the
    classification authority.  A whole-expression Deferred rejects the shape
    the native grammar does not accept, so it classifies as empty structure
    even when the legacy Python AST could parse the same text; that is a
    deliberate difference, not a legacy-fact generalization."""

    for text in ("[sqrt(id)]", "[sqrt(id)] + 1"):
        expression = _typed_expression(text)
        assert isinstance(expression, DeferredExpressionError), text
        assert _SEMANTICS.function_names_typed(expression) == (), text
        assert _SEMANTICS.column_names_typed(expression) == (), text
        assert _SEMANTICS.contains_function_call_typed(expression) is False
        assert (
            _SEMANTICS.find_unsupported_function_typed(
                expression, supported_functions=()
            )
            is None
        )
        # documented difference: the old Python AST classified the very shape
        # the native grammar refuses
        assert _legacy_function_names(text) == ("sqrt",), text

    # the apply inventory on the deferred descriptor stays intact and is never
    # consumed as a classification substitute
    deferred_apply = _typed_expression("[apply('h', a)]")
    assert isinstance(deferred_apply, DeferredExpressionError)
    assert [reference.target for reference in deferred_apply.apply_references] == ["h"]
    assert _SEMANTICS.function_names_typed(deferred_apply) == ()

    # no fabricated partial payload: the classifier receives only the
    # deferred descriptor (whose fields hold no nested expression structure)
    # and reports the same empty classification; reparse guards stay in
    # test_typed_classification_avoids_ast_compile_and_legacy_cache_entries
    deferred_identity = _typed_expression("[sqrt(id)] + 1")
    assert isinstance(deferred_identity, DeferredExpressionError)
    assert deferred_identity.apply_references == ()

    # for inputs both grammars refuse, the empty classification coincides —
    # coincidence documented, not a parity guarantee
    for text in ("x +", "1 2", "left >< right"):
        expression = _typed_expression(text)
        assert isinstance(expression, DeferredExpressionError), text
        assert _SEMANTICS.function_names_typed(expression) == (), text
        assert _legacy_function_names(text) == (), text


def test_legacy_lenient_empty_operand_stays_a_difference() -> None:
    """The legacy string normalizer completed a missing concat operand with ''
    and accepted ``a ||``; the new frontend structure is canonical and keeps
    the whole-expression Deferred with the empty classification — no lenient
    lexical rewriting is restored, no early rejection is added."""

    assert _legacy_function_names("a ||") == ("__dsl_concat__",)
    expression = _typed_expression("a ||")
    assert isinstance(expression, DeferredExpressionError)
    assert _typed_function_names("a ||") == ()
    assert _SEMANTICS.column_names_typed(expression) == ()


def test_dotted_call_names_classify_as_their_own_name_policy() -> None:
    """Policy case: the new structure classifies ``a.b.c.f`` under its own
    dotted name.  Dotted-name existence and executability belong to the
    runtime binding owner; the classification tuple alone adds no early
    rejection (the legacy direct-Name limit is not reproduced)."""

    expression = _typed_expression("a.b.c.f(x)")
    assert isinstance(expression, CallSyntax)
    assert _typed_function_names("a.b.c.f(x)") == ("a.b.c.f",)
    assert _SEMANTICS.column_names_typed(expression) == ("x",)
    # documented policy difference: the legacy direct-Name classification saw
    # no function name for the attribute call
    assert _legacy_function_names("a.b.c.f(x)") == ()


def test_brace_and_bracket_subscript_are_typed_equivalent_canonical() -> None:
    """Policy case: the native grammar accepts ``f(x{g(id)})`` and
    ``f(x[g(id)])`` as the same typed shape, so both classify ``f, g``.  The
    legacy Python AST only classified the bracket spelling."""

    for text in ("f(x{g(id)})", "f(x[g(id)])"):
        assert _typed_function_names(text) == ("f", "g"), text
    assert _legacy_function_names("f(x[g(id)])") == ("f", "g")
    # documented policy difference: the old Python AST refused the brace form
    assert _legacy_function_names("f(x{g(id)})") == ()


def test_new_native_accepted_literals_keep_their_calls() -> None:
    """Policy case: for shapes the new native accepts but the old Python AST
    could not classify, the call itself is never dropped."""

    assert _typed_function_names("f(01)") == ("f",)
    numeric_expression = _typed_expression("f(01)")
    assert isinstance(numeric_expression, CallSyntax)
    assert isinstance(numeric_expression.arguments[0], LiteralSyntax)
    assert _typed_function_names("f($x)") == ("f",)
    assert _legacy_function_names("f(01)") == ()
    assert _legacy_function_names("f($x)") == ()


# --------------------------------------------------------------------------
# typed statement → expression slots (no reparse, no display info, no apply)
# --------------------------------------------------------------------------


def test_assignment_slots_cover_array_target_index_and_value() -> None:
    slots = statement_expression_slots(
        _statements("data out; set s; arr[i] = 1; run;")[2]
    )
    assert [(slot.role, type(slot.expression).__name__) for slot in slots] == [
        ("assign_target_index", "NameSyntax"),
        ("assign_value", "LiteralSyntax"),
    ]
    assert isinstance(slots[0].expression, NameSyntax)
    assert slots[0].expression.name == "i"
    assert _SEMANTICS.function_names_typed(slots[0].expression) == ()
    # name targets contribute no expression slot (the target is a defined column)
    plain = statement_expression_slots(
        _statements("data out; set s; y = arr(i) + 1; run;")[2]
    )
    assert [slot.role for slot in plain] == ["assign_value"]
    assert _SEMANTICS.function_names_typed(plain[0].expression) == ("arr",)


def test_sum_slot_classifies_increment_like_legacy_text() -> None:
    statement = _statements("data out; set s; t + u(v); run;")[2]
    slots = statement_expression_slots(statement)
    assert [slot.role for slot in slots] == ["sum_increment"]
    assert (
        _SEMANTICS.function_names_typed(slots[0].expression)
        == _legacy_function_names("t + u(v)")
        == ("u",)
    )
    assert _SEMANTICS.column_names_typed(slots[0].expression) == ("v",)


def test_if_else_slots_classify_condition_and_structured_actions() -> None:
    statements = _statements(
        "data out; set s; if f(x) then y = catx('-', a, b); else z = w(2); run;"
    )
    if_slots = statement_expression_slots(statements[2])
    assert [slot.role for slot in if_slots] == [
        "if_condition",
        "assign_value",
    ]
    condition, then_value = if_slots
    assert (
        _SEMANTICS.function_names_typed(condition.expression)
        == _legacy_function_names("f(x)")
        == ("f",)
    )
    assert _SEMANTICS.function_names_typed(then_value.expression) == ("catx",)

    else_slots = statement_expression_slots(statements[3])
    assert [slot.role for slot in else_slots] == ["assign_value"]
    assert _SEMANTICS.function_names_typed(else_slots[0].expression) == ("w",)


def test_if_deferred_action_is_not_classified_as_executable_statement() -> None:
    statement = _statements("data out; set s; if x then foo; run;")[2]
    slots = statement_expression_slots(statement)
    assert [slot.role for slot in slots] == ["if_condition"]
    payload = statement.payload
    assert isinstance(payload, IfStatementSyntax)
    assert isinstance(payload.then_action, DeferredStatementError)


def test_if_action_apply_inventory_stays_separate_from_classification() -> None:
    statement = _statements(
        "data out; set s; if x then y = apply('host', a); run;"
    )[2]
    slots = statement_expression_slots(statement)
    assert [slot.role for slot in slots] == ["if_condition", "assign_value"]
    # the outer inventory is untouched: it belongs to the resolve stage
    assert [reference.target for reference in statement.apply_references] == ["host"]
    typed_value = slots[1].expression
    # the structure still carries the apply call form; the classifier reads the
    # structure (never the inventory) and both agree with the legacy text
    assert (
        _SEMANTICS.function_names_typed(typed_value)
        == _legacy_function_names("apply('host', a)")
        == ("apply",)
    )
    assert _SEMANTICS.column_names_typed(typed_value) == ("a",)


def test_do_bound_slots_classify_start_and_end_with_legacy_tail_parity() -> None:
    statements = _statements("data out; set s; do i = f(1) to g(2); run;")
    slots = statement_expression_slots(statements[2])
    assert [slot.role for slot in slots] == ["do_start", "do_end"]
    assert _SEMANTICS.function_names_typed(slots[0].expression) == ("f",)
    assert _SEMANTICS.function_names_typed(slots[1].expression) == ("g",)
    do_payload = statements[2].payload
    assert isinstance(do_payload, DoStatementSyntax)
    assert do_payload.boundary_display_spans is not None

    by_typed = _statements("data out; set s; do i = f(1) to g(2) by h(3); run;")[2]
    by_slots = statement_expression_slots(by_typed)
    assert [slot.role for slot in by_slots] == ["do_start", "do_end"]
    # the BY tail stays in the deferred end shape; both legs classify as empty
    assert isinstance(by_slots[1].expression, DeferredExpressionError)
    assert _SEMANTICS.function_names_typed(by_slots[1].expression) == ()
    by_payload = by_typed.payload
    assert isinstance(by_payload, DoStatementSyntax)
    assert by_payload.loop_var == "i"


def test_where_slot_classifies_valid_and_deferred_conditions() -> None:
    statements = _statements("data out; set s; where f(x) and y; run;")
    slots = statement_expression_slots(statements[2])
    assert [slot.role for slot in slots] == ["where_condition"]
    assert (
        _SEMANTICS.function_names_typed(slots[0].expression)
        == _legacy_function_names("f(x) and y")
        == ("f",)
    )

    deferred_statements = _statements("data out; set s; where x + ; run;")
    deferred_slots = statement_expression_slots(deferred_statements[2])
    assert [slot.role for slot in deferred_slots] == ["where_condition"]
    assert isinstance(deferred_slots[0].expression, DeferredExpressionError)
    assert _SEMANTICS.function_names_typed(deferred_slots[0].expression) == ()


def test_block_slots_aggregate_statements_in_source_order() -> None:
    parsed = parse_program("data out; set s; do i = 1 to n; x = f(i); t + g(i); run;")
    assert parsed.program is not None
    slots = block_expression_slots(parsed.program.blocks[0])
    assert [slot.role for slot in slots] == [
        "do_start",
        "do_end",
        "assign_value",
        "sum_increment",
    ]


def test_non_expression_statements_yield_no_slots() -> None:
    parsed = parse_program("data out; set s; keep x; rename a = b; run;")
    assert parsed.program is not None
    assert block_expression_slots(parsed.program.blocks[0]) == ()


# --------------------------------------------------------------------------
# guards: no ast/compile/reparse/legacy cache/display projections reached
# --------------------------------------------------------------------------


def test_typed_classification_avoids_ast_compile_and_legacy_cache_entries() -> None:
    parsed_expressions = [
        _typed_expression("f(a, g(b))"),
        _typed_expression("x + y"),
        _typed_expression("put(x, comma8.)"),
        _typed_expression("arr(i) + x[1]"),
        _typed_expression("upper(a) || b"),
        _typed_expression("x +"),
    ]
    parsed_program = parse_program(
        "data out; set s; if f(x) then y = catx('-', a, b);"
        " do i = f(1) to g(2); arr[i] = 1; where x + ; run;"
    )
    assert parsed_program.program is not None

    def _forbidden(name: str):
        def _raise(*args, **kwargs):
            raise AssertionError(f"typed classification reached {name}")

        return _raise

    with (
        patch.object(builtins, "compile", _forbidden("compile")),
        patch.object(ast_module, "parse", _forbidden("ast.parse")),
        patch.object(
            ExpressionSemantics, "prepare_expression", _forbidden("normalize cache")
        ),
        patch.object(
            ExpressionSemantics, "normalize_expression", _forbidden("string normalizer")
        ),
        patch.object(
            ExpressionSemantics, "parse_expression", _forbidden("string ast parser")
        ),
        patch.object(
            ExpressionSemantics, "compile_expression", _forbidden("string compile")
        ),
        patch.object(syntax_compat, "project_result", _forbidden("legacy projection")),
        patch.object(syntax_compat, "project_program", _forbidden("legacy projection")),
        patch.object(syntax_compat, "project_block", _forbidden("legacy projection")),
        patch.object(syntax_compat, "project_statement", _forbidden("legacy projection")),
    ):
        for expression in parsed_expressions:
            _SEMANTICS.function_names_typed(expression)
            _SEMANTICS.column_names_typed(expression)
            _SEMANTICS.contains_function_call_typed(expression)
            _SEMANTICS.find_unsupported_function_typed(expression, supported_functions=())
        for statement in parsed_program.program.blocks[0].statements:
            statement_expression_slots(statement)


def test_typed_classification_keeps_the_column_api_string_entrance() -> None:
    legacy_before = _SEMANTICS.prepare_expression("catx('-', upcase(name)) ^='x'")
    compiled_before = _SEMANTICS.compile_expression("amount + 1")
    # exercising the typed entries must not disturb the string entries/cache
    _SEMANTICS.function_names_typed(_typed_expression("f(a) + g(b)"))
    _SEMANTICS.column_names_typed(_typed_expression("h(c)"))

    assert (
        _SEMANTICS.prepare_expression("catx('-', upcase(name)) ^='x'") == legacy_before
    )
    assert _SEMANTICS.compile_expression("amount + 1") is compiled_before
    assert _SEMANTICS.function_names("catx('-', upcase(name)) ^='x'") == (
        "catx",
        "upcase",
    )
    assert _SEMANTICS.contains_function_call("f(a)") is True


def test_classification_module_imports_stay_pure_and_light() -> None:
    _run_isolated(
        """
        import sys
        import limulus.expression_semantics
        from limulus.expression_semantics import ExpressionSemantics
        from limulus.syntax.models import SourceDocument, SourceSpan
        from limulus.syntax.expressions import parse_general_expression

        semantics = ExpressionSemantics()
        text = "f(a, g(b)) || c"
        result = parse_general_expression(
            SourceDocument(text), SourceSpan(0, len(text), 1, 1)
        )
        names = semantics.function_names_typed(result.expression)
        columns = semantics.column_names_typed(result.expression)
        assert names == ("__dsl_concat__", "f", "g"), names
        # column order is the ast.walk depth order: depth-1 `c` precedes the
        # depth-2 arguments under `f`, matching the folded legacy Name order
        assert columns == ("c", "a", "b"), columns

        for denied in ("pyarrow", "polars", "limulus_native", "limulus.session"):
            assert denied not in sys.modules, denied
        """
    )
    for module_name in (
        "limulus.expression_semantics",
        "limulus.execution.statement_expressions",
    ):
        module = importlib.import_module(module_name)
        source = inspect.getsource(module)
        tree = ast_module.parse(source)
        imported = {
            node.module
            for node in ast_module.walk(tree)
            if isinstance(node, ast_module.ImportFrom) and node.module is not None
        }
        imported.update(
            alias.name
            for node in ast_module.walk(tree)
            if isinstance(node, ast_module.Import)
            for alias in node.names
        )
        for module_import in imported:
            assert not module_import.startswith(
                ("limulus.runtime", "limulus.session")
            ), (module_name, module_import)
            assert module_import
            assert module_import not in {"pyarrow", "polars", "limulus_native"}, (
                module_name,
                module_import,
            )
    assert "limulus.execution.statement_expressions" in sys.modules


_HOOK = textwrap.dedent(
    """
    import sys

    _DENIED = ("pyarrow", "polars", "limulus_native", "limulus.session")


    class _DeniedDependencyGate:
        def find_spec(self, fullname, path=None, target=None):
            for root in _DENIED:
                if fullname == root or fullname.startswith(root + "."):
                    raise ImportError(
                        f"module {fullname!r} is denied by the isolation test hook"
                    )
            return None


    sys.meta_path.insert(0, _DeniedDependencyGate())
    """
)


def _run_isolated(snippet: str) -> None:
    script = _HOOK + "\n" + textwrap.dedent(snippet)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"isolated import failed:\n{result.stdout}\n{result.stderr}"
    )
