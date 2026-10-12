from __future__ import annotations

from limulus.dataset_option_expressions import (
    WhereComparison,
    WhereLogicalExpression,
    WhereNullPredicate,
)
from limulus.parser import ParserService, ast_to_dict


def test_dataset_reference_where_preserves_nested_and_quoted_parentheses() -> None:
    result = ParserService().parse(
        'data out; set inp(keep=id amount drop=tmp '
        'where=((amount >= 20 and amount < 40) or name = "A) (B") '
        'rename=(amount=score) firstobs=2 obs=1); run;'
    )

    assert result.diagnostics == ()
    options = result.ast.statements[1].dataset_refs[0].options
    assert options.keep_vars == ("id", "amount")
    assert options.drop_vars == ("tmp",)
    assert options.where_expr == WhereLogicalExpression(
        "or",
        (
            WhereLogicalExpression(
                "and",
                (
                    WhereComparison("amount", ">=", 20),
                    WhereComparison("amount", "<", 40),
                ),
            ),
            WhereComparison("name", "=", "A) (B"),
        ),
    )
    assert options.rename_map == {"amount": "score"}
    assert options.firstobs == 2
    assert options.obs == 1


def test_dataset_reference_where_serializes_as_typed_ast() -> None:
    result = ParserService().parse("data out; set inp(where=(missing is not null)); run;")

    assert result.diagnostics == ()
    serialized = ast_to_dict(result.ast)
    assert serialized["statements"][1]["dataset_refs"][0]["options"]["where_expr"] == {
        "kind": "null",
        "column": "missing",
        "negated": True,
    }


def test_dataset_reference_where_keeps_quoted_semicolon_and_adjacent_options() -> None:
    result = ParserService().parse(
        "data out; set inp(where=(name = 'A;B') rename=(name=label) obs=2); run;"
    )

    assert result.diagnostics == ()
    options = result.ast.statements[1].dataset_refs[0].options
    assert options.where_expr == WhereComparison("name", "=", "A;B")
    assert options.rename_map == {"name": "label"}
    assert options.obs == 2


def test_dataset_reference_where_keeps_backslash_and_doubled_quote_escapes() -> None:
    backslash_result = ParserService().parse(
        r'''data out; set inp(where=(name = "A\";B") rename=(name=escaped) firstobs=3); run;'''
    )
    doubled_quote_result = ParserService().parse(
        '''data out; set inp(where=(name = 'A''B;C') rename=(name=doubled) obs=4); run;'''
    )

    assert backslash_result.diagnostics == ()
    backslash_options = backslash_result.ast.statements[1].dataset_refs[0].options
    assert backslash_options.where_expr == WhereComparison("name", "=", 'A";B')
    assert backslash_options.rename_map == {"name": "escaped"}
    assert backslash_options.firstobs == 3

    assert doubled_quote_result.diagnostics == ()
    doubled_options = doubled_quote_result.ast.statements[1].dataset_refs[0].options
    assert doubled_options.where_expr == WhereComparison("name", "=", "A'B;C")
    assert doubled_options.rename_map == {"name": "doubled"}
    assert doubled_options.obs == 4


def test_dataset_reference_where_rejects_unsupported_syntax_at_statement_boundary() -> None:
    for expression, reason in (
        ("func(amount) = 1", "function"),
        ("record.amount = 1", "attribute"),
        ("items[0] = 1", "subscript"),
        ("amount + 1 > 10", "arithmetic"),
        ("amount := 10", "assignment"),
        ("amount >", "incomplete"),
    ):
        result = ParserService().parse(f"data out; set inp(where=({expression})); run;")

        assert result.ast.statements == ()
        assert len(result.diagnostics) == 1
        diagnostic = result.diagnostics[0]
        assert diagnostic.code == "PARSE_DATASET_WHERE_UNSUPPORTED"
        assert diagnostic.location == "statement:2"
        assert "WHERE" in diagnostic.message
        assert reason in diagnostic.message.lower()
