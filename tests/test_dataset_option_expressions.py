from __future__ import annotations

import pytest
import pyarrow as pa

from limulus.dataset_option_expressions import (
    DatasetWhereParseResult,
    DatasetWhereParser,
    DatasetWhereEvaluationResult,
    WhereComparison,
    WhereLogicalExpression,
    WhereNotExpression,
    WhereNullPredicate,
    parse_dataset_option_where,
    evaluate_dataset_where,
)


@pytest.fixture()
def parser() -> DatasetWhereParser:
    return DatasetWhereParser()


def test_parser_builds_comparison_for_each_operator(parser: DatasetWhereParser) -> None:
    for operator in ("=", "==", "!=", ">", ">=", "<", "<="):
        result = parser.parse(f"amount {operator} 10")
        assert result.diagnostics == ()
        assert result.expression == WhereComparison("amount", operator, 10)


def test_parser_keeps_string_float_and_boolean_literal_types(parser: DatasetWhereParser) -> None:
    assert parser.parse("name = 'Alice'").expression == WhereComparison("name", "=", "Alice")
    assert parser.parse("ratio >= -1.25e2").expression == WhereComparison("ratio", ">=", -125.0)
    assert parser.parse("active == TRUE").expression == WhereComparison("active", "==", True)
    assert parser.parse('note != "O""Brien"').expression == WhereComparison("note", "!=", 'O"Brien')


def test_parser_preserves_boolean_precedence_and_parentheses(parser: DatasetWhereParser) -> None:
    result = parser.parse("not amount = 0 and score >= 10 or active = true")

    assert result.diagnostics == ()
    assert result.expression == WhereLogicalExpression(
        "or",
        (
            WhereLogicalExpression(
                "and",
                (
                    WhereNotExpression(WhereComparison("amount", "=", 0)),
                    WhereComparison("score", ">=", 10),
                ),
            ),
            WhereComparison("active", "=", True),
        ),
    )

    grouped = parser.parse("not (amount = 0 or score >= 10)")
    assert grouped.expression == WhereNotExpression(
        WhereLogicalExpression(
            "or",
            (
                WhereComparison("amount", "=", 0),
                WhereComparison("score", ">=", 10),
            ),
        )
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ("missing is null", WhereNullPredicate("missing", False)),
        ("missing IS NOT NULL", WhereNullPredicate("missing", True)),
    ),
)
def test_parser_builds_null_predicates(
    parser: DatasetWhereParser,
    source: str,
    expected: WhereNullPredicate,
) -> None:
    result = parser.parse(source)
    assert result.diagnostics == ()
    assert result.expression == expected


@pytest.mark.parametrize(
    "source",
    (
        "amount >",
        "(amount > 10",
        "amount > 10 and",
        "",
    ),
)
def test_parser_returns_incomplete_where_diagnostic(
    parser: DatasetWhereParser,
    source: str,
) -> None:
    result = parser.parse(source)
    assert result.expression is None
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "PARSE_DATASET_WHERE_UNSUPPORTED"
    assert "WHERE=" in diagnostic.message
    assert "incomplete" in diagnostic.message.lower()


@pytest.mark.parametrize(
    ("source", "reason"),
    (
        ("amount + 1 > 10", "arithmetic"),
        ("func(amount) = 1", "function"),
        ("record.amount = 1", "attribute"),
        ("items[0] = 1", "subscript"),
        ("amount := 10", "assignment"),
        ("amount = other_amount", "column"),
    ),
)
def test_parser_rejects_python_and_non_scalar_syntax(
    parser: DatasetWhereParser,
    source: str,
    reason: str,
) -> None:
    result = parser.parse(source)
    assert result.expression is None
    assert result.diagnostics
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "PARSE_DATASET_WHERE_UNSUPPORTED"
    assert "WHERE=" in diagnostic.message
    assert "unsupported" in diagnostic.message.lower()
    assert reason in diagnostic.message.lower()


def test_parser_accepts_optional_where_wrapper_and_convenience_function() -> None:
    result = parse_dataset_option_where("WHERE=(amount >= 20)")

    assert isinstance(result, DatasetWhereParseResult)
    assert result.diagnostics == ()
    assert result.expression == WhereComparison("amount", ">=", 20)


@pytest.fixture()
def evaluation_table() -> pa.Table:
    return pa.table(
        {
            "ID": pa.array([1, 2, 3, None], type=pa.int64()),
            "Name": pa.array(["Alice", "Bob", None, "Dora"], type=pa.string()),
            "Active": pa.array([True, False, None, True], type=pa.bool_()),
        }
    )


@pytest.mark.parametrize(
    ("operator", "expected"),
    (
        ("=", [False, True, False, False]),
        ("==", [False, True, False, False]),
        ("!=", [True, False, True, False]),
        (">", [False, False, True, False]),
        (">=", [False, True, True, False]),
        ("<", [True, False, False, False]),
        ("<=", [True, True, False, False]),
    ),
)
def test_evaluator_comparison_operators_return_arrow_boolean_mask(
    evaluation_table: pa.Table,
    operator: str,
    expected: list[bool | None],
) -> None:
    result = evaluate_dataset_where(
        WhereComparison("id", operator, 2),
        evaluation_table,
        source_name="source",
    )

    assert isinstance(result, DatasetWhereEvaluationResult)
    assert result.diagnostics == ()
    assert result.mask is not None
    assert result.mask.to_pylist() == expected


def test_evaluator_resolves_columns_case_insensitively_and_handles_scalar_types(
    evaluation_table: pa.Table,
) -> None:
    string_result = evaluate_dataset_where(
        WhereComparison("name", "=", "Bob"), evaluation_table, source_name="source"
    )
    bool_result = evaluate_dataset_where(
        WhereComparison("ACTIVE", "==", True), evaluation_table, source_name="source"
    )

    assert string_result.diagnostics == ()
    assert string_result.mask.to_pylist() == [False, True, False, False]
    assert bool_result.diagnostics == ()
    assert bool_result.mask.to_pylist() == [True, False, False, True]


def test_evaluator_null_predicates_are_explicit_boolean_masks(
    evaluation_table: pa.Table,
) -> None:
    is_null = evaluate_dataset_where(
        WhereNullPredicate("name", False), evaluation_table, source_name="source"
    )
    is_not_null = evaluate_dataset_where(
        WhereNullPredicate("NAME", True), evaluation_table, source_name="source"
    )

    assert is_null.diagnostics == ()
    assert is_null.mask.to_pylist() == [False, False, True, False]
    assert is_not_null.diagnostics == ()
    assert is_not_null.mask.to_pylist() == [True, True, False, True]


@pytest.mark.parametrize(
    ("nested_type", "values"),
    (
        (pa.list_(pa.int64()), [[1], None]),
        (pa.large_list(pa.int64()), [[1], None]),
        (pa.list_(pa.int64(), 1), [[1], None]),
        (pa.struct([pa.field("item", pa.int64())]), [{"item": 1}, None]),
    ),
)
def test_evaluator_rejects_null_predicates_for_nested_operands(
    nested_type: pa.DataType,
    values: list[object],
) -> None:
    table = pa.table({"nested": pa.array(values, type=nested_type)})

    for predicate in (
        WhereNullPredicate("nested", False),
        WhereNullPredicate("nested", True),
    ):
        result = evaluate_dataset_where(predicate, table, source_name="input")

        assert result.mask is None
        assert len(result.diagnostics) == 1
        diagnostic = result.diagnostics[0]
        assert diagnostic.code == "RUNTIME_DATASET_OPTION_INVALID"
        assert "input" in diagnostic.message
        assert "nested" in diagnostic.message
        assert "null" in diagnostic.message.lower()
        assert "nested" in diagnostic.message.lower()


def test_evaluator_reports_unknown_and_ambiguous_columns() -> None:
    table = pa.table({"Amount": [1, 2], "amount": [3, 4]})

    unknown = evaluate_dataset_where(
        WhereComparison("missing", "=", 1), table, source_name="input.csv"
    )
    ambiguous = evaluate_dataset_where(
        WhereComparison("AMOUNT", "=", 1), table, source_name="input.csv"
    )

    for result, reason, column in (
        (unknown, "not found", "missing"),
        (ambiguous, "ambiguous", "amount"),
    ):
        assert result.mask is None
        assert len(result.diagnostics) == 1
        diagnostic = result.diagnostics[0]
        assert diagnostic.code == "RUNTIME_DATASET_OPTION_INVALID"
        assert "input.csv" in diagnostic.message
        assert column in diagnostic.message.lower()
        assert reason in diagnostic.message.lower()


def test_evaluator_converts_kernel_type_errors_to_dataset_option_diagnostic(
    evaluation_table: pa.Table,
) -> None:
    result = evaluate_dataset_where(
        WhereComparison("name", ">", 10), evaluation_table, source_name="input.csv"
    )

    assert result.mask is None
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "RUNTIME_DATASET_OPTION_INVALID"
    assert "input.csv" in diagnostic.message
    assert "name" in diagnostic.message.lower()
    assert ">" in diagnostic.message
    assert "type" in diagnostic.message.lower() or "compare" in diagnostic.message.lower()


def test_evaluator_composes_nested_boolean_tree_with_kleene_kernels(
    evaluation_table: pa.Table,
) -> None:
    expression = WhereLogicalExpression(
        "or",
        (
            WhereComparison("id", "=", 2),
            WhereLogicalExpression(
                "and",
                (
                    WhereNullPredicate("name", True),
                    WhereNotExpression(WhereComparison("active", "=", False)),
                ),
            ),
        ),
    )

    result = evaluate_dataset_where(expression, evaluation_table, source_name="source")

    assert result.diagnostics == ()
    assert result.mask.to_pylist() == [True, True, False, True]


def test_evaluator_final_mask_fills_nullable_boolean_results_with_false(
    evaluation_table: pa.Table,
) -> None:
    expression = WhereLogicalExpression(
        "or",
        (
            WhereComparison("id", "=", 2),
            WhereComparison("active", "=", True),
        ),
    )

    result = evaluate_dataset_where(expression, evaluation_table, source_name="source")

    assert result.diagnostics == ()
    assert result.mask.to_pylist() == [True, True, False, True]


@pytest.mark.parametrize("operator", ("and", "or"))
def test_evaluator_rejects_logical_expression_without_two_operands(
    evaluation_table: pa.Table,
    operator: str,
) -> None:
    result = evaluate_dataset_where(
        WhereLogicalExpression(operator, (WhereComparison("id", "=", 1),)),
        evaluation_table,
        source_name="source",
    )

    assert result.mask is None
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].code == "RUNTIME_DATASET_OPTION_INVALID"
    assert "operand" in result.diagnostics[0].message.lower()
