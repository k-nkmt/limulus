import pytest

from limulus.session_parsing import classify_sql, parse_simple_filter


def test_parse_simple_filter_supports_quoted_strings_with_operators() -> None:
    result = parse_simple_filter("src", "status = '>= ready'")

    assert result.variable_name == "status"
    assert result.operator == "="
    assert result.scalar_value == ">= ready"


def test_parse_simple_filter_rejects_unquoted_string_literal() -> None:
    with pytest.raises(ValueError, match="SESSION_FILTER_PARSE_ERROR"):
        parse_simple_filter("src", "status = ready")


def test_classify_sql_identifies_create_table_and_rewrites_dictionary_reference() -> None:
    result = classify_sql("create table out as select * from dictionary.columns where memname = 'SRC'")

    assert result.kind == "create_table"
    assert result.target == "out"
    assert 'from "dictionary.columns"' in result.query


def test_classify_sql_identifies_drop_table_with_work_prefix() -> None:
    result = classify_sql("drop table work.out;")

    assert result.kind == "drop_table"
    assert result.target == "work.out"


def test_classify_sql_defaults_to_select_and_preserves_underscore_alias() -> None:
    result = classify_sql("select * from dictionary_columns order by memname")

    assert result.kind == "select"
    assert result.target is None
    assert result.query == "select * from dictionary_columns order by memname"


def test_classify_sql_rewrites_dictionary_reference_only_outside_string_literals() -> None:
    result = classify_sql("select 'dictionary.columns' as label from dictionary.columns")

    assert result.kind == "select"
    assert result.query == "select 'dictionary.columns' as label from \"dictionary.columns\""


def test_classify_sql_rejects_invalid_drop_table_form() -> None:
    with pytest.raises(ValueError, match="SESSION_SQL_CLASSIFICATION_ERROR"):
        classify_sql("drop table")


def test_classify_sql_rejects_invalid_create_table_form() -> None:
    with pytest.raises(ValueError, match="SESSION_SQL_CLASSIFICATION_ERROR"):
        classify_sql("create table out select * from src")