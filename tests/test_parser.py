import unittest

import limulus
import limulus.parser as parser_module
from limulus.parser import (
    LarkParserFrontend,
    ParserService,
)
from limulus.session_parsing import classify_sql, parse_simple_filter


PARSER_SCENARIOS = {
    "lark_parser_initialization": {
        "overview": "Lark parser is initialized and parses a basic DATA/SET/RUN block",
        "dsl": "data out; set in; run;",
        "expected_kinds": ["DATA", "SET", "RUN"],
    },
}


class ParserServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = ParserService()

    def test_parser_service_is_the_lark_frontend_alias(self) -> None:
        self.assertIs(ParserService, LarkParserFrontend)
        self.assertIs(type(self.parser), LarkParserFrontend)
        self.assertIs(limulus.ParserService, LarkParserFrontend)

    def test_lark_parser_is_initialized_and_used(self) -> None:
        scenario = PARSER_SCENARIOS["lark_parser_initialization"]
        # Lazy facade contract: construction builds no
        # grammar; the pure program frontend initializes on first parse.
        self.assertIsNone(getattr(self.parser, "_lark_parser", None))
        self.assertIsNotNone(self.parser._program_frontend)

        result = self.parser.parse(scenario["dsl"])

        self.assertFalse(result.has_errors)
        self.assertEqual([statement.kind for statement in result.ast.statements], scenario["expected_kinds"])

    def test_canonical_frontend_accepts_keep_and_merge_statements(self) -> None:
        keep_result = self.parser.parse("data out; set in; keep id; run;")
        merge_result = self.parser.parse("data out; merge left right; by id; run;")

        self.assertFalse(keep_result.has_errors)
        self.assertFalse(merge_result.has_errors)
        self.assertEqual(
            [statement.kind for statement in keep_result.ast.statements],
            ["DATA", "SET", "KEEP", "RUN"],
        )
        self.assertEqual(
            [statement.kind for statement in merge_result.ast.statements],
            ["DATA", "MERGE", "BY", "RUN"],
        )

    def test_parser_module_has_no_alternate_backend_symbols(self) -> None:
        for symbol in (
            "ParserBackend",
            "PythonParserBackend",
            "RustNativeParserBackend",
            "ParserBackendSelector",
            "ParserExecutionContext",
        ):
            self.assertFalse(hasattr(parser_module, symbol), symbol)

    def test_parse_preserves_ast_lowering_and_source_spans(self) -> None:
        result = self.parser.parse("data out;\nset in;\nrun;")

        self.assertFalse(result.has_errors)
        data_statement, set_statement, run_statement = result.ast.statements
        self.assertEqual(data_statement.output_refs[0].name, "out")
        self.assertEqual(set_statement.dataset_refs[0].name, "in")
        assert run_statement.span is not None
        self.assertEqual(
            [(statement.span.start, statement.span.end) for statement in result.ast.statements if statement.span],
            [(0, 8), (10, 16), (18, 21)],
        )
        self.assertEqual(run_statement.span.line, 3)

    def test_parse_preserves_diagnostic_code_severity_and_span(self) -> None:
        result = self.parser.parse("data out;\n???;\nrun;")

        self.assertTrue(result.has_errors)
        diagnostic = result.diagnostics[0]
        self.assertEqual(diagnostic.code, "PARSE_UNSUPPORTED_STATEMENT")
        self.assertEqual(diagnostic.severity, "error")
        self.assertEqual(diagnostic.message, "Unsupported statement syntax near line 2, column 1.")
        assert diagnostic.span is not None
        self.assertEqual(diagnostic.span.start, 10)
        self.assertEqual(diagnostic.span.end, 13)


class SessionParsingTests(unittest.TestCase):
    def test_parse_simple_filter_supports_quoted_strings_with_operators(self) -> None:
        result = parse_simple_filter("src", "status = '>= ready'")

        self.assertEqual(result.variable_name, "status")
        self.assertEqual(result.operator, "=")
        self.assertEqual(result.scalar_value, ">= ready")

    def test_parse_simple_filter_rejects_unquoted_string_literal(self) -> None:
        with self.assertRaisesRegex(ValueError, "SESSION_FILTER_PARSE_ERROR"):
            parse_simple_filter("src", "status = ready")

    def test_classify_sql_identifies_create_table_and_rewrites_dictionary_reference(self) -> None:
        result = classify_sql("create table out as select * from dictionary.columns where memname = 'SRC'")

        self.assertEqual(result.kind, "create_table")
        self.assertEqual(result.target, "out")
        self.assertIn('from "dictionary.columns"', result.query)

    def test_classify_sql_identifies_drop_table_with_work_prefix(self) -> None:
        result = classify_sql("drop table work.out;")

        self.assertEqual(result.kind, "drop_table")
        self.assertEqual(result.target, "work.out")

    def test_classify_sql_defaults_to_select_and_preserves_underscore_alias(self) -> None:
        result = classify_sql("select * from dictionary_columns order by memname")

        self.assertEqual(result.kind, "select")
        self.assertIsNone(result.target)
        self.assertEqual(result.query, "select * from dictionary_columns order by memname")

    def test_classify_sql_rewrites_dictionary_reference_only_outside_string_literals(self) -> None:
        result = classify_sql("select 'dictionary.columns' as label from dictionary.columns")

        self.assertEqual(result.kind, "select")
        self.assertEqual(result.query, "select 'dictionary.columns' as label from \"dictionary.columns\"")

    def test_classify_sql_rejects_invalid_drop_table_form(self) -> None:
        with self.assertRaisesRegex(ValueError, "SESSION_SQL_CLASSIFICATION_ERROR"):
            classify_sql("drop table")

    def test_classify_sql_rejects_invalid_create_table_form(self) -> None:
        with self.assertRaisesRegex(ValueError, "SESSION_SQL_CLASSIFICATION_ERROR"):
            classify_sql("create table out select * from src")


if __name__ == "__main__":
    unittest.main()
