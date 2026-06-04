import unittest

from limulus.parser import ParserExecutionContext, ParserService, RustNativeParserBackend, SplitStageParserService
from limulus.session_parsing import classify_sql, parse_simple_filter


PARSER_SCENARIOS = {
    "lark_parser_initialization": {
        "overview": "Lark parser is initialized and parses a basic DATA/SET/RUN block",
        "dsl": "data out; set in; run;",
        "expected_kinds": ["DATA", "SET", "RUN"],
    },
}


NATIVE_PARSER_SCENARIOS = {
    "native_matches_python_subset": {
        "overview": "Rust parser backend matches Python parser for supported subset",
        "dsl": "data out; set in(keep=id amount where=(amount >= 0)); where amount >= 0; output out; run;",
    },
    "native_unsupported_keep": {
        "overview": "Rust parser backend returns capability diagnostic for unsupported KEEP",
        "dsl": "data out; set in; keep id; run;",
        "expected_code": "PARSE_BACKEND_CAPABILITY_MISSING",
    },
    "native_unsupported_merge": {
        "overview": "Rust parser backend returns capability diagnostic for unsupported MERGE",
        "dsl": "data out; set in; merge b; run;",
        "expected_code": "PARSE_BACKEND_CAPABILITY_MISSING",
    },
}


class ParserServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = ParserService()

    def test_lark_parser_is_initialized_and_used(self) -> None:
        scenario = PARSER_SCENARIOS["lark_parser_initialization"]
        self.assertIsNotNone(self.parser._lark_parser)

        result = self.parser.parse(scenario["dsl"])

        self.assertFalse(result.has_errors)
        self.assertEqual([statement.kind for statement in result.ast.statements], scenario["expected_kinds"])


class SplitStageParserServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = ParserService()

    def test_extracts_proc_and_macro_skip_regions_via_parser(self) -> None:
        parser = SplitStageParserService()
        dsl_text = """
        %let cutoff = 10;
        %macro noop();
        data hidden;
        set inp;
        run;
        %mend noop;
        proc sort data=inp out=sorted;
        by id;
        run;
        data out;
        set inp;
        value = "proc sort data=inp; quit; %macro noop();";
        run;
        """

        regions = parser.extract_statement_regions(dsl_text)

        self.assertIsNotNone(regions)
        assert regions is not None
        kinds = [region.kind for region in regions]
        self.assertIn("SKIP", kinds)
        self.assertEqual(kinds[-4:], ["DATA", "OTHER", "OTHER", "RUN"])

        skipped_texts = [dsl_text[region.start:region.end].strip() for region in regions if region.kind == "SKIP"]
        self.assertTrue(any(text.lower().startswith("%let") for text in skipped_texts))
        self.assertTrue(any(text.lower().startswith("%macro") for text in skipped_texts))
        self.assertTrue(any(text.lower().startswith("proc sort") for text in skipped_texts))


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


class ParserNativeIntegrationTests(unittest.TestCase):
    def test_native_matches_python_for_supported_subset(self) -> None:
        scenario = NATIVE_PARSER_SCENARIOS["native_matches_python_subset"]
        parser = ParserService()
        native = RustNativeParserBackend(parser)
        dsl = scenario["dsl"]

        python_result = parser.parse(dsl)
        rust_result = native.parse(ParserExecutionContext(dsl_text=dsl))

        self.assertFalse(python_result.has_errors)
        self.assertFalse(rust_result.has_errors)
        self.assertEqual(
            [statement.kind for statement in rust_result.ast.statements],
            [statement.kind for statement in python_result.ast.statements],
        )

    def test_native_returns_capability_diagnostics_for_unsupported_statement(self) -> None:
        scenario = NATIVE_PARSER_SCENARIOS["native_unsupported_keep"]
        parser = RustNativeParserBackend()

        result = parser.parse(ParserExecutionContext(dsl_text=scenario["dsl"]))

        self.assertTrue(result.has_errors)
        self.assertEqual(result.diagnostics[0].code, scenario["expected_code"])

    def test_rust_parser_returns_capability_diagnostic_for_out_of_subset(self) -> None:
        scenario = NATIVE_PARSER_SCENARIOS["native_unsupported_merge"]
        parser = RustNativeParserBackend()

        result = parser.parse(ParserExecutionContext(dsl_text=scenario["dsl"]))

        self.assertTrue(result.has_errors)
        self.assertEqual(result.diagnostics[0].code, scenario["expected_code"])


if __name__ == "__main__":
    unittest.main()
