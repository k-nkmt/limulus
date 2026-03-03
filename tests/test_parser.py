import unittest

from limulus.parser import ParserExecutionContext, ParserService, RustNativeParserBackend


PARSER_SCENARIOS = {
    "phase1_normalized_ast": {
        "overview": "Parses phase-1 statements and returns normalized AST kinds",
        "dsl": (
            "data out; "
            "set in; "
            "if amount > 0 then output out; "
            "else if amount = 0 then do; keep amount; end; "
            "else drop amount; "
            "where amount >= 0; "
            "run;"
        ),
        "expected_kinds": ["DATA", "SET", "IF", "ELSE IF", "KEEP", "END", "ELSE", "WHERE", "RUN"],
    },
    "invalid_syntax": {
        "overview": "Returns diagnostic details for invalid syntax",
        "dsl": "data out; invalid syntax; run;",
        "expected_code": "PARSE_UNSUPPORTED_STATEMENT",
        "expected_location": "statement:2",
    },
    "set_options_and_statement_options": {
        "overview": "Parses dataset IN= options and SET-level INDSNAME=/END= options",
        "dsl": "data out; set in_a(in=in_left) in_b(in=in_right) indsname=src end=last; run;",
    },
    "set_options_parser_based_interleaved": {
        "overview": "Parses interleaved dataset options and SET statement options via parser structure",
        "dsl": (
            "data out; "
            "set in_a(keep=id amount drop=tmp rename=(amount=amt) where=(amt > 0)) "
            "in_b(in=in_right keep=id) "
            "indsname=src end=eof ; "
            "run;"
        ),
    },
    "invalid_option_scope": {
        "overview": "Returns PARSE_SET_OPTION_SCOPE_ERROR when option scope is invalid",
        "dsl_in_statement_scope": "data out; set in_a in=in_left; run;",
        "dsl_indsname_dataset_scope": "data out; set in_a(indsname=src); run;",
    },
    "set_multiple_inputs_delete": {
        "overview": "Parses SET with multiple inputs and DELETE statement",
        "dsl": "data out; set a b c; if amount > 0 then output out; delete; run;",
        "expected_kinds": ["DATA", "SET", "IF", "DELETE", "RUN"],
        "expected_inputs": ("a", "b", "c"),
    },
    "dataset_options_where_rename_keep_drop": {
        "overview": "Parses KEEP/DROP/RENAME/WHERE dataset options",
        "dsl": "data out; set in(keep=id amount drop=tmp rename=(amount=amt) where=(amt > 0)); run;",
    },
    "data_output_dataset_options": {
        "overview": "Parses DATA statement output dataset options",
        "dsl": "data a(keep=id) b(drop=tmp rename=(name=full_name)); set in; run;",
    },
    "rename_statement": {
        "overview": "Parses RENAME statement mapping",
        "dsl": "data out; set in; rename amount=amt score=score_new; run;",
    },
    "rename_statement_with_spaces": {
        "overview": "Parses RENAME statement with whitespace around equals",
        "dsl": "data out; set in; rename id = id2 amount = amt; run;",
    },
    "case_insensitive_keywords": {
        "overview": "Parses keywords case-insensitively",
        "dsl": "DaTa Out; SeT Work.Input; WhErE amount > 0; OuTpUt OUT; RuN;",
        "expected_kinds": ["DATA", "SET", "WHERE", "OUTPUT", "RUN"],
    },
    "lark_parser_initialization": {
        "overview": "Lark parser is initialized and parses a basic DATA/SET/RUN block",
        "dsl": "data out; set in; run;",
        "expected_kinds": ["DATA", "SET", "RUN"],
    },
    "line_comment": {
        "overview": "Ignores line comments during parsing",
        "dsl": "* Comment; data out; set inp; run;",
    },
    "block_comment_inline": {
        "overview": "Ignores inline block comments during parsing",
        "dsl": "data out; /* block comment */ set inp; run;",
    },
    "block_comment_tail": {
        "overview": "Ignores tail block comments during parsing",
        "dsl": "data out; set inp; /* block comment */ run;",
    },
    "set_options_equals_whitespace": {
        "overview": "Parses SET options with flexible whitespace around equals",
        "dsl": (
            "data out; "
            "set inp end=eof; "
            "set inp end = eof; "
            "set inp end= eof; "
            "set inp end =eof; "
            "set inp indsname=src; "
            "set inp indsname = src; "
            "set inp(in=flag1); "
            "set inp(in = flag2); "
            "set inp(in= flag3); "
            "set inp(in =flag4); "
            "run;"
        ),
    },
    "unsupported_as_skipped": {
        "overview": "Parses unsupported statements as SKIPPED",
        "dsl": (
            "data out; "
            "length name 8; "
            "attrib amount length=8; "
            "format amount 8.2; "
            "label amount = \"Amount\"; "
            "informat amount 8.; "
            "set inp; "
            "run;"
        ),
    },
    "call_statement_as_skipped": {
        "overview": "Parses CALL statements as SKIPPED",
        "dsl": "data out; set inp; call missing(var1); run;",
    },
    "stop_statement": {
        "overview": "Parses STOP statement as executable STOP kind",
        "dsl": "data out; set in; stop; run;",
        "expected_kinds": ["DATA", "SET", "STOP", "RUN"],
    },
    "if_structured_spec": {
        "overview": "Parses structured IF metadata for subset IF and IF THEN action",
        "dsl": "data out; set in; if amount > 0; if amount = 0 then output out; run;",
    },
    "do_array_structured_spec": {
        "overview": "Parses structured DO/ARRAY metadata",
        "dsl": "data out; set in; array vars[*] a b c; do i = 1 to 3; output out; end; run;",
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

    def test_accepts_phase1_statements_and_returns_normalized_ast(self) -> None:
        scenario = PARSER_SCENARIOS["phase1_normalized_ast"]
        dsl_text = scenario["dsl"]

        result = self.parser.parse(dsl_text)

        self.assertFalse(result.has_errors)
        self.assertEqual(
            [statement.kind for statement in result.ast.statements],
            scenario["expected_kinds"],
        )

    def test_returns_diagnostics_with_kind_position_and_cause_on_invalid_syntax(self) -> None:
        scenario = PARSER_SCENARIOS["invalid_syntax"]
        dsl_text = scenario["dsl"]

        result = self.parser.parse(dsl_text)

        self.assertTrue(result.has_errors)
        self.assertEqual(result.diagnostics[0].code, scenario["expected_code"])
        self.assertEqual(result.diagnostics[0].severity, "error")
        self.assertEqual(result.diagnostics[0].location, scenario["expected_location"])
        self.assertTrue(
            "invalid syntax" in result.diagnostics[0].message
            or "near line" in result.diagnostics[0].message
        )

    def test_parses_set_dataset_ref_options_and_set_statement_options_with_scope_boundary(self) -> None:
        scenario = PARSER_SCENARIOS["set_options_and_statement_options"]
        dsl_text = scenario["dsl"]

        result = self.parser.parse(dsl_text)

        self.assertFalse(result.has_errors)
        set_statement = next(statement for statement in result.ast.statements if statement.kind == "SET")
        self.assertEqual(tuple(ref.name for ref in set_statement.dataset_refs), ("in_a", "in_b"))
        self.assertEqual(set_statement.dataset_refs[0].options.in_var, "in_left")
        self.assertEqual(set_statement.dataset_refs[1].options.in_var, "in_right")
        self.assertEqual(set_statement.statement_options.indsname_var, "src")
        self.assertEqual(set_statement.statement_options.end_var, "last")

    def test_parses_set_dataset_options_with_interleaved_statement_options(self) -> None:
        scenario = PARSER_SCENARIOS["set_options_parser_based_interleaved"]
        result = self.parser.parse(scenario["dsl"])

        self.assertFalse(result.has_errors)
        set_statement = next(statement for statement in result.ast.statements if statement.kind == "SET")
        self.assertEqual(set_statement.statement_options.end_var, "eof")
        self.assertEqual(set_statement.statement_options.indsname_var, "src")

        first_options = set_statement.dataset_refs[0].options
        self.assertEqual(first_options.keep_vars, ("id", "amount"))
        self.assertEqual(first_options.drop_vars, ("tmp",))
        self.assertEqual(first_options.rename_map, {"amount": "amt"})
        self.assertEqual(first_options.where_expr, "amt > 0")

        second_options = set_statement.dataset_refs[1].options
        self.assertEqual(second_options.in_var, "in_right")
        self.assertEqual(second_options.keep_vars, ("id",))

    def test_returns_parse_set_option_scope_error_for_invalid_option_scope(self) -> None:
        scenario = PARSER_SCENARIOS["invalid_option_scope"]
        with_in_as_statement_option = scenario["dsl_in_statement_scope"]
        with_indsname_as_dataset_option = scenario["dsl_indsname_dataset_scope"]

        result_in = self.parser.parse(with_in_as_statement_option)
        result_inds = self.parser.parse(with_indsname_as_dataset_option)

        self.assertTrue(result_in.has_errors)
        self.assertEqual(result_in.diagnostics[0].code, "PARSE_UNSUPPORTED_STATEMENT")
        self.assertTrue(result_inds.has_errors)
        self.assertEqual(result_inds.diagnostics[0].code, "PARSE_SET_OPTION_SCOPE_ERROR")
        self.assertIn("INDSNAME=", result_inds.diagnostics[0].message)

    def test_parses_set_multiple_inputs_and_delete_statement(self) -> None:
        scenario = PARSER_SCENARIOS["set_multiple_inputs_delete"]
        dsl_text = scenario["dsl"]

        result = self.parser.parse(dsl_text)

        self.assertFalse(result.has_errors)
        self.assertEqual(
            [statement.kind for statement in result.ast.statements],
            scenario["expected_kinds"],
        )
        set_statement = next(statement for statement in result.ast.statements if statement.kind == "SET")
        self.assertEqual(tuple(ref.name for ref in set_statement.dataset_refs), scenario["expected_inputs"])

    def test_parses_dataset_options_where_rename_keep_drop(self) -> None:
        scenario = PARSER_SCENARIOS["dataset_options_where_rename_keep_drop"]
        dsl_text = scenario["dsl"]

        result = self.parser.parse(dsl_text)

        self.assertFalse(result.has_errors)
        set_statement = next(statement for statement in result.ast.statements if statement.kind == "SET")
        options = set_statement.dataset_refs[0].options
        self.assertEqual(options.keep_vars, ("id", "amount"))
        self.assertEqual(options.drop_vars, ("tmp",))
        self.assertEqual(options.rename_map, {"amount": "amt"})
        self.assertEqual(options.where_expr, "amt > 0")

    def test_parses_data_output_dataset_options(self) -> None:
        scenario = PARSER_SCENARIOS["data_output_dataset_options"]
        dsl_text = scenario["dsl"]

        result = self.parser.parse(dsl_text)

        self.assertFalse(result.has_errors)
        data_statement = next(statement for statement in result.ast.statements if statement.kind == "DATA")
        self.assertEqual(tuple(ref.name for ref in data_statement.dataset_refs), ("a", "b"))
        self.assertEqual(data_statement.dataset_refs[0].options.keep_vars, ("id",))
        self.assertEqual(data_statement.dataset_refs[1].options.drop_vars, ("tmp",))
        self.assertEqual(data_statement.dataset_refs[1].options.rename_map, {"name": "full_name"})

    def test_parses_rename_statement(self) -> None:
        scenario = PARSER_SCENARIOS["rename_statement"]
        dsl_text = scenario["dsl"]

        result = self.parser.parse(dsl_text)

        self.assertFalse(result.has_errors)
        rename_statement = next(statement for statement in result.ast.statements if statement.kind == "RENAME")
        self.assertEqual(rename_statement.rename_map, {"amount": "amt", "score": "score_new"})

    def test_parses_rename_statement_with_whitespace_around_equals(self) -> None:
        scenario = PARSER_SCENARIOS["rename_statement_with_spaces"]
        dsl_text = scenario["dsl"]

        result = self.parser.parse(dsl_text)

        self.assertFalse(result.has_errors)
        rename_statement = next(statement for statement in result.ast.statements if statement.kind == "RENAME")
        self.assertEqual(rename_statement.rename_map, {"id": "id2", "amount": "amt"})

    def test_parses_keywords_case_insensitively(self) -> None:
        scenario = PARSER_SCENARIOS["case_insensitive_keywords"]
        dsl_text = scenario["dsl"]

        result = self.parser.parse(dsl_text)

        self.assertFalse(result.has_errors)
        self.assertEqual(
            [statement.kind for statement in result.ast.statements],
            scenario["expected_kinds"],
        )

    def test_lark_parser_is_initialized_and_used(self) -> None:
        scenario = PARSER_SCENARIOS["lark_parser_initialization"]
        self.assertIsNotNone(self.parser._lark_parser)

        result = self.parser.parse(scenario["dsl"])

        self.assertFalse(result.has_errors)
        self.assertEqual([statement.kind for statement in result.ast.statements], scenario["expected_kinds"])

    def test_parses_line_comment(self) -> None:
        scenario = PARSER_SCENARIOS["line_comment"]
        dsl_text = scenario["dsl"]

        result = self.parser.parse(dsl_text)

        self.assertFalse(result.has_errors)
        self.assertEqual([statement.kind for statement in result.ast.statements], ["DATA", "SET", "RUN"])

    def test_parses_block_comment_inline(self) -> None:
        scenario = PARSER_SCENARIOS["block_comment_inline"]
        dsl_text = scenario["dsl"]

        result = self.parser.parse(dsl_text)

        self.assertFalse(result.has_errors)
        self.assertEqual([statement.kind for statement in result.ast.statements], ["DATA", "SET", "RUN"])

    def test_parses_block_comment_tail(self) -> None:
        scenario = PARSER_SCENARIOS["block_comment_tail"]
        dsl_text = scenario["dsl"]

        result = self.parser.parse(dsl_text)

        self.assertFalse(result.has_errors)
        self.assertEqual([statement.kind for statement in result.ast.statements], ["DATA", "SET", "RUN"])

    def test_parses_set_statement_options_with_whitespace_around_equals(self) -> None:
        scenario = PARSER_SCENARIOS["set_options_equals_whitespace"]
        dsl_text = scenario["dsl"]

        result = self.parser.parse(dsl_text)

        self.assertFalse(result.has_errors)
        set_statements = [statement for statement in result.ast.statements if statement.kind == "SET"]
        self.assertEqual(set_statements[0].statement_options.end_var, "eof")
        self.assertEqual(set_statements[1].statement_options.end_var, "eof")
        self.assertEqual(set_statements[2].statement_options.end_var, "eof")
        self.assertEqual(set_statements[3].statement_options.end_var, "eof")
        self.assertEqual(set_statements[4].statement_options.indsname_var, "src")
        self.assertEqual(set_statements[5].statement_options.indsname_var, "src")
        self.assertEqual(set_statements[6].dataset_refs[0].options.in_var, "flag1")
        self.assertEqual(set_statements[7].dataset_refs[0].options.in_var, "flag2")
        self.assertEqual(set_statements[8].dataset_refs[0].options.in_var, "flag3")
        self.assertEqual(set_statements[9].dataset_refs[0].options.in_var, "flag4")

    def test_parses_unsupported_statements_as_skipped(self) -> None:
        scenario = PARSER_SCENARIOS["unsupported_as_skipped"]
        dsl_text = scenario["dsl"]

        result = self.parser.parse(dsl_text)

        self.assertFalse(result.has_errors)
        skipped = [statement for statement in result.ast.statements if statement.kind == "SKIPPED"]
        self.assertEqual(len(skipped), 5)

    def test_parses_call_statement_as_skipped(self) -> None:
        scenario = PARSER_SCENARIOS["call_statement_as_skipped"]
        result = self.parser.parse(scenario["dsl"])

        self.assertFalse(result.has_errors)
        skipped = [statement for statement in result.ast.statements if statement.kind == "SKIPPED"]
        self.assertEqual(len(skipped), 1)
        self.assertIn("call", skipped[0].text.lower())

    def test_parses_stop_statement(self) -> None:
        scenario = PARSER_SCENARIOS["stop_statement"]
        result = self.parser.parse(scenario["dsl"])

        self.assertFalse(result.has_errors)
        self.assertEqual([statement.kind for statement in result.ast.statements], scenario["expected_kinds"])

    def test_parses_if_structured_metadata(self) -> None:
        scenario = PARSER_SCENARIOS["if_structured_spec"]
        result = self.parser.parse(scenario["dsl"])

        self.assertFalse(result.has_errors)
        if_statements = [statement for statement in result.ast.statements if statement.kind == "IF"]
        self.assertEqual(len(if_statements), 2)

        subset_if = if_statements[0]
        self.assertIsNotNone(subset_if.if_spec)
        self.assertTrue(subset_if.if_spec.is_subset)
        self.assertEqual(subset_if.if_spec.condition, "amount > 0")
        self.assertIsNone(subset_if.if_spec.then_action)

        then_if = if_statements[1]
        self.assertIsNotNone(then_if.if_spec)
        self.assertFalse(then_if.if_spec.is_subset)
        self.assertEqual(then_if.if_spec.condition, "amount == 0")
        self.assertEqual(then_if.if_spec.then_action, "output out")

    def test_parses_do_and_array_structured_metadata(self) -> None:
        scenario = PARSER_SCENARIOS["do_array_structured_spec"]
        result = self.parser.parse(scenario["dsl"])

        self.assertFalse(result.has_errors)
        array_statement = next(statement for statement in result.ast.statements if statement.kind == "ARRAY")
        do_statement = next(statement for statement in result.ast.statements if statement.kind == "DO")

        self.assertIsNotNone(array_statement.array_spec)
        self.assertEqual(array_statement.array_spec.array_name, "vars")
        self.assertEqual(array_statement.array_spec.variables, ("a", "b", "c"))
        self.assertTrue(array_statement.array_spec.wildcard_size)

        self.assertIsNotNone(do_statement.do_spec)
        self.assertEqual(do_statement.do_spec.loop_var, "i")
        self.assertEqual(do_statement.do_spec.start_expr, "1")
        self.assertEqual(do_statement.do_spec.end_expr, "3")


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
