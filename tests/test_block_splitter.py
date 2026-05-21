from limulus.block_splitter import DataStepBlockPreparser, DataStepBlockSplitter


def test_block_preparser_skips_leading_comments_and_extracts_single_block() -> None:
    preparser = DataStepBlockPreparser()
    dsl_text = """* heading comment;
/* block comment */
data out;
  set in;
run;
"""

    assert preparser.split(dsl_text) == ("data out;\n  set in;\nrun;",)


def test_block_preparser_handles_block_comments_and_multiple_blocks() -> None:
    preparser = DataStepBlockPreparser()
    dsl_text = """data first;
  set a;
run;
/* between blocks */
data second;
  set b;
run;
"""

    assert preparser.split(dsl_text) == (
        "data first;\n  set a;\nrun;",
        "data second;\n  set b;\nrun;",
    )


def test_block_splitter_falls_back_to_preparser_for_syntax_error_mixed_blocks() -> None:
    splitter = DataStepBlockSplitter()
    dsl_text = """* parser should fail on ??? but block boundaries must survive;
data bad;
  ???;
run;
data good;
  set inp;
run;
"""

    assert splitter.split(dsl_text) == (
        "data bad;\n  ???;\nrun;",
        "data good;\n  set inp;\nrun;",
    )


def test_block_preparser_keeps_semicolons_inside_quoted_strings() -> None:
    preparser = DataStepBlockPreparser()
    dsl_text = """data first;
  note = 'alpha;beta';
run;
data second;
  note = \"gamma;delta\";
run;
"""

    assert preparser.split(dsl_text) == (
        "data first;\n  note = 'alpha;beta';\nrun;",
        'data second;\n  note = "gamma;delta";\nrun;',
    )