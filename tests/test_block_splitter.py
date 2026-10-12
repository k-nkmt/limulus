from limulus.block_splitter import DataStepBlockSplitter


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
