import re

import pyarrow as pa
import pytest

from limulus import Session
from limulus.native_bridge import load_native_module
from limulus.models import DiagnosticLabel, DiagnosticSpan, LogEntry, SubmitResult


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _dataset_rows(dataset) -> list[dict[str, object]]:
    if hasattr(dataset, "to_pylist"):
        return dataset.to_pylist()
    return list(dataset)

SESSION_SCENARIOS = {
    "load_submit_dataset_access": {
        "overview": "Session.load + submit exposes work datasets and filters rows by IF condition",
        "inputs": {"inp": {"id": [1, 2], "amount": [10, -5]}},
        "dsl": """
        data out;
        set inp;
        if amount >= 0 then output out;
        run;
        """,
        "expected_output": [{"id": 1, "amount": 10}],
    },
    "log_returns_entries": {
        "overview": "Invalid blank submit populates session log with error severity",
        "dsl": " ",
        "expected_success": False,
        "expected_first_severity": "error",
    },
    "acceptance_skipped_unsupported_logs": {
        "overview": "Unsupported statements are skipped and logged as info notices",
        "dsl": """
        data out;
        length id 8;
        attrib id length=8;
        format id 8.;
        label id = "Identifier";
        informat id 8.;
        set inp;
        output out;
        run;
        """,
        "expected_skip_count": 4,
    },
}


def test_session_log_returns_entries() -> None:
    scenario = SESSION_SCENARIOS["log_returns_entries"]
    session = Session()
    result = session.submit(scenario["dsl"])

    assert result.success is scenario["expected_success"]
    assert len(session.log.log) >= 1
    assert session.log.log[0].severity == scenario["expected_first_severity"]
    assert session.get_log() == session.log


def test_session_submit_prints_log_on_failure(capsys: pytest.CaptureFixture[str]) -> None:
    scenario = SESSION_SCENARIOS["log_returns_entries"]
    session = Session()

    result = session.submit(scenario["dsl"])
    captured = capsys.readouterr()

    assert result.success is False
    assert "Error" in captured.out


def test_session_submit_parse_failure_prints_source_excerpt(capsys: pytest.CaptureFixture[str]) -> None:
    session = Session()

    result = session.submit("data out;\n    invalid syntax;\nrun;")
    captured = capsys.readouterr()
    rendered = _strip_ansi(captured.out)

    assert result.success is False
    assert "parse_unsupported_statement" in rendered.lower()
    assert "<dsl>:2:5" in rendered
    assert "invalid syntax;" in rendered
    assert "syntax error" in rendered.lower()


def test_session_submit_validation_error_shows_category_label() -> None:
    session = Session()

    result = session.submit("data out;\n  set dummy;\nrun;")

    rendered = _strip_ansi(result.format_log())
    assert result.success is False
    assert "runtime_set_dataset_not_found" in rendered.lower()
    assert "missing dataset" in rendered.lower()
    assert "set dummy;" in rendered.lower()


def test_session_submit_execute_error_uses_statement_excerpt() -> None:
    session = Session()
    session.load("inp", pa.table({"amount": [10]}))

    result = session.submit("data out;\n  set inp;\n  total = unknown_func(amount);\nrun;")

    rendered = _strip_ansi(result.format_log())
    assert result.success is False
    assert "runtime_expression_evaluation_error" in rendered.lower()
    assert "[stage: execute]" in rendered.lower()
    assert "total = unknown_func(amount)" in rendered
    assert "expression error" in rendered.lower()
    assert "^" in rendered


def test_session_submit_execute_where_error_uses_statement_excerpt() -> None:
    session = Session()
    session.load("inp", pa.table({"amount": [10]}))

    result = session.submit("data out;\n  set inp;\n  where amount >< 1;\nrun;")

    rendered = _strip_ansi(result.format_log())
    assert result.success is False
    assert "runtime_operator_not_supported" in rendered.lower()
    assert "[stage: execute]" in rendered.lower()
    assert "where amount >< 1" in rendered.lower()
    assert "operator error" in rendered.lower()
    assert "^" in rendered


def test_session_tracks_last_submit_result() -> None:
    scenario = SESSION_SCENARIOS["load_submit_dataset_access"]
    session = Session()
    assert session.log is None

    session.load("inp", pa.table(scenario["inputs"]["inp"]))
    result = session.submit(scenario["dsl"])

    assert result.success is True
    assert session.get_log() is result


def test_submit_result_print_log_on_failure_shows_status_elapsed_and_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    scenario = SESSION_SCENARIOS["log_returns_entries"]
    session = Session()

    result = session.submit(scenario["dsl"])
    _ = capsys.readouterr()

    result.print_log()
    captured = capsys.readouterr()

    assert result.success is False
    assert "success: False" in captured.out
    assert "seconds elapsed" in captured.out
    assert "Error" in captured.out


def test_session_submit_invalid_prx_pattern_renders_source_aware_excerpt() -> None:
    session = Session()
    session.load("in", pa.table({"name": ["Alice"]}))

    result = session.submit("data out; set in; where prxmatch('/[/', name) > 0; output out; run;")

    rendered = _strip_ansi(result.format_log())
    assert result.success is False
    assert "runtime_function_argument_invalid" in rendered.lower()
    assert "<prxmatch-pattern>" in rendered
    assert "/[/" in rendered
    assert "regex syntax" in rendered.lower()
    assert "^" in rendered


def test_session_submit_invalid_prx_flag_renders_source_aware_excerpt() -> None:
    session = Session()
    session.load("in", pa.table({"name": ["Alice"]}))

    result = session.submit("data out; set in; where prxmatch('/foo/z', name) > 0; output out; run;")

    rendered = _strip_ansi(result.format_log())
    assert result.success is False
    assert "runtime_function_argument_invalid" in rendered.lower()
    assert "<prxmatch-pattern>" in rendered
    assert "/foo/z" in rendered
    assert "unsupported prx flag" in rendered.lower()
    assert "^" in rendered


def test_native_ariadne_renderer_renders_multiple_diagnostics_and_notes() -> None:
    native_module, error = load_native_module()

    assert native_module is not None, error
    if not callable(getattr(native_module, "render_diagnostics_ariadne", None)):
        pytest.skip("installed native module does not expose render_diagnostics_ariadne in this test environment")

    rendered = native_module.render_diagnostics_ariadne(
        {
            "source_id": "<dsl>",
            "source_text": "data out; set in; keep id missing; run;",
            "diagnostics": [
                {
                    "code": "VALIDATE_COLUMN_NOT_FOUND",
                    "severity": "error",
                    "message": "KEEP statement references unknown variable: missing",
                    "stage": "validate",
                    "span": {
                        "start": 18,
                        "end": 33,
                        "line": 1,
                        "column": 19,
                        "end_line": 1,
                        "end_column": 34,
                        "source_id": "<dsl>",
                    },
                    "labels": [
                        {
                            "span": {
                                "start": 23,
                                "end": 30,
                                "line": 1,
                                "column": 24,
                                "end_line": 1,
                                "end_column": 31,
                                "source_id": "<dsl>",
                            },
                            "message": "unknown variable",
                            "kind": "primary",
                        },
                        {
                            "span": {
                                "start": 18,
                                "end": 22,
                                "line": 1,
                                "column": 19,
                                "end_line": 1,
                                "end_column": 23,
                                "source_id": "<dsl>",
                            },
                            "message": "KEEP clause",
                            "kind": "secondary",
                        },
                    ],
                    "notes": ["validate checks only statically derivable identifiers"],
                },
                {
                    "code": "PARSE_UNSUPPORTED_STATEMENT",
                    "severity": "error",
                    "message": "Unsupported statement syntax",
                    "stage": "parse",
                    "span": {
                        "start": 10,
                        "end": 17,
                        "line": 1,
                        "column": 11,
                        "end_line": 1,
                        "end_column": 18,
                        "source_id": "<dsl>",
                    },
                    "labels": [
                        {
                            "span": {
                                "start": 10,
                                "end": 17,
                                "line": 1,
                                "column": 11,
                                "end_line": 1,
                                "end_column": 18,
                                "source_id": "<dsl>",
                            },
                            "message": "syntax error",
                            "kind": "primary",
                        }
                    ],
                    "notes": ["second diagnostic"],
                    "source_text": "data out; invalid syntax; run;",
                },
            ],
        }
    )
    stripped = _strip_ansi(rendered)

    assert "VALIDATE_COLUMN_NOT_FOUND" in stripped
    assert "PARSE_UNSUPPORTED_STATEMENT" in stripped
    assert "KEEP clause" in stripped
    assert "unknown variable" in stripped
    assert "syntax error" in stripped
    assert "validate checks only statically derivable identifiers" in stripped
    assert "second diagnostic" in stripped


def test_session_logs_info_for_skipped_unsupported_statements() -> None:
    scenario = SESSION_SCENARIOS["acceptance_skipped_unsupported_logs"]
    session = Session()
    session.load("inp", pa.table({"id": [1]}))

    result = session.submit(scenario["dsl"])

    assert result.success is True
    skipped_logs = [entry for entry in result.log if entry.severity == "info" and "unsupported statement skipped" in entry.message]
    assert len(skipped_logs) == scenario["expected_skip_count"]


def test_session_submit_staged_pipeline_matches_session_method_reference() -> None:
    source_rows = {
        "id": [1, 2, 3, 4],
        "amount": [10, 20, 30, 40],
        "tmp": ["a", "b", "c", "d"],
    }
    dsl = (
        "data out; "
        "set inp(keep=id amount tmp drop=tmp where=(amount >= 20) rename=(amount=amt)); "
        "where amt >= 30; "
        "score = amt + 1; "
        "output out; "
        "keep id score; "
        "rename score=final_score; "
        "run;"
    )

    reference = Session()
    reference.load("inp", pa.table(source_rows))
    reference.apply_dataset_options(
        "inp",
        where="amount >= 20",
        keep=["id", "amount", "tmp"],
        drop=["tmp"],
        rename={"amount": "amt"},
        out="stage",
    )
    reference.where("stage", "amt >= 30", out="flt")
    reference.assign("flt", out="calc", score="amt + 1")
    reference.keep("calc", ["id", "score"], out="kept")
    reference.rename("kept", {"score": "final_score"}, out="expected")
    expected_rows = reference["expected"].to_pylist()

    session = Session(parser_backend="python")
    session.load("inp", pa.table(source_rows))
    result = session.submit(dsl)

    assert result.success is True
    assert _dataset_rows(result.datasets["out"]) == expected_rows
