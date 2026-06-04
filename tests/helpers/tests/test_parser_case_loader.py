from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.helpers.parser_case_loader import ParserCaseLoaderError, load_parser_ast_case_packs


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_loader_reads_grouped_parser_ast_cases(tmp_path: Path) -> None:
    root = tmp_path / "cases" / "parser"
    case_dir = root / "core"
    (case_dir / "programs").mkdir(parents=True, exist_ok=True)
    (case_dir / "expected").mkdir(parents=True, exist_ok=True)

    _write_json(
        case_dir / "case.json",
        {
            "owner": "parser",
            "variants": [
                {
                    "case_id": "basic_ast",
                    "program": "programs/basic_ast.txt",
                    "expected": "expected/basic_ast.json",
                }
            ],
        },
    )
    (case_dir / "programs" / "basic_ast.txt").write_text("data out; set in; run;\n", encoding="utf-8")
    _write_json(case_dir / "expected" / "basic_ast.json", {"statements": [{"kind": "DATA"}, {"kind": "SET"}, {"kind": "RUN"}]})

    cases = load_parser_ast_case_packs(root)

    assert len(cases) == 1
    assert cases[0].case_id == "basic_ast"
    assert cases[0].owner == "parser"
    assert cases[0].expected_ast["statements"][1]["kind"] == "SET"


def test_loader_rejects_missing_statement_expectations(tmp_path: Path) -> None:
    root = tmp_path / "cases" / "parser"
    case_dir = root / "core"
    (case_dir / "programs").mkdir(parents=True, exist_ok=True)
    (case_dir / "expected").mkdir(parents=True, exist_ok=True)

    _write_json(
        case_dir / "case.json",
        {
            "owner": "parser",
            "variants": [
                {
                    "case_id": "broken_ast",
                    "program": "programs/broken_ast.txt",
                    "expected": "expected/broken_ast.json",
                }
            ],
        },
    )
    (case_dir / "programs" / "broken_ast.txt").write_text("data out; run;\n", encoding="utf-8")
    _write_json(case_dir / "expected" / "broken_ast.json", {})

    with pytest.raises(ParserCaseLoaderError, match="non-empty statements list"):
        load_parser_ast_case_packs(root)
