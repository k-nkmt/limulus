from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from limulus.parser import ParserService, ast_to_dict
from tests.helpers.parser_case_loader import ParserAstCase, load_parser_ast_case_packs


CASE_ROOT = Path(__file__).resolve().parents[2] / "cases" / "parser"


def _assert_subset(expected: Any, actual: Any, *, path: str) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected dict, got {type(actual).__name__}"
        for key, value in expected.items():
            assert key in actual, f"{path}: missing key {key}"
            _assert_subset(value, actual[key], path=f"{path}.{key}")
        return

    if isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected list, got {type(actual).__name__}"
        assert len(actual) == len(expected), f"{path}: expected {len(expected)} items, got {len(actual)}"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            _assert_subset(expected_item, actual_item, path=f"{path}[{index}]")
        return

    assert actual == expected, f"{path}: expected {expected!r}, got {actual!r}"


@pytest.mark.parametrize("case", load_parser_ast_case_packs(CASE_ROOT), ids=lambda c: c.case_id)
def test_parser_ast_case_packs_match_expected_structure(case: ParserAstCase) -> None:
    result = ParserService().parse(case.program_text)

    assert result.has_errors is False, f"case={case.case_id}: expected parser success"
    _assert_subset(case.expected_ast, ast_to_dict(result.ast), path=case.case_id)
