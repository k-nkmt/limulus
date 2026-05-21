from __future__ import annotations

from pathlib import Path

try:
    from lark import Lark
except Exception:  # pragma: no cover
    Lark = None  # type: ignore[assignment]


def build_lark_parser_from_file(
    grammar_file_name: str,
    *,
    start: str = "start",
    parser: str = "lalr",
    propagate_positions: bool = True,
):
    if Lark is None:
        return None

    grammar_path = Path(__file__).with_name("grammar") / grammar_file_name
    try:
        grammar = grammar_path.read_text(encoding="utf-8")
        return Lark(grammar, start=start, parser=parser, propagate_positions=propagate_positions)
    except Exception:
        return None