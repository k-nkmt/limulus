from __future__ import annotations

import json
from pathlib import Path

from tests.helpers.datastep_case_formatter import format_case_json_file, render_case_json


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_formatter_converts_object_rows_to_table_json(tmp_path: Path) -> None:
    path = tmp_path / "inputs" / "formatter_input.json"
    _write_json(
        path,
        {
            "rows": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ]
        },
    )

    changed = format_case_json_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert changed is True
    assert payload["columns"] == ["id", "name"]
    assert payload["rows"] == [[1, "Alice"], [2, "Bob"]]



def test_formatter_preserves_non_row_metadata_key_order(tmp_path: Path) -> None:
    path = tmp_path / "expected" / "output.json"
    _write_json(
        path,
        {
            "note": "keep-order",
            "datasets": {
                "out": {
                    "schema": {"id": "int"},
                    "rows": [{"id": 1}],
                }
            },
            "meta": {"owner": "demo"},
        },
    )

    format_case_json_file(path)
    text = path.read_text(encoding="utf-8")

    assert text.find('"note"') < text.find('"datasets"')
    assert text.find('"datasets"') < text.find('"meta"')
    payload = json.loads(text)
    assert payload["datasets"]["out"]["columns"] == ["id"]
    assert payload["datasets"]["out"]["rows"] == [[1]]



def test_formatter_is_stable_when_reapplied(tmp_path: Path) -> None:
    path = tmp_path / "inputs" / "in.json"
    _write_json(
        path,
        {
            "columns": ["id"],
            "rows": [[1], [2]],
        },
    )

    first = format_case_json_file(path)
    second = format_case_json_file(path)

    assert first is True
    assert second is False


def test_formatter_renders_columns_and_short_rows_compact(tmp_path: Path) -> None:
    path = tmp_path / "expected" / "output.json"
    _write_json(
        path,
        {
            "datasets": {
                "out": {
                    "columns": ["id", "amount"],
                    "rows": [[1, 10], [2, 20]],
                }
            }
        },
    )

    format_case_json_file(path)
    text = path.read_text(encoding="utf-8")

    assert '"columns": ["id", "amount"]' in text
    assert '      [1, 10],' in text
    assert '      [2, 20]' in text


def test_repository_case_fixtures_already_match_canonical_formatter_output() -> None:
    case_root = Path(__file__).resolve().parents[2] / "cases" / "datastep"

    mismatches: list[str] = []
    for path in sorted(case_root.rglob("*.json")):
        if not ({"inputs", "expected"} & set(path.parts)):
            continue
        before = path.read_text(encoding="utf-8")
        rendered = render_case_json(json.loads(before))
        if before != rendered:
            mismatches.append(str(path.relative_to(case_root.parent)))

    assert mismatches == []
