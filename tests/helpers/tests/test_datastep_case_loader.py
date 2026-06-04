from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.helpers.datastep_case_loader import DataStepCaseLoaderError, load_case_packs


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_case(root: Path, *, case_id: str = "sample") -> Path:
    case_dir = root / "set" / case_id
    (case_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (case_dir / "expected").mkdir(parents=True, exist_ok=True)

    _write_json(
        case_dir / "case.json",
        {
            "case_id": case_id,
            "category": "set",
            "inputs": {"in": "inputs/in.json"},
            "expected": "expected/output.json"
        },
    )
    (case_dir / "program.txt").write_text(
        "data out; set in; output out; run;\n",
        encoding="utf-8",
    )
    _write_json(case_dir / "inputs" / "in.json", {"rows": [{"id": 1}, {"id": 2}]})
    _write_json(
        case_dir / "expected" / "output.json",
        {"datasets": {"out": {"rows": [{"id": 1}, {"id": 2}]}}},
    )
    return case_dir


def test_loader_reads_case_schema_and_dataset_keyed_expected(tmp_path: Path) -> None:
    root = tmp_path / "cases" / "datastep"
    _write_case(root)

    cases = load_case_packs(root)

    assert len(cases) == 1
    loaded = cases[0]
    assert loaded.case_id == "sample"
    assert loaded.category == "set"
    assert loaded.backend == "rust"
    assert loaded.extended is False
    assert loaded.inputs["in"] == [{"id": 1}, {"id": 2}]
    assert loaded.expected_datasets["out"].rows == [{"id": 1}, {"id": 2}]
    assert loaded.expected_datasets["out"].memlabel is None
    assert loaded.expected_datasets["out"].column_labels == {}


def test_loader_reads_grouped_layout_variants(tmp_path: Path) -> None:
    root = tmp_path / "cases" / "datastep"
    case_dir = root / "functions" / "numeric"
    (case_dir / "programs").mkdir(parents=True, exist_ok=True)
    (case_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (case_dir / "expected").mkdir(parents=True, exist_ok=True)

    _write_json(
        case_dir / "case.json",
        {
            "category": "functions",
            "variants": [
                {
                    "case_id": "numeric_alpha",
                    "backend": "python",
                    "extended": True,
                    "program": "programs/numeric_alpha.txt",
                    "inputs": {"in": "inputs/alpha.json"},
                    "expected": "expected/alpha.json",
                },
                {
                    "case_id": "numeric_beta",
                    "backend": "rust",
                    "program": "programs/numeric_beta.txt",
                    "inputs": {"in": "inputs/beta.json"},
                    "expected": "expected/beta.json",
                },
            ],
        },
    )
    (case_dir / "programs" / "numeric_alpha.txt").write_text(
        "data out; set in; output out; run;\n",
        encoding="utf-8",
    )
    (case_dir / "programs" / "numeric_beta.txt").write_text(
        "data out; set in; output out; run;\n",
        encoding="utf-8",
    )
    _write_json(case_dir / "inputs" / "alpha.json", {"rows": [{"id": 1}]})
    _write_json(case_dir / "inputs" / "beta.json", {"rows": [{"id": 2}]})
    _write_json(case_dir / "expected" / "alpha.json", {"datasets": {"out": {"rows": [{"id": 1}]}}})
    _write_json(case_dir / "expected" / "beta.json", {"datasets": {"out": {"rows": [{"id": 2}]}}})

    cases = load_case_packs(root)

    assert [case.case_id for case in cases] == ["numeric_alpha", "numeric_beta"]
    assert cases[0].extended is True
    assert cases[1].backend == "rust"
    assert cases[1].extended is False
    assert cases[0].inputs["in"] == [{"id": 1}]
    assert cases[1].expected_datasets["out"].rows == [{"id": 2}]


def test_loader_rejects_path_traversal_in_manifest(tmp_path: Path) -> None:
    root = tmp_path / "cases" / "datastep"
    case_dir = _write_case(root, case_id="bad_path")

    manifest = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    manifest["inputs"]["in"] = "../escape.json"
    (case_dir / "case.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises(DataStepCaseLoaderError, match="path traversal"):
        load_case_packs(root)


def test_loader_reports_case_id_for_broken_fixture(tmp_path: Path) -> None:
    root = tmp_path / "cases" / "datastep"
    case_dir = _write_case(root, case_id="broken_expected")

    (case_dir / "expected" / "output.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(DataStepCaseLoaderError, match="broken_expected"):
        load_case_packs(root)


def test_loader_rejects_engine_named_expected_artifact(tmp_path: Path) -> None:
    root = tmp_path / "cases" / "datastep"
    case_dir = _write_case(root, case_id="engine_named_expected")

    manifest = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    manifest["expected"] = "expected/limulus.json"
    (case_dir / "case.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_json(case_dir / "expected" / "limulus.json", {"datasets": {"out": {"rows": [{"id": 1}]}}})

    with pytest.raises(DataStepCaseLoaderError, match="engine-named artifact"):
        load_case_packs(root)


def test_loader_reads_table_json_for_inputs_and_expected(tmp_path: Path) -> None:
    root = tmp_path / "cases" / "datastep"
    case_dir = _write_case(root, case_id="table_json")

    _write_json(
        case_dir / "inputs" / "in.json",
        {
            "columns": ["id", "name"],
            "rows": [
                [1, "Alice"],
                [2, "Bob"],
            ],
        },
    )
    _write_json(
        case_dir / "expected" / "output.json",
        {
            "datasets": {
                "out": {
                    "columns": ["id", "name"],
                    "rows": [
                        [1, "Alice"],
                        [2, "Bob"],
                    ],
                }
            }
        },
    )

    cases = load_case_packs(root)

    assert cases[0].inputs["in"] == [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    assert cases[0].expected_datasets["out"].rows == [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]


def test_loader_reads_expected_dataset_and_column_labels(tmp_path: Path) -> None:
    root = tmp_path / "cases" / "datastep"
    case_dir = _write_case(root, case_id="labels")

    _write_json(
        case_dir / "expected" / "output.json",
        {
            "datasets": {
                "out": {
                    "memlabel": "OUT",
                    "columns": {
                        "id": "Identifier",
                    },
                    "rows": [{"id": 1}, {"id": 2}],
                }
            }
        },
    )

    cases = load_case_packs(root)

    assert cases[0].expected_datasets["out"].memlabel == "OUT"
    assert cases[0].expected_datasets["out"].column_labels == {"id": "Identifier"}


def test_loader_accepts_columns_dict_with_null_labels(tmp_path: Path) -> None:
    root = tmp_path / "cases" / "datastep"
    case_dir = _write_case(root, case_id="null_labels")

    _write_json(
        case_dir / "expected" / "output.json",
        {
            "datasets": {
                "out": {
                    "columns": {
                        "id": None,
                        "amount": "Amount",
                    },
                    "rows": [
                        [1, 10],
                    ],
                }
            }
        },
    )

    cases = load_case_packs(root)

    assert cases[0].expected_datasets["out"].rows == [{"id": 1, "amount": 10}]
    assert cases[0].expected_datasets["out"].column_labels == {"amount": "Amount"}


def test_loader_rejects_row_length_mismatch_for_table_json(tmp_path: Path) -> None:
    root = tmp_path / "cases" / "datastep"
    case_dir = _write_case(root, case_id="row_length_mismatch")

    _write_json(
        case_dir / "inputs" / "in.json",
        {
            "columns": ["id", "name"],
            "rows": [[1]],
        },
    )

    with pytest.raises(DataStepCaseLoaderError, match="length does not match columns length"):
        load_case_packs(root)
