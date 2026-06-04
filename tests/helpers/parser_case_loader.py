from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ParserCaseLoaderError(ValueError):
    """Raised when an external parser fixture is invalid."""


@dataclass(frozen=True)
class ParserAstCase:
    case_id: str
    owner: str
    case_dir: Path
    program_text: str
    expected_ast: dict[str, Any]


def _read_json(path: Path, *, case_id: str, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ParserCaseLoaderError(f"case={case_id}: missing {label} file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ParserCaseLoaderError(f"case={case_id}: invalid JSON in {label}: {path}") from exc

    if not isinstance(payload, dict):
        raise ParserCaseLoaderError(f"case={case_id}: {label} must be a JSON object")
    return payload


def _resolve_relative_file(case_dir: Path, relative_path: str, *, case_id: str, label: str) -> Path:
    candidate = (case_dir / relative_path).resolve()
    case_root = case_dir.resolve()

    try:
        candidate.relative_to(case_root)
    except ValueError as exc:
        raise ParserCaseLoaderError(f"case={case_id}: path traversal detected for {label}: {relative_path}") from exc

    if not candidate.exists():
        raise ParserCaseLoaderError(f"case={case_id}: missing {label} file: {relative_path}")
    return candidate


def _load_expected_ast(path: Path, *, case_id: str) -> dict[str, Any]:
    payload = _read_json(path, case_id=case_id, label="expected AST")
    statements = payload.get("statements")
    if not isinstance(statements, list) or not statements:
        raise ParserCaseLoaderError(f"case={case_id}: expected AST must define non-empty statements list")
    return payload


def _load_case(case_json_path: Path) -> list[ParserAstCase]:
    case_dir = case_json_path.parent
    manifest = _read_json(case_json_path, case_id=case_dir.name, label="case manifest")
    owner = manifest.get("owner")
    variants = manifest.get("variants")
    if not isinstance(owner, str) or not owner:
        raise ParserCaseLoaderError(f"case={case_dir.name}: owner must be a non-empty string")
    if not isinstance(variants, list) or not variants:
        raise ParserCaseLoaderError(f"case={case_dir.name}: variants must be a non-empty list")

    cases: list[ParserAstCase] = []
    for variant in variants:
        if not isinstance(variant, dict):
            raise ParserCaseLoaderError(f"case={case_dir.name}: variants must contain objects")
        case_id = variant.get("case_id")
        program_ref = variant.get("program")
        expected_ref = variant.get("expected")
        if not isinstance(case_id, str) or not case_id:
            raise ParserCaseLoaderError(f"case={case_dir.name}: variant case_id must be non-empty string")
        if not isinstance(program_ref, str) or not program_ref.endswith(".txt"):
            raise ParserCaseLoaderError(f"case={case_id}: program must be .txt path")
        if not isinstance(expected_ref, str) or not expected_ref.endswith(".json"):
            raise ParserCaseLoaderError(f"case={case_id}: expected must be expected/*.json")

        program_path = _resolve_relative_file(case_dir, program_ref, case_id=case_id, label="program")
        expected_path = _resolve_relative_file(case_dir, expected_ref, case_id=case_id, label="expected")
        cases.append(
            ParserAstCase(
                case_id=case_id,
                owner=owner,
                case_dir=case_dir,
                program_text=program_path.read_text(encoding="utf-8"),
                expected_ast=_load_expected_ast(expected_path, case_id=case_id),
            )
        )
    return cases


def load_parser_ast_case_packs(case_root: Path) -> list[ParserAstCase]:
    if not case_root.exists():
        return []

    cases: list[ParserAstCase] = []
    for case_json_path in sorted(case_root.rglob("case.json")):
        cases.extend(_load_case(case_json_path))
    return sorted(cases, key=lambda case: (str(case.case_dir), case.case_id))
