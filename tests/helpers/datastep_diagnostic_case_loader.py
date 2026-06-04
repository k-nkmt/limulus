from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa

from limulus.models import DataSetRef


class DataStepDiagnosticCaseLoaderError(ValueError):
    """Raised when an external Data Step diagnostic fixture is invalid."""


@dataclass(frozen=True)
class DiagnosticExpectation:
    stage: str
    code: str
    severity: str
    location: str | None
    message_fragments: tuple[str, ...]
    span: dict[str, int] | None
    labels: tuple[dict[str, Any], ...]
    source_text: str | None


@dataclass(frozen=True)
class DataStepDiagnosticCase:
    case_id: str
    owner: str
    backend: str
    extended: bool
    case_dir: Path
    program_text: str
    inputs: dict[str, list[dict[str, Any]]]
    expected_diagnostics: tuple[DiagnosticExpectation, ...]
    output_targets: tuple[str, ...] | None

    def input_dataset_refs(self) -> dict[str, DataSetRef]:
        return {
            name: DataSetRef(
                kind="arrow_table",
                location=f"dataset://{name}",
                payload=pa.Table.from_pylist(rows),
            )
            for name, rows in self.inputs.items()
        }


def _read_json(path: Path, *, case_id: str, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DataStepDiagnosticCaseLoaderError(f"case={case_id}: missing {label} file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DataStepDiagnosticCaseLoaderError(f"case={case_id}: invalid JSON in {label}: {path}") from exc

    if not isinstance(payload, dict):
        raise DataStepDiagnosticCaseLoaderError(f"case={case_id}: {label} must be a JSON object")
    return payload


def _resolve_relative_file(case_dir: Path, relative_path: str, *, case_id: str, label: str) -> Path:
    candidate = (case_dir / relative_path).resolve()
    case_root = case_dir.resolve()

    try:
        candidate.relative_to(case_root)
    except ValueError as exc:
        raise DataStepDiagnosticCaseLoaderError(
            f"case={case_id}: path traversal detected for {label}: {relative_path}"
        ) from exc

    if not candidate.exists():
        raise DataStepDiagnosticCaseLoaderError(f"case={case_id}: missing {label} file: {relative_path}")
    return candidate


def _validate_backend(backend: Any, *, case_id: str) -> str:
    if backend not in {"python", "rust", "auto"}:
        raise DataStepDiagnosticCaseLoaderError(f"case={case_id}: backend must be one of python/rust/auto")
    return backend


def _validate_extended(extended: Any, *, case_id: str) -> bool:
    if not isinstance(extended, bool):
        raise DataStepDiagnosticCaseLoaderError(f"case={case_id}: extended must be a boolean")
    return extended


def _validate_output_targets(output_targets: Any, *, case_id: str) -> tuple[str, ...] | None:
    if output_targets is None:
        return None
    if not isinstance(output_targets, list) or not all(isinstance(name, str) and name for name in output_targets):
        raise DataStepDiagnosticCaseLoaderError(
            f"case={case_id}: output_targets must be a list of non-empty strings"
        )
    return tuple(output_targets)


def _load_rows_table(path: Path, *, case_id: str, label: str) -> list[dict[str, Any]]:
    payload = _read_json(path, case_id=case_id, label=label)
    rows = payload.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise DataStepDiagnosticCaseLoaderError(f"case={case_id}: {label} must have object rows list")
    return rows


def _load_expected_diagnostics(path: Path, *, case_id: str) -> tuple[DiagnosticExpectation, ...]:
    payload = _read_json(path, case_id=case_id, label="expected diagnostics")
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, list) or not diagnostics:
        raise DataStepDiagnosticCaseLoaderError(f"case={case_id}: expected diagnostics must be non-empty list")

    normalized: list[DiagnosticExpectation] = []
    for idx, item in enumerate(diagnostics):
        if not isinstance(item, dict):
            raise DataStepDiagnosticCaseLoaderError(f"case={case_id}: diagnostics[{idx}] must be object")
        stage = item.get("stage")
        code = item.get("code")
        severity = item.get("severity")
        if not isinstance(stage, str) or not stage:
            raise DataStepDiagnosticCaseLoaderError(f"case={case_id}: diagnostics[{idx}].stage must be string")
        if not isinstance(code, str) or not code:
            raise DataStepDiagnosticCaseLoaderError(f"case={case_id}: diagnostics[{idx}].code must be string")
        if not isinstance(severity, str) or not severity:
            raise DataStepDiagnosticCaseLoaderError(
                f"case={case_id}: diagnostics[{idx}].severity must be string"
            )
        location = item.get("location")
        if location is not None and (not isinstance(location, str) or not location):
            raise DataStepDiagnosticCaseLoaderError(
                f"case={case_id}: diagnostics[{idx}].location must be non-empty string when provided"
            )
        fragments = item.get("message_fragments")
        if not isinstance(fragments, list) or not all(isinstance(v, str) and v for v in fragments):
            raise DataStepDiagnosticCaseLoaderError(
                f"case={case_id}: diagnostics[{idx}].message_fragments must be non-empty strings"
            )
        span = item.get("span")
        if span is not None and not isinstance(span, dict):
            raise DataStepDiagnosticCaseLoaderError(
                f"case={case_id}: diagnostics[{idx}].span must be object when provided"
            )
        labels = item.get("labels", [])
        if not isinstance(labels, list) or not all(isinstance(v, dict) for v in labels):
            raise DataStepDiagnosticCaseLoaderError(
                f"case={case_id}: diagnostics[{idx}].labels must be list of objects"
            )
        source_text = item.get("source_text")
        if source_text is not None and not isinstance(source_text, str):
            raise DataStepDiagnosticCaseLoaderError(
                f"case={case_id}: diagnostics[{idx}].source_text must be string when provided"
            )
        normalized.append(
            DiagnosticExpectation(
                stage=stage,
                code=code,
                severity=severity,
                location=location,
                message_fragments=tuple(fragments),
                span=span,
                labels=tuple(labels),
                source_text=source_text,
            )
        )
    return tuple(normalized)


def _load_case(case_json_path: Path) -> list[DataStepDiagnosticCase]:
    case_dir = case_json_path.parent
    manifest = _read_json(case_json_path, case_id=case_dir.name, label="case manifest")
    required = {"owner", "variants"}
    missing = sorted(required - set(manifest.keys()))
    if missing:
        raise DataStepDiagnosticCaseLoaderError(
            f"case={case_dir.name}: missing required fields: {', '.join(missing)}"
        )

    owner = manifest["owner"]
    if not isinstance(owner, str) or not owner:
        raise DataStepDiagnosticCaseLoaderError(f"case={case_dir.name}: owner must be a non-empty string")

    variants = manifest["variants"]
    if not isinstance(variants, list) or not variants:
        raise DataStepDiagnosticCaseLoaderError(f"case={case_dir.name}: variants must be a non-empty list")

    cases: list[DataStepDiagnosticCase] = []
    for variant in variants:
        if not isinstance(variant, dict):
            raise DataStepDiagnosticCaseLoaderError(f"case={case_dir.name}: variants must contain objects")

        case_id = variant.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise DataStepDiagnosticCaseLoaderError(f"case={case_dir.name}: variant case_id must be non-empty string")

        program_ref = variant.get("program")
        expected_ref = variant.get("expected")
        inputs_ref = variant.get("inputs")
        if not isinstance(program_ref, str) or not program_ref.endswith(".txt"):
            raise DataStepDiagnosticCaseLoaderError(f"case={case_id}: program must be .txt path")
        if not isinstance(expected_ref, str) or not expected_ref.endswith(".diagnostics.json"):
            raise DataStepDiagnosticCaseLoaderError(
                f"case={case_id}: expected must be expected/*.diagnostics.json"
            )
        if not isinstance(inputs_ref, dict) or not inputs_ref:
            raise DataStepDiagnosticCaseLoaderError(f"case={case_id}: inputs must be a non-empty object")

        program_path = _resolve_relative_file(case_dir, program_ref, case_id=case_id, label="program")
        expected_path = _resolve_relative_file(case_dir, expected_ref, case_id=case_id, label="expected")

        inputs: dict[str, list[dict[str, Any]]] = {}
        for dataset_name, relative_path in inputs_ref.items():
            if not isinstance(dataset_name, str) or not dataset_name:
                raise DataStepDiagnosticCaseLoaderError(f"case={case_id}: invalid dataset key")
            if not isinstance(relative_path, str) or not relative_path:
                raise DataStepDiagnosticCaseLoaderError(f"case={case_id}: invalid dataset path")
            input_path = _resolve_relative_file(
                case_dir,
                relative_path,
                case_id=case_id,
                label=f"input:{dataset_name}",
            )
            inputs[dataset_name] = _load_rows_table(input_path, case_id=case_id, label=f"input:{dataset_name}")

        cases.append(
            DataStepDiagnosticCase(
                case_id=case_id,
                owner=owner,
                backend=_validate_backend(variant.get("backend", "rust"), case_id=case_id),
                extended=_validate_extended(variant.get("extended", False), case_id=case_id),
                case_dir=case_dir,
                program_text=program_path.read_text(encoding="utf-8"),
                inputs=inputs,
                expected_diagnostics=_load_expected_diagnostics(expected_path, case_id=case_id),
                output_targets=_validate_output_targets(variant.get("output_targets"), case_id=case_id),
            )
        )
    return cases


def load_diagnostic_case_packs(case_root: Path) -> list[DataStepDiagnosticCase]:
    if not case_root.exists():
        return []

    cases: list[DataStepDiagnosticCase] = []
    for case_json_path in sorted(case_root.rglob("case.json")):
        cases.extend(_load_case(case_json_path))
    return sorted(cases, key=lambda case: (str(case.case_dir), case.case_id))
