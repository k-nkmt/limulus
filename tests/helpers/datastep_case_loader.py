from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa


class DataStepCaseLoaderError(ValueError):
    """Raised when an external Data Step case fixture is invalid."""


@dataclass(frozen=True)
class ExpectedDataset:
    rows: list[dict]
    memlabel: str | None
    column_labels: dict[str, str]


@dataclass(frozen=True)
class DataStepCase:
    case_id: str
    category: str
    backend: str
    extended: bool
    compatibility_notes: tuple[str, ...]
    case_dir: Path
    program_text: str
    inputs: dict[str, list[dict]]
    expected_datasets: dict[str, ExpectedDataset]

    def arrow_inputs(self) -> dict[str, pa.Table]:
        return {name: pa.Table.from_pylist(rows) for name, rows in self.inputs.items()}


def _read_json(path: Path, *, case_id: str, label: str) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise DataStepCaseLoaderError(f"case={case_id}: missing {label} file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DataStepCaseLoaderError(f"case={case_id}: invalid JSON in {label}: {path}") from exc

    if not isinstance(payload, dict):
        raise DataStepCaseLoaderError(f"case={case_id}: {label} must be a JSON object")
    return payload


def _resolve_relative_file(case_dir: Path, relative_path: str, *, case_id: str, label: str) -> Path:
    candidate = (case_dir / relative_path).resolve()
    case_root = case_dir.resolve()

    try:
        candidate.relative_to(case_root)
    except ValueError as exc:
        raise DataStepCaseLoaderError(
            f"case={case_id}: path traversal detected for {label}: {relative_path}"
        ) from exc

    if not candidate.exists():
        raise DataStepCaseLoaderError(f"case={case_id}: missing {label} file: {relative_path}")
    return candidate


def _validate_case_id(case_id: Any, *, context: str) -> str:
    if not isinstance(case_id, str) or not case_id:
        raise DataStepCaseLoaderError(f"{context}: case_id must be a non-empty string")
    return case_id


def _validate_category(category: Any, *, case_id: str) -> str:
    if not isinstance(category, str) or not category:
        raise DataStepCaseLoaderError(f"case={case_id}: category must be a non-empty string")
    return category


def _validate_backend(backend: Any, *, case_id: str) -> str:
    if backend not in {"python", "rust", "auto"}:
        raise DataStepCaseLoaderError(f"case={case_id}: backend must be one of python/rust/auto")
    return backend


def _validate_extended(extended: Any, *, case_id: str) -> bool:
    if not isinstance(extended, bool):
        raise DataStepCaseLoaderError(f"case={case_id}: extended must be a boolean")
    return extended


def _validate_input_refs(inputs: Any, *, case_id: str) -> dict[str, str]:
    if not isinstance(inputs, dict) or not inputs:
        raise DataStepCaseLoaderError(f"case={case_id}: inputs must be a non-empty object")
    if not all(isinstance(key, str) and key and isinstance(value, str) and value for key, value in inputs.items()):
        raise DataStepCaseLoaderError(f"case={case_id}: inputs must map dataset names to relative file paths")
    return inputs


def _validate_expected_ref(expected: Any, *, case_id: str) -> str:
    if not isinstance(expected, str) or not expected:
        raise DataStepCaseLoaderError(f"case={case_id}: expected must be a non-empty relative path string")
    expected_path = Path(expected)
    if expected_path.parts[:1] != ("expected",) or expected_path.suffix != ".json":
        raise DataStepCaseLoaderError(
            f"case={case_id}: expected must use a neutral artifact path under expected/*.json"
        )
    if expected_path.name in {"limulus.json", "sas.json", "r.json"}:
        raise DataStepCaseLoaderError(
            f"case={case_id}: expected must not use engine-named artifact files"
        )
    return expected


def _validate_compatibility_notes(compatibility_notes: Any, *, case_id: str) -> tuple[str, ...]:
    if not isinstance(compatibility_notes, list) or not all(isinstance(note, str) for note in compatibility_notes):
        raise DataStepCaseLoaderError(f"case={case_id}: compatibility_notes must be a list of strings")
    return tuple(compatibility_notes)


def _validate_program_ref(program: Any, *, case_id: str) -> str:
    if not isinstance(program, str) or not program.endswith(".txt"):
        raise DataStepCaseLoaderError(f"case={case_id}: program must be a relative .txt path")
    return program


def _parse_columns_spec(
    columns: Any,
    *,
    case_id: str,
    label: str,
) -> tuple[list[str] | None, dict[str, str]]:
    if columns is None:
        return None, {}
    if isinstance(columns, list):
        if not all(isinstance(col, str) and col for col in columns):
            raise DataStepCaseLoaderError(
                f"case={case_id}: {label} columns must be a non-empty list of strings"
            )
        if len(set(columns)) != len(columns):
            raise DataStepCaseLoaderError(f"case={case_id}: {label} columns must not contain duplicates")
        return list(columns), {}
    if isinstance(columns, dict):
        ordered = list(columns.keys())
        if not all(isinstance(col, str) and col for col in ordered):
            raise DataStepCaseLoaderError(
                f"case={case_id}: {label} columns keys must be non-empty strings"
            )
        if not all(value is None or isinstance(value, str) for value in columns.values()):
            raise DataStepCaseLoaderError(
                f"case={case_id}: {label} columns values must be strings or null"
            )
        return ordered, {key: value for key, value in columns.items() if value is not None}
    raise DataStepCaseLoaderError(
        f"case={case_id}: {label} columns must be a list of strings or an object"
    )


def _normalize_rows_payload(payload: dict, *, case_id: str, label: str) -> list[dict]:
    rows = payload.get("rows")
    columns = payload.get("columns")

    if not isinstance(rows, list):
        raise DataStepCaseLoaderError(f"case={case_id}: {label} must contain rows list")

    ordered_columns, _column_labels = _parse_columns_spec(columns, case_id=case_id, label=label)

    normalized: list[dict] = []
    for idx, row in enumerate(rows):
        if isinstance(row, dict):
            if ordered_columns is None:
                normalized.append(row)
                continue
            normalized.append({name: row.get(name) for name in ordered_columns})
            continue
        if not isinstance(row, list):
            raise DataStepCaseLoaderError(
                f"case={case_id}: {label} row[{idx}] must be an object or list"
            )
        if ordered_columns is None:
            raise DataStepCaseLoaderError(
                f"case={case_id}: {label} row[{idx}] must be an object when columns is omitted"
            )
        if len(row) != len(ordered_columns):
            raise DataStepCaseLoaderError(
                f"case={case_id}: {label} row[{idx}] length does not match columns length"
            )
        normalized.append(dict(zip(ordered_columns, row)))
    return normalized


def _load_rows_table(path: Path, *, case_id: str, label: str) -> list[dict]:
    payload = _read_json(path, case_id=case_id, label=label)
    return _normalize_rows_payload(payload, case_id=case_id, label=label)


def _load_expected_datasets(expected_path: Path, *, case_id: str) -> dict[str, ExpectedDataset]:
    expected_payload = _read_json(expected_path, case_id=case_id, label="expected")
    datasets = expected_payload.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise DataStepCaseLoaderError(f"case={case_id}: expected must contain non-empty datasets object")

    expected_datasets: dict[str, ExpectedDataset] = {}
    for dataset_name, dataset_payload in datasets.items():
        if not isinstance(dataset_name, str) or not dataset_name:
            raise DataStepCaseLoaderError(f"case={case_id}: dataset name must be non-empty string")
        if not isinstance(dataset_payload, dict):
            raise DataStepCaseLoaderError(f"case={case_id}: dataset payload for {dataset_name} must be object")
        dataset_label = dataset_payload.get("memlabel")
        if dataset_label is not None and not isinstance(dataset_label, str):
            raise DataStepCaseLoaderError(f"case={case_id}: expected dataset {dataset_name} memlabel must be string")
        _ordered_columns, column_labels = _parse_columns_spec(
            dataset_payload.get("columns"),
            case_id=case_id,
            label=f"expected dataset {dataset_name}",
        )
        expected_datasets[dataset_name] = ExpectedDataset(
            rows=_normalize_rows_payload(
                dataset_payload,
                case_id=case_id,
                label=f"expected dataset {dataset_name}",
            ),
            memlabel=dataset_label,
            column_labels=column_labels,
        )
    return expected_datasets


def _build_case(
    *,
    case_dir: Path,
    category: str,
    case_id: str,
    backend: str,
    extended: bool,
    program_ref: str,
    input_refs: dict[str, str],
    expected_ref: str,
    compatibility_notes: tuple[str, ...],
) -> DataStepCase:
    program_path = _resolve_relative_file(case_dir, program_ref, case_id=case_id, label="program")
    program_text = program_path.read_text(encoding="utf-8")

    inputs: dict[str, list[dict]] = {}
    for dataset_name, relative_path in input_refs.items():
        input_path = _resolve_relative_file(
            case_dir,
            relative_path,
            case_id=case_id,
            label=f"input:{dataset_name}",
        )
        inputs[dataset_name] = _load_rows_table(
            input_path,
            case_id=case_id,
            label=f"input:{dataset_name}",
        )

    expected_path = _resolve_relative_file(
        case_dir,
        expected_ref,
        case_id=case_id,
        label="expected",
    )
    expected_datasets = _load_expected_datasets(expected_path, case_id=case_id)

    return DataStepCase(
        case_id=case_id,
        category=category,
        backend=backend,
        extended=extended,
        compatibility_notes=compatibility_notes,
        case_dir=case_dir,
        program_text=program_text,
        inputs=inputs,
        expected_datasets=expected_datasets,
    )


def _load_case(case_json_path: Path) -> list[DataStepCase]:
    case_dir = case_json_path.parent
    manifest = _read_json(case_json_path, case_id=case_dir.name, label="case manifest")
    if "variants" in manifest:
        shared_required = {"category", "variants"}
        missing = sorted(shared_required - set(manifest.keys()))
        if missing:
            raise DataStepCaseLoaderError(f"case={case_dir.name}: missing required fields: {', '.join(missing)}")

        shared_case_id = case_dir.name
        category = _validate_category(manifest["category"], case_id=shared_case_id)
        shared_notes = _validate_compatibility_notes(manifest.get("compatibility_notes", []), case_id=shared_case_id)
        variants = manifest["variants"]
        if not isinstance(variants, list) or not variants:
            raise DataStepCaseLoaderError(f"case={shared_case_id}: variants must be a non-empty list")

        cases: list[DataStepCase] = []
        seen_case_ids: set[str] = set()
        for variant in variants:
            if not isinstance(variant, dict):
                raise DataStepCaseLoaderError(f"case={shared_case_id}: variants must contain objects")
            case_id = _validate_case_id(variant.get("case_id"), context=f"case={shared_case_id}")
            if case_id in seen_case_ids:
                raise DataStepCaseLoaderError(f"case={shared_case_id}: duplicate variant case_id: {case_id}")
            seen_case_ids.add(case_id)
            cases.append(
                _build_case(
                    case_dir=case_dir,
                    category=category,
                    case_id=case_id,
                    backend=_validate_backend(variant.get("backend", manifest.get("backend", "rust")), case_id=case_id),
                    extended=_validate_extended(variant.get("extended", manifest.get("extended", False)), case_id=case_id),
                    program_ref=_validate_program_ref(variant.get("program", "program.txt"), case_id=case_id),
                    input_refs=_validate_input_refs(variant.get("inputs"), case_id=case_id),
                    expected_ref=_validate_expected_ref(variant.get("expected"), case_id=case_id),
                    compatibility_notes=_validate_compatibility_notes(
                        variant.get("compatibility_notes", list(shared_notes)),
                        case_id=case_id,
                    ),
                )
            )
        return cases

    required = {
        "case_id",
        "category",
        "inputs",
        "expected",
    }
    missing = sorted(required - set(manifest.keys()))
    if missing:
        raise DataStepCaseLoaderError(f"case={case_dir.name}: missing required fields: {', '.join(missing)}")

    case_id = _validate_case_id(manifest["case_id"], context=f"case={case_dir.name}")
    return [
        _build_case(
            case_dir=case_dir,
            category=_validate_category(manifest["category"], case_id=case_id),
            case_id=case_id,
            backend=_validate_backend(manifest.get("backend", "rust"), case_id=case_id),
            extended=_validate_extended(manifest.get("extended", False), case_id=case_id),
            program_ref="program.txt",
            input_refs=_validate_input_refs(manifest["inputs"], case_id=case_id),
            expected_ref=_validate_expected_ref(manifest["expected"], case_id=case_id),
            compatibility_notes=_validate_compatibility_notes(manifest.get("compatibility_notes", []), case_id=case_id),
        )
    ]


def load_case_packs(case_root: Path) -> list[DataStepCase]:
    if not case_root.exists():
        return []

    cases: list[DataStepCase] = []
    for case_json_path in sorted(case_root.rglob("case.json")):
        cases.extend(_load_case(case_json_path))
    return sorted(cases, key=lambda case: (str(case.case_dir), case.case_id))
