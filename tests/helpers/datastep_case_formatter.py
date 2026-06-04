from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _rows_dict_to_table(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        return payload
    if not all(isinstance(row, dict) for row in rows):
        return payload

    seen: set[str] = set()
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)

    converted_rows = [[row.get(column) for column in columns] for row in rows]

    result: dict[str, Any] = {}
    inserted_columns = False
    for key, value in payload.items():
        if key == "rows":
            result["columns"] = columns
            result["rows"] = converted_rows
            inserted_columns = True
            continue
        result[key] = value

    if not inserted_columns:
        result["columns"] = columns
        result["rows"] = converted_rows
    return result


def _convert_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        if "datasets" in payload and isinstance(payload["datasets"], dict):
            converted: dict[str, Any] = {}
            for key, value in payload.items():
                if key == "datasets":
                    converted_datasets: dict[str, Any] = {}
                    for dataset_name, dataset_payload in value.items():
                        if isinstance(dataset_payload, dict):
                            converted_datasets[dataset_name] = _rows_dict_to_table(dataset_payload)
                        else:
                            converted_datasets[dataset_name] = dataset_payload
                    converted["datasets"] = converted_datasets
                else:
                    converted[key] = value
            return converted
        return _rows_dict_to_table(payload)
    return payload


def _json_atom(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True)


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _render_list(values: list[Any], indent: int) -> str:
    if not values:
        return "[]"

    if all(_is_scalar(value) for value in values):
        inline = ", ".join(_json_atom(value) for value in values)
        return f"[{inline}]"

    child_indent = indent + 2
    child_prefix = " " * child_indent
    rendered_items = [f"{child_prefix}{_render_value(value, child_indent)}" for value in values]
    return "[\n" + ",\n".join(rendered_items) + "\n" + (" " * indent) + "]"


def _render_dict(mapping: dict[str, Any], indent: int) -> str:
    if not mapping:
        return "{}"

    child_indent = indent + 2
    child_prefix = " " * child_indent
    rendered_items = [
        f'{child_prefix}{json.dumps(key, ensure_ascii=True)}: {_render_value(value, child_indent)}'
        for key, value in mapping.items()
    ]
    return "{\n" + ",\n".join(rendered_items) + "\n" + (" " * indent) + "}"


def _render_value(value: Any, indent: int) -> str:
    if isinstance(value, dict):
        return _render_dict(value, indent)
    if isinstance(value, list):
        return _render_list(value, indent)
    return _json_atom(value)


def render_case_json(payload: Any) -> str:
    converted = _convert_payload(payload)
    return _render_value(converted, 0) + "\n"


def format_case_json_file(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rendered = render_case_json(payload)
    before = path.read_text(encoding="utf-8")
    if before == rendered:
        return False
    path.write_text(rendered, encoding="utf-8")
    return True


def format_case_json_tree(root: Path) -> list[Path]:
    changed: list[Path] = []
    for path in sorted(root.rglob("*.json")):
        if format_case_json_file(path):
            changed.append(path)
    return changed
