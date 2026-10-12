"""Durable boundary checks for removed Python row runtime internals."""

from __future__ import annotations

import ast
import io
import importlib.util
import inspect
import json
import re
import tokenize
from pathlib import Path
from typing import Iterable
from unittest.mock import patch

import pyarrow as pa
import pytest

from limulus import Session
import limulus.runtime as runtime
from limulus.backend_integration.contracts import RuntimeExecutionResult
from limulus.execution import DataStepExecutor
from limulus.execution import coordinator as coordinator_module
from limulus.models import DataSetRef, Diagnostic, ExecuteRequest
from limulus.syntax.program import LarkProgramFrontend


_LEGACY_BOUNDARY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("legacy_inputs", re.compile(r"\blegacy_inputs\b")),
    ("diagnostic_memory_rows", re.compile(r"\bdiagnostic_memory_rows\b")),
    ("from_row_list", re.compile(r"\bfrom_row_list\b")),
    ("return_outputs", re.compile(r"\breturn_outputs\b")),
    ("legacy_outputs", re.compile(r"\blegacy_outputs\b")),
    ("prepared_merge_mode", re.compile(r"\bprepared_merge_mode\b")),
    ("transport_mode", re.compile(r"\btransport_mode\b")),
    (
        "special_transport_path",
        re.compile(
            r"\b(?:_SPECIAL_TRANSPORT_PATH|SPECIAL_TRANSPORT_PATH|"
            r"special_transport|diagnostic_memory_transport)\b"
        ),
    ),
    (
        "prepared_transport_helpers",
        re.compile(
            r"\b(?:_has_prepared_transport_marker|_has_prepared_transport_rows|"
            r"_has_prepared_transport_arrow|_uses_prepared_merge_transport|"
            r"_uses_prepared_set_transport)\b"
        ),
    ),
    (
        "compatibility_path_reason",
        re.compile(r"\b(?:compatibility_path_reason|compat_reason)\b"),
    ),
    (
        "compatibility_reason_helpers",
        re.compile(
            r"\b(?:CANONICAL_COMPATIBILITY_PATH_REASONS|"
            r"LEGACY_OUTPUT_MATERIALIZATION_REASON|SourceOwnerEvidenceContract|"
            r"MetricsAdapter|canonicalize_compatibility_path_reason|"
            r"normalize_compatibility_path_reason|classify_source_access_path|"
            r"normalize_source_access_mode|build_source_owner_evidence_contract)\b"
        ),
    ),
)


def _mask_python_comments(source: str) -> str:
    masked = list(source)
    line_offsets: list[int] = [0]
    for line in source.splitlines(keepends=True):
        line_offsets.append(line_offsets[-1] + len(line))

    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            if token.type != tokenize.COMMENT:
                continue
            start_line, start_column = token.start
            end_line, end_column = token.end
            start = line_offsets[start_line - 1] + start_column
            end = line_offsets[end_line - 1] + end_column
            masked[start:end] = " " * (end - start)
    except (IndentationError, tokenize.TokenError):
        # Production modules are valid Python. The fallback keeps an invalid
        # injected fixture useful without making the absence guard crash.
        for line_start, line in enumerate(source.splitlines(keepends=True)):
            comment_index = line.find("#")
            if comment_index >= 0:
                offset = sum(len(item) for item in source.splitlines(keepends=True)[:line_start])
                end = offset + comment_index + len(line[comment_index:].rstrip("\r\n"))
                masked[offset + comment_index:end] = " " * (end - offset - comment_index)
    return "".join(masked)


def _mask_rust_comments(source: str) -> str:
    masked = list(source)
    index = 0
    block_depth = 0
    in_line_comment = False
    in_string = False
    escaped = False

    while index < len(source):
        if in_line_comment:
            if source[index] in "\r\n":
                in_line_comment = False
            else:
                masked[index] = " "
            index += 1
            continue
        if block_depth:
            if source.startswith("/*", index):
                masked[index:index + 2] = "  "
                block_depth += 1
                index += 2
            elif source.startswith("*/", index):
                masked[index:index + 2] = "  "
                block_depth -= 1
                index += 2
            else:
                if source[index] not in "\r\n":
                    masked[index] = " "
                index += 1
            continue
        if in_string:
            if escaped:
                escaped = False
            elif source[index] == "\\":
                escaped = True
            elif source[index] == '"':
                in_string = False
            index += 1
            continue
        if source.startswith("//", index):
            masked[index:index + 2] = "  "
            in_line_comment = True
            index += 2
        elif source.startswith("/*", index):
            masked[index:index + 2] = "  "
            block_depth = 1
            index += 2
        else:
            if source[index] == '"':
                in_string = True
            index += 1
    return "".join(masked)


_PYTHON_PAYLOAD_CARRIERS = frozenset({"payload", "native_payload", "serialized"})
_PYTHON_RESULT_CARRIERS = frozenset({"native_result"})
_PYTHON_PAYLOAD_KEYS = frozenset({"inputs", "outputs", "return_outputs"})
_PYTHON_RESULT_KEYS = frozenset({"outputs", "return_outputs"})


def _python_constant_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


class _PythonTransportVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[tuple[int, str]] = []

    def _carrier_kind(self, name: str) -> str | None:
        if name in _PYTHON_PAYLOAD_CARRIERS:
            return "payload"
        if name in _PYTHON_RESULT_CARRIERS:
            return "result"
        return None

    def _record(self, node: ast.AST, *, carrier: str, key: str, operation: str) -> None:
        allowed_keys = _PYTHON_PAYLOAD_KEYS if carrier == "payload" else _PYTHON_RESULT_KEYS
        if key not in allowed_keys:
            return
        if operation == "literal":
            label = f"serialized_literal_{key}"
        elif operation == "method" and carrier == "payload":
            label = f"payload_{key}"
        elif operation == "method":
            label = f"result_{key}"
        elif carrier == "payload":
            label = f"payload_{operation}_{key}"
        else:
            label = f"result_{operation}_{key}"
        self.violations.append((getattr(node, "lineno", 1), label))

    def _scan_dict_assignment(self, value: ast.AST, carrier: str, node: ast.AST) -> None:
        if not isinstance(value, ast.Dict):
            return
        for key_node in value.keys:
            key = _python_constant_string(key_node)
            if key is not None:
                self._record(
                    key_node or node,
                    carrier=carrier,
                    key=key,
                    operation="literal",
                )

    def _scan_assignment_target(self, target: ast.AST, value: ast.AST, node: ast.AST) -> None:
        if not isinstance(target, ast.Name):
            return
        carrier = self._carrier_kind(target.id)
        if carrier is not None:
            self._scan_dict_assignment(value, carrier, node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._scan_assignment_target(target, node.value, node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._scan_assignment_target(node.target, node.value, node)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name):
            carrier = self._carrier_kind(node.value.id)
            key = _python_constant_string(node.slice)
            if carrier is not None and key is not None:
                self._record(node, carrier=carrier, key=key, operation="mapping")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        function = node.func
        if isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name):
            carrier = self._carrier_kind(function.value.id)
            key = _python_constant_string(node.args[0]) if node.args else None
            if carrier is not None and key is not None and function.attr in {
                "get",
                "get_item",
                "set_item",
            }:
                self._record(node, carrier=carrier, key=key, operation="method")
        self.generic_visit(node)


def _find_python_transport_violations(source_path: str, source: str) -> list[str]:
    try:
        tree = ast.parse(source, filename=source_path)
    except SyntaxError as error:
        line = error.lineno or 1
        return [f"{source_path}:{line}:python_parse_failure"]

    visitor = _PythonTransportVisitor()
    visitor.visit(tree)
    return [f"{source_path}:{line}:{label}" for line, label in visitor.violations]


_RUST_REQUEST_RECEIVERS = frozenset({"payload", "request", "native_payload"})
_RUST_RESULT_RECEIVERS = frozenset({"result", "native_result"})
_RUST_PAYLOAD_KEYS = frozenset({"inputs", "outputs", "return_outputs"})
_RUST_RESULT_KEYS = frozenset({"outputs", "return_outputs"})


def _rust_raw_string_end(source: str, start: int) -> int | None:
    cursor = start
    if source.startswith("br", cursor):
        cursor += 2
    elif source.startswith("r", cursor):
        cursor += 1
    else:
        return None
    hash_start = cursor
    while cursor < len(source) and source[cursor] == "#":
        cursor += 1
    if cursor >= len(source) or source[cursor] != '"':
        return None
    hashes = source[hash_start:cursor]
    terminator = '"' + hashes
    end = source.find(terminator, cursor + 1)
    return len(source) if end < 0 else end + len(terminator)


def _rust_lex_transport_tokens(source: str) -> list[tuple[str, str, int]]:
    tokens: list[tuple[str, str, int]] = []
    index = 0
    line = 1
    block_depth = 0

    while index < len(source):
        if block_depth:
            if source.startswith("/*", index):
                block_depth += 1
                index += 2
            elif source.startswith("*/", index):
                block_depth -= 1
                index += 2
            else:
                line += source[index] == "\n"
                index += 1
            continue

        if source.startswith("//", index):
            index += 2
            while index < len(source) and source[index] not in "\r\n":
                index += 1
            continue
        if source.startswith("/*", index):
            block_depth = 1
            index += 2
            continue

        if source[index].isspace():
            line += source[index] == "\n"
            index += 1
            continue

        raw_end = _rust_raw_string_end(source, index)
        if raw_end is not None:
            line += source[index:raw_end].count("\n")
            index = raw_end
            continue

        if source[index] == '"':
            token_line = line
            index += 1
            value: list[str] = []
            while index < len(source):
                character = source[index]
                if character == "\\" and index + 1 < len(source):
                    value.append(source[index + 1])
                    index += 2
                    continue
                if character == '"':
                    index += 1
                    break
                value.append(character)
                line += character == "\n"
                index += 1
            tokens.append(("string", "".join(value), token_line))
            continue

        if source[index] == "'":
            # Rust lifetimes (`'_`, `'py`, `'static`) do not have a closing
            # quote. Skip only the apostrophe so the following identifier is
            # still lexed; skip the whole token only for a char literal.
            char_end = index + 1
            if char_end < len(source) and source[char_end] == "\\":
                char_end += 2
            else:
                char_end += 1
            if char_end < len(source) and source[char_end] == "'":
                index = char_end + 1
            else:
                index += 1
            continue

        if source[index].isalpha() or source[index] == "_":
            token_line = line
            start = index
            index += 1
            while index < len(source) and (source[index].isalnum() or source[index] == "_"):
                index += 1
            tokens.append(("ident", source[start:index], token_line))
            continue

        if source[index] in ".(),?":
            tokens.append(("punct", source[index], line))
        index += 1

    return tokens


def _find_rust_transport_violations(source_path: str, source: str) -> list[str]:
    tokens = _rust_lex_transport_tokens(source)
    violations: list[str] = []
    for index in range(len(tokens) - 4):
        receiver_kind, receiver, line = tokens[index]
        if receiver_kind != "ident":
            continue
        if tokens[index + 1][:2] != ("punct", "."):
            continue
        method_kind, method, _ = tokens[index + 2]
        if method_kind != "ident" or method not in {"get_item", "set_item"}:
            continue
        if tokens[index + 3][:2] != ("punct", "("):
            continue
        key_kind, key, _ = tokens[index + 4]
        if key_kind != "string":
            continue
        if receiver in _RUST_REQUEST_RECEIVERS and key in _RUST_PAYLOAD_KEYS:
            violations.append(f"{source_path}:{line}:rust_payload_{key}")
        elif receiver in _RUST_RESULT_RECEIVERS and key in _RUST_RESULT_KEYS:
            violations.append(f"{source_path}:{line}:rust_result_{key}")
    return violations


def _find_legacy_boundary_violations(
    sources: Iterable[tuple[str, str]],
) -> list[str]:
    violations: list[str] = []
    for source_path, source in sources:
        if source_path.endswith(".py"):
            scan_source = _mask_python_comments(source)
        elif source_path.endswith(".rs"):
            scan_source = _mask_rust_comments(source)
        else:
            continue
        for label, pattern in _LEGACY_BOUNDARY_PATTERNS:
            for match in pattern.finditer(scan_source):
                line = source.count("\n", 0, match.start()) + 1
                violations.append(f"{source_path}:{line}:{label}")
        if source_path.endswith(".py"):
            violations.extend(_find_python_transport_violations(source_path, source))
        elif source_path.endswith(".rs"):
            violations.extend(_find_rust_transport_violations(source_path, source))
    return violations


def _production_source_items() -> Iterable[tuple[str, str]]:
    repository_root = Path(__file__).resolve().parents[1]
    source_roots = (
        repository_root / "limulus",
        repository_root / "native" / "limulus_native" / "src",
    )
    for source_root in source_roots:
        for source_path in sorted(source_root.rglob("*")):
            if source_path.suffix not in {".py", ".rs"}:
                continue
            yield (
                str(source_path.relative_to(repository_root)),
                source_path.read_text(encoding="utf-8"),
            )


def test_python_runtime_modules_are_removed() -> None:
    for module_name in (
        "limulus.runtime.row_runtime",
        "limulus.runtime.conditions",
        "limulus.runtime.expressions",
        "limulus.runtime.arrow_cursor",
    ):
        assert importlib.util.find_spec(module_name) is None


def test_runtime_package_exposes_only_arrow_compatibility_helpers() -> None:
    removed = {
        "ConditionEvaluator",
        "ExpressionEvaluator",
        "FunctionArgumentError",
        "FunctionRegistryService",
        "LoopControlService",
        "RetainArrayRuntimeService",
        "RowRuntimeService",
        "RowView",
        "RuntimeContext",
        "RuntimeExecutionError",
        "ArrowRowCursorProtocol",
        "ArrowTableRowCursor",
        "create_arrow_row_cursor",
    }

    assert removed.isdisjoint(runtime.__all__)
    for name in removed:
        assert not hasattr(runtime, name)


def test_coordinator_and_column_helper_have_no_python_runtime_construction() -> None:
    from limulus import column_api
    from limulus.execution import coordinator

    for module in (column_api, coordinator):
        source = inspect.getsource(module)
        assert "runtime.row_runtime" not in source
        assert "runtime.expressions" not in source
        assert "ExpressionEvaluator(" not in source
        assert "RowRuntimeService(" not in source


def test_compatibility_surface_has_one_pipeline_owner() -> None:
    """Record the compatibility-surface inventory before the coordinator became internal."""

    from limulus.execution import coordinator as coordinator_module
    from limulus.execution import output_handoff as output_handoff_module

    repository_root = Path(__file__).resolve().parents[1]
    coordinator_path = repository_root / "limulus" / "execution" / "coordinator.py"
    output_handoff_path = repository_root / "limulus" / "execution" / "output_handoff.py"
    compatibility_path = (
        repository_root / "limulus" / "execution" / "compatibility.py"
    )

    def module_definitions(source: str, name: str) -> list[ast.AST]:
        tree = ast.parse(source)
        return [
            node
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ]

    # The unified-parser pipeline shim module is gone and must stay gone.
    assert importlib.util.find_spec("limulus.execution.pipeline") is None
    assert not (repository_root / "limulus" / "execution" / "pipeline.py").exists()

    coordinator_source = coordinator_path.read_text(encoding="utf-8")
    output_handoff_source = output_handoff_path.read_text(encoding="utf-8")
    compatibility_source = compatibility_path.read_text(encoding="utf-8")
    for helper_name in ("_collect_compatibility_notices",):
        assert module_definitions(coordinator_source, helper_name) == []
        assert len(module_definitions(compatibility_source, helper_name)) == 1
        from limulus.execution import compatibility as compatibility_module

        assert getattr(coordinator_module, helper_name) is getattr(
            compatibility_module, helper_name
        )

    assert len(module_definitions(output_handoff_source, "_LazyArrowOutputs")) == 1
    assert hasattr(output_handoff_module, "_LazyArrowOutputs")

    # The coordinator is internalized: the public package must not retain
    # the old advanced export or constructor name.
    from limulus import execution as execution_package

    assert not hasattr(execution_package, "ExecutionPipelineCoordinator")
    executor = coordinator_module.DataStepExecutor(runtime_backend="rust")
    assert not hasattr(executor, "_pipeline")
    assert not hasattr(executor, "_block_splitter")


def test_moved_helpers_have_one_production_owner() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    expected_owners = {
        "_collect_compatibility_notices": "limulus/execution/compatibility.py",
        "_LazyArrowOutputs": "limulus/execution/output_handoff.py",
    }
    definitions: dict[str, list[str]] = {name: [] for name in expected_owners}
    defined_names: set[str] = set()

    for source_path in sorted((repository_root / "limulus").rglob("*.py")):
        relative_path = source_path.relative_to(repository_root).as_posix()
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(
                node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                continue
            defined_names.add(node.name)
            if node.name in definitions:
                definitions[node.name].append(relative_path)

    assert definitions == {
        name: [owner] for name, owner in expected_owners.items()
    }

    # The legacy text-extraction validators were deleted with the legacy
    # request contract; the symbols must not come back.
    assert defined_names.isdisjoint(
        {
            "_validate_reserved_output_targets",
            "_validate_input_dataset_refs",
            "_validate_static_column_refs",
            "_validate_statement_column_refs",
        }
    )


def test_legacy_engine_and_fallback_symbols_are_absent_from_python_package() -> None:
    """Keep deleted row-engine and fallback metadata from returning silently."""

    package_root = Path(__file__).resolve().parents[1] / "limulus"
    forbidden_symbols = (
        "PythonRuntimeBackend",
        "PythonBackendExecutionService",
        "RuntimeBackendSelector",
        "ProgramExecutionService",
        "python_fallback",
        "python_limited_mode",
        "RowRuntimeService",
        "ExpressionEvaluator",
        "ArrowRowCursorProtocol",
        "create_arrow_row_cursor",
        "FunctionRegistryService",
        "RetainArrayRuntimeService",
        "LoopControlService",
    )

    violations = []
    for source_path in package_root.rglob("*.py"):
        source = source_path.read_text(encoding="utf-8")
        for symbol in forbidden_symbols:
            if symbol in source:
                violations.append(f"{source_path.relative_to(package_root.parent)}: {symbol}")

    assert violations == [], "legacy runtime/fallback symbols found: " + ", ".join(violations)


def test_legacy_native_transport_contract_is_absent_from_production_sources() -> None:
    violations = _find_legacy_boundary_violations(_production_source_items())

    assert violations == [], "legacy native transport symbols found: " + ", ".join(violations)


def test_prepared_marker_literals_are_owned_by_provenance_module() -> None:
    marker_literals = ("#prepared_set_rows", "#prepared_merge_rows")
    violations: list[str] = []
    for source_path, source in _production_source_items():
        if source_path == "limulus/backend_integration/transport.py":
            continue
        for marker in marker_literals:
            if marker in source:
                violations.append(f"{source_path}: {marker}")

    assert violations == [], "prepared marker literals escaped provenance owner: " + ", ".join(violations)


def test_callable_format_and_informat_registration_remains_unsupported() -> None:
    session = Session(backend="rust")

    with pytest.raises(ValueError, match="Callable format registration is not supported"):
        session.register_format("tag", lambda value: f"TAG-{value}")
    with pytest.raises(ValueError, match="Callable informat registration is not supported"):
        session.register_informat("pipeint", lambda value: value, kind="integer")


def test_case_manifests_have_no_legacy_engine_or_fallback_metadata() -> None:
    cases_root = Path(__file__).resolve().parent / "cases"
    forbidden_keys = {"backend", "backends", "runtime_backend", "engine", "engines"}
    forbidden_markers = ("python_fallback", "python_limited_mode", "fallback_success")
    violations: list[str] = []

    for manifest_path in cases_root.rglob("case.json"):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw = manifest_path.read_text(encoding="utf-8")
        for marker in forbidden_markers:
            if marker in raw:
                violations.append(f"{manifest_path.relative_to(cases_root)}: {marker}")

        variants = payload.get("variants", ()) if isinstance(payload, dict) else ()
        records = (payload, *variants) if isinstance(variants, list) else (payload,)
        for record in records:
            if not isinstance(record, dict):
                continue
            for key in forbidden_keys.intersection(record):
                violations.append(f"{manifest_path.relative_to(cases_root)}: {key}")

    assert violations == [], "legacy case metadata found: " + ", ".join(violations)


def test_legacy_boundary_guard_rejects_injected_payload_key_but_ignores_comment() -> None:
    fixture = """
# payload.get_item(\"return_outputs\") is only a comment.
def execute(payload):
    return payload.get_item(\"inputs\")
"""

    violations = _find_legacy_boundary_violations((("fixture.py", fixture),))

    assert any("fixture.py" in violation and "payload_inputs" in violation for violation in violations)
    assert not any("return_outputs" in violation for violation in violations)


def test_legacy_boundary_guard_detects_runtime_payload_key_mutation() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    source_path = repository_root / "native" / "limulus_native" / "src" / "runtime" / "mod.rs"
    source = source_path.read_text(encoding="utf-8")
    needle = 'get_item("input_streams")'
    replacement = 'get_item("inputs")'
    first_index = source.find(needle)

    assert first_index >= 0
    mutated = source[:first_index] + replacement + source[first_index + len(needle):]
    assert mutated[first_index:first_index + len(replacement)] == replacement

    violations = _find_rust_transport_violations(
        str(source_path.relative_to(repository_root)),
        mutated,
    )

    assert any("rust_payload_inputs" in violation for violation in violations), violations


@pytest.mark.parametrize(
    ("label", "snippet"),
    (
        ("payload_mapping_inputs", 'return payload["inputs"]'),
        ("payload_mapping_return_outputs", 'return payload["return_outputs"]'),
        ("payload_mapping_outputs", 'return payload["outputs"]'),
        ("serialized_literal_inputs", 'payload = {"inputs": streams}'),
        ("serialized_literal_outputs", 'payload = {"outputs": outputs}'),
    ),
)
def test_legacy_boundary_guard_rejects_direct_mapping_and_literal_forms(
    label: str,
    snippet: str,
) -> None:
    fixture = f"# {snippet}\ndef build_payload(payload):\n    {snippet}\n"

    violations = _find_legacy_boundary_violations((("fixture.py", fixture),))

    assert any(f"fixture.py" in violation and label in violation for violation in violations)


def test_legacy_boundary_guard_rejects_nonfirst_payload_literal_key() -> None:
    fixture = 'def build_payload(streams):\n    payload = {"safe": 1, "inputs": streams}\n'

    violations = _find_legacy_boundary_violations((("fixture.py", fixture),))

    assert any("fixture.py" in violation and "serialized_literal_inputs" in violation for violation in violations)


def test_syntax_aware_transport_guard_handles_multiline_and_nontransport_cases() -> None:
    positive_fixtures = (
        (
            "python_multiline_payload",
            """
def build(payload):
    return (
        payload
        .get(
            "inputs",
        )
    )
""",
            "payload_inputs",
        ),
        (
            "python_result_subscript",
            """
def read(native_result):
    return native_result[
        "outputs"
    ]
""",
            "result_mapping_outputs",
        ),
        (
            "rust_multiline_payload",
            """
fn build(payload: &PyDict) {
    payload
        .get_item(
            "inputs",
        );
}
""",
            "rust_payload_inputs",
        ),
        (
            "rust_multiline_result",
            """
fn build(result: &PyDict) {
    result
        .set_item(
            "outputs",
            output_streams,
        );
}
""",
            "rust_result_outputs",
        ),
    )

    for source_path, fixture, label in positive_fixtures:
        violations = _find_legacy_boundary_violations(((source_path + ".py", fixture),))
        if source_path.startswith("rust_"):
            violations = _find_legacy_boundary_violations(((source_path + ".rs", fixture),))
        assert any(label in violation for violation in violations), (source_path, violations)

    negative_fixtures = (
        (
            "python_comment.py",
            '# payload.get("inputs")\n# native_result["outputs"]\n',
        ),
        (
            "python_string.py",
            'source = "payload.get(\\"inputs\\")"\n'
            'description = "native_result[\\"outputs\\"]"\n',
        ),
        (
            "python_nontransport.py",
            'records = {"inputs": streams, "outputs": rows}\n'
            'value = records.get("inputs")\n',
        ),
        (
            "rust_comment.rs",
            '// payload.get_item("inputs")\n// result.set_item("outputs", rows)\n',
        ),
        (
            "rust_string.rs",
            'let source = "payload.get_item(\\"inputs\\")";\n',
        ),
        (
            "rust_nontransport.rs",
            'let table = source_table.get_item("inputs");\n',
        ),
    )

    for source_path, fixture in negative_fixtures:
        assert _find_legacy_boundary_violations(((source_path, fixture),)) == [], source_path


def test_python_transport_guard_reports_parse_failure_explicitly() -> None:
    violations = _find_legacy_boundary_violations((("broken.py", "payload = {"),))

    assert any("broken.py" in violation and "python_parse_failure" in violation for violation in violations)


# --- typed facade behavior guards (moved from the removed facade test file) -------


def _boundary_input() -> DataSetRef:
    return DataSetRef(
        kind="arrow_table",
        location="dataset://input",
        payload=pa.table({"id": [1]}),
    )


def test_facade_keeps_parser_projection_without_legacy_composition() -> None:
    executor = DataStepExecutor(runtime_backend="rust")

    assert isinstance(executor._program_frontend, LarkProgramFrontend)
    assert not hasattr(executor, "_pipeline")
    assert not hasattr(executor, "_block_splitter")


def test_invalid_block_uses_one_frontend_parse_without_fallback(monkeypatch) -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    frontend = executor._program_frontend
    calls: list[str] = []
    original_parse = frontend.parse_program

    def parse_once(dsl_text: str, *, source_id: str = "<dsl>"):
        calls.append(dsl_text)
        return original_parse(dsl_text, source_id=source_id)

    monkeypatch.setattr(frontend, "parse_program", parse_once)

    response = executor.execute(
        ExecuteRequest(dsl_text="data out; ???; run;", inputs={"input": _boundary_input()})
    )

    assert response.has_errors is True
    assert len(calls) == 1
    assert response.diagnostics[0].code == "PARSE_UNSUPPORTED_STATEMENT"


def test_multiple_blocks_share_one_whole_program_parse(monkeypatch) -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    frontend = executor._program_frontend
    calls: list[str] = []
    original_parse = frontend.parse_program

    def parse_once(dsl_text: str, *, source_id: str = "<dsl>"):
        calls.append(dsl_text)
        return original_parse(dsl_text, source_id=source_id)

    monkeypatch.setattr(frontend, "parse_program", parse_once)

    response = executor.execute(
        ExecuteRequest(
            dsl_text=(
                "data first; set input; marker=1; run; "
                "data second; set input; marker=2; run;"
            ),
            inputs={"input": _boundary_input()},
        )
    )

    assert response.has_errors is False
    assert len(calls) == 1
    assert calls[0].lower().startswith("data first;")
    assert "data second;" in calls[0].lower()
    assert tuple(response.outputs) == ("first", "second")
    assert response.outputs["first"].payload.to_pylist() == [{"id": 1, "marker": 1.0}]
    assert response.outputs["second"].payload.to_pylist() == [{"id": 1, "marker": 2.0}]


def test_typed_facade_preserves_stage_order(monkeypatch) -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    events: list[str] = []

    def record_method(owner, name: str, event: str) -> None:
        original = getattr(owner, name)

        def wrapped(*args, **kwargs):
            events.append(event)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, wrapped)

    record_method(executor._macro_hook, "evaluate", "macro")
    record_method(executor._program_frontend, "parse_program", "parse")
    record_method(executor._validation_service, "validate_typed", "validate")
    record_method(executor._plan_service, "plan_typed_helpers", "helpers")
    record_method(executor._input_preparation_service, "resolve_typed", "resolve")
    record_method(executor._input_preparation_service, "prepare_typed", "prepare")
    record_method(executor._output_handoff_service, "plan_typed_shaping", "shape")
    record_method(executor._runtime_handoff_service, "decide_strategy", "strategy")
    record_method(executor._plan_service, "build_typed", "plan")
    record_method(executor._rust_backend, "execute_typed", "runtime")
    record_method(executor._output_handoff_service, "finalize_typed", "output")

    response = executor.execute(
        ExecuteRequest(dsl_text="data out; set input; run;", inputs={"input": _boundary_input()})
    )

    assert response.has_errors is False
    assert events == [
        "macro",
        "parse",
        "validate",
        "helpers",
        "resolve",
        "prepare",
        "shape",
        "strategy",
        "plan",
        "runtime",
        "output",
    ]


def test_facade_passes_prior_generated_catalog_to_the_next_request(monkeypatch) -> None:
    """The typed program runner receives a fresh generated-catalog snapshot."""

    executor = DataStepExecutor(runtime_backend="rust")
    program_requests = []
    original_execute = executor._execute_typed_program

    def capture_request(request):
        program_requests.append(request)
        return original_execute(request)

    monkeypatch.setattr(executor, "_execute_typed_program", capture_request)

    first = executor.execute(
        ExecuteRequest(
            dsl_text="data staged; set input; run;",
            inputs={"input": _boundary_input()},
        )
    )
    second = executor.execute(
        ExecuteRequest(dsl_text="data out; set staged; run;")
    )

    assert first.has_errors is False
    assert second.has_errors is False
    assert tuple(program_requests[0].generated_tables) == ()
    assert tuple(program_requests[1].generated_tables) == ("staged",)
    assert second.outputs["out"].payload.to_pylist() == [{"id": 1}]


def test_typed_handoff_invokes_native_backend_once() -> None:
    executor = DataStepExecutor(runtime_backend="rust")

    with patch.object(
        executor._rust_backend,
        "execute_typed",
        wraps=executor._rust_backend.execute_typed,
    ) as native_execute:
        response = executor.execute(
            ExecuteRequest(
                dsl_text="data out; set input; run;",
                inputs={"input": _boundary_input()},
            )
        )

    assert response.has_errors is False
    native_execute.assert_called_once()


def test_pipeline_runtime_target_failure_is_fail_closed_and_invoked_once(monkeypatch) -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    calls = []
    partial_output = DataSetRef(
        kind="arrow_table",
        location="dataset://out",
        payload=pa.table({"id": [1]}),
    )
    target_diagnostic = Diagnostic(
        code="RUNTIME_RUST_NATIVE_EXECUTION_FAILED",
        severity="error",
        location="dataset:out",
        message="native output target contract failed",
    )

    def fail_target(context):
        calls.append(context)
        return RuntimeExecutionResult(
            outputs={"out": partial_output},
            diagnostics=(target_diagnostic,),
            actual_engine="rust",
            native_invoked=True,
            phase_metrics={"row_execution_ms": 1.0},
        )

    monkeypatch.setattr(executor._rust_backend, "execute_typed", fail_target)
    response = executor.execute(
        ExecuteRequest(
            dsl_text="data out; set input; run;",
            inputs={"input": _boundary_input()},
        )
    )

    assert len(calls) == 1
    assert response.has_errors is True
    assert response.outputs == {}
    assert response.outputs_arrow == {}
    assert response.diagnostics[0].code == target_diagnostic.code
    assert response.diagnostics[0].stage == "execute"
    assert response.diagnostics[0].location == "dataset:out"
    assert "out" not in executor._generated_tables


def test_later_failure_keeps_prefix_catalog_observation_and_notice() -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    response = executor.execute(
        ExecuteRequest(
            dsl_text=(
                "data stage1; set input; value = .; output stage1; run; "
                "data stage2; set missing; output stage2; run;"
            ),
            inputs={"input": _boundary_input()},
        )
    )

    assert response.has_errors is True
    assert response.diagnostics[0].stage == "validate"
    assert set(executor._generated_tables) == {"stage1"}
    assert [notice.id for notice in response.notices] == [
        "COMPAT_MISSING_VALUE_SEMANTICS"
    ]
    assert executor.last_strategy_decision is not None
    assert executor.last_runtime_backend == "rust"


def test_facade_commits_success_and_error_results_once() -> None:
    success_executor = DataStepExecutor(runtime_backend="rust")
    with patch.object(
        success_executor,
        "_commit_pipeline_result",
        wraps=success_executor._commit_pipeline_result,
    ) as commit:
        response = success_executor.execute(
            ExecuteRequest(
                dsl_text="data out; set input; run;",
                inputs={"input": _boundary_input()},
            )
        )

    assert response.has_errors is False
    commit.assert_called_once()

    error_executor = DataStepExecutor(runtime_backend="rust")
    with patch.object(
        error_executor,
        "_commit_pipeline_result",
        wraps=error_executor._commit_pipeline_result,
    ) as commit:
        response = error_executor.execute(
            ExecuteRequest(
                dsl_text="data out; set missing; run;",
                inputs={"input": _boundary_input()},
            )
        )

    assert response.has_errors is True
    commit.assert_called_once()
    assert error_executor._generated_tables == {}


def test_facade_composes_stage_owner_dependencies_without_thin_adapters() -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    coordinator_source = inspect.getsource(coordinator_module.DataStepExecutor)

    assert not hasattr(executor, "_pipeline")
    assert not hasattr(executor, "_block_splitter")
    assert "_ExecutionPipelineCoordinator" not in coordinator_source
    assert "SplitStageParserService" not in coordinator_source
    for removed in (
        "_resolve_inputs",
        "_prepare_runtime_evaluations",
        "_split_data_step_blocks",
        "_dictionary_input_cache",
    ):
        assert f"def {removed}" not in coordinator_source
        assert removed not in coordinator_source


def test_pipeline_validation_short_circuits_input_plan_and_runtime() -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    request = ExecuteRequest(
        dsl_text="data out; set source; keep missing; run;",
        inputs={
            "source": DataSetRef(
                kind="arrow_table",
                location="dataset://source",
                payload=pa.table({"id": [1]}),
            )
        },
    )

    with (
        patch.object(
            executor._input_preparation_service,
            "resolve_typed",
            wraps=executor._input_preparation_service.resolve_typed,
        ) as resolve_typed,
        patch.object(
            executor._plan_service,
            "plan_typed_helpers",
            wraps=executor._plan_service.plan_typed_helpers,
        ) as plan_helpers,
        patch.object(
            executor._plan_service,
            "build_typed",
            wraps=executor._plan_service.build_typed,
        ) as build_plan,
        patch.object(
            executor._rust_backend,
            "execute_typed",
            wraps=executor._rust_backend.execute_typed,
        ) as runtime,
    ):
        response = executor.execute(request)

    assert response.has_errors is True
    assert response.diagnostics[0].code == "VALIDATE_COLUMN_NOT_FOUND"
    assert response.diagnostics[0].stage == "validate"
    resolve_typed.assert_not_called()
    plan_helpers.assert_not_called()
    build_plan.assert_not_called()
    runtime.assert_not_called()


def test_coordinator_no_longer_contains_validation_owner() -> None:
    source = inspect.getsource(coordinator_module)

    assert "def _validate_ast_block" not in source
    assert "def _validate_reserved_output_targets" not in source
    assert "def _validate_input_dataset_refs" not in source
    assert "def _validate_static_column_refs" not in source
    assert "def _validate_statement_column_refs" not in source


def test_public_typed_execution_uses_typed_output_plan_owner() -> None:
    executor = DataStepExecutor(runtime_backend="rust")
    source = DataSetRef(
        kind="arrow_table",
        location="dataset://source",
        payload=pa.table({"id": [1]}),
    )
    with patch.object(
        executor._output_handoff_service,
        "plan_typed",
        wraps=executor._output_handoff_service.plan_typed,
    ) as plan_call:
        response = executor.execute(
            ExecuteRequest(
                dsl_text="data out; set source; run;",
                inputs={"source": source},
            )
        )

    assert not response.has_errors
    assert plan_call.call_count == 1
