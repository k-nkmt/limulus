from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

from limulus.expression_semantics import ExpressionSemantics


def test_expression_semantics_normalizes_existing_dsl_operators_and_literals() -> None:
    semantics = ExpressionSemantics()

    assert semantics.prepare_expression("score ^= 1 || put(code, $tag.) || .") == (
        "__dsl_concat__(__dsl_concat__(score != 1, put(code, '$tag.')), None)"
    )


def test_expression_semantics_normalizes_nested_put_input_format_arguments() -> None:
    semantics = ExpressionSemantics()

    assert semantics.normalize_expression("put(input(raw, comma8.), z5.)") == (
        "put(input(raw, 'comma8.'), 'z5.')"
    )
    assert semantics.normalize_expression("left >< right") == "left  @__unsupported_operator__  right"


def test_expression_semantics_classifies_ast_calls_without_string_false_positives() -> None:
    semantics = ExpressionSemantics()

    expression = "catx('-', upcase(name), 'find(value)')"
    assert semantics.function_names(expression) == ("catx", "upcase")
    assert semantics.contains_function_call(expression) is True
    assert semantics.find_unsupported_function(expression, supported_functions=("CATX", "UPCASE")) is None
    assert semantics.find_unsupported_function(expression, supported_functions=("catx",)) == "upcase"

    assert semantics.function_names("'put(value, z5.)'") == ()
    assert semantics.contains_function_call("'put(value, z5.)'") is False


def test_expression_semantics_compile_cache_returns_stable_code_without_eval_scope() -> None:
    semantics = ExpressionSemantics()

    first = semantics.compile_expression("amount + 1")
    second = semantics.compile_expression("amount + 1")

    assert first is second
    assert isinstance(first, type(compile("amount + 1", "<test>", "eval")))
    assert not hasattr(semantics, "evaluate_scalar")
    assert not hasattr(semantics, "_eval_scope_provider")
    assert not hasattr(semantics, "_context")


def test_expression_semantics_module_does_not_import_runtime() -> None:
    source = inspect.getsource(__import__("limulus.expression_semantics", fromlist=["*"]))
    tree = ast.parse(source)

    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert all(not module.startswith("limulus.runtime") for module in imported_modules)


def test_expression_semantics_all_compatibility_symbols_import() -> None:
    module = importlib.import_module("limulus.expression_semantics")

    for symbol in module.__all__:
        assert getattr(module, symbol) is not None


def test_column_and_shared_validation_layers_do_not_construct_row_expression_evaluator() -> None:
    package_root = Path(__file__).parents[1] / "limulus"
    column_source = (package_root / "column_api.py").read_text()
    coordinator_source = (package_root / "execution" / "coordinator.py").read_text()

    assert "runtime.expressions" not in column_source
    assert "runtime.row_runtime import RowRuntimeService" not in column_source
    assert "runtime.expressions" not in coordinator_source
    assert "ExpressionEvaluator(" not in coordinator_source
    assert "RowRuntimeService(" not in coordinator_source
