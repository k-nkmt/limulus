from __future__ import annotations

from types import MappingProxyType

import pytest

from limulus.host_functions import (
    HostFunctionResolver,
    capture_host_function_context,
)
from limulus.models import HostFunctionContext


def _increment(value: object) -> object:
    return value + 1  # type: ignore[operator]


def test_capture_keeps_only_callable_values_in_caller_snapshots() -> None:
    caller_locals = {
        "callable_local": _increment,
        "plain_local": 1,
        "none_local": None,
    }
    caller_globals = {
        "callable_global": str,
        "plain_global": "not callable",
    }

    captured = capture_host_function_context(caller_locals, caller_globals)

    assert captured.diagnostics == ()
    assert captured.context.caller_locals == {"callable_local": _increment}
    assert captured.context.caller_globals == {"callable_global": str}
    assert isinstance(captured.context.caller_locals, MappingProxyType)
    assert isinstance(captured.context.caller_globals, MappingProxyType)


def test_context_mappings_are_immutable_and_snapshotted() -> None:
    functions = {"increment": _increment}
    caller_locals = {"local": _increment}

    captured = capture_host_function_context(caller_locals, {}, functions=functions)
    functions["later"] = str
    caller_locals["changed"] = str

    assert captured.context.explicit == {"increment": _increment}
    assert captured.context.caller_locals == {"local": _increment}
    with pytest.raises(TypeError):
        captured.context.explicit["new"] = str  # type: ignore[index]


def test_context_does_not_retain_a_frame_object() -> None:
    captured = capture_host_function_context(locals(), globals())

    assert not any(type(value).__name__ == "frame" for value in captured.context.explicit.values())
    assert not any(type(value).__name__ == "frame" for value in captured.context.caller_locals.values())
    assert not any(type(value).__name__ == "frame" for value in captured.context.caller_globals.values())
    assert not hasattr(captured.context, "frame")


@pytest.mark.parametrize(
    ("functions", "reason"),
    (
        ([_increment], "mapping"),
        ({1: _increment}, "string key"),
        ({"increment": 1}, "callable"),
    ),
)
def test_invalid_explicit_functions_return_diagnostic_and_empty_context(
    functions: object,
    reason: str,
) -> None:
    captured = capture_host_function_context(
        {"caller": _increment},
        {"global": str},
        functions=functions,
    )

    assert captured.context == HostFunctionContext()
    assert captured.diagnostics
    assert captured.diagnostics[0].code == "RUNTIME_APPLY_FUNCTIONS_INVALID"
    assert reason in captured.diagnostics[0].message


def test_each_capture_is_request_scoped_and_does_not_share_mutable_state() -> None:
    first = capture_host_function_context({"first": _increment}, {})
    second = capture_host_function_context({"second": str}, {})

    assert first.context.caller_locals == {"first": _increment}
    assert second.context.caller_locals == {"second": str}
    assert first.context.caller_locals is not second.context.caller_locals


def test_resolver_uses_reserved_builtin_before_every_shadowing_source() -> None:
    def shadowed(value: object) -> object:
        return value

    context = capture_host_function_context(
        {"len": shadowed},
        {"len": shadowed},
        functions={"len": shadowed},
    ).context

    result = HostFunctionResolver().resolve(("len",), context)

    assert result.diagnostics == ()
    assert result.registry["len"] is len


def test_resolver_priority_is_explicit_then_local_then_global() -> None:
    def explicit_target(value: object) -> object:
        return value

    def local_target(value: object) -> object:
        return value

    def global_target(value: object) -> object:
        return value

    context = capture_host_function_context(
        {"local_target": local_target, "global_target": local_target},
        {"global_target": global_target},
        functions={"explicit_target": explicit_target, "global_target": explicit_target},
    ).context

    result = HostFunctionResolver().resolve(
        ("explicit_target", "local_target", "global_target"),
        context,
    )

    assert result.diagnostics == ()
    assert result.registry == {
        "explicit_target": explicit_target,
        "local_target": local_target,
        "global_target": explicit_target,
    }


def test_resolver_prefers_caller_locals_over_caller_globals_for_same_target() -> None:
    def local_target(value: object) -> object:
        return "local"

    def global_target(value: object) -> object:
        return "global"

    context = capture_host_function_context(
        {"shared_target": local_target},
        {"shared_target": global_target},
    ).context

    result = HostFunctionResolver().resolve(("shared_target",), context)

    assert result.diagnostics == ()
    assert result.registry["shared_target"] is local_target


def test_resolver_resolves_qualified_module_target_after_scope_snapshots() -> None:
    context = capture_host_function_context({}, {}).context

    result = HostFunctionResolver().resolve(("math.sqrt",), context)

    assert result.diagnostics == ()
    assert result.registry["math.sqrt"](9) == 3


def test_resolver_explicit_mapping_can_override_a_qualified_target() -> None:
    def explicit_sqrt(value: object) -> object:
        return value

    context = capture_host_function_context({}, {}, functions={"math.sqrt": explicit_sqrt}).context

    result = HostFunctionResolver().resolve(("math.sqrt",), context)

    assert result.diagnostics == ()
    assert result.registry["math.sqrt"] is explicit_sqrt


def test_resolver_prefers_caller_scope_for_qualified_target_before_module_import() -> None:
    def local_sqrt(value: object) -> object:
        return "local"

    def global_sqrt(value: object) -> object:
        return "global"

    context = capture_host_function_context(
        {"math.sqrt": local_sqrt},
        {"math.sqrt": global_sqrt},
    ).context

    result = HostFunctionResolver().resolve(("math.sqrt",), context)

    assert result.diagnostics == ()
    assert result.registry["math.sqrt"] is local_sqrt


def test_resolver_rejects_dunder_qualified_attribute() -> None:
    context = capture_host_function_context({}, {}).context

    result = HostFunctionResolver().resolve(("math.__dict__",), context)

    assert result.registry == {}
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "RUNTIME_APPLY_TARGET_UNRESOLVED"
    assert "math.__dict__" in diagnostic.message
    assert "source=qualified module" in diagnostic.message
    assert "dunder" in diagnostic.message


def test_resolver_reports_qualified_non_callable_as_invalid() -> None:
    context = capture_host_function_context({}, {}).context

    result = HostFunctionResolver().resolve(("math.pi",), context)

    assert result.registry == {}
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "RUNTIME_APPLY_TARGET_INVALID"
    assert "math.pi" in diagnostic.message
    assert "source=qualified module" in diagnostic.message
    assert "not callable" in diagnostic.message


def test_resolver_reports_missing_target_with_target_and_source_reason() -> None:
    context = capture_host_function_context({}, {}).context

    result = HostFunctionResolver().resolve(("missing_apply_target",), context)

    assert result.registry == {}
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "RUNTIME_APPLY_TARGET_UNRESOLVED"
    assert "missing_apply_target" in diagnostic.message
    assert "source=resolver" in diagnostic.message
    assert "reason" in diagnostic.message


def test_resolver_is_fail_closed_when_any_target_fails() -> None:
    context = capture_host_function_context({}, {}, functions={"len": len}).context

    result = HostFunctionResolver().resolve(("len", "missing_apply_target"), context)

    assert result.registry == {}
    assert len(result.diagnostics) == 1
    assert "missing_apply_target" in result.diagnostics[0].message


def test_resolver_deduplicates_target_names_in_registry_and_diagnostics() -> None:
    context = capture_host_function_context({}, {}).context

    result = HostFunctionResolver().resolve(("len", "len", "len"), context)

    assert result.diagnostics == ()
    assert tuple(result.registry) == ("len",)
