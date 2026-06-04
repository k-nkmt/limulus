from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re
from typing import Any

from .transport import _classify_transport_input, _REJECTED_TRANSPORT_PATH, _SPECIAL_TRANSPORT_PATH, _STANDARD_TRANSPORT_PATH


_ADVANCED_RUNTIME_STATEMENT_KINDS = frozenset({"SUM", "DO", "ARRAY", "ASSIGN", "RETAIN", "STOP"})


@dataclass(frozen=True)
class CompatibilityPathPlan:
    owner: str
    selected_path: str
    path_role: str
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "selected_path": self.selected_path,
            "path_role": self.path_role,
            "reason_codes": self.reason_codes,
        }


@dataclass(frozen=True)
class RuntimeSelectionDecision:
    selected_backend: str
    normalized_mode: str
    reason_codes: tuple[str, ...]


class BackendDispatchPolicy:
    _FIRST_LAST_FLAG_PATTERN = re.compile(r"\b(?:first|last)\.[A-Za-z_]\w*", re.IGNORECASE)
    _PYTHON_LIMITED_APPLY_PATTERN = re.compile(r"\bapply\s*\(", re.IGNORECASE)
    _PYTHON_LIMITED_FORMAT_PATTERN = re.compile(r"\b(?:put|input)\s*\(", re.IGNORECASE)

    @classmethod
    def build_compatibility_path_plan(cls, ast_statements: Sequence[Any]) -> CompatibilityPathPlan:
        reason_codes: list[str] = []
        if any(getattr(statement, "kind", None) in _ADVANCED_RUNTIME_STATEMENT_KINDS for statement in ast_statements):
            reason_codes.append("PLANNER_TOGGLE_READY_STATEMENT")
        if any(cls._is_if_then_do(statement) for statement in ast_statements):
            reason_codes.append("PLANNER_TOGGLE_READY_IF_THEN_DO")

        if reason_codes:
            return CompatibilityPathPlan(
                owner="backend_dispatch_policy",
                selected_path="advanced",
                path_role="planner_toggle_ready",
                reason_codes=tuple(dict.fromkeys(reason_codes)),
            )

        return CompatibilityPathPlan(
            owner="backend_dispatch_policy",
            selected_path="basic",
            path_role="standard_path",
            reason_codes=(),
        )

    @classmethod
    def uses_advanced_compatibility_path(cls, ast_statements: Sequence[Any]) -> bool:
        return cls.build_compatibility_path_plan(ast_statements).selected_path == "advanced"

    @classmethod
    def choose_engine(cls, context: Any, preferred_backend: str, *, rust_available: bool) -> str:
        return cls.build_runtime_selection_decision(
            context,
            preferred_backend,
            rust_available=rust_available,
        ).selected_backend

    @classmethod
    def build_runtime_selection_decision(
        cls,
        context: Any,
        preferred_backend: str,
        *,
        rust_available: bool,
    ) -> RuntimeSelectionDecision:
        preferred = cls.normalize_preferred_backend(preferred_backend)
        if preferred == "python_fallback":
            return RuntimeSelectionDecision(
                selected_backend="python",
                normalized_mode="runtime_retry",
                reason_codes=("AUTO_RUNTIME_RETRY",),
            )
        if preferred == "auto" and context is None:
            return RuntimeSelectionDecision(
                selected_backend="rust" if rust_available else "python",
                normalized_mode="rust_first",
                reason_codes=(),
            )
        if preferred == "rust":
            return RuntimeSelectionDecision(
                selected_backend="rust" if rust_available else "python",
                normalized_mode="rust_first",
                reason_codes=(),
            )

        compatibility_path_plan = cls.build_compatibility_path_plan(getattr(context, "ast_statements", ()) or ())
        if cls.references_first_last_flags(context) and cls.uses_rust_standard_row_loop(context):
            return RuntimeSelectionDecision(
                selected_backend="rust" if rust_available else "python",
                normalized_mode="rust_first",
                reason_codes=(),
            )
        if cls.requires_python_limited_backend(context):
            return RuntimeSelectionDecision(
                selected_backend="python",
                normalized_mode="python_limited_facade",
                reason_codes=cls.python_limited_reason_codes(context),
            )

        if cls.uses_rust_standard_row_loop(context):
            return RuntimeSelectionDecision(
                selected_backend="rust" if rust_available else "python",
                normalized_mode="rust_first",
                reason_codes=(),
            )

        fallback_mode = (
            "planner_toggle_ready"
            if compatibility_path_plan.selected_path == "advanced"
            else "compatibility_fallback"
        )
        return RuntimeSelectionDecision(
            selected_backend="python",
            normalized_mode=fallback_mode,
            reason_codes=compatibility_path_plan.reason_codes,
        )

    @staticmethod
    def normalize_preferred_backend(preferred_backend: str) -> str:
        return preferred_backend.strip().lower()

    @classmethod
    def prefers_rust_first_execution(cls, preferred_backend: str) -> bool:
        return cls.normalize_preferred_backend(preferred_backend) in {"auto", "python", "rust"}

    @classmethod
    @classmethod
    def requires_python_limited_backend(cls, context: Any) -> bool:
        return bool(cls.python_limited_reason_codes(context))

    @classmethod
    def python_limited_reason_codes(cls, context: Any) -> tuple[str, ...]:
        ast_statements = getattr(context, "ast_statements", ()) or ()
        request = getattr(context, "request", None)
        request_options = getattr(request, "options", {}) if request is not None else {}
        dispatch_hints = request_options.get("dispatch_hints", {}) if isinstance(request_options, dict) else {}
        has_callable_format_registry = bool(
            dispatch_hints.get("has_callable_format_registry", dispatch_hints.get("has_custom_format_registry"))
        ) if isinstance(dispatch_hints, dict) else False
        reason_codes: list[str] = []
        for statement in ast_statements:
            statement_text = getattr(statement, "text", None)
            if not isinstance(statement_text, str):
                continue
            if cls._PYTHON_LIMITED_APPLY_PATTERN.search(statement_text):
                reason_codes.append("PYTHON_LIMITED_APPLY")
            if has_callable_format_registry and cls._PYTHON_LIMITED_FORMAT_PATTERN.search(statement_text):
                reason_codes.append("PYTHON_LIMITED_FORMAT_REGISTRY")
        return tuple(dict.fromkeys(reason_codes))

    @classmethod
    def references_first_last_flags(cls, context: Any) -> bool:
        ast_statements = getattr(context, "ast_statements", ()) or ()
        for statement in ast_statements:
            statement_text = getattr(statement, "text", None)
            if isinstance(statement_text, str) and cls._FIRST_LAST_FLAG_PATTERN.search(statement_text):
                return True
        return False

    @classmethod
    def uses_rust_standard_row_loop(cls, context: Any) -> bool:
        execution_plan = getattr(context, "execution_plan", None)
        if not isinstance(execution_plan, dict):
            return False

        execution_readiness = execution_plan.get("execution_readiness")
        if isinstance(execution_readiness, dict) and execution_readiness.get("decision") == "blocked":
            return False

        ast_statements = getattr(context, "ast_statements", ()) or ()
        if any(getattr(statement, "kind", None) == "SKIPPED" for statement in ast_statements):
            return False
        if not any(getattr(statement, "kind", None) in {"SET", "MERGE"} for statement in ast_statements):
            return False

        row_loop_plan = execution_plan.get("row_loop_plan")
        if not isinstance(row_loop_plan, dict):
            return False
        if row_loop_plan.get("mode") != "arrow_row_loop":
            return False
        if row_loop_plan.get("engine_mode") != "unified_row_loop":
            return False
        if row_loop_plan.get("backend_mode") != "rust_first":
            return False

        resolved_inputs = getattr(context, "resolved_inputs", {}) or {}
        return all(
            cls._supports_rust_transport(dataset_ref)
            for dataset_ref in resolved_inputs.values()
        )

    @staticmethod
    def _supports_rust_transport(dataset_ref: Any) -> bool:
        transport_path = _classify_transport_input(dataset_ref)
        if transport_path == _SPECIAL_TRANSPORT_PATH:
            return True
        if transport_path == _STANDARD_TRANSPORT_PATH:
            payload = getattr(dataset_ref, "payload", None)
            return payload is not None and hasattr(payload, "__arrow_c_stream__")
        return transport_path != _REJECTED_TRANSPORT_PATH

    @staticmethod
    def _is_if_then_do(statement: Any) -> bool:
        if getattr(statement, "kind", None) != "IF":
            return False
        if_spec = getattr(statement, "if_spec", None)
        if if_spec is None:
            return False
        if not getattr(if_spec, "is_then_do", False):
            return False
        condition = getattr(if_spec, "condition", None)
        return isinstance(condition, str) and bool(condition)