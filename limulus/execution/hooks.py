"""Macro hooks for the public execution surface."""

from __future__ import annotations

from typing import Protocol


class MacroHook(Protocol):
    """Public extension point evaluated before the one whole-program parse."""

    def evaluate(self, dsl_text: str) -> str:
        ...


class NoOpMacroHook:
    def evaluate(self, dsl_text: str) -> str:
        return dsl_text


__all__ = ["MacroHook", "NoOpMacroHook"]
