from __future__ import annotations

"""Compatibility package for the remaining advanced execution imports."""

def __getattr__(name: str):
    if name in {"DataStepExecutor", "FormatSupportResult", "RuntimeRequirements"}:
        from ..execution import DataStepExecutor, FormatSupportResult, RuntimeRequirements

        return {
            "DataStepExecutor": DataStepExecutor,
            "FormatSupportResult": FormatSupportResult,
            "RuntimeRequirements": RuntimeRequirements,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DataStepExecutor",
    "FormatSupportResult",
    "RuntimeRequirements",
]
