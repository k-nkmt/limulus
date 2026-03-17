from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DataSetRef:
    kind: str
    location: str
    payload: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Diagnostic:
    code: str
    severity: str
    message: str
    location: str = ""
    stage: str = ""


@dataclass(frozen=True)
class CompatibilityNotice:
    id: str
    category: str
    summary: str
    link: str = ""


@dataclass(frozen=True)
class ExecuteRequest:
    dsl_text: str
    inputs: Mapping[str, DataSetRef] = field(default_factory=dict)
    output_targets: tuple[str, ...] = field(default_factory=tuple)
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecuteResponse:
    outputs: Mapping[str, DataSetRef] = field(default_factory=dict)
    outputs_arrow: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)
    notices: tuple[CompatibilityNotice, ...] = field(default_factory=tuple)

    @property
    def has_errors(self) -> bool:
        return any(diag.severity == "error" for diag in self.diagnostics)


@dataclass(frozen=True)
class OutputConversionResult:
    outputs: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    @property
    def has_errors(self) -> bool:
        return any(diag.severity == "error" for diag in self.diagnostics)


@dataclass(frozen=True)
class LogEntry:
    code: str
    severity: str
    message: str
    location: str = ""
    stage: str = ""


@dataclass
class SubmitResult:
    success: bool
    datasets: Mapping[str, Any] = field(default_factory=dict)
    log: tuple[LogEntry, ...] = field(default_factory=tuple)
    elapsed_seq: float = 0.0
    display_result: bool = field(default=True, repr=False)

    def __bool__(self) -> bool:
        return self.success

    def __repr__(self) -> str:
        if not self.display_result:
            return ""
        return (
            "SubmitResult("
            f"success={self.success}, "
            f"datasets={self.datasets}, "
            f"log={self.log}, "
            f"elapsed_seq={self.elapsed_seq}"
            ")"
        )

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        printer.text(repr(self))

    def format_log(self) -> str:
        lines: list[str] = [
            f"success: {self.success}",
            f"{round(self.elapsed_seq, 2)} seconds elapsed",
        ]
        if not self.log:
            lines.append("(no log entries)")
            return "\n".join(lines)

        for entry in self.log:
            location_suffix = f" ({entry.location}) - " if entry.location else ""
            stage_suffix = f" [stage: {entry.stage}]" if entry.stage else ""
            lines.append(f"{location_suffix}{entry.severity.title()}\n{stage_suffix}: {entry.message}")
        return "\n".join(lines)

    def print_log(self) -> None:
        print(self.format_log())


class DatasetCatalog(Mapping[str, Any]):
    def __init__(self) -> None:
        self._datasets: MutableMapping[str, Any] = {}

    def set(self, name: str, table: Any) -> None:
        self._datasets[self._normalize(name)] = table

    def update(self, datasets: Mapping[str, Any]) -> None:
        for name, table in datasets.items():
            self.set(name, table)

    def delete(self, name: str, *, missing_ok: bool = True) -> bool:
        normalized = self._normalize(name)
        if normalized in self._datasets:
            del self._datasets[normalized]
            return True
        if missing_ok:
            return False
        raise KeyError(name)

    def __getitem__(self, name: str) -> Any:
        return self._datasets[self._normalize(name)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._datasets)

    def __len__(self) -> int:
        return len(self._datasets)

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return self._normalize(name) in self._datasets

    @staticmethod
    def _normalize(name: str) -> str:
        return name.strip().lower()


__all__ = [
    "DataSetRef",
    "Diagnostic",
    "ExecuteRequest",
    "ExecuteResponse",
    "OutputConversionResult",
    "CompatibilityNotice",
    "LogEntry",
    "SubmitResult",
    "DatasetCatalog",
]
