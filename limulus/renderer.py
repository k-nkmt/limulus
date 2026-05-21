from __future__ import annotations

from typing import Any, Protocol

from .models import Diagnostic, RenderRequest
from .native_bridge import load_native_module


class DiagnosticRenderer(Protocol):
    name: str

    def render_many(self, request: RenderRequest) -> str:
        ...


class PlainTextDiagnosticRenderer:
    name = "plain"

    def render_many(self, request: RenderRequest) -> str:
        rendered: list[str] = []
        for diagnostic in request.diagnostics:
            rendered.append(self._render_one(diagnostic, fallback_source_id=request.source_id))
        return "\n\n".join(rendered)

    def _render_one(self, diagnostic: Diagnostic, *, fallback_source_id: str) -> str:
        if diagnostic.span is None or not diagnostic.source_text:
            return self._render_simple(diagnostic)

        stage_suffix = f" [stage: {diagnostic.stage}]" if diagnostic.stage else ""
        source_id = diagnostic.span.source_id or fallback_source_id
        excerpt = self._extract_line(diagnostic.source_text, diagnostic.span.line)
        if excerpt is None:
            return self._render_simple(diagnostic)

        pointer_width = self._pointer_width(diagnostic, excerpt)
        pointer = " " * max(diagnostic.span.column - 1, 0) + "^" * pointer_width
        primary_label = next((label for label in diagnostic.labels if label.kind == "primary"), None)
        if primary_label is not None and primary_label.message:
            pointer = f"{pointer} {primary_label.message}"

        lines = [
            f"{diagnostic.severity.title()}[{diagnostic.code}]{stage_suffix}: {diagnostic.message}",
            f" --> {source_id}:{diagnostic.span.line}:{diagnostic.span.column}",
            "  |",
            f"{diagnostic.span.line} | {excerpt}",
            f"  | {pointer}",
        ]
        for note in diagnostic.notes:
            lines.append(f"  = note: {note}")
        return "\n".join(lines)

    def _render_simple(self, diagnostic: Diagnostic) -> str:
        location_suffix = f" ({diagnostic.location}) - " if diagnostic.location else ""
        stage_suffix = f" [stage: {diagnostic.stage}]" if diagnostic.stage else ""
        lines = [f"{location_suffix}{diagnostic.severity.title()}\n{stage_suffix}: {diagnostic.message}"]
        for note in diagnostic.notes:
            lines.append(f"note: {note}")
        return "\n".join(lines)

    @staticmethod
    def _extract_line(source_text: str, line_number: int) -> str | None:
        if line_number <= 0:
            return None
        lines = source_text.splitlines() or [source_text]
        if line_number > len(lines):
            return None
        return lines[line_number - 1]

    @staticmethod
    def _pointer_width(diagnostic: Diagnostic, line_text: str) -> int:
        if diagnostic.span is None:
            return 1
        if diagnostic.span.end_line == diagnostic.span.line and diagnostic.span.end_column is not None:
            return max(diagnostic.span.end_column - diagnostic.span.column, 1)
        available = max(len(line_text) - diagnostic.span.column + 1, 1)
        consumed = max(diagnostic.span.end - diagnostic.span.start, 1)
        return max(min(consumed, available), 1)


class NativeAriadneDiagnosticRenderer:
    name = "native-ariadne"

    def __init__(self, native_module: Any) -> None:
        self._native_module = native_module

    def render_many(self, request: RenderRequest) -> str:
        render_fn = getattr(self._native_module, "render_diagnostics_ariadne", None)
        if not callable(render_fn):
            raise AttributeError("native renderer bridge is unavailable")
        rendered = render_fn(_render_request_to_payload(request))
        return str(rendered)


def _span_to_payload(span: Any) -> dict[str, Any] | None:
    if span is None:
        return None
    return {
        "start": span.start,
        "end": span.end,
        "line": span.line,
        "column": span.column,
        "end_line": span.end_line,
        "end_column": span.end_column,
        "source_id": span.source_id,
    }


def _diagnostic_to_payload(diagnostic: Diagnostic) -> dict[str, Any]:
    return {
        "code": diagnostic.code,
        "severity": diagnostic.severity,
        "message": diagnostic.message,
        "location": diagnostic.location,
        "stage": diagnostic.stage,
        "span": _span_to_payload(diagnostic.span),
        "labels": [
            {
                "span": _span_to_payload(label.span),
                "message": label.message,
                "kind": label.kind,
            }
            for label in diagnostic.labels
        ],
        "notes": list(diagnostic.notes),
        "source_text": diagnostic.source_text,
    }


def _render_request_to_payload(request: RenderRequest) -> dict[str, Any]:
    return {
        "source_id": request.source_id,
        "source_text": request.source_text,
        "diagnostics": [_diagnostic_to_payload(diagnostic) for diagnostic in request.diagnostics],
    }


def get_default_diagnostic_renderer() -> DiagnosticRenderer:
    native_module, _ = load_native_module()
    if native_module is not None and callable(getattr(native_module, "render_diagnostics_ariadne", None)):
        return NativeAriadneDiagnosticRenderer(native_module)
    return PlainTextDiagnosticRenderer()


def render_diagnostics(request: RenderRequest) -> str:
    plain_rendered = PlainTextDiagnosticRenderer().render_many(request)
    renderer = get_default_diagnostic_renderer()
    if isinstance(renderer, PlainTextDiagnosticRenderer):
        return plain_rendered
    try:
        rendered = renderer.render_many(request)
    except Exception:
        return plain_rendered

    if _should_append_plain_fallback(request, rendered, plain_rendered):
        return f"{rendered}\n\n{plain_rendered}"
    return rendered


def _should_append_plain_fallback(
    request: RenderRequest,
    rendered: str,
    plain_rendered: str,
) -> bool:
    if not rendered or rendered == plain_rendered:
        return False
    if "^" in rendered or "^" not in plain_rendered:
        return False
    return any(diagnostic.span is not None and diagnostic.source_text for diagnostic in request.diagnostics)


__all__ = [
    "DiagnosticRenderer",
    "NativeAriadneDiagnosticRenderer",
    "PlainTextDiagnosticRenderer",
    "get_default_diagnostic_renderer",
    "render_diagnostics",
]