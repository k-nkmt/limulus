"""ParserService facade and flat parser-result models."""

from __future__ import annotations

from typing import Any

from .models import Diagnostic
from .syntax.legacy import (
    ArrayStatementSpec,
    DataStepAst,
    DatasetReference,
    DatasetReferenceOptionSpec,
    DoStatementSpec,
    IfStatementSpec,
    ParseResult,
    ParsedStatement,
    SetStatementOptionSpec,
    StatementRegion,
    ast_statements_to_dict,
    ast_to_dict,
    statement_to_dict,
)
from .syntax.compat import project_result


def _capability_missing_diagnostic() -> Diagnostic:
    """Report an unavailable mandatory parser backend."""

    return Diagnostic(
        code="PARSE_BACKEND_CAPABILITY_MISSING",
        severity="error",
        location="",
        message="lark parser is not available; parsing cannot proceed.",
    )


def _new_program_frontend() -> Any | None:
    """One pure program frontend, or ``None`` when Lark is unavailable."""

    try:
        from .syntax.program import LarkProgramFrontend
    except Exception:  # pragma: no cover - lark-free cold import
        return None
    return LarkProgramFrontend()


def _parse_program_or_capability(frontend: Any, dsl_text: str) -> Any:
    """One pure whole-text parse; ``None`` only for the capability family.

    The facade converts ``ParserInitializationError``
    into the existing ``PARSE_BACKEND_CAPABILITY_MISSING`` diagnostic
    contract and never masks any other failure.
    """

    if frontend is None:
        return None
    try:
        from .syntax.program import ParserInitializationError
    except Exception:  # pragma: no cover - lark-free cold import
        return None
    try:
        return frontend.parse_program(dsl_text)
    except ParserInitializationError:
        return None


class LarkParserFrontend:
    """Expose flat parser results and statement regions from whole-program parsing."""

    def __init__(self) -> None:
        self._program_frontend: Any | None = _new_program_frontend()

    def parse(self, dsl_text: str) -> ParseResult:
        result = _parse_program_or_capability(self._program_frontend, dsl_text)
        if result is None:
            return ParseResult(diagnostics=(_capability_missing_diagnostic(),))
        return project_result(result)

    def extract_statement_regions(
        self, dsl_text: str
    ) -> tuple[StatementRegion, ...] | None:
        result = _parse_program_or_capability(self._program_frontend, dsl_text)
        if result is None or result.program is None:
            return None
        return tuple(
            StatementRegion(kind=statement.kind, start=statement.span.start, end=statement.span.end)
            for block in result.program.blocks
            for statement in block.statements
        )


ParserService = LarkParserFrontend
