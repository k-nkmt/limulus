from __future__ import annotations

from typing import Any

from .syntax.compat import block_display_slices, project_block_displays

__all__ = ["DataStepBlockSplitter"]


def _new_program_frontend() -> Any | None:
    """One pure program frontend, or ``None`` when Lark is unavailable."""

    try:
        from .syntax.program import LarkProgramFrontend
    except Exception:  # pragma: no cover - lark-free cold import
        return None
    return LarkProgramFrontend()


def _split_blocks_from_regions(dsl_text: str, statement_regions) -> tuple[str, ...]:
    blocks: list[str] = []
    current_block_start: int | None = None
    last_statement_end: int | None = None

    for region in statement_regions:
        region_end = _extend_statement_end(dsl_text, region.end)
        if current_block_start is None:
            current_block_start = region.start
        elif region.kind == "DATA":
            previous_block = dsl_text[current_block_start:region.start].strip()
            if previous_block:
                blocks.append(previous_block)
            current_block_start = region.start

        last_statement_end = region_end
        if region.kind == "RUN":
            block_text = dsl_text[current_block_start:region_end].strip()
            if block_text:
                blocks.append(block_text)
            current_block_start = None
            last_statement_end = None

    if current_block_start is not None and last_statement_end is not None:
        trailing_block = dsl_text[current_block_start:last_statement_end].strip()
        if trailing_block:
            blocks.append(trailing_block)

    return tuple(blocks)


def _extend_statement_end(dsl_text: str, end: int) -> int:
    cursor = end
    while cursor < len(dsl_text) and dsl_text[cursor].isspace():
        cursor += 1
    if cursor < len(dsl_text) and dsl_text[cursor] == ";":
        return cursor + 1
    return end


class DataStepBlockSplitter:
    """Return DATA block source slices from whole-program parse facts.

    Parse failures use established region facts when available, otherwise
    the stripped source is returned as one block.
    """

    def __init__(self) -> None:
        self._program_frontend: Any | None = _new_program_frontend()

    def split(self, dsl_text: str) -> tuple[str, ...]:
        if not dsl_text.strip():
            return ()
        if self._program_frontend is None:
            stripped = dsl_text.strip()
            return (stripped,) if stripped else ()

        result = self._parse_program(self._program_frontend, dsl_text)
        if result is None:
            # No grammar facts are available; expose the source as one block.
            stripped = dsl_text.strip()
            return (stripped,) if stripped else ()

        if result.program is not None:
            return project_block_displays(result.program)

        projection = result.projection
        if (
            projection is not None
            and projection.source.text == dsl_text
            and projection.block_display_spans
        ):
            return block_display_slices(
                dsl_text,
                [
                    (span.start, span.end, terminated)
                    for span, terminated in projection.block_display_spans
                ],
                [(region.span.start, region.span.end) for region in projection.skip_regions],
            )
        # Region-refusing input: no display region was established — the
        # one raw source block.
        stripped = dsl_text.strip()
        return (stripped,) if stripped else ()

    @staticmethod
    def _parse_program(frontend: Any, dsl_text: str) -> Any:
        """One pure parse; ``None`` only for the capability family."""

        try:
            from .syntax.program import ParserInitializationError
        except Exception:  # pragma: no cover - lark-free cold import
            return None
        try:
            return frontend.parse_program(dsl_text)
        except ParserInitializationError:
            return None
