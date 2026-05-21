from __future__ import annotations

from .parser import ParserService, SplitStageParserService


def _extend_statement_end(dsl_text: str, end: int) -> int:
    cursor = end
    while cursor < len(dsl_text) and dsl_text[cursor].isspace():
        cursor += 1
    if cursor < len(dsl_text) and dsl_text[cursor] == ";":
        return cursor + 1
    return end


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


class DataStepBlockPreparser:
    def __init__(self, split_parser_service: SplitStageParserService | None = None) -> None:
        self._split_parser_service = split_parser_service or SplitStageParserService()

    def split(self, dsl_text: str) -> tuple[str, ...]:
        if not dsl_text.strip():
            return ()

        statement_regions = self._split_parser_service.extract_statement_regions(dsl_text)
        if statement_regions is None:
            stripped = dsl_text.strip()
            return (stripped,) if stripped else ()
        return _split_blocks_from_regions(dsl_text, statement_regions)


class DataStepBlockSplitter:
    def __init__(self, parser_service: ParserService | None = None) -> None:
        self._parser_service = parser_service or ParserService()
        self._preparser = DataStepBlockPreparser()

    def split(self, dsl_text: str) -> tuple[str, ...]:
        if not dsl_text.strip():
            return ()

        parser_blocks = self._split_with_parser(dsl_text)
        if parser_blocks is not None:
            return parser_blocks
        return self._split_with_scanner(dsl_text)

    def _split_with_parser(self, dsl_text: str) -> tuple[str, ...] | None:
        statement_regions = self._parser_service.extract_statement_regions(dsl_text)
        if statement_regions is None:
            return None
        if not statement_regions:
            return ()
        return _split_blocks_from_regions(dsl_text, statement_regions)

    def _split_with_scanner(self, dsl_text: str) -> tuple[str, ...]:
        return self._preparser.split(dsl_text)

__all__ = ["DataStepBlockPreparser", "DataStepBlockSplitter"]