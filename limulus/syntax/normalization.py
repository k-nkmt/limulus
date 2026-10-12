"""Position-preserving normalization of analysis text."""

from __future__ import annotations

import re

__all__ = ("initial_analysis_text",)


def _skip_quoted(text: str, start: int) -> int:
    """Skip one quoted region with the legacy scanner quote rules.

    Mirrors the existing quoted-region scan of the initial lexical apply
    inventory (``limulus/syntax/expressions.py`` ``_scanner_skip_quoted``):
    backslash escape pairs are skipped blindly and a doubled quote
    (SAS-style ``''`` / ``""``) continues the string.  An unterminated
    string consumes the rest of the text.
    """

    quote = text[start]
    cursor = start + 1
    length = len(text)
    while cursor < length:
        current = text[cursor]
        if current == "\\":
            cursor += 2
            continue
        if current == quote:
            # Doubled-quote escape inside the SAS-style literal.
            if cursor + 1 < length and text[cursor + 1] == quote:
                cursor += 2
                continue
            return cursor + 1
        cursor += 1
    return cursor


def _skip_quoted_or_open_fragment(text: str, start: int) -> int:
    """One quoted token for the tab scan, bounded like the region grammar.

    A closed literal skips with the legacy scanner rules unchanged.  An
    ordinary unclosed run keeps its content raw until the statement
    ``;`` or EOF (the open-quote fragment
    truth of the pure region grammar), so following statements normalize
    their tabs again.  Region-control token internals are excluded: they
    are handled by their own quote-aware span scan.
    """

    quote = text[start]
    cursor = start + 1
    length = len(text)
    while cursor < length:
        current = text[cursor]
        if current == "\\":
            cursor += 2
            continue
        if current == quote:
            # Doubled-quote escape inside the SAS-style literal.
            if cursor + 1 < length and text[cursor + 1] == quote:
                cursor += 2
                continue
            return cursor + 1
        cursor += 1
    semicolon = text.find(";", start + 1)
    return semicolon if semicolon >= 0 else length


_MACRO_LINE_HEAD = re.compile(r"%(?:[lL][eE][tT]|[pP][uU][tT])\b")
_MACRO_BLOCK_HEAD = re.compile(r"%(?i:macro)\b")
_MACRO_REGION_END = re.compile(r"%(?i:mend)\b")
_MACRO_REGION_TAIL = re.compile(r"(?:\s+[A-Za-z_][A-Za-z0-9_]*)?\s*")
_PROC_HEAD = re.compile(r"(?i:proc)\b")
_PROC_REGION_END = re.compile(r"\b(?i:run|quit)\b")
_WHITESPACE_TAIL = re.compile(r"\s*")


def _scan_macro_line_end(text: str, start: int) -> int:
    """End of one ``%let``/``%put``/``%*`` line macro token.

    The raw ``;`` ends the token and stays unconsumed (the real
    ``MACRO_LINE`` regex excludes it from the token); quoted runs inside
    the token are consumed with the legacy quote rules.  Source without a
    terminator consumes the rest of the request text the way the regex
    alternation would.
    """

    cursor = start
    length = len(text)
    while cursor < length:
        character = text[cursor]
        if character == ";":
            return cursor
        if character == "'" or character == '"':
            cursor = _skip_quoted(text, cursor)
            continue
        cursor += 1
    return length


def _scan_macro_block_end(text: str, start: int) -> int | None:
    """End of one ``%macro ... %mend`` block token, or ``None``.

    Quote-aware scan to the ``%mend`` boundary; the optional macro name and
    trailing whitespace belong to the real token.  ``None`` mirrors the
    lexer when no ``%mend`` terminates the block: the region token cannot
    match, so the text fragments back into ordinary chunks.
    """

    cursor = start
    length = len(text)
    while cursor < length:
        character = text[cursor]
        if character == "'" or character == '"':
            cursor = _skip_quoted(text, cursor)
            continue
        match = _MACRO_REGION_END.match(text, cursor)
        if match is not None:
            tail = _MACRO_REGION_TAIL.match(text, match.end())
            if tail is None:  # pragma: no cover - the tail pattern always matches
                return match.end()
            return tail.end()
        cursor += 1
    return None


def _scan_proc_block_end(text: str, start: int) -> int | None:
    """End of one ``proc ... run|quit`` block token, or ``None``.

    Quote-aware non-greedy scan to the first ``run``/``quit`` word boundary
    plus the token's trailing whitespace.  ``None`` mirrors the lexer when
    no ``run``/``quit`` terminates the block.
    """

    cursor = start
    length = len(text)
    while cursor < length:
        character = text[cursor]
        if character == "'" or character == '"':
            cursor = _skip_quoted(text, cursor)
            continue
        match = _PROC_REGION_END.match(text, cursor)
        if match is not None:
            tail = _WHITESPACE_TAIL.match(text, match.end())
            if tail is None:  # pragma: no cover - the tail pattern always matches
                return match.end()
            return tail.end()
        cursor += 1
    return None


def _region_control_token_end(text: str, start: int) -> int | None:
    """End offset of the region control token starting at ``start``.

    Only MACRO_LINE / MACRO_BLOCK / PROC_BLOCK qualify (``data``/``run``
    keywords contain no tab-relevant span and stay ordinary chunks).
    ``None`` keeps the ordinary chunk path, matching the real lexer.
    """

    if start >= len(text):
        return None
    if text[start] == "%":
        if _MACRO_BLOCK_HEAD.match(text, start) is not None:
            return _scan_macro_block_end(text, start)
        if _MACRO_LINE_HEAD.match(text, start) is not None or text.startswith(
            "%*", start
        ):
            return _scan_macro_line_end(text, start)
        return None
    if _PROC_HEAD.match(text, start) is not None:
        return _scan_proc_block_end(text, start)
    return None


def _normalize_region_span(
    text: str, characters: list[str], start: int, end: int
) -> int:
    """Normalize tabs outside quoted runs of one region control token.

    Region control tokens create no comment boundaries of their own; only
    the quoted runs inside them keep tabs (no macro/PROC exemption).  The
    scan consumes quotes with the same legacy rule the end detection used,
    so an unterminated quote may overshoot ``end``; tab replacements stay
    1:1 inside the visited span.
    """

    cursor = start
    while cursor < end:
        character = text[cursor]
        if character == "'" or character == '"':
            cursor = _skip_quoted(text, cursor)
            continue
        if character == "\t":
            characters[cursor] = " "
        cursor += 1
    return max(cursor, end)


def initial_analysis_text(text: str) -> str:
    """Normalize the raw request text for the initial frontend lexing.

    Returns text where every tab outside quoted strings, comments, and
    quoted runs of region control tokens became exactly one space, with
    the same length and newline layout as ``text``.  ``text`` is never
    modified and nothing is cached between requests: the
    frontend owns the bounded most-recent-document buffer, this function
    is pure.

    The comment detection is token-boundary aware: comment and
    region-control openers exist only where the real region lexer opens a
    STATEMENT token (statement start), never inside a greedy
    ``OTHER_TEXT`` chunk or after in-statement material.
    """

    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    if "\t" not in text:
        return text
    characters = list(text)
    index = 0
    length = len(text)
    at_statement_start = True
    while index < length:
        character = text[index]
        if character == "\t":
            # Between tokens: the fixed 1 char -> 1 char replacement keeps
            # the request's codepoint and newline layout intact.
            characters[index] = " "
            index += 1
            continue
        if character.isspace():
            index += 1
            continue
        if character == "'" or character == '"':
            # Quoted string token: closed literal content (doubled quotes
            # and backslash escapes included) is untouched, tabs kept.  An
            # ordinary unclosed run is bounded at the next statement ``;``
            # or EOF with its content raw, so following
            # statements are reachable again.  A closed or unterminated
            # quoted run is still statement material: the statements that
            # follow its terminator, and any ``*`` right after it, stay in
            # the same statement (region-grammar tail truth).
            index = _skip_quoted_or_open_fragment(text, index)
            at_statement_start = False
            continue
        if character == ";":
            index += 1
            at_statement_start = True
            continue
        if text.startswith("/*", index):
            # BLOCK_COMMENT: only a real token start can open it.  The real
            # lexer ignores it at any statement state, so it neither opens
            # nor closes a statement boundary; the surrounding statement
            # state is unchanged.
            block_end = text.find("*/", index + 2)
            index = length if block_end < 0 else block_end + 2
            continue
        if (
            character == "*"
            and text[index + 1 : index + 2] != "/"
            and at_statement_start
        ):
            # LINE_COMMENT ``\*...;``: only at a STATEMENT start (the text
            # start, or after ``;`` / a closed region-control token), the
            # Statement-position truth of the region grammar.  A
            # ``*`` reached with in-statement material before it is
            # arithmetic chunk material and falls through below; the
            # ``;`` terminator keeps its legacy comment-tail truth.
            line_end = text.find(";", index + 1)
            index = length if line_end < 0 else line_end + 1
            continue
        region_end = _region_control_token_end(text, index)
        if region_end is not None:
            index = _normalize_region_span(text, characters, index, region_end)
            at_statement_start = not (
                index < length and text[index] == ";"
            )
            continue
        # Ordinary OTHER_TEXT / keyword chunk: greedy ``[^;'"\s]+``.  The
        # chunk swallows ``*`` and ``/*`` as material, so the next quote /
        # semicolon / whitespace is the next token boundary; such chunks
        # cannot contain the whitespace they would have to contain, so
        # tabs were already normalized by the whitespace branch above.
        at_statement_start = False
        while index < length:
            character = text[index]
            if (
                character == "'"
                or character == '"'
                or character == ";"
                or character.isspace()
            ):
                break
            index += 1
        continue
    return "".join(characters)
