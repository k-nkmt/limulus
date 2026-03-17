from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Protocol, Sequence

from .models import Diagnostic
from .native_bridge import load_native_module

try:
    from lark import Lark
    from lark.exceptions import UnexpectedInput
except Exception:  # pragma: no cover
    Lark = None  # type: ignore[assignment]
    UnexpectedInput = Exception  # type: ignore[assignment]


@dataclass(frozen=True)
class DatasetReferenceOptionSpec:
    in_var: str | None = None
    keep_vars: tuple[str, ...] = field(default_factory=tuple)
    drop_vars: tuple[str, ...] = field(default_factory=tuple)
    where_expr: str | None = None
    rename_map: dict[str, str] = field(default_factory=dict)
    firstobs: int | None = None
    obs: int | None = None
    label: str | None = None


@dataclass(frozen=True)
class SetStatementOptionSpec:
    indsname_var: str | None = None
    end_var: str | None = None


@dataclass(frozen=True)
class DatasetReference:
    name: str
    options: DatasetReferenceOptionSpec = field(default_factory=DatasetReferenceOptionSpec)


@dataclass(frozen=True)
class ParsedStatement:
    kind: str
    text: str
    dataset_refs: tuple[DatasetReference, ...] = field(default_factory=tuple)
    output_refs: tuple[DatasetReference, ...] = field(default_factory=tuple)
    statement_options: SetStatementOptionSpec = field(default_factory=SetStatementOptionSpec)
    rename_map: dict[str, str] = field(default_factory=dict)
    label_map: dict[str, str] = field(default_factory=dict)
    if_spec: "IfStatementSpec | None" = None
    do_spec: "DoStatementSpec | None" = None
    array_spec: "ArrayStatementSpec | None" = None


@dataclass(frozen=True)
class IfStatementSpec:
    condition: str
    then_action: str | None = None
    is_subset: bool = False
    is_then_do: bool = False


@dataclass(frozen=True)
class DoStatementSpec:
    loop_var: str
    start_expr: str
    end_expr: str


@dataclass(frozen=True)
class ArrayStatementSpec:
    array_name: str
    variables: tuple[str, ...]
    declared_size: int | None = None
    wildcard_size: bool = False
    character_array: bool = False


@dataclass(frozen=True)
class DataStepAst:
    statements: tuple[ParsedStatement, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ParseResult:
    ast: DataStepAst = field(default_factory=DataStepAst)
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    @property
    def has_errors(self) -> bool:
        return any(diag.severity == "error" for diag in self.diagnostics)


def statement_to_dict(statement: ParsedStatement) -> dict[str, Any]:
    if_spec = statement.if_spec
    do_spec = statement.do_spec
    array_spec = statement.array_spec
    return {
        "kind": statement.kind,
        "text": statement.text,
        "dataset_refs": [
            {
                "name": dataset_ref.name,
                "options": {
                    "in_var": dataset_ref.options.in_var,
                    "keep_vars": list(dataset_ref.options.keep_vars),
                    "drop_vars": list(dataset_ref.options.drop_vars),
                    "where_expr": dataset_ref.options.where_expr,
                    "rename_map": dict(dataset_ref.options.rename_map),
                    "firstobs": dataset_ref.options.firstobs,
                    "obs": dataset_ref.options.obs,
                    "label": dataset_ref.options.label,
                },
            }
            for dataset_ref in statement.dataset_refs
        ],
        "output_refs": [
            {
                "name": dataset_ref.name,
                "options": {
                    "in_var": dataset_ref.options.in_var,
                    "keep_vars": list(dataset_ref.options.keep_vars),
                    "drop_vars": list(dataset_ref.options.drop_vars),
                    "where_expr": dataset_ref.options.where_expr,
                    "rename_map": dict(dataset_ref.options.rename_map),
                    "firstobs": dataset_ref.options.firstobs,
                    "obs": dataset_ref.options.obs,
                    "label": dataset_ref.options.label,
                },
            }
            for dataset_ref in statement.output_refs
        ],
        "statement_options": {
            "indsname_var": statement.statement_options.indsname_var,
            "end_var": statement.statement_options.end_var,
        },
        "rename_map": dict(statement.rename_map),
        "label_map": dict(statement.label_map),
        "if_spec": (
            {
                "condition": if_spec.condition,
                "then_action": if_spec.then_action,
                "is_subset": if_spec.is_subset,
                "is_then_do": if_spec.is_then_do,
            }
            if if_spec is not None
            else None
        ),
        "do_spec": (
            {
                "loop_var": do_spec.loop_var,
                "start_expr": do_spec.start_expr,
                "end_expr": do_spec.end_expr,
            }
            if do_spec is not None
            else None
        ),
        "array_spec": (
            {
                "array_name": array_spec.array_name,
                "variables": list(array_spec.variables),
                "declared_size": array_spec.declared_size,
                "wildcard_size": array_spec.wildcard_size,
                "character_array": array_spec.character_array,
            }
            if array_spec is not None
            else None
        ),
    }


def ast_to_dict(ast: DataStepAst) -> dict[str, Any]:
    return {"statements": [statement_to_dict(statement) for statement in ast.statements]}


def ast_statements_to_dict(statements: list[ParsedStatement] | tuple[ParsedStatement, ...]) -> dict[str, Any]:
    return {"statements": [statement_to_dict(statement) for statement in statements]}


class LarkParserService:
    _RULE_KIND_MAP: dict[str, str] = {
        "data_stmt": "DATA",
        "set_stmt": "SET",
        "merge_stmt": "MERGE",
        "if_stmt": "IF",
        "if_then_stmt": "IF",
        "if_then_do_stmt": "IF",
        "subset_if_stmt": "IF",
        "else_if_stmt": "ELSE IF",
        "else_if_then_stmt": "ELSE IF",
        "else_if_then_do_stmt": "ELSE IF",
        "else_stmt": "ELSE",
        "where_stmt": "WHERE",
        "by_stmt": "BY",
        "output_stmt": "OUTPUT",
        "drop_stmt": "DROP",
        "keep_stmt": "KEEP",
        "delete_stmt": "DELETE",
        "rename_stmt": "RENAME",
        "retain_stmt": "RETAIN",
        "array_stmt": "ARRAY",
        "stop_stmt": "STOP",
        "do_stmt": "DO",
        "end_stmt": "END",
        "run_stmt": "RUN",
        "sum_stmt": "SUM",
        "assign_stmt": "ASSIGN",
        "skipped_stmt": "SKIPPED",
        "length_stmt": "SKIPPED",
        "attrib_stmt": "SKIPPED",
        "format_stmt": "SKIPPED",
        "label_stmt": "LABEL",
        "informat_stmt": "SKIPPED",
        "call_stmt": "SKIPPED",
    }
    _DO_TO_STATEMENT = re.compile(r"^do\s+([A-Za-z_][\w\.]*)\s*=\s*(.+?)\s+to\s+(.+)$", re.IGNORECASE)
    _ARRAY_DECLARATION = re.compile(
        r"^array\s+([A-Za-z_][\w\.]*)\s*(\[[^\]]+\])?\s+(\$\s+)?(.+)$",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        self._lark_parser = self._build_lark_parser()

    def parse(self, dsl_text: str) -> ParseResult:
        segments, syntax_diagnostic = self._extract_statement_segments(dsl_text)
        if syntax_diagnostic is not None:
            return ParseResult(diagnostics=(syntax_diagnostic,))

        diagnostics: list[Diagnostic] = []
        statements: list[ParsedStatement] = []

        for index, (kind, segment, variant, rule_node) in enumerate(segments, start=1):

            if kind == "DATA":
                dataset_ref_specs = self._extract_dataset_ref_specs(
                    dsl_text=dsl_text,
                    rule_node=rule_node,
                )
                dataset_tokens = self._extract_ordered_rule_texts(
                    dsl_text=dsl_text,
                    rule_node=rule_node,
                    rule_names={"dataset_ref"},
                )
                statement, diagnostic = self._parse_data_statement(
                    segment,
                    index,
                    dataset_tokens=dataset_tokens,
                    dataset_ref_specs=dataset_ref_specs,
                )
                if diagnostic is not None:
                    diagnostics.append(diagnostic)
                    continue
                statements.append(statement)
                continue

            if kind in {"SET", "MERGE"}:
                dataset_ref_specs = self._extract_dataset_ref_specs(
                    dsl_text=dsl_text,
                    rule_node=rule_node,
                )
                dataset_tokens = self._extract_ordered_rule_texts(
                    dsl_text=dsl_text,
                    rule_node=rule_node,
                    rule_names={"dataset_ref"},
                )
                statement_tokens = self._extract_ordered_rule_texts(
                    dsl_text=dsl_text,
                    rule_node=rule_node,
                    rule_names={"set_statement_option"},
                )
                statement, diagnostic = self._parse_data_source_statement(
                    kind,
                    segment,
                    index,
                    dataset_tokens=dataset_tokens,
                    statement_tokens=statement_tokens,
                    dataset_ref_specs=dataset_ref_specs,
                )
                if diagnostic is not None:
                    diagnostics.append(diagnostic)
                    continue
                statements.append(statement)
                continue

            if kind == "RENAME":
                statement, diagnostic = self._parse_rename_statement(segment, index)
                if diagnostic is not None:
                    diagnostics.append(diagnostic)
                    continue
                statements.append(statement)
                continue

            if kind == "LABEL":
                statement, diagnostic = self._parse_label_statement(segment, index)
                if diagnostic is not None:
                    diagnostics.append(diagnostic)
                    continue
                statements.append(statement)
                continue

            if kind in {"IF", "ELSE IF"}:
                statements.append(
                    ParsedStatement(
                        kind=kind,
                        text=segment,
                        if_spec=self._parse_if_statement_spec(segment, kind, variant),
                    )
                )
                continue

            if kind == "DO":
                statements.append(
                    ParsedStatement(
                        kind=kind,
                        text=segment,
                        do_spec=self._parse_do_statement_spec(segment),
                    )
                )
                continue

            if kind == "ARRAY":
                statements.append(
                    ParsedStatement(
                        kind=kind,
                        text=segment,
                        array_spec=self._parse_array_statement_spec(segment),
                    )
                )
                continue

            statements.append(ParsedStatement(kind=kind, text=segment))

        if diagnostics:
            return ParseResult(diagnostics=tuple(diagnostics))

        return ParseResult(ast=DataStepAst(statements=tuple(statements)))

    def _parse_data_statement(
        self,
        segment: str,
        statement_index: int,
        dataset_tokens: list[str] | None = None,
        dataset_ref_specs: list[tuple[str, tuple[str, ...]]] | None = None,
    ) -> tuple[ParsedStatement, Diagnostic | None]:
        body = segment[len("data") :].strip()
        if not body:
            return ParsedStatement(kind="DATA", text=segment), None

        output_refs: list[DatasetReference] = []
        if dataset_ref_specs:
            for dataset_name, option_tokens in dataset_ref_specs:
                parsed_ref, diagnostic = self._parse_dataset_reference_from_parts(
                    dataset_name=dataset_name,
                    option_tokens=option_tokens,
                    statement_index=statement_index,
                )
                if diagnostic is not None:
                    return ParsedStatement(kind="DATA", text=segment), diagnostic
                output_refs.append(parsed_ref)
        else:
            tokens = dataset_tokens if dataset_tokens else self._normalize_equals_tokens(self._split_top_level_tokens(body))
            for token in tokens:
                parsed_ref, diagnostic = self._parse_dataset_reference(token, statement_index)
                if diagnostic is not None:
                    return ParsedStatement(kind="DATA", text=segment), diagnostic
                output_refs.append(parsed_ref)

        refs = tuple(output_refs)
        return ParsedStatement(kind="DATA", text=segment, dataset_refs=refs, output_refs=refs), None

    def _build_lark_parser(self):
        if Lark is None:
            return None

        grammar_path = Path(__file__).with_name("grammar") / "datastep.lark"
        try:
            grammar = grammar_path.read_text(encoding="utf-8")
            return Lark(grammar, start="start", parser="lalr", propagate_positions=True)
        except Exception:
            return None

    def _extract_statement_segments(
        self,
        dsl_text: str,
    ) -> tuple[list[tuple[str, str, str, Any]], Diagnostic | None]:
        if self._lark_parser is None:
            return [], Diagnostic(
                code="PARSE_BACKEND_CAPABILITY_MISSING",
                severity="error",
                location="",
                message="lark parser is not available; parsing cannot proceed.",
            )

        try:
            tree = self._lark_parser.parse(dsl_text)
            segments: list[tuple[str, str, str, Any]] = []
            for index, statement_node in enumerate(getattr(tree, "children", ()), start=1):
                children = getattr(statement_node, "children", ())
                if not children:
                    continue
                rule_node = children[0]
                rule_name = str(getattr(rule_node, "data", ""))
                variant_name = rule_name
                nested_children = getattr(rule_node, "children", ())
                if len(nested_children) == 1 and hasattr(nested_children[0], "data"):
                    nested_rule = str(getattr(nested_children[0], "data", ""))
                    if nested_rule in self._RULE_KIND_MAP:
                        variant_name = nested_rule
                kind = self._RULE_KIND_MAP.get(variant_name)
                if kind is None:
                    kind = self._RULE_KIND_MAP.get(rule_name)
                if kind is None:
                    return [], Diagnostic(
                        code="PARSE_UNSUPPORTED_STATEMENT",
                        severity="error",
                        location=f"statement:{index}",
                        message=f"Unsupported statement syntax: {rule_name}",
                    )

                node_meta = getattr(rule_node, "meta", None)
                if node_meta is None:
                    node_meta = getattr(statement_node, "meta", None)
                start = getattr(node_meta, "start_pos", 0)
                end = getattr(node_meta, "end_pos", start)
                text = dsl_text[start:end].strip()
                if text:
                    segments.append((kind, text, variant_name, rule_node))

            return segments, None
        except UnexpectedInput as error:
            position = getattr(error, "pos_in_stream", 0)
            line = getattr(error, "line", 1)
            column = getattr(error, "column", 1)
            statement_index = dsl_text[:position].count(";") + 1
            message = f"Unsupported statement syntax near line {line}, column {column}."
            return [], Diagnostic(
                code="PARSE_UNSUPPORTED_STATEMENT",
                severity="error",
                location=f"statement:{statement_index}",
                message=message,
            )

    def _parse_data_source_statement(
        self,
        kind: str,
        segment: str,
        statement_index: int,
        dataset_tokens: list[str] | None = None,
        statement_tokens: list[str] | None = None,
        dataset_ref_specs: list[tuple[str, tuple[str, ...]]] | None = None,
    ) -> tuple[ParsedStatement, Diagnostic | None]:
        keyword = kind.lower()
        body = segment[len(keyword):].strip()
        if not body:
            return (
                ParsedStatement(kind=kind, text=segment),
                None,
            )

        if dataset_ref_specs:
            tokens = []
        elif dataset_tokens and kind in {"SET", "MERGE"}:
            tokens = list(dataset_tokens)
        else:
            tokens = self._split_top_level_tokens(body)
            tokens = self._normalize_equals_tokens(tokens)
        dataset_refs: list[DatasetReference] = []
        statement_options = SetStatementOptionSpec()

        for token in statement_tokens or ():
            option_key_match = re.match(r"^\s*(indsname|end|in)\s*=", token.lower())
            if option_key_match is None:
                continue
            option_key = option_key_match.group(1)
            if option_key == "indsname" and kind != "SET":
                return (
                    ParsedStatement(kind=kind, text=segment),
                    Diagnostic(
                        code="PARSE_SET_OPTION_SCOPE_ERROR",
                        severity="error",
                        location=f"statement:{statement_index}",
                        message=f"{option_key.upper()}= is only allowed as a SET statement option.",
                    ),
                )
            updated, diagnostic = self._parse_set_statement_option_token(
                token=token,
                existing=statement_options,
                statement_index=statement_index,
            )
            if diagnostic is not None:
                return ParsedStatement(kind=kind, text=segment), diagnostic
            statement_options = updated

        if dataset_ref_specs:
            for dataset_name, option_tokens in dataset_ref_specs:
                parsed_ref, diagnostic = self._parse_dataset_reference_from_parts(
                    dataset_name=dataset_name,
                    option_tokens=option_tokens,
                    statement_index=statement_index,
                )
                if diagnostic is not None:
                    return ParsedStatement(kind=kind, text=segment), diagnostic
                dataset_refs.append(parsed_ref)
        else:
            for token in tokens:
                token_lower = token.lower()
                set_option_match = re.match(r"^\s*(indsname|end|in)\s*=", token_lower)
                if set_option_match is not None and "(" not in token:
                    option_key = set_option_match.group(1)
                    if option_key == "indsname" and kind != "SET":
                        return (
                            ParsedStatement(kind=kind, text=segment),
                            Diagnostic(
                                code="PARSE_SET_OPTION_SCOPE_ERROR",
                                severity="error",
                                location=f"statement:{statement_index}",
                                message=f"{option_key.upper()}= is only allowed as a SET statement option.",
                            ),
                        )
                    updated, diagnostic = self._parse_set_statement_option_token(
                        token=token,
                        existing=statement_options,
                        statement_index=statement_index,
                    )
                    if diagnostic is not None:
                        return ParsedStatement(kind=kind, text=segment), diagnostic
                    statement_options = updated
                    continue

                parsed_ref, diagnostic = self._parse_dataset_reference(token, statement_index)
                if diagnostic is not None:
                    return ParsedStatement(kind=kind, text=segment), diagnostic
                dataset_refs.append(parsed_ref)

        return (
            ParsedStatement(
                kind=kind,
                text=segment,
                dataset_refs=tuple(dataset_refs),
                statement_options=statement_options,
            ),
            None,
        )

    def _extract_ordered_rule_texts(
        self,
        dsl_text: str,
        rule_node: Any,
        rule_names: set[str],
    ) -> list[str]:
        spans: list[tuple[int, int]] = []

        def walk(node: Any) -> None:
            node_data = getattr(node, "data", None)
            node_meta = getattr(node, "meta", None)
            if node_data in rule_names and node_meta is not None:
                start = getattr(node_meta, "start_pos", None)
                end = getattr(node_meta, "end_pos", None)
                if isinstance(start, int) and isinstance(end, int) and end > start:
                    spans.append((start, end))
            for child in getattr(node, "children", ()):
                if hasattr(child, "data"):
                    walk(child)

        walk(rule_node)
        spans.sort(key=lambda item: item[0])
        return [dsl_text[start:end].strip() for start, end in spans if dsl_text[start:end].strip()]

    def _extract_dataset_ref_specs(
        self,
        dsl_text: str,
        rule_node: Any,
    ) -> list[tuple[str, tuple[str, ...]]]:
        refs: list[tuple[int, str, tuple[str, ...]]] = []

        def node_text(node: Any) -> str:
            node_meta = getattr(node, "meta", None)
            start = getattr(node_meta, "start_pos", None)
            end = getattr(node_meta, "end_pos", None)
            if isinstance(start, int) and isinstance(end, int) and end > start:
                return dsl_text[start:end].strip()
            return ""

        def collect_dataset_option_tokens(dataset_options_node: Any) -> tuple[str, ...]:
            tokens: list[tuple[int, str]] = []

            def walk_options(node: Any) -> None:
                node_data = getattr(node, "data", None)
                if node_data == "dataset_option":
                    option_text = node_text(node)
                    option_meta = getattr(node, "meta", None)
                    option_start = getattr(option_meta, "start_pos", 0)
                    if option_text:
                        tokens.append((option_start, option_text))
                for child in getattr(node, "children", ()):
                    if hasattr(child, "data"):
                        walk_options(child)

            walk_options(dataset_options_node)
            tokens.sort(key=lambda item: item[0])
            return tuple(token for _, token in tokens)

        def walk(node: Any) -> None:
            node_data = getattr(node, "data", None)
            if node_data == "dataset_ref":
                node_meta = getattr(node, "meta", None)
                start = getattr(node_meta, "start_pos", 0)
                dataset_name = ""
                option_tokens: tuple[str, ...] = ()

                for child in getattr(node, "children", ()):
                    child_data = getattr(child, "data", None)
                    if child_data == "dataset_name":
                        dataset_name = node_text(child)
                    elif child_data == "dataset_options":
                        option_tokens = collect_dataset_option_tokens(child)

                if dataset_name:
                    refs.append((start, dataset_name, option_tokens))

            for child in getattr(node, "children", ()):
                if hasattr(child, "data"):
                    walk(child)

        walk(rule_node)
        refs.sort(key=lambda item: item[0])
        return [(name, options) for _, name, options in refs]

    def _parse_dataset_reference_from_parts(
        self,
        dataset_name: str,
        option_tokens: Sequence[str],
        statement_index: int,
    ) -> tuple[DatasetReference, Diagnostic | None]:
        if not option_tokens:
            return DatasetReference(name=dataset_name), None

        options, diagnostic = self._parse_dataset_reference_option_tokens(option_tokens, statement_index)
        if diagnostic is not None:
            return DatasetReference(name=dataset_name), diagnostic
        return DatasetReference(name=dataset_name, options=options), None

    def _parse_set_statement_option_token(
        self,
        token: str,
        existing: SetStatementOptionSpec,
        statement_index: int,
    ) -> tuple[SetStatementOptionSpec, Diagnostic | None]:
        match = re.match(r"^\s*([A-Za-z_][\w]*)\s*=\s*(.+?)\s*$", token)
        if match is None:
            return existing, Diagnostic(
                code="PARSE_UNSUPPORTED_STATEMENT",
                severity="error",
                location=f"statement:{statement_index}",
                message=f"Unsupported SET statement option: {token}",
            )

        key = match.group(1)
        value = match.group(2)
        normalized_key = key.strip().lower()
        normalized_value = value.strip()

        if normalized_key == "in":
            return existing, Diagnostic(
                code="PARSE_SET_OPTION_SCOPE_ERROR",
                severity="error",
                location=f"statement:{statement_index}",
                message="IN= is only allowed as a dataset reference option (e.g. set in(in=flag)).",
            )

        if normalized_key not in {"indsname", "end"}:
            return existing, Diagnostic(
                code="PARSE_UNSUPPORTED_STATEMENT",
                severity="error",
                location=f"statement:{statement_index}",
                message=f"Unsupported SET statement option: {token}",
            )

        if not normalized_value:
            return existing, Diagnostic(
                code="PARSE_UNSUPPORTED_STATEMENT",
                severity="error",
                location=f"statement:{statement_index}",
                message=f"SET statement option requires a variable name: {token}",
            )

        if normalized_key == "indsname":
            return SetStatementOptionSpec(indsname_var=normalized_value, end_var=existing.end_var), None

        return SetStatementOptionSpec(indsname_var=existing.indsname_var, end_var=normalized_value), None

    def _parse_dataset_reference(
        self,
        token: str,
        statement_index: int,
    ) -> tuple[DatasetReference, Diagnostic | None]:
        if "(" not in token:
            return DatasetReference(name=token), None

        if not token.endswith(")"):
            return DatasetReference(name=token), Diagnostic(
                code="PARSE_UNSUPPORTED_STATEMENT",
                severity="error",
                location=f"statement:{statement_index}",
                message=f"Invalid dataset reference option syntax: {token}",
            )

        open_index = token.find("(")
        name = token[:open_index].strip()
        option_body = token[open_index + 1 : -1].strip()

        options, diagnostic = self._parse_dataset_reference_options(option_body, statement_index)
        if diagnostic is not None:
            return DatasetReference(name=name), diagnostic

        return DatasetReference(name=name, options=options), None

    def _parse_dataset_reference_options(
        self,
        option_body: str,
        statement_index: int,
    ) -> tuple[DatasetReferenceOptionSpec, Diagnostic | None]:
        tokens = self._normalize_equals_tokens(self._split_top_level_tokens(option_body))
        return self._parse_dataset_reference_option_tokens(tokens, statement_index)

    def _parse_dataset_reference_option_tokens(
        self,
        tokens: Sequence[str],
        statement_index: int,
    ) -> tuple[DatasetReferenceOptionSpec, Diagnostic | None]:
        in_var: str | None = None
        keep_vars: list[str] = []
        drop_vars: list[str] = []
        where_expr: str | None = None
        rename_map: dict[str, str] = {}
        firstobs: int | None = None
        obs: int | None = None
        label: str | None = None
        active_collect: str | None = None

        for token in tokens:
            if "=" in token:
                key, raw_value = token.split("=", maxsplit=1)
                normalized_key = key.strip().lower()
                value = raw_value.strip()

                if normalized_key in {"indsname", "end"}:
                    return DatasetReferenceOptionSpec(), Diagnostic(
                        code="PARSE_SET_OPTION_SCOPE_ERROR",
                        severity="error",
                        location=f"statement:{statement_index}",
                        message=f"{normalized_key.upper()}= is only allowed as a SET statement option.",
                    )

                if normalized_key == "in":
                    in_var = value
                    active_collect = None
                    continue

                if normalized_key == "keep":
                    keep_vars = [part for part in value.split() if part]
                    active_collect = "keep"
                    continue

                if normalized_key == "drop":
                    drop_vars = [part for part in value.split() if part]
                    active_collect = "drop"
                    continue

                if normalized_key == "where":
                    if value.startswith("(") and value.endswith(")"):
                        where_expr = value[1:-1].strip()
                    else:
                        where_expr = value
                    active_collect = None
                    continue

                if normalized_key == "rename":
                    value_body = value
                    if value_body.startswith("(") and value_body.endswith(")"):
                        value_body = value_body[1:-1]
                    parsed_map, diagnostic = self._parse_rename_pairs(value_body, statement_index)
                    if diagnostic is not None:
                        return DatasetReferenceOptionSpec(), diagnostic
                    rename_map = parsed_map
                    active_collect = None
                    continue

                if normalized_key == "firstobs":
                    try:
                        firstobs = int(value)
                    except ValueError:
                        return DatasetReferenceOptionSpec(), Diagnostic(
                            code="PARSE_UNSUPPORTED_STATEMENT",
                            severity="error",
                            location=f"statement:{statement_index}",
                            message=f"Unsupported dataset reference option: {token}",
                        )
                    active_collect = None
                    continue

                if normalized_key == "obs":
                    try:
                        obs = int(value)
                    except ValueError:
                        return DatasetReferenceOptionSpec(), Diagnostic(
                            code="PARSE_UNSUPPORTED_STATEMENT",
                            severity="error",
                            location=f"statement:{statement_index}",
                            message=f"Unsupported dataset reference option: {token}",
                        )
                    active_collect = None
                    continue

                if normalized_key == "label":
                    label = self._strip_quoted_value(value)
                    active_collect = None
                    continue

                return DatasetReferenceOptionSpec(), Diagnostic(
                    code="PARSE_UNSUPPORTED_STATEMENT",
                    severity="error",
                    location=f"statement:{statement_index}",
                    message=f"Unsupported dataset reference option: {token}",
                )

            if active_collect == "keep":
                keep_vars.append(token)
                continue

            if active_collect == "drop":
                drop_vars.append(token)
                continue

            return DatasetReferenceOptionSpec(), Diagnostic(
                code="PARSE_UNSUPPORTED_STATEMENT",
                severity="error",
                location=f"statement:{statement_index}",
                message=f"Invalid dataset reference option token: {token}",
            )

        return (
            DatasetReferenceOptionSpec(
                in_var=in_var,
                keep_vars=tuple(keep_vars),
                drop_vars=tuple(drop_vars),
                where_expr=where_expr,
                rename_map=rename_map,
                firstobs=firstobs,
                obs=obs,
                label=label,
            ),
            None,
        )

    def _parse_rename_statement(
        self,
        segment: str,
        statement_index: int,
    ) -> tuple[ParsedStatement, Diagnostic | None]:
        body = segment[len("rename") :].strip()
        rename_map, diagnostic = self._parse_rename_pairs(body, statement_index)
        if diagnostic is not None:
            return ParsedStatement(kind="RENAME", text=segment), diagnostic
        return ParsedStatement(kind="RENAME", text=segment, rename_map=rename_map), None

    def _parse_label_statement(
        self,
        segment: str,
        statement_index: int,
    ) -> tuple[ParsedStatement, Diagnostic | None]:
        body = segment[len("label") :].strip()
        if not body:
            return ParsedStatement(kind="LABEL", text=segment), None

        label_map: dict[str, str] = {}
        pattern = re.compile(r'([A-Za-z_][\w\.]*)\s*=\s*("[^"]*"|\'[^\']*\')')
        for match in pattern.finditer(body):
            label_map[match.group(1)] = self._strip_quoted_value(match.group(2))

        if not label_map:
            return ParsedStatement(kind="LABEL", text=segment), Diagnostic(
                code="PARSE_UNSUPPORTED_STATEMENT",
                severity="error",
                location=f"statement:{statement_index}",
                message=f"Unsupported LABEL statement syntax: {segment}",
            )

        return ParsedStatement(kind="LABEL", text=segment, label_map=label_map), None

    def _parse_rename_pairs(
        self,
        body: str,
        statement_index: int,
    ) -> tuple[dict[str, str], Diagnostic | None]:
        rename_map: dict[str, str] = {}
        tokens = self._normalize_equals_tokens(self._split_top_level_tokens(body))
        for token in tokens:
            if "=" not in token:
                return {}, Diagnostic(
                    code="PARSE_RENAME_STATEMENT_INVALID",
                    severity="error",
                    location=f"statement:{statement_index}",
                    message=f"Invalid rename mapping: {token}",
                )
            old_name, new_name = token.split("=", maxsplit=1)
            normalized_old = old_name.strip()
            normalized_new = new_name.strip()
            if not normalized_old or not normalized_new:
                return {}, Diagnostic(
                    code="PARSE_RENAME_STATEMENT_INVALID",
                    severity="error",
                    location=f"statement:{statement_index}",
                    message=f"Invalid rename mapping: {token}",
                )
            rename_map[normalized_old] = normalized_new
        return rename_map, None

    def _strip_quoted_value(self, value: str) -> str:
        normalized = value.strip()
        if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {'"', "'"}:
            return normalized[1:-1]
        return normalized

    def _split_top_level_tokens(self, text: str) -> list[str]:
        tokens: list[str] = []
        current: list[str] = []
        depth = 0

        for char in text:
            if char == "(":
                depth += 1
                current.append(char)
                continue
            if char == ")":
                depth -= 1
                current.append(char)
                continue
            if char.isspace() and depth == 0:
                if current:
                    tokens.append("".join(current))
                    current = []
                continue
            current.append(char)

        if current:
            tokens.append("".join(current))

        return [token for token in tokens if token]

    def _normalize_equals_tokens(self, tokens: list[str]) -> list[str]:
        normalized: list[str] = []
        index = 0

        while index < len(tokens):
            token = tokens[index].strip()
            if not token:
                index += 1
                continue

            if token == "=" and normalized and index + 1 < len(tokens):
                previous = normalized.pop().strip()
                next_token = tokens[index + 1].strip()
                normalized.append(f"{previous}={next_token}")
                index += 2
                continue

            if token.endswith("=") and index + 1 < len(tokens):
                next_token = tokens[index + 1].strip()
                normalized.append(f"{token}{next_token}")
                index += 2
                continue

            if token.startswith("=") and normalized:
                previous = normalized.pop().strip()
                normalized.append(f"{previous}{token}")
                index += 1
                continue

            normalized.append(token)
            index += 1

        return normalized

    def _parse_if_statement_spec(self, segment: str, kind: str, variant: str) -> IfStatementSpec | None:
        keyword = "else if" if kind == "ELSE IF" else "if"
        lowered_segment = segment.lower()
        prefix = f"{keyword} "
        if not lowered_segment.startswith(prefix):
            return None

        body = segment[len(prefix) :].strip()
        if not body:
            return None

        if variant == "subset_if_stmt":
            condition = self._normalize_if_expression(body)
            if not condition:
                return None
            return IfStatementSpec(condition=condition, is_subset=True)

        lowered_body = body.lower()
        then_index = lowered_body.find(" then ")
        if then_index < 0:
            condition = self._normalize_if_expression(body)
            if not condition:
                return None
            return IfStatementSpec(condition=condition, is_subset=True)

        raw_condition = body[:then_index].strip()
        action = body[then_index + len(" then ") :].strip()
        if not raw_condition or not action:
            return None

        condition = self._normalize_if_expression(raw_condition)
        return IfStatementSpec(
            condition=condition,
            then_action=action,
            is_subset=False,
            is_then_do=(variant in {"if_then_do_stmt", "else_if_then_do_stmt"} or action.lower() == "do"),
        )

    def _normalize_if_expression(self, expression: str) -> str:
        normalized = expression.strip()
        normalized = re.sub(r"\bEQ\b", "==", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bNE\b", "!=", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bGT\b", ">", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bLT\b", "<", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bGE\b", ">=", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bLE\b", "<=", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bAND\b", "and", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bOR\b", "or", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bNOT\b", "not", normalized, flags=re.IGNORECASE)
        normalized = normalized.replace("^=", "!=")
        return re.sub(r"(?<![<>!])=(?!=)", "==", normalized)

    def _parse_do_statement_spec(self, segment: str) -> DoStatementSpec | None:
        matched = self._DO_TO_STATEMENT.match(segment.strip())
        if matched is None:
            return None
        return DoStatementSpec(
            loop_var=matched.group(1),
            start_expr=matched.group(2).strip(),
            end_expr=matched.group(3).strip(),
        )

    def _parse_array_statement_spec(self, segment: str) -> ArrayStatementSpec | None:
        matched = self._ARRAY_DECLARATION.match(segment.strip())
        if matched is None:
            return None

        array_name = matched.group(1)
        dimension_token = matched.group(2).strip() if matched.group(2) else None
        character_array = bool(matched.group(3))
        remainder = matched.group(4).strip()
        tokens = [token for token in remainder.split() if token]
        if not tokens:
            return None

        declared_size: int | None = None
        wildcard_size = False
        if dimension_token is not None:
            if re.match(r"^\[\s*\*\s*\]$", dimension_token):
                wildcard_size = True
            elif re.match(r"^\[\s*\d+\s*\]$", dimension_token):
                declared_size = int(dimension_token.strip("[] "))
        else:
            first = tokens[0]
            if re.match(r"^\[\s*\*\s*\]$", first):
                wildcard_size = True
                tokens = tokens[1:]
            elif re.match(r"^\[\s*\d+\s*\]$", first):
                declared_size = int(first.strip("[] "))
                tokens = tokens[1:]
            elif re.match(r"^\d+$", first):
                declared_size = int(first)
                tokens = tokens[1:]

        variables = tuple(tokens)
        if not variables:
            return None

        return ArrayStatementSpec(
            array_name=array_name,
            variables=variables,
            declared_size=declared_size,
            wildcard_size=wildcard_size,
            character_array=character_array,
        )


@dataclass(frozen=True)
class ParserExecutionContext:
    dsl_text: str


ParserService = LarkParserService


class ParserBackend(Protocol):
    name: str

    def is_available(self) -> bool:
        ...

    def parse(self, context: ParserExecutionContext) -> ParseResult:
        ...


class PythonParserBackend:
    name = "python"

    def __init__(self, parser_service: LarkParserService) -> None:
        self._parser_service = parser_service

    def is_available(self) -> bool:
        return True

    def parse(self, context: ParserExecutionContext) -> ParseResult:
        return self._parser_service.parse(context.dsl_text)


class RustNativeParserBackend:
    name = "rust"
    _SUPPORTED_STATEMENTS = {"DATA", "SET", "WHERE", "OUTPUT", "STOP", "RUN"}

    def __init__(self, parser_service: LarkParserService | None = None) -> None:
        self._parser_service = parser_service or LarkParserService()

    def is_available(self) -> bool:
        return True

    def parse(self, context: ParserExecutionContext) -> ParseResult:
        result = self._parser_service.parse(context.dsl_text)
        if result.has_errors:
            return result

        for index, statement in enumerate(result.ast.statements, start=1):
            if statement.kind not in self._SUPPORTED_STATEMENTS:
                return ParseResult(
                    diagnostics=(
                        Diagnostic(
                            code="PARSE_BACKEND_CAPABILITY_MISSING",
                            severity="error",
                            location=f"statement:{index}",
                            message=f"Rust native parser does not support statement kind: {statement.kind}",
                        ),
                    )
                )

        native_module, _import_error = load_native_module()
        if native_module is None:
            return result

        try:
            native_result = native_module.parse_subset(context.dsl_text)
        except Exception as error:
            return ParseResult(
                diagnostics=(
                    Diagnostic(
                        code="PARSE_RUST_NATIVE_EXECUTION_FAILED",
                        severity="error",
                        location="",
                        message=f"Rust native parser execution failed: {error}",
                    ),
                )
            )

        native_diagnostics = self._extract_native_diagnostics(native_result)
        if native_diagnostics:
            return ParseResult(diagnostics=tuple(native_diagnostics))

        return result

    def _extract_native_diagnostics(self, native_result: Any) -> list[Diagnostic]:
        if not isinstance(native_result, dict):
            return [
                Diagnostic(
                    code="PARSE_RUST_NATIVE_EXECUTION_FAILED",
                    severity="error",
                    location="",
                    message="Rust native parser returned invalid result type.",
                )
            ]

        raw = native_result.get("diagnostics", [])
        if not isinstance(raw, list):
            return []

        diagnostics: list[Diagnostic] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            diagnostics.append(
                Diagnostic(
                    code=str(item.get("code", "PARSE_BACKEND_CAPABILITY_MISSING")),
                    severity=str(item.get("severity", "error")),
                    location=str(item.get("location", "")),
                    message=str(item.get("message", "Rust native parser returned diagnostic.")),
                )
            )
        return diagnostics


class ParserBackendSelector:
    def __init__(self, python_backend: ParserBackend, rust_backend: ParserBackend) -> None:
        self._python_backend = python_backend
        self._rust_backend = rust_backend

    def select(self, preferred_backend: str) -> ParserBackend:
        preferred = preferred_backend.strip().lower()
        if preferred == "rust" and self._rust_backend.is_available():
            return self._rust_backend
        return self._python_backend