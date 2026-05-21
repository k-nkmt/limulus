from dataclasses import dataclass, field, replace
import re
from typing import Any, Protocol, Sequence

from .lark_support import build_lark_parser_from_file
from .models import Diagnostic, DiagnosticLabel, DiagnosticSpan
try:
    from lark.exceptions import UnexpectedInput
except Exception:  # pragma: no cover
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
    span: DiagnosticSpan | None = None
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
class StatementRegion:
    kind: str
    start: int
    end: int


@dataclass(frozen=True)
class ParseResult:
    ast: DataStepAst = field(default_factory=DataStepAst)
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    @property
    def has_errors(self) -> bool:
        return any(diag.severity == "error" for diag in self.diagnostics)


def statement_to_dict(statement: ParsedStatement) -> dict[str, Any]:
    span = statement.span
    if_spec = statement.if_spec
    do_spec = statement.do_spec
    array_spec = statement.array_spec
    return {
        "kind": statement.kind,
        "text": statement.text,
        "span": (
            {
                "start": span.start,
                "end": span.end,
                "line": span.line,
                "column": span.column,
                "end_line": span.end_line,
                "end_column": span.end_column,
                "source_id": span.source_id,
            }
            if span is not None
            else None
        ),
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
    def __init__(self) -> None:
        self._lark_parser = self._build_lark_parser()
        self._label_pairs_parser = build_lark_parser_from_file("label_pairs.lark")
        self._if_structured_parser = build_lark_parser_from_file("if_structured.lark")
        self._rename_pairs_parser = build_lark_parser_from_file("rename_pairs.lark")
        self._assignment_token_parser = build_lark_parser_from_file("assignment_token.lark")
        self._do_loop_parser = build_lark_parser_from_file("do_loop.lark")
        self._array_declaration_parser = build_lark_parser_from_file("array_declaration.lark")

    def parse(self, dsl_text: str) -> ParseResult:
        segments, syntax_diagnostic = self._extract_statement_segments(dsl_text)
        if syntax_diagnostic is not None:
            return ParseResult(diagnostics=(syntax_diagnostic,))

        diagnostics: list[Diagnostic] = []
        statements: list[ParsedStatement] = []

        for index, (kind, segment, variant, rule_node) in enumerate(segments, start=1):
            statement_span = self._span_from_meta(dsl_text, getattr(rule_node, "meta", None))

            if kind == "DATA":
                dataset_ref_specs = self._extract_dataset_ref_specs(
                    dsl_text=dsl_text,
                    rule_node=rule_node,
                )
                statement, diagnostic = self._parse_data_statement(
                    segment,
                    index,
                    dataset_ref_specs=dataset_ref_specs,
                )
                if diagnostic is not None:
                    diagnostics.append(self._attach_statement_context(diagnostic, dsl_text=dsl_text, span=statement_span))
                    continue
                statements.append(replace(statement, span=statement_span))
                continue

            if kind in {"SET", "MERGE"}:
                dataset_ref_specs = self._extract_dataset_ref_specs(
                    dsl_text=dsl_text,
                    rule_node=rule_node,
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
                    statement_tokens=statement_tokens,
                    dataset_ref_specs=dataset_ref_specs,
                )
                if diagnostic is not None:
                    diagnostics.append(self._attach_statement_context(diagnostic, dsl_text=dsl_text, span=statement_span))
                    continue
                statements.append(replace(statement, span=statement_span))
                continue

            if kind == "RENAME":
                statement, diagnostic = self._parse_rename_statement(segment, index)
                if diagnostic is not None:
                    diagnostics.append(self._attach_statement_context(diagnostic, dsl_text=dsl_text, span=statement_span))
                    continue
                statements.append(replace(statement, span=statement_span))
                continue

            if kind == "LABEL":
                statement, diagnostic = self._parse_label_statement(segment, index)
                if diagnostic is not None:
                    diagnostics.append(self._attach_statement_context(diagnostic, dsl_text=dsl_text, span=statement_span))
                    continue
                statements.append(replace(statement, span=statement_span))
                continue

            if kind in {"IF", "ELSE IF"}:
                statements.append(
                    ParsedStatement(
                        kind=kind,
                        text=segment,
                        span=statement_span,
                        if_spec=self._parse_if_statement_spec(segment, kind, variant),
                    )
                )
                continue

            if kind == "DO":
                statements.append(
                    ParsedStatement(
                        kind=kind,
                        text=segment,
                        span=statement_span,
                        do_spec=self._parse_do_statement_spec(segment),
                    )
                )
                continue

            if kind == "ARRAY":
                statements.append(
                    ParsedStatement(
                        kind=kind,
                        text=segment,
                        span=statement_span,
                        array_spec=self._parse_array_statement_spec(segment),
                    )
                )
                continue

            statements.append(ParsedStatement(kind=kind, text=segment, span=statement_span))

        if diagnostics:
            return ParseResult(diagnostics=tuple(diagnostics))

        return ParseResult(ast=DataStepAst(statements=tuple(statements)))

    def extract_statement_regions(self, dsl_text: str) -> tuple[StatementRegion, ...] | None:
        segments, syntax_diagnostic = self._extract_statement_segments(dsl_text)
        if syntax_diagnostic is not None:
            return None

        regions: list[StatementRegion] = []
        for kind, _, _, rule_node in segments:
            node_meta = getattr(rule_node, "meta", None)
            if node_meta is None:
                continue
            start = max(getattr(node_meta, "start_pos", 0), 0)
            end = max(getattr(node_meta, "end_pos", start), start)
            regions.append(StatementRegion(kind=kind, start=start, end=end))
        return tuple(regions)

    def _parse_data_statement(
        self,
        segment: str,
        statement_index: int,
        dataset_ref_specs: list[tuple[str, tuple[str, ...]]] | None = None,
    ) -> tuple[ParsedStatement, Diagnostic | None]:
        body = segment[len("data") :].strip()
        if not body:
            return ParsedStatement(kind="DATA", text=segment), None

        if not dataset_ref_specs:
            return ParsedStatement(kind="DATA", text=segment), Diagnostic(
                code="PARSE_UNSUPPORTED_STATEMENT",
                severity="error",
                location=f"statement:{statement_index}",
                message=f"Unable to resolve DATA output dataset references: {segment}",
            )

        output_refs: list[DatasetReference] = []
        for dataset_name, option_tokens in dataset_ref_specs:
            parsed_ref, diagnostic = self._parse_dataset_reference_from_parts(
                dataset_name=dataset_name,
                option_tokens=option_tokens,
                statement_index=statement_index,
            )
            if diagnostic is not None:
                return ParsedStatement(kind="DATA", text=segment), diagnostic
            output_refs.append(parsed_ref)

        refs = tuple(output_refs)
        return ParsedStatement(kind="DATA", text=segment, dataset_refs=refs, output_refs=refs), None

    def _build_lark_parser(self):
        return build_lark_parser_from_file("datastep.lark")

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
                node_meta = getattr(rule_node, "meta", None)
                if node_meta is None:
                    node_meta = getattr(statement_node, "meta", None)
                nested_children = getattr(rule_node, "children", ())
                if len(nested_children) == 1 and hasattr(nested_children[0], "data"):
                    nested_rule = str(getattr(nested_children[0], "data", ""))
                    if nested_rule in self._RULE_KIND_MAP:
                        variant_name = nested_rule
                kind = self._RULE_KIND_MAP.get(variant_name)
                if kind is None:
                    kind = self._RULE_KIND_MAP.get(rule_name)
                if kind is None:
                    span = self._span_from_meta(dsl_text, node_meta)
                    return [], Diagnostic(
                        code="PARSE_UNSUPPORTED_STATEMENT",
                        severity="error",
                        location=f"statement:{index}",
                        message=f"Unsupported statement syntax: {rule_name}",
                        span=span,
                        labels=((DiagnosticLabel(span=span, message="unsupported statement"),) if span is not None else ()),
                        source_text=dsl_text,
                    )

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
            span = self._span_from_unexpected_input(dsl_text, error)
            return [], Diagnostic(
                code="PARSE_UNSUPPORTED_STATEMENT",
                severity="error",
                location=f"statement:{statement_index}",
                message=message,
                span=span,
                labels=((DiagnosticLabel(span=span, message="syntax error"),) if span is not None else ()),
                source_text=dsl_text,
            )

    def _span_from_meta(self, dsl_text: str, node_meta: Any | None) -> DiagnosticSpan | None:
        if node_meta is None:
            return None
        start = getattr(node_meta, "start_pos", None)
        end = getattr(node_meta, "end_pos", None)
        line = getattr(node_meta, "line", None)
        column = getattr(node_meta, "column", None)
        end_line = getattr(node_meta, "end_line", None)
        end_column = getattr(node_meta, "end_column", None)
        if start is None or end is None or line is None or column is None:
            return None
        end_position = max(end, start + 1)
        return DiagnosticSpan(
            start=max(start, 0),
            end=end_position,
            line=max(line, 1),
            column=max(column, 1),
            end_line=end_line,
            end_column=end_column,
            source_id="<dsl>",
        )

    def _span_from_unexpected_input(self, dsl_text: str, error: Any) -> DiagnosticSpan:
        start = max(getattr(error, "pos_in_stream", 0), 0)
        line = max(getattr(error, "line", 1), 1)
        column = max(getattr(error, "column", 1), 1)
        end = self._unexpected_span_end(dsl_text, start)
        return DiagnosticSpan(
            start=start,
            end=end,
            line=line,
            column=column,
            end_line=line,
            end_column=column + max(end - start, 1),
            source_id="<dsl>",
        )

    def _unexpected_span_end(self, dsl_text: str, start: int) -> int:
        if start >= len(dsl_text):
            return start + 1
        match = re.match(r"[^\s;]+", dsl_text[start:])
        if match is None:
            return min(start + 1, len(dsl_text))
        return start + max(len(match.group(0)), 1)

    def _attach_statement_context(
        self,
        diagnostic: Diagnostic,
        *,
        dsl_text: str,
        span: DiagnosticSpan | None,
    ) -> Diagnostic:
        labels = diagnostic.labels
        if not labels and span is not None:
            labels = (DiagnosticLabel(span=span, message="statement"),)
        return Diagnostic(
            code=diagnostic.code,
            severity=diagnostic.severity,
            message=diagnostic.message,
            location=diagnostic.location,
            stage=diagnostic.stage,
            span=diagnostic.span or span,
            labels=labels,
            notes=diagnostic.notes,
            source_text=diagnostic.source_text or dsl_text,
        )

    def _parse_data_source_statement(
        self,
        kind: str,
        segment: str,
        statement_index: int,
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

        dataset_refs: list[DatasetReference] = []
        statement_options = SetStatementOptionSpec()

        for token in statement_tokens or ():
            parsed_token = self._parse_assignment_token(token)
            if parsed_token is None:
                continue
            option_key = parsed_token[0].strip().lower()
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
        parsed_token = self._parse_assignment_token(token)
        if parsed_token is None:
            return existing, Diagnostic(
                code="PARSE_UNSUPPORTED_STATEMENT",
                severity="error",
                location=f"statement:{statement_index}",
                message=f"Unsupported SET statement option: {token}",
            )

        key, value = parsed_token
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
            parsed_token = self._parse_assignment_token(token)
            if parsed_token is not None:
                key, raw_value = parsed_token
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

        if self._label_pairs_parser is None:
            return ParsedStatement(kind="LABEL", text=segment), Diagnostic(
                code="PARSE_UNSUPPORTED_STATEMENT",
                severity="error",
                location=f"statement:{statement_index}",
                message="LABEL parser backend is unavailable.",
            )

        try:
            parsed = self._label_pairs_parser.parse(body)
        except UnexpectedInput:
            return ParsedStatement(kind="LABEL", text=segment), Diagnostic(
                code="PARSE_UNSUPPORTED_STATEMENT",
                severity="error",
                location=f"statement:{statement_index}",
                message=f"Unsupported LABEL statement syntax: {segment}",
            )

        label_map: dict[str, str] = {}
        for pair in getattr(parsed, "children", ()):
            if getattr(pair, "data", None) != "label_pair":
                continue
            if len(getattr(pair, "children", ())) != 3:
                continue
            name_token, _, value_token = pair.children
            label_map[str(name_token)] = self._strip_quoted_value(str(value_token))

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
        if self._rename_pairs_parser is None:
            return {}, Diagnostic(
                code="PARSE_RENAME_STATEMENT_INVALID",
                severity="error",
                location=f"statement:{statement_index}",
                message="RENAME parser backend is unavailable.",
            )

        try:
            parsed = self._rename_pairs_parser.parse(body)
        except UnexpectedInput:
            return {}, Diagnostic(
                code="PARSE_RENAME_STATEMENT_INVALID",
                severity="error",
                location=f"statement:{statement_index}",
                message=f"Invalid rename mapping: {body}",
            )

        rename_map: dict[str, str] = {}
        for pair in getattr(parsed, "children", ()): 
            if getattr(pair, "data", None) != "rename_pair":
                continue
            if len(getattr(pair, "children", ())) != 3:
                return {}, Diagnostic(
                    code="PARSE_RENAME_STATEMENT_INVALID",
                    severity="error",
                    location=f"statement:{statement_index}",
                    message=f"Invalid rename mapping: {body}",
                )
            old_name, _, new_name = pair.children
            normalized_old = old_name.strip()
            normalized_new = new_name.strip()
            if not normalized_old or not normalized_new:
                return {}, Diagnostic(
                    code="PARSE_RENAME_STATEMENT_INVALID",
                    severity="error",
                    location=f"statement:{statement_index}",
                    message=f"Invalid rename mapping: {body}",
                )
            rename_map[normalized_old] = normalized_new
        return rename_map, None

    def _strip_quoted_value(self, value: str) -> str:
        normalized = value.strip()
        if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {'"', "'"}:
            return normalized[1:-1]
        return normalized

    def _parse_if_statement_spec(self, segment: str, kind: str, variant: str) -> IfStatementSpec | None:
        if self._if_structured_parser is None:
            return None

        source_text = segment.strip()
        parse_source = source_text
        if kind == "ELSE IF":
            lowered = parse_source.lower()
            if not lowered.startswith("else "):
                return None
            parse_source = parse_source[4:].strip()

        try:
            parsed = self._if_structured_parser.parse(parse_source)
        except UnexpectedInput:
            return None

        root = self._root_tree(parsed, "if_stmt")
        children = list(getattr(root, "children", ()))
        if len(children) < 2:
            return None

        condition_node = children[1]
        condition = self._normalize_if_expression(parse_source, condition_node)
        if not condition:
            return None

        if len(children) == 2:
            return IfStatementSpec(condition=condition, is_subset=True)

        action = self._slice_node_text(parse_source, children[3]) if len(children) >= 4 else ""
        if not action:
            return None
        return IfStatementSpec(
            condition=condition,
            then_action=action,
            is_subset=False,
            is_then_do=(variant in {"if_then_do_stmt", "else_if_then_do_stmt"} or action.lower() == "do"),
        )

    def _normalize_if_expression(self, source_text: str, condition_node: Any) -> str:
        start = getattr(getattr(condition_node, "meta", None), "start_pos", None)
        end = getattr(getattr(condition_node, "meta", None), "end_pos", None)
        if not isinstance(start, int) or not isinstance(end, int) or end < start:
            return self._slice_node_text(source_text, condition_node)

        tokens = self._collect_tokens(condition_node)
        if not tokens:
            return source_text[start:end].strip()

        normalized_parts: list[str] = []
        cursor = start
        for token in sorted(tokens, key=lambda item: getattr(item, "start_pos", 0)):
            token_start = getattr(token, "start_pos", cursor)
            token_end = getattr(token, "end_pos", token_start)
            normalized_parts.append(source_text[cursor:token_start])
            normalized_parts.append(self._normalize_if_token_text(token))
            cursor = token_end
        normalized_parts.append(source_text[cursor:end])
        return "".join(normalized_parts).strip()

    def _collect_tokens(self, node: Any) -> list[Any]:
        tokens: list[Any] = []
        for child in getattr(node, "children", ()):
            if hasattr(child, "children"):
                tokens.extend(self._collect_tokens(child))
                continue
            tokens.append(child)
        return tokens

    @staticmethod
    def _normalize_if_token_text(token: Any) -> str:
        token_type = getattr(token, "type", "")
        token_text = str(token)
        if token_type == "COMPARE_WORD":
            mapping = {
                "eq": "==",
                "ne": "!=",
                "gt": ">",
                "lt": "<",
                "ge": ">=",
                "le": "<=",
            }
            return mapping.get(token_text.lower(), token_text)
        if token_type == "BOOL_WORD":
            return token_text.lower()
        if token_type == "SYMBOL_OP":
            if token_text == "=":
                return "=="
            if token_text in {"^=", "~=", "¬="}:
                return "!="
        return token_text

    def _parse_do_statement_spec(self, segment: str) -> DoStatementSpec | None:
        if self._do_loop_parser is None:
            return None

        source_text = segment.strip()
        try:
            parsed = self._do_loop_parser.parse(source_text)
        except UnexpectedInput:
            return None

        root = self._root_tree(parsed, "do_stmt")
        children = list(getattr(root, "children", ()))
        if len(children) != 6:
            return None
        return DoStatementSpec(
            loop_var=str(children[1]),
            start_expr=self._slice_node_text(source_text, children[3]),
            end_expr=self._slice_node_text(source_text, children[5]),
        )

    def _parse_array_statement_spec(self, segment: str) -> ArrayStatementSpec | None:
        if self._array_declaration_parser is None:
            return None

        try:
            parsed = self._array_declaration_parser.parse(segment.strip())
        except UnexpectedInput:
            return None

        root = self._root_tree(parsed, "array_stmt")
        children = list(getattr(root, "children", ()))
        if len(children) < 3:
            return None

        declared_size: int | None = None
        wildcard_size = False
        character_array = False
        variables: tuple[str, ...] = ()
        for child in children[2:]:
            child_data = getattr(child, "data", None)
            if child_data == "wildcard_size":
                wildcard_size = True
                continue
            if child_data == "bracket_size":
                declared_size = int(str(child.children[0]).strip("[] "))
                continue
            if child_data == "numeric_size":
                declared_size = int(str(child.children[0]))
                continue
            if child_data == "character_array":
                character_array = True
                continue
            if child_data == "variable_list":
                variables = tuple(str(token) for token in child.children)

        if not variables:
            return None

        return ArrayStatementSpec(
            array_name=str(children[1]),
            variables=variables,
            declared_size=declared_size,
            wildcard_size=wildcard_size,
            character_array=character_array,
        )

    def _parse_assignment_token(self, token: str) -> tuple[str, str] | None:
        if self._assignment_token_parser is None:
            return None

        try:
            parsed = self._assignment_token_parser.parse(token)
        except UnexpectedInput:
            return None

        root = self._root_tree(parsed, "assignment_token")
        children = list(getattr(root, "children", ()))
        if len(children) != 3:
            return None
        return str(children[0]), str(children[2]).strip()

    @staticmethod
    def _root_tree(parsed: Any, expected: str) -> Any:
        if getattr(parsed, "data", None) == expected:
            return parsed
        children = getattr(parsed, "children", ())
        if children and getattr(children[0], "data", None) == expected:
            return children[0]
        return parsed

    @staticmethod
    def _slice_node_text(source_text: str, node: Any) -> str:
        node_meta = getattr(node, "meta", None)
        start = getattr(node_meta, "start_pos", None)
        end = getattr(node_meta, "end_pos", None)
        if isinstance(start, int) and isinstance(end, int) and end >= start:
            return source_text[start:end].strip()
        return str(node).strip()


class SplitStageParserService:
    _RULE_KIND_MAP: dict[str, str] = {
        "data_stmt": "DATA",
        "run_stmt": "RUN",
        "skip_stmt": "SKIP",
        "other_stmt": "OTHER",
    }

    def __init__(self) -> None:
        self._lark_parser = build_lark_parser_from_file("datastep_split.lark")

    def extract_statement_regions(self, dsl_text: str) -> tuple[StatementRegion, ...] | None:
        if self._lark_parser is None:
            return None

        try:
            tree = self._lark_parser.parse(dsl_text)
        except UnexpectedInput:
            return None

        regions: list[StatementRegion] = []
        for statement_node in getattr(tree, "children", ()): 
            children = getattr(statement_node, "children", ())
            if not children:
                continue
            rule_node = children[0]
            rule_name = str(getattr(rule_node, "data", ""))
            kind = self._RULE_KIND_MAP.get(rule_name)
            if kind is None:
                continue
            node_meta = getattr(rule_node, "meta", None)
            if node_meta is None:
                node_meta = getattr(statement_node, "meta", None)
            if node_meta is None:
                continue
            start = max(getattr(node_meta, "start_pos", 0), 0)
            end = max(getattr(node_meta, "end_pos", start), start)
            regions.append(StatementRegion(kind=kind, start=start, end=end))
        return tuple(regions)

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

        return result

class ParserBackendSelector:
    def __init__(self, python_backend: ParserBackend, rust_backend: ParserBackend) -> None:
        self._python_backend = python_backend
        self._rust_backend = rust_backend

    def select(self, preferred_backend: str) -> ParserBackend:
        preferred = preferred_backend.strip().lower()
        if preferred == "rust" and self._rust_backend.is_available():
            return self._rust_backend
        return self._python_backend