from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .models import Diagnostic, DiagnosticLabel, DiagnosticSpan


class _PythonStatementExecutionMixin:
    _EXECUTE_LABELS = {
        "RUNTIME_EXPRESSION_EVALUATION_ERROR": "expression error",
        "RUNTIME_UNSUPPORTED_FUNCTION": "unsupported function",
        "RUNTIME_FUNCTION_ARGUMENT_INVALID": "invalid arguments",
        "RUNTIME_OPERATOR_NOT_SUPPORTED": "operator error",
        "RUNTIME_INVALID_REFERENCE": "invalid reference",
        "RUNTIME_LOOP_EVALUATION_ERROR": "loop error",
        "RUNTIME_LOOP_LIMIT_EXCEEDED": "loop limit",
        "RUNTIME_ARRAY_INVALID": "array error",
        "RUNTIME_SUM_STATEMENT_INVALID": "invalid sum statement",
    }

    def _extract_executable_statements(self, ast_statements: Sequence[Any]) -> list[Any]:
        return [
            statement
            for statement in ast_statements
            if statement.kind not in {"DATA", "SET", "MERGE", "BY", "WHERE", "DROP", "KEEP", "RENAME", "RUN"}
        ]

    def _uses_row_view_functions(self, ast_statements: Sequence[Any]) -> bool:
        for statement in ast_statements:
            statement_text = getattr(statement, "text", "")
            if isinstance(statement_text, str) and self._ROW_VIEW_FUNCTION_PATTERN.search(statement_text):
                return True
        return False

    def _run_statement_block(
        self,
        statements: Sequence[Any],
        index: int,
        stop_index: int,
        row: dict[str, Any],
        context: Any,
        emitted_rows: list[tuple[str | None, dict[str, Any]]],
        array_defs: dict[str, tuple[str, ...]],
        sum_state: dict[str, Any],
        nesting_level: int = 0,
    ) -> tuple[bool, bool, Diagnostic | None]:
        if nesting_level > self._MAX_DO_NESTING:
            return False, False, self._attach_execute_context(
                Diagnostic(
                    code="RUNTIME_LOOP_NESTING_LIMIT_EXCEEDED",
                    severity="error",
                    message=f"DO block nesting exceeded limit: {self._MAX_DO_NESTING}",
                ),
                statement=statements[index] if index < len(statements) else None,
            )

        cursor = index
        while cursor < stop_index:
            statement = statements[cursor]

            if statement.kind == "END":
                return False, False, None

            if statement.kind == "DO":
                end_index = self._find_matching_end(statements, cursor, stop_index)
                if end_index < 0:
                    return False, False, self._attach_execute_context(
                        Diagnostic(
                            code="RUNTIME_LOOP_EVALUATION_ERROR",
                            severity="error",
                            message="DO statement is missing matching END.",
                        ),
                        statement=statement,
                    )

                spec = self._parse_do_to_spec(statement)
                if spec is None:
                    return False, False, self._attach_execute_context(
                        Diagnostic(
                            code="RUNTIME_LOOP_EVALUATION_ERROR",
                            severity="error",
                            message=f"Unsupported DO syntax: {statement.text}",
                        ),
                        statement=statement,
                    )

                loop_var, start_expr, end_expr = spec
                start_value, start_error = self._evaluate_scalar(start_expr, row=row, context=context, array_defs=array_defs)
                if start_error is not None:
                    return False, False, self._attach_execute_context(start_error, statement=statement, expression_text=start_expr)
                end_value, end_error = self._evaluate_scalar(end_expr, row=row, context=context, array_defs=array_defs)
                if end_error is not None:
                    return False, False, self._attach_execute_context(end_error, statement=statement, expression_text=end_expr)

                try:
                    start_int = int(start_value)
                    end_int = int(end_value)
                except Exception:
                    return False, False, self._attach_execute_context(
                        Diagnostic(
                            code="RUNTIME_LOOP_EVALUATION_ERROR",
                            severity="error",
                            message=f"DO bounds must be numeric: {statement.text}",
                        ),
                        statement=statement,
                    )

                for iteration, value in enumerate(range(start_int, end_int + 1), start=1):
                    if iteration > 10000:
                        return False, False, self._attach_execute_context(
                            Diagnostic(
                                code="RUNTIME_LOOP_LIMIT_EXCEEDED",
                                severity="error",
                                message="DO TO loop exceeded safety iteration limit.",
                            ),
                            statement=statement,
                        )
                    self._runtime.set_row_value(row, context, loop_var, value)
                    deleted, stopped, nested_error = self._run_statement_block(
                        statements=statements,
                        index=cursor + 1,
                        stop_index=end_index,
                        row=row,
                        context=context,
                        emitted_rows=emitted_rows,
                        array_defs=array_defs,
                        sum_state=sum_state,
                        nesting_level=nesting_level + 1,
                    )
                    if nested_error is not None:
                        return False, False, nested_error
                    if stopped:
                        return deleted, True, None
                    if deleted:
                        return True, False, None

                cursor = end_index + 1
                continue

            if statement.kind == "ARRAY":
                declaration = self._parse_array_declaration(statement)
                if declaration is None:
                    return False, False, self._attach_execute_context(
                        Diagnostic(
                            code="RUNTIME_ARRAY_INVALID",
                            severity="error",
                            message=f"Invalid ARRAY declaration: {statement.text}",
                        ),
                        statement=statement,
                    )
                array_name, variables = declaration
                array_defs[array_name] = variables
                cursor += 1
                continue

            if statement.kind == "RETAIN":
                cursor += 1
                continue

            if statement.kind == "SUM":
                sum_error = self._apply_sum_statement(
                    statement.text,
                    row=row,
                    context=context,
                    array_defs=array_defs,
                    sum_state=sum_state,
                )
                if sum_error is not None:
                    matched = self._SUM_STATEMENT.match(statement.text)
                    expression_text = matched.group(2).strip() if matched is not None else None
                    return False, False, self._attach_execute_context(sum_error, statement=statement, expression_text=expression_text)
                cursor += 1
                continue

            if statement.kind == "ASSIGN":
                assign_error = self._apply_assignment_statement(statement.text, row=row, context=context, array_defs=array_defs)
                if assign_error is not None:
                    matched = self._ASSIGN_STATEMENT.match(statement.text)
                    expression_text = matched.group(2).strip() if matched is not None else None
                    return False, False, self._attach_execute_context(assign_error, statement=statement, expression_text=expression_text)
                cursor += 1
                continue

            if statement.kind == "IF":
                if_do_condition = self._parse_if_then_do_condition(statement)
                if if_do_condition is not None:
                    end_index = self._find_matching_end_for_if_do(statements, cursor, stop_index)
                    if end_index < 0:
                        return False, False, self._attach_execute_context(
                            Diagnostic(
                                code="RUNTIME_LOOP_EVALUATION_ERROR",
                                severity="error",
                                message="IF THEN DO block is missing matching END.",
                            ),
                            statement=statement,
                        )

                    try:
                        matched = self._runtime.passes_where(row, context, if_do_condition)
                    except self._runtime.RuntimeExecutionError as error:
                        return False, False, self._attach_execute_context(
                            error.diagnostic,
                            statement=statement,
                            expression_text=self._extract_if_condition_source(statement.text),
                        )

                    if matched:
                        deleted, stopped, nested_error = self._run_statement_block(
                            statements=statements,
                            index=cursor + 1,
                            stop_index=end_index,
                            row=row,
                            context=context,
                            emitted_rows=emitted_rows,
                            array_defs=array_defs,
                            sum_state=sum_state,
                            nesting_level=nesting_level + 1,
                        )
                        if nested_error is not None:
                            return False, False, nested_error
                        if stopped:
                            return deleted, True, None
                        if deleted:
                            return True, False, None

                    cursor = end_index + 1
                    if cursor < stop_index and statements[cursor].kind == "ELSE":
                        else_text = statements[cursor].text.strip()
                        else_action = else_text[len("else"):].strip() if else_text.lower().startswith("else") else ""

                        if else_action.lower() == "do":
                            else_end_index = self._find_matching_end(statements, cursor, stop_index)
                            if else_end_index < 0:
                                return False, False, self._attach_execute_context(
                                    Diagnostic(
                                        code="RUNTIME_LOOP_EVALUATION_ERROR",
                                        severity="error",
                                        message="ELSE DO block is missing matching END.",
                                    ),
                                    statement=statements[cursor],
                                )
                            if not matched:
                                deleted, stopped, nested_error = self._run_statement_block(
                                    statements=statements,
                                    index=cursor + 1,
                                    stop_index=else_end_index,
                                    row=row,
                                    context=context,
                                    emitted_rows=emitted_rows,
                                    array_defs=array_defs,
                                    sum_state=sum_state,
                                    nesting_level=nesting_level + 1,
                                )
                                if nested_error is not None:
                                    return False, False, nested_error
                                if stopped:
                                    return deleted, True, None
                                if deleted:
                                    return True, False, None
                            cursor = else_end_index + 1
                            continue

                        if not matched and else_action:
                            deleted, action_error = self._execute_inline_action(
                                action_text=else_action,
                                row=row,
                                context=context,
                                emitted_rows=emitted_rows,
                                array_defs=array_defs,
                                sum_state=sum_state,
                            )
                            if action_error is not None:
                                return False, False, self._attach_execute_context(
                                    action_error,
                                    statement=statements[cursor],
                                    expression_text=else_action,
                                )
                            if deleted:
                                return True, False, None
                        cursor += 1
                        continue

                    continue

                parsed_if_action = self._parse_if_then_action(statement)
                if parsed_if_action is not None:
                    condition, action_text = parsed_if_action
                    try:
                        matched = self._runtime.passes_where(row, context, condition)
                    except self._runtime.RuntimeExecutionError as error:
                        return False, False, self._attach_execute_context(
                            error.diagnostic,
                            statement=statement,
                            expression_text=self._extract_if_condition_source(statement.text),
                        )

                    consumed_else = False
                    if matched:
                        deleted, action_error = self._execute_inline_action(
                            action_text=action_text,
                            row=row,
                            context=context,
                            emitted_rows=emitted_rows,
                            array_defs=array_defs,
                            sum_state=sum_state,
                        )
                        if action_error is not None:
                            return False, False, self._attach_execute_context(
                                action_error,
                                statement=statement,
                                expression_text=action_text,
                            )
                        if row.pop("__stop__", False):
                            return False, True, None
                        if deleted:
                            return True, False, None
                        if cursor + 1 < stop_index and statements[cursor + 1].kind == "ELSE":
                            consumed_else = True
                    elif cursor + 1 < stop_index and statements[cursor + 1].kind == "ELSE":
                        else_text = statements[cursor + 1].text.strip()
                        else_action = else_text[len("else"):].strip()
                        deleted, action_error = self._execute_inline_action(
                            action_text=else_action,
                            row=row,
                            context=context,
                            emitted_rows=emitted_rows,
                            array_defs=array_defs,
                            sum_state=sum_state,
                        )
                        if action_error is not None:
                            return False, False, self._attach_execute_context(
                                action_error,
                                statement=statements[cursor + 1],
                                expression_text=else_action,
                            )
                        if row.pop("__stop__", False):
                            return False, True, None
                        if deleted:
                            return True, False, None
                        consumed_else = True

                    cursor += 2 if consumed_else else 1
                    continue

                subset_if_condition = self._parse_subset_if_condition(statement)
                if subset_if_condition is not None:
                    try:
                        matched = self._runtime.passes_where(row, context, subset_if_condition)
                    except self._runtime.RuntimeExecutionError as error:
                        return False, False, self._attach_execute_context(
                            error.diagnostic,
                            statement=statement,
                            expression_text=self._extract_if_condition_source(statement.text),
                        )
                    if not matched:
                        return True, False, None
                    cursor += 1
                    continue
                cursor += 1
                continue

            if statement.kind == "DELETE":
                return True, False, None

            if statement.kind == "STOP":
                return False, True, None

            if statement.kind == "OUTPUT":
                tokens = statement.text.split()
                emitted_rows.append((tokens[1] if len(tokens) > 1 else None, self._runtime.export_row(row, context)))
                cursor += 1
                continue

            cursor += 1

        return False, False, None

    def _attach_execute_context(
        self,
        diagnostic: Diagnostic,
        *,
        statement: Any | None,
        expression_text: str | None = None,
    ) -> Diagnostic:
        if diagnostic.source_text and diagnostic.span is not None:
            return diagnostic

        statement_text = getattr(statement, "text", None)
        if not isinstance(statement_text, str) or not statement_text.strip():
            return diagnostic

        source_text = statement_text.strip()
        source_id = f"<execute:{str(getattr(statement, 'kind', 'statement')).lower().replace(' ', '-')}>"
        span = self._local_execute_span(source_text, expression_text=expression_text, source_id=source_id)
        if span is None:
            return diagnostic

        labels = diagnostic.labels or (
            DiagnosticLabel(span=span, message=self._EXECUTE_LABELS.get(diagnostic.code, "execute error")),
        )
        return Diagnostic(
            code=diagnostic.code,
            severity=diagnostic.severity,
            message=diagnostic.message,
            location=diagnostic.location,
            stage=diagnostic.stage,
            span=span,
            labels=labels,
            notes=diagnostic.notes,
            source_text=source_text,
        )

    def _local_execute_span(
        self,
        source_text: str,
        *,
        expression_text: str | None,
        source_id: str,
    ) -> DiagnosticSpan | None:
        stripped = source_text.strip()
        if not stripped:
            return None
        if expression_text:
            start = source_text.lower().find(expression_text.strip().lower())
            if start >= 0:
                end = start + len(expression_text.strip())
                return DiagnosticSpan(
                    start=start,
                    end=max(end, start + 1),
                    line=1,
                    column=start + 1,
                    end_line=1,
                    end_column=max(end, start + 1) + 1,
                    source_id=source_id,
                )

        start = source_text.find(stripped)
        end = start + len(stripped)
        return DiagnosticSpan(
            start=start,
            end=end,
            line=1,
            column=start + 1,
            end_line=1,
            end_column=end + 1,
            source_id=source_id,
        )

    @staticmethod
    def _extract_if_condition_source(statement_text: str) -> str | None:
        stripped = statement_text.strip()
        lowered = stripped.lower()
        if not lowered.startswith("if "):
            return None
        body = stripped[3:]
        then_index = body.lower().find(" then ")
        if then_index >= 0:
            return body[:then_index].strip()
        return body.strip() or None

    def _parse_if_then_action(self, statement: Any) -> tuple[str, str] | None:
        if_spec = getattr(statement, "if_spec", None)
        if if_spec is None:
            return None
        if getattr(if_spec, "is_subset", False) or getattr(if_spec, "is_then_do", False):
            return None
        condition = getattr(if_spec, "condition", None)
        action = getattr(if_spec, "then_action", None)
        if not isinstance(condition, str) or not condition or not isinstance(action, str) or not action:
            return None
        return condition, action

    def _parse_subset_if_condition(self, statement: Any) -> str | None:
        if_spec = getattr(statement, "if_spec", None)
        if if_spec is None:
            return None
        if not getattr(if_spec, "is_subset", False):
            return None
        condition = getattr(if_spec, "condition", None)
        if not isinstance(condition, str) or not condition:
            return None
        return condition

    def _execute_inline_action(
        self,
        action_text: str,
        row: dict[str, Any],
        context: Any,
        emitted_rows: list[tuple[str | None, dict[str, Any]]],
        array_defs: Mapping[str, tuple[str, ...]],
        sum_state: dict[str, Any],
    ) -> tuple[bool, Diagnostic | None]:
        normalized = action_text.strip()
        if not normalized:
            return False, None

        lowered = normalized.lower()
        if lowered == "delete":
            return True, None
        if lowered == "stop":
            row["__stop__"] = True
            return False, None

        target = self._parse_output_target(normalized)
        if target is not None:
            if target == "__default__":
                emitted_rows.append((None, self._runtime.export_row(row, context)))
            else:
                emitted_rows.append((target, self._runtime.export_row(row, context)))
            return False, None

        if self._SUM_STATEMENT.match(normalized):
            sum_error = self._apply_sum_statement(
                normalized,
                row=row,
                context=context,
                array_defs=array_defs,
                sum_state=sum_state,
            )
            return False, sum_error

        if self._ASSIGN_STATEMENT.match(normalized):
            assign_error = self._apply_assignment_statement(normalized, row=row, context=context, array_defs=array_defs)
            return False, assign_error

        return False, Diagnostic(
            code="RUNTIME_EXPRESSION_EVALUATION_ERROR",
            severity="error",
            message=f"Unsupported IF/ELSE action syntax: {action_text}",
        )

    def _find_matching_end(self, statements: Sequence[Any], do_index: int, stop_index: int) -> int:
        def opens_do_block(statement: Any) -> bool:
            if statement.kind == "DO":
                return True
            if statement.kind == "IF":
                return self._parse_if_then_do_condition(statement) is not None
            if statement.kind == "ELSE":
                else_text = statement.text.strip()
                else_action = else_text[len("else") :].strip() if else_text.lower().startswith("else") else ""
                return else_action.lower() == "do"
            return False

        depth = 0
        for cursor in range(do_index, stop_index):
            current = statements[cursor]
            if opens_do_block(current):
                depth += 1
            elif current.kind == "END":
                depth -= 1
                if depth == 0:
                    return cursor
        return -1

    def _find_matching_end_for_if_do(self, statements: Sequence[Any], if_index: int, stop_index: int) -> int:
        depth = 1
        for cursor in range(if_index + 1, stop_index):
            current = statements[cursor]
            if current.kind == "DO":
                depth += 1
            elif current.kind == "IF" and self._parse_if_then_do_condition(current) is not None:
                depth += 1
            elif current.kind == "END":
                depth -= 1
                if depth == 0:
                    return cursor
        return -1

    def _parse_do_to_spec(self, statement: Any) -> tuple[str, str, str] | None:
        do_spec = getattr(statement, "do_spec", None)
        if do_spec is None:
            return None
        loop_var = str(getattr(do_spec, "loop_var", "")).strip()
        start_expr = str(getattr(do_spec, "start_expr", "")).strip()
        end_expr = str(getattr(do_spec, "end_expr", "")).strip()
        if not loop_var or not start_expr or not end_expr:
            return None
        return loop_var, start_expr, end_expr

    def _parse_if_then_do_condition(self, statement: Any) -> str | None:
        if_spec = getattr(statement, "if_spec", None)
        if if_spec is None:
            return None
        if not getattr(if_spec, "is_then_do", False):
            return None
        condition = getattr(if_spec, "condition", None)
        if not isinstance(condition, str) or not condition:
            return None
        return condition

    def _parse_array_declaration(self, statement: Any) -> tuple[str, tuple[str, ...]] | None:
        array_spec = getattr(statement, "array_spec", None)
        if array_spec is None:
            return None
        array_name = str(getattr(array_spec, "array_name", "")).strip()
        variables = tuple(str(token) for token in getattr(array_spec, "variables", ()) if str(token))
        if not array_name or not variables:
            return None
        return array_name, variables

    def _extract_retain_variable_names(self, ast_statements: Sequence[Any]) -> tuple[str, ...]:
        names: list[str] = []
        for statement in ast_statements:
            if statement.kind != "RETAIN":
                continue
            body = statement.text[len("retain"):].strip()
            names.extend(token for token in body.split() if token)
        return tuple(dict.fromkeys(names))

    def _extract_sum_variable_names(self, ast_statements: Sequence[Any]) -> tuple[str, ...]:
        names: list[str] = []
        for statement in ast_statements:
            if statement.kind != "SUM":
                continue
            matched = self._SUM_STATEMENT.match(statement.text)
            if matched is None:
                continue
            names.append(matched.group(1))
        return tuple(dict.fromkeys(names))

    def _apply_sum_statement(
        self,
        statement_text: str,
        row: dict[str, Any],
        context: Any,
        array_defs: Mapping[str, tuple[str, ...]],
        sum_state: Mapping[str, Any] | None = None,
    ) -> Diagnostic | None:
        matched = self._SUM_STATEMENT.match(statement_text)
        if matched is None:
            return Diagnostic(
                code="RUNTIME_SUM_STATEMENT_INVALID",
                severity="error",
                message=f"Invalid sum statement: {statement_text}",
            )

        variable_name = matched.group(1)
        expression = matched.group(2)
        add_value, add_error = self._evaluate_scalar(expression, row=row, context=context, array_defs=array_defs)
        if add_error is not None:
            return add_error

        resolved_name = self._runtime.resolve_row_name(row, context, variable_name)
        current = self._runtime.get_row_value(row, context, resolved_name)
        if self._is_missing_value(current) and sum_state is not None and resolved_name in sum_state:
            current = sum_state.get(resolved_name)
        current_numeric = 0.0 if self._is_missing_value(current) else float(current)
        add_numeric = 0.0 if self._is_missing_value(add_value) else float(add_value)
        summed = current_numeric + add_numeric
        stored_value = int(summed) if float(summed).is_integer() else summed
        resolved_name = self._runtime.set_row_value(row, context, resolved_name, stored_value)
        if sum_state is not None:
            sum_state[resolved_name] = stored_value
        return None

    def _apply_assignment_statement(
        self,
        statement_text: str,
        row: dict[str, Any],
        context: Any,
        array_defs: Mapping[str, tuple[str, ...]],
    ) -> Diagnostic | None:
        matched = self._ASSIGN_STATEMENT.match(statement_text)
        if matched is None:
            return Diagnostic(
                code="RUNTIME_EXPRESSION_EVALUATION_ERROR",
                severity="error",
                message=f"Invalid assignment statement: {statement_text}",
            )

        left = matched.group(1).strip()
        right = matched.group(2).strip()
        right_value, right_error = self._evaluate_scalar(right, row=row, context=context, array_defs=array_defs)
        if right_error is not None:
            return right_error

        array_ref = self._parse_array_reference(left)
        if array_ref is None:
            self._runtime.set_row_value(row, context, left, right_value)
            return None

        array_name, index_expr = array_ref
        index_value, index_error = self._evaluate_scalar(index_expr, row=row, context=context, array_defs=array_defs)
        if index_error is not None:
            return index_error
        variable_name, variable_error = self._resolve_array_variable(array_name, index_value, array_defs)
        if variable_error is not None:
            return variable_error
        self._runtime.set_row_value(row, context, variable_name, right_value)
        return None

    def _evaluate_scalar(
        self,
        expression: str,
        row: Mapping[str, Any],
        context: Any,
        array_defs: Mapping[str, tuple[str, ...]],
    ) -> tuple[Any, Diagnostic | None]:
        return self._evaluator.evaluate_scalar(expression, row=row, context=context, array_defs=array_defs)

    def _transform_array_expression(self, expression: str, array_defs: Mapping[str, tuple[str, ...]]) -> str:
        return self._evaluator.transform_array_expression(expression, array_defs=array_defs)

    def _transform_dotted_variable_expression(self, expression: str, row: Mapping[str, Any]) -> str:
        return self._evaluator.transform_dotted_variable_expression(expression, row=row)

    def _parse_array_reference(self, token: str) -> tuple[str, str] | None:
        return self._evaluator.parse_array_reference(token)

    def _resolve_array_variable(
        self,
        array_name: str,
        index_value: Any,
        array_defs: Mapping[str, tuple[str, ...]],
    ) -> tuple[str, Diagnostic | None]:
        return self._evaluator.resolve_array_variable(array_name, index_value, array_defs)

    def _is_missing_value(self, value: Any) -> bool:
        return self._evaluator.is_missing_value(value)

    def _normalize_expression(self, expression: str) -> str:
        return self._evaluator.normalize_expression(expression)

    def _normalize_concat_operator(self, expression: str) -> str:
        return self._evaluator.normalize_concat_operator(expression)