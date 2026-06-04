use crate::ast::{self, AstDatasetRefOptions, AstStatement};
use crate::expressions::{
    self,
    condition_syntax_error_message,
    evaluate_scalar_expression,
    get_stateful_row_item,
    get_stateful_scalar_value,
    is_expression_parse_error,
    optional_source_values_equal,
    py_to_scalar,
    resolve_array_variable_name,
    scalar_to_bool,
    scalar_to_number,
    Expr,
    ScalarValue,
};
use crate::output::{
    all_output_builders_use_planned_columns,
    append_projected_row_to_target,
    export_output_streams,
    initialize_output_builders,
    resolve_declared_target_name,
};
use crate::row_cursor::NativeArrowRowCursor;
use crate::value_ref::OwnedValue;
use crate::apply_dataset_ref_options_to_row;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::time::Instant;

mod state;
use self::state::{
    mark_automatic_error,
    reset_automatic_scalars,
    set_runtime_flag,
    set_runtime_text,
    store_runtime_scalar,
};

#[derive(Default)]
pub(crate) struct EvalRuntimeState {
    pub(crate) lag_queues: HashMap<usize, Vec<ScalarValue>>,
    pub(crate) sum_totals: HashMap<String, f64>,
    pub(crate) retain_values: HashMap<String, OwnedValue>,
    pub(crate) automatic_values: HashMap<String, OwnedValue>,
    pub(crate) mutable_values: HashMap<String, OwnedValue>,
    pub(crate) materialize_mutable_values: bool,
    pub(crate) array_defs: HashMap<String, Vec<String>>,
    pub(crate) parsed_expr_cache: HashMap<String, Expr>,
    pub(crate) parsed_assignment_cache: HashMap<String, Option<(String, String)>>,
    pub(crate) parsed_sum_cache: HashMap<String, Option<(String, String)>>,
    pub(crate) dataset_option_survivor_counts: HashMap<String, usize>,
    pub(crate) row_index: isize,
    pub(crate) source_cursor: Option<Rc<NativeArrowRowCursor>>,
    pub(crate) apply_registry: HashMap<String, Py<PyAny>>,
    pub(crate) numeric_format_catalogs: HashMap<String, Vec<(ScalarValue, String)>>,
    pub(crate) character_format_catalogs: HashMap<String, Vec<(ScalarValue, String)>>,
    pub(crate) informat_catalogs: HashMap<String, Vec<(ScalarValue, f64)>>,
}

#[derive(Default)]
pub(crate) struct RowExecutionOutcome {
    pub(crate) deleted: bool,
    pub(crate) stopped: bool,
}

fn parse_text_catalog_group(
    catalogs_any: Option<&Bound<'_, PyAny>>,
) -> PyResult<HashMap<String, Vec<(ScalarValue, String)>>> {
    let Some(catalogs_any) = catalogs_any else {
        return Ok(HashMap::new());
    };
    let catalogs_dict = catalogs_any
        .cast::<PyDict>()
        .map_err(|_| PyValueError::new_err("format catalog group must be a dict"))?;
    let mut resolved = HashMap::new();
    for (name_any, entries_any) in catalogs_dict.iter() {
        let name = name_any.extract::<String>()?;
        let entries_dict = entries_any
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err("format catalog entries must be a dict"))?;
        let mut entries = Vec::new();
        for (key_any, value_any) in entries_dict.iter() {
            let key = py_to_scalar(&key_any)?;
            let value = value_any.extract::<String>()?;
            entries.push((key, value));
        }
        resolved.insert(name, entries);
    }
    Ok(resolved)
}

fn parse_numeric_catalog_group(
    catalogs_any: Option<&Bound<'_, PyAny>>,
) -> PyResult<HashMap<String, Vec<(ScalarValue, f64)>>> {
    let Some(catalogs_any) = catalogs_any else {
        return Ok(HashMap::new());
    };
    let catalogs_dict = catalogs_any
        .cast::<PyDict>()
        .map_err(|_| PyValueError::new_err("informat catalog group must be a dict"))?;
    let mut resolved = HashMap::new();
    for (name_any, entries_any) in catalogs_dict.iter() {
        let name = name_any.extract::<String>()?;
        let entries_dict = entries_any
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err("informat catalog entries must be a dict"))?;
        let mut entries = Vec::new();
        for (key_any, value_any) in entries_dict.iter() {
            let key = py_to_scalar(&key_any)?;
            let value = value_any.extract::<f64>()?;
            entries.push((key, value));
        }
        resolved.insert(name, entries);
    }
    Ok(resolved)
}

fn diag(py: Python<'_>, code: &str, location: &str, message: &str) -> PyResult<Py<PyDict>> {
    let item = PyDict::new(py);
    item.set_item("code", code)?;
    item.set_item("severity", "error")?;
    item.set_item("location", location)?;
    item.set_item("message", message)?;
    Ok(item.unbind())
}

fn statement_body(text: &str, keyword: &str) -> String {
    let lowered = text.to_lowercase();
    if lowered.starts_with(&format!("{} ", keyword)) {
        return text[keyword.len()..].trim().to_string();
    }
    if lowered == keyword {
        return String::new();
    }
    text.to_string()
}

fn parse_assignment(text: &str) -> Option<(String, String)> {
    let (left, right) = text.split_once('=')?;
    let name = left.trim();
    let expr = right.trim();
    if name.is_empty() || expr.is_empty() {
        return None;
    }
    Some((name.to_string(), expr.to_string()))
}

fn parse_array_reference_token(token: &str) -> Option<(String, String)> {
    let candidates = [('(', ')'), ('[', ']'), ('{', '}')];
    for (open, close) in candidates {
        let Some(start) = token.find(open) else {
            continue;
        };
        if !token.ends_with(close) || start == 0 {
            continue;
        }
        let name = token[..start].trim();
        let index_expr = token[start + 1..token.len() - 1].trim();
        if name.is_empty() || index_expr.is_empty() {
            continue;
        }
        return Some((name.to_string(), index_expr.to_string()));
    }
    None
}

fn parse_sum_statement(text: &str) -> Option<(String, String)> {
    let (left, right) = text.split_once('+')?;
    let name = left.trim();
    let expr = right.trim();
    if name.is_empty() || expr.is_empty() {
        return None;
    }
    Some((name.to_string(), expr.to_string()))
}

fn parse_assignment_cached(text: &str, state: &mut EvalRuntimeState) -> Option<(String, String)> {
    if let Some(cached) = state.parsed_assignment_cache.get(text) {
        return cached.clone();
    }
    let parsed = parse_assignment(text);
    state
        .parsed_assignment_cache
        .insert(text.to_string(), parsed.clone());
    parsed
}

fn parse_sum_statement_cached(text: &str, state: &mut EvalRuntimeState) -> Option<(String, String)> {
    if let Some(cached) = state.parsed_sum_cache.get(text) {
        return cached.clone();
    }
    let parsed = parse_sum_statement(text);
    state
        .parsed_sum_cache
        .insert(text.to_string(), parsed.clone());
    parsed
}

fn dataset_ref_options_are_passthrough(options: &AstDatasetRefOptions) -> bool {
    options.keep_vars.is_empty()
        && options.drop_vars.is_empty()
        && options.where_expr.is_none()
        && options.rename_map.is_empty()
        && options.firstobs.is_none()
        && options.obs.is_none()
}

fn dataset_option_error_message(scope_key: &str, error: &str) -> String {
    let (subject_kind, subject_name) = if let Some(name) = scope_key.strip_prefix("input:") {
        ("source", name)
    } else if let Some(name) = scope_key.strip_prefix("output:") {
        ("output", name)
    } else {
        ("dataset", scope_key)
    };

    if let Some(detail) = error.strip_prefix("dataset option WHERE= evaluation failed: ") {
        return format!("Dataset option WHERE= evaluation failed for {subject_kind} '{subject_name}': {detail}");
    }
    format!("Invalid dataset option for {subject_kind} '{subject_name}': {error}")
}

fn statement_if_then_do(statement: &AstStatement) -> bool {
    statement.if_spec.as_ref().map(|spec| spec.is_then_do).unwrap_or(false)
}

fn statement_if_then_action(statement: &AstStatement) -> Option<(String, String)> {
    let spec = statement.if_spec.as_ref()?;
    let action = spec.then_action.as_ref()?;
    let condition = spec.condition.trim();
    let action_trimmed = action.trim();
    if condition.is_empty() || action_trimmed.is_empty() || spec.is_subset || spec.is_then_do {
        return None;
    }
    Some((condition.to_string(), action_trimmed.to_string()))
}

fn statement_subset_if_condition(statement: &AstStatement) -> Option<String> {
    let spec = statement.if_spec.as_ref()?;
    if !spec.is_subset {
        return None;
    }
    let condition = spec.condition.trim();
    if condition.is_empty() {
        return None;
    }
    Some(condition.to_string())
}

fn statement_do_spec(statement: &AstStatement) -> Option<(String, String, String)> {
    let spec = statement.do_spec.as_ref()?;
    let loop_var = spec.loop_var.trim();
    let start_expr = spec.start_expr.trim();
    let end_expr = spec.end_expr.trim();
    if loop_var.is_empty() || start_expr.is_empty() || end_expr.is_empty() {
        return None;
    }
    Some((
        loop_var.to_string(),
        start_expr.to_string(),
        end_expr.to_string(),
    ))
}

fn statement_array_spec(statement: &AstStatement) -> Option<(String, Vec<String>)> {
    let spec = statement.array_spec.as_ref()?;
    let array_name = spec.array_name.trim();
    if array_name.is_empty() {
        return None;
    }
    let variables: Vec<String> = spec
        .variables
        .iter()
        .map(|item| item.trim())
        .filter(|item| !item.is_empty())
        .map(|item| item.to_string())
        .collect();
    if variables.is_empty() {
        return None;
    }
    Some((array_name.to_string(), variables))
}

fn find_matching_end(statements: &[AstStatement], start: usize, stop: usize) -> Option<usize> {
    let mut depth = 1;
    let mut cursor = start + 1;
    while cursor < stop {
        let statement = &statements[cursor];
        if statement.kind == "DO" || (statement.kind == "IF" && statement_if_then_do(statement)) {
            depth += 1;
        } else if statement.kind == "END" {
            depth -= 1;
            if depth == 0 {
                return Some(cursor);
            }
        }
        cursor += 1;
    }
    None
}

fn has_inline_output_action(statement: &AstStatement) -> bool {
    if statement.kind == "OUTPUT" {
        return true;
    }
    if statement.kind == "IF" || statement.kind == "ELSE IF" || statement.kind == "ELSE" {
        if let Some(spec) = &statement.if_spec {
            if let Some(action) = &spec.then_action {
                if action.trim().to_lowercase().starts_with("output") {
                    return true;
                }
            }
        }
        return statement.text.to_lowercase().contains(" output");
    }
    false
}

fn get_plan_item<'py>(owner: &Bound<'py, PyAny>, name: &str) -> Result<Option<Bound<'py, PyAny>>, String> {
    if let Ok(dict) = owner.cast::<PyDict>() {
        return dict
            .get_item(name)
            .map_err(|error| format!("failed to read plan key '{name}': {error}"));
    }
    match owner.getattr(name) {
        Ok(value) => Ok(Some(value)),
        Err(error) => {
            if error.is_instance_of::<pyo3::exceptions::PyAttributeError>(owner.py()) {
                Ok(None)
            } else {
                Err(format!("failed to read plan attr '{name}': {error}"))
            }
        }
    }
}

fn get_plan_text(owner: &Bound<'_, PyAny>, name: &str) -> Result<Option<String>, String> {
    let Some(value) = get_plan_item(owner, name)? else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    value
        .extract::<String>()
        .map(Some)
        .map_err(|error| format!("failed to parse plan field '{name}' as text: {error}"))
}

fn get_plan_text_list(owner: &Bound<'_, PyAny>, name: &str) -> Result<Vec<String>, String> {
    let Some(value) = get_plan_item(owner, name)? else {
        return Ok(Vec::new());
    };
    if value.is_none() {
        return Ok(Vec::new());
    }
    value
        .extract::<Vec<String>>()
        .map_err(|error| format!("failed to parse plan field '{name}' as list[str]: {error}"))
}

fn execute_inline_action(
    _py: Python<'_>,
    action: &str,
    row: &Bound<'_, PyDict>,
    state: &mut EvalRuntimeState,
    outcome: &mut RowExecutionOutcome,
    output_router: &mut impl FnMut(&Bound<'_, PyDict>, Option<&str>, &mut EvalRuntimeState) -> Result<(), String>,
) -> Result<(), String> {
    let normalized = action.trim();
    if normalized.is_empty() {
        return Ok(());
    }

    let lowered = normalized.to_lowercase();
    if lowered == "delete" {
        outcome.deleted = true;
        return Ok(());
    }
    if lowered == "stop" {
        outcome.stopped = true;
        return Ok(());
    }

    if lowered.starts_with("output") {
        let body = statement_body(normalized, "output");
        if body.is_empty() {
            output_router(row, None, state)?;
        } else {
            output_router(row, Some(body.as_str()), state)?;
        }
        return Ok(());
    }

    if let Some((name, expr)) = parse_sum_statement_cached(normalized, state) {
        let increment = scalar_to_number(&evaluate_scalar_expression(row, &expr, state)?)?;

        let base = if let Some(existing) = get_stateful_scalar_value(row, &name, state)? {
            scalar_to_number(&existing).unwrap_or(0.0)
        } else {
            *state.sum_totals.entry(name.clone()).or_insert(0.0)
        };
        let total = base + increment;
        state.sum_totals.insert(name.clone(), total);
        store_runtime_scalar(row, state, &name, &ScalarValue::Number(total))?;
        return Ok(());
    }

    if let Some((name, expr)) = parse_assignment_cached(normalized, state) {
        let value = evaluate_scalar_expression(row, &expr, state)?;
        if let Some((array_name, index_expr)) = parse_array_reference_token(&name) {
            let index_value = scalar_to_number(&evaluate_scalar_expression(row, &index_expr, state)?)?;
            let variable_name = resolve_array_variable_name(state, &array_name, index_value)?;
            store_runtime_scalar(row, state, &variable_name, &value)?;
        } else {
            store_runtime_scalar(row, state, &name, &value)?;
        }
        return Ok(());
    }

    Err(format!("unsupported inline action: {normalized}"))
}

fn execute_statement_block(
    py: Python<'_>,
    statements: &[AstStatement],
    start: usize,
    stop: usize,
    row: &Bound<'_, PyDict>,
    state: &mut EvalRuntimeState,
    output_router: &mut impl FnMut(&Bound<'_, PyDict>, Option<&str>, &mut EvalRuntimeState) -> Result<(), String>,
) -> Result<(usize, RowExecutionOutcome), String> {
    let mut cursor = start;
    let mut outcome = RowExecutionOutcome::default();

    while cursor < stop {
        let statement = &statements[cursor];
        match statement.kind.as_str() {
            "END" => return Ok((cursor + 1, outcome)),
            "ARRAY" => {
                let (name, variables) = statement_array_spec(statement)
                    .ok_or_else(|| format!("invalid ARRAY declaration: {}", statement.text))?;
                state.array_defs.insert(name, variables);
                cursor += 1;
            }
            "RETAIN" => {
                cursor += 1;
            }
            "SUM" => {
                let (name, expr) = parse_sum_statement_cached(&statement.text, state)
                    .ok_or_else(|| format!("invalid SUM statement: {}", statement.text))?;
                let increment = scalar_to_number(&evaluate_scalar_expression(row, &expr, state)?)?;

                let base = if let Some(existing) = get_stateful_scalar_value(row, &name, state)? {
                    scalar_to_number(&existing).unwrap_or(0.0)
                } else {
                    *state.sum_totals.entry(name.clone()).or_insert(0.0)
                };
                let total = base + increment;
                state.sum_totals.insert(name.clone(), total);
                store_runtime_scalar(row, state, &name, &ScalarValue::Number(total))?;
                cursor += 1;
            }
            "ASSIGN" => {
                let (name, expr) = parse_assignment_cached(&statement.text, state)
                    .ok_or_else(|| format!("invalid assignment statement: {}", statement.text))?;
                let value = evaluate_scalar_expression(row, &expr, state)
                    .map_err(|error| format!("assignment expression failed: '{expr}': {error}"))?;
                if let Some((array_name, index_expr)) = parse_array_reference_token(&name) {
                    let index_value = scalar_to_number(&evaluate_scalar_expression(row, &index_expr, state)?)?;
                    let variable_name = resolve_array_variable_name(state, &array_name, index_value)?;
                    store_runtime_scalar(row, state, &variable_name, &value)?;
                } else {
                    store_runtime_scalar(row, state, &name, &value)?;
                }
                cursor += 1;
            }
            "DELETE" => {
                outcome.deleted = true;
                return Ok((cursor + 1, outcome));
            }
            "OUTPUT" => {
                let body = statement_body(&statement.text, "output");
                if body.is_empty() {
                    output_router(row, None, state)?;
                } else {
                    output_router(row, Some(body.as_str()), state)?;
                }
                cursor += 1;
            }
            "DO" => {
                let end_index = find_matching_end(statements, cursor, stop)
                    .ok_or_else(|| "DO statement missing matching END".to_string())?;
                let (var_name, from_expr, to_expr) = statement_do_spec(statement)
                    .ok_or_else(|| format!("unsupported DO statement: {}", statement.text))?;
                let from_number = scalar_to_number(&evaluate_scalar_expression(row, &from_expr, state)?)?;
                let to_number = scalar_to_number(&evaluate_scalar_expression(row, &to_expr, state)?)?;
                let from_value = from_number as i64;
                let to_value = to_number as i64;
                let mut iteration = from_value;
                while iteration <= to_value {
                    store_runtime_scalar(row, state, &var_name, &ScalarValue::Number(iteration as f64))?;
                    let (_, nested) = execute_statement_block(
                        py,
                        statements,
                        cursor + 1,
                        end_index,
                        row,
                        state,
                        output_router,
                    )?;
                    if nested.deleted {
                        outcome.deleted = true;
                        return Ok((end_index + 1, outcome));
                    }
                    if nested.stopped {
                        outcome.stopped = true;
                        return Ok((end_index + 1, outcome));
                    }
                    iteration += 1;
                }
                cursor = end_index + 1;
            }
            "IF" => {
                if statement_if_then_do(statement) {
                    let condition = statement
                        .if_spec
                        .as_ref()
                        .map(|spec| spec.condition.trim().to_string())
                        .filter(|condition| !condition.is_empty())
                        .ok_or_else(|| format!("invalid IF THEN DO statement: {}", statement.text))?;
                    let end_index = find_matching_end(statements, cursor, stop)
                        .ok_or_else(|| "IF THEN DO block missing matching END".to_string())?;
                    let matched = scalar_to_bool(&evaluate_scalar_expression(row, &condition, state)?)?;
                    if matched {
                        let (_, nested) = execute_statement_block(
                            py,
                            statements,
                            cursor + 1,
                            end_index,
                            row,
                            state,
                            output_router,
                        )?;
                        if nested.deleted {
                            outcome.deleted = true;
                            return Ok((end_index + 1, outcome));
                        }
                        if nested.stopped {
                            outcome.stopped = true;
                            return Ok((end_index + 1, outcome));
                        }
                    }
                    cursor = end_index + 1;
                    if cursor < stop && statements[cursor].kind == "ELSE" {
                        let else_text = statement_body(&statements[cursor].text, "else");
                        if else_text.to_lowercase() == "do" {
                            let else_end_index = find_matching_end(statements, cursor, stop)
                                .ok_or_else(|| "ELSE DO block missing matching END".to_string())?;
                            if !matched {
                                let (_, nested) = execute_statement_block(
                                    py,
                                    statements,
                                    cursor + 1,
                                    else_end_index,
                                    row,
                                    state,
                                    output_router,
                                )?;
                                if nested.deleted {
                                    outcome.deleted = true;
                                    return Ok((else_end_index + 1, outcome));
                                }
                                if nested.stopped {
                                    outcome.stopped = true;
                                    return Ok((else_end_index + 1, outcome));
                                }
                            }
                            cursor = else_end_index + 1;
                            continue;
                        }
                        if !matched && !else_text.is_empty() {
                            execute_inline_action(py, &else_text, row, state, &mut outcome, output_router)?;
                            if outcome.deleted {
                                return Ok((cursor + 1, outcome));
                            }
                        }
                        cursor += 1;
                        continue;
                    }
                    continue;
                }

                if let Some((condition, action)) = statement_if_then_action(statement) {
                    let matched = scalar_to_bool(&evaluate_scalar_expression(row, &condition, state)?)?;
                    if matched {
                        execute_inline_action(py, &action, row, state, &mut outcome, output_router)?;
                        if outcome.deleted || outcome.stopped {
                            return Ok((cursor + 1, outcome));
                        }
                        if cursor + 1 < stop && (statements[cursor + 1].kind == "ELSE" || statements[cursor + 1].kind == "ELSE IF") {
                            cursor += 2;
                        } else {
                            cursor += 1;
                        }
                        continue;
                    }

                    let mut next_cursor = cursor + 1;
                    if next_cursor < stop && statements[next_cursor].kind == "ELSE IF" {
                        if let Some((else_if_condition, else_if_action)) = statement_if_then_action(&statements[next_cursor]) {
                            if scalar_to_bool(&evaluate_scalar_expression(row, &else_if_condition, state)?)? {
                                execute_inline_action(py, &else_if_action, row, state, &mut outcome, output_router)?;
                                if outcome.deleted || outcome.stopped {
                                    return Ok((next_cursor + 1, outcome));
                                }
                                next_cursor += 1;
                                if next_cursor < stop && statements[next_cursor].kind == "ELSE" {
                                    next_cursor += 1;
                                }
                                cursor = next_cursor;
                                continue;
                            }
                            next_cursor += 1;
                        }
                    }
                    if next_cursor < stop && statements[next_cursor].kind == "ELSE" {
                        let else_text = statement_body(&statements[next_cursor].text, "else");
                        execute_inline_action(py, &else_text, row, state, &mut outcome, output_router)?;
                        if outcome.deleted || outcome.stopped {
                            return Ok((next_cursor + 1, outcome));
                        }
                        cursor = next_cursor + 1;
                    } else {
                        cursor += 1;
                    }
                    continue;
                }

                if let Some(condition) = statement_subset_if_condition(statement) {
                    let matched = scalar_to_bool(&evaluate_scalar_expression(row, &condition, state)?)?;
                    if !matched {
                        outcome.deleted = true;
                        return Ok((cursor + 1, outcome));
                    }
                    cursor += 1;
                    continue;
                }

                return Err(format!("invalid IF statement: {}", statement.text));
            }
            "ELSE" | "ELSE IF" => {
                cursor += 1;
            }
            "STOP" => {
                outcome.stopped = true;
                return Ok((cursor + 1, outcome));
            }
            _ => {
                cursor += 1;
            }
        }
    }

    Ok((cursor, outcome))
}

pub(crate) fn execute_block(py: Python<'_>, payload: &Bound<'_, PyDict>) -> PyResult<Py<PyDict>> {
    let result = PyDict::new(py);
    let diagnostics = PyList::empty(py);
    let input_import_start = Instant::now();
    let ast_json = payload
        .get_item("ast_json")?
        .ok_or_else(|| PyValueError::new_err("payload.ast_json is required"))?
        .extract::<String>()?;
    let runtime_statements = match ast::parse_runtime_statements(&ast_json) {
        Ok(parsed_statements) => parsed_statements,
        Err(error) => {
            diagnostics.append(diag(
                py,
                "PARSE_RUST_AST_DESERIALIZE_FAILED",
                "ast",
                &format!("Rust native parser/runtime failed to deserialize AST payload: {}", error),
            )?)?;
            result.set_item("outputs", PyDict::new(py))?;
            result.set_item("diagnostics", diagnostics)?;
            return Ok(result.unbind());
        }
    };

    let kinds: Vec<String> = runtime_statements.iter().map(|statement| statement.kind.clone()).collect();

    let supported = [
        "DATA", "SET", "WHERE", "OUTPUT", "KEEP", "DROP", "RUN", "IF", "ELSE IF", "ELSE", "DO", "END",
        "ASSIGN", "SUM", "DELETE", "RETAIN", "ARRAY", "BY", "STOP", "RENAME",
    ];
    for (index, kind) in kinds.iter().enumerate() {
        if !supported.contains(&kind.as_str()) {
            diagnostics.append(diag(
                py,
                "RUNTIME_RUST_STATEMENT_UNSUPPORTED",
                &format!("statement:{}", index + 1),
                &format!("Rust native runtime does not support statement kind: {}", kind),
            )?)?;
            result.set_item("outputs", PyDict::new(py))?;
            result.set_item("diagnostics", diagnostics)?;
            return Ok(result.unbind());
        }
    }

    let mut set_input_names: Vec<String> = Vec::new();
    let mut where_expr: Option<String> = None;
    let mut keep_vars: Vec<String> = Vec::new();
    let mut drop_vars: Vec<String> = Vec::new();
    let mut rename_map: HashMap<String, String> = HashMap::new();
    let mut output_targets_from_stmt: Vec<String> = Vec::new();
    let mut executable: Vec<AstStatement> = Vec::new();
    let mut by_vars: Vec<String> = Vec::new();
    let mut retain_vars: Vec<String> = Vec::new();
    let mut set_in_var_by_dataset: HashMap<String, String> = HashMap::new();
    let mut set_options_by_dataset: HashMap<String, AstDatasetRefOptions> = HashMap::new();
    let mut output_options_by_target: HashMap<String, AstDatasetRefOptions> = HashMap::new();
    let mut all_in_vars: Vec<String> = Vec::new();
    let mut indsname_var: Option<String> = None;
    let mut end_var: Option<String> = None;

    for statement in &runtime_statements {
        if statement.kind == "DATA" {
            for dataset_ref in &statement.dataset_refs {
                let effective_options = dataset_ref.options.clone();

                if !effective_options.keep_vars.is_empty()
                    || !effective_options.drop_vars.is_empty()
                    || !effective_options.rename_map.is_empty()
                {
                    output_options_by_target.insert(dataset_ref.name.to_lowercase(), effective_options);
                }
            }
            continue;
        }
        if statement.kind == "SET" {
            if !statement.dataset_refs.is_empty() {
                for dataset_ref in &statement.dataset_refs {
                    let effective_options = dataset_ref.options.clone();

                    set_input_names.push(dataset_ref.name.clone());
                    set_options_by_dataset.insert(dataset_ref.name.to_lowercase(), effective_options.clone());
                    if let Some(in_var_name) = &effective_options.in_var {
                        set_in_var_by_dataset.insert(dataset_ref.name.to_lowercase(), in_var_name.clone());
                        all_in_vars.push(in_var_name.clone());
                    }
                }
                indsname_var = statement.statement_options.indsname_var.clone();
                end_var = statement.statement_options.end_var.clone();
            }
            continue;
        }
        if statement.kind == "WHERE" {
            where_expr = Some(statement_body(&statement.text, "where"));
            continue;
        }
        if statement.kind == "KEEP" {
            for name in statement_body(&statement.text, "keep").split_whitespace() {
                keep_vars.push(name.to_string());
            }
            continue;
        }
        if statement.kind == "DROP" {
            for name in statement_body(&statement.text, "drop").split_whitespace() {
                drop_vars.push(name.to_string());
            }
            continue;
        }
        if statement.kind == "RENAME" {
            rename_map = statement.rename_map.clone();
            continue;
        }
        if statement.kind == "BY" {
            for name in statement_body(&statement.text, "by").split_whitespace() {
                if !name.is_empty() {
                    by_vars.push(name.to_string());
                }
            }
            continue;
        }
        if statement.kind == "RETAIN" {
            for name in statement_body(&statement.text, "retain").split_whitespace() {
                if !name.is_empty() {
                    retain_vars.push(name.to_string());
                }
            }
            executable.push(statement.clone());
            continue;
        }
        if statement.kind == "OUTPUT" {
            for target in statement_body(&statement.text, "output").split_whitespace() {
                output_targets_from_stmt.push(target.to_string());
            }
            executable.push(statement.clone());
            continue;
        }
        if !["DATA", "RUN"].contains(&statement.kind.as_str()) {
            executable.push(statement.clone());
        }
    }

    if set_input_names.is_empty() {
        diagnostics.append(diag(
            py,
            "RUNTIME_SET_DATASET_NOT_FOUND",
            "statement:set",
            "SET statement requires at least one input dataset.",
        )?)?;
        result.set_item("outputs", PyDict::new(py))?;
        result.set_item("diagnostics", diagnostics)?;
        return Ok(result.unbind());
    }

    all_in_vars.sort();
    all_in_vars.dedup();

    let mut excluded_output_vars: HashSet<String> = HashSet::new();
    for in_var in &all_in_vars {
        excluded_output_vars.insert(in_var.clone());
    }
    if let Some(name) = &indsname_var {
        excluded_output_vars.insert(name.clone());
    }
    if let Some(name) = &end_var {
        excluded_output_vars.insert(name.clone());
    }
    for by_var in &by_vars {
        excluded_output_vars.insert(format!("first.{}", by_var));
        excluded_output_vars.insert(format!("FIRST.{}", by_var));
        excluded_output_vars.insert(format!("last.{}", by_var));
        excluded_output_vars.insert(format!("LAST.{}", by_var));
    }

    let output_targets_any = payload
        .get_item("output_targets")?
        .ok_or_else(|| PyValueError::new_err("payload.output_targets is required"))?;
    let output_targets = output_targets_any.cast::<PyList>()?;

    let targets = if output_targets_from_stmt.is_empty() {
        output_targets
            .iter()
            .map(|item| item.extract::<String>())
            .collect::<PyResult<Vec<String>>>()?
    } else {
        output_targets_from_stmt
    };

    let prepared_merge_mode = payload
        .get_item("prepared_merge_mode")?
        .and_then(|value| value.extract::<bool>().ok())
        .unwrap_or(false);

    let execution_plan = payload.get_item("execution_plan")?;
    let mut planned_source_slot_order: Vec<String> = Vec::new();
    if !prepared_merge_mode {
        if let Some(plan_any) = execution_plan.as_ref() {
        if let Some(row_loop_plan_any) = get_plan_item(plan_any, "row_loop_plan")
            .map_err(PyValueError::new_err)?
        {
            planned_source_slot_order =
                get_plan_text_list(&row_loop_plan_any, "source_slot_order").map_err(PyValueError::new_err)?;
            if let Some(reason_code) = get_plan_text(&row_loop_plan_any, "unsupported_reason")
                .map_err(PyValueError::new_err)?
            {
                if !reason_code.trim().is_empty() {
                    diagnostics.append(diag(
                        py,
                        &reason_code,
                        "row_loop_plan",
                        "Execution plan indicates row-loop unsupported condition for native runtime.",
                    )?)?;
                    result.set_item("outputs", PyDict::new(py))?;
                    result.set_item("diagnostics", diagnostics)?;
                    return Ok(result.unbind());
                }
            }

            let mode = get_plan_text(&row_loop_plan_any, "mode").map_err(PyValueError::new_err)?;
            let engine_mode = get_plan_text(&row_loop_plan_any, "engine_mode").map_err(PyValueError::new_err)?;
            let backend_mode = get_plan_text(&row_loop_plan_any, "backend_mode").map_err(PyValueError::new_err)?;
            let where_mode = get_plan_text(&row_loop_plan_any, "where_mode").map_err(PyValueError::new_err)?;
            let rewrite_mode = get_plan_text(&row_loop_plan_any, "rewrite_mode").map_err(PyValueError::new_err)?;
            let cursor_kind = get_plan_text(&row_loop_plan_any, "cursor_kind").map_err(PyValueError::new_err)?;
            let builder_mode = get_plan_text(&row_loop_plan_any, "builder_mode").map_err(PyValueError::new_err)?;
            let output_mode = get_plan_text(&row_loop_plan_any, "output_mode").map_err(PyValueError::new_err)?;
            let materialization_policy =
                get_plan_text(&row_loop_plan_any, "materialization_policy").map_err(PyValueError::new_err)?;

            if mode.as_deref() != Some("arrow_row_loop")
                || engine_mode.as_deref() != Some("unified_row_loop")
                || backend_mode.as_deref() != Some("rust_first")
                || !matches!(where_mode.as_deref(), Some("none") | Some("shared_pre_row_filter"))
                || !matches!(rewrite_mode.as_deref(), Some("planner_owned_pending") | Some("planner_owned_active"))
                || cursor_kind.as_deref() != Some("arrow_row_cursor")
                || builder_mode.as_deref() != Some("targeted_output_handoff")
                || output_mode.as_deref() != Some("targeted_output_handoff")
                || materialization_policy.as_deref() != Some("arrow_cursor")
            {
                diagnostics.append(diag(
                    py,
                    "ROW_LOOP_UNSUPPORTED_PLAN",
                    "row_loop_plan",
                    "Native runtime requires the unified arrow_row_loop/targeted_output_handoff contract.",
                )?)?;
                result.set_item("outputs", PyDict::new(py))?;
                result.set_item("diagnostics", diagnostics)?;
                return Ok(result.unbind());
            }
        }
        }
    }

    let input_streams: Option<Py<PyDict>> = if let Some(input_streams_any) = payload.get_item("input_streams")? {
        Some(
            input_streams_any
                .cast::<PyDict>()
                .map_err(|_| PyValueError::new_err("payload.input_streams must be a dict"))?
                .clone()
                .unbind(),
        )
    } else {
        None
    };

    let legacy_inputs: Option<Py<PyDict>> = if let Some(inputs_any) = payload.get_item("inputs")? {
        Some(inputs_any.cast::<PyDict>()?.clone().unbind())
    } else {
        None
    };
    let apply_registry: HashMap<String, Py<PyAny>> = if let Some(apply_registry_any) = payload.get_item("apply_registry")? {
        let apply_registry_dict = apply_registry_any
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err("payload.apply_registry must be a dict"))?;
        let mut resolved = HashMap::new();
        for (key_any, value_any) in apply_registry_dict.iter() {
            let key = key_any.extract::<String>()?;
            resolved.insert(key, value_any.unbind());
        }
        resolved
    } else {
        HashMap::new()
    };
    let (numeric_format_catalogs, character_format_catalogs, informat_catalogs) =
        if let Some(format_catalog_payload_any) = payload.get_item("format_catalog_payload")? {
            let format_catalog_payload = format_catalog_payload_any
                .cast::<PyDict>()
                .map_err(|_| PyValueError::new_err("payload.format_catalog_payload must be a dict"))?;
            let informats_any = format_catalog_payload.get_item("informats")?;
            let numeric_formats_any = if let Some(formats_any) = format_catalog_payload.get_item("formats")? {
                let formats_dict = formats_any
                    .cast::<PyDict>()
                    .map_err(|_| PyValueError::new_err("payload.format_catalog_payload.formats must be a dict"))?;
                formats_dict.get_item("numeric")?
            } else {
                None
            };
            let character_formats_any = if let Some(formats_any) = format_catalog_payload.get_item("formats")? {
                let formats_dict = formats_any
                    .cast::<PyDict>()
                    .map_err(|_| PyValueError::new_err("payload.format_catalog_payload.formats must be a dict"))?;
                formats_dict.get_item("character")?
            } else {
                None
            };
            (
                parse_text_catalog_group(numeric_formats_any.as_ref())?,
                parse_text_catalog_group(character_formats_any.as_ref())?,
                parse_numeric_catalog_group(informats_any.as_ref())?,
            )
        } else {
            (HashMap::new(), HashMap::new(), HashMap::new())
        };
    let return_outputs = payload
        .get_item("return_outputs")?
        .and_then(|value| value.extract::<bool>().ok())
        .unwrap_or(legacy_inputs.is_some());

    let mut output_builders = match initialize_output_builders(execution_plan.as_ref(), &targets) {
        Ok(value) => value,
        Err(error) => {
            diagnostics.append(diag(
                py,
                "ROW_LOOP_UNSUPPORTED_PLAN",
                "output_handoff_plan",
                &format!("Rust output handoff plan resolution failed: {}", error),
            )?)?;
            result.set_item("outputs", PyDict::new(py))?;
            result.set_item("diagnostics", diagnostics)?;
            return Ok(result.unbind());
        }
    };

    let legacy_outputs = if return_outputs {
        let output_rows = PyDict::new(py);
        for target in &targets {
            output_rows.set_item(target, PyList::empty(py))?;
        }
        Some(output_rows)
    } else {
        None
    };

    let has_explicit_output_statement = executable.iter().any(has_inline_output_action);
    let default_target = targets
        .first()
        .cloned()
        .unwrap_or_else(|| "out".to_string());

    let uses_planned_output_handoff = all_output_builders_use_planned_columns(&output_builders);
    let has_output_dataset_options = output_options_by_target
        .values()
        .any(|option| !dataset_ref_options_are_passthrough(option));

    let mut eval_state = EvalRuntimeState {
        apply_registry,
        numeric_format_catalogs,
        character_format_catalogs,
        informat_catalogs,
        ..Default::default()
    };
    let mut global_row_index: i64 = 0;
    let mut stop_execution = false;

    let load_rows_for_input = |input_name: &str| -> Result<Option<Rc<NativeArrowRowCursor>>, String> {
        if let Some(streams) = input_streams.as_ref() {
            let streams = streams.bind(py);
            if let Some(stream_any) = streams
                .get_item(input_name)
                .map_err(|error| format!("input_streams lookup failed: {error}"))?
            {
                return NativeArrowRowCursor::from_arrow_stream(py, &stream_any)
                    .map(Rc::new)
                    .map(Some);
            }
            if let Some(inputs) = legacy_inputs.as_ref() {
                let inputs = inputs.bind(py);
                let rows_any = match inputs
                    .get_item(input_name)
                    .map_err(|error| format!("legacy_inputs lookup failed: {error}"))?
                {
                    Some(value) => value,
                    None => return Ok(None),
                };
                let rows = rows_any
                    .cast::<PyList>()
                    .map_err(|error| format!("legacy input is not a row list: {error}"))?;
                return NativeArrowRowCursor::from_row_list(py, &rows)
                    .map(Rc::new)
                    .map(Some);
            }
            return Ok(None);
        }

        if let Some(inputs) = legacy_inputs.as_ref() {
            let inputs = inputs.bind(py);
            let rows_any = match inputs
                .get_item(input_name)
                .map_err(|error| format!("inputs lookup failed: {error}"))?
            {
                Some(value) => value,
                None => return Ok(None),
            };
            let rows = rows_any
                .cast::<PyList>()
                .map_err(|error| format!("input is not a row list: {error}"))?;
            return NativeArrowRowCursor::from_row_list(py, &rows)
                .map(Rc::new)
                .map(Some);
        }

        Err("Rust native runtime requires input_streams for execution.".to_string())
    };

    let mut loaded_set_inputs: Vec<(String, Rc<NativeArrowRowCursor>)> = Vec::new();
    for input_name in &set_input_names {
        let row_cursor = match load_rows_for_input(input_name) {
            Ok(Some(rows)) => rows,
            Ok(None) => continue,
            Err(error) => {
                let code = if error == "Rust native runtime requires input_streams for execution." {
                    "RUNTIME_RUST_BRIDGE_ARROW_EXPORT_MISSING"
                } else {
                    "RUNTIME_RUST_BRIDGE_ARROW_EXPORT_FAILED"
                };
                diagnostics.append(diag(
                    py,
                    code,
                    &format!("dataset:{}", input_name),
                    &format!("Rust native runtime failed to read input rows: {}", error),
                )?)?;
                result.set_item("outputs", PyDict::new(py))?;
                result.set_item("diagnostics", diagnostics)?;
                return Ok(result.unbind());
            }
        };
        if let Err(error) = row_cursor.validate_source_slot_order(&planned_source_slot_order) {
            diagnostics.append(diag(
                py,
                "ROW_LOOP_UNSUPPORTED_PLAN",
                &format!("dataset:{}", input_name),
                &format!("Rust row cursor failed source slot order validation: {}", error),
            )?)?;
            result.set_item("outputs", PyDict::new(py))?;
            result.set_item("diagnostics", diagnostics)?;
            return Ok(result.unbind());
        }
        loaded_set_inputs.push((input_name.clone(), row_cursor));
    }

    let set_options_are_passthrough = loaded_set_inputs.iter().all(|(input_name, _)| {
        set_options_by_dataset
            .get(&input_name.to_lowercase())
            .map(dataset_ref_options_are_passthrough)
            .unwrap_or(true)
    });
    let can_use_sparse_working_row =
        set_options_are_passthrough && !has_output_dataset_options && legacy_outputs.is_none() && uses_planned_output_handoff;

    let input_import_ms = input_import_start.elapsed().as_secs_f64() * 1000.0;
    let row_execution_start = Instant::now();

    let total_surviving_set_rows = if end_var.is_some() && can_use_sparse_working_row {
        Some(loaded_set_inputs.iter().map(|(_, row_cursor)| row_cursor.row_count()).sum())
    } else if end_var.is_some() {
        let mut count_state = EvalRuntimeState::default();
        let mut total: usize = 0;
        for (input_name, row_cursor) in &loaded_set_inputs {
            let set_options = set_options_by_dataset
                .get(&input_name.to_lowercase())
                .cloned()
                .unwrap_or_default();
            let option_scope_key = format!("input:{}", input_name.to_lowercase());
            for row_index in 0..row_cursor.row_count() {
                let row_obj = row_cursor
                    .row_dict_at(py, row_index)
                    .map_err(PyValueError::new_err)?;
                let row = row_obj.bind(py);
                match apply_dataset_ref_options_to_row(py, &row, &set_options, &mut count_state, &option_scope_key) {
                    Ok(Some(_)) => total += 1,
                    Ok(None) => {}
                    Err(error) => {
                        diagnostics.append(diag(
                            py,
                            "RUNTIME_DATASET_OPTION_INVALID",
                            &format!("dataset:{}", input_name),
                            &dataset_option_error_message(&option_scope_key, &error),
                        )?)?;
                        result.set_item("outputs", PyDict::new(py))?;
                        result.set_item("diagnostics", diagnostics)?;
                        return Ok(result.unbind());
                    }
                }
            }
        }
        Some(total)
    } else {
        None
    };

    for (input_name, row_cursor) in &loaded_set_inputs {
        let row_count = row_cursor.row_count();
        eval_state.source_cursor = Some(Rc::clone(row_cursor));
        eval_state.materialize_mutable_values = !can_use_sparse_working_row;

        for row_index in 0..row_count {
            eval_state.row_index = row_index as isize;
            let set_options = set_options_by_dataset
                .get(&input_name.to_lowercase())
                .cloned()
                .unwrap_or_default();
            let option_scope_key = format!("input:{}", input_name.to_lowercase());
            let working_row_obj = if can_use_sparse_working_row {
                PyDict::new(py).unbind()
            } else {
                let row_obj = row_cursor
                    .row_dict_at(py, row_index)
                    .map_err(PyValueError::new_err)?;
                let row = row_obj.bind(py);
                let projected_row = match apply_dataset_ref_options_to_row(
                    py,
                    row,
                    &set_options,
                    &mut eval_state,
                    &option_scope_key,
                ) {
                    Ok(Some(projected_row)) => projected_row,
                    Ok(None) => continue,
                    Err(error) => {
                        diagnostics.append(diag(
                            py,
                            "RUNTIME_DATASET_OPTION_INVALID",
                            &format!("dataset:{}", input_name),
                            &dataset_option_error_message(&option_scope_key, &error),
                        )?)?;
                        result.set_item("outputs", PyDict::new(py))?;
                        result.set_item("diagnostics", diagnostics)?;
                        return Ok(result.unbind());
                    }
                };
                projected_row
            };
            let working_row = working_row_obj.bind(py);

            global_row_index += 1;
            reset_automatic_scalars(&mut eval_state, global_row_index);

            for in_var in &all_in_vars {
                set_runtime_flag(&mut eval_state, in_var, false);
            }
            if let Some(in_var_name) = set_in_var_by_dataset.get(&input_name.to_lowercase()) {
                set_runtime_flag(&mut eval_state, in_var_name, true);
            }
            if let Some(name) = &indsname_var {
                set_runtime_text(&mut eval_state, name, input_name.clone());
            }
            if let Some(name) = &end_var {
                let is_last = total_surviving_set_rows
                    .map(|total| total > 0 && global_row_index as usize == total)
                    .unwrap_or(false);
                set_runtime_flag(&mut eval_state, name, is_last);
            }

            for retained_name in &retain_vars {
                if get_stateful_row_item(&working_row, retained_name, &eval_state)
                    .map_err(PyValueError::new_err)?
                    .is_none()
                {
                    if let Some(value) = eval_state.retain_values.get(retained_name).cloned() {
                        store_runtime_scalar(&working_row, &mut eval_state, retained_name, value.as_scalar()).ok();
                    }
                }
            }

            if !by_vars.is_empty() {
                let prev_index = if row_index > 0 { Some(row_index - 1) } else { None };
                let next_index = if row_index + 1 < row_count { Some(row_index + 1) } else { None };
                for by_var in &by_vars {
                    let current_value = row_cursor
                        .source_value_by_name(row_index, by_var)
                        .map_err(PyValueError::new_err)?;
                    let is_first = if let Some(prev_idx) = prev_index {
                        let prev_value = row_cursor
                            .source_value_by_name(prev_idx, by_var)
                            .map_err(PyValueError::new_err)?;
                        !optional_source_values_equal(current_value, prev_value)
                            .map_err(PyValueError::new_err)?
                    } else {
                        true
                    };
                    let is_last = if let Some(next_idx) = next_index {
                        let next_value = row_cursor
                            .source_value_by_name(next_idx, by_var)
                            .map_err(PyValueError::new_err)?;
                        !optional_source_values_equal(current_value, next_value)
                            .map_err(PyValueError::new_err)?
                    } else {
                        true
                    };
                    set_runtime_flag(&mut eval_state, &format!("first.{}", by_var), is_first);
                    set_runtime_flag(&mut eval_state, &format!("FIRST.{}", by_var), is_first);
                    set_runtime_flag(&mut eval_state, &format!("last.{}", by_var), is_last);
                    set_runtime_flag(&mut eval_state, &format!("LAST.{}", by_var), is_last);
                }
            }

            if let Some(expression) = &where_expr {
                let pass = match expressions::evaluate_simple_where(&working_row, expression, &mut eval_state) {
                    Ok(value) => value,
                    Err(error) => {
                        mark_automatic_error(&mut eval_state);
                        let (code, message) = if is_expression_parse_error(&error) {
                            let syntax_error = condition_syntax_error_message(py, expression, "<limulus-condition>")
                                .map_err(PyValueError::new_err)?
                                .unwrap_or(error.clone());
                            (
                                "RUNTIME_OPERATOR_NOT_SUPPORTED",
                                format!("Unsupported operator in expression '{expression}': {syntax_error}"),
                            )
                        } else {
                            (
                                "RUNTIME_EXPRESSION_EVALUATION_ERROR",
                                format!("Rust WHERE evaluation failed: {error}"),
                            )
                        };
                        diagnostics.append(diag(
                            py,
                            code,
                            "where",
                            &message,
                        )?)?;
                        result.set_item("outputs", PyDict::new(py))?;
                        result.set_item("diagnostics", diagnostics)?;
                        return Ok(result.unbind());
                    }
                };
                if !pass {
                    continue;
                }
            }

            let mut output_router = |
                source_row: &Bound<'_, PyDict>,
                target_opt: Option<&str>,
                state: &mut EvalRuntimeState,
            | -> Result<(), String> {
                let requested_target = target_opt.unwrap_or(default_target.as_str());
                let resolved_target = resolve_declared_target_name(requested_target, &targets);
                if !targets.contains(&resolved_target) {
                    return Err(format!("Output target is not declared: {requested_target}"));
                }

                if uses_planned_output_handoff && legacy_outputs.is_none() {
                    append_projected_row_to_target(
                        &mut output_builders,
                        &resolved_target,
                        source_row,
                        state,
                        &keep_vars,
                        &drop_vars,
                        &rename_map,
                    )?;
                    return Ok(());
                }

                let mut row_to_write = source_row.clone().unbind();
                if let Some(option_spec) = output_options_by_target.get(&resolved_target.to_lowercase()) {
                    let emitted_bound = row_to_write.bind(py);
                    let option_scope_key = format!("output:{}", resolved_target.to_lowercase());
                    let projected = apply_dataset_ref_options_to_row(
                        py,
                        &emitted_bound,
                        option_spec,
                        state,
                        &option_scope_key,
                    )?;
                    let Some(projected_row) = projected else {
                        return Ok(());
                    };
                    row_to_write = projected_row;
                }

                let emitted_bound = row_to_write.bind(py);
                for excluded_name in &excluded_output_vars {
                    emitted_bound.del_item(excluded_name).ok();
                }
                if keep_vars.is_empty() {
                    emitted_bound.del_item("_N_").ok();
                    emitted_bound.del_item("_ERROR_").ok();
                    emitted_bound.del_item("_n_").ok();
                    emitted_bound.del_item("_error_").ok();
                }
                append_projected_row_to_target(
                    &mut output_builders,
                    &resolved_target,
                    &emitted_bound,
                    state,
                    &keep_vars,
                    &drop_vars,
                    &rename_map,
                )?;

                if let Some(output_rows) = legacy_outputs.as_ref() {
                    let target_rows_any = output_rows
                        .get_item(&resolved_target)
                        .map_err(|error| format!("legacy output lookup failed: {error}"))?
                        .ok_or_else(|| format!("legacy output target is not declared: {resolved_target}"))?;
                    let target_rows = target_rows_any
                        .cast::<PyList>()
                        .map_err(|_| format!("legacy output target is not a list: {resolved_target}"))?;
                    target_rows
                        .append(&emitted_bound)
                        .map_err(|error| format!("legacy output append failed: {error}"))?;
                }

                Ok(())
            };

            let (_, outcome) = match execute_statement_block(
                py,
                &executable,
                0,
                executable.len(),
                &working_row,
                &mut eval_state,
                &mut output_router,
            ) {
                Ok(value) => value,
                Err(error) => {
                    mark_automatic_error(&mut eval_state);
                    diagnostics.append(diag(
                        py,
                        "RUNTIME_EXPRESSION_EVALUATION_ERROR",
                        "statement",
                        &format!("Rust statement execution failed: {}", error),
                    )?)?;
                    result.set_item("outputs", PyDict::new(py))?;
                    result.set_item("diagnostics", diagnostics)?;
                    return Ok(result.unbind());
                }
            };

            for retained_name in &retain_vars {
                if let Some(value_any) = get_stateful_row_item(&working_row, retained_name, &eval_state)
                    .map_err(PyValueError::new_err)?
                {
                    if let Ok(value) = py_to_scalar(&value_any) {
                        eval_state
                            .retain_values
                            .insert(retained_name.clone(), OwnedValue::from(value));
                    }
                }
            }

            if !outcome.deleted && !has_explicit_output_statement {
                if let Err(error) = output_router(&working_row, Some(default_target.as_str()), &mut eval_state) {
                    diagnostics.append(diag(
                        py,
                        "RUNTIME_EXPRESSION_EVALUATION_ERROR",
                        "output",
                        &format!("Rust output projection failed: {}", error),
                    )?)?;
                    result.set_item("outputs", PyDict::new(py))?;
                    result.set_item("diagnostics", diagnostics)?;
                    return Ok(result.unbind());
                }
            }

            if outcome.stopped {
                stop_execution = true;
                break;
            }
        }

        if stop_execution {
            break;
        }
    }

    let row_execution_ms = row_execution_start.elapsed().as_secs_f64() * 1000.0;

    let output_export_start = Instant::now();
    let output_streams = match export_output_streams(py, &output_builders) {
        Ok(value) => value,
        Err(error) => {
            diagnostics.append(diag(
                py,
                "RUNTIME_RUST_NATIVE_EXECUTION_FAILED",
                "output_streams",
                &format!("Rust native runtime failed to export output streams: {}", error),
            )?)?;
            result.set_item("outputs", PyDict::new(py))?;
            result.set_item("diagnostics", diagnostics)?;
            return Ok(result.unbind());
        }
    };
    let output_export_ms = output_export_start.elapsed().as_secs_f64() * 1000.0;

    if return_outputs {
        if let Some(output_rows) = legacy_outputs {
            result.set_item("outputs", output_rows)?;
        }
    }
    let phase_metrics = PyDict::new(py);
    phase_metrics.set_item("input_import_ms", input_import_ms)?;
    phase_metrics.set_item("row_execution_ms", row_execution_ms)?;
    phase_metrics.set_item("output_export_ms", output_export_ms)?;
    result.set_item("phase_metrics", phase_metrics)?;
    result.set_item("output_streams", output_streams)?;
    result.set_item("diagnostics", diagnostics)?;
    Ok(result.unbind())
}
