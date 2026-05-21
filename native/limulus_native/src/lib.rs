use ariadne::{Label, Report, ReportKind, Source};
use chrono::{Datelike, NaiveDate, NaiveDateTime, NaiveTime, Timelike};
use std::collections::{HashMap, HashSet};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use regex::{Regex, RegexBuilder};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct AstPayload {
    statements: Vec<AstStatement>,
}

#[derive(Debug, Clone, Deserialize)]
struct AstStatement {
    kind: String,
    text: String,
    #[serde(default)]
    dataset_refs: Vec<AstDatasetRef>,
    #[serde(default)]
    statement_options: AstStatementOptions,
    #[serde(default)]
    rename_map: HashMap<String, String>,
    #[serde(default)]
    if_spec: Option<AstIfSpec>,
    #[serde(default)]
    do_spec: Option<AstDoSpec>,
    #[serde(default)]
    array_spec: Option<AstArraySpec>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct AstIfSpec {
    condition: String,
    then_action: Option<String>,
    is_subset: bool,
    is_then_do: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct AstDoSpec {
    loop_var: String,
    start_expr: String,
    end_expr: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct AstArraySpec {
    array_name: String,
    #[serde(default)]
    variables: Vec<String>,
    declared_size: Option<usize>,
    wildcard_size: bool,
    character_array: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct AstDatasetRef {
    name: String,
    #[serde(default)]
    options: AstDatasetRefOptions,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct AstDatasetRefOptions {
    in_var: Option<String>,
    #[serde(default)]
    keep_vars: Vec<String>,
    #[serde(default)]
    drop_vars: Vec<String>,
    where_expr: Option<String>,
    #[serde(default)]
    rename_map: HashMap<String, String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct AstStatementOptions {
    indsname_var: Option<String>,
    end_var: Option<String>,
}

#[derive(Debug, Clone)]
struct RenderPayload {
    source_id: String,
    source_text: Option<String>,
    diagnostics: Vec<RenderDiagnostic>,
}

#[derive(Debug, Clone)]
struct RenderDiagnostic {
    code: String,
    severity: String,
    message: String,
    location: String,
    stage: String,
    span: Option<RenderSpan>,
    labels: Vec<RenderLabel>,
    notes: Vec<String>,
    source_text: Option<String>,
}

#[derive(Debug, Clone)]
struct RenderLabel {
    span: Option<RenderSpan>,
    message: String,
    kind: String,
}

#[derive(Debug, Clone)]
struct RenderSpan {
    start: usize,
    end: usize,
    line: usize,
    column: usize,
    end_line: Option<usize>,
    end_column: Option<usize>,
    source_id: String,
}

#[derive(Debug, Clone)]
enum Expr {
    Literal(ScalarValue),
    Variable(String),
    ArrayRef {
        name: String,
        index: Box<Expr>,
    },
    UnaryNot(Box<Expr>),
    UnaryNeg(Box<Expr>),
    Binary {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
    },
    FunctionCall {
        name: String,
        args: Vec<Expr>,
    },
}

#[derive(Debug, Clone)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Mod,
    And,
    Or,
    Ge,
    Le,
    Ne,
    Eq,
    Gt,
    Lt,
}

#[derive(Debug, Clone)]
enum ScalarValue {
    Number(f64),
    Text(String),
    Bool(bool),
    Date(NaiveDate),
    DateTime(NaiveDateTime),
    Time(NaiveTime),
    Null,
}

#[derive(Debug, Clone)]
enum Token {
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Op(String),
    Number(f64),
    Text(String),
    Identifier(String),
}

struct TokenCursor {
    tokens: Vec<Token>,
    position: usize,
}

impl TokenCursor {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, position: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.position)
    }

    fn next(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.position).cloned();
        if token.is_some() {
            self.position += 1;
        }
        token
    }

    fn expect_rparen(&mut self) -> Result<(), String> {
        match self.next() {
            Some(Token::RParen) => Ok(()),
            _ => Err("expected ')'".to_string()),
        }
    }
}

#[derive(Default)]
struct EvalRuntimeState {
    lag_queues: HashMap<usize, Vec<ScalarValue>>,
    sum_totals: HashMap<String, f64>,
    retain_values: HashMap<String, ScalarValue>,
    array_defs: HashMap<String, Vec<String>>,
    parsed_expr_cache: HashMap<String, Expr>,
    parsed_assignment_cache: HashMap<String, Option<(String, String)>>,
    parsed_sum_cache: HashMap<String, Option<(String, String)>>,
    row_index: isize,
    row_view: Vec<Py<PyDict>>,
}

#[derive(Default)]
struct RowExecutionOutcome {
    deleted: bool,
    stopped: bool,
    emitted_rows: Vec<(Option<String>, Py<PyDict>)>,
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

fn column_key(name: &str) -> String {
    name.trim().to_uppercase()
}

fn resolve_row_key(row: &Bound<'_, PyDict>, name: &str) -> Result<Option<String>, String> {
    if row
        .get_item(name)
        .map_err(|error| format!("row lookup failed: {error}"))?
        .is_some()
    {
        return Ok(Some(name.to_string()));
    }

    let normalized = column_key(name);
    for (key_any, _) in row.iter() {
        let candidate = key_any
            .extract::<String>()
            .map_err(|error| format!("row key extract failed: {error}"))?;
        if column_key(&candidate) == normalized {
            return Ok(Some(candidate));
        }
    }

    Ok(None)
}

fn get_row_item<'py>(row: &Bound<'py, PyDict>, name: &str) -> Result<Option<Bound<'py, PyAny>>, String> {
    if let Some(value) = row
        .get_item(name)
        .map_err(|error| format!("row get_item failed: {error}"))?
    {
        return Ok(Some(value));
    }

    let Some(resolved_name) = resolve_row_key(row, name)? else {
        return Ok(None);
    };

    row.get_item(&resolved_name)
        .map_err(|error| format!("row get_item failed: {error}"))
}

fn delete_row_item(row: &Bound<'_, PyDict>, name: &str) -> Result<(), String> {
    if let Some(resolved_name) = resolve_row_key(row, name)? {
        row.del_item(&resolved_name)
            .map_err(|error| format!("row del_item failed: {error}"))?;
    }
    Ok(())
}

fn build_case_insensitive_scope<'py>(py: Python<'py>, row: &Bound<'py, PyDict>) -> Result<Bound<'py, PyDict>, String> {
    let locals = PyDict::new(py);
    for (key, value) in row.iter() {
        locals
            .set_item(&key, &value)
            .map_err(|error| format!("case-insensitive scope set failed: {error}"))?;

        let Ok(name) = key.extract::<String>() else {
            continue;
        };

        let normalized = column_key(&name);
        if locals
            .get_item(&normalized)
            .map_err(|error| format!("case-insensitive scope get failed: {error}"))?
            .is_none()
        {
            locals
                .set_item(&normalized, &value)
                .map_err(|error| format!("case-insensitive scope normalized set failed: {error}"))?;
        }

        let lowered = name.to_lowercase();
        if locals
            .get_item(&lowered)
            .map_err(|error| format!("case-insensitive scope get failed: {error}"))?
            .is_none()
        {
            locals
                .set_item(&lowered, &value)
                .map_err(|error| format!("case-insensitive scope lowered set failed: {error}"))?;
        }
    }
    Ok(locals)
}

fn extract_render_payload(payload: &Bound<'_, PyDict>) -> PyResult<RenderPayload> {
    let source_id = py_dict_string(payload, "source_id", "<dsl>")?;
    let source_text = py_dict_optional_string(payload, "source_text")?;
    let diagnostics_any = payload
        .get_item("diagnostics")?
        .ok_or_else(|| PyValueError::new_err("payload.diagnostics is required"))?;
    let diagnostics_list = diagnostics_any.downcast::<PyList>()?;
    let mut diagnostics: Vec<RenderDiagnostic> = Vec::new();
    for item in diagnostics_list.iter() {
        let diagnostic_dict = item.downcast::<PyDict>()?;
        diagnostics.push(extract_render_diagnostic(&diagnostic_dict)?);
    }
    Ok(RenderPayload {
        source_id,
        source_text,
        diagnostics,
    })
}

fn extract_render_diagnostic(payload: &Bound<'_, PyDict>) -> PyResult<RenderDiagnostic> {
    let labels_any = payload.get_item("labels")?;
    let mut labels: Vec<RenderLabel> = Vec::new();
    if let Some(labels_value) = labels_any {
        let labels_list = labels_value.downcast::<PyList>()?;
        for item in labels_list.iter() {
            let label_dict = item.downcast::<PyDict>()?;
            labels.push(extract_render_label(&label_dict)?);
        }
    }

    let notes_any = payload.get_item("notes")?;
    let mut notes: Vec<String> = Vec::new();
    if let Some(notes_value) = notes_any {
        let notes_list = notes_value.downcast::<PyList>()?;
        for item in notes_list.iter() {
            notes.push(item.extract::<String>()?);
        }
    }

    Ok(RenderDiagnostic {
        code: py_dict_string(payload, "code", "UNKNOWN")?,
        severity: py_dict_string(payload, "severity", "error")?,
        message: py_dict_string(payload, "message", "")?,
        location: py_dict_string(payload, "location", "")?,
        stage: py_dict_string(payload, "stage", "")?,
        span: extract_optional_render_span(payload.get_item("span")?)?,
        labels,
        notes,
        source_text: py_dict_optional_string(payload, "source_text")?,
    })
}

fn extract_render_label(payload: &Bound<'_, PyDict>) -> PyResult<RenderLabel> {
    Ok(RenderLabel {
        span: extract_optional_render_span(payload.get_item("span")?)?,
        message: py_dict_string(payload, "message", "")?,
        kind: py_dict_string(payload, "kind", "primary")?,
    })
}

fn extract_optional_render_span(value: Option<Bound<'_, PyAny>>) -> PyResult<Option<RenderSpan>> {
    let Some(span_value) = value else {
        return Ok(None);
    };
    if span_value.is_none() {
        return Ok(None);
    }
    let span_dict = span_value.downcast::<PyDict>()?;
    Ok(Some(RenderSpan {
        start: py_dict_usize(&span_dict, "start", 0)?,
        end: py_dict_usize(&span_dict, "end", 1)?,
        line: py_dict_usize(&span_dict, "line", 1)?,
        column: py_dict_usize(&span_dict, "column", 1)?,
        end_line: py_dict_optional_usize(&span_dict, "end_line")?,
        end_column: py_dict_optional_usize(&span_dict, "end_column")?,
        source_id: py_dict_string(&span_dict, "source_id", "<dsl>")?,
    }))
}

fn py_dict_string(payload: &Bound<'_, PyDict>, key: &str, default: &str) -> PyResult<String> {
    let Some(value) = payload.get_item(key)? else {
        return Ok(default.to_string());
    };
    if value.is_none() {
        return Ok(default.to_string());
    }
    value.extract::<String>()
}

fn py_dict_optional_string(payload: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    let Some(value) = payload.get_item(key)? else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    Ok(Some(value.extract::<String>()?))
}

fn py_dict_usize(payload: &Bound<'_, PyDict>, key: &str, default: usize) -> PyResult<usize> {
    let Some(value) = payload.get_item(key)? else {
        return Ok(default);
    };
    if value.is_none() {
        return Ok(default);
    }
    value.extract::<usize>()
}

fn py_dict_optional_usize(payload: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<usize>> {
    let Some(value) = payload.get_item(key)? else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    Ok(Some(value.extract::<usize>()?))
}

fn normalize_render_range(span: &RenderSpan, source_text: &str) -> std::ops::Range<usize> {
    let start = span.start.min(source_text.len());
    let end = span.end.max(start + 1).min(source_text.len().max(start + 1));
    start..end
}

fn render_report_kind(severity: &str) -> ReportKind<'static> {
    match severity.to_lowercase().as_str() {
        "warning" => ReportKind::Warning,
        "info" | "note" => ReportKind::Advice,
        _ => ReportKind::Error,
    }
}

fn render_header_message(diagnostic: &RenderDiagnostic) -> String {
    if diagnostic.stage.is_empty() {
        diagnostic.message.clone()
    } else {
        format!("[stage: {}] {}", diagnostic.stage, diagnostic.message)
    }
}

fn render_simple_diagnostic(diagnostic: &RenderDiagnostic) -> String {
    let mut header = format!("{}[{}]", diagnostic.severity.to_lowercase(), diagnostic.code);
    if !diagnostic.stage.is_empty() {
        header.push_str(&format!(" [stage: {}]", diagnostic.stage));
    }
    header.push_str(&format!(": {}", diagnostic.message));
    if !diagnostic.location.is_empty() {
        header.push_str(&format!(" ({})", diagnostic.location));
    }
    let mut lines = vec![header];
    for note in &diagnostic.notes {
        lines.push(format!("note: {}", note));
    }
    lines.join("\n")
}

fn render_ariadne_diagnostic(diagnostic: &RenderDiagnostic, payload: &RenderPayload) -> Result<String, String> {
    let source_text = diagnostic
        .source_text
        .as_deref()
        .or(payload.source_text.as_deref());
    let span = diagnostic.span.as_ref();
    let Some(source_text) = source_text else {
        return Ok(render_simple_diagnostic(diagnostic));
    };
    let Some(span) = span else {
        return Ok(render_simple_diagnostic(diagnostic));
    };

    let source_id = if span.source_id.is_empty() {
        payload.source_id.clone()
    } else {
        span.source_id.clone()
    };
    let report_range = normalize_render_range(span, source_text);
    let mut builder = Report::build(
        render_report_kind(&diagnostic.severity),
        (source_id.clone(), report_range.clone()),
    )
        .with_code(diagnostic.code.clone())
        .with_message(render_header_message(diagnostic));

    let mut has_label = false;
    for label in &diagnostic.labels {
        let Some(label_span) = label.span.as_ref() else {
            continue;
        };
        let range = normalize_render_range(label_span, source_text);
        let mut ariadne_label = Label::new((source_id.clone(), range));
        if !label.message.is_empty() {
            ariadne_label = ariadne_label.with_message(label.message.clone());
        }
        builder = builder.with_label(ariadne_label);
        has_label = true;
    }

    if !has_label {
        let range = normalize_render_range(span, source_text);
        builder = builder.with_label(Label::new((source_id.clone(), range)));
    }

    for note in &diagnostic.notes {
        builder = builder.with_note(note.clone());
    }

    let report = builder.finish();
    let mut buffer: Vec<u8> = Vec::new();
    if let Err(error) = report.write((source_id, Source::from(source_text)), &mut buffer) {
        return Err(error.to_string());
    }
    String::from_utf8(buffer).map_err(|error| error.to_string())
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

fn resolve_array_variable_name(
    state: &EvalRuntimeState,
    array_name: &str,
    index_value: f64,
) -> Result<String, String> {
    let vars = state
        .array_defs
        .get(array_name)
        .ok_or_else(|| format!("array is not defined: {array_name}"))?;
    let index = index_value as isize;
    if index < 1 || (index as usize) > vars.len() {
        return Err(format!("array index out of bounds: {array_name}[{index}]") );
    }
    Ok(vars[index as usize - 1].clone())
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

fn set_row_scalar(row: &Bound<'_, PyDict>, name: &str, value: &ScalarValue) -> Result<(), String> {
    let normalized = column_key(name);
    let mut target_name: Option<String> = None;
    let mut duplicate_names: Vec<String> = Vec::new();
    for (key_any, _) in row.iter() {
        let candidate = key_any
            .extract::<String>()
            .map_err(|error| format!("set_item key extract failed: {error}"))?;
        if column_key(&candidate) != normalized {
            continue;
        }
        if target_name.is_none() {
            target_name = Some(candidate.clone());
        } else {
            duplicate_names.push(candidate);
        }
    }
    let target_name = target_name.unwrap_or_else(|| name.to_string());
    for duplicate_name in duplicate_names {
        row.del_item(&duplicate_name).ok();
    }
    match value {
        ScalarValue::Number(number) => row.set_item(&target_name, *number),
        ScalarValue::Text(text) => row.set_item(&target_name, text.clone()),
        ScalarValue::Bool(boolean) => row.set_item(&target_name, *boolean),
        ScalarValue::Date(date) => row.set_item(&target_name, temporal_scalar_to_pyany(row.py(), &ScalarValue::Date(*date))?),
        ScalarValue::DateTime(datetime) => {
            row.set_item(&target_name, temporal_scalar_to_pyany(row.py(), &ScalarValue::DateTime(*datetime))?)
        }
        ScalarValue::Time(time) => row.set_item(&target_name, temporal_scalar_to_pyany(row.py(), &ScalarValue::Time(*time))?),
        ScalarValue::Null => row.set_item(&target_name, py_none(row.py())),
    }
    .map_err(|error| format!("set_item failed: {error}"))
}

fn temporal_scalar_to_pyany(py: Python<'_>, value: &ScalarValue) -> Result<Py<PyAny>, String> {
    let datetime_module = py.import("datetime").map_err(|error| error.to_string())?;
    match value {
        ScalarValue::Date(date) => datetime_module
            .getattr("date")
            .map_err(|error| error.to_string())?
            .call1((date.year(), date.month(), date.day()))
            .map_err(|error| error.to_string())
            .map(|item| item.unbind()),
        ScalarValue::DateTime(datetime) => datetime_module
            .getattr("datetime")
            .map_err(|error| error.to_string())?
            .call1((
                datetime.year(),
                datetime.month(),
                datetime.day(),
                datetime.hour(),
                datetime.minute(),
                datetime.second(),
                datetime.and_utc().timestamp_subsec_micros(),
            ))
            .map_err(|error| error.to_string())
            .map(|item| item.unbind()),
        ScalarValue::Time(time) => datetime_module
            .getattr("time")
            .map_err(|error| error.to_string())?
            .call1((
                time.hour(),
                time.minute(),
                time.second(),
                time.nanosecond() / 1_000,
            ))
            .map_err(|error| error.to_string())
            .map(|item| item.unbind()),
        _ => Err("temporal scalar expected".to_string()),
    }
}

fn py_none(py: Python<'_>) -> Py<PyAny> {
    py.None()
}

fn parse_expression_cached(expr_text: &str, state: &mut EvalRuntimeState) -> Result<Expr, String> {
    if let Some(parsed) = state.parsed_expr_cache.get(expr_text) {
        return Ok(parsed.clone());
    }

    let tokens = tokenize_expression(expr_text)?;
    let mut cursor = TokenCursor::new(tokens);
    let parsed = parse_expression(&mut cursor, 0)?;
    if cursor.peek().is_some() {
        return Err("unexpected token at end of expression".to_string());
    }

    state
        .parsed_expr_cache
        .insert(expr_text.to_string(), parsed.clone());
    Ok(parsed)
}

fn evaluate_scalar_expression(
    row: &Bound<'_, PyDict>,
    expr_text: &str,
    state: &mut EvalRuntimeState,
) -> Result<ScalarValue, String> {
    let native_result = (|| {
        let parsed = parse_expression_cached(expr_text, state)?;
        evaluate_expression(row, &parsed, state)
    })();

    match native_result {
        Ok(value) => Ok(value),
        Err(native_error) => {
            let should_try_python_fallback = expr_text.contains("**")
                || native_error == "expression expected"
                || native_error == "unexpected token at end of expression";
            if !should_try_python_fallback {
                return Err(native_error);
            }

            let py = row.py();
            let globals = PyDict::new(py);
            globals
                .set_item("__builtins__", PyDict::new(py))
                .map_err(|error| format!("python pow fallback globals failed: {error}"))?;

            let locals = build_case_insensitive_scope(py, row)?;

            let builtins = py
                .import("builtins")
                .map_err(|error| format!("python pow fallback import failed: {error}"))?;
            for name in ["round", "abs", "min", "max", "pow"] {
                let function = builtins
                    .getattr(name)
                    .map_err(|error| format!("python pow fallback getattr failed for {name}: {error}"))?;
                locals
                    .set_item(name, function)
                    .map_err(|error| format!("python pow fallback locals set failed for {name}: {error}"))?;
            }

            let evaluated = builtins
                .getattr("eval")
                .map_err(|error| format!("python pow fallback eval lookup failed: {error}"))?
                .call1((expr_text, globals, locals))
                .map_err(|error| format!("{native_error}; python pow fallback failed: {error}"))?;
            py_to_scalar(&evaluated).map_err(|error| format!("python pow fallback conversion failed: {error}"))
        }
    }
}

fn append_projected_row_to_target(
    py: Python<'_>,
    outputs: &Bound<'_, PyDict>,
    target: &str,
    row: &Bound<'_, PyDict>,
    keep_vars: &[String],
    drop_vars: &[String],
) -> Result<(), String> {
    let projected = PyDict::new(py);
    if keep_vars.is_empty() {
        for (key, value) in row.iter() {
            projected
                .set_item(key, value)
                .map_err(|error| format!("projection set_item failed: {error}"))?;
        }
    } else {
        for name in keep_vars {
            if let Some(value) = get_row_item(row, name)? {
                projected
                    .set_item(name, value)
                    .map_err(|error| format!("projection keep set_item failed: {error}"))?;
            }
        }
    }

    if !drop_vars.is_empty() {
        for name in drop_vars {
            delete_row_item(&projected, name).ok();
        }
    }

    let list_any = outputs
        .get_item(target)
        .map_err(|error| format!("output lookup failed: {error}"))?
        .ok_or_else(|| format!("output target is not declared: {target}"))?;
    let list = list_any
        .downcast::<PyList>()
        .map_err(|_| format!("output target is not a list: {target}"))?;
    list.append(&projected)
        .map_err(|error| format!("output append failed: {error}"))?;
    Ok(())
}

fn clone_row_dict(py: Python<'_>, row: &Bound<'_, PyDict>) -> Result<Py<PyDict>, String> {
    let cloned = PyDict::new(py);
    for (key, value) in row.iter() {
        cloned
            .set_item(key, value)
            .map_err(|error| format!("row clone failed: {error}"))?;
    }
    Ok(cloned.unbind())
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

fn execute_inline_action(
    py: Python<'_>,
    action: &str,
    row: &Bound<'_, PyDict>,
    state: &mut EvalRuntimeState,
    outcome: &mut RowExecutionOutcome,
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
            outcome.emitted_rows.push((None, clone_row_dict(py, row)?));
        } else {
            outcome.emitted_rows.push((Some(body), clone_row_dict(py, row)?));
        }
        return Ok(());
    }

    if let Some((name, expr)) = parse_sum_statement_cached(normalized, state) {
        let increment = scalar_to_number(&evaluate_scalar_expression(row, &expr, state)?)?;

        let base = if let Some(existing) = get_row_item(row, &name)? {
            py_to_scalar(&existing).ok()
                .and_then(|s| scalar_to_number(&s).ok())
                .unwrap_or(0.0)
        } else {
            *state.sum_totals.entry(name.clone()).or_insert(0.0)
        };
        let total = base + increment;
        state.sum_totals.insert(name.clone(), total);
        set_row_scalar(row, &name, &ScalarValue::Number(total))?;
        return Ok(());
    }

    if let Some((name, expr)) = parse_assignment_cached(normalized, state) {
        let value = evaluate_scalar_expression(row, &expr, state)?;
        if let Some((array_name, index_expr)) = parse_array_reference_token(&name) {
            let index_value = scalar_to_number(&evaluate_scalar_expression(row, &index_expr, state)?)?;
            let variable_name = resolve_array_variable_name(state, &array_name, index_value)?;
            set_row_scalar(row, &variable_name, &value)?;
        } else {
            set_row_scalar(row, &name, &value)?;
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

                let base = if let Some(existing) = get_row_item(row, &name)? {
                    py_to_scalar(&existing).ok()
                        .and_then(|s| scalar_to_number(&s).ok())
                        .unwrap_or(0.0)
                } else {
                    *state.sum_totals.entry(name.clone()).or_insert(0.0)
                };
                let total = base + increment;
                state.sum_totals.insert(name.clone(), total);
                set_row_scalar(row, &name, &ScalarValue::Number(total))?;
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
                    set_row_scalar(row, &variable_name, &value)?;
                } else {
                    set_row_scalar(row, &name, &value)?;
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
                    outcome.emitted_rows.push((None, clone_row_dict(py, row)?));
                } else {
                    outcome.emitted_rows.push((Some(body), clone_row_dict(py, row)?));
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
                    set_row_scalar(row, &var_name, &ScalarValue::Number(iteration as f64))?;
                    let (_, nested) = execute_statement_block(py, statements, cursor + 1, end_index, row, state)?;
                    if nested.deleted {
                        outcome.deleted = true;
                        return Ok((end_index + 1, outcome));
                    }
                    if nested.stopped {
                        outcome.stopped = true;
                        return Ok((end_index + 1, outcome));
                    }
                    outcome.emitted_rows.extend(nested.emitted_rows);
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
                        let (_, nested) = execute_statement_block(py, statements, cursor + 1, end_index, row, state)?;
                        if nested.deleted {
                            outcome.deleted = true;
                            return Ok((end_index + 1, outcome));
                        }
                        if nested.stopped {
                            outcome.stopped = true;
                            return Ok((end_index + 1, outcome));
                        }
                        outcome.emitted_rows.extend(nested.emitted_rows);
                    }
                    cursor = end_index + 1;
                    if cursor < stop && statements[cursor].kind == "ELSE" {
                        let else_text = statement_body(&statements[cursor].text, "else");
                        if else_text.to_lowercase() == "do" {
                            let else_end_index = find_matching_end(statements, cursor, stop)
                                .ok_or_else(|| "ELSE DO block missing matching END".to_string())?;
                            if !matched {
                                let (_, nested) = execute_statement_block(py, statements, cursor + 1, else_end_index, row, state)?;
                                if nested.deleted {
                                    outcome.deleted = true;
                                    return Ok((else_end_index + 1, outcome));
                                }
                                if nested.stopped {
                                    outcome.stopped = true;
                                    return Ok((else_end_index + 1, outcome));
                                }
                                outcome.emitted_rows.extend(nested.emitted_rows);
                            }
                            cursor = else_end_index + 1;
                            continue;
                        }
                        if !matched && !else_text.is_empty() {
                            execute_inline_action(py, &else_text, row, state, &mut outcome)?;
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
                        execute_inline_action(py, &action, row, state, &mut outcome)?;
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
                                execute_inline_action(py, &else_if_action, row, state, &mut outcome)?;
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
                        execute_inline_action(py, &else_text, row, state, &mut outcome)?;
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

fn tokenize_expression(expression: &str) -> Result<Vec<Token>, String> {
    let chars: Vec<char> = expression.chars().collect();
    let mut index = 0;
    let mut tokens: Vec<Token> = Vec::new();

    while index < chars.len() {
        let ch = chars[index];
        if ch.is_whitespace() {
            index += 1;
            continue;
        }

        if ch == '(' {
            tokens.push(Token::LParen);
            index += 1;
            continue;
        }
        if ch == ')' {
            tokens.push(Token::RParen);
            index += 1;
            continue;
        }
        if ch == '[' {
            tokens.push(Token::LBracket);
            index += 1;
            continue;
        }
        if ch == ']' {
            tokens.push(Token::RBracket);
            index += 1;
            continue;
        }
        if ch == '{' {
            tokens.push(Token::LBrace);
            index += 1;
            continue;
        }
        if ch == '}' {
            tokens.push(Token::RBrace);
            index += 1;
            continue;
        }
        if ch == ',' {
            tokens.push(Token::Comma);
            index += 1;
            continue;
        }

        if ch == '\'' || ch == '"' {
            let quote = ch;
            index += 1;
            let start = index;
            while index < chars.len() && chars[index] != quote {
                index += 1;
            }
            if index >= chars.len() {
                return Err("unterminated string literal".to_string());
            }
            let text: String = chars[start..index].iter().collect();
            tokens.push(Token::Text(text));
            index += 1;
            continue;
        }

        if ch.is_ascii_digit() || (ch == '.' && index + 1 < chars.len() && chars[index + 1].is_ascii_digit()) {
            let start = index;
            index += 1;
            while index < chars.len() && (chars[index].is_ascii_digit() || chars[index] == '.') {
                index += 1;
            }
            let raw: String = chars[start..index].iter().collect();
            let next_non_space = chars[index..].iter().copied().find(|candidate| !candidate.is_whitespace());
            if raw.ends_with('.') && matches!(next_non_space, Some(',') | Some(')')) {
                tokens.push(Token::Text(raw));
                continue;
            }
            if raw.ends_with('.') && raw[..raw.len() - 1].contains('.') {
                tokens.push(Token::Identifier(raw));
                continue;
            }
            let number = raw.parse::<f64>().map_err(|_| format!("invalid number literal: {raw}"))?;
            tokens.push(Token::Number(number));
            continue;
        }

        if ch.is_ascii_alphabetic() || ch == '_' {
            let start = index;
            index += 1;
            while index < chars.len()
                && (chars[index].is_ascii_alphanumeric() || chars[index] == '_' || chars[index] == '.')
            {
                index += 1;
            }
            let ident: String = chars[start..index].iter().collect();
            tokens.push(Token::Identifier(ident));
            continue;
        }

        if index + 1 < chars.len() {
            let pair = format!("{}{}", chars[index], chars[index + 1]);
            if ["**"].contains(&pair.as_str()) {
                tokens.push(Token::Op(pair));
                index += 2;
                continue;
            }
            if [">=", "<=", "!=", "=="].contains(&pair.as_str()) {
                tokens.push(Token::Op(pair));
                index += 2;
                continue;
            }
            if ["^=", "~=", "¬="].contains(&pair.as_str()) {
                tokens.push(Token::Op("!=".to_string()));
                index += 2;
                continue;
            }
        }

        if ['+', '-', '*', '/', '>', '<', '='].contains(&ch) {
            tokens.push(Token::Op(ch.to_string()));
            index += 1;
            continue;
        }

        return Err(format!("unexpected token: {ch}"));
    }

    Ok(tokens)
}

fn parse_expression(cursor: &mut TokenCursor, min_bp: u8) -> Result<Expr, String> {
    let mut lhs = match cursor.next() {
        Some(Token::Number(value)) => Expr::Literal(ScalarValue::Number(value)),
        Some(Token::Text(value)) => Expr::Literal(ScalarValue::Text(value)),
        Some(Token::Identifier(name)) => {
            let lowered = name.to_lowercase();
            if lowered == "true" {
                Expr::Literal(ScalarValue::Bool(true))
            } else if lowered == "false" {
                Expr::Literal(ScalarValue::Bool(false))
            } else if lowered == "none" || lowered == "null" {
                Expr::Literal(ScalarValue::Null)
            } else if lowered == "not" {
                let rhs = parse_expression(cursor, 11)?;
                Expr::UnaryNot(Box::new(rhs))
            } else if matches!(cursor.peek(), Some(Token::LParen)) {
                cursor.next();
                let mut args: Vec<Expr> = Vec::new();
                if !matches!(cursor.peek(), Some(Token::RParen)) {
                    loop {
                        args.push(parse_expression(cursor, 0)?);
                        if matches!(cursor.peek(), Some(Token::Comma)) {
                            cursor.next();
                            continue;
                        }
                        break;
                    }
                }
                cursor.expect_rparen()?;
                Expr::FunctionCall { name, args }
            } else if matches!(cursor.peek(), Some(Token::LBracket)) {
                cursor.next();
                let index = parse_expression(cursor, 0)?;
                match cursor.next() {
                    Some(Token::RBracket) => Expr::ArrayRef {
                        name,
                        index: Box::new(index),
                    },
                    _ => return Err("expected ']'".to_string()),
                }
            } else if matches!(cursor.peek(), Some(Token::LBrace)) {
                cursor.next();
                let index = parse_expression(cursor, 0)?;
                match cursor.next() {
                    Some(Token::RBrace) => Expr::ArrayRef {
                        name,
                        index: Box::new(index),
                    },
                    _ => return Err("expected '}'".to_string()),
                }
            } else {
                Expr::Variable(name)
            }
        }
        Some(Token::Op(op)) if op == "-" => {
            let rhs = parse_expression(cursor, 11)?;
            Expr::UnaryNeg(Box::new(rhs))
        }
        Some(Token::LParen) => {
            let expr = parse_expression(cursor, 0)?;
            cursor.expect_rparen()?;
            expr
        }
        _ => return Err("expression expected".to_string()),
    };

    loop {
        let op_token = match cursor.peek() {
            Some(Token::Op(op)) => Some(op.clone()),
            Some(Token::Identifier(ident)) => {
                let lowered = ident.to_lowercase();
                if ["and", "or", "mod"].contains(&lowered.as_str()) {
                    Some(lowered)
                } else {
                    None
                }
            }
            _ => None,
        };
        let Some(op_text) = op_token else {
            break;
        };

        let (lbp, rbp, op) = match op_text.as_str() {
            "or" => (1, 2, BinaryOp::Or),
            "and" => (3, 4, BinaryOp::And),
            "==" | "=" => (5, 6, BinaryOp::Eq),
            "!=" => (5, 6, BinaryOp::Ne),
            ">" => (5, 6, BinaryOp::Gt),
            ">=" => (5, 6, BinaryOp::Ge),
            "<" => (5, 6, BinaryOp::Lt),
            "<=" => (5, 6, BinaryOp::Le),
            "+" => (7, 8, BinaryOp::Add),
            "-" => (7, 8, BinaryOp::Sub),
            "*" => (9, 10, BinaryOp::Mul),
            "/" => (9, 10, BinaryOp::Div),
            "**" => (11, 11, BinaryOp::Pow),
            "mod" => (9, 10, BinaryOp::Mod),
            _ => break,
        };

        if lbp < min_bp {
            break;
        }

        cursor.next();
        let rhs = parse_expression(cursor, rbp)?;
        lhs = Expr::Binary {
            left: Box::new(lhs),
            op,
            right: Box::new(rhs),
        };
    }

    Ok(lhs)
}

fn py_to_scalar(value: &Bound<'_, PyAny>) -> PyResult<ScalarValue> {
    if value.is_none() {
        return Ok(ScalarValue::Null);
    }
    if let Some(temporal) = py_temporal_to_scalar(value)? {
        return Ok(temporal);
    }
    if let Ok(boolean) = value.extract::<bool>() {
        return Ok(ScalarValue::Bool(boolean));
    }
    if let Ok(number) = value.extract::<f64>() {
        return Ok(ScalarValue::Number(number));
    }
    Ok(ScalarValue::Text(value.str()?.to_string()))
}

fn py_temporal_to_scalar(value: &Bound<'_, PyAny>) -> PyResult<Option<ScalarValue>> {
    let py = value.py();
    let datetime_module = py.import("datetime")?;
    let datetime_type = datetime_module.getattr("datetime")?;
    if value.is_instance(&datetime_type)? {
        let year = value.getattr("year")?.extract::<i32>()?;
        let month = value.getattr("month")?.extract::<u32>()?;
        let day = value.getattr("day")?.extract::<u32>()?;
        let hour = value.getattr("hour")?.extract::<u32>()?;
        let minute = value.getattr("minute")?.extract::<u32>()?;
        let second = value.getattr("second")?.extract::<u32>()?;
        let microsecond = value.getattr("microsecond")?.extract::<u32>()?;
        let date = NaiveDate::from_ymd_opt(year, month, day)
            .ok_or_else(|| PyValueError::new_err("invalid python datetime date component"))?;
        let datetime = date
            .and_hms_micro_opt(hour, minute, second, microsecond)
            .ok_or_else(|| PyValueError::new_err("invalid python datetime time component"))?;
        return Ok(Some(ScalarValue::DateTime(datetime)));
    }

    let date_type = datetime_module.getattr("date")?;
    if value.is_instance(&date_type)? {
        let year = value.getattr("year")?.extract::<i32>()?;
        let month = value.getattr("month")?.extract::<u32>()?;
        let day = value.getattr("day")?.extract::<u32>()?;
        let date = NaiveDate::from_ymd_opt(year, month, day)
            .ok_or_else(|| PyValueError::new_err("invalid python date component"))?;
        return Ok(Some(ScalarValue::Date(date)));
    }

    let time_type = datetime_module.getattr("time")?;
    if value.is_instance(&time_type)? {
        let hour = value.getattr("hour")?.extract::<u32>()?;
        let minute = value.getattr("minute")?.extract::<u32>()?;
        let second = value.getattr("second")?.extract::<u32>()?;
        let microsecond = value.getattr("microsecond")?.extract::<u32>()?;
        let time = NaiveTime::from_hms_micro_opt(hour, minute, second, microsecond)
            .ok_or_else(|| PyValueError::new_err("invalid python time component"))?;
        return Ok(Some(ScalarValue::Time(time)));
    }

    Ok(None)
}

fn scalar_to_number(value: &ScalarValue) -> Result<f64, String> {
    match value {
        ScalarValue::Number(number) => Ok(*number),
        ScalarValue::Bool(boolean) => Ok(if *boolean { 1.0 } else { 0.0 }),
        ScalarValue::Text(text) => text
            .parse::<f64>()
            .map_err(|_| format!("numeric value required, got '{text}'")),
        ScalarValue::Date(_) => Err("date value cannot be converted to number".to_string()),
        ScalarValue::DateTime(_) => Err("datetime value cannot be converted to number".to_string()),
        ScalarValue::Time(_) => Err("time value cannot be converted to number".to_string()),
        ScalarValue::Null => Ok(0.0),
    }
}

fn scalar_to_text(value: &ScalarValue) -> String {
    match value {
        ScalarValue::Text(text) => text.clone(),
        ScalarValue::Number(number) => number.to_string(),
        ScalarValue::Bool(boolean) => {
            if *boolean {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        ScalarValue::Date(date) => date.to_string(),
        ScalarValue::DateTime(datetime) => datetime.format("%Y-%m-%dT%H:%M:%S").to_string(),
        ScalarValue::Time(time) => time.format("%H:%M:%S").to_string(),
        ScalarValue::Null => "".to_string(),
    }
}

fn scalar_to_date(value: &ScalarValue) -> Result<NaiveDate, String> {
    match value {
        ScalarValue::Date(date) => Ok(*date),
        ScalarValue::DateTime(datetime) => Ok(datetime.date()),
        ScalarValue::Text(text) => NaiveDate::parse_from_str(text, "%Y-%m-%d")
            .map_err(|_| format!("date value required, got '{text}'")),
        _ => Err("date value required".to_string()),
    }
}

fn scalar_to_datetime(value: &ScalarValue) -> Result<NaiveDateTime, String> {
    match value {
        ScalarValue::DateTime(datetime) => Ok(*datetime),
        ScalarValue::Date(date) => date
            .and_hms_opt(0, 0, 0)
            .ok_or_else(|| "datetime value required".to_string()),
        ScalarValue::Text(text) => NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S")
            .map_err(|_| format!("datetime value required, got '{text}'")),
        _ => Err("datetime value required".to_string()),
    }
}

fn scalar_to_time(value: &ScalarValue) -> Result<NaiveTime, String> {
    match value {
        ScalarValue::Time(time) => Ok(*time),
        ScalarValue::DateTime(datetime) => Ok(datetime.time()),
        ScalarValue::Text(text) => parse_time_text(text),
        _ => Err("time value required".to_string()),
    }
}

fn parse_time_text(text: &str) -> Result<NaiveTime, String> {
    NaiveTime::parse_from_str(text, "%H:%M:%S")
        .or_else(|_| NaiveTime::parse_from_str(text, "%H:%M"))
        .map_err(|_| format!("time value required, got '{text}'"))
}

fn scalar_is_missing(value: &ScalarValue) -> bool {
    matches!(value, ScalarValue::Null) || matches!(value, ScalarValue::Text(text) if text.is_empty())
}

fn scalar_to_bool(value: &ScalarValue) -> Result<bool, String> {
    match value {
        ScalarValue::Bool(boolean) => Ok(*boolean),
        ScalarValue::Number(number) => Ok(number.abs() >= f64::EPSILON),
        ScalarValue::Text(text) => Ok(!text.is_empty()),
        ScalarValue::Date(_) => Ok(true),
        ScalarValue::DateTime(_) => Ok(true),
        ScalarValue::Time(_) => Ok(true),
        ScalarValue::Null => Ok(false),
    }
}

fn scalar_eq(left: &ScalarValue, right: &ScalarValue) -> Result<bool, String> {
    match (left, right) {
        (ScalarValue::Null, ScalarValue::Null) => Ok(true),
        (ScalarValue::Number(_), _) | (_, ScalarValue::Number(_)) | (ScalarValue::Bool(_), _) | (_, ScalarValue::Bool(_)) => {
            let l = scalar_to_number(left)?;
            let r = scalar_to_number(right)?;
            Ok((l - r).abs() < f64::EPSILON)
        }
        (ScalarValue::Text(l), ScalarValue::Text(r)) => Ok(l == r),
        (ScalarValue::Date(l), ScalarValue::Date(r)) => Ok(l == r),
        (ScalarValue::DateTime(l), ScalarValue::DateTime(r)) => Ok(l == r),
        (ScalarValue::Time(l), ScalarValue::Time(r)) => Ok(l == r),
        _ => Ok(false),
    }
}

fn optional_pyvalues_equal(
    left: Option<&Bound<'_, PyAny>>,
    right: Option<&Bound<'_, PyAny>>,
) -> Result<bool, String> {
    match (left, right) {
        (Some(lhs), Some(rhs)) => {
            let lhs_scalar = py_to_scalar(lhs).map_err(|error| error.to_string())?;
            let rhs_scalar = py_to_scalar(rhs).map_err(|error| error.to_string())?;
            scalar_eq(&lhs_scalar, &rhs_scalar)
        }
        (None, None) => Ok(true),
        _ => Ok(false),
    }
}

fn scalar_compare(left: &ScalarValue, right: &ScalarValue) -> Result<std::cmp::Ordering, String> {
    match (left, right) {
        (ScalarValue::Text(l), ScalarValue::Text(r)) => Ok(l.cmp(r)),
        (ScalarValue::Date(l), ScalarValue::Date(r)) => Ok(l.cmp(r)),
        (ScalarValue::DateTime(l), ScalarValue::DateTime(r)) => Ok(l.cmp(r)),
        (ScalarValue::Time(l), ScalarValue::Time(r)) => Ok(l.cmp(r)),
        _ => {
            let l = scalar_to_number(left)?;
            let r = scalar_to_number(right)?;
            l.partial_cmp(&r)
                .ok_or_else(|| "comparison failed".to_string())
        }
    }
}

fn normalize_format_name(value: &ScalarValue) -> (String, bool) {
    let raw = scalar_to_text(value).trim().to_ascii_lowercase();
    let has_dot = raw.contains('.');
    (raw.trim_end_matches('.').to_string(), has_dot)
}

fn requires_numeric_format_dot(format_name: &str, has_dot: bool) -> bool {
    if has_dot {
        return false;
    }

    parse_width_precision(format_name).is_some()
        || format_name
            .strip_prefix('z')
            .and_then(parse_width_precision)
            .is_some()
        || format_name
            .strip_prefix("comma")
            .and_then(parse_width_precision)
            .is_some()
}

fn parse_width_precision(format_name: &str) -> Option<(usize, Option<usize>)> {
    let (width_part, precision_part) = format_name.split_once('.').unwrap_or((format_name, ""));
    let width = width_part.parse::<usize>().ok()?;
    if precision_part.is_empty() {
        return Some((width, None));
    }
    let precision = precision_part.parse::<usize>().ok()?;
    Some((width, Some(precision)))
}

fn format_fixed_number(number: f64, decimals: usize) -> String {
    format!("{number:.decimals$}")
}

fn format_zero_filled_number(number: f64, width: usize, decimals: usize) -> String {
    let rendered = format_fixed_number(number, decimals);
    let (sign, unsigned) = match rendered.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", rendered.as_str()),
    };
    let padding = width.saturating_sub(sign.len() + unsigned.len());
    format!("{sign}{}{unsigned}", "0".repeat(padding))
}

fn format_grouped_number(number: f64, decimals: usize) -> String {
    let rendered = format_fixed_number(number, decimals);
    let (sign, unsigned) = match rendered.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", rendered.as_str()),
    };
    let (integer_part, fractional_part) = unsigned.split_once('.').unwrap_or((unsigned, ""));
    let grouped_integer = group_digits(integer_part);
    if fractional_part.is_empty() {
        return format!("{sign}{grouped_integer}");
    }
    format!("{sign}{grouped_integer}.{fractional_part}")
}

fn format_best_number(number: f64) -> String {
    if number.fract().abs() < f64::EPSILON {
        return format!("{:.0}", number);
    }
    number.to_string()
}

fn group_digits(integer_part: &str) -> String {
    let mut grouped = String::with_capacity(integer_part.len() + integer_part.len() / 3);
    for (index, ch) in integer_part.chars().enumerate() {
        if index > 0 && (integer_part.len() - index) % 3 == 0 {
            grouped.push(',');
        }
        grouped.push(ch);
    }
    grouped
}

fn put_builtin(value: &ScalarValue, format_name: &str) -> Result<ScalarValue, String> {
    if matches!(value, ScalarValue::Null) {
        return Ok(ScalarValue::Null);
    }

    if let Some(z_spec) = format_name.strip_prefix('z') {
        let (width, precision) = parse_width_precision(z_spec)
            .ok_or_else(|| format!("Unsupported format: {format_name}"))?;
        let number = scalar_to_number(value)?;
        let rendered = format_zero_filled_number(number, width, precision.unwrap_or(0));
        return Ok(ScalarValue::Text(rendered));
    }

    if let Some((width, precision)) = parse_width_precision(format_name) {
        let number = scalar_to_number(value)?;
        let _ = width;
        let rendered = format_fixed_number(number, precision.unwrap_or(0));
        return Ok(ScalarValue::Text(rendered));
    }

    if let Some(comma_spec) = format_name.strip_prefix("comma") {
        if let Some((_, precision)) = parse_width_precision(comma_spec) {
            let number = scalar_to_number(value)?;
            let rendered = format_grouped_number(number, precision.unwrap_or(0));
            return Ok(ScalarValue::Text(rendered));
        }
    }

    match format_name {
        "best" => Ok(ScalarValue::Text(format_best_number(scalar_to_number(value)?))),
        "e8601da" => Ok(ScalarValue::Text(scalar_to_date(value)?.format("%Y-%m-%d").to_string())),
        "e8601dt" => Ok(ScalarValue::Text(
            scalar_to_datetime(value)?.format("%Y-%m-%dT%H:%M:%S").to_string(),
        )),
        "time" => Ok(ScalarValue::Text(scalar_to_time(value)?.format("%H:%M:%S").to_string())),
        _ => Err(format!("Unsupported format: {format_name}")),
    }
}

fn input_builtin(value: &ScalarValue, format_name: &str) -> Result<ScalarValue, String> {
    let raw = scalar_to_text(value);
    let text = raw.trim();
    if text.is_empty() {
        return Ok(ScalarValue::Null);
    }

    match format_name {
        "best" => Ok(ScalarValue::Number(
            text.parse::<f64>()
                .map_err(|_| format!("Invalid numeric value: {text}"))?,
        )),
        "yymmdd6" => {
            let year = text[0..2]
                .parse::<i32>()
                .map_err(|_| format!("Invalid yymmdd6 value: {text}"))?;
            let month = text[2..4]
                .parse::<u32>()
                .map_err(|_| format!("Invalid yymmdd6 value: {text}"))?;
            let day = text[4..6]
                .parse::<u32>()
                .map_err(|_| format!("Invalid yymmdd6 value: {text}"))?;
            let date = NaiveDate::from_ymd_opt(2000 + year, month, day)
                .ok_or_else(|| format!("Invalid yymmdd6 value: {text}"))?;
            Ok(ScalarValue::Date(date))
        }
        "yymmdd8" => {
            let year = text[0..4]
                .parse::<i32>()
                .map_err(|_| format!("Invalid yymmdd8 value: {text}"))?;
            let month = text[4..6]
                .parse::<u32>()
                .map_err(|_| format!("Invalid yymmdd8 value: {text}"))?;
            let day = text[6..8]
                .parse::<u32>()
                .map_err(|_| format!("Invalid yymmdd8 value: {text}"))?;
            let date = NaiveDate::from_ymd_opt(year, month, day)
                .ok_or_else(|| format!("Invalid yymmdd8 value: {text}"))?;
            Ok(ScalarValue::Date(date))
        }
        "yymmdd10" | "e8601da" => Ok(ScalarValue::Date(
            NaiveDate::parse_from_str(text, "%Y-%m-%d")
                .map_err(|_| format!("Invalid date value: {text}"))?,
        )),
        "e8601dt" => Ok(ScalarValue::DateTime(
            NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S")
                .map_err(|_| format!("Invalid datetime value: {text}"))?,
        )),
        "time" => Ok(ScalarValue::Time(parse_time_text(text)?)),
        _ => Err(format!("Unsupported informat: {format_name}")),
    }
}

fn hour_builtin(value: &ScalarValue) -> Result<ScalarValue, String> {
    if matches!(value, ScalarValue::Null) {
        return Ok(ScalarValue::Null);
    }

    let time = match value {
        ScalarValue::Time(time) => *time,
        ScalarValue::DateTime(datetime) => datetime.time(),
        ScalarValue::Text(text) => parse_time_text(text.trim())?,
        _ => return Err("hour() requires a time-like value".to_string()),
    };

    let hour_value = f64::from(time.hour())
        + f64::from(time.minute()) / 60.0
        + f64::from(time.second()) / 3600.0;
    Ok(ScalarValue::Number(hour_value))
}

fn parse_prx_flags(flags_part: &str) -> Result<(bool, bool, bool, bool), String> {
    let mut case_insensitive = false;
    let mut multi_line = false;
    let mut dot_all = false;
    let mut verbose = false;
    for ch in flags_part.chars() {
        match ch.to_ascii_lowercase() {
            'i' => case_insensitive = true,
            'm' => multi_line = true,
            's' => dot_all = true,
            'x' => verbose = true,
            other => return Err(format!("Unsupported PRX flag: {other}")),
        }
    }
    Ok((case_insensitive, multi_line, dot_all, verbose))
}

fn compile_prx_pattern(raw: &str) -> Result<Regex, String> {
    let (pattern_body, flags_part) = if raw.len() >= 2 && raw.starts_with('/') && raw.matches('/').count() >= 2 {
        let last_delim = raw.rfind('/').unwrap();
        (&raw[1..last_delim], &raw[last_delim + 1..])
    } else {
        (raw, "")
    };
    let (ci, ml, da, _verbose) = parse_prx_flags(flags_part)?;
    RegexBuilder::new(pattern_body)
        .case_insensitive(ci)
        .multi_line(ml)
        .dot_matches_new_line(da)
        .build()
        .map_err(|e| format!("Invalid prxmatch pattern: {e}"))
}

fn parse_prxchange_pattern(raw: &str) -> Result<(Regex, String), String> {
    if !raw.starts_with("s/") {
        return Err("prxchange pattern must start with s/".to_string());
    }
    let first_sep = raw[2..].find('/').ok_or("prxchange pattern is missing replacement separator")?;
    let first_sep = first_sep + 2;
    let second_sep = raw[first_sep + 1..].find('/').ok_or("prxchange pattern is missing closing separator")?;
    let second_sep = second_sep + first_sep + 1;
    let regex_body = &raw[2..first_sep];
    let replacement = &raw[first_sep + 1..second_sep];
    let flags_part = &raw[second_sep + 1..];
    let (ci, ml, da, _verbose) = parse_prx_flags(flags_part)?;
    let compiled = RegexBuilder::new(regex_body)
        .case_insensitive(ci)
        .multi_line(ml)
        .dot_matches_new_line(da)
        .build()
        .map_err(|e| format!("Invalid prxchange pattern: {e}"))?;
    Ok((compiled, replacement.to_string()))
}

fn evaluate_function(
    py: Python<'_>,
    name: &str,
    args: &[ScalarValue],
    state: &mut EvalRuntimeState,
) -> Result<ScalarValue, String> {
    fn parse_steps(raw: Option<&ScalarValue>, default_value: isize) -> Result<isize, String> {
        let Some(value) = raw else {
            return Ok(default_value);
        };
        Ok(scalar_to_number(value)? as isize)
    }

    fn shift_by_variable(
        py: Python<'_>,
        state: &EvalRuntimeState,
        variable_name: &str,
        steps: isize,
        default: ScalarValue,
    ) -> Result<ScalarValue, String> {
        let target_index = state.row_index - steps;
        if target_index < 0 || (target_index as usize) >= state.row_view.len() {
            return Ok(default);
        }

        let row = state.row_view[target_index as usize].bind(py);
        let value_any = row.get_item(variable_name).map_err(|error| error.to_string())?;
        let Some(value) = value_any else {
            return Ok(default.clone());
        };
        py_to_scalar(&value).map_err(|error| error.to_string())
    }

    let lowered = name.to_lowercase();
    match lowered.as_str() {
        "put" => {
            if args.len() != 2 {
                return Err("put() expects 2 arguments".to_string());
            }
            let (format_name, has_dot) = normalize_format_name(&args[1]);
            if requires_numeric_format_dot(&format_name, has_dot) {
                return Err(format!("Unsupported format: {}", scalar_to_text(&args[1])));
            }
            put_builtin(&args[0], &format_name)
        }
        "input" => {
            if args.len() != 2 {
                return Err("input() expects 2 arguments".to_string());
            }
            let (format_name, _) = normalize_format_name(&args[1]);
            input_builtin(&args[0], &format_name)
        }
        "hour" => {
            if args.len() != 1 {
                return Err("hour() expects 1 argument".to_string());
            }
            hour_builtin(&args[0])
        }
        "substr" => {
            if args.len() < 2 || args.len() > 3 {
                return Err("substr() expects 2 or 3 arguments".to_string());
            }
            let source = scalar_to_text(&args[0]);
            let start_index = scalar_to_number(&args[1])? as isize - 1;
            let start = start_index.max(0) as usize;
            if start >= source.len() {
                return Ok(ScalarValue::Text("".to_string()));
            }
            if args.len() == 2 {
                return Ok(ScalarValue::Text(source[start..].to_string()));
            }
            let length = scalar_to_number(&args[2])?.max(0.0) as usize;
            let end = (start + length).min(source.len());
            Ok(ScalarValue::Text(source[start..end].to_string()))
        }
        "scan" => {
            if args.len() < 2 || args.len() > 3 {
                return Err("scan() expects 2 or 3 arguments".to_string());
            }
            let source = scalar_to_text(&args[0]);
            let index = scalar_to_number(&args[1])? as isize - 1;
            let delimiters = if args.len() == 3 {
                scalar_to_text(&args[2])
            } else {
                " ".to_string()
            };
            let parts = source
                .split(|ch| delimiters.contains(ch))
                .filter(|part| !part.is_empty())
                .collect::<Vec<_>>();
            if index < 0 || (index as usize) >= parts.len() {
                return Ok(ScalarValue::Text("".to_string()));
            }
            Ok(ScalarValue::Text(parts[index as usize].to_string()))
        }
        "compress" => {
            if args.is_empty() || args.len() > 2 {
                return Err("compress() expects 1 or 2 arguments".to_string());
            }
            let source = scalar_to_text(&args[0]);
            if args.len() == 1 {
                return Ok(ScalarValue::Text(source.chars().filter(|ch| !ch.is_whitespace()).collect()));
            }
            let remove_chars = scalar_to_text(&args[1]);
            Ok(ScalarValue::Text(
                source
                    .chars()
                    .filter(|ch| !remove_chars.contains(*ch))
                    .collect(),
            ))
        }
        "trim" => {
            if args.len() != 1 {
                return Err("trim() expects 1 argument".to_string());
            }
            Ok(ScalarValue::Text(scalar_to_text(&args[0]).trim().to_string()))
        }
        "upcase" => {
            if args.len() != 1 {
                return Err("upcase() expects 1 argument".to_string());
            }
            Ok(ScalarValue::Text(scalar_to_text(&args[0]).to_uppercase()))
        }
        "lowcase" => {
            if args.len() != 1 {
                return Err("lowcase() expects 1 argument".to_string());
            }
            Ok(ScalarValue::Text(scalar_to_text(&args[0]).to_lowercase()))
        }
        "abs" => {
            if args.len() != 1 {
                return Err("abs() expects 1 argument".to_string());
            }
            Ok(ScalarValue::Number(scalar_to_number(&args[0])?.abs()))
        }
        "ceil" => {
            if args.len() != 1 {
                return Err("ceil() expects 1 argument".to_string());
            }
            Ok(ScalarValue::Number(scalar_to_number(&args[0])?.ceil()))
        }
        "floor" => {
            if args.len() != 1 {
                return Err("floor() expects 1 argument".to_string());
            }
            Ok(ScalarValue::Number(scalar_to_number(&args[0])?.floor()))
        }
        "round" => {
            if args.is_empty() || args.len() > 2 {
                return Err("round() expects 1 or 2 arguments".to_string());
            }
            let value = scalar_to_number(&args[0])?;
            if args.len() == 1 {
                return Ok(ScalarValue::Number(value.round()));
            }
            let unit = scalar_to_number(&args[1])?;
            if unit.abs() < f64::EPSILON {
                return Ok(ScalarValue::Number(value));
            }
            Ok(ScalarValue::Number((value / unit).round() * unit))
        }
        "pow" => {
            if args.len() != 2 {
                return Err("pow() expects 2 arguments".to_string());
            }
            let base = scalar_to_number(&args[0])?;
            let exponent = scalar_to_number(&args[1])?;
            Ok(ScalarValue::Number(base.powf(exponent)))
        }
        "mod" => {
            if args.len() != 2 {
                return Err("mod() expects 2 arguments".to_string());
            }
            let left = scalar_to_number(&args[0])?;
            let right = scalar_to_number(&args[1])?;
            Ok(ScalarValue::Number(left % right))
        }
        "max" => {
            if args.is_empty() {
                return Ok(ScalarValue::Null);
            }
            let mut values: Vec<f64> = args
                .iter()
                .filter(|value| !scalar_is_missing(value))
                .map(scalar_to_number)
                .collect::<Result<Vec<_>, _>>()?;
            if values.is_empty() {
                return Ok(ScalarValue::Null);
            }
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            Ok(ScalarValue::Number(*values.last().unwrap_or(&0.0)))
        }
        "min" => {
            if args.is_empty() {
                return Ok(ScalarValue::Null);
            }
            let mut values: Vec<f64> = args
                .iter()
                .filter(|value| !scalar_is_missing(value))
                .map(scalar_to_number)
                .collect::<Result<Vec<_>, _>>()?;
            if values.is_empty() {
                return Ok(ScalarValue::Null);
            }
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            Ok(ScalarValue::Number(*values.first().unwrap_or(&0.0)))
        }
        "missing" => {
            if args.len() != 1 {
                return Err("missing() expects 1 argument".to_string());
            }
            Ok(ScalarValue::Bool(scalar_is_missing(&args[0])))
        }
        "nmiss" => {
            let count = args.iter().filter(|value| scalar_is_missing(value)).count();
            Ok(ScalarValue::Number(count as f64))
        }
        "shift" => {
            if args.is_empty() || args.len() > 3 {
                return Err("shift() expects 1 to 3 arguments".to_string());
            }

            let default = if args.len() == 3 {
                args[2].clone()
            } else {
                ScalarValue::Null
            };
            let steps = parse_steps(args.get(1), 1)?;

            if let ScalarValue::Text(variable_name) = &args[0] {
                return shift_by_variable(py, state, variable_name, steps, default);
            }

            if steps < 0 {
                return Err("shift() with negative offset requires variable name text argument".to_string());
            }
            if steps == 0 {
                return Ok(args[0].clone());
            }

            let offset = steps as usize;
            let queue = state.lag_queues.entry(offset).or_default();
            queue.push(args[0].clone());
            if queue.len() <= offset {
                return Ok(default);
            }
            Ok(queue.remove(0))
        }
        "lag" => {
            if args.is_empty() || args.len() > 3 {
                return Err("lag() expects 1 to 3 arguments".to_string());
            }
            let value = args[0].clone();
            let offset = parse_steps(args.get(1), 1)?;
            if offset < 1 {
                return Err("lag() offset must be >= 1".to_string());
            }
            let default = if args.len() == 3 {
                args[2].clone()
            } else {
                ScalarValue::Null
            };
            let offset = offset as usize;
            let queue = state.lag_queues.entry(offset).or_default();
            queue.push(value);
            if queue.len() <= offset {
                return Ok(default);
            }
            Ok(queue.remove(0))
        }
        "lead" => {
            if args.is_empty() || args.len() > 3 {
                return Err("lead() expects 1 to 3 arguments".to_string());
            }
            let variable_name = scalar_to_text(&args[0]);
            let offset = parse_steps(args.get(1), 1)?;
            if offset < 1 {
                return Err("lead() offset must be >= 1".to_string());
            }
            let default = if args.len() == 3 {
                args[2].clone()
            } else {
                ScalarValue::Null
            };
            shift_by_variable(py, state, &variable_name, -offset, default)
        }
        "mdy" => {
            if args.len() != 3 {
                return Err("mdy() expects 3 arguments".to_string());
            }
            let month = scalar_to_number(&args[0])? as u32;
            let day = scalar_to_number(&args[1])? as u32;
            let year = scalar_to_number(&args[2])? as i32;
            let date = NaiveDate::from_ymd_opt(year, month, day)
                .ok_or_else(|| "invalid date in mdy()".to_string())?;
            Ok(ScalarValue::Date(date))
        }
        "year" => {
            if args.len() != 1 {
                return Err("year() expects 1 argument".to_string());
            }
            let date = scalar_to_date(&args[0])?;
            Ok(ScalarValue::Number(date.year() as f64))
        }
        "intck" => {
            if args.len() != 3 {
                return Err("intck() expects 3 arguments".to_string());
            }
            let interval = scalar_to_text(&args[0]).to_lowercase();
            let start = scalar_to_date(&args[1])?;
            let end = scalar_to_date(&args[2])?;
            let value = match interval.as_str() {
                "day" => end.signed_duration_since(start).num_days() as f64,
                "month" => ((end.year() - start.year()) * 12 + (end.month() as i32 - start.month() as i32)) as f64,
                "year" => (end.year() - start.year()) as f64,
                _ => return Err(format!("unsupported intck interval: {interval}")),
            };
            Ok(ScalarValue::Number(value))
        }
        // --- String functions ---
        "propcase" => {
            if args.len() != 1 {
                return Err("propcase() expects 1 argument".to_string());
            }
            let source = scalar_to_text(&args[0]);
            let result: String = source
                .split_inclusive(char::is_whitespace)
                .map(|word| {
                    let mut chars = word.chars();
                    match chars.next() {
                        None => String::new(),
                        Some(first) => {
                            let upper: String = first.to_uppercase().collect();
                            upper + &chars.as_str().to_lowercase()
                        }
                    }
                })
                .collect();
            Ok(ScalarValue::Text(result))
        }
        "__dsl_concat__" | "cat" => {
            // cat() concatenates all arguments as-is (None → "")
            let result: String = args.iter().map(scalar_to_text).collect();
            Ok(ScalarValue::Text(result))
        }
        "cats" => {
            // cats() strips leading/trailing blanks from each argument then concatenates
            let result: String = args.iter().map(|a| scalar_to_text(a).trim().to_string()).collect();
            Ok(ScalarValue::Text(result))
        }
        "catt" => {
            // catt() trims trailing blanks from each argument then concatenates
            let result: String = args.iter().map(|a| scalar_to_text(a).trim_end().to_string()).collect();
            Ok(ScalarValue::Text(result))
        }
        "catx" => {
            // catx(delimiter, arg1, arg2, ...) joins non-empty stripped args with delimiter
            if args.is_empty() {
                return Err("catx() expects at least 1 argument".to_string());
            }
            let delimiter = scalar_to_text(&args[0]);
            let parts: Vec<String> = args[1..]
                .iter()
                .filter_map(|a| {
                    if scalar_is_missing(a) {
                        return None;
                    }
                    let trimmed = scalar_to_text(a).trim().to_string();
                    if trimmed.is_empty() { None } else { Some(trimmed) }
                })
                .collect();
            Ok(ScalarValue::Text(parts.join(&delimiter)))
        }
        "index" => {
            if args.len() != 2 {
                return Err("index() expects 2 arguments".to_string());
            }
            let source = scalar_to_text(&args[0]);
            let excerpt = scalar_to_text(&args[1]);
            if excerpt.is_empty() {
                return Ok(ScalarValue::Number(1.0));
            }
            match source.find(&excerpt) {
                Some(pos) => Ok(ScalarValue::Number((pos + 1) as f64)),
                None => Ok(ScalarValue::Number(0.0)),
            }
        }
        "find" => {
            if args.is_empty() || args.len() > 4 {
                return Err("find() expects 2 to 4 arguments".to_string());
            }
            let mut source = scalar_to_text(&args[0]);
            let mut excerpt = scalar_to_text(&args[1]);
            let start_index = if args.len() >= 3 {
                (scalar_to_number(&args[2])? as isize - 1).max(0) as usize
            } else {
                0
            };
            let modifiers = if args.len() >= 4 {
                scalar_to_text(&args[3]).to_lowercase()
            } else {
                String::new()
            };
            if modifiers.contains('i') {
                source = source.to_lowercase();
                excerpt = excerpt.to_lowercase();
            }
            match source[start_index..].find(&excerpt) {
                Some(pos) => Ok(ScalarValue::Number((start_index + pos + 1) as f64)),
                None => Ok(ScalarValue::Number(0.0)),
            }
        }
        "tranwrd" => {
            if args.len() != 3 {
                return Err("tranwrd() expects 3 arguments".to_string());
            }
            let source = scalar_to_text(&args[0]);
            let target = scalar_to_text(&args[1]);
            let replacement = scalar_to_text(&args[2]);
            Ok(ScalarValue::Text(source.replace(&target, &replacement)))
        }
        "translate" => {
            if args.len() != 3 {
                return Err("translate() expects 3 arguments".to_string());
            }
            let source = scalar_to_text(&args[0]);
            let to_chars: Vec<char> = scalar_to_text(&args[1]).chars().collect();
            let from_chars: Vec<char> = scalar_to_text(&args[2]).chars().collect();
            let mut map: HashMap<char, String> = HashMap::new();
            for (i, src_char) in from_chars.iter().enumerate() {
                map.insert(
                    *src_char,
                    if i < to_chars.len() {
                        to_chars[i].to_string()
                    } else {
                        String::new()
                    },
                );
            }
            let result: String = source.chars().map(|c| {
                map.get(&c).cloned().unwrap_or_else(|| c.to_string())
            }).collect();
            Ok(ScalarValue::Text(result))
        }
        "length" => {
            if args.len() != 1 {
                return Err("length() expects 1 argument".to_string());
            }
            let source = scalar_to_text(&args[0]);
            Ok(ScalarValue::Number(source.len() as f64))
        }
        "lengthn" => {
            if args.len() != 1 {
                return Err("lengthn() expects 1 argument".to_string());
            }
            let source = scalar_to_text(&args[0]);
            if source.is_empty() {
                Ok(ScalarValue::Number(0.0))
            } else {
                Ok(ScalarValue::Number(source.len() as f64))
            }
        }
        "strip" => {
            if args.len() != 1 {
                return Err("strip() expects 1 argument".to_string());
            }
            Ok(ScalarValue::Text(scalar_to_text(&args[0]).trim().to_string()))
        }
        "reverse" => {
            if args.len() != 1 {
                return Err("reverse() expects 1 argument".to_string());
            }
            let source = scalar_to_text(&args[0]);
            Ok(ScalarValue::Text(source.chars().rev().collect()))
        }
        "repeat" => {
            if args.len() != 2 {
                return Err("repeat() expects 2 arguments".to_string());
            }
            let source = scalar_to_text(&args[0]);
            let count = scalar_to_number(&args[1])?.max(0.0) as usize;
            Ok(ScalarValue::Text(source.repeat(count)))
        }
        "countw" => {
            if args.is_empty() || args.len() > 2 {
                return Err("countw() expects 1 or 2 arguments".to_string());
            }
            let source = scalar_to_text(&args[0]).trim().to_string();
            let delimiters = if args.len() == 2 {
                scalar_to_text(&args[1])
            } else {
                " ".to_string()
            };
            let count = source
                .split(|ch: char| delimiters.contains(ch))
                .filter(|part| !part.is_empty())
                .count();
            Ok(ScalarValue::Number(count as f64))
        }
        // --- Numeric functions ---
        "int" => {
            if args.len() != 1 {
                return Err("int() expects 1 argument".to_string());
            }
            let value = scalar_to_number(&args[0])?;
            Ok(ScalarValue::Number(value.trunc()))
        }
        "sum" => {
            let mut total = 0.0f64;
            for arg in args {
                if scalar_is_missing(arg) {
                    continue;
                }
                total += scalar_to_number(arg)?;
            }
            Ok(ScalarValue::Number(total))
        }
        "mean" => {
            let values: Vec<f64> = args
                .iter()
                .filter(|a| !scalar_is_missing(a))
                .map(scalar_to_number)
                .collect::<Result<Vec<_>, _>>()?;
            if values.is_empty() {
                return Ok(ScalarValue::Number(f64::NAN));
            }
            let sum: f64 = values.iter().sum();
            Ok(ScalarValue::Number(sum / values.len() as f64))
        }
        "sqrt" => {
            if args.len() != 1 {
                return Err("sqrt() expects 1 argument".to_string());
            }
            Ok(ScalarValue::Number(scalar_to_number(&args[0])?.sqrt()))
        }
        "log" => {
            if args.is_empty() || args.len() > 2 {
                return Err("log() expects 1 or 2 arguments".to_string());
            }
            let value = scalar_to_number(&args[0])?;
            if args.len() == 1 {
                Ok(ScalarValue::Number(value.ln()))
            } else {
                let base = scalar_to_number(&args[1])?;
                Ok(ScalarValue::Number(value.ln() / base.ln()))
            }
        }
        "exp" => {
            if args.len() != 1 {
                return Err("exp() expects 1 argument".to_string());
            }
            Ok(ScalarValue::Number(scalar_to_number(&args[0])?.exp()))
        }
        "sign" => {
            if args.len() != 1 {
                return Err("sign() expects 1 argument".to_string());
            }
            let value = scalar_to_number(&args[0])?;
            let result = if value > 0.0 { 1.0 } else if value < 0.0 { -1.0 } else { 0.0 };
            Ok(ScalarValue::Number(result))
        }
        // --- Missing value functions ---
        "cmiss" => {
            let count = args.iter().filter(|a| scalar_is_missing(a)).count();
            Ok(ScalarValue::Number(count as f64))
        }
        // --- Regex functions ---
        "prxmatch" => {
            if args.len() != 2 {
                return Err("prxmatch() expects 2 arguments".to_string());
            }
            let pattern_raw = scalar_to_text(&args[0]);
            let source = scalar_to_text(&args[1]);
            let compiled = compile_prx_pattern(&pattern_raw)?;
            match compiled.find(&source) {
                Some(m) => Ok(ScalarValue::Number((m.start() + 1) as f64)),
                None => Ok(ScalarValue::Number(0.0)),
            }
        }
        "prxchange" => {
            if args.len() != 3 {
                return Err("prxchange() expects 3 arguments".to_string());
            }
            let pattern_raw = scalar_to_text(&args[0]);
            let times = scalar_to_number(&args[1])? as i64;
            let source = scalar_to_text(&args[2]);
            let (compiled, replacement) = parse_prxchange_pattern(&pattern_raw)?;
            if times <= 0 {
                // times <= 0 means replace all
                Ok(ScalarValue::Text(compiled.replace_all(&source, replacement.as_str()).to_string()))
            } else {
                // replace exactly `times` occurrences
                let mut result = source.clone();
                let mut replaced = 0i64;
                loop {
                    if replaced >= times {
                        break;
                    }
                    if let Some(m) = compiled.find(&result) {
                        let before = &result[..m.start()];
                        let after = &result[m.end()..];
                        result = format!("{before}{replacement}{after}");
                        replaced += 1;
                    } else {
                        break;
                    }
                }
                Ok(ScalarValue::Text(result))
            }
        }
        _ => Err(format!("unsupported function: {name}")),
    }
}

fn evaluate_expression(row: &Bound<'_, PyDict>, expr: &Expr, state: &mut EvalRuntimeState) -> Result<ScalarValue, String> {
    match expr {
        Expr::Literal(value) => Ok(value.clone()),
        Expr::Variable(name) => {
            let value_any = get_row_item(row, name)?;
            let Some(value) = value_any else {
                if name.ends_with('.') {
                    return Ok(ScalarValue::Text(name.to_string()));
                }
                return Ok(ScalarValue::Null);
            };
            py_to_scalar(&value).map_err(|error| error.to_string())
        }
        Expr::ArrayRef { name, index } => {
            let index_value = scalar_to_number(&evaluate_expression(row, index, state)?)?;
            let variable_name = resolve_array_variable_name(state, name, index_value)?;
            let value_any = get_row_item(row, &variable_name)?;
            let Some(value) = value_any else {
                return Ok(ScalarValue::Null);
            };
            py_to_scalar(&value).map_err(|error| error.to_string())
        }
        Expr::UnaryNot(inner) => Ok(ScalarValue::Bool(!scalar_to_bool(&evaluate_expression(row, inner, state)?)?)),
        Expr::UnaryNeg(inner) => Ok(ScalarValue::Number(-scalar_to_number(&evaluate_expression(row, inner, state)?)?)),
        Expr::Binary { left, op, right } => {
            let left_value = evaluate_expression(row, left, state)?;
            let right_value = evaluate_expression(row, right, state)?;
            match op {
                BinaryOp::Add => Ok(ScalarValue::Number(scalar_to_number(&left_value)? + scalar_to_number(&right_value)?)),
                BinaryOp::Sub => Ok(ScalarValue::Number(scalar_to_number(&left_value)? - scalar_to_number(&right_value)?)),
                BinaryOp::Mul => Ok(ScalarValue::Number(scalar_to_number(&left_value)? * scalar_to_number(&right_value)?)),
                BinaryOp::Div => Ok(ScalarValue::Number(scalar_to_number(&left_value)? / scalar_to_number(&right_value)?)),
                BinaryOp::Pow => Ok(ScalarValue::Number(scalar_to_number(&left_value)?.powf(scalar_to_number(&right_value)?))),
                BinaryOp::Mod => Ok(ScalarValue::Number(scalar_to_number(&left_value)? % scalar_to_number(&right_value)?)),
                BinaryOp::And => Ok(ScalarValue::Bool(scalar_to_bool(&left_value)? && scalar_to_bool(&right_value)?)),
                BinaryOp::Or => Ok(ScalarValue::Bool(scalar_to_bool(&left_value)? || scalar_to_bool(&right_value)?)),
                BinaryOp::Eq => Ok(ScalarValue::Bool(scalar_eq(&left_value, &right_value)?)),
                BinaryOp::Ne => Ok(ScalarValue::Bool(!scalar_eq(&left_value, &right_value)?)),
                BinaryOp::Gt => Ok(ScalarValue::Bool(scalar_compare(&left_value, &right_value)? == std::cmp::Ordering::Greater)),
                BinaryOp::Ge => Ok(ScalarValue::Bool(matches!(scalar_compare(&left_value, &right_value)?, std::cmp::Ordering::Greater | std::cmp::Ordering::Equal))),
                BinaryOp::Lt => Ok(ScalarValue::Bool(scalar_compare(&left_value, &right_value)? == std::cmp::Ordering::Less)),
                BinaryOp::Le => Ok(ScalarValue::Bool(matches!(scalar_compare(&left_value, &right_value)?, std::cmp::Ordering::Less | std::cmp::Ordering::Equal))),
            }
        }
        Expr::FunctionCall { name, args } => {
            let lowered = name.to_lowercase();
            if lowered == "dim" {
                if args.len() != 1 {
                    return Err("dim() expects 1 argument".to_string());
                }
                let array_name = match &args[0] {
                    Expr::Variable(var_name) => var_name.clone(),
                    Expr::ArrayRef { name, .. } => name.clone(),
                    Expr::FunctionCall { name, .. } => name.clone(),
                    _ => return Err("dim() expects an array name".to_string()),
                };
                let length = state
                    .array_defs
                    .get(&array_name)
                    .map(|items| items.len())
                    .ok_or_else(|| format!("array is not defined: {array_name}"))?;
                return Ok(ScalarValue::Number(length as f64));
            }

            if lowered == "vname" {
                if args.len() != 1 {
                    return Err("vname() expects 1 argument".to_string());
                }
                let (array_name, index_expr) = match &args[0] {
                    Expr::ArrayRef { name, index } => (name.clone(), index.as_ref().clone()),
                    Expr::FunctionCall { name, args } if args.len() == 1 && state.array_defs.contains_key(name) => {
                        (name.clone(), args[0].clone())
                    }
                    _ => return Err("vname() expects an array element".to_string()),
                };
                let index_value = scalar_to_number(&evaluate_expression(row, &index_expr, state)?)?;
                let variable_name = resolve_array_variable_name(state, &array_name, index_value)?;
                return Ok(ScalarValue::Text(variable_name));
            }

            if state.array_defs.contains_key(name) && args.len() == 1 {
                let index_value = scalar_to_number(&evaluate_expression(row, &args[0], state)?)?;
                let variable_name = resolve_array_variable_name(state, name, index_value)?;
                let value_any = get_row_item(row, &variable_name)?;
                let Some(value) = value_any else {
                    return Ok(ScalarValue::Null);
                };
                return py_to_scalar(&value).map_err(|error| error.to_string());
            }

            let evaluated_args = args
                .iter()
                .map(|arg| evaluate_expression(row, arg, state))
                .collect::<Result<Vec<_>, _>>()?;
            evaluate_function(row.py(), name, &evaluated_args, state)
        }
    }
}

fn evaluate_simple_where(row: &Bound<'_, PyDict>, expr: &str, state: &mut EvalRuntimeState) -> Result<bool, String> {
    let parsed = parse_expression_cached(expr, state)?;
    let evaluated = evaluate_expression(row, &parsed, state)?;
    scalar_to_bool(&evaluated)
}

fn apply_dataset_ref_options_to_row(
    py: Python<'_>,
    source_row: &Bound<'_, PyDict>,
    options: &AstDatasetRefOptions,
    state: &mut EvalRuntimeState,
) -> Result<Option<Py<PyDict>>, String> {
    let mut working = PyDict::new(py);
    for (key, value) in source_row.iter() {
        working
            .set_item(key, value)
            .map_err(|error| format!("dataset option copy failed: {error}"))?;
    }

    if !options.keep_vars.is_empty() {
        let projected = PyDict::new(py);
        for name in &options.keep_vars {
            if let Some(value) = get_row_item(&working, name)? {
                projected
                    .set_item(name, value)
                    .map_err(|error| format!("dataset option keep set failed: {error}"))?;
            }
        }
        working = projected;
    }

    if !options.drop_vars.is_empty() {
        for name in &options.drop_vars {
            delete_row_item(&working, name).ok();
        }
    }

    if !options.rename_map.is_empty() {
        let mut rename_targets: Vec<String> = Vec::new();
        for value in options.rename_map.values() {
            let normalized_target = column_key(value);
            if rename_targets.contains(&normalized_target) {
                return Err("dataset option RENAME= has duplicate target names".to_string());
            }
            rename_targets.push(normalized_target);
        }

        for old_name in options.rename_map.keys() {
            if get_row_item(&working, old_name)?.is_none() {
                return Err(format!("dataset option RENAME= references unknown variable: {old_name}"));
            }
        }

        let renamed = PyDict::new(py);
        let mut resolved_rename_map: HashMap<String, String> = HashMap::new();
        for (old_name, new_name) in &options.rename_map {
            if let Some(resolved_name) = resolve_row_key(&working, old_name)? {
                resolved_rename_map.insert(resolved_name, new_name.clone());
            }
        }
        for (key_any, value_any) in working.iter() {
            let key = key_any
                .extract::<String>()
                .map_err(|error| format!("dataset option rename key extract failed: {error}"))?;
            let renamed_key = resolved_rename_map.get(&key).cloned().unwrap_or(key);
            renamed
                .set_item(renamed_key, value_any)
                .map_err(|error| format!("dataset option rename set failed: {error}"))?;
        }
        working = renamed;
    }

    if let Some(expression) = &options.where_expr {
        let pass = evaluate_simple_where(&working, expression, state)?;
        if !pass {
            return Ok(None);
        }
    }

    Ok(Some(working.unbind()))
}

fn resolve_declared_target_name(requested: &str, declared: &[String]) -> String {
    let requested_lower = requested.to_lowercase();
    for target in declared {
        if target.to_lowercase() == requested_lower {
            return target.clone();
        }
    }
    requested.to_string()
}

fn read_rows_from_arrow_stream(py: Python<'_>, stream_obj: &Bound<'_, PyAny>) -> Result<Py<PyList>, String> {
    let pyarrow = py.import("pyarrow").map_err(|error| format!("failed to import pyarrow: {error}"))?;
    let reader_class = pyarrow
        .getattr("RecordBatchReader")
        .map_err(|error| format!("pyarrow.RecordBatchReader is missing: {error}"))?;
    let import_method = match reader_class.getattr("_import_from_c_capsule") {
        Ok(method) => method,
        Err(_) => reader_class
            .getattr("_import_from_c")
            .map_err(|error| format!("pyarrow.RecordBatchReader import API is missing: {error}"))?,
    };
    let reader = import_method
        .call1((stream_obj.clone(),))
        .map_err(|error| format!("failed to import Arrow C stream: {error}"))?;

    let rows = PyList::empty(py);
    loop {
        let batch = match reader.call_method0("read_next_batch") {
            Ok(value) => value,
            Err(error) => {
                if error.is_instance_of::<pyo3::exceptions::PyStopIteration>(py) {
                    break;
                }
                return Err(format!("failed to read next record batch: {error}"));
            }
        };

        let batch_rows_any = batch
            .call_method0("to_pylist")
            .map_err(|error| format!("failed to convert record batch to row list: {error}"))?;
        let batch_rows = batch_rows_any
            .downcast::<PyList>()
            .map_err(|_| "RecordBatch to_pylist did not return a list".to_string())?;

        for row_any in batch_rows.iter() {
            rows
                .append(row_any)
                .map_err(|error| format!("failed to append row from record batch: {error}"))?;
        }
    }

    Ok(rows.unbind())
}

fn export_output_streams(py: Python<'_>, outputs: &Bound<'_, PyDict>) -> Result<Py<PyDict>, String> {
    let pyarrow = py.import("pyarrow").map_err(|error| format!("failed to import pyarrow: {error}"))?;
    let table_class = pyarrow
        .getattr("Table")
        .map_err(|error| format!("pyarrow.Table is missing: {error}"))?;

    let output_streams = PyDict::new(py);
    for (target_any, rows_any) in outputs.iter() {
        let rows_list = rows_any
            .downcast::<PyList>()
            .map_err(|_| "failed to treat output rows as list".to_string())?;
        let mut all_keys: Vec<String> = Vec::new();
        for row_any in rows_list.iter() {
            let row = row_any
                .downcast::<PyDict>()
                .map_err(|_| "failed to treat output row as dict".to_string())?;
            for (key_any, _) in row.iter() {
                let key = key_any
                    .extract::<String>()
                    .map_err(|error| format!("failed to read output row key: {error}"))?;
                if !all_keys.contains(&key) {
                    all_keys.push(key);
                }
            }
        }

        let normalized_rows = PyList::empty(py);
        for row_any in rows_list.iter() {
            let row = row_any
                .downcast::<PyDict>()
                .map_err(|_| "failed to treat output row as dict".to_string())?;
            let normalized = PyDict::new(py);
            for key in &all_keys {
                if let Some(value) = row
                    .get_item(key)
                    .map_err(|error| format!("failed to read output row value: {error}"))?
                {
                    normalized
                        .set_item(key, value)
                        .map_err(|error| format!("failed to normalize output row: {error}"))?;
                } else {
                    normalized
                        .set_item(key, py.None())
                        .map_err(|error| format!("failed to normalize missing output value: {error}"))?;
                }
            }
            normalized_rows
                .append(normalized)
                .map_err(|error| format!("failed to append normalized output row: {error}"))?;
        }

        let table = table_class
            .call_method1("from_pylist", (normalized_rows,))
            .map_err(|error| format!("failed to convert output rows to Arrow table: {error}"))?;
        let stream = table
            .call_method0("__arrow_c_stream__")
            .map_err(|error| format!("failed to export output Arrow C stream: {error}"))?;
        output_streams
            .set_item(target_any, stream)
            .map_err(|error| format!("failed to set output stream: {error}"))?;
    }

    Ok(output_streams.unbind())
}

#[pyfunction]
fn execute_block(py: Python<'_>, payload: &Bound<'_, PyDict>) -> PyResult<Py<PyDict>> {
    let result = PyDict::new(py);
    let diagnostics = PyList::empty(py);
    let ast_json = payload
        .get_item("ast_json")?
        .ok_or_else(|| PyValueError::new_err("payload.ast_json is required"))?
        .extract::<String>()?;
    let runtime_statements = match serde_json::from_str::<AstPayload>(&ast_json) {
        Ok(parsed_ast) => parsed_ast.statements,
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
        "ASSIGN", "SUM", "DELETE", "RETAIN", "ARRAY", "BY", "STOP",
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
    let output_targets = output_targets_any.downcast::<PyList>()?;

    let targets = if output_targets_from_stmt.is_empty() {
        output_targets
            .iter()
            .map(|item| item.extract::<String>())
            .collect::<PyResult<Vec<String>>>()?
    } else {
        output_targets_from_stmt
    };

    let input_streams: Option<Py<PyDict>> = if let Some(input_streams_any) = payload.get_item("input_streams")? {
        Some(
            input_streams_any
                .downcast::<PyDict>()
                .map_err(|_| PyValueError::new_err("payload.input_streams must be a dict"))?
                .clone()
                .unbind(),
        )
    } else {
        None
    };

    let legacy_inputs: Option<Py<PyDict>> = if let Some(inputs_any) = payload.get_item("inputs")? {
        Some(inputs_any.downcast::<PyDict>()?.clone().unbind())
    } else {
        None
    };

    let outputs = PyDict::new(py);
    for target in &targets {
        outputs.set_item(target, PyList::empty(py))?;
    }

    let has_explicit_output_statement = executable.iter().any(has_inline_output_action);
    let default_target = targets
        .first()
        .cloned()
        .unwrap_or_else(|| "out".to_string());

    let mut eval_state = EvalRuntimeState::default();
    let mut global_row_index: i64 = 0;
    let mut stop_execution = false;

    for input_name in set_input_names {
        let rows_list = if let Some(streams) = input_streams.as_ref() {
            let streams = streams.bind(py);
            if let Some(stream_any) = streams.get_item(&input_name)? {
                match read_rows_from_arrow_stream(py, &stream_any) {
                    Ok(rows) => rows,
                    Err(error) => {
                        diagnostics.append(diag(
                            py,
                            "RUNTIME_RUST_BRIDGE_ARROW_EXPORT_FAILED",
                            &format!("dataset:{}", input_name),
                            &format!("Rust native runtime failed to read Arrow C stream: {}", error),
                        )?)?;
                        result.set_item("outputs", PyDict::new(py))?;
                        result.set_item("diagnostics", diagnostics)?;
                        return Ok(result.unbind());
                    }
                }
            } else if let Some(inputs) = legacy_inputs.as_ref() {
                let inputs = inputs.bind(py);
                let rows_any = match inputs.get_item(&input_name)? {
                    Some(value) => value,
                    None => continue,
                };
                let rows = rows_any.downcast::<PyList>()?;
                rows.clone().unbind()
            } else {
                continue
            }
        } else if let Some(inputs) = legacy_inputs.as_ref() {
            let inputs = inputs.bind(py);
            let rows_any = match inputs.get_item(&input_name)? {
                Some(value) => value,
                None => continue,
            };
            let rows = rows_any.downcast::<PyList>()?;
            rows.clone().unbind()
        } else {
            diagnostics.append(diag(
                py,
                "RUNTIME_RUST_BRIDGE_ARROW_EXPORT_MISSING",
                &format!("dataset:{}", input_name),
                "Rust native runtime requires input_streams for execution.",
            )?)?;
            result.set_item("outputs", PyDict::new(py))?;
            result.set_item("diagnostics", diagnostics)?;
            return Ok(result.unbind());
        };
        let rows = rows_list.bind(py);
        eval_state.row_view = rows
            .iter()
            .map(|row_any| {
                row_any
                    .downcast::<PyDict>()
                    .map_err(PyErr::from)
                    .map(|row| row.clone().unbind())
            })
            .collect::<PyResult<Vec<_>>>()?;
        let row_count = eval_state.row_view.len();

        for row_index in 0..row_count {
            eval_state.row_index = row_index as isize;
            let row_obj = eval_state.row_view[row_index].clone_ref(py);
            let row = row_obj.bind(py);
            let set_options = set_options_by_dataset
                .get(&input_name.to_lowercase())
                .cloned()
                .unwrap_or_default();
            let Some(working_row_obj) = apply_dataset_ref_options_to_row(py, row, &set_options, &mut eval_state)
                .map_err(PyValueError::new_err)?
            else {
                continue;
            };
            let working_row = working_row_obj.bind(py);

            global_row_index += 1;
            working_row.set_item("_N_", global_row_index)?;
            working_row.set_item("_ERROR_", 0)?;
            working_row.set_item("_n_", global_row_index)?;
            working_row.set_item("_error_", 0)?;

            for in_var in &all_in_vars {
                working_row.set_item(in_var, false)?;
            }
            if let Some(in_var_name) = set_in_var_by_dataset.get(&input_name.to_lowercase()) {
                working_row.set_item(in_var_name, true)?;
            }
            if let Some(name) = &indsname_var {
                working_row.set_item(name, input_name.clone())?;
            }
            if let Some(name) = &end_var {
                working_row.set_item(name, row_index + 1 == row_count)?;
            }

            for retained_name in &retain_vars {
                if get_row_item(&working_row, retained_name)
                    .map_err(PyValueError::new_err)?
                    .is_none()
                {
                    if let Some(value) = eval_state.retain_values.get(retained_name).cloned() {
                        set_row_scalar(&working_row, retained_name, &value).ok();
                    }
                }
            }

            if !by_vars.is_empty() {
                let prev_index = if row_index > 0 { Some(row_index - 1) } else { None };
                let next_index = if row_index + 1 < row_count { Some(row_index + 1) } else { None };
                for by_var in &by_vars {
                    let current_value = get_row_item(&working_row, by_var)
                        .map_err(PyValueError::new_err)?;
                    let is_first = if let Some(prev_idx) = prev_index {
                        let prev_row = eval_state.row_view[prev_idx].bind(py);
                        let prev_value = get_row_item(&prev_row, by_var)
                            .map_err(PyValueError::new_err)?;
                        !optional_pyvalues_equal(current_value.as_ref(), prev_value.as_ref())
                            .map_err(PyValueError::new_err)?
                    } else {
                        true
                    };
                    let is_last = if let Some(next_idx) = next_index {
                        let next_row = eval_state.row_view[next_idx].bind(py);
                        let next_value = get_row_item(&next_row, by_var)
                            .map_err(PyValueError::new_err)?;
                        !optional_pyvalues_equal(current_value.as_ref(), next_value.as_ref())
                            .map_err(PyValueError::new_err)?
                    } else {
                        true
                    };
                    working_row.set_item(format!("first.{}", by_var), is_first)?;
                    working_row.set_item(format!("FIRST.{}", by_var), is_first)?;
                    working_row.set_item(format!("last.{}", by_var), is_last)?;
                    working_row.set_item(format!("LAST.{}", by_var), is_last)?;
                }
            }

            if let Some(expression) = &where_expr {
                let pass = match evaluate_simple_where(&working_row, expression, &mut eval_state) {
                    Ok(value) => value,
                    Err(error) => {
                        working_row.set_item("_ERROR_", 1).ok();
                        diagnostics.append(diag(
                            py,
                            "RUNTIME_EXPRESSION_EVALUATION_ERROR",
                            "where",
                            &format!("Rust WHERE evaluation failed: {}", error),
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

            let (_, outcome) = match execute_statement_block(
                py,
                &executable,
                0,
                executable.len(),
                &working_row,
                &mut eval_state,
            ) {
                Ok(value) => value,
                Err(error) => {
                    working_row.set_item("_ERROR_", 1).ok();
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
                if let Some(value_any) = get_row_item(&working_row, retained_name)
                    .map_err(PyValueError::new_err)?
                {
                    if let Ok(value) = py_to_scalar(&value_any) {
                        eval_state.retain_values.insert(retained_name.clone(), value);
                    }
                }
            }

            let mut emitted_rows: Vec<(String, Py<PyDict>)> = Vec::new();
            if outcome.emitted_rows.is_empty() {
                if !outcome.deleted && !has_explicit_output_statement {
                    let cloned = match clone_row_dict(py, &working_row) {
                        Ok(value) => value,
                        Err(error) => {
                            diagnostics.append(diag(
                                py,
                                "RUNTIME_EXPRESSION_EVALUATION_ERROR",
                                "output",
                                &format!("Rust output row clone failed: {}", error),
                            )?)?;
                            result.set_item("outputs", PyDict::new(py))?;
                            result.set_item("diagnostics", diagnostics)?;
                            return Ok(result.unbind());
                        }
                    };
                    emitted_rows.push((default_target.clone(), cloned));
                }
            } else {
                for (target_opt, emitted_row) in &outcome.emitted_rows {
                    let target = target_opt.clone().unwrap_or_else(|| default_target.clone());
                    emitted_rows.push((target, emitted_row.clone_ref(py)));
                }
            }

            for (target, emitted_row) in emitted_rows {
                let resolved_target = resolve_declared_target_name(&target, &targets);
                if !targets.contains(&resolved_target) {
                    diagnostics.append(diag(
                        py,
                        "RUNTIME_OUTPUT_TARGET_NOT_FOUND",
                        "output",
                        &format!("Output target is not declared: {}", target),
                    )?)?;
                    result.set_item("outputs", PyDict::new(py))?;
                    result.set_item("diagnostics", diagnostics)?;
                    return Ok(result.unbind());
                }
                let mut row_to_write = emitted_row.clone_ref(py);
                if let Some(option_spec) = output_options_by_target.get(&resolved_target.to_lowercase()) {
                    let emitted_bound = row_to_write.bind(py);
                    let projected = apply_dataset_ref_options_to_row(py, emitted_bound, option_spec, &mut eval_state)
                        .map_err(PyValueError::new_err)?;
                    let Some(projected_row) = projected else {
                        continue;
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
                if let Err(error) = append_projected_row_to_target(py, &outputs, &resolved_target, emitted_bound, &keep_vars, &drop_vars) {
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

    let output_streams = match export_output_streams(py, &outputs) {
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

    result.set_item("outputs", outputs)?;
    result.set_item("output_streams", output_streams)?;
    result.set_item("diagnostics", diagnostics)?;
    Ok(result.unbind())
}

#[pyfunction]
fn render_diagnostics_ariadne(payload: &Bound<'_, PyDict>) -> PyResult<String> {
    let render_payload = extract_render_payload(payload)?;
    let mut rendered: Vec<String> = Vec::new();
    for diagnostic in &render_payload.diagnostics {
        match render_ariadne_diagnostic(diagnostic, &render_payload) {
            Ok(output) => rendered.push(output),
            Err(_) => rendered.push(render_simple_diagnostic(diagnostic)),
        }
    }
    Ok(rendered.join("\n"))
}

#[pymodule]
fn limulus_native(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(execute_block, module)?)?;
    module.add_function(wrap_pyfunction!(render_diagnostics_ariadne, module)?)?;
    Ok(())
}
