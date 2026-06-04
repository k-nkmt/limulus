use crate::runtime::EvalRuntimeState;
use crate::value_ref::{ResolvedValue, SourceValueRef};
use chrono::{Datelike, NaiveDate, NaiveDateTime, NaiveTime, Timelike};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyTuple};
use regex::Regex;
use std::collections::HashSet;

mod functions;
pub use functions::evaluate_function;

#[derive(Debug, Clone)]
pub(crate) enum Expr {
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
pub(crate) enum BinaryOp {
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

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ScalarValue {
    Number(f64),
    Decimal(String),
    Text(String),
    Bool(bool),
    Date(chrono::NaiveDate),
    DateTime(chrono::NaiveDateTime),
    Time(chrono::NaiveTime),
    List(Vec<ScalarValue>),
    Struct(Vec<(String, ScalarValue)>),
    Null,
}

#[derive(Debug, Clone)]
pub(crate) enum Token {
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

pub(crate) struct TokenCursor {
    tokens: Vec<Token>,
    position: usize,
}

impl TokenCursor {
    pub(crate) fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, position: 0 }
    }

    pub(crate) fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.position)
    }

    pub(crate) fn next(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.position).cloned();
        if token.is_some() {
            self.position += 1;
        }
        token
    }

    pub(crate) fn expect_rparen(&mut self) -> Result<(), String> {
        match self.next() {
            Some(Token::RParen) => Ok(()),
            _ => Err("expected ')'".to_string()),
        }
    }
}

pub(crate) fn column_key(name: &str) -> String {
    name.trim().to_uppercase()
}

pub(crate) fn resolve_row_key(row: &Bound<'_, PyDict>, name: &str) -> Result<Option<String>, String> {
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

pub(crate) fn get_row_item<'py>(row: &Bound<'py, PyDict>, name: &str) -> Result<Option<Bound<'py, PyAny>>, String> {
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

pub(crate) fn get_stateful_row_item<'py>(
    row: &Bound<'py, PyDict>,
    name: &str,
    state: &EvalRuntimeState,
) -> Result<Option<Bound<'py, PyAny>>, String> {
    let Some(value) = get_stateful_scalar_value(row, name, state)? else {
        return Ok(None);
    };
    let value_any = scalar_to_pyobject(row.py(), &value)?;
    Ok(Some(value_any.bind(row.py()).clone()))
}

pub(crate) fn resolve_value<'a>(
    row: &Bound<'_, PyDict>,
    name: &str,
    state: &'a EvalRuntimeState,
) -> Result<Option<ResolvedValue<'a>>, String> {
    let _ = row;
    let normalized_name = column_key(name);

    if let Some(value) = state.automatic_values.get(&normalized_name) {
        return Ok(Some(ResolvedValue::Owned(value)));
    }

    if let Some(value) = state.mutable_values.get(&normalized_name) {
        return Ok(Some(ResolvedValue::Owned(value)));
    }

    let Some(cursor) = state.source_cursor.as_ref() else {
        return Ok(None);
    };
    if state.row_index < 0 {
        return Ok(None);
    }

    cursor
        .source_value_by_normalized_name(state.row_index as usize, &normalized_name)
        .map(|value| value.map(ResolvedValue::Source))
}

pub(crate) fn get_stateful_scalar_value(
    row: &Bound<'_, PyDict>,
    name: &str,
    state: &EvalRuntimeState,
) -> Result<Option<ScalarValue>, String> {
    if !state.materialize_mutable_values {
        return resolve_value(row, name, state).map(|value| {
            value.map(|resolved| resolved.into_owned_value().into_scalar_value())
        });
    }

    if let Some(value) = resolve_value(row, name, state)? {
        return Ok(Some(value.into_owned_value().into_scalar_value()));
    }

    if let Some(value) = get_row_item(row, name)? {
        return py_to_scalar(&value)
            .map(Some)
            .map_err(|error| error.to_string());
    }

    Ok(None)
}

pub(crate) fn scalar_to_pyobject(py: Python<'_>, value: &ScalarValue) -> Result<Py<PyAny>, String> {
    let datetime_module = py.import("datetime").map_err(|error| error.to_string())?;
    match value {
        ScalarValue::Number(number) => Ok(number.into_pyobject(py).map_err(|error| error.to_string())?.unbind().into_any()),
        ScalarValue::Decimal(decimal) => decimal_pyobject(py, decimal),
        ScalarValue::Text(text) => Ok(text.clone().into_pyobject(py).map_err(|error| error.to_string())?.unbind().into_any()),
        ScalarValue::Bool(boolean) => Ok(boolean
            .into_pyobject(py)
            .map_err(|error| error.to_string())?
            .to_owned()
            .into_any()
            .unbind()),
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
        ScalarValue::List(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(scalar_to_pyobject(py, item)?.bind(py))
                    .map_err(|error| error.to_string())?;
            }
            Ok(list.unbind().into_any())
        }
        ScalarValue::Struct(fields) => {
            let mapping = PyDict::new(py);
            for (name, item) in fields {
                mapping
                    .set_item(name, scalar_to_pyobject(py, item)?.bind(py))
                    .map_err(|error| error.to_string())?;
            }
            Ok(mapping.unbind().into_any())
        }
        ScalarValue::Null => Ok(py.None()),
    }
}

pub(crate) fn is_expression_parse_error(error: &str) -> bool {
    matches!(
        error,
        "expression expected" | "unexpected token at end of expression" | "unterminated string literal"
    ) || error.starts_with("unexpected token:")
}

pub(crate) fn condition_syntax_error_message(
    py: Python<'_>,
    expression: &str,
    filename: &str,
) -> Result<Option<String>, String> {
    let builtins = py
        .import("builtins")
        .map_err(|error| format!("condition syntax import failed: {error}"))?;
    let compile_fn = builtins
        .getattr("compile")
        .map_err(|error| format!("condition syntax compile lookup failed: {error}"))?;
    match compile_fn.call1((expression, filename, "eval")) {
        Ok(_) => Ok(None),
        Err(error) => {
            let message = error.to_string();
            Ok(Some(
                message
                    .strip_prefix("SyntaxError: ")
                    .unwrap_or(&message)
                    .to_string(),
            ))
        }
    }
}

pub(crate) fn resolve_array_variable_name(
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
        return Err(format!("array index out of bounds: {array_name}[{index}]"));
    }
    Ok(vars[index as usize - 1].clone())
}

fn build_case_insensitive_scope<'py>(
    py: Python<'py>,
    row: &Bound<'py, PyDict>,
    state: &EvalRuntimeState,
    names: &[String],
) -> Result<Bound<'py, PyDict>, String> {
    let locals = PyDict::new(py);
    for name in names {
        let Some(value) = get_stateful_row_item(row, name, state)? else {
            continue;
        };

        locals
            .set_item(name, &value)
            .map_err(|error| format!("case-insensitive scope set failed: {error}"))?;

        let normalized = name.trim().to_uppercase();
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

fn fallback_identifier_names(expr_text: &str) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let pattern = Regex::new(r"\$?[A-Za-z_][A-Za-z0-9_\.]*").expect("identifier regex must compile");
    for matched in pattern.find_iter(expr_text) {
        let candidate = matched.as_str();
        let normalized = column_key(candidate);
        if seen.insert(normalized) {
            names.push(candidate.to_string());
        }
    }
    names
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

        if ch.is_ascii_alphabetic() || ch == '_' || ch == '$' {
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

pub(crate) fn py_to_scalar(value: &Bound<'_, PyAny>) -> PyResult<ScalarValue> {
    convert_py_to_scalar(value)
}

pub(crate) fn scalar_to_number(value: &ScalarValue) -> Result<f64, String> {
    coerce_scalar_to_number(value)
}

pub(crate) fn scalar_to_bool(value: &ScalarValue) -> Result<bool, String> {
    coerce_scalar_to_bool(value)
}

fn decimal_pyobject(py: Python<'_>, decimal: &str) -> Result<Py<PyAny>, String> {
    py.import("decimal")
        .map_err(|error| error.to_string())?
        .getattr("Decimal")
        .map_err(|error| error.to_string())?
        .call1((decimal,))
        .map_err(|error| error.to_string())
        .map(|value| value.unbind())
}

fn convert_py_to_scalar(value: &Bound<'_, PyAny>) -> PyResult<ScalarValue> {
    if value.is_none() {
        return Ok(ScalarValue::Null);
    }
    if let Some(temporal) = py_temporal_to_scalar(value)? {
        return Ok(temporal);
    }
    if py_decimal_to_scalar(value)? {
        return Ok(ScalarValue::Decimal(value.str()?.to_string()));
    }
    if let Ok(items) = value.cast::<PyList>() {
        return py_sequence_to_scalar_list(items.iter());
    }
    if let Ok(items) = value.cast::<PyTuple>() {
        return py_sequence_to_scalar_list(items.iter());
    }
    if let Ok(mapping) = value.cast::<PyDict>() {
        let mut fields = Vec::with_capacity(mapping.len());
        for (key, item) in mapping.iter() {
            let name = key
                .extract::<String>()
                .or_else(|_| key.str().map(|rendered| rendered.to_string()))?;
            fields.push((name, convert_py_to_scalar(&item)?));
        }
        return Ok(ScalarValue::Struct(fields));
    }
    if let Ok(boolean) = value.extract::<bool>() {
        return Ok(ScalarValue::Bool(boolean));
    }
    if let Ok(number) = value.extract::<f64>() {
        return Ok(ScalarValue::Number(number));
    }
    Ok(ScalarValue::Text(value.str()?.to_string()))
}

fn py_decimal_to_scalar(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    let decimal_module = value.py().import("decimal")?;
    let decimal_type = decimal_module.getattr("Decimal")?;
    value.is_instance(&decimal_type)
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

fn py_sequence_to_scalar_list<'py, I>(items: I) -> PyResult<ScalarValue>
where
    I: IntoIterator<Item = Bound<'py, PyAny>>,
{
    let mut values = Vec::new();
    for item in items {
        values.push(convert_py_to_scalar(&item)?);
    }
    Ok(ScalarValue::List(values))
}

fn coerce_scalar_to_number(value: &ScalarValue) -> Result<f64, String> {
    match value {
        ScalarValue::Number(number) => Ok(*number),
        ScalarValue::Decimal(decimal) => decimal
            .parse::<f64>()
            .map_err(|_| format!("numeric value required, got '{decimal}'")),
        ScalarValue::Bool(boolean) => Ok(if *boolean { 1.0 } else { 0.0 }),
        ScalarValue::Text(text) => text
            .parse::<f64>()
            .map_err(|_| format!("numeric value required, got '{text}'")),
        ScalarValue::Date(_) => Err("date value cannot be converted to number".to_string()),
        ScalarValue::DateTime(_) => Err("datetime value cannot be converted to number".to_string()),
        ScalarValue::Time(_) => Err("time value cannot be converted to number".to_string()),
        ScalarValue::List(_) => Err("list value cannot be converted to number".to_string()),
        ScalarValue::Struct(_) => Err("struct value cannot be converted to number".to_string()),
        ScalarValue::Null => Ok(0.0),
    }
}

pub(crate) fn scalar_to_text(value: &ScalarValue) -> String {
    match value {
        ScalarValue::Text(text) => text.clone(),
        ScalarValue::Number(number) => number.to_string(),
        ScalarValue::Decimal(decimal) => decimal.clone(),
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
        ScalarValue::List(values) => {
            let rendered: Vec<String> = values.iter().map(scalar_to_text).collect();
            format!("[{}]", rendered.join(", "))
        }
        ScalarValue::Struct(fields) => {
            let rendered: Vec<String> = fields
                .iter()
                .map(|(name, item)| format!("{name}: {}", scalar_to_text(item)))
                .collect();
            format!("{{{}}}", rendered.join(", "))
        }
        ScalarValue::Null => "".to_string(),
    }
}

fn scalar_to_date(value: &ScalarValue) -> Result<NaiveDate, String> {
    match value {
        ScalarValue::Date(date) => Ok(*date),
        ScalarValue::DateTime(datetime) => Ok(datetime.date()),
        ScalarValue::Text(text) => NaiveDate::parse_from_str(text, "%Y-%m-%d")
            .map_err(|_| format!("date value required, got '{text}'")),
        ScalarValue::List(_) => Err("date value required, got list".to_string()),
        ScalarValue::Struct(_) => Err("date value required, got struct".to_string()),
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
        ScalarValue::List(_) => Err("datetime value required, got list".to_string()),
        ScalarValue::Struct(_) => Err("datetime value required, got struct".to_string()),
        _ => Err("datetime value required".to_string()),
    }
}

fn scalar_to_time(value: &ScalarValue) -> Result<NaiveTime, String> {
    match value {
        ScalarValue::Time(time) => Ok(*time),
        ScalarValue::DateTime(datetime) => Ok(datetime.time()),
        ScalarValue::Text(text) => parse_time_text(text),
        ScalarValue::List(_) => Err("time value required, got list".to_string()),
        ScalarValue::Struct(_) => Err("time value required, got struct".to_string()),
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

fn coerce_scalar_to_bool(value: &ScalarValue) -> Result<bool, String> {
    match value {
        ScalarValue::Bool(boolean) => Ok(*boolean),
        ScalarValue::Number(number) => Ok(number.abs() >= f64::EPSILON),
        ScalarValue::Decimal(decimal) => Ok(decimal.parse::<f64>().unwrap_or(0.0).abs() >= f64::EPSILON),
        ScalarValue::Text(text) => Ok(!text.is_empty()),
        ScalarValue::Date(_) => Ok(true),
        ScalarValue::DateTime(_) => Ok(true),
        ScalarValue::Time(_) => Ok(true),
        ScalarValue::List(values) => Ok(!values.is_empty()),
        ScalarValue::Struct(fields) => Ok(!fields.is_empty()),
        ScalarValue::Null => Ok(false),
    }
}

fn scalar_eq(left: &ScalarValue, right: &ScalarValue) -> Result<bool, String> {
    match (left, right) {
        (ScalarValue::Null, ScalarValue::Null) => Ok(true),
        (ScalarValue::Number(_), _)
        | (_, ScalarValue::Number(_))
        | (ScalarValue::Decimal(_), _)
        | (_, ScalarValue::Decimal(_))
        | (ScalarValue::Bool(_), _)
        | (_, ScalarValue::Bool(_)) => {
            let l = scalar_to_number(left)?;
            let r = scalar_to_number(right)?;
            Ok((l - r).abs() < f64::EPSILON)
        }
        (ScalarValue::Text(l), ScalarValue::Text(r)) => Ok(l == r),
        (ScalarValue::Date(l), ScalarValue::Date(r)) => Ok(l == r),
        (ScalarValue::DateTime(l), ScalarValue::DateTime(r)) => Ok(l == r),
        (ScalarValue::Time(l), ScalarValue::Time(r)) => Ok(l == r),
        (ScalarValue::List(l), ScalarValue::List(r)) => Ok(l == r),
        (ScalarValue::Struct(l), ScalarValue::Struct(r)) => Ok(l == r),
        _ => Ok(false),
    }
}

pub(crate) fn optional_source_values_equal(
    left: Option<SourceValueRef<'_>>,
    right: Option<SourceValueRef<'_>>,
) -> Result<bool, String> {
    match (left, right) {
        (Some(lhs), Some(rhs)) => scalar_eq(lhs.as_scalar(), rhs.as_scalar()),
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
        (ScalarValue::List(_), ScalarValue::List(_)) | (ScalarValue::Struct(_), ScalarValue::Struct(_)) => {
            Ok(scalar_to_text(left).cmp(&scalar_to_text(right)))
        }
        _ => {
            let l = scalar_to_number(left)?;
            let r = scalar_to_number(right)?;
            l.partial_cmp(&r)
                .ok_or_else(|| "comparison failed".to_string())
        }
    }
}


pub(crate) fn evaluate_scalar_expression(
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

            let locals = build_case_insensitive_scope(py, row, state, &fallback_identifier_names(expr_text))?;

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

fn evaluate_expression(
    row: &Bound<'_, PyDict>,
    expr: &Expr,
    state: &mut EvalRuntimeState,
) -> Result<ScalarValue, String> {
    eval_expression_tree(row, expr, state)
}

enum EvalOperand<'a> {
    Resolved(ResolvedValue<'a>),
    Scalar(ScalarValue),
}

impl<'a> EvalOperand<'a> {
    fn as_scalar(&self) -> &ScalarValue {
        match self {
            Self::Resolved(value) => value.as_scalar(),
            Self::Scalar(value) => value,
        }
    }
}

fn resolve_simple_expression_value<'a>(
    row: &Bound<'_, PyDict>,
    expr: &Expr,
    state: &'a EvalRuntimeState,
) -> Result<Option<EvalOperand<'a>>, String> {
    match expr {
        Expr::Literal(value) => Ok(Some(EvalOperand::Scalar(value.clone()))),
        Expr::Variable(name) => {
            if let Some(value) = resolve_value(row, name, state)? {
                return Ok(Some(EvalOperand::Resolved(value)));
            }
            if state.materialize_mutable_values {
                if let Some(value) = get_row_item(row, name)? {
                    let scalar = py_to_scalar(&value).map_err(|error| error.to_string())?;
                    return Ok(Some(EvalOperand::Scalar(scalar)));
                }
            }
            if name.ends_with('.') {
                return Ok(Some(EvalOperand::Scalar(ScalarValue::Text(name.to_string()))));
            }
            Ok(Some(EvalOperand::Scalar(ScalarValue::Null)))
        }
        _ => Ok(None),
    }
}

fn operand_to_number(value: &EvalOperand<'_>) -> Result<f64, String> {
    coerce_scalar_to_number(value.as_scalar())
}

fn operand_to_bool(value: &EvalOperand<'_>) -> Result<bool, String> {
    coerce_scalar_to_bool(value.as_scalar())
}

fn operand_eq(left: &EvalOperand<'_>, right: &EvalOperand<'_>) -> Result<bool, String> {
    scalar_eq(left.as_scalar(), right.as_scalar())
}

fn operand_compare(left: &EvalOperand<'_>, right: &EvalOperand<'_>) -> Result<std::cmp::Ordering, String> {
    scalar_compare(left.as_scalar(), right.as_scalar())
}

fn eval_binary_operands(
    left_value: &EvalOperand<'_>,
    op: &BinaryOp,
    right_value: &EvalOperand<'_>,
) -> Result<ScalarValue, String> {
    match op {
        BinaryOp::Add => Ok(ScalarValue::Number(operand_to_number(left_value)? + operand_to_number(right_value)?)),
        BinaryOp::Sub => Ok(ScalarValue::Number(operand_to_number(left_value)? - operand_to_number(right_value)?)),
        BinaryOp::Mul => Ok(ScalarValue::Number(operand_to_number(left_value)? * operand_to_number(right_value)?)),
        BinaryOp::Div => Ok(ScalarValue::Number(operand_to_number(left_value)? / operand_to_number(right_value)?)),
        BinaryOp::Pow => Ok(ScalarValue::Number(operand_to_number(left_value)?.powf(operand_to_number(right_value)?))),
        BinaryOp::Mod => Ok(ScalarValue::Number(operand_to_number(left_value)? % operand_to_number(right_value)?)),
        BinaryOp::And => Ok(ScalarValue::Bool(operand_to_bool(left_value)? && operand_to_bool(right_value)?)),
        BinaryOp::Or => Ok(ScalarValue::Bool(operand_to_bool(left_value)? || operand_to_bool(right_value)?)),
        BinaryOp::Eq => Ok(ScalarValue::Bool(operand_eq(left_value, right_value)?)),
        BinaryOp::Ne => Ok(ScalarValue::Bool(!operand_eq(left_value, right_value)?)),
        BinaryOp::Gt => Ok(ScalarValue::Bool(operand_compare(left_value, right_value)? == std::cmp::Ordering::Greater)),
        BinaryOp::Ge => Ok(ScalarValue::Bool(matches!(
            operand_compare(left_value, right_value)?,
            std::cmp::Ordering::Greater | std::cmp::Ordering::Equal
        ))),
        BinaryOp::Lt => Ok(ScalarValue::Bool(operand_compare(left_value, right_value)? == std::cmp::Ordering::Less)),
        BinaryOp::Le => Ok(ScalarValue::Bool(matches!(
            operand_compare(left_value, right_value)?,
            std::cmp::Ordering::Less | std::cmp::Ordering::Equal
        ))),
    }
}

fn eval_expression_tree(row: &Bound<'_, PyDict>, expr: &Expr, state: &mut EvalRuntimeState) -> Result<ScalarValue, String> {
    match expr {
        Expr::Literal(value) => Ok(value.clone()),
        Expr::Variable(name) => {
            let Some(value) = get_stateful_scalar_value(row, name, state)? else {
                if name.ends_with('.') {
                    return Ok(ScalarValue::Text(name.to_string()));
                }
                return Ok(ScalarValue::Null);
            };
            Ok(value)
        }
        Expr::ArrayRef { name, index } => {
            let index_value = scalar_to_number(&eval_expression_tree(row, index, state)?)?;
            let variable_name = resolve_array_variable_name(state, name, index_value)?;
            let Some(value) = get_stateful_scalar_value(row, &variable_name, state)? else {
                return Ok(ScalarValue::Null);
            };
            Ok(value)
        }
        Expr::UnaryNot(inner) => {
            if let Some(value) = resolve_simple_expression_value(row, inner, &*state)? {
                return Ok(ScalarValue::Bool(!operand_to_bool(&value)?));
            }
            Ok(ScalarValue::Bool(!scalar_to_bool(&eval_expression_tree(row, inner, state)?)?))
        }
        Expr::UnaryNeg(inner) => {
            if let Some(value) = resolve_simple_expression_value(row, inner, &*state)? {
                return Ok(ScalarValue::Number(-operand_to_number(&value)?));
            }
            Ok(ScalarValue::Number(-scalar_to_number(&eval_expression_tree(row, inner, state)?)?))
        }
        Expr::Binary { left, op, right } => {
            if let (Some(left_value), Some(right_value)) = (
                resolve_simple_expression_value(row, left, &*state)?,
                resolve_simple_expression_value(row, right, &*state)?,
            ) {
                return eval_binary_operands(&left_value, op, &right_value);
            }

            let left_value = EvalOperand::Scalar(eval_expression_tree(row, left, state)?);
            let right_value = EvalOperand::Scalar(eval_expression_tree(row, right, state)?);
            eval_binary_operands(&left_value, op, &right_value)
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
                let index_value = scalar_to_number(&eval_expression_tree(row, &index_expr, state)?)?;
                let variable_name = resolve_array_variable_name(state, &array_name, index_value)?;
                return Ok(ScalarValue::Text(variable_name));
            }

            if state.array_defs.contains_key(name) && args.len() == 1 {
                let index_value = scalar_to_number(&eval_expression_tree(row, &args[0], state)?)?;
                let variable_name = resolve_array_variable_name(state, name, index_value)?;
                let Some(value) = get_stateful_scalar_value(row, &variable_name, state)? else {
                    return Ok(ScalarValue::Null);
                };
                return Ok(value);
            }

            let evaluated_args = args
                .iter()
                .map(|arg| eval_expression_tree(row, arg, state))
                .collect::<Result<Vec<_>, _>>()?;
            evaluate_function(row.py(), name, &evaluated_args, state)
        }
    }
}

pub(crate) fn evaluate_simple_where(
    row: &Bound<'_, PyDict>,
    expr: &str,
    state: &mut EvalRuntimeState,
) -> Result<bool, String> {
    let parsed = parse_expression_cached(expr, state)?;
    let evaluated = evaluate_expression(row, &parsed, state)?;
    scalar_to_bool(&evaluated)
}
