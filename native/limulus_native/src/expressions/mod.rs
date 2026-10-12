use crate::ast::{
    AstWhereComparisonOperator,
    AstWhereExpression,
    AstWhereLogicalOperator,
    AstWhereScalar,
};
use crate::runtime::EvalRuntimeState;
use crate::row_view::LogicalRowAccess;
use crate::value_ref::{ResolvedValue, SourceValueRef};
use arrow::datatypes::TimeUnit;
use chrono::{Datelike, NaiveDate, NaiveDateTime, NaiveTime, Timelike};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyTuple};

mod functions;
pub use functions::evaluate_function;
use functions::evaluate_function_with_context;

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
    DeferredError {
        cause: String,
    },
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
    Integer(i64),
    Unsigned(u64),
    Number(f64),
    RawTimestamp {
        value: i64,
        unit: TimeUnit,
        timezone: Option<String>,
    },
    RawTime {
        value: i64,
        unit: TimeUnit,
    },
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

impl ScalarValue {
    pub(crate) fn as_integer(&self) -> Option<i64> {
        match self {
            Self::Integer(value) => Some(*value),
            _ => None,
        }
    }

    pub(crate) fn as_unsigned(&self) -> Option<u64> {
        match self {
            Self::Unsigned(value) => Some(*value),
            _ => None,
        }
    }

    pub(crate) fn as_raw_timestamp(&self) -> Option<(i64, TimeUnit, Option<&str>)> {
        match self {
            Self::RawTimestamp {
                value,
                unit,
                timezone,
            } => Some((*value, *unit, timezone.as_deref())),
            _ => None,
        }
    }

    pub(crate) fn as_raw_time(&self) -> Option<(i64, TimeUnit)> {
        match self {
            Self::RawTime { value, unit } => Some((*value, *unit)),
            _ => None,
        }
    }
}

#[cfg(test)]
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
    Integer(i64),
    Unsigned(u64),
    Number(f64),
    Text(String),
    Identifier(String),
    Missing,
}

#[cfg(test)]
struct TokenCursor {
    tokens: Vec<Token>,
    position: usize,
}

#[cfg(test)]
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

pub(crate) fn resolve_value<'a>(
    row: &Bound<'_, PyDict>,
    name: &str,
    state: &'a EvalRuntimeState,
) -> Result<Option<ResolvedValue<'a>>, String> {
    let _ = row;
    let logical_row = state
        .logical_row
        .as_ref()
        .ok_or_else(|| "logical row view is required for native value resolution".to_string())?;
    logical_row
        .resolve(name)
        .map_err(|error| error.to_string())
        .map(|value| {
            value.map(|resolved| match resolved {
                crate::row_view::ResolvedValue::SourceScalar(value_ref) => {
                    ResolvedValue::Source(value_ref)
                }
                crate::row_view::ResolvedValue::SourceCell(cell) => {
                    ResolvedValue::Source(SourceValueRef::Cell(cell))
                }
                crate::row_view::ResolvedValue::Owned(value)
                | crate::row_view::ResolvedValue::Automatic(value) => ResolvedValue::Owned(value),
            })
        })
}

pub(crate) fn get_stateful_scalar_value(
    row: &Bound<'_, PyDict>,
    name: &str,
    state: &EvalRuntimeState,
) -> Result<Option<ScalarValue>, String> {
    let _ = row;
    let Some(value) = resolve_value(row, name, state)? else {
        return Ok(None);
    };
    let scalar = value
        .into_owned_value()
        .map_err(|error| {
            format!(
                "complex access is unsupported for column '{name}' during expression: {error}"
            )
        })?
        .into_scalar_value()
        .map_err(|error| {
            format!(
                "complex access is unsupported for column '{name}' during expression: {error}"
            )
        })?;
    if matches!(scalar, ScalarValue::Null) {
        Ok(None)
    } else {
        Ok(Some(scalar))
    }
}

pub(crate) fn scalar_to_pyobject(py: Python<'_>, value: &ScalarValue) -> Result<Py<PyAny>, String> {
    let datetime_module = py.import("datetime").map_err(|error| error.to_string())?;
    match value {
        ScalarValue::Integer(integer) => Ok(integer
            .into_pyobject(py)
            .map_err(|error| error.to_string())?
            .unbind()
            .into_any()),
        ScalarValue::Unsigned(unsigned) => Ok(unsigned
            .into_pyobject(py)
            .map_err(|error| error.to_string())?
            .unbind()
            .into_any()),
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
        ScalarValue::RawTimestamp { .. } | ScalarValue::RawTime { .. } => {
            Err(raw_temporal_conversion_error(value, "a Python object"))
        }
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

fn python_boundary_argument_context(expr: &Expr) -> String {
    match expr {
        Expr::Variable(name) => name.trim().to_string(),
        Expr::ArrayRef { name, .. } => name.trim().to_string(),
        Expr::FunctionCall { name, .. } => format!("{name}() result"),
        _ => "expression".to_string(),
    }
}

#[cfg(test)]
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
            if raw.contains('.') {
                let number = raw
                    .parse::<f64>()
                    .map_err(|_| format!("invalid number literal: {raw}"))?;
                tokens.push(Token::Number(number));
            } else if let Ok(integer) = raw.parse::<i64>() {
                tokens.push(Token::Integer(integer));
            } else {
                let unsigned = raw
                    .parse::<u64>()
                    .map_err(|_| format!("invalid integer literal: {raw}"))?;
                tokens.push(Token::Unsigned(unsigned));
            }
            continue;
        }

        // Standalone dot is the missing-value token; `.5`, quoted ".", and dots
        // inside identifiers were already handled above.
        if ch == '.' {
            tokens.push(Token::Missing);
            index += 1;
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

#[cfg(test)]
fn parse_expression(cursor: &mut TokenCursor, min_bp: u8) -> Result<Expr, String> {
    let mut lhs = match cursor.next() {
        Some(Token::Integer(value)) => Expr::Literal(ScalarValue::Integer(value)),
        Some(Token::Unsigned(value)) => Expr::Literal(ScalarValue::Unsigned(value)),
        Some(Token::Number(value)) => Expr::Literal(ScalarValue::Number(value)),
        Some(Token::Text(value)) => Expr::Literal(ScalarValue::Text(value)),
        Some(Token::Missing) => Expr::Literal(ScalarValue::Null),
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
            match rhs {
                Expr::Literal(ScalarValue::Integer(value)) => match value.checked_neg() {
                    Some(negated) => Expr::Literal(ScalarValue::Integer(negated)),
                    None => Expr::UnaryNeg(Box::new(Expr::Literal(ScalarValue::Integer(value)))),
                },
                Expr::Literal(ScalarValue::Unsigned(value))
                    if value <= (i64::MAX as u64).saturating_add(1) =>
                {
                    Expr::Literal(ScalarValue::Integer(-(value as i128) as i64))
                }
                other => Expr::UnaryNeg(Box::new(other)),
            }
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

#[cfg(test)]
fn parse_expression_for_internal_test(expr_text: &str) -> Result<Expr, String> {
    let tokens = tokenize_expression(expr_text)?;
    let mut cursor = TokenCursor::new(tokens);
    let parsed = parse_expression(&mut cursor, 0)?;
    if cursor.peek().is_some() {
        return Err("unexpected token at end of expression".to_string());
    }
    Ok(parsed)
}

pub(crate) fn py_to_scalar(value: &Bound<'_, PyAny>) -> PyResult<ScalarValue> {
    convert_py_to_scalar(value)
}

pub(crate) fn scalar_to_number(value: &ScalarValue) -> Result<f64, String> {
    coerce_scalar_to_number_for_operation(value, "numeric coercion")
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

fn exact_integer_to_f64(value: &ScalarValue, operation: &str) -> Result<f64, String> {
    let (number, round_tripped) = match value {
        ScalarValue::Integer(integer) => {
            let number = *integer as f64;
            (number, number as i128 == i128::from(*integer))
        }
        ScalarValue::Unsigned(unsigned) => {
            let number = *unsigned as f64;
            (number, number as i128 == i128::from(*unsigned))
        }
        _ => unreachable!("exact integer coercion requires an integer scalar"),
    };

    if round_tripped {
        return Ok(number);
    }

    Err(format!(
        "exact integer requires lossless numeric coercion during {operation}: {value:?} cannot be represented losslessly as f64"
    ))
}

fn coerce_scalar_to_number_for_operation(value: &ScalarValue, operation: &str) -> Result<f64, String> {
    match value {
        ScalarValue::Integer(_) | ScalarValue::Unsigned(_) => exact_integer_to_f64(value, operation),
        ScalarValue::Number(number) => Ok(*number),
        ScalarValue::Decimal(decimal) => decimal
            .parse::<f64>()
            .map_err(|_| format!("numeric value required during {operation}, got '{decimal}'")),
        ScalarValue::Bool(boolean) => Ok(if *boolean { 1.0 } else { 0.0 }),
        ScalarValue::Text(text) => text
            .parse::<f64>()
            .map_err(|_| format!("numeric value required, got '{text}'")),
        ScalarValue::Date(_) => Err("date value cannot be converted to number".to_string()),
        ScalarValue::DateTime(_) => Err("datetime value cannot be converted to number".to_string()),
        ScalarValue::Time(_) => Err("time value cannot be converted to number".to_string()),
        ScalarValue::RawTimestamp { .. } | ScalarValue::RawTime { .. } => {
            Err(format!(
                "{} during {operation}",
                raw_temporal_conversion_error(value, "number")
            ))
        }
        ScalarValue::List(_) => Err("list value cannot be converted to number".to_string()),
        ScalarValue::Struct(_) => Err("struct value cannot be converted to number".to_string()),
        ScalarValue::Null => Ok(0.0),
    }
}

pub(crate) fn raw_temporal_conversion_error(value: &ScalarValue, target: &str) -> String {
    match value {
        ScalarValue::RawTimestamp {
            value,
            unit,
            timezone,
        } => format!(
            "raw timestamp (value={value}, unit={unit:?}, timezone={timezone:?}) cannot be converted to {target}"
        ),
        ScalarValue::RawTime { value, unit } => {
            format!("raw time (value={value}, unit={unit:?}) cannot be converted to {target}")
        }
        _ => unreachable!("raw temporal conversion error requires a raw temporal value"),
    }
}

pub(crate) fn scalar_to_text(value: &ScalarValue) -> Result<String, String> {
    match value {
        ScalarValue::Integer(integer) => Ok(integer.to_string()),
        ScalarValue::Unsigned(unsigned) => Ok(unsigned.to_string()),
        ScalarValue::Text(text) => Ok(text.clone()),
        ScalarValue::Number(number) => Ok(number.to_string()),
        ScalarValue::Decimal(decimal) => Ok(decimal.clone()),
        ScalarValue::Bool(boolean) => {
            if *boolean {
                Ok("true".to_string())
            } else {
                Ok("false".to_string())
            }
        }
        ScalarValue::Date(date) => Ok(date.to_string()),
        ScalarValue::DateTime(datetime) => Ok(datetime.format("%Y-%m-%dT%H:%M:%S").to_string()),
        ScalarValue::Time(time) => Ok(time.format("%H:%M:%S").to_string()),
        ScalarValue::RawTimestamp { .. } | ScalarValue::RawTime { .. } => {
            Err(raw_temporal_conversion_error(value, "text"))
        }
        ScalarValue::List(values) => {
            let rendered = values
                .iter()
                .map(scalar_to_text)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(format!("[{}]", rendered.join(", ")))
        }
        ScalarValue::Struct(fields) => {
            let rendered: Vec<String> = fields
                .iter()
                .map(|(name, item)| scalar_to_text(item).map(|value| format!("{name}: {value}")))
                .collect::<Result<_, _>>()?;
            Ok(format!("{{{}}}", rendered.join(", ")))
        }
        ScalarValue::Null => Ok("".to_string()),
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
        ScalarValue::RawTimestamp { .. } | ScalarValue::RawTime { .. } => {
            Err(raw_temporal_conversion_error(value, "date"))
        }
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
        ScalarValue::RawTimestamp { .. } | ScalarValue::RawTime { .. } => {
            Err(raw_temporal_conversion_error(value, "datetime"))
        }
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
        ScalarValue::RawTimestamp { .. } | ScalarValue::RawTime { .. } => {
            Err(raw_temporal_conversion_error(value, "time"))
        }
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
        ScalarValue::Integer(integer) => Ok(*integer != 0),
        ScalarValue::Unsigned(unsigned) => Ok(*unsigned != 0),
        ScalarValue::Bool(boolean) => Ok(*boolean),
        ScalarValue::Number(number) => Ok(number.abs() >= f64::EPSILON),
        // Decimal parse failure must not silently fold to false; boolean
        // consumers get a fail-closed error carrying the value and requested type.
        ScalarValue::Decimal(decimal) => decimal
            .parse::<f64>()
            .map(|number| number.abs() >= f64::EPSILON)
            .map_err(|_| format!("boolean value required, got '{decimal}'")),
        ScalarValue::Text(text) => Ok(!text.is_empty()),
        ScalarValue::Date(_) => Ok(true),
        ScalarValue::DateTime(_) => Ok(true),
        ScalarValue::Time(_) => Ok(true),
        ScalarValue::RawTimestamp { .. } | ScalarValue::RawTime { .. } => {
            Err(raw_temporal_conversion_error(value, "boolean"))
        }
        ScalarValue::List(values) => Ok(!values.is_empty()),
        ScalarValue::Struct(fields) => Ok(!fields.is_empty()),
        ScalarValue::Null => Ok(false),
    }
}

fn exact_integer_comparison(
    left: &ScalarValue,
    right: &ScalarValue,
) -> Option<std::cmp::Ordering> {
    match (left, right) {
        (ScalarValue::Integer(left), ScalarValue::Integer(right)) => Some(left.cmp(right)),
        (ScalarValue::Unsigned(left), ScalarValue::Unsigned(right)) => Some(left.cmp(right)),
        (ScalarValue::Integer(left), ScalarValue::Unsigned(right)) => {
            if *left < 0 {
                Some(std::cmp::Ordering::Less)
            } else {
                Some((*left as u64).cmp(right))
            }
        }
        (ScalarValue::Unsigned(left), ScalarValue::Integer(right)) => {
            if *right < 0 {
                Some(std::cmp::Ordering::Greater)
            } else {
                Some(left.cmp(&(*right as u64)))
            }
        }
        _ => None,
    }
}

fn temporal_unit_rank(unit: TimeUnit) -> u8 {
    match unit {
        TimeUnit::Second => 0,
        TimeUnit::Millisecond => 1,
        TimeUnit::Microsecond => 2,
        TimeUnit::Nanosecond => 3,
    }
}

fn temporal_unit_factor(from: TimeUnit, to: TimeUnit) -> Result<i128, String> {
    let rank_delta = i32::from(temporal_unit_rank(to)) - i32::from(temporal_unit_rank(from));
    if rank_delta < 0 {
        return Err(format!(
            "temporal conversion from {from:?} to {to:?} would discard precision"
        ));
    }
    let mut factor = 1_i128;
    for _ in 0..rank_delta {
        factor = factor
            .checked_mul(1_000)
            .ok_or_else(|| format!("temporal conversion from {from:?} to {to:?} overflowed"))?;
    }
    Ok(factor)
}

fn scale_temporal_value(value: i64, from: TimeUnit, to: TimeUnit) -> Result<i128, String> {
    let factor = temporal_unit_factor(from, to)?;
    i128::from(value)
        .checked_mul(factor)
        .ok_or_else(|| format!("temporal value {value} overflowed during {from:?}->{to:?} conversion"))
}

fn common_temporal_unit(left: TimeUnit, right: TimeUnit) -> TimeUnit {
    if temporal_unit_rank(left) >= temporal_unit_rank(right) {
        left
    } else {
        right
    }
}

fn datetime_epoch_nanoseconds(value: NaiveDateTime) -> Result<i128, String> {
    let utc = value.and_utc();
    i128::from(utc.timestamp())
        .checked_mul(1_000_000_000)
        .and_then(|seconds| seconds.checked_add(i128::from(utc.timestamp_subsec_nanos())))
        .ok_or_else(|| "datetime value overflowed during nanosecond conversion".to_string())
}

fn time_nanoseconds(value: NaiveTime) -> Result<i128, String> {
    let seconds = i128::from(value.num_seconds_from_midnight());
    let nanoseconds = i128::from(value.nanosecond());
    seconds
        .checked_mul(1_000_000_000)
        .and_then(|value| value.checked_add(nanoseconds))
        .ok_or_else(|| "time value overflowed during nanosecond conversion".to_string())
}

fn exact_temporal_comparison(
    left: &ScalarValue,
    right: &ScalarValue,
) -> Result<Option<std::cmp::Ordering>, String> {
    match (left, right) {
        (
            ScalarValue::RawTimestamp {
                value: left_value,
                unit: left_unit,
                ..
            },
            ScalarValue::RawTimestamp {
                value: right_value,
                unit: right_unit,
                ..
            },
        ) => {
            let unit = common_temporal_unit(*left_unit, *right_unit);
            let left_value = scale_temporal_value(*left_value, *left_unit, unit)?;
            let right_value = scale_temporal_value(*right_value, *right_unit, unit)?;
            Ok(Some(left_value.cmp(&right_value)))
        }
        (
            ScalarValue::RawTime {
                value: left_value,
                unit: left_unit,
            },
            ScalarValue::RawTime {
                value: right_value,
                unit: right_unit,
            },
        ) => {
            let unit = common_temporal_unit(*left_unit, *right_unit);
            let left_value = scale_temporal_value(*left_value, *left_unit, unit)?;
            let right_value = scale_temporal_value(*right_value, *right_unit, unit)?;
            Ok(Some(left_value.cmp(&right_value)))
        }
        (
            ScalarValue::RawTimestamp {
                value,
                unit,
                ..
            },
            ScalarValue::DateTime(datetime),
        ) => {
            let left_value = scale_temporal_value(*value, *unit, TimeUnit::Nanosecond)?;
            let right_value = datetime_epoch_nanoseconds(*datetime)?;
            Ok(Some(left_value.cmp(&right_value)))
        }
        (
            ScalarValue::DateTime(datetime),
            ScalarValue::RawTimestamp {
                value,
                unit,
                ..
            },
        ) => {
            let left_value = datetime_epoch_nanoseconds(*datetime)?;
            let right_value = scale_temporal_value(*value, *unit, TimeUnit::Nanosecond)?;
            Ok(Some(left_value.cmp(&right_value)))
        }
        (ScalarValue::RawTime { value, unit }, ScalarValue::Time(time)) => {
            let left_value = scale_temporal_value(*value, *unit, TimeUnit::Nanosecond)?;
            let right_value = time_nanoseconds(*time)?;
            Ok(Some(left_value.cmp(&right_value)))
        }
        (ScalarValue::Time(time), ScalarValue::RawTime { value, unit }) => {
            let left_value = time_nanoseconds(*time)?;
            let right_value = scale_temporal_value(*value, *unit, TimeUnit::Nanosecond)?;
            Ok(Some(left_value.cmp(&right_value)))
        }
        (left, right)
            if matches!(
                left,
                ScalarValue::RawTimestamp { .. } | ScalarValue::RawTime { .. }
            ) || matches!(
                right,
                ScalarValue::RawTimestamp { .. } | ScalarValue::RawTime { .. }
            ) => Err(format!(
                "raw temporal values cannot be compared with {left:?} and {right:?}"
            )),
        _ => Ok(None),
    }
}

fn is_numeric_coercion_operand(value: &ScalarValue) -> bool {
    matches!(
        value,
        ScalarValue::Integer(_)
            | ScalarValue::Unsigned(_)
            | ScalarValue::Number(_)
            | ScalarValue::Decimal(_)
            | ScalarValue::Bool(_)
    )
}

fn scalar_eq(left: &ScalarValue, right: &ScalarValue) -> Result<bool, String> {
    scalar_eq_for_operation(left, right, "comparison")
}

fn scalar_eq_for_operation(
    left: &ScalarValue,
    right: &ScalarValue,
    operation: &str,
) -> Result<bool, String> {
    if let Some(ordering) = exact_temporal_comparison(left, right).map_err(|error| {
        format!("{error} during {operation}")
    })? {
        return Ok(ordering == std::cmp::Ordering::Equal);
    }
    if let Some(ordering) = exact_integer_comparison(left, right) {
        return Ok(ordering == std::cmp::Ordering::Equal);
    }

    match (left, right) {
        (ScalarValue::Null, ScalarValue::Null) => Ok(true),
        (ScalarValue::Text(l), ScalarValue::Text(r)) => Ok(l == r),
        (ScalarValue::Date(l), ScalarValue::Date(r)) => Ok(l == r),
        (ScalarValue::DateTime(l), ScalarValue::DateTime(r)) => Ok(l == r),
        (ScalarValue::Time(l), ScalarValue::Time(r)) => Ok(l == r),
        (ScalarValue::List(l), ScalarValue::List(r)) => Ok(l == r),
        (ScalarValue::Struct(l), ScalarValue::Struct(r)) => Ok(l == r),
        _ if is_numeric_coercion_operand(left) || is_numeric_coercion_operand(right) => {
            let l = coerce_scalar_to_number_for_operation(left, operation)?;
            let r = coerce_scalar_to_number_for_operation(right, operation)?;
            Ok((l - r).abs() < f64::EPSILON)
        }
        _ => Ok(false),
    }
}

pub(crate) fn optional_source_values_equal(
    left: Option<SourceValueRef<'_>>,
    right: Option<SourceValueRef<'_>>,
) -> Result<bool, String> {
    match (left, right) {
        (Some(lhs), Some(rhs)) => scalar_eq(lhs.as_scalar()?, rhs.as_scalar()?),
        (None, None) => Ok(true),
        _ => Ok(false),
    }
}

pub(crate) fn optional_logical_values_equal(
    left: Option<crate::row_view::ResolvedValue<'_>>,
    right: Option<crate::row_view::ResolvedValue<'_>>,
    column: &str,
    operation: &str,
) -> Result<bool, String> {
    let left = logical_value_to_scalar(left, column, operation)?;
    let right = logical_value_to_scalar(right, column, operation)?;
    match (left, right) {
        (Some(lhs), Some(rhs)) => scalar_eq_for_operation(&lhs, &rhs, operation),
        (None, None) => Ok(true),
        _ => Ok(false),
    }
}

fn logical_value_to_scalar(
    value: Option<crate::row_view::ResolvedValue<'_>>,
    column: &str,
    operation: &str,
) -> Result<Option<ScalarValue>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    match value {
        crate::row_view::ResolvedValue::SourceScalar(value_ref) => value_ref
            .as_scalar()
            .map(|scalar| Some(scalar.clone())),
        crate::row_view::ResolvedValue::SourceCell(cell) => Err(format!(
            "complex access is unsupported for column '{column}' during {operation}: {}",
            cell.access_error(operation)
        )),
        crate::row_view::ResolvedValue::Owned(value)
        | crate::row_view::ResolvedValue::Automatic(value) => value
            .as_scalar()
            .map(|scalar| Some(scalar.clone())),
    }
}

fn scalar_compare(left: &ScalarValue, right: &ScalarValue) -> Result<std::cmp::Ordering, String> {
    scalar_compare_for_operation(left, right, "comparison")
}

fn scalar_compare_for_operation(
    left: &ScalarValue,
    right: &ScalarValue,
    operation: &str,
) -> Result<std::cmp::Ordering, String> {
    if let Some(ordering) = exact_temporal_comparison(left, right).map_err(|error| {
        format!("{error} during {operation}")
    })? {
        return Ok(ordering);
    }
    if let Some(ordering) = exact_integer_comparison(left, right) {
        return Ok(ordering);
    }

    match (left, right) {
        (ScalarValue::Text(l), ScalarValue::Text(r)) => Ok(l.cmp(r)),
        (ScalarValue::Date(l), ScalarValue::Date(r)) => Ok(l.cmp(r)),
        (ScalarValue::DateTime(l), ScalarValue::DateTime(r)) => Ok(l.cmp(r)),
        (ScalarValue::Time(l), ScalarValue::Time(r)) => Ok(l.cmp(r)),
        (ScalarValue::List(_), ScalarValue::List(_)) | (ScalarValue::Struct(_), ScalarValue::Struct(_)) => {
            Ok(scalar_to_text(left)?.cmp(&scalar_to_text(right)?))
        }
        _ => {
            let l = coerce_scalar_to_number_for_operation(left, operation)?;
            let r = coerce_scalar_to_number_for_operation(right, operation)?;
            l.partial_cmp(&r)
                .ok_or_else(|| "comparison failed".to_string())
        }
    }
}

// NativeExpressionFailClosedBoundary: parse and evaluation failures stay on the
// native error path; only `apply("name", ...)` may reach a Python callable.
#[cfg(test)]
pub(crate) fn evaluate_scalar_expression(
    row: &Bound<'_, PyDict>,
    expr_text: &str,
    state: &mut EvalRuntimeState,
) -> Result<ScalarValue, String> {
    let parsed = parse_expression_for_internal_test(expr_text)?;
    evaluate_expression(row, &parsed, state)
}

#[cfg(test)]
fn evaluate_expression(
    row: &Bound<'_, PyDict>,
    expr: &Expr,
    state: &mut EvalRuntimeState,
) -> Result<ScalarValue, String> {
    eval_expression_tree(row, expr, state)
}

/// Evaluate an already-lowered expression without parsing source text.
pub(crate) fn evaluate_typed_expression(
    row: &Bound<'_, PyDict>,
    expr: &Expr,
    state: &mut EvalRuntimeState,
) -> Result<ScalarValue, String> {
    eval_expression_tree(row, expr, state)
}

/// Evaluate an already-lowered condition with the existing scalar-to-bool
/// rules, without parsing display text.
pub(crate) fn evaluate_typed_condition(
    row: &Bound<'_, PyDict>,
    expr: &Expr,
    state: &mut EvalRuntimeState,
) -> Result<bool, String> {
    let value = evaluate_typed_expression(row, expr, state)?;
    scalar_to_bool(&value)
}

enum EvalOperand<'a> {
    Resolved {
        value: ResolvedValue<'a>,
        column: String,
    },
    Scalar(ScalarValue),
}

impl<'a> EvalOperand<'a> {
    fn as_scalar_for_operation(&self, operation: &str) -> Result<&ScalarValue, String> {
        match self {
            Self::Scalar(value) => Ok(value),
            Self::Resolved { value, column } => match value {
                ResolvedValue::Source(SourceValueRef::Cell(cell)) => Err(format!(
                    "complex access is unsupported for column '{column}' during {operation}: {}",
                    cell.access_error(operation)
                )),
                ResolvedValue::Source(SourceValueRef::Scalar(value)) => Ok(value),
                ResolvedValue::Owned(value) => value.as_scalar().map_err(|error| {
                    format!(
                        "complex access is unsupported for column '{column}' during {operation}: {error}"
                    )
                }),
            },
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
                return Ok(Some(EvalOperand::Resolved {
                    value,
                    column: name.trim().to_string(),
                }));
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
    coerce_scalar_to_number_for_operation(
        value.as_scalar_for_operation("numeric expression")?,
        "numeric expression",
    )
}

fn operand_to_bool(value: &EvalOperand<'_>) -> Result<bool, String> {
    coerce_scalar_to_bool(value.as_scalar_for_operation("boolean expression")?)
}

fn operand_eq(left: &EvalOperand<'_>, right: &EvalOperand<'_>) -> Result<bool, String> {
    scalar_eq_for_operation(
        left.as_scalar_for_operation("expression comparison")?,
        right.as_scalar_for_operation("expression comparison")?,
        "expression comparison",
    )
}

fn operand_compare(left: &EvalOperand<'_>, right: &EvalOperand<'_>) -> Result<std::cmp::Ordering, String> {
    scalar_compare_for_operation(
        left.as_scalar_for_operation("expression comparison")?,
        right.as_scalar_for_operation("expression comparison")?,
        "expression comparison",
    )
}

fn eval_binary_operands(
    left_value: &EvalOperand<'_>,
    op: &BinaryOp,
    right_value: &EvalOperand<'_>,
) -> Result<ScalarValue, String> {
    match op {
        BinaryOp::Add => add_operands(left_value, right_value),
        BinaryOp::Sub => arithmetic_operands(left_value, right_value, |left, right| left - right),
        BinaryOp::Mul => arithmetic_operands(left_value, right_value, |left, right| left * right),
        BinaryOp::Div => arithmetic_operands(left_value, right_value, |left, right| left / right),
        BinaryOp::Pow => arithmetic_operands(left_value, right_value, |left, right| left.powf(right)),
        BinaryOp::Mod => arithmetic_operands(left_value, right_value, |left, right| left % right),
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

// Arithmetic NULL propagation: detect NULL before numeric coercion so missing
// operands never silently collapse to 0. Empty text and non-numeric values keep
// their fail-closed coercion errors; comparisons and logical operators are not
// NULL-propagating and keep the coercion rules below.
fn arithmetic_operands(
    left_value: &EvalOperand<'_>,
    right_value: &EvalOperand<'_>,
    operation: impl Fn(f64, f64) -> f64,
) -> Result<ScalarValue, String> {
    if arithmetic_operand_is_null(left_value)? || arithmetic_operand_is_null(right_value)? {
        return Ok(ScalarValue::Null);
    }
    Ok(ScalarValue::Number(operation(
        operand_to_number(left_value)?,
        operand_to_number(right_value)?,
    )))
}

fn add_operands(
    left_value: &EvalOperand<'_>,
    right_value: &EvalOperand<'_>,
) -> Result<ScalarValue, String> {
    // Keep the established NULL short-circuit before inspecting the right
    // operand. Only two Text scalars use concatenation; every other pair
    // follows the existing numeric coercion path and its error ordering.
    if arithmetic_operand_is_null(left_value)? || arithmetic_operand_is_null(right_value)? {
        return Ok(ScalarValue::Null);
    }

    let left = left_value.as_scalar_for_operation("numeric expression")?;
    let right = right_value.as_scalar_for_operation("numeric expression")?;
    if let (ScalarValue::Text(left), ScalarValue::Text(right)) = (left, right) {
        return Ok(ScalarValue::Text(format!("{left}{right}")));
    }

    let left = coerce_scalar_to_number_for_operation(left, "numeric expression")?;
    let right = coerce_scalar_to_number_for_operation(right, "numeric expression")?;
    Ok(ScalarValue::Number(left + right))
}

fn arithmetic_operand_is_null(value: &EvalOperand<'_>) -> Result<bool, String> {
    Ok(matches!(
        value.as_scalar_for_operation("numeric expression")?,
        ScalarValue::Null
    ))
}

fn eval_expression_tree(row: &Bound<'_, PyDict>, expr: &Expr, state: &mut EvalRuntimeState) -> Result<ScalarValue, String> {
    match expr {
        Expr::Literal(value) => Ok(value.clone()),
        Expr::DeferredError { cause } => Err(cause.clone()),
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
                if arithmetic_operand_is_null(&value)? {
                    return Ok(ScalarValue::Null);
                }
                return Ok(ScalarValue::Number(-operand_to_number(&value)?));
            }
            let inner_value = eval_expression_tree(row, inner, state)?;
            if matches!(inner_value, ScalarValue::Null) {
                return Ok(ScalarValue::Null);
            }
            Ok(ScalarValue::Number(-scalar_to_number(&inner_value)?))
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
                .enumerate()
                .map(|(index, arg)| {
                    eval_expression_tree(row, arg, state).map_err(|error| {
                        if lowered == "apply" && index > 0 {
                            format!(
                                "apply() argument {} ({}) evaluation failed: {error}",
                                index - 1,
                                python_boundary_argument_context(arg)
                            )
                        } else {
                            error
                        }
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            if lowered == "apply" {
                let argument_contexts: Vec<String> = args
                    .iter()
                    .skip(1)
                    .map(python_boundary_argument_context)
                    .collect();
                evaluate_function_with_context(
                    row.py(),
                    name,
                    &evaluated_args,
                    &argument_contexts,
                    state,
                )
            } else {
                evaluate_function(row.py(), name, &evaluated_args, state)
            }
        }
    }
}

pub(crate) fn evaluate_structured_where(
    expression: &AstWhereExpression,
    state: &EvalRuntimeState,
) -> Result<bool, String> {
    let view = state
        .logical_row
        .as_ref()
        .ok_or_else(|| "logical row view is required for structured WHERE= evaluation".to_string())?;
    evaluate_structured_where_expression(expression, view)
}

pub(crate) fn evaluate_structured_where_on_view(
    expression: &AstWhereExpression,
    view: &crate::row_view::LogicalRowView,
) -> Result<bool, String> {
    evaluate_structured_where_expression(expression, view)
}

fn evaluate_structured_where_expression<V: LogicalRowAccess + ?Sized>(
    expression: &AstWhereExpression,
    view: &V,
) -> Result<bool, String> {
    match expression {
        AstWhereExpression::Comparison {
            column,
            operator,
            value,
        } => {
            let left = resolve_structured_scalar(column, view, "dataset WHERE=")?;
            let right = structured_literal_to_scalar(value);
            match operator {
                AstWhereComparisonOperator::Eq | AstWhereComparisonOperator::Equal => {
                    scalar_eq_for_operation(&left, &right, "dataset WHERE=")
                }
                AstWhereComparisonOperator::NotEqual => {
                    scalar_eq_for_operation(&left, &right, "dataset WHERE=").map(|value| !value)
                }
                AstWhereComparisonOperator::Greater => {
                    scalar_compare_for_operation(&left, &right, "dataset WHERE=")
                        .map(|value| value == std::cmp::Ordering::Greater)
                }
                AstWhereComparisonOperator::GreaterEqual => {
                    scalar_compare_for_operation(&left, &right, "dataset WHERE=").map(|value| {
                        matches!(value, std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
                    })
                }
                AstWhereComparisonOperator::Less => {
                    scalar_compare_for_operation(&left, &right, "dataset WHERE=")
                        .map(|value| value == std::cmp::Ordering::Less)
                }
                AstWhereComparisonOperator::LessEqual => {
                    scalar_compare_for_operation(&left, &right, "dataset WHERE=").map(|value| {
                        matches!(value, std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                    })
                }
            }
        }
        AstWhereExpression::Null { column, negated } => {
            let is_null = resolve_structured_null(column, view)?;
            Ok(if *negated { !is_null } else { is_null })
        }
        AstWhereExpression::Logical { operator, operands } => match operator {
            AstWhereLogicalOperator::And => operands.iter().try_fold(true, |result, operand| {
                if !result {
                    return Ok(false);
                }
                evaluate_structured_where_expression(operand, view).map(|value| result && value)
            }),
            AstWhereLogicalOperator::Or => operands.iter().try_fold(false, |result, operand| {
                if result {
                    return Ok(true);
                }
                evaluate_structured_where_expression(operand, view).map(|value| result || value)
            }),
        },
        AstWhereExpression::Not { operand } => {
            evaluate_structured_where_expression(operand, view).map(|value| !value)
        }
    }
}

fn structured_literal_to_scalar(value: &AstWhereScalar) -> ScalarValue {
    match value {
        AstWhereScalar::Text(value) => ScalarValue::Text(value.clone()),
        AstWhereScalar::Integer(value) => ScalarValue::Integer(*value),
        AstWhereScalar::Unsigned(value) => ScalarValue::Unsigned(*value),
        AstWhereScalar::Number(value) => ScalarValue::Number(*value),
        AstWhereScalar::Bool(value) => ScalarValue::Bool(*value),
    }
}

fn resolve_structured_scalar<V: LogicalRowAccess + ?Sized>(
    column: &str,
    view: &V,
    operation: &str,
) -> Result<ScalarValue, String> {
    let Some(value) = view.resolve(column).map_err(|error| error.to_string())? else {
        return Err(format!("column '{column}' is not visible during {operation}"));
    };
    match value {
        crate::row_view::ResolvedValue::SourceScalar(value) => value
            .as_scalar()
            .map(|value| value.clone())
            .map_err(|error| format!("complex access is unsupported for column '{column}' during {operation}: {error}")),
        crate::row_view::ResolvedValue::SourceCell(cell) => Err(format!(
            "complex access is unsupported for column '{column}' during {operation}: {}",
            cell.access_error(operation)
        )),
        crate::row_view::ResolvedValue::Owned(value)
        | crate::row_view::ResolvedValue::Automatic(value) => value
            .as_scalar()
            .map(|value| value.clone())
            .map_err(|error| format!("complex access is unsupported for column '{column}' during {operation}: {error}")),
    }
}

fn resolve_structured_null<V: LogicalRowAccess + ?Sized>(
    column: &str,
    view: &V,
) -> Result<bool, String> {
    let Some(value) = view.resolve(column).map_err(|error| error.to_string())? else {
        return Err(format!("column '{column}' is not visible during dataset WHERE="));
    };
    match value {
        crate::row_view::ResolvedValue::SourceScalar(value) => {
            Ok(matches!(value.as_scalar()?, ScalarValue::Null))
        }
        crate::row_view::ResolvedValue::SourceCell(cell) => Err(format!(
            "complex access is unsupported for column '{column}' during dataset WHERE=: {}",
            cell.access_error("dataset WHERE=")
        )),
        crate::row_view::ResolvedValue::Owned(value)
        | crate::row_view::ResolvedValue::Automatic(value) => {
            if let crate::value_ref::OwnedValue::SourceCell(cell) = value {
                return Err(format!(
                    "complex access is unsupported for column '{column}' during dataset WHERE=: {}",
                    cell.access_error("dataset WHERE=")
                ));
            }
            Ok(matches!(value.as_scalar()?, ScalarValue::Null))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        coerce_scalar_to_bool, coerce_scalar_to_number_for_operation, eval_binary_operands,
        evaluate_expression, evaluate_scalar_expression, parse_expression_for_internal_test,
        scalar_compare, scalar_eq, scalar_to_bool, scalar_to_number, scalar_to_text,
        tokenize_expression, BinaryOp, EvalOperand, Expr, ScalarValue, Token,
    };
    use crate::runtime::EvalRuntimeState;
    use crate::typed_ast::{lower_expression, parse_typed_program, TypedStatementPayload};
    use arrow::datatypes::TimeUnit;
    use pyo3::types::{PyAnyMethods, PyDictMethods};
    use serde_json::{json, Value};

    fn typed_test_span() -> Value {
        json!({
            "start": 0,
            "end": 1,
            "line": 1,
            "column": 1,
            "end_line": 1,
            "end_column": 2,
            "source_id": "typed-expression-evaluation-test"
        })
    }

    fn typed_test_literal(kind: &str, lexeme: &str) -> Value {
        json!({
            "tag": "literal",
            "span": typed_test_span(),
            "origin": null,
            "kind": kind,
            "lexeme": lexeme
        })
    }

    fn lower_typed_test_expression(expression: Value) -> Expr {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/typed_payload_v1.json"
        )))
        .expect("shared typed payload fixture is valid JSON");
        let mut program = fixture["valid_program"].clone();
        program["statements"][4]["payload"]["value"] = expression;
        let parsed =
            parse_typed_program(&program.to_string()).expect("test typed expression is valid");
        match &parsed.statements[4].payload {
            TypedStatementPayload::Assign { value, .. } => {
                lower_expression(value).expect("test typed expression lowers")
            }
            _ => panic!("fixture statement is not ASSIGN"),
        }
    }

    fn lower_typed_test_binary(operator: &str, left: Value, right: Value) -> Expr {
        lower_typed_test_expression(json!({
            "tag": "binary",
            "span": typed_test_span(),
            "origin": null,
            "operator": operator,
            "left": left,
            "right": right
        }))
    }

    fn typed_test_logical_row(bindings: &[(&str, ScalarValue)]) -> crate::row_view::LogicalRowView {
        let mut values = std::collections::HashMap::new();
        let columns = bindings
            .iter()
            .map(|(name, value)| {
                let normalized_name = name.trim().to_uppercase();
                values.insert(
                    normalized_name.clone(),
                    crate::value_ref::OwnedValue::from(value.clone()),
                );
                (
                    (*name).to_string(),
                    crate::row_view::ColumnBinding::Owned { normalized_name },
                )
            })
            .collect();
        crate::row_view::LogicalRowView::with_values(
            0,
            std::rc::Rc::new(crate::source_registry::ExecutionSourceRegistry::<
                crate::row_cursor::NativeArrowRowCursor,
            >::new()),
            columns,
            values,
            std::collections::HashMap::new(),
        )
        .expect("native logical row view")
    }

    #[test]
    fn exact_scalar_payloads_are_inspectable_without_loss() {
        let signed = ScalarValue::Integer(i64::MIN);
        assert_eq!(signed.as_integer(), Some(i64::MIN));
        assert_eq!(signed.as_unsigned(), None);

        let unsigned = ScalarValue::Unsigned(u64::MAX);
        assert_eq!(unsigned.as_unsigned(), Some(u64::MAX));
        assert_eq!(unsigned.as_integer(), None);

        let timestamp = ScalarValue::RawTimestamp {
            value: i64::MIN,
            unit: TimeUnit::Nanosecond,
            timezone: Some("UTC".to_string()),
        };
        assert_eq!(
            timestamp.as_raw_timestamp(),
            Some((i64::MIN, TimeUnit::Nanosecond, Some("UTC")))
        );

        let time = ScalarValue::RawTime {
            value: 86_399_999_999_999,
            unit: TimeUnit::Nanosecond,
        };
        assert_eq!(
            time.as_raw_time(),
            Some((86_399_999_999_999, TimeUnit::Nanosecond))
        );
    }

    #[test]
    fn exact_scalar_clone_preserves_domain_and_temporal_identity() {
        let value = ScalarValue::RawTimestamp {
            value: 1_234_567_890,
            unit: TimeUnit::Microsecond,
            timezone: Some("America/New_York".to_string()),
        };

        assert_eq!(value.clone(), value);
        assert_eq!(
            value.as_raw_timestamp().unwrap().2,
            Some("America/New_York")
        );
    }

    #[test]
    fn exact_integer_comparison_preserves_domains_and_fails_lossy_mixes() {
        let boundary = 1_i64 << 53;
        assert!(scalar_eq(
            &ScalarValue::Integer(boundary + 1),
            &ScalarValue::Integer(boundary + 1),
        )
        .unwrap());
        assert_eq!(
            scalar_compare(
                &ScalarValue::Integer(boundary + 1),
                &ScalarValue::Integer(boundary + 2),
            )
            .unwrap(),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            scalar_compare(&ScalarValue::Integer(-1), &ScalarValue::Unsigned(0)).unwrap(),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            scalar_compare(
                &ScalarValue::Unsigned(u64::MAX),
                &ScalarValue::Integer(i64::MAX),
            )
            .unwrap(),
            std::cmp::Ordering::Greater
        );
        assert!(scalar_eq(
            &ScalarValue::Integer(boundary + 1),
            &ScalarValue::Number((boundary + 1) as f64),
        )
        .is_err());
        assert_eq!(scalar_to_number(&ScalarValue::Integer(42)).unwrap(), 42.0);
        let error = scalar_to_number(&ScalarValue::Integer(boundary + 1)).unwrap_err();
        assert!(error.contains("lossless") && error.contains("numeric coercion"));
    }

    #[test]
    fn raw_temporal_comparison_uses_checked_common_units_and_legacy_views() {
        let timestamp_ns = ScalarValue::RawTimestamp {
            value: 1_001,
            unit: TimeUnit::Nanosecond,
            timezone: Some("UTC".to_string()),
        };
        let timestamp_us = ScalarValue::RawTimestamp {
            value: 1,
            unit: TimeUnit::Microsecond,
            timezone: Some("Asia/Tokyo".to_string()),
        };
        assert_eq!(
            scalar_compare(&timestamp_ns, &timestamp_us).unwrap(),
            std::cmp::Ordering::Greater
        );
        assert!(scalar_eq(
            &timestamp_ns,
            &ScalarValue::RawTimestamp {
                value: 1_001,
                unit: TimeUnit::Nanosecond,
                timezone: Some("America/New_York".to_string()),
            },
        )
        .unwrap());

        let time_ns = ScalarValue::RawTime {
            value: 1_000_001,
            unit: TimeUnit::Nanosecond,
        };
        let time_us = ScalarValue::RawTime {
            value: 1_000,
            unit: TimeUnit::Microsecond,
        };
        assert_eq!(scalar_eq(&time_ns, &time_us).unwrap(), false);
        assert_eq!(
            scalar_compare(&time_ns, &time_us).unwrap(),
            std::cmp::Ordering::Greater
        );

        let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1)
            .unwrap()
            .and_hms_nano_opt(0, 0, 0, 1_001)
            .unwrap();
        assert!(scalar_eq(
            &timestamp_ns,
            &ScalarValue::DateTime(epoch),
        )
        .unwrap());
        assert!(scalar_eq(&timestamp_ns, &time_ns).is_err());
    }

    #[test]
    fn raw_temporal_conversions_fail_closed_with_identity_context() {
        let timestamp = ScalarValue::RawTimestamp {
            value: -7,
            unit: TimeUnit::Microsecond,
            timezone: Some("Asia/Tokyo".to_string()),
        };
        for error in [
            scalar_to_text(&timestamp).unwrap_err(),
            scalar_to_bool(&timestamp).unwrap_err(),
            scalar_to_number(&timestamp).unwrap_err(),
        ] {
            assert!(error.contains("-7"));
            assert!(error.contains("Microsecond"));
            assert!(error.contains("Asia/Tokyo"));
        }

        let time = ScalarValue::RawTime {
            value: 11,
            unit: TimeUnit::Nanosecond,
        };
        let error = scalar_to_text(&time).unwrap_err();
        assert!(error.contains("11"));
        assert!(error.contains("Nanosecond"));
        assert!(scalar_to_bool(&time).is_err());
    }

    #[test]
    fn standalone_dot_tokenizes_as_missing_without_conflicts() {
        assert!(matches!(
            tokenize_expression(".").unwrap().as_slice(),
            [Token::Missing]
        ));
        assert!(matches!(
            tokenize_expression("x = .").unwrap().as_slice(),
            [Token::Identifier(_), Token::Op(op), Token::Missing] if op == "="
        ));

        // `.5` stays on the existing decimal number path.
        assert!(matches!(
            tokenize_expression(".5").unwrap().as_slice(),
            [Token::Number(number)] if *number == 0.5
        ));
        assert!(matches!(
            tokenize_expression("x = .5").unwrap().as_slice(),
            [Token::Identifier(_), Token::Op(op), Token::Number(number)]
                if op == "=" && *number == 0.5
        ));

        // Quoted dots stay text literals.
        assert!(matches!(
            tokenize_expression("\".\"").unwrap().as_slice(),
            [Token::Text(text)] if text == "."
        ));
        assert!(matches!(
            tokenize_expression("'.'").unwrap().as_slice(),
            [Token::Text(text)] if text == "."
        ));

        // Dots inside identifiers stay identifiers.
        assert!(matches!(
            tokenize_expression("a.b").unwrap().as_slice(),
            [Token::Identifier(name)] if name == "a.b"
        ));
        assert!(matches!(
            tokenize_expression("x = a. ").unwrap().as_slice(),
            [Token::Identifier(_), Token::Op(op), Token::Identifier(name)]
                if op == "=" && name == "a."
        ));

        // `1 . 2` isolates the dot as Missing between integers.
        assert!(matches!(
            tokenize_expression("1 . 2").unwrap().as_slice(),
            [Token::Integer(1), Token::Missing, Token::Integer(2)]
        ));
    }

    #[test]
    fn numdot_text_before_closer_keeps_legacy_text_value() {
        assert!(matches!(
            tokenize_expression("(5.)").unwrap().as_slice(),
            [Token::LParen, Token::Text(value), Token::RParen] if value == "5."
        ));
    }

    #[test]
    fn standalone_dot_parses_to_null_literal_independent_of_context() {
        assert!(matches!(
            parse_expression_for_internal_test(".").unwrap(),
            Expr::Literal(ScalarValue::Null)
        ));
        assert!(matches!(
            parse_expression_for_internal_test("( . )").unwrap(),
            Expr::Literal(ScalarValue::Null)
        ));
        assert!(matches!(
            parse_expression_for_internal_test("missing(.)").unwrap(),
            Expr::FunctionCall { name, args }
                if name == "missing"
                    && matches!(args.as_slice(), [Expr::Literal(ScalarValue::Null)])
        ));

        // Existing null keywords keep their Literal(Null) behavior.
        assert!(matches!(
            parse_expression_for_internal_test("none").unwrap(),
            Expr::Literal(ScalarValue::Null)
        ));
    }

    #[test]
    fn standalone_dot_evaluates_to_scalar_null() {
        pyo3::Python::attach(|py| {
            let row = pyo3::types::PyDict::new(py);
            let mut state = EvalRuntimeState::default();
            let expr = parse_expression_for_internal_test(".").unwrap();
            let value = evaluate_expression(&row, &expr, &mut state).unwrap();
            assert!(matches!(value, ScalarValue::Null));
        });
    }

    #[test]
    fn misplaced_dot_fails_closed_as_expression_parse_error() {
        let error = parse_expression_for_internal_test("1 . 2").unwrap_err();
        assert!(error.contains("unexpected token at end of expression"));

        assert!(parse_expression_for_internal_test("x .").is_err());
        assert!(matches!(
            tokenize_expression("x .").unwrap().as_slice(),
            [Token::Identifier(_), Token::Missing]
        ));
    }

    fn evaluate_expression_text(expr_text: &str) -> Result<ScalarValue, String> {
        pyo3::Python::attach(|py| {
            let row = pyo3::types::PyDict::new(py);
            let mut state = EvalRuntimeState::default();
            let expr = parse_expression_for_internal_test(expr_text)?;
            evaluate_expression(&row, &expr, &mut state)
        })
    }

    #[test]
    fn typed_lower_deferred_error_returns_saved_cause_without_evaluating() {
        let cause = "unsupported inline action: saved cause";
        pyo3::Python::attach(|py| {
            let row = pyo3::types::PyDict::new(py);
            let mut state = EvalRuntimeState::default();
            let expression = Expr::DeferredError {
                cause: cause.to_string(),
            };
            assert_eq!(
                evaluate_expression(&row, &expression, &mut state).unwrap_err(),
                cause
            );
        });
    }

    #[test]
    fn typed_eval_entry_preserves_scalar_coercion_and_deferred_faults() {
        let sum = lower_typed_test_expression(json!({
            "tag": "binary",
            "span": typed_test_span(),
            "origin": null,
            "operator": "+",
            "left": typed_test_literal("numeric", "1"),
            "right": typed_test_literal("numeric", "2")
        }));
        let coercion_cases = vec![
            (
                lower_typed_test_expression(json!({
                    "tag": "binary",
                    "span": typed_test_span(),
                    "origin": null,
                    "operator": "+",
                    "left": typed_test_literal("numeric", "1"),
                    "right": typed_test_literal("string", "'2'")
                })),
                ScalarValue::Number(3.0),
            ),
            (
                lower_typed_test_expression(json!({
                    "tag": "binary",
                    "span": typed_test_span(),
                    "origin": null,
                    "operator": "+",
                    "left": typed_test_literal("missing", "."),
                    "right": typed_test_literal("numeric", "2")
                })),
                ScalarValue::Null,
            ),
        ];
        let deferred_cause = "saved typed expression cause";
        let deferred = lower_typed_test_expression(json!({
            "tag": "deferred",
            "span": typed_test_span(),
            "origin": null,
            "error_kind": "unsupported_expression",
            "native_cause": deferred_cause,
            "where_syntax_summary": "saved summary",
            "display_text": "display only",
            "apply_references": []
        }));

        pyo3::Python::attach(|py| {
            let row = pyo3::types::PyDict::new(py);
            let mut state = EvalRuntimeState::default();
            assert_eq!(
                super::evaluate_typed_expression(&row, &sum, &mut state).unwrap(),
                ScalarValue::Number(3.0)
            );
            for (expression, expected) in &coercion_cases {
                assert_eq!(
                    super::evaluate_typed_expression(&row, expression, &mut state).unwrap(),
                    *expected
                );
            }
            assert_eq!(
                super::evaluate_typed_expression(&row, &deferred, &mut state).unwrap_err(),
                deferred_cause
            );
        });
    }

    #[test]
    fn typed_string_add_concatenates_only_two_text_values() {
        let cases = vec![
            (
                lower_typed_test_binary(
                    "+",
                    typed_test_literal("string", "'a'"),
                    typed_test_literal("string", "'B'"),
                ),
                ScalarValue::Text("aB".to_string()),
            ),
            (
                lower_typed_test_binary(
                    "+",
                    typed_test_literal("string", "'1'"),
                    typed_test_literal("string", "'2'"),
                ),
                ScalarValue::Text("12".to_string()),
            ),
            (
                lower_typed_test_binary(
                    "+",
                    typed_test_literal("string", "''"),
                    typed_test_literal("string", "'  雪  '"),
                ),
                ScalarValue::Text("  雪  ".to_string()),
            ),
            (
                lower_typed_test_binary(
                    "+",
                    typed_test_literal("string", "'1'"),
                    typed_test_literal("numeric", "2"),
                ),
                ScalarValue::Number(3.0),
            ),
            (
                lower_typed_test_binary(
                    "+",
                    typed_test_literal("missing", "."),
                    typed_test_literal("string", "'invalid numeric text'"),
                ),
                ScalarValue::Null,
            ),
            (
                lower_typed_test_binary(
                    "+",
                    typed_test_literal("string", "'a'"),
                    typed_test_literal("missing", "."),
                ),
                ScalarValue::Null,
            ),
        ];
        let invalid_numeric = lower_typed_test_binary(
            "+",
            typed_test_literal("string", "'a'"),
            typed_test_literal("numeric", "2"),
        );
        let list_addition = lower_typed_test_binary(
            "+",
            json!({"tag": "name", "span": typed_test_span(), "origin": null, "name": "items"}),
            typed_test_literal("numeric", "1"),
        );
        let struct_addition = lower_typed_test_binary(
            "+",
            json!({"tag": "name", "span": typed_test_span(), "origin": null, "name": "record"}),
            typed_test_literal("numeric", "1"),
        );

        pyo3::Python::attach(|py| {
            let row = pyo3::types::PyDict::new(py);
            let mut state = EvalRuntimeState::default();
            state.logical_row = Some(typed_test_logical_row(&[
                ("items", ScalarValue::List(vec![ScalarValue::Number(1.0)])),
                (
                    "record",
                    ScalarValue::Struct(vec![("field".to_string(), ScalarValue::Number(1.0))]),
                ),
            ]));

            for (expression, expected) in &cases {
                assert_eq!(
                    super::evaluate_typed_expression(&row, expression, &mut state).unwrap(),
                    *expected
                );
            }
            assert_eq!(
                super::evaluate_typed_expression(&row, &invalid_numeric, &mut state).unwrap_err(),
                "numeric value required, got 'a'"
            );
            assert_eq!(
                super::evaluate_typed_expression(&row, &list_addition, &mut state).unwrap_err(),
                "list value cannot be converted to number"
            );
            assert_eq!(
                super::evaluate_typed_expression(&row, &struct_addition, &mut state).unwrap_err(),
                "struct value cannot be converted to number"
            );
        });
    }

    #[test]
    fn typed_eval_entry_resolves_names_from_logical_row_view() {
        let expression = lower_typed_test_expression(json!({
            "tag": "name",
            "span": typed_test_span(),
            "origin": null,
            "name": "value"
        }));

        pyo3::Python::attach(|py| {
            let row = pyo3::types::PyDict::new(py);
            let mut values = std::collections::HashMap::new();
            values.insert(
                "VALUE".to_string(),
                crate::value_ref::OwnedValue::from(ScalarValue::Integer(41)),
            );
            let logical_row = crate::row_view::LogicalRowView::with_values(
                0,
                std::rc::Rc::new(crate::source_registry::ExecutionSourceRegistry::<
                    crate::row_cursor::NativeArrowRowCursor,
                >::new()),
                vec![(
                    "Value".to_string(),
                    crate::row_view::ColumnBinding::Owned {
                        normalized_name: "VALUE".to_string(),
                    },
                )],
                values,
                std::collections::HashMap::new(),
            )
            .expect("native logical row view");
            let mut state = EvalRuntimeState::default();
            state.logical_row = Some(logical_row);

            assert_eq!(
                super::evaluate_typed_expression(&row, &expression, &mut state).unwrap(),
                ScalarValue::Integer(41)
            );
        });
    }

    #[test]
    fn typed_condition_entry_coerces_values_and_returns_saved_fault_cause() {
        let truthy = lower_typed_test_expression(typed_test_literal("string", "'yes'"));
        let missing = lower_typed_test_expression(typed_test_literal("missing", "."));
        let bad_numeric = lower_typed_test_expression(json!({
            "tag": "binary",
            "span": typed_test_span(),
            "origin": null,
            "operator": "+",
            "left": typed_test_literal("string", "'bad'"),
            "right": typed_test_literal("numeric", "1")
        }));
        let unknown_function = lower_typed_test_expression(json!({
            "tag": "call",
            "span": typed_test_span(),
            "origin": null,
            "name": "mystery",
            "arguments": []
        }));
        let deferred_cause = "unexpected token: **";
        let deferred = lower_typed_test_expression(json!({
            "tag": "deferred",
            "span": typed_test_span(),
            "origin": null,
            "error_kind": "unsupported_expression",
            "native_cause": deferred_cause,
            "where_syntax_summary": "saved summary",
            "display_text": "not valid (",
            "apply_references": []
        }));

        pyo3::Python::attach(|py| {
            let row = pyo3::types::PyDict::new(py);
            let mut state = EvalRuntimeState::default();
            assert!(super::evaluate_typed_condition(&row, &truthy, &mut state).unwrap());
            assert!(!super::evaluate_typed_condition(&row, &missing, &mut state).unwrap());
            assert_eq!(
                super::evaluate_typed_condition(&row, &bad_numeric, &mut state).unwrap_err(),
                "numeric value required, got 'bad'"
            );
            assert_eq!(
                super::evaluate_typed_condition(&row, &unknown_function, &mut state).unwrap_err(),
                "unsupported function: mystery"
            );
            assert_eq!(
                super::evaluate_typed_condition(&row, &deferred, &mut state).unwrap_err(),
                deferred_cause
            );
        });
    }

    #[test]
    fn typed_eval_power_keeps_right_association_parentheses_unary_and_scalar_domains() {
        let right_associative = lower_typed_test_expression(json!({
            "tag": "binary",
            "span": typed_test_span(),
            "origin": null,
            "operator": "**",
            "left": {"tag": "name", "span": typed_test_span(), "origin": null, "name": "a"},
            "right": {
                "tag": "binary",
                "span": typed_test_span(),
                "origin": null,
                "operator": "**",
                "left": typed_test_literal("numeric", "2"),
                "right": typed_test_literal("numeric", "3")
            }
        }));
        let explicitly_grouped = lower_typed_test_expression(json!({
            "tag": "binary",
            "span": typed_test_span(),
            "origin": null,
            "operator": "**",
            "left": {
                "tag": "binary",
                "span": typed_test_span(),
                "origin": null,
                "operator": "**",
                "left": {"tag": "name", "span": typed_test_span(), "origin": null, "name": "a"},
                "right": typed_test_literal("numeric", "2")
            },
            "right": typed_test_literal("numeric", "3")
        }));
        let unary_outside_power = lower_typed_test_expression(json!({
            "tag": "unary",
            "span": typed_test_span(),
            "origin": null,
            "operator": "-",
            "operand": {
                "tag": "binary",
                "span": typed_test_span(),
                "origin": null,
                "operator": "**",
                "left": typed_test_literal("numeric", "2"),
                "right": typed_test_literal("numeric", "2")
            }
        }));
        let unary_power_exponent = lower_typed_test_expression(json!({
            "tag": "binary",
            "span": typed_test_span(),
            "origin": null,
            "operator": "**",
            "left": typed_test_literal("numeric", "2"),
            "right": {
                "tag": "unary",
                "span": typed_test_span(),
                "origin": null,
                "operator": "-",
                "operand": typed_test_literal("numeric", "2")
            }
        }));

        let scalar_cases = vec![
            (
                lower_typed_test_expression(typed_test_literal(
                    "numeric",
                    "9223372036854775807",
                )),
                ScalarValue::Integer(i64::MAX),
            ),
            (
                lower_typed_test_expression(typed_test_literal(
                    "numeric",
                    "-9223372036854775808",
                )),
                ScalarValue::Integer(i64::MIN),
            ),
            (
                lower_typed_test_expression(typed_test_literal(
                    "numeric",
                    "18446744073709551615",
                )),
                ScalarValue::Unsigned(u64::MAX),
            ),
            (
                lower_typed_test_expression(typed_test_literal("missing", ".")),
                ScalarValue::Null,
            ),
        ];

        pyo3::Python::attach(|py| {
            let row = pyo3::types::PyDict::new(py);
            let mut state = EvalRuntimeState::default();
            state.logical_row = Some(typed_test_logical_row(&[("a", ScalarValue::Integer(2))]));

            assert_eq!(
                super::evaluate_typed_expression(&row, &right_associative, &mut state).unwrap(),
                ScalarValue::Number(256.0)
            );
            assert_eq!(
                super::evaluate_typed_expression(&row, &explicitly_grouped, &mut state).unwrap(),
                ScalarValue::Number(64.0)
            );
            assert_eq!(
                super::evaluate_typed_expression(&row, &unary_outside_power, &mut state).unwrap(),
                ScalarValue::Number(-4.0)
            );
            assert_eq!(
                super::evaluate_typed_expression(&row, &unary_power_exponent, &mut state).unwrap(),
                ScalarValue::Number(0.25)
            );
            for (expression, expected) in &scalar_cases {
                assert_eq!(
                    super::evaluate_typed_expression(&row, expression, &mut state).unwrap(),
                    *expected
                );
            }
        });
    }

    #[test]
    fn typed_eval_preserves_quoted_backslash_and_format_token_contents() {
        let quoted_operator = lower_typed_test_expression(typed_test_literal("string", "'a ** 2'"));
        let quoted_backslashes =
            lower_typed_test_expression(typed_test_literal("string", r#"'C:\\tmp'"#));
        let format_variable =
            lower_typed_test_expression(typed_test_literal("format", "best12."));
        let format_text = lower_typed_test_expression(typed_test_literal("format", "8."));

        pyo3::Python::attach(|py| {
            let row = pyo3::types::PyDict::new(py);
            let mut state = EvalRuntimeState::default();
            state.logical_row = Some(typed_test_logical_row(&[(
                "best12.",
                ScalarValue::Text("format token value".to_string()),
            )]));

            assert_eq!(
                super::evaluate_typed_expression(&row, &quoted_operator, &mut state).unwrap(),
                ScalarValue::Text("a ** 2".to_string())
            );
            assert_eq!(
                super::evaluate_typed_expression(&row, &quoted_backslashes, &mut state).unwrap(),
                ScalarValue::Text(r"C:\\tmp".to_string())
            );
            assert_eq!(
                super::evaluate_typed_expression(&row, &format_variable, &mut state).unwrap(),
                ScalarValue::Text("format token value".to_string())
            );
            assert_eq!(
                super::evaluate_typed_expression(&row, &format_text, &mut state).unwrap(),
                ScalarValue::Text("8.".to_string())
            );

            let rejected = lower_typed_test_expression(json!({
                "tag": "binary",
                "span": typed_test_span(),
                "origin": null,
                "operator": "*",
                "left": typed_test_literal("string", "'not numeric'"),
                "right": typed_test_literal("numeric", "2")
            }));
            assert_eq!(
                super::evaluate_typed_expression(&row, &rejected, &mut state).unwrap_err(),
                "numeric value required, got 'not numeric'"
            );
        });
    }

    #[test]
    fn typed_lowered_concat_evaluates_parts_in_order_and_treats_null_as_empty() {
        let expression = lower_typed_test_expression(json!({
            "tag": "concat",
            "span": typed_test_span(),
            "origin": null,
            "parts": [
                typed_test_literal("string", "'left'"),
                typed_test_literal("missing", "."),
                typed_test_literal("string", "'right'")
            ]
        }));

        pyo3::Python::attach(|py| {
            let row = pyo3::types::PyDict::new(py);
            let mut state = EvalRuntimeState::default();
            assert!(matches!(
                evaluate_expression(&row, &expression, &mut state).unwrap(),
                ScalarValue::Text(value) if value == "leftright"
            ));
        });
    }

    #[test]
    fn typed_concat_formats_scalars_and_evaluates_each_operand_once_in_order() {
        let numeric_concat = lower_typed_test_expression(json!({
            "tag": "concat",
            "span": typed_test_span(),
            "origin": null,
            "parts": [
                typed_test_literal("numeric", "1"),
                typed_test_literal("missing", "."),
                typed_test_literal("numeric", "2")
            ]
        }));
        let add_inside_concat = lower_typed_test_expression(json!({
            "tag": "concat",
            "span": typed_test_span(),
            "origin": null,
            "parts": [
                typed_test_literal("string", "'a'"),
                {
                    "tag": "binary",
                    "span": typed_test_span(),
                    "origin": null,
                    "operator": "+",
                    "left": typed_test_literal("numeric", "1"),
                    "right": typed_test_literal("numeric", "2")
                }
            ]
        }));
        let grouped_concat = lower_typed_test_expression(json!({
            "tag": "concat",
            "span": typed_test_span(),
            "origin": null,
            "parts": [
                typed_test_literal("string", "'a'"),
                {
                    "tag": "concat",
                    "span": typed_test_span(),
                    "origin": null,
                    "parts": [
                        typed_test_literal("string", "'b'"),
                        typed_test_literal("string", "'c'")
                    ]
                }
            ]
        }));
        let apply_call = |value: &str| {
            json!({
                "tag": "call",
                "span": typed_test_span(),
                "origin": null,
                "name": "apply",
                "arguments": [
                    typed_test_literal("string", "'tick'"),
                    typed_test_literal("string", &format!("'{value}'"))
                ]
            })
        };
        let add = lower_typed_test_binary("+", apply_call("left"), apply_call("right"));
        let concat = lower_typed_test_expression(json!({
            "tag": "concat",
            "span": typed_test_span(),
            "origin": null,
            "parts": [apply_call("left"), apply_call("middle"), apply_call("right")]
        }));

        pyo3::Python::attach(|py| {
            let calls = pyo3::types::PyList::empty(py);
            let globals = pyo3::types::PyDict::new(py);
            globals.set_item("calls", &calls).unwrap();
            let callback = py
                .eval(
                    c"lambda value: (calls.append(value), value)[1]",
                    Some(&globals),
                    None,
                )
                .expect("test callback is callable");
            let row = pyo3::types::PyDict::new(py);
            let mut state = EvalRuntimeState::default();
            state
                .apply_registry
                .insert("tick".to_string(), callback.unbind());

            assert_eq!(
                super::evaluate_typed_expression(&row, &numeric_concat, &mut state).unwrap(),
                ScalarValue::Text("12".to_string())
            );
            assert_eq!(
                super::evaluate_typed_expression(&row, &add_inside_concat, &mut state).unwrap(),
                ScalarValue::Text("a3".to_string())
            );
            assert_eq!(
                super::evaluate_typed_expression(&row, &grouped_concat, &mut state).unwrap(),
                ScalarValue::Text("abc".to_string())
            );
            assert_eq!(
                super::evaluate_typed_expression(&row, &add, &mut state).unwrap(),
                ScalarValue::Text("leftright".to_string())
            );
            assert_eq!(
                super::evaluate_typed_expression(&row, &concat, &mut state).unwrap(),
                ScalarValue::Text("leftmiddleright".to_string())
            );
            assert_eq!(
                calls.extract::<Vec<String>>().unwrap(),
                vec!["left", "right", "left", "middle", "right"]
            );
        });
    }

    fn assert_eval_null(expr_text: &str) {
        match evaluate_expression_text(expr_text) {
            Ok(ScalarValue::Null) => {}
            other => panic!("{expr_text} evaluated to {other:?}, expected NULL"),
        }
    }

    fn assert_eval_number(expr_text: &str, expected: f64) {
        match evaluate_expression_text(expr_text) {
            Ok(ScalarValue::Number(value)) => assert_eq!(value, expected, "{expr_text}"),
            other => panic!("{expr_text} evaluated to {other:?}, expected number {expected:?}"),
        }
    }

    fn assert_eval_bool(expr_text: &str, expected: bool) {
        match evaluate_expression_text(expr_text) {
            Ok(ScalarValue::Bool(value)) => assert_eq!(value, expected, "{expr_text}"),
            other => panic!("{expr_text} evaluated to {other:?}, expected bool {expected:?}"),
        }
    }

    #[test]
    fn binary_arithmetic_propagates_null_before_numeric_coercion() {
        for op in ["+", "-", "*", "/", "**", "mod"] {
            assert_eval_null(&format!(". {op} 1"));
            assert_eval_null(&format!("1 {op} ."));
            assert_eval_null(&format!(". {op} ."));
        }
    }

    #[test]
    fn unary_negation_propagates_null() {
        // Literal/variable fast path.
        assert_eval_null("-.");
        // Recursive evaluation path with a NULL-producing inner expression.
        assert_eval_null("-(1 + .)");
    }

    #[test]
    fn non_null_arithmetic_results_are_unchanged() {
        assert_eval_number("1 + 2", 3.0);
        assert_eval_number("7 / 2", 3.5);
        assert_eval_number("2 ** 3", 8.0);
        assert_eval_number("5 mod 3", 2.0);
        assert_eval_number("-(1 + 2)", -3.0);
    }

    #[test]
    fn comparison_and_logical_null_semantics_are_unchanged() {
        assert_eval_bool(". = .", true);
        assert_eval_bool(". = 1", false);
        assert_eval_bool("1 = .", false);
        assert_eval_bool(". != .", false);
        assert_eval_bool("not .", true);
        assert_eval_bool(". or true", true);
        assert_eval_bool(". or false", false);
        assert_eval_bool(". and false", false);
        assert_eval_bool(". and true", false);
    }

    #[test]
    fn null_ordering_semantics_are_unchanged() {
        assert_eval_bool("1 > .", true);
        assert_eval_bool(". < 1", true);
        assert_eval_bool(". > 1", false);
        assert_eval_bool(". >= .", true);
        assert_eval_bool(". > .", false);
        assert_eval_bool("1 >= .", true);
        assert_eval_bool("1 <= .", false);
    }

    #[test]
    fn invalid_decimal_numeric_conversion_carries_operation_context() {
        let error = coerce_scalar_to_number_for_operation(
            &ScalarValue::Decimal("not-a-number".to_string()),
            "numeric expression",
        )
        .unwrap_err();
        assert!(error.contains("numeric expression"), "{error}");
        assert!(error.contains("not-a-number"), "{error}");
    }

    #[test]
    fn invalid_decimal_boolean_conversion_fails_closed_with_value_and_type() {
        let error = coerce_scalar_to_bool(&ScalarValue::Decimal("not-a-number".to_string())).unwrap_err();
        assert!(error.contains("boolean"), "{error}");
        assert!(error.contains("not-a-number"), "{error}");
    }

    #[test]
    fn invalid_decimal_boolean_consumer_fails_closed_in_expression_evaluation() {
        pyo3::Python::attach(|py| {
            let row = pyo3::types::PyDict::new(py);
            let mut state = EvalRuntimeState::default();
            let expr = Expr::UnaryNot(Box::new(Expr::Literal(ScalarValue::Decimal(
                "not-a-number".to_string(),
            ))));
            let error = evaluate_expression(&row, &expr, &mut state).unwrap_err();
            assert!(error.contains("boolean"), "{error}");
            assert!(error.contains("not-a-number"), "{error}");
        });
    }

    #[test]
    fn invalid_decimal_numeric_consumer_fails_closed_in_expression_evaluation() {
        pyo3::Python::attach(|py| {
            let row = pyo3::types::PyDict::new(py);
            let mut state = EvalRuntimeState::default();
            let expr = Expr::Binary {
                left: Box::new(Expr::Literal(ScalarValue::Decimal("not-a-number".to_string()))),
                op: BinaryOp::Add,
                right: Box::new(Expr::Literal(ScalarValue::Number(1.0))),
            };
            let error = evaluate_expression(&row, &expr, &mut state).unwrap_err();
            assert!(error.contains("numeric"), "{error}");
            assert!(error.contains("not-a-number"), "{error}");
        });
    }

    #[test]
    fn valid_decimal_conversions_keep_existing_results() {
        assert_eq!(
            coerce_scalar_to_number_for_operation(
                &ScalarValue::Decimal("1.5".to_string()),
                "numeric expression",
            )
            .unwrap(),
            1.5
        );
        assert!(coerce_scalar_to_bool(&ScalarValue::Decimal("1".to_string())).unwrap());
        assert!(!coerce_scalar_to_bool(&ScalarValue::Decimal("0".to_string())).unwrap());
        assert!(!coerce_scalar_to_bool(&ScalarValue::Decimal("NaN".to_string())).unwrap());
        assert!(coerce_scalar_to_bool(&ScalarValue::Decimal("Infinity".to_string())).unwrap());
    }

    #[test]
    fn arithmetic_null_propagation_precedes_decimal_coercion() {
        let result = eval_binary_operands(
            &EvalOperand::Scalar(ScalarValue::Null),
            &BinaryOp::Add,
            &EvalOperand::Scalar(ScalarValue::Decimal("1.5".to_string())),
        )
        .unwrap();
        assert!(matches!(result, ScalarValue::Null));
    }

    #[test]
    fn unsupported_list_literal_fails_closed_without_python_eval_fallback() {
        pyo3::Python::attach(|py| {
            let row = pyo3::types::PyDict::new(py);
            let mut state = EvalRuntimeState::default();
            let error = evaluate_scalar_expression(&row, "[1, 2, 3]", &mut state)
                .expect_err("list literal must fail closed without a Python eval fallback");
            assert!(error.contains("expression expected"), "{error}");
        });
    }

    #[test]
    fn native_parse_error_is_returned_unchanged_without_python_fallback() {
        pyo3::Python::attach(|py| {
            let row = pyo3::types::PyDict::new(py);
            let mut state = EvalRuntimeState::default();
            let error = evaluate_scalar_expression(&row, "1 . 2", &mut state)
                .expect_err("misplaced missing literal must fail closed");
            assert_eq!(
                error, "unexpected token at end of expression",
                "native parse error must propagate without a Python fallback"
            );
        });
    }

    #[test]
    fn native_power_missing_literal_and_builtin_guard_evaluate_without_fallback() {
        pyo3::Python::attach(|py| {
            let row = pyo3::types::PyDict::new(py);
            let mut state = EvalRuntimeState::default();

            let power = evaluate_scalar_expression(&row, "2 ** 3", &mut state).unwrap();
            assert_eq!(scalar_to_number(&power).unwrap(), 8.0, "** stays native");

            let missing = evaluate_scalar_expression(&row, ".", &mut state).unwrap();
            assert!(matches!(missing, ScalarValue::Null), "missing literal stays native");

            let builtin = evaluate_scalar_expression(&row, "abs(-3)", &mut state).unwrap();
            assert_eq!(
                scalar_to_number(&builtin).unwrap(),
                3.0,
                "builtin dispatch stays native"
            );
        });
    }
}
