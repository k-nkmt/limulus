use chrono::{Datelike, Timelike};
use crate::expressions::{
    py_to_scalar, scalar_is_missing, scalar_to_date, scalar_to_datetime, scalar_to_number,
    scalar_to_pyobject, scalar_to_text, scalar_to_time, ScalarValue,
};
use crate::runtime::EvalRuntimeState;
use crate::row_view::LogicalRowAccess;
use chrono::NaiveDate;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use regex::{Regex, RegexBuilder};
use std::collections::HashMap;

// Format parsing and number formatting
pub(crate) fn parse_width_precision(format_name: &str) -> Option<(usize, Option<usize>)> {
    let (width_part, precision_part) = format_name.split_once('.').unwrap_or((format_name, ""));
    let width = width_part.parse::<usize>().ok()?;
    if precision_part.is_empty() {
        return Some((width, None));
    }
    let precision = precision_part.parse::<usize>().ok()?;
    Some((width, Some(precision)))
}

pub(crate) fn normalize_format_name(value: &ScalarValue) -> Result<(String, bool), String> {
    let raw = scalar_to_text(value)?.trim().to_ascii_lowercase();
    let has_dot = raw.contains('.');
    Ok((raw.trim_end_matches('.').to_string(), has_dot))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CatalogLookupState {
    Absent,
    PresentNoMatch,
}

fn normalize_custom_format_name(value: &ScalarValue) -> Result<(String, bool, bool), String> {
    let raw = scalar_to_text(value)?.trim().to_ascii_lowercase();
    let has_dot = raw.contains('.');
    let is_character = raw.starts_with('$');
    let normalized = raw
        .strip_prefix('$')
        .unwrap_or(raw.as_str())
        .trim_end_matches('.')
        .to_string();
    Ok((normalized, has_dot, is_character))
}

fn lookup_put_catalog(
    state: &EvalRuntimeState,
    format_name: &str,
    is_character: bool,
    value: &ScalarValue,
) -> Result<ScalarValue, CatalogLookupState> {
    let catalogs = if is_character {
        &state.character_format_catalogs
    } else {
        &state.numeric_format_catalogs
    };
    let Some(entries) = catalogs.get(format_name) else {
        return Err(CatalogLookupState::Absent);
    };
    for (candidate, rendered) in entries {
        if candidate == value {
            return Ok(ScalarValue::Text(rendered.clone()));
        }
    }
    Err(CatalogLookupState::PresentNoMatch)
}

fn lookup_input_catalog(
    state: &EvalRuntimeState,
    format_name: &str,
    value: &ScalarValue,
) -> Result<ScalarValue, CatalogLookupState> {
    let Some(entries) = state.informat_catalogs.get(format_name) else {
        return Err(CatalogLookupState::Absent);
    };
    for (candidate, parsed) in entries {
        if candidate == value {
            return Ok(ScalarValue::Number(*parsed));
        }
    }
    Err(CatalogLookupState::PresentNoMatch)
}

pub(crate) fn requires_numeric_format_dot(format_name: &str, has_dot: bool) -> bool {
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

// Format/input built-in functions
pub(crate) fn put_builtin(value: &ScalarValue, format_name: &str) -> Result<ScalarValue, String> {
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
        "yymmdd6" => Ok(ScalarValue::Text(
            scalar_to_date(value)?.format("%y%m%d").to_string(),
        )),
        "yymmdd8" => Ok(ScalarValue::Text(
            scalar_to_date(value)?.format("%Y%m%d").to_string(),
        )),
        "yymmdd10" => Ok(ScalarValue::Text(
            scalar_to_date(value)?.format("%Y-%m-%d").to_string(),
        )),
        "e8601da" => Ok(ScalarValue::Text(scalar_to_date(value)?.format("%Y-%m-%d").to_string())),
        "e8601dt" => Ok(ScalarValue::Text(
            scalar_to_datetime(value)?.format("%Y-%m-%dT%H:%M:%S").to_string(),
        )),
        "time" => Ok(ScalarValue::Text(scalar_to_time(value)?.format("%H:%M:%S").to_string())),
        _ => Err(format!("Unsupported format: {format_name}")),
    }
}

fn parse_time_text(text: &str) -> Result<chrono::NaiveTime, String> {
    chrono::NaiveTime::parse_from_str(text, "%H:%M:%S")
        .or_else(|_| chrono::NaiveTime::parse_from_str(text, "%H:%M"))
        .map_err(|_| format!("time value required, got '{text}'"))
}

pub(crate) fn input_builtin(value: &ScalarValue, format_name: &str) -> Result<ScalarValue, String> {
    let raw = scalar_to_text(value)?;
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
            chrono::NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S")
                .map_err(|_| format!("Invalid datetime value: {text}"))?,
        )),
        "time" => Ok(ScalarValue::Time(parse_time_text(text)?)),
        _ => Err(format!("Unsupported informat: {format_name}")),
    }
}

pub(crate) fn hour_builtin(value: &ScalarValue) -> Result<ScalarValue, String> {
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

// PRX (regex) helper functions
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
        .map_err(|error| format!("Invalid prxmatch pattern: {error}"))
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
        .map_err(|error| format!("Invalid prxchange pattern: {error}"))?;
    Ok((compiled, replacement.to_string()))
}

// Unicode scalar position helpers.
// Public string functions expose 1-based Unicode scalar positions and scalar counts;
// UTF-8 byte offsets stay internal to slicing/search and must never reach public results.
fn byte_offset_at_scalar(value: &str, zero_based_scalar: usize) -> usize {
    value
        .char_indices()
        .nth(zero_based_scalar)
        .map(|(byte_offset, _)| byte_offset)
        .unwrap_or(value.len())
}

fn scalar_count_before_byte(value: &str, byte_offset: usize) -> Result<usize, String> {
    if byte_offset > value.len() || !value.is_char_boundary(byte_offset) {
        return Err(format!(
            "byte offset {byte_offset} is not a char boundary of a {}-byte string",
            value.len()
        ));
    }
    Ok(value[..byte_offset].chars().count())
}

fn scalar_slice(value: &str, start: usize, length: Option<usize>) -> String {
    let byte_start = byte_offset_at_scalar(value, start);
    let byte_end = match length {
        None => value.len(),
        Some(length) => byte_offset_at_scalar(value, start.saturating_add(length)),
    };
    let end = byte_end.min(value.len());
    if byte_start >= end {
        return String::new();
    }
    value[byte_start..end].to_string()
}

fn find_scalar_position(
    value: &str,
    pattern: &str,
    start: usize,
    ignore_case: bool,
) -> Result<Option<usize>, String> {
    let scalar_total = value.chars().count();
    if pattern.is_empty() {
        return Ok(Some(start.min(scalar_total)));
    }

    if !ignore_case {
        let byte_start = byte_offset_at_scalar(value, start);
        if byte_start >= value.len() {
            return Ok(None);
        }
        return match value[byte_start..].find(pattern) {
            Some(relative) => {
                let absolute = byte_start + relative;
                Ok(Some(scalar_count_before_byte(value, absolute)?))
            }
            None => Ok(None),
        };
    }

    // Case-insensitive search lowercases per-char and keeps a map from every
    // lowercase scalar back to the original scalar ordinal it came from; chars
    // whose lowercase expands to several scalars all map to the same ordinal.
    // Match positions are never derived from lowercase-buffer byte offsets.
    let mut lowered_value = String::with_capacity(value.len());
    let mut lowered_origins: Vec<usize> = Vec::with_capacity(value.len());
    let mut lowered_start_byte: Option<usize> = None;
    for (ordinal, ch) in value.chars().enumerate() {
        if ordinal >= start && lowered_start_byte.is_none() {
            lowered_start_byte = Some(lowered_value.len());
        }
        for lowered in ch.to_lowercase() {
            lowered_value.push(lowered);
            lowered_origins.push(ordinal);
        }
    }
    let lowered_pattern: String = pattern.chars().flat_map(char::to_lowercase).collect();
    let lowered_start_byte = lowered_start_byte.unwrap_or(lowered_value.len());

    match lowered_value[lowered_start_byte..].find(&lowered_pattern) {
        Some(relative) => {
            let absolute = lowered_start_byte + relative;
            let lowered_ordinal = scalar_count_before_byte(&lowered_value, absolute)?;
            Ok(lowered_origins.get(lowered_ordinal).copied())
        }
        None => Ok(None),
    }
}

// Main function evaluation dispatch
pub fn evaluate_function(
    py: Python<'_>,
    name: &str,
    args: &[ScalarValue],
    state: &mut EvalRuntimeState,
) -> Result<ScalarValue, String> {
    evaluate_function_with_context(py, name, args, &[], state)
}

pub(crate) fn evaluate_function_with_context(
    py: Python<'_>,
    name: &str,
    args: &[ScalarValue],
    argument_contexts: &[String],
    state: &mut EvalRuntimeState,
) -> Result<ScalarValue, String> {
    fn call_python_apply_target(
        py: Python<'_>,
        target: &Bound<'_, PyAny>,
        args: &[ScalarValue],
        argument_contexts: &[String],
    ) -> Result<ScalarValue, String> {
        let py_args = args
            .iter()
            .enumerate()
            .map(|(index, arg)| {
                scalar_to_pyobject(py, arg).map_err(|error| {
                    let context = argument_contexts
                        .get(index)
                        .map(String::as_str)
                        .filter(|value| !value.is_empty())
                        .unwrap_or("argument");
                    format!(
                        "apply() argument {index} ({context}) could not cross the Python boundary: {error}"
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let tuple = PyTuple::new(py, py_args.iter().map(|item| item.bind(py))).map_err(|error| error.to_string())?;
        let result = target.call1(tuple).map_err(|error| error.to_string())?;
        py_to_scalar(&result)
            .map_err(|error| format!("apply() result conversion failed: {error}"))
    }

    fn parse_steps(raw: Option<&ScalarValue>, default_value: isize) -> Result<isize, String> {
        let Some(value) = raw else {
            return Ok(default_value);
        };
        Ok(scalar_to_number(value)? as isize)
    }

    enum ShiftInput<'a> {
        Value(ScalarValue),
        Variable(&'a str),
    }

    #[derive(Clone, Copy)]
    enum ShiftMode {
        Auto,
        ValueQueue,
        VariableLookup,
    }

    fn shift_variable_lookup(
        state: &EvalRuntimeState,
        variable_name: &str,
        steps: isize,
        default: ScalarValue,
    ) -> Result<ScalarValue, String> {
        let target_index = state.row_index + steps;
        let logical_row = state
            .logical_row
            .as_ref()
            .ok_or_else(|| "logical row view is required for shift() variable lookup".to_string())?;
        if target_index < 0 || (target_index as usize) >= logical_row.row_count() {
            return Ok(default);
        }

        let target_row = logical_row
            .for_row_index(target_index as usize)
            .map_err(|error| error.to_string())?;
        let Some(value) = target_row
            .resolve(variable_name)
            .map_err(|error| error.to_string())?
        else {
            return Ok(default.clone());
        };
        match value {
            crate::row_view::ResolvedValue::SourceScalar(value) => {
                Ok(value.as_scalar().map_err(|error| error.to_string())?.clone())
            }
            crate::row_view::ResolvedValue::SourceCell(cell) => Err(format!(
                "complex access is unsupported for column '{variable_name}' during shift(): {}",
                cell.access_error("shift()")
            )),
            crate::row_view::ResolvedValue::Owned(value)
            | crate::row_view::ResolvedValue::Automatic(value) => Ok(
                value
                    .as_scalar()
                    .map_err(|error| error.to_string())?
                    .clone(),
            ),
        }
    }

    fn shift_value_queue(
        state: &mut EvalRuntimeState,
        value: ScalarValue,
        steps: isize,
        default: ScalarValue,
    ) -> Result<ScalarValue, String> {
        if steps > 0 {
            return Err("shift() with positive offset requires variable name text argument".to_string());
        }
        if steps == 0 {
            return Ok(value);
        }

        let offset = (-steps) as usize;
        let queue = state.lag_queues.entry(offset).or_default();
        queue.push(value);
        if queue.len() <= offset {
            return Ok(default);
        }
        Ok(queue.remove(0))
    }

    fn shift_family(
        state: &mut EvalRuntimeState,
        input: ShiftInput<'_>,
        steps: isize,
        default: ScalarValue,
        mode: ShiftMode,
    ) -> Result<ScalarValue, String> {
        match mode {
            ShiftMode::ValueQueue => {
                let ShiftInput::Value(value) = input else {
                    return Err("shift() queue mode requires a value input".to_string());
                };
                shift_value_queue(state, value, steps, default)
            }
            ShiftMode::VariableLookup => {
                match input {
                    ShiftInput::Variable(name) => shift_variable_lookup(state, name, steps, default),
                    ShiftInput::Value(ScalarValue::Text(name)) => {
                        shift_variable_lookup(state, name.as_str(), steps, default)
                    }
                    ShiftInput::Value(_) => {
                        Err("shift() variable lookup mode requires variable name text".to_string())
                    }
                }
            }
            ShiftMode::Auto => match input {
                ShiftInput::Variable(name) => shift_variable_lookup(state, name, steps, default),
                ShiftInput::Value(ScalarValue::Text(name)) => shift_variable_lookup(state, name.as_str(), steps, default),
                ShiftInput::Value(value) => shift_value_queue(state, value, steps, default),
            },
        }
    }

    let lowered = name.to_lowercase();
    match lowered.as_str() {
        "put" => {
            if args.len() != 2 {
                return Err("put() expects 2 arguments".to_string());
            }
            let (format_name, has_dot, is_character) = normalize_custom_format_name(&args[1])?;
            match lookup_put_catalog(state, &format_name, is_character, &args[0]) {
                Ok(value) => return Ok(value),
                Err(CatalogLookupState::PresentNoMatch) => {
                    return Ok(ScalarValue::Text(scalar_to_text(&args[0])?));
                }
                Err(CatalogLookupState::Absent) => {}
            }
            if is_character {
                return Err(format!("Unsupported format: {}", scalar_to_text(&args[1])?));
            }
            if requires_numeric_format_dot(&format_name, has_dot) {
                return Err(format!("Unsupported format: {}", scalar_to_text(&args[1])?));
            }
            put_builtin(&args[0], &format_name)
        }
        "input" => {
            if args.len() != 2 {
                return Err("input() expects 2 arguments".to_string());
            }
            let (format_name, _) = normalize_format_name(&args[1])?;
            match lookup_input_catalog(state, &format_name, &args[0]) {
                Ok(value) => return Ok(value),
                Err(CatalogLookupState::PresentNoMatch) => return Ok(ScalarValue::Null),
                Err(CatalogLookupState::Absent) => {}
            }
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
            let source = scalar_to_text(&args[0])?;
            let start_index = scalar_to_number(&args[1])? as isize - 1;
            let start = start_index.max(0) as usize;
            let length = if args.len() == 3 {
                Some(scalar_to_number(&args[2])?.max(0.0) as usize)
            } else {
                None
            };
            Ok(ScalarValue::Text(scalar_slice(&source, start, length)))
        }
        "scan" => {
            if args.len() < 2 || args.len() > 3 {
                return Err("scan() expects 2 or 3 arguments".to_string());
            }
            let source = scalar_to_text(&args[0])?;
            let index = scalar_to_number(&args[1])? as isize - 1;
            let delimiters = if args.len() == 3 {
                scalar_to_text(&args[2])?
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
            let source = scalar_to_text(&args[0])?;
            if args.len() == 1 {
                return Ok(ScalarValue::Text(source.chars().filter(|ch| !ch.is_whitespace()).collect()));
            }
            let remove_chars = scalar_to_text(&args[1])?;
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
            Ok(ScalarValue::Text(scalar_to_text(&args[0])?.trim().to_string()))
        }
        "upcase" => {
            if args.len() != 1 {
                return Err("upcase() expects 1 argument".to_string());
            }
            Ok(ScalarValue::Text(scalar_to_text(&args[0])?.to_uppercase()))
        }
        "lowcase" => {
            if args.len() != 1 {
                return Err("lowcase() expects 1 argument".to_string());
            }
            Ok(ScalarValue::Text(scalar_to_text(&args[0])?.to_lowercase()))
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
            // The bridge normalizes `var ** lit` assignments to pow(var, lit),
            // so this builtin must keep the binary-arithmetic NULL guard of
            // BinaryOp::Pow: a NULL operand propagates as NULL instead of
            // silently folding to 0 through numeric coercion.
            if matches!(args[0], ScalarValue::Null) || matches!(args[1], ScalarValue::Null) {
                return Ok(ScalarValue::Null);
            }
            let base = scalar_to_number(&args[0])?;
            let exponent = scalar_to_number(&args[1])?;
            Ok(ScalarValue::Number(base.powf(exponent)))
        }
        "mod" => {
            if args.len() != 2 {
                return Err("mod() expects 2 arguments".to_string());
            }
            // `mod(x, y)` is the function spelling of the binary Mod
            // operator, so it shares the same NULL-propagation boundary.
            if matches!(args[0], ScalarValue::Null) || matches!(args[1], ScalarValue::Null) {
                return Ok(ScalarValue::Null);
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
            shift_family(state, ShiftInput::Value(args[0].clone()), steps, default, ShiftMode::Auto)
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
            shift_family(state, ShiftInput::Value(value), -offset, default, ShiftMode::ValueQueue)
        }
        "lead" => {
            if args.is_empty() || args.len() > 3 {
                return Err("lead() expects 1 to 3 arguments".to_string());
            }
            let variable_name = scalar_to_text(&args[0])?;
            let offset = parse_steps(args.get(1), 1)?;
            if offset < 1 {
                return Err("lead() offset must be >= 1".to_string());
            }
            let default = if args.len() == 3 {
                args[2].clone()
            } else {
                ScalarValue::Null
            };
            shift_family(
                state,
                ShiftInput::Variable(&variable_name),
                offset,
                default,
                ShiftMode::VariableLookup,
            )
        }
        "apply" => {
            if args.is_empty() {
                return Err("apply() expects at least 1 argument".to_string());
            }

            let function_name = scalar_to_text(&args[0])?;
            let apply_args = &args[1..];
            let Some(target) = state.apply_registry.get(&function_name) else {
                return Err(format!("Unknown apply function: {function_name}"));
            };
            let target = target.bind(py);
            if !target.is_callable() {
                return Err(format!("Resolved apply target is not callable: {function_name}"));
            }
            call_python_apply_target(py, &target, apply_args, argument_contexts)
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
            let interval = scalar_to_text(&args[0])?.to_lowercase();
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
        "propcase" => {
            if args.len() != 1 {
                return Err("propcase() expects 1 argument".to_string());
            }
            let source = scalar_to_text(&args[0])?;
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
            let result = args.iter().try_fold(String::new(), |mut result, arg| {
                result.push_str(&scalar_to_text(arg)?);
                Ok::<_, String>(result)
            })?;
            Ok(ScalarValue::Text(result))
        }
        "cats" => {
            let result = args
                .iter()
                .map(|arg| scalar_to_text(arg).map(|value| value.trim().to_string()))
                .collect::<Result<Vec<_>, _>>()?
                .concat();
            Ok(ScalarValue::Text(result))
        }
        "catt" => {
            let result = args
                .iter()
                .map(|arg| scalar_to_text(arg).map(|value| value.trim_end().to_string()))
                .collect::<Result<Vec<_>, _>>()?
                .concat();
            Ok(ScalarValue::Text(result))
        }
        "catx" => {
            if args.is_empty() {
                return Err("catx() expects at least 1 argument".to_string());
            }
            let delimiter = scalar_to_text(&args[0])?;
            let mut parts = Vec::new();
            for arg in &args[1..] {
                if scalar_is_missing(arg) {
                    continue;
                }
                let trimmed = scalar_to_text(arg)?.trim().to_string();
                if !trimmed.is_empty() {
                    parts.push(trimmed);
                }
            }
            Ok(ScalarValue::Text(parts.join(&delimiter)))
        }
        "index" => {
            if args.len() != 2 {
                return Err("index() expects 2 arguments".to_string());
            }
            let source = scalar_to_text(&args[0])?;
            let excerpt = scalar_to_text(&args[1])?;
            if excerpt.is_empty() {
                return Ok(ScalarValue::Number(1.0));
            }
            match find_scalar_position(&source, &excerpt, 0, false)? {
                Some(ordinal) => Ok(ScalarValue::Number((ordinal + 1) as f64)),
                None => Ok(ScalarValue::Number(0.0)),
            }
        }
        "find" => {
            if args.len() < 2 || args.len() > 4 {
                return Err("find() expects 2 to 4 arguments".to_string());
            }
            let source = scalar_to_text(&args[0])?;
            let excerpt = scalar_to_text(&args[1])?;
            let start_index = if args.len() >= 3 {
                (scalar_to_number(&args[2])? as isize - 1).max(0) as usize
            } else {
                0
            };
            let modifiers = if args.len() >= 4 {
                scalar_to_text(&args[3])?.to_lowercase()
            } else {
                String::new()
            };
            let ignore_case = modifiers.contains('i');
            match find_scalar_position(&source, &excerpt, start_index, ignore_case)? {
                Some(ordinal) => Ok(ScalarValue::Number((ordinal + 1) as f64)),
                None => Ok(ScalarValue::Number(0.0)),
            }
        }
        "tranwrd" => {
            if args.len() != 3 {
                return Err("tranwrd() expects 3 arguments".to_string());
            }
            let source = scalar_to_text(&args[0])?;
            let target = scalar_to_text(&args[1])?;
            let replacement = scalar_to_text(&args[2])?;
            Ok(ScalarValue::Text(source.replace(&target, &replacement)))
        }
        "translate" => {
            if args.len() != 3 {
                return Err("translate() expects 3 arguments".to_string());
            }
            let source = scalar_to_text(&args[0])?;
            let to_chars: Vec<char> = scalar_to_text(&args[1])?.chars().collect();
            let from_chars: Vec<char> = scalar_to_text(&args[2])?.chars().collect();
            let mut map: HashMap<char, String> = HashMap::new();
            for (index, src_char) in from_chars.iter().enumerate() {
                map.insert(
                    *src_char,
                    if index < to_chars.len() {
                        to_chars[index].to_string()
                    } else {
                        String::new()
                    },
                );
            }
            let result: String = source
                .chars()
                .map(|ch| map.get(&ch).cloned().unwrap_or_else(|| ch.to_string()))
                .collect();
            Ok(ScalarValue::Text(result))
        }
        "length" => {
            if args.len() != 1 {
                return Err("length() expects 1 argument".to_string());
            }
            let source = scalar_to_text(&args[0])?;
            Ok(ScalarValue::Number(source.chars().count() as f64))
        }
        "lengthn" => {
            if args.len() != 1 {
                return Err("lengthn() expects 1 argument".to_string());
            }
            let source = scalar_to_text(&args[0])?;
            Ok(ScalarValue::Number(source.chars().count() as f64))
        }
        "strip" => {
            if args.len() != 1 {
                return Err("strip() expects 1 argument".to_string());
            }
            Ok(ScalarValue::Text(scalar_to_text(&args[0])?.trim().to_string()))
        }
        "reverse" => {
            if args.len() != 1 {
                return Err("reverse() expects 1 argument".to_string());
            }
            let source = scalar_to_text(&args[0])?;
            Ok(ScalarValue::Text(source.chars().rev().collect()))
        }
        "repeat" => {
            if args.len() != 2 {
                return Err("repeat() expects 2 arguments".to_string());
            }
            let source = scalar_to_text(&args[0])?;
            let count = scalar_to_number(&args[1])?.max(0.0) as usize;
            Ok(ScalarValue::Text(source.repeat(count)))
        }
        "countw" => {
            if args.is_empty() || args.len() > 2 {
                return Err("countw() expects 1 or 2 arguments".to_string());
            }
            let source = scalar_to_text(&args[0])?.trim().to_string();
            let delimiters = if args.len() == 2 {
                scalar_to_text(&args[1])?
            } else {
                " ".to_string()
            };
            let count = source
                .split(|ch: char| delimiters.contains(ch))
                .filter(|part| !part.is_empty())
                .count();
            Ok(ScalarValue::Number(count as f64))
        }
        "int" => {
            if args.len() != 1 {
                return Err("int() expects 1 argument".to_string());
            }
            Ok(ScalarValue::Number(scalar_to_number(&args[0])?.trunc()))
        }
        "sum" => {
            let total = args
                .iter()
                .filter(|arg| !scalar_is_missing(arg))
                .try_fold(0.0, |acc, arg| Ok::<f64, String>(acc + scalar_to_number(arg)?))?;
            Ok(ScalarValue::Number(total))
        }
        "mean" => {
            let values = args
                .iter()
                .filter(|arg| !scalar_is_missing(arg))
                .map(scalar_to_number)
                .collect::<Result<Vec<_>, _>>()?;
            if values.is_empty() {
                return Ok(ScalarValue::Number(f64::NAN));
            }
            let total = values.iter().sum::<f64>();
            Ok(ScalarValue::Number(total / values.len() as f64))
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
            let result = if value > 0.0 {
                1.0
            } else if value < 0.0 {
                -1.0
            } else {
                0.0
            };
            Ok(ScalarValue::Number(result))
        }
        "cmiss" => {
            let count = args.iter().filter(|arg| scalar_is_missing(arg)).count();
            Ok(ScalarValue::Number(count as f64))
        }
        "prxmatch" => {
            if args.len() != 2 {
                return Err("prxmatch() expects 2 arguments".to_string());
            }
            let pattern_raw = scalar_to_text(&args[0])?;
            let source = scalar_to_text(&args[1])?;
            let regex = compile_prx_pattern(&pattern_raw)?;
            match regex.find(&source) {
                Some(found) => {
                    let ordinal = scalar_count_before_byte(&source, found.start())?;
                    Ok(ScalarValue::Number((ordinal + 1) as f64))
                }
                None => Ok(ScalarValue::Number(0.0)),
            }
        }
        "prxchange" => {
            if args.len() != 3 {
                return Err("prxchange() expects 3 arguments".to_string());
            }
            let pattern_raw = scalar_to_text(&args[0])?;
            let times = scalar_to_number(&args[1])? as isize;
            let source = scalar_to_text(&args[2])?;
            let (regex, replacement) = parse_prxchange_pattern(&pattern_raw)?;
            if times <= 0 {
                return Ok(ScalarValue::Text(regex.replace_all(&source, replacement.as_str()).into_owned()));
            }

            let mut result = source;
            let mut replaced = 0isize;
            while replaced < times {
                if regex.is_match(&result) {
                    result = regex.replacen(&result, 1, replacement.as_str()).into_owned();
                    replaced += 1;
                } else {
                    break;
                }
            }
            Ok(ScalarValue::Text(result))
        }
        _ => Err(format!("unsupported function: {name}")),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        byte_offset_at_scalar, evaluate_function, find_scalar_position, scalar_count_before_byte,
        scalar_slice,
    };
    use crate::expressions::ScalarValue;
    use crate::runtime::EvalRuntimeState;

    const ZWJ_FAMILY: &str = "👨‍👩‍👧‍👦";
    const JP_FLAG: &str = "🇯🇵";

    fn evaluate(name: &str, args: &[ScalarValue]) -> Result<ScalarValue, String> {
        pyo3::Python::attach(|py| {
            let mut state = EvalRuntimeState::default();
            evaluate_function(py, name, args, &mut state)
        })
    }

    fn text_arg(value: &str) -> ScalarValue {
        ScalarValue::Text(value.to_string())
    }

    fn number_arg(value: f64) -> ScalarValue {
        ScalarValue::Number(value)
    }

    fn assert_text(name: &str, args: &[ScalarValue], expected: &str) {
        match evaluate(name, args) {
            Ok(ScalarValue::Text(value)) => assert_eq!(value, expected),
            other => panic!("{name}{args:?} returned {other:?}, expected text {expected:?}"),
        }
    }

    fn assert_number(name: &str, args: &[ScalarValue], expected: f64) {
        match evaluate(name, args) {
            Ok(ScalarValue::Number(value)) => assert_eq!(value, expected),
            other => panic!("{name}{args:?} returned {other:?}, expected number {expected:?}"),
        }
    }

    fn assert_error_without_panic(name: &str, args: &[ScalarValue]) {
        let error = evaluate(name, args)
            .err()
            .unwrap_or_else(|| panic!("{name}{args:?} unexpectedly succeeded"));
        assert!(!error.is_empty(), "{name}{args:?} returned an empty error");
    }

    #[test]
    fn substr_ascii_scalar_positions() {
        assert_text("substr", &[text_arg("abc"), number_arg(2.0)], "bc");
        assert_text("substr", &[text_arg("abc"), number_arg(2.0), number_arg(1.0)], "b");
        assert_text("substr", &[text_arg("abc"), number_arg(-3.0)], "abc");
    }

    #[test]
    fn substr_cjk_scalar_positions() {
        assert_text("substr", &[text_arg("日本語"), number_arg(2.0)], "本語");
        assert_text("substr", &[text_arg("日本語"), number_arg(2.0), number_arg(1.0)], "本");
    }

    #[test]
    fn substr_and_length_treat_precomposed_and_decomposed_accents_as_written() {
        assert_number("length", &[text_arg("café")], 4.0);
        assert_text("substr", &[text_arg("café"), number_arg(4.0)], "é");

        let decomposed = "cafe\u{0301}";
        assert_number("length", &[text_arg(decomposed)], 5.0);
        assert_text("substr", &[text_arg(decomposed), number_arg(5.0)], "\u{0301}");
    }

    #[test]
    fn length_and_lengthn_count_unicode_scalars_not_bytes() {
        assert_number("length", &[text_arg("abc")], 3.0);
        assert_number("length", &[text_arg("日本語")], 3.0);
        assert_number("lengthn", &[text_arg("日本語")], 3.0);
        assert_number("length", &[text_arg("")], 0.0);
        assert_number("lengthn", &[text_arg("")], 0.0);
    }

    #[test]
    fn substr_counts_multi_scalar_emoji_and_flag_inputs_individually() {
        assert_number("length", &[text_arg(ZWJ_FAMILY)], 7.0);
        assert_text("substr", &[text_arg(ZWJ_FAMILY), number_arg(3.0), number_arg(1.0)], "👩");

        assert_number("length", &[text_arg(JP_FLAG)], 2.0);
        assert_text("substr", &[text_arg(JP_FLAG), number_arg(2.0), number_arg(1.0)], "\u{1F1F5}");
    }

    #[test]
    fn substr_out_of_range_start_and_oversized_length_stay_bounded() {
        assert_text("substr", &[text_arg("日本語"), number_arg(99.0)], "");
        assert_text("substr", &[text_arg("日本語"), number_arg(2.0), number_arg(999.0)], "本語");
    }

    #[test]
    fn substr_negative_length_is_empty() {
        assert_text("substr", &[text_arg("abc"), number_arg(1.0), number_arg(-5.0)], "");
    }

    #[test]
    fn string_function_arity_errors_do_not_panic() {
        assert_error_without_panic("substr", &[text_arg("a")]);
        assert_error_without_panic("substr", &[
            text_arg("a"),
            number_arg(1.0),
            number_arg(2.0),
            number_arg(3.0),
        ]);
        assert_error_without_panic("length", &[]);
        assert_error_without_panic("length", &[text_arg("a"), text_arg("b")]);
        assert_error_without_panic("lengthn", &[]);
        assert_error_without_panic("lengthn", &[text_arg("a"), text_arg("b")]);
    }

    #[test]
    fn scalar_helpers_agree_on_scalar_and_byte_boundaries() {
        assert_eq!(byte_offset_at_scalar("日本語", 0), 0);
        assert_eq!(byte_offset_at_scalar("日本語", 1), 3);
        assert_eq!(byte_offset_at_scalar("日本語", 3), 9);
        assert_eq!(byte_offset_at_scalar("日本語", 99), 9);

        assert_eq!(scalar_count_before_byte("日本語", 0).unwrap(), 0);
        assert_eq!(scalar_count_before_byte("日本語", 3).unwrap(), 1);
        assert_eq!(scalar_count_before_byte("日本語", 9).unwrap(), 3);
        assert!(scalar_count_before_byte("日本語", 1).is_err());
        assert!(scalar_count_before_byte("日本語", 10).is_err());

        assert_eq!(scalar_slice("日本語", 1, Some(1)), "本");
        assert_eq!(scalar_slice("日本語", 1, None), "本語");
        assert_eq!(scalar_slice("日本語", 5, None), "");
        assert_eq!(scalar_slice("日本語", 0, Some(0)), "");
    }

    #[test]
    fn find_scalar_position_maps_matches_to_original_scalar_ordinals() {
        assert_eq!(find_scalar_position("日本語のテキスト", "の", 0, false).unwrap(), Some(3));
        assert_eq!(find_scalar_position("日本語", "本", 1, false).unwrap(), Some(1));
        assert_eq!(find_scalar_position("日本語", "本", 99, false).unwrap(), None);
        assert_eq!(find_scalar_position("日本語", "", 1, false).unwrap(), Some(1));
        assert_eq!(find_scalar_position("日本ABC語", "abc", 0, true).unwrap(), Some(2));
        assert_eq!(find_scalar_position("İXİy", "İy", 0, true).unwrap(), Some(2));
    }

    #[test]
    fn find_and_index_return_one_based_scalar_positions() {
        assert_number("find", &[text_arg("日本語のテキスト"), text_arg("テキスト")], 5.0);
        assert_number("index", &[text_arg("日本語のテキスト"), text_arg("の")], 4.0);
    }

    #[test]
    fn find_start_position_is_a_scalar_ordinal_and_stays_bounded() {
        assert_number("find", &[text_arg("日本語"), text_arg("本"), number_arg(1.0)], 2.0);
        // start is inclusive: scalar position 2 is '本' itself, so the match is at 2.
        assert_number("find", &[text_arg("日本語"), text_arg("本"), number_arg(2.0)], 2.0);
        // No match at or after the start keeps the existing non-match 0 contract.
        assert_number("find", &[text_arg("日本語"), text_arg("日"), number_arg(2.0)], 0.0);
        assert_number("find", &[text_arg("日本語"), text_arg("本"), number_arg(99.0)], 0.0);
    }

    #[test]
    fn find_case_insensitive_positions_map_to_original_scalars() {
        assert_number(
            "find",
            &[text_arg("日本ABC語"), text_arg("abc"), number_arg(1.0), text_arg("i")],
            3.0,
        );
        // 'İ' (U+0130) lowers to two scalars ('i' + U+0307) in both the source and
        // the excerpt; the reported position must be the matched İ's scalar ordinal
        // in the original string, never a lowercase-buffer index.
        assert_number(
            "find",
            &[text_arg("İXİy"), text_arg("İy"), number_arg(1.0), text_arg("i")],
            3.0,
        );
        assert_number(
            "find",
            &[text_arg("İXy"), text_arg("xy"), number_arg(1.0), text_arg("i")],
            2.0,
        );
    }

    #[test]
    fn prxmatch_reports_scalar_positions() {
        assert_number("prxmatch", &[text_arg("/テキスト/"), text_arg("日本語のテキスト")], 5.0);
        assert_number("prxmatch", &[text_arg("/ZZZ/"), text_arg("日本語")], 0.0);
    }

    #[test]
    fn string_search_compares_stored_scalars_without_normalization() {
        assert_number("find", &[text_arg("cafe\u{0301}"), text_arg("café")], 0.0);
        assert_number("index", &[text_arg("café"), text_arg("e\u{0301}")], 0.0);
    }

    #[test]
    fn empty_excerpt_keeps_existing_position_contracts() {
        assert_number("index", &[text_arg("日本語"), text_arg("")], 1.0);
        assert_number("find", &[text_arg("日本語"), text_arg(""), number_arg(2.0)], 2.0);
    }

    #[test]
    fn search_function_arity_errors_do_not_panic() {
        assert_error_without_panic("find", &[text_arg("a")]);
        assert_error_without_panic("find", &[
            text_arg("a"),
            text_arg("b"),
            number_arg(1.0),
            text_arg("i"),
            number_arg(1.0),
        ]);
        assert_error_without_panic("index", &[text_arg("a")]);
        assert_error_without_panic("prxmatch", &[text_arg("a")]);
    }

    #[test]
    fn pow_propagates_null_before_numeric_coercion() {
        // The bridge rewrites `var ** lit` assignments to pow(var, lit), so the
        // builtin must keep the binary-arithmetic NULL guard of BinaryOp::Pow:
        // a NULL operand propagates as NULL, and non-NULL operands keep the
        // existing checked-coercion results.
        assert_number("pow", &[number_arg(2.0), number_arg(3.0)], 8.0);
        assert_number("pow", &[number_arg(7.0), number_arg(2.0)], 49.0);

        for args in [
            vec![ScalarValue::Null, number_arg(2.0)],
            vec![number_arg(2.0), ScalarValue::Null],
            vec![ScalarValue::Null, ScalarValue::Null],
        ] {
            match evaluate("pow", &args) {
                Ok(ScalarValue::Null) => {}
                other => panic!("pow({args:?}) returned {other:?}, expected NULL"),
            }
        }
    }

    #[test]
    fn mod_function_propagates_null_before_numeric_coercion() {
        assert_number("mod", &[number_arg(5.0), number_arg(3.0)], 2.0);

        for args in [
            vec![ScalarValue::Null, number_arg(3.0)],
            vec![number_arg(3.0), ScalarValue::Null],
            vec![ScalarValue::Null, ScalarValue::Null],
        ] {
            match evaluate("mod", &args) {
                Ok(ScalarValue::Null) => {}
                other => panic!("mod({args:?}) returned {other:?}, expected NULL"),
            }
        }
    }
}
