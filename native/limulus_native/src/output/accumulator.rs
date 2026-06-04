#![allow(dead_code)]

use chrono::{Datelike, Timelike};
use crate::expressions::ScalarValue;
use crate::output::OutputBuilderAccumulator;
use crate::value_ref::OwnedValue;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};
use std::collections::HashMap;

// Public helper: invert rename map by target name
pub(crate) fn invert_rename_map_by_target(
    rename_map: &HashMap<String, String>,
    target_name: &str,
) -> Option<String> {
    for (source_name, mapped_name) in rename_map {
        if mapped_name.eq_ignore_ascii_case(target_name) {
            return Some(source_name.clone());
        }
    }
    None
}

// Arrow type resolution helpers
fn resolve_arrow_type(
    pyarrow: &Bound<'_, PyModule>,
    type_name: Option<&str>,
) -> Result<Option<Py<PyAny>>, String> {
    let Some(type_name) = type_name else {
        return Ok(None);
    };
    if type_name.is_empty() || type_name == "dynamic" {
        return Ok(None);
    }
    if type_name.contains('<') {
        return Ok(None);
    }

    let alias_constructor = match type_name {
        "int" => Some("int64"),
        "float" => Some("float64"),
        "str" | "string" => Some("string"),
        "bool" => Some("bool_"),
        _ => None,
    };

    if let Some(constructor_name) = alias_constructor {
        return pyarrow
            .getattr(constructor_name)
            .map_err(|error| {
                format!("failed to resolve Arrow type constructor '{constructor_name}': {error}")
            })?
            .call0()
            .map(|value| Some(value.unbind()))
            .map_err(|error| format!("failed to construct Arrow type '{type_name}': {error}"));
    }

    if let Some((precision, scale)) = parse_decimal_type_name(type_name) {
        return pyarrow
            .getattr("decimal128")
            .map_err(|error| format!("failed to resolve pyarrow.decimal128: {error}"))?
            .call1((precision, scale))
            .map(|value| Some(value.unbind()))
            .map_err(|error| format!("failed to construct Arrow type '{type_name}': {error}"));
    }

    pyarrow
        .getattr("type_for_alias")
        .map_err(|error| format!("failed to resolve pyarrow.type_for_alias: {error}"))?
        .call1((type_name,))
        .map(|value| Some(value.unbind()))
        .map_err(|error| format!("failed to resolve Arrow type alias '{type_name}': {error}"))
}

fn parse_decimal_type_name(type_name: &str) -> Option<(u8, i8)> {
    let normalized = type_name.trim().to_ascii_lowercase();
    let inner = normalized
        .strip_prefix("decimal128(")?
        .strip_suffix(')')?;
    let (precision, scale) = inner.split_once(',')?;
    Some((precision.trim().parse().ok()?, scale.trim().parse().ok()?))
}

// Build Arrow arrays from accumulated output
pub(crate) fn build_output_arrays(
    py: Python<'_>,
    pyarrow: &Bound<'_, PyModule>,
    output: &OutputBuilderAccumulator,
) -> Result<(Py<PyAny>, Py<PyAny>), String> {
    let array_fn = pyarrow
        .getattr("array")
        .map_err(|error| format!("pyarrow.array is missing: {error}"))?;
    let arrays = PyList::empty(py);
    let names = PyList::empty(py);

    for name in &output.column_order {
        let values = output
            .columns
            .get(name)
            .ok_or_else(|| format!("missing collected output values for '{name}'"))?;
        let value_list = PyList::empty(py);
        for value in values {
            value_list.append(scalar_value_to_pyobject(py, value)?.bind(py)).map_err(|error| {
                format!("failed to append output value for '{name}': {error}")
            })?;
        }

        let array = if let Some(data_type) = resolve_arrow_type(
            pyarrow,
            output.type_expectations.get(name).map(String::as_str),
        )? {
            let kwargs = PyDict::new(py);
            kwargs
                .set_item("type", data_type.bind(py))
                .map_err(|error| format!("failed to set Arrow type keyword for '{name}': {error}"))?;
            array_fn
                .call((value_list,), Some(&kwargs))
                .map_err(|error| format!("failed to build typed Arrow array for '{name}': {error}"))?
        } else {
            array_fn
                .call1((value_list,))
                .map_err(|error| format!("failed to build Arrow array for '{name}': {error}"))?
        };
        arrays
            .append(array)
            .map_err(|error| format!("failed to append Arrow array for '{name}': {error}"))?;
        names
            .append(name)
            .map_err(|error| format!("failed to append Arrow name for '{name}': {error}"))?;
    }

    Ok((arrays.unbind().into(), names.unbind().into()))
}

// Normalize output numbers
fn normalize_output_number(value: f64) -> f64 {
    if !value.is_finite() {
        return value;
    }

    let tolerance = 1e-12 * value.abs().max(1.0);
    for places in 0..=13 {
        let rounded = round_to_places(value, places);
        if (value - rounded).abs() <= tolerance {
            return if rounded == -0.0 { 0.0 } else { rounded };
        }
    }

    if value == -0.0 { 0.0 } else { value }
}

fn round_to_places(value: f64, places: u32) -> f64 {
    let factor = 10f64.powi(places as i32);
    if factor == 0.0 || !factor.is_finite() {
        return value;
    }
    (value * factor).round() / factor
}

// Convert scalar values to Python objects for output
fn scalar_value_to_pyobject(py: Python<'_>, value: &OwnedValue) -> Result<Py<PyAny>, String> {
    scalar_to_pyobject(py, value.as_scalar())
}

fn scalar_to_pyobject(py: Python<'_>, value: &ScalarValue) -> Result<Py<PyAny>, String> {
    let datetime_module = py.import("datetime").map_err(|error| error.to_string())?;
    match value {
        ScalarValue::Number(number) => Ok(normalize_output_number(*number)
            .into_pyobject(py)
            .map_err(|error| error.to_string())?
            .unbind()
            .into_any()),
        ScalarValue::Decimal(decimal) => py
            .import("decimal")
            .map_err(|error| error.to_string())?
            .getattr("Decimal")
            .map_err(|error| error.to_string())?
            .call1((decimal,))
            .map_err(|error| error.to_string())
            .map(|value| value.unbind()),
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
