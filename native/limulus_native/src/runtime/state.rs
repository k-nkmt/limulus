use chrono::{Datelike, Timelike};
use crate::expressions::ScalarValue;
use crate::runtime::EvalRuntimeState;
use crate::value_ref::OwnedValue;
use pyo3::prelude::*;
use pyo3::types::PyDict;

// State setter utilities - internal helpers for managing runtime state
pub(crate) fn set_automatic_scalar(state: &mut EvalRuntimeState, name: &str, value: ScalarValue) {
    state
        .automatic_values
        .insert(crate::expressions::column_key(name), OwnedValue::from(value));
}

pub(crate) fn set_mutable_scalar(state: &mut EvalRuntimeState, name: &str, value: ScalarValue) {
    state
        .mutable_values
        .insert(crate::expressions::column_key(name), OwnedValue::from(value));
}

pub(crate) fn reset_automatic_scalars(state: &mut EvalRuntimeState, global_row_index: i64) {
    state.automatic_values.clear();
    state.mutable_values.clear();
    set_automatic_scalar(state, "_N_", ScalarValue::Number(global_row_index as f64));
    set_automatic_scalar(state, "_ERROR_", ScalarValue::Number(0.0));
}

pub(crate) fn mark_automatic_error(state: &mut EvalRuntimeState) {
    set_automatic_scalar(state, "_ERROR_", ScalarValue::Number(1.0));
}

pub(crate) fn set_runtime_flag(state: &mut EvalRuntimeState, name: &str, enabled: bool) {
    set_automatic_scalar(state, name, ScalarValue::Bool(enabled));
}

pub(crate) fn set_runtime_text(state: &mut EvalRuntimeState, name: &str, value: String) {
    set_automatic_scalar(state, name, ScalarValue::Text(value));
}

pub(crate) fn store_runtime_scalar(
    row: &Bound<'_, PyDict>,
    state: &mut EvalRuntimeState,
    name: &str,
    value: &ScalarValue,
) -> Result<(), String> {
    set_mutable_scalar(state, name, value.clone());
    if state.materialize_mutable_values {
        set_row_scalar(row, name, value)?;
    }
    Ok(())
}

fn set_row_scalar(row: &Bound<'_, PyDict>, name: &str, value: &ScalarValue) -> Result<(), String> {
    use crate::expressions::{column_key, scalar_to_pyobject};

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
        ScalarValue::Decimal(_) => row.set_item(&target_name, scalar_to_pyobject(row.py(), value)?),
        ScalarValue::Text(text) => row.set_item(&target_name, text.clone()),
        ScalarValue::Bool(boolean) => row.set_item(&target_name, *boolean),
        ScalarValue::Date(date) => row.set_item(&target_name, temporal_scalar_to_pyany(row.py(), &ScalarValue::Date(*date))?),
        ScalarValue::DateTime(datetime) => {
            row.set_item(&target_name, temporal_scalar_to_pyany(row.py(), &ScalarValue::DateTime(*datetime))?)
        }
        ScalarValue::Time(time) => row.set_item(&target_name, temporal_scalar_to_pyany(row.py(), &ScalarValue::Time(*time))?),
        ScalarValue::List(_) | ScalarValue::Struct(_) => {
            row.set_item(&target_name, scalar_to_pyobject(row.py(), value)?)
        }
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
