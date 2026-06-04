//! Native limulus runtime entry points and row-level dataset-option handling.

mod ast;
mod diagnostics;
mod expressions;
mod output;
mod row_cursor;
mod runtime;
mod source_view;
mod value_ref;

use crate::ast::AstDatasetRefOptions;
use crate::expressions::{
    condition_syntax_error_message,
    column_key,
    evaluate_simple_where,
    get_row_item,
    is_expression_parse_error,
    resolve_row_key,
};
use crate::runtime::EvalRuntimeState;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

fn delete_row_item(row: &Bound<'_, PyDict>, name: &str) -> Result<(), String> {
    if let Some(resolved_name) = resolve_row_key(row, name)? {
        row.del_item(&resolved_name)
            .map_err(|error| format!("row del_item failed: {error}"))?;
    }
    Ok(())
}

fn apply_dataset_ref_options_to_row(
    py: Python<'_>,
    source_row: &Bound<'_, PyDict>,
    options: &AstDatasetRefOptions,
    state: &mut EvalRuntimeState,
    scope_key: &str,
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
            if let Some(resolved_name) = resolve_row_key(&working, name)? {
                if let Some(value) = get_row_item(&working, &resolved_name)? {
                    projected
                        .set_item(resolved_name, value)
                        .map_err(|error| format!("dataset option keep set failed: {error}"))?;
                }
            } else if let Some(value) = get_row_item(&working, name)? {
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

    if let Some(expression) = &options.where_expr {
        let pass = match evaluate_simple_where(&working, expression, state) {
            Ok(value) => value,
            Err(error) => {
                if is_expression_parse_error(&error) {
                    if let Some(syntax_error) = condition_syntax_error_message(py, expression, "<string>")? {
                        return Err(format!("dataset option WHERE= evaluation failed: {syntax_error}"));
                    }
                }
                return Err(format!("dataset option WHERE= evaluation failed: {error}"));
            }
        };
        if !pass {
            return Ok(None);
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

    if let Some(firstobs) = options.firstobs {
        if firstobs <= 0 {
            return Err("dataset option FIRSTOBS= must be positive".to_string());
        }
    }
    if let Some(obs) = options.obs {
        if obs < 0 {
            return Err("dataset option OBS= must be non-negative".to_string());
        }
    }

    if options.firstobs.is_some() || options.obs.is_some() {
        let firstobs = options.firstobs.unwrap_or(1);
        let counter = state
            .dataset_option_survivor_counts
            .entry(scope_key.to_string())
            .or_insert(0);
        *counter += 1;
        let position = *counter as i64;

        if position < firstobs {
            return Ok(None);
        }
        if let Some(obs) = options.obs {
            if position >= firstobs + obs {
                return Ok(None);
            }
        }
    }

    Ok(Some(working.unbind()))
}

#[pyfunction]
fn execute_block(py: Python<'_>, payload: &Bound<'_, PyDict>) -> PyResult<Py<PyDict>> {
    let _ = ast::parse_runtime_statements;
    runtime::execute_block(py, payload)
}

#[pyfunction]
fn render_diagnostics_ariadne(payload: &Bound<'_, PyDict>) -> PyResult<String> {
    diagnostics::render_diagnostics_ariadne(payload)
}

#[pymodule]
fn limulus_native(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(execute_block, module)?)?;
    module.add_function(wrap_pyfunction!(render_diagnostics_ariadne, module)?)?;
    Ok(())
}
