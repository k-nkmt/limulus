use crate::expressions::{get_row_item, py_to_scalar, resolve_row_key, resolve_value, ScalarValue};
use crate::runtime::EvalRuntimeState;
use crate::value_ref::{AppendValue, OwnedValue, ResolvedValue};
use chrono::{Datelike, Timelike};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyModule};
use std::collections::HashMap;

mod accumulator;

#[derive(Default)]
pub(crate) struct OutputHandoffTargetSpec {
    projected_columns: Vec<String>,
    rename_map: HashMap<String, String>,
    type_expectations: HashMap<String, String>,
}

#[derive(Default)]
pub(crate) struct OutputBuilderAccumulator {
    column_order: Vec<String>,
    columns: HashMap<String, Vec<OwnedValue>>,
    row_count: usize,
    planned_columns: bool,
    handoff_rename_map: HashMap<String, String>,
    handoff_rename_inverse: HashMap<String, String>,
    type_expectations: HashMap<String, String>,
}

impl OutputBuilderAccumulator {
    pub(crate) fn uses_planned_columns(&self) -> bool {
        self.planned_columns
    }

    fn new(target_spec: Option<OutputHandoffTargetSpec>) -> Self {
        let Some(spec) = target_spec else {
            return Self::default();
        };

        let mut columns: HashMap<String, Vec<OwnedValue>> = HashMap::new();
        for name in &spec.projected_columns {
            columns.insert(name.clone(), Vec::new());
        }
        let handoff_rename_inverse = spec
            .rename_map
            .iter()
            .map(|(source, target)| (target.clone(), source.clone()))
            .collect();

        Self {
            planned_columns: !spec.projected_columns.is_empty(),
            column_order: spec.projected_columns,
            columns,
            row_count: 0,
            handoff_rename_map: spec.rename_map,
            handoff_rename_inverse,
            type_expectations: spec.type_expectations,
        }
    }

    fn append(
        &mut self,
        row: &Bound<'_, PyDict>,
        state: &EvalRuntimeState,
        keep_vars: &[String],
        drop_vars: &[String],
        rename_map: &HashMap<String, String>,
    ) -> Result<(), String> {
        if self.planned_columns {
            for column_name in &self.column_order {
                let value = self.resolve_projected_value(
                    row,
                    state,
                    column_name,
                    keep_vars,
                    drop_vars,
                    rename_map,
                )?;
                self.columns
                    .get_mut(column_name)
                    .ok_or_else(|| format!("missing output column buffer for '{column_name}'"))?
                    .push(value);
            }
            self.row_count += 1;
            return Ok(());
        }

        let resolved_runtime_rename_map = self.resolve_runtime_rename_map(row, rename_map)?;
        let resolved_handoff_rename_map = self.resolve_handoff_rename_map(row, &resolved_runtime_rename_map)?;

        for (key_any, _) in row.iter() {
            let key = key_any
                .extract::<String>()
                .map_err(|error| format!("failed to read output row key: {error}"))?;
            if !self.passes_keep_drop_filters(&key, keep_vars, drop_vars) {
                continue;
            }

            let runtime_name = resolved_runtime_rename_map
                .get(&key)
                .cloned()
                .unwrap_or_else(|| key.clone());
            let final_name = resolved_handoff_rename_map
                .get(&runtime_name)
                .cloned()
                .unwrap_or(runtime_name);

            if self.columns.contains_key(&final_name) {
                continue;
            }
            self.column_order.push(final_name.clone());
            let mut values: Vec<OwnedValue> = Vec::with_capacity(self.row_count);
            for _ in 0..self.row_count {
                values.push(OwnedValue::null());
            }
            self.columns.insert(final_name, values);
        }

        for column_name in &self.column_order {
            let value = self.resolve_projected_value(
                row,
                state,
                column_name,
                keep_vars,
                drop_vars,
                rename_map,
            )?;
            self.columns
                .get_mut(column_name)
                .ok_or_else(|| format!("missing output column buffer for '{column_name}'"))?
                .push(value);
        }
        self.row_count += 1;
        Ok(())
    }

    fn resolve_projected_value(
        &self,
        row: &Bound<'_, PyDict>,
        state: &EvalRuntimeState,
        projected_name: &str,
        keep_vars: &[String],
        drop_vars: &[String],
        rename_map: &HashMap<String, String>,
    ) -> Result<OwnedValue, String> {
        let runtime_name = self
            .handoff_rename_inverse
            .get(projected_name)
            .cloned()
            .unwrap_or_else(|| projected_name.to_string());
        let source_name = invert_rename_map_by_target(rename_map, &runtime_name)
            .unwrap_or_else(|| runtime_name.clone());

        if !self.passes_keep_drop_filters(&source_name, keep_vars, drop_vars) {
            return Ok(OwnedValue::null());
        }

        if let Some(value) = self.resolve_append_value(row, state, &source_name)? {
            return Ok(value.into_owned_value());
        }
        if let Some(value) = self.resolve_materialized_row_value(row, state, &source_name)? {
            return Ok(value);
        }
        if !runtime_name.eq_ignore_ascii_case(&source_name) {
            if let Some(value) = self.resolve_materialized_row_value(row, state, &runtime_name)? {
                return Ok(value);
            }
        }
        if let Some(value) = self.resolve_append_value(row, state, projected_name)? {
            return Ok(value.into_owned_value());
        }
        if !projected_name.eq_ignore_ascii_case(&runtime_name) {
            if let Some(value) = self.resolve_materialized_row_value(row, state, projected_name)? {
                return Ok(value);
            }
        }
        Ok(OwnedValue::null())
    }

    fn resolve_append_value<'a>(
        &self,
        row: &Bound<'_, PyDict>,
        state: &'a EvalRuntimeState,
        name: &str,
    ) -> Result<Option<AppendValue<'a>>, String> {
        resolve_value(row, name, state).map(|value| {
            value.map(|resolved| match resolved {
                ResolvedValue::Source(value_ref) => AppendValue::Source(value_ref),
                ResolvedValue::Owned(value) => AppendValue::Owned(value),
            })
        })
    }

    fn resolve_materialized_row_value(
        &self,
        row: &Bound<'_, PyDict>,
        state: &EvalRuntimeState,
        name: &str,
    ) -> Result<Option<OwnedValue>, String> {
        if !state.materialize_mutable_values {
            return Ok(None);
        }
        let Some(value_any) = get_row_item(row, name)? else {
            return Ok(None);
        };
        py_to_scalar(&value_any)
            .map(OwnedValue::from)
            .map(Some)
            .map_err(|error| error.to_string())
    }

    fn passes_keep_drop_filters(
        &self,
        source_name: &str,
        keep_vars: &[String],
        drop_vars: &[String],
    ) -> bool {
        if !keep_vars.is_empty() && !keep_vars.iter().any(|name| name.eq_ignore_ascii_case(source_name)) {
            return false;
        }
        if drop_vars.iter().any(|name| name.eq_ignore_ascii_case(source_name)) {
            return false;
        }
        true
    }

    fn resolve_runtime_rename_map(
        &self,
        row: &Bound<'_, PyDict>,
        rename_map: &HashMap<String, String>,
    ) -> Result<HashMap<String, String>, String> {
        let mut resolved = HashMap::new();
        for (source_name, target_name) in rename_map {
            let Some(resolved_source_name) = resolve_row_key(row, source_name)? else {
                continue;
            };
            resolved.insert(resolved_source_name, target_name.clone());
        }
        Ok(resolved)
    }

    fn resolve_handoff_rename_map(
        &self,
        row: &Bound<'_, PyDict>,
        resolved_runtime_rename_map: &HashMap<String, String>,
    ) -> Result<HashMap<String, String>, String> {
        let mut resolved = HashMap::new();
        for (source_name, target_name) in &self.handoff_rename_map {
            let runtime_source = invert_rename_map_by_target(resolved_runtime_rename_map, source_name)
                .unwrap_or_else(|| source_name.clone());
            let Some(resolved_source_name) = resolve_row_key(row, &runtime_source)? else {
                continue;
            };
            let resolved_runtime_name = resolved_runtime_rename_map
                .get(&resolved_source_name)
                .cloned()
                .unwrap_or(resolved_source_name);
            resolved.insert(resolved_runtime_name, target_name.clone());
        }
        Ok(resolved)
    }
}

fn invert_rename_map_by_target(
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

fn get_attr_or_item<'py>(
    obj: &Bound<'py, PyAny>,
    name: &str,
) -> Result<Option<Bound<'py, PyAny>>, String> {
    if let Ok(dict) = obj.cast::<PyDict>() {
        return dict
            .get_item(name)
            .map_err(|error| format!("failed to read '{name}' from mapping: {error}"));
    }

    match obj.getattr(name) {
        Ok(value) => Ok(Some(value)),
        Err(error) => {
            if error.is_instance_of::<pyo3::exceptions::PyAttributeError>(obj.py()) {
                Ok(None)
            } else {
                Err(format!("failed to read attribute '{name}': {error}"))
            }
        }
    }
}

fn get_nested_mapping_item<'py>(
    obj: &Bound<'py, PyAny>,
    mapping_name: &str,
    key: &str,
) -> Result<Option<Bound<'py, PyAny>>, String> {
    let Some(mapping_any) = get_attr_or_item(obj, mapping_name)? else {
        return Ok(None);
    };

    if let Ok(dict) = mapping_any.cast::<PyDict>() {
        return dict
            .get_item(key)
            .map_err(|error| format!("failed to read '{mapping_name}[{key}]': {error}"));
    }

    match mapping_any.call_method1("get", (key,)) {
        Ok(value) => {
            if value.is_none() {
                Ok(None)
            } else {
                Ok(Some(value))
            }
        }
        Err(error) => Err(format!("failed to read '{mapping_name}[{key}]': {error}")),
    }
}

pub(crate) fn initialize_output_builders(
    execution_plan: Option<&Bound<'_, PyAny>>,
    targets: &[String],
) -> Result<HashMap<String, OutputBuilderAccumulator>, String> {
    let mut outputs: HashMap<String, OutputBuilderAccumulator> = HashMap::new();
    for target in targets {
        let target_spec = resolve_output_handoff_target_spec(execution_plan, target)?;
        outputs.insert(target.clone(), OutputBuilderAccumulator::new(target_spec));
    }
    Ok(outputs)
}

pub(crate) fn append_projected_row_to_target(
    outputs: &mut HashMap<String, OutputBuilderAccumulator>,
    target: &str,
    row: &Bound<'_, PyDict>,
    state: &EvalRuntimeState,
    keep_vars: &[String],
    drop_vars: &[String],
    rename_map: &HashMap<String, String>,
) -> Result<(), String> {
    let builder = outputs
        .get_mut(target)
        .ok_or_else(|| format!("output target is not declared: {target}"))?;
    builder.append(row, state, keep_vars, drop_vars, rename_map)
}

pub(crate) fn all_output_builders_use_planned_columns(
    outputs: &HashMap<String, OutputBuilderAccumulator>,
) -> bool {
    !outputs.is_empty() && outputs.values().all(OutputBuilderAccumulator::uses_planned_columns)
}

pub(crate) fn resolve_declared_target_name(requested: &str, declared: &[String]) -> String {
    let requested_lower = requested.to_lowercase();
    for target in declared {
        if target.to_lowercase() == requested_lower {
            return target.clone();
        }
    }
    requested.to_string()
}

fn resolve_output_handoff_target_spec(
    execution_plan: Option<&Bound<'_, PyAny>>,
    target_name: &str,
) -> Result<Option<OutputHandoffTargetSpec>, String> {
    let Some(plan_any) = execution_plan else {
        return Ok(None);
    };
    let Some(raw_plan_any) = get_attr_or_item(plan_any, "output_handoff_plan")? else {
        return Ok(None);
    };

    let projected_columns =
        match get_nested_mapping_item(&raw_plan_any, "projected_columns_by_target", target_name)? {
            Some(value) => value.extract::<Vec<String>>().map_err(|error| {
                format!("failed to deserialize projected columns for '{target_name}': {error}")
            })?,
            None => Vec::new(),
        };
    let rename_map = match get_nested_mapping_item(&raw_plan_any, "rename_map_by_target", target_name)? {
        Some(value) => value.extract::<HashMap<String, String>>().map_err(|error| {
            format!("failed to deserialize rename map for '{target_name}': {error}")
        })?,
        None => HashMap::new(),
    };
    let type_expectations =
        match get_nested_mapping_item(&raw_plan_any, "type_expectations_by_target", target_name)? {
            Some(value) => value.extract::<HashMap<String, String>>().map_err(|error| {
                format!("failed to deserialize type expectations for '{target_name}': {error}")
            })?,
            None => HashMap::new(),
        };

    if projected_columns.is_empty() && rename_map.is_empty() && type_expectations.is_empty() {
        return Ok(None);
    }

    Ok(Some(OutputHandoffTargetSpec {
        projected_columns,
        rename_map,
        type_expectations,
    }))
}

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

fn build_output_arrays(
    py: Python<'_>,
    pyarrow: &Bound<'_, PyModule>,
    output: &OutputBuilderAccumulator,
) -> Result<(Py<PyList>, Py<PyList>), String> {
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

    Ok((arrays.unbind(), names.unbind()))
}

pub(crate) fn export_output_streams(
    py: Python<'_>,
    outputs: &HashMap<String, OutputBuilderAccumulator>,
) -> Result<Py<PyDict>, String> {
    let pyarrow = py
        .import("pyarrow")
        .map_err(|error| format!("failed to import pyarrow: {error}"))?;
    let table_class = pyarrow
        .getattr("Table")
        .map_err(|error| format!("pyarrow.Table is missing: {error}"))?;

    let output_streams = PyDict::new(py);
    for (target_name, output) in outputs {
        let (arrays, names) = build_output_arrays(py, &pyarrow, output)?;
        let kwargs = PyDict::new(py);
        kwargs
            .set_item("names", names.bind(py))
            .map_err(|error| format!("failed to set Arrow table names for '{target_name}': {error}"))?;
        let table = table_class
            .call_method("from_arrays", (arrays.bind(py),), Some(&kwargs))
            .map_err(|error| format!("failed to convert output rows to Arrow table: {error}"))?;
        let stream = table
            .call_method0("__arrow_c_stream__")
            .map_err(|error| format!("failed to export output Arrow C stream: {error}"))?;
        output_streams
            .set_item(target_name, stream)
            .map_err(|error| format!("failed to set output stream: {error}"))?;
    }

    Ok(output_streams.unbind())
}

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
