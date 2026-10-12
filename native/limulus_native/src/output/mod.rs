use crate::expressions::ScalarValue;
use crate::row_cursor::NativeArrowRowCursor;
use crate::row_view::LogicalRowAccess;
use crate::runtime::EvalRuntimeState;
use crate::source_registry::{ExecutionSourceRegistry, SourceId};
use crate::value_ref::OwnedValue;
use arrow::array::{new_null_array, Array, ArrayRef, UInt64Array};
use arrow::compute::{concat, take, TakeOptions};
use arrow::datatypes::{Field, TimeUnit};
use arrow::ffi::{to_ffi, FFI_ArrowSchema};
use chrono::{Datelike, Timelike};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyCapsule, PyDict, PyList, PyModule};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::rc::Rc;

#[derive(Default)]
pub(crate) struct OutputHandoffTargetSpec {
    projected_columns: Vec<String>,
    rename_map: HashMap<String, String>,
    type_expectations: HashMap<String, String>,
}

#[derive(Default)]
pub(crate) struct OutputBuilderAccumulator {
    pub(crate) source_registry: Rc<ExecutionSourceRegistry<NativeArrowRowCursor>>,
    column_order: Vec<String>,
    columns: HashMap<String, Vec<OwnedValue>>,
    row_count: usize,
    planned_columns: bool,
    handoff_rename_inverse: HashMap<String, String>,
    type_expectations: HashMap<String, String>,
    field_hints: HashMap<String, Field>,
}

impl OutputBuilderAccumulator {
    pub(crate) fn uses_planned_columns(&self) -> bool {
        self.planned_columns
    }

    fn new(
        target_spec: Option<OutputHandoffTargetSpec>,
        source_registry: Rc<ExecutionSourceRegistry<NativeArrowRowCursor>>,
    ) -> Self {
        let Some(spec) = target_spec else {
            return Self {
                source_registry,
                ..Self::default()
            };
        };

        let mut columns: HashMap<String, Vec<OwnedValue>> = HashMap::new();
        for name in &spec.projected_columns {
            columns.insert(name.clone(), Vec::new());
        }
        let handoff_rename_inverse = spec
            .rename_map
            .iter()
            .map(|(source, target)| (target.to_ascii_lowercase(), source.clone()))
            .collect();

        Self {
            source_registry,
            planned_columns: !spec.projected_columns.is_empty(),
            column_order: spec.projected_columns,
            columns,
            row_count: 0,
            handoff_rename_inverse,
            type_expectations: spec.type_expectations,
            field_hints: HashMap::new(),
        }
    }

    fn ensure_projected_shape(
        &mut self,
        view: &crate::row_view::LogicalRowView,
        excluded_names: &HashSet<String>,
    ) -> Result<(), String> {
        for column in view.visible_columns().map_err(|error| error.to_string())? {
            let name = column.logical_name().to_string();
            if excluded_names
                .iter()
                .any(|excluded| excluded.eq_ignore_ascii_case(&name))
            {
                continue;
            }
            if !self.planned_columns
                && !self
                    .columns
                    .keys()
                    .any(|existing| existing.eq_ignore_ascii_case(&name))
            {
                let mut values = Vec::with_capacity(self.row_count);
                values.resize(self.row_count, OwnedValue::null());
                self.column_order.push(name.clone());
                self.columns.insert(name.clone(), values);
            }
            if let Some(source_field) = column.source_field() {
                let field = source_field.clone().with_name(&name);
                let existing = self
                    .field_hints
                    .iter()
                    .find(|(existing_name, _)| existing_name.eq_ignore_ascii_case(&name))
                    .map(|(_, field)| field);
                if let Some(existing) = existing {
                    if !source_fields_are_compatible(existing, &field) {
                        return Err(format!(
                            "output column '{name}' has incompatible source field ownership"
                        ));
                    }
                } else {
                    self.field_hints.insert(name, field);
                }
            }
        }
        Ok(())
    }

    fn source_field_for_output_column(&self, output_name: &str) -> Result<Option<Field>, String> {
        if let Some((_, field)) = self
            .field_hints
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(output_name))
        {
            return Ok(Some(field.clone().with_name(output_name)));
        }
        let source_name = self
            .handoff_rename_inverse
            .get(&output_name.to_ascii_lowercase())
            .map(String::as_str)
            .unwrap_or(output_name);
        let mut resolved: Option<Field> = None;
        for source_index in 0..self.source_registry.len() {
            let cursor = self
                .source_registry
                .lookup(crate::source_registry::SourceId::new(source_index))
                .map_err(|error| error.to_string())?
                .cursor();
            let Some(slot_index) = cursor
                .slot_order()
                .iter()
                .position(|name| name.eq_ignore_ascii_case(source_name))
            else {
                continue;
            };
            let field = cursor.field_at(slot_index).ok_or_else(|| {
                format!("source field is missing for output column '{output_name}'")
            })?;
            if !is_source_backed_type(field.data_type()) {
                continue;
            }
            if let Some(expected) = &resolved {
                if !source_fields_are_compatible(expected, &field) {
                    return Err(format!(
                        "output column '{output_name}' has incompatible source field ownership"
                    ));
                }
            } else {
                resolved = Some(field.as_ref().clone());
            }
        }
        Ok(resolved.map(|field| field.with_name(output_name)))
    }

    fn append(
        &mut self,
        state: &EvalRuntimeState,
        global_keep: &[String],
        global_drop: &[String],
        global_rename: &HashMap<String, String>,
        target_keep: &[String],
        target_drop: &[String],
        target_rename: &HashMap<String, String>,
        excluded_names: &HashSet<String>,
    ) -> Result<(), String> {
        let row_view = state
            .logical_row
            .as_ref()
            .ok_or_else(|| "logical row view is required for output projection".to_string())?;
        let target_view = project_output_view(
            row_view, global_keep, global_drop, global_rename,
            target_keep, target_drop, target_rename,
        )?;
        self.ensure_projected_shape(&target_view, excluded_names)?;

        if self.planned_columns {
            for column_name in &self.column_order {
                let value = resolve_logical_value(&target_view, column_name)?;
                self.columns
                    .get_mut(column_name)
                    .ok_or_else(|| format!("missing output column buffer for '{column_name}'"))?
                    .push(value);
            }
            self.row_count += 1;
            return Ok(());
        }

        for column_name in &self.column_order {
            let value = resolve_logical_value(&target_view, column_name)?;
            self.columns
                .get_mut(column_name)
                .ok_or_else(|| format!("missing output column buffer for '{column_name}'"))?
                .push(value);
        }
        self.row_count += 1;
        Ok(())
    }
}

pub(crate) fn ensure_output_target_shape_from_view(
    outputs: &mut HashMap<String, OutputBuilderAccumulator>,
    target: &str,
    view: &crate::row_view::LogicalRowView,
    global_keep: &[String],
    global_drop: &[String],
    global_rename: &HashMap<String, String>,
    target_keep: &[String],
    target_drop: &[String],
    target_rename: &HashMap<String, String>,
    excluded_names: &HashSet<String>,
) -> Result<(), String> {
    let target_view = project_output_view(
        view, global_keep, global_drop, global_rename,
        target_keep, target_drop, target_rename,
    )?;
    outputs
        .get_mut(target)
        .ok_or_else(|| format!("output target is not declared: {target}"))?
        .ensure_projected_shape(&target_view, excluded_names)
}

fn project_output_view<'a>(
    view: &'a crate::row_view::LogicalRowView,
    global_keep: &[String],
    global_drop: &[String],
    global_rename: &HashMap<String, String>,
    target_keep: &[String],
    target_drop: &[String],
    target_rename: &HashMap<String, String>,
) -> Result<Cow<'a, crate::row_view::LogicalRowView>, String> {
    if global_keep.is_empty() && global_drop.is_empty() && global_rename.is_empty()
        && target_keep.is_empty() && target_drop.is_empty() && target_rename.is_empty()
    {
        // Output consumers only read the view. Visibility and values do not
        // change when both projections are empty, so no snapshot is needed.
        return Ok(Cow::Borrowed(view));
    }
    let mut projected = view.clone();
    projected
        .project_columns(global_keep, global_drop, global_rename)
        .and_then(|_| projected.project_columns(target_keep, target_drop, target_rename))
        .map_err(|error| error.to_string())?;
    Ok(Cow::Owned(projected))
}

fn resolve_logical_value(
    view: &crate::row_view::LogicalRowView,
    logical_name: &str,
) -> Result<OwnedValue, String> {
    let Some(value) = view
        .resolve(logical_name)
        .map_err(|error| error.to_string())?
    else {
        return Ok(OwnedValue::null());
    };
    match value {
        crate::row_view::ResolvedValue::SourceScalar(value) => value.to_owned_value(),
        crate::row_view::ResolvedValue::SourceCell(cell) => Ok(OwnedValue::SourceCell(cell)),
        crate::row_view::ResolvedValue::Owned(value)
        | crate::row_view::ResolvedValue::Automatic(value) => Ok(value.clone()),
    }
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

    // Typed execution plans are exposed as immutable MappingProxyType values.
    // Read those through their mapping protocol just like plain dictionaries;
    // otherwise output shaping silently falls back to inference and can lose
    // exact Arrow types such as uint64.
    if let Ok(get_method) = obj.getattr("get") {
        let value = get_method
            .call1((name,))
            .map_err(|error| format!("failed to read '{name}' from mapping: {error}"))?;
        return if value.is_none() {
            Ok(None)
        } else {
            Ok(Some(value))
        };
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
    source_registry: Rc<ExecutionSourceRegistry<NativeArrowRowCursor>>,
) -> Result<HashMap<String, OutputBuilderAccumulator>, String> {
    let mut outputs: HashMap<String, OutputBuilderAccumulator> = HashMap::new();
    for target in targets {
        let target_spec = resolve_output_handoff_target_spec(execution_plan, target)?;
        outputs.insert(
            target.clone(),
            OutputBuilderAccumulator::new(target_spec, Rc::clone(&source_registry)),
        );
    }
    Ok(outputs)
}

pub(crate) fn append_projected_row_to_target(
    outputs: &mut HashMap<String, OutputBuilderAccumulator>,
    target: &str,
    state: &EvalRuntimeState,
    global_keep: &[String],
    global_drop: &[String],
    global_rename: &HashMap<String, String>,
    target_keep: &[String],
    target_drop: &[String],
    target_rename: &HashMap<String, String>,
    excluded_names: &HashSet<String>,
) -> Result<(), String> {
    let builder = outputs
        .get_mut(target)
        .ok_or_else(|| format!("output target is not declared: {target}"))?;
    builder.append(
        state,
        global_keep,
        global_drop,
        global_rename,
        target_keep,
        target_drop,
        target_rename,
        excluded_names,
    )
}

pub(crate) fn all_output_builders_use_planned_columns(
    outputs: &HashMap<String, OutputBuilderAccumulator>,
) -> bool {
    !outputs.is_empty()
        && outputs
            .values()
            .all(OutputBuilderAccumulator::uses_planned_columns)
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
    let rename_map =
        match get_nested_mapping_item(&raw_plan_any, "rename_map_by_target", target_name)? {
            Some(value) => value
                .extract::<HashMap<String, String>>()
                .map_err(|error| {
                    format!("failed to deserialize rename map for '{target_name}': {error}")
                })?,
            None => HashMap::new(),
        };
    let type_expectations =
        match get_nested_mapping_item(&raw_plan_any, "type_expectations_by_target", target_name)? {
            Some(value) => value
                .extract::<HashMap<String, String>>()
                .map_err(|error| {
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

    if let Some(temporal_type) = parse_temporal_type_name(type_name)? {
        return temporal_type
            .to_pyarrow(pyarrow)
            .map(|value| Some(value.unbind()));
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

#[derive(Debug, Clone, PartialEq, Eq)]
enum TemporalOutputType {
    Timestamp {
        unit: TimeUnit,
        timezone: Option<String>,
    },
    Time64 {
        unit: TimeUnit,
    },
}

impl TemporalOutputType {
    fn to_pyarrow<'py>(&self, pyarrow: &Bound<'py, PyModule>) -> Result<Bound<'py, PyAny>, String> {
        match self {
            Self::Timestamp { unit, timezone } => {
                let constructor = pyarrow
                    .getattr("timestamp")
                    .map_err(|error| format!("failed to resolve pyarrow.timestamp: {error}"))?;
                if let Some(timezone) = timezone {
                    let kwargs = PyDict::new(pyarrow.py());
                    kwargs
                        .set_item("tz", timezone)
                        .map_err(|error| format!("failed to set timestamp timezone: {error}"))?;
                    constructor
                        .call((arrow_time_unit_name(unit),), Some(&kwargs))
                        .map_err(|error| format!("failed to construct timestamp type: {error}"))
                } else {
                    constructor
                        .call1((arrow_time_unit_name(unit),))
                        .map_err(|error| format!("failed to construct timestamp type: {error}"))
                }
            }
            Self::Time64 { unit } => pyarrow
                .getattr("time64")
                .map_err(|error| format!("failed to resolve pyarrow.time64: {error}"))?
                .call1((arrow_time_unit_name(unit),))
                .map_err(|error| format!("failed to construct time64 type: {error}")),
        }
    }
}

fn parse_temporal_type_name(type_name: &str) -> Result<Option<TemporalOutputType>, String> {
    if let Some(inner) = type_name
        .strip_prefix("timestamp[")
        .and_then(|value| value.strip_suffix(']'))
    {
        let (unit_name, timezone) = match inner.split_once(", tz=") {
            Some((_unit, "")) => {
                return Err(format!(
                    "timestamp type has an empty timezone: '{type_name}'"
                ));
            }
            Some((unit, timezone)) => (unit, Some(timezone.to_string())),
            None => (inner, None),
        };
        return Ok(Some(TemporalOutputType::Timestamp {
            unit: parse_timestamp_unit(unit_name)
                .ok_or_else(|| format!("unsupported timestamp unit in '{type_name}'"))?,
            timezone,
        }));
    }
    if type_name.starts_with("timestamp[") {
        return Err(format!("invalid timestamp type expectation: '{type_name}'"));
    }
    if let Some(inner) = type_name
        .strip_prefix("time64[")
        .and_then(|value| value.strip_suffix(']'))
    {
        let unit = match inner {
            "us" => TimeUnit::Microsecond,
            "ns" => TimeUnit::Nanosecond,
            _ => return Err(format!("unsupported time64 unit in '{type_name}'")),
        };
        return Ok(Some(TemporalOutputType::Time64 { unit }));
    }
    if type_name.starts_with("time64[") {
        return Err(format!("invalid time64 type expectation: '{type_name}'"));
    }
    Ok(None)
}

fn parse_timestamp_unit(unit: &str) -> Option<TimeUnit> {
    match unit {
        "s" => Some(TimeUnit::Second),
        "ms" => Some(TimeUnit::Millisecond),
        "us" => Some(TimeUnit::Microsecond),
        "ns" => Some(TimeUnit::Nanosecond),
        _ => None,
    }
}

fn arrow_time_unit_name(unit: &TimeUnit) -> &'static str {
    match unit {
        TimeUnit::Second => "s",
        TimeUnit::Millisecond => "ms",
        TimeUnit::Microsecond => "us",
        TimeUnit::Nanosecond => "ns",
    }
}

fn parse_decimal_type_name(type_name: &str) -> Option<(u8, i8)> {
    let normalized = type_name.trim().to_ascii_lowercase();
    let inner = normalized.strip_prefix("decimal128(")?.strip_suffix(')')?;
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
    let fields = PyList::empty(py);

    for name in &output.column_order {
        let values = output
            .columns
            .get(name)
            .ok_or_else(|| format!("missing collected output values for '{name}'"))?;
        let has_source_cell = values
            .iter()
            .any(|value| matches!(value, OwnedValue::SourceCell(_)));
        let all_null = values
            .iter()
            .all(|value| matches!(value, OwnedValue::Scalar(ScalarValue::Null)));
        let planned_source_field = if !has_source_cell && all_null {
            output.source_field_for_output_column(name)?
        } else {
            None
        };
        let source_backed_empty_or_null = planned_source_field
            .as_ref()
            .is_some_and(|field| is_source_backed_type(field.data_type()))
            && all_null;
        if has_source_cell || source_backed_empty_or_null {
            let (array, field) = if has_source_cell {
                gather_source_backed_column(name, values)?
            } else {
                let field = planned_source_field
                    .as_ref()
                    .expect("source field checked above")
                    .clone();
                (
                    new_null_array(field.data_type(), values.len()),
                    field.with_nullable(true),
                )
            };
            let py_array = import_native_array(py, pyarrow, &array, name)?;
            let py_field = import_native_field(py, pyarrow, &field, name)?;
            arrays.append(py_array).map_err(|error| {
                format!("failed to append source-backed Arrow array for '{name}': {error}")
            })?;
            fields.append(py_field).map_err(|error| {
                format!("failed to append source-backed Arrow field for '{name}': {error}")
            })?;
            continue;
        }

        if all_null {
            if let Some(field) = planned_source_field {
                let field = field.with_nullable(true);
                let array = new_null_array(field.data_type(), values.len());
                let py_array = import_native_array(py, pyarrow, &array, name)?;
                let py_field = import_native_field(py, pyarrow, &field, name)?;
                arrays.append(py_array).map_err(|error| {
                    format!("failed to append typed null Arrow array for '{name}': {error}")
                })?;
                fields.append(py_field).map_err(|error| {
                    format!("failed to append typed null Arrow field for '{name}': {error}")
                })?;
                continue;
            }
        }

        let planned_type_name = output.type_expectations.get(name).map(String::as_str);
        let temporal_type = resolve_temporal_output_type(values, planned_type_name, name)?;
        let value_list = PyList::empty(py);
        for value in values {
            value_list
                .append(output_scalar_to_pyobject(py, value, temporal_type.as_ref())?.bind(py))
                .map_err(|error| format!("failed to append output value for '{name}': {error}"))?;
        }

        let resolved_type = if let Some(temporal_type) = &temporal_type {
            Some(temporal_type.to_pyarrow(pyarrow)?.unbind())
        } else {
            resolve_arrow_type(pyarrow, planned_type_name)?
        };
        let array = if let Some(data_type) = resolved_type {
            let kwargs = PyDict::new(py);
            kwargs
                .set_item("type", data_type.bind(py))
                .map_err(|error| {
                    format!("failed to set Arrow type keyword for '{name}': {error}")
                })?;
            array_fn
                .call((value_list,), Some(&kwargs))
                .map_err(|error| {
                    format!("failed to build typed Arrow array for '{name}': {error}")
                })?
        } else {
            array_fn
                .call1((value_list,))
                .map_err(|error| format!("failed to build Arrow array for '{name}': {error}"))?
        };
        arrays
            .append(&array)
            .map_err(|error| format!("failed to append Arrow array for '{name}': {error}"))?;
        let field = pyarrow
            .getattr("field")
            .map_err(|error| format!("pyarrow.field is missing: {error}"))?
            .call1((
                name,
                array.getattr("type").map_err(|error| {
                    format!("failed to inspect Arrow array type for '{name}': {error}")
                })?,
            ))
            .map_err(|error| format!("failed to build Arrow field for '{name}': {error}"))?;
        fields
            .append(field)
            .map_err(|error| format!("failed to append Arrow field for '{name}': {error}"))?;
    }

    Ok((arrays.unbind(), fields.unbind()))
}

fn resolve_temporal_output_type(
    values: &[OwnedValue],
    planned_type_name: Option<&str>,
    column_name: &str,
) -> Result<Option<TemporalOutputType>, String> {
    let planned = planned_type_name
        .map(parse_temporal_type_name)
        .transpose()?
        .flatten();
    let mut derived: Option<TemporalOutputType> = None;
    for value in values {
        let scalar = value.as_scalar()?;
        let current = match scalar {
            ScalarValue::RawTimestamp { unit, timezone, .. } => {
                Some(TemporalOutputType::Timestamp {
                    unit: unit.clone(),
                    timezone: timezone.clone(),
                })
            }
            ScalarValue::RawTime { unit, .. } => {
                Some(TemporalOutputType::Time64 { unit: unit.clone() })
            }
            ScalarValue::Null => None,
            _ if planned.is_some() => {
                return Err(format!(
                    "output column '{column_name}' does not match its temporal type expectation"
                ));
            }
            _ => None,
        };
        if let Some(current) = current {
            if let Some(expected) = planned.as_ref().or(derived.as_ref()) {
                if expected != &current {
                    return Err(format!(
                        "output column '{column_name}' contains incompatible temporal values: expected {expected:?}, got {current:?}"
                    ));
                }
            } else {
                derived = Some(current);
            }
        }
    }
    if derived.is_some() && planned_type_name.is_some() && planned.is_none() {
        return Err(format!(
            "raw temporal output column '{column_name}' has non-temporal type expectation '{}'",
            planned_type_name.unwrap_or_default()
        ));
    }
    Ok(planned.or(derived))
}

fn output_scalar_to_pyobject(
    py: Python<'_>,
    value: &OwnedValue,
    temporal_type: Option<&TemporalOutputType>,
) -> Result<Py<PyAny>, String> {
    match value.as_scalar()? {
        ScalarValue::RawTimestamp { value, .. } | ScalarValue::RawTime { value, .. }
            if temporal_type.is_some() =>
        {
            value
                .into_pyobject(py)
                .map_err(|error| error.to_string())
                .map(|value| value.unbind().into_any())
        }
        scalar => scalar_to_pyobject(py, scalar),
    }
}

fn gather_source_backed_column(
    column_name: &str,
    values: &[OwnedValue],
) -> Result<(ArrayRef, Field), String> {
    struct GatherGroup {
        array: ArrayRef,
        local_rows: Vec<u64>,
    }

    enum TargetSelection {
        Source {
            group_index: usize,
            row_index: usize,
        },
        Null {
            row_index: usize,
        },
    }

    let first_cell = values
        .iter()
        .find_map(|value| match value {
            OwnedValue::SourceCell(cell) => Some(cell),
            OwnedValue::Scalar(_) => None,
        })
        .ok_or_else(|| format!("output column '{column_name}' has no source-backed cells"))?;
    let source_field = first_cell.field()?;
    if !is_source_backed_type(source_field.data_type()) {
        return Err(format!(
            "output column '{column_name}' references unsupported source-backed type {:?}",
            source_field.data_type()
        ));
    }

    let mut groups: Vec<GatherGroup> = Vec::new();
    let mut group_index_by_location: HashMap<(SourceId, usize, usize), usize> = HashMap::new();
    let mut target_selection: Vec<TargetSelection> = Vec::with_capacity(values.len());
    let mut null_count = 0_usize;
    let mut nullable = source_field.is_nullable();
    for value in values {
        match value {
            OwnedValue::SourceCell(cell) => {
                let field = cell.field()?;
                if !source_fields_are_compatible(&source_field, &field) {
                    return Err(format!(
                        "output column '{column_name}' combines incompatible source-backed fields {:?} and {:?}",
                        source_field, field
                    ));
                }
                let local_row = u64::try_from(cell.batch_local_row()?)
                    .map_err(|_| format!("source row index is too large for '{column_name}'"))?;
                let location = (cell.source_id(), cell.batch_index()?, cell.slot_index());
                let group_index = if let Some(index) = group_index_by_location.get(&location) {
                    *index
                } else {
                    let index = groups.len();
                    groups.push(GatherGroup {
                        array: cell.array()?,
                        local_rows: Vec::new(),
                    });
                    group_index_by_location.insert(location, index);
                    index
                };
                let row_index = groups[group_index].local_rows.len();
                groups[group_index].local_rows.push(local_row);
                target_selection.push(TargetSelection::Source {
                    group_index,
                    row_index,
                });
            }
            OwnedValue::Scalar(ScalarValue::Null) => {
                nullable = true;
                target_selection.push(TargetSelection::Null {
                    row_index: null_count,
                });
                null_count += 1;
            }
            OwnedValue::Scalar(_) => {
                return Err(format!(
                    "output column '{column_name}' mixes source-backed and owned scalar values"
                ));
            }
        }
    }

    let mut chunks: Vec<ArrayRef> = Vec::with_capacity(groups.len() + usize::from(null_count > 0));
    let mut group_offsets: Vec<usize> = Vec::with_capacity(groups.len());
    let mut combined_len = 0_usize;
    for group in groups {
        group_offsets.push(combined_len);
        let gathered = take(
            group.array.as_ref(),
            &UInt64Array::from(group.local_rows),
            Some(TakeOptions { check_bounds: true }),
        )
        .map_err(|error| {
            format!("failed to gather source-backed output column '{column_name}': {error}")
        })?;
        combined_len += gathered.len();
        chunks.push(gathered);
    }
    let null_offset = combined_len;
    if null_count > 0 {
        chunks.push(new_null_array(source_field.data_type(), null_count));
    }
    let chunk_refs: Vec<&dyn Array> = chunks.iter().map(|chunk| chunk.as_ref()).collect();
    let grouped = concat(&chunk_refs).map_err(|error| {
        format!("failed to concatenate source-backed output column '{column_name}': {error}")
    })?;
    let target_indices = target_selection
        .into_iter()
        .map(|selection| match selection {
            TargetSelection::Source {
                group_index,
                row_index,
            } => group_offsets[group_index] + row_index,
            TargetSelection::Null { row_index } => null_offset + row_index,
        })
        .map(|index| {
            u64::try_from(index)
                .map_err(|_| format!("target row index is too large for '{column_name}'"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let array = take(
        grouped.as_ref(),
        &UInt64Array::from(target_indices),
        Some(TakeOptions { check_bounds: true }),
    )
    .map_err(|error| {
        format!("failed to restore target order for output column '{column_name}': {error}")
    })?;
    let field = source_field
        .as_ref()
        .clone()
        .with_name(column_name)
        .with_nullable(nullable);
    Ok((array, field))
}

fn source_fields_are_compatible(expected: &Field, actual: &Field) -> bool {
    expected.data_type() == actual.data_type()
        && expected.is_nullable() == actual.is_nullable()
        && expected.metadata() == actual.metadata()
}

fn is_source_backed_type(data_type: &arrow::datatypes::DataType) -> bool {
    matches!(
        data_type,
        arrow::datatypes::DataType::List(_)
            | arrow::datatypes::DataType::LargeList(_)
            | arrow::datatypes::DataType::FixedSizeList(_, _)
            | arrow::datatypes::DataType::Struct(_)
    )
}

fn import_native_array<'py>(
    py: Python<'py>,
    pyarrow: &Bound<'py, PyModule>,
    array: &ArrayRef,
    column_name: &str,
) -> Result<Bound<'py, PyAny>, String> {
    let (ffi_array, ffi_schema) = to_ffi(&array.to_data()).map_err(|error| {
        format!("failed to export source-backed Arrow array for '{column_name}': {error}")
    })?;
    let schema_capsule = ffi_capsule(py, ffi_schema, "arrow_schema")?;
    let array_capsule = ffi_capsule(py, ffi_array, "arrow_array")?;
    pyarrow
        .getattr("Array")
        .map_err(|error| format!("pyarrow.Array is missing: {error}"))?
        .call_method1("_import_from_c_capsule", (schema_capsule, array_capsule))
        .map_err(|error| {
            format!("failed to import source-backed Arrow array for '{column_name}': {error}")
        })
}

fn import_native_field<'py>(
    py: Python<'py>,
    pyarrow: &Bound<'py, PyModule>,
    field: &Field,
    column_name: &str,
) -> Result<Bound<'py, PyAny>, String> {
    let ffi_schema = FFI_ArrowSchema::try_from(field).map_err(|error| {
        format!("failed to export source-backed Arrow field for '{column_name}': {error}")
    })?;
    let schema_capsule = ffi_capsule(py, ffi_schema, "arrow_schema")?;
    pyarrow
        .getattr("Field")
        .map_err(|error| format!("pyarrow.Field is missing: {error}"))?
        .call_method1("_import_from_c_capsule", (schema_capsule,))
        .map_err(|error| {
            format!("failed to import source-backed Arrow field for '{column_name}': {error}")
        })
}

fn ffi_capsule<'py, T: 'static + Send>(
    py: Python<'py>,
    value: T,
    name: &str,
) -> Result<Bound<'py, PyCapsule>, String> {
    let name =
        CString::new(name).map_err(|error| format!("invalid Arrow capsule name: {error}"))?;
    PyCapsule::new(py, value, Some(name))
        .map_err(|error| format!("failed to create Arrow C Data capsule: {error}"))
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
        let (arrays, fields) = build_output_arrays(py, &pyarrow, output)?;
        let schema = pyarrow
            .getattr("schema")
            .map_err(|error| format!("pyarrow.schema is missing: {error}"))?
            .call1((fields.bind(py),))
            .map_err(|error| {
                format!("failed to build output schema for '{target_name}': {error}")
            })?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("schema", schema).map_err(|error| {
            format!("failed to set Arrow table schema for '{target_name}': {error}")
        })?;
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

fn scalar_to_pyobject(py: Python<'_>, value: &ScalarValue) -> Result<Py<PyAny>, String> {
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
        ScalarValue::Text(text) => Ok(text
            .clone()
            .into_pyobject(py)
            .map_err(|error| error.to_string())?
            .unbind()
            .into_any()),
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
        ScalarValue::RawTimestamp { .. } => Err(
            "raw timestamp cannot be converted to a Python object without losing its Arrow type"
                .to_string(),
        ),
        ScalarValue::RawTime { .. } => Err(
            "raw time cannot be converted to a Python object without losing its Arrow type"
                .to_string(),
        ),
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

    if value == -0.0 {
        0.0
    } else {
        value
    }
}

fn round_to_places(value: f64, places: u32) -> f64 {
    let factor = 10f64.powi(places as i32);
    if factor == 0.0 || !factor.is_finite() {
        return value;
    }
    (value * factor).round() / factor
}

#[cfg(test)]
mod tests {
    use super::gather_source_backed_column;
    use crate::row_cursor::NativeArrowRowCursor;
    use crate::source_registry::ExecutionSourceRegistry;
    use crate::source_view::NativeSourceView;
    use crate::value_ref::{OwnedValue, SourceCellRef};
    use arrow::array::{new_null_array, Array, ArrayRef, ListArray};
    use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
    use arrow::record_batch::RecordBatch;
    use std::collections::HashMap;
    use std::rc::Rc;
    use std::sync::Arc;

    fn registered_cursor(
        schema: Arc<Schema>,
        batches: Vec<RecordBatch>,
    ) -> (
        Rc<NativeArrowRowCursor>,
        Rc<ExecutionSourceRegistry<NativeArrowRowCursor>>,
    ) {
        let view = NativeSourceView::from_record_batches(batches).expect("source view");
        let cursor = Rc::new(NativeArrowRowCursor::from_source_view(view));
        let mut registry = ExecutionSourceRegistry::new();
        let source_id = registry.register(Rc::clone(&cursor));
        let registry = Rc::new(registry);
        cursor.assign_source_id(source_id).expect("source id");
        cursor
            .assign_source_registry(Rc::downgrade(&registry))
            .expect("source registry");
        assert_eq!(cursor.field_at(0), schema.fields().first().cloned());
        (cursor, registry)
    }

    fn source_value(
        cursor: &NativeArrowRowCursor,
        registry: &Rc<ExecutionSourceRegistry<NativeArrowRowCursor>>,
        row: usize,
    ) -> OwnedValue {
        OwnedValue::SourceCell(
            SourceCellRef::new(Rc::downgrade(registry), cursor.source_id().unwrap(), row, 0)
                .expect("source cell"),
        )
    }

    #[test]
    fn source_backed_gather_preserves_order_duplicates_nulls_and_field_metadata() {
        let item = Arc::new(Field::new("item", DataType::Int32, true));
        let field = Field::new("nested", DataType::List(item), true).with_metadata(HashMap::from(
            [("column-role".to_string(), "nested".to_string())],
        ));
        let schema = Arc::new(Schema::new(vec![field.clone()]));
        let first = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(ListArray::from_iter_primitive::<
                arrow::datatypes::Int32Type,
                _,
                _,
            >(vec![
                Some(vec![Some(1_i32)]),
                Some(vec![Some(2_i32)]),
            ])) as ArrayRef],
        )
        .unwrap();
        let second = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(ListArray::from_iter_primitive::<
                arrow::datatypes::Int32Type,
                _,
                _,
            >(vec![None, Some(vec![Some(3_i32)])]))
                as ArrayRef],
        )
        .unwrap();
        let (cursor, registry) = registered_cursor(Arc::clone(&schema), vec![first, second]);
        let values = vec![
            source_value(&cursor, &registry, 3),
            source_value(&cursor, &registry, 0),
            source_value(&cursor, &registry, 2),
            source_value(&cursor, &registry, 0),
        ];

        let (array, output_field) =
            gather_source_backed_column("renamed", &values).expect("source gather");
        let expected = ListArray::from_iter_primitive::<arrow::datatypes::Int32Type, _, _>(vec![
            Some(vec![Some(3_i32)]),
            Some(vec![Some(1_i32)]),
            None,
            Some(vec![Some(1_i32)]),
        ]);

        assert_eq!(array.to_data(), expected.to_data());
        assert_eq!(output_field.name(), "renamed");
        assert_eq!(output_field.data_type(), field.data_type());
        assert_eq!(output_field.metadata(), field.metadata());
    }

    #[test]
    fn source_backed_gather_preserves_nested_temporal_identity() {
        let timestamp = DataType::Timestamp(TimeUnit::Nanosecond, Some("America/New_York".into()));
        let item = Arc::new(
            Field::new("instant", timestamp, true).with_metadata(HashMap::from([(
                "unit-owner".to_string(),
                "source".to_string(),
            )])),
        );
        let field = Field::new("events", DataType::List(item), true);
        let schema = Arc::new(Schema::new(vec![field.clone()]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![new_null_array(field.data_type(), 2)],
        )
        .unwrap();
        let (cursor, registry) = registered_cursor(schema, vec![batch]);
        let values = vec![
            source_value(&cursor, &registry, 1),
            source_value(&cursor, &registry, 0),
        ];

        let (array, output_field) =
            gather_source_backed_column("events", &values).expect("source gather");

        assert_eq!(array.data_type(), field.data_type());
        assert_eq!(output_field.data_type(), field.data_type());
        assert_eq!(array.null_count(), 2);
    }

    #[test]
    fn source_backed_gather_rejects_owned_nested_value_mixing() {
        let item = Arc::new(Field::new("item", DataType::Int32, true));
        let field = Field::new("nested", DataType::List(item), true);
        let schema = Arc::new(Schema::new(vec![field.clone()]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![new_null_array(field.data_type(), 1)],
        )
        .unwrap();
        let (cursor, registry) = registered_cursor(schema, vec![batch]);
        let values = vec![
            source_value(&cursor, &registry, 0),
            OwnedValue::from(crate::expressions::ScalarValue::List(vec![
                crate::expressions::ScalarValue::Integer(1),
            ])),
        ];

        let error = gather_source_backed_column("nested", &values)
            .expect_err("mixed ownership must fail closed");

        assert!(error.contains("source-backed"));
        assert!(error.contains("nested"));
    }
}
