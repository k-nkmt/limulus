use crate::expressions::{column_key, py_to_scalar, scalar_to_pyobject, ScalarValue};
use crate::source_view::NativeSourceView;
use crate::value_ref::SourceValueRef;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use std::collections::HashMap;

pub(crate) struct NativeArrowRowCursor {
    slot_order: Vec<String>,
    row_count: usize,
    normalized_slot_index: HashMap<String, usize>,
    source_view: Option<NativeSourceView>,
    materialized_columns: Vec<Vec<ScalarValue>>,
}

impl NativeArrowRowCursor {
    pub(crate) fn from_arrow_stream(
        _py: Python<'_>,
        stream_obj: &Bound<'_, PyAny>,
    ) -> Result<Self, String> {
        NativeSourceView::from_arrow_stream(stream_obj).map(Self::from_source_view)
    }

    pub(crate) fn from_row_list(py: Python<'_>, row_list: &Bound<'_, PyList>) -> Result<Self, String> {
        let mut rows: Vec<Py<PyDict>> = Vec::with_capacity(row_list.len());
        for row_any in row_list.iter() {
            let row = row_any
                .cast::<PyDict>()
                .map_err(|error| format!("legacy input row is not a dict: {error}"))?;
            rows.push(row.clone().unbind());
        }

        let slot_order = rows
            .first()
            .map(|row| {
                let bound = row.bind(py);
                bound
                    .iter()
                    .map(|(key_any, _)| {
                        key_any
                            .extract::<String>()
                            .map_err(|error| format!("row key extract failed: {error}"))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?
            .unwrap_or_default();

        let mut materialized_columns: Vec<Vec<ScalarValue>> = (0..slot_order.len())
            .map(|_| Vec::with_capacity(rows.len()))
            .collect();

        for row in &rows {
            let bound = row.bind(py);
            for (slot_index, slot_name) in slot_order.iter().enumerate() {
                let value_any = bound
                    .get_item(slot_name)
                    .map_err(|error| format!("failed to access legacy slot value '{slot_name}': {error}"))?;
                let value = match value_any {
                    Some(item) => py_to_scalar(&item).map_err(|error| error.to_string())?,
                    None => ScalarValue::Null,
                };
                materialized_columns[slot_index].push(value);
            }
        }

        Ok(Self::from_materialized_columns(
            slot_order,
            materialized_columns,
            rows.len(),
        ))
    }

    fn from_source_view(source_view: NativeSourceView) -> Self {
        let slot_order = source_view.slot_order().to_vec();
        let row_count = source_view.row_count();
        Self {
            normalized_slot_index: normalized_slot_index(&slot_order),
            slot_order,
            row_count,
            source_view: Some(source_view),
            materialized_columns: Vec::new(),
        }
    }

    fn from_materialized_columns(
        slot_order: Vec<String>,
        materialized_columns: Vec<Vec<ScalarValue>>,
        row_count: usize,
    ) -> Self {
        Self {
            normalized_slot_index: normalized_slot_index(&slot_order),
            slot_order,
            row_count,
            source_view: None,
            materialized_columns,
        }
    }

    pub(crate) fn row_count(&self) -> usize {
        self.row_count
    }

    pub(crate) fn row_dict_at(&self, py: Python<'_>, row_index: usize) -> Result<Py<PyDict>, String> {
        if row_index >= self.row_count {
            return Err(format!("row index out of range: {row_index}"));
        }

        let row = PyDict::new(py);
        for (slot_index, slot_name) in self.slot_order.iter().enumerate() {
            if let Some(value) = self.value_at_slot(py, row_index, slot_index)? {
                row.set_item(slot_name, value)
                    .map_err(|error| format!("failed to materialize row slot '{slot_name}': {error}"))?;
            }
        }
        Ok(row.unbind())
    }

    pub(crate) fn value_at_slot(
        &self,
        py: Python<'_>,
        row_index: usize,
        slot_index: usize,
    ) -> Result<Option<Py<PyAny>>, String> {
        self.source_value_at_slot(row_index, slot_index)
            .and_then(|value| value.map(|value_ref| scalar_to_pyobject(py, value_ref.as_scalar())).transpose())
    }

    pub(crate) fn source_value_at_slot(
        &self,
        row_index: usize,
        slot_index: usize,
    ) -> Result<Option<SourceValueRef<'_>>, String> {
        if slot_index >= self.slot_order.len() {
            return Ok(None);
        }
        if row_index >= self.row_count {
            return Err(format!("row index out of range: {row_index}"));
        }
        if let Some(source_view) = self.source_view.as_ref() {
            return source_view.value_at(row_index, slot_index);
        }
        let column = self
            .materialized_columns
            .get(slot_index)
            .ok_or_else(|| format!("slot index out of range: {slot_index}"))?;
        let value = column
            .get(row_index)
            .ok_or_else(|| format!("row index out of range: {row_index}"))?;
        Ok(Some(SourceValueRef::Scalar(value)))
    }

    #[allow(dead_code)]
    pub(crate) fn scalar_at_slot(
        &self,
        row_index: usize,
        slot_index: usize,
    ) -> Result<Option<ScalarValue>, String> {
        self.source_value_at_slot(row_index, slot_index)
            .map(|value| value.map(|value_ref| value_ref.to_owned_value().into_scalar_value()))
    }

    #[allow(dead_code)]
    pub(crate) fn value_by_name(
        &self,
        py: Python<'_>,
        row_index: usize,
        name: &str,
    ) -> Result<Option<Py<PyAny>>, String> {
        self.source_value_by_name(row_index, name)
            .and_then(|value| value.map(|value_ref| scalar_to_pyobject(py, value_ref.as_scalar())).transpose())
    }

    pub(crate) fn source_value_by_name(
        &self,
        row_index: usize,
        name: &str,
    ) -> Result<Option<SourceValueRef<'_>>, String> {
        let normalized = column_key(name);
        self.source_value_by_normalized_name(row_index, &normalized)
    }

    pub(crate) fn source_value_by_normalized_name(
        &self,
        row_index: usize,
        normalized_name: &str,
    ) -> Result<Option<SourceValueRef<'_>>, String> {
        let Some(slot_index) = self.normalized_slot_index.get(normalized_name).copied() else {
            return Ok(None);
        };
        self.source_value_at_slot(row_index, slot_index)
    }

    #[allow(dead_code)]
    pub(crate) fn scalar_by_name(
        &self,
        row_index: usize,
        name: &str,
    ) -> Result<Option<ScalarValue>, String> {
        self.source_value_by_name(row_index, name)
            .map(|value| value.map(|value_ref| value_ref.to_owned_value().into_scalar_value()))
    }

    pub(crate) fn validate_source_slot_order(&self, source_slot_order: &[String]) -> Result<(), String> {
        if source_slot_order.is_empty() {
            return Ok(());
        }

        let mut plan_index_by_key: HashMap<String, usize> = HashMap::new();
        for (index, slot_name) in source_slot_order.iter().enumerate() {
            plan_index_by_key.insert(column_key(slot_name), index);
        }

        let mut plan_index: usize = 0;
        for cursor_slot in &self.slot_order {
            let cursor_key = column_key(cursor_slot);
            let Some(target_index) = plan_index_by_key.get(&cursor_key).copied() else {
                continue;
            };
            if target_index < plan_index {
                return Err(format!(
                    "source slot order mismatch: cursor slot '{cursor_slot}' violates row_loop_plan.source_slot_order"
                ));
            }
            plan_index = target_index + 1;
        }

        Ok(())
    }
}

fn normalized_slot_index(slot_order: &[String]) -> HashMap<String, usize> {
    let mut normalized_slot_index = HashMap::new();
    for (slot_index, slot_name) in slot_order.iter().enumerate() {
        normalized_slot_index.insert(column_key(slot_name), slot_index);
    }
    normalized_slot_index
}
