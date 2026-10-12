use crate::expressions::column_key;
use crate::row_cursor::NativeArrowRowCursor;
use crate::source_registry::{ExecutionSourceRegistry, SourceId};
use crate::value_ref::{OwnedValue, SourceCellRef, SourceValueRef};
use arrow::datatypes::Field;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

type SourceRegistry = ExecutionSourceRegistry<NativeArrowRowCursor>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ColumnBinding {
    Source { source_id: SourceId, slot: usize },
    Owned { normalized_name: String },
    Automatic { normalized_name: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LogicalRowError {
    InvalidAlias {
        name: String,
        reason: String,
    },
    InvalidBindingName {
        logical_name: String,
        normalized_name: String,
        reason: String,
    },
    NameCollision {
        normalized_name: String,
        first: String,
        second: String,
    },
    InvalidSourceBinding {
        logical_name: String,
        source_id: SourceId,
        slot: usize,
        reason: String,
    },
    ColumnNotVisible {
        logical_name: String,
    },
    AutomaticColumnReadOnly {
        logical_name: String,
    },
    MissingOwnedValue {
        logical_name: String,
        normalized_name: String,
    },
    MissingAutomaticValue {
        logical_name: String,
        normalized_name: String,
    },
}

impl fmt::Display for LogicalRowError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAlias { name, reason } => {
                write!(formatter, "invalid logical column alias '{name}': {reason}")
            }
            Self::InvalidBindingName {
                logical_name,
                normalized_name,
                reason,
            } => write!(
                formatter,
                "invalid binding name '{normalized_name}' for logical column '{logical_name}': {reason}"
            ),
            Self::NameCollision {
                normalized_name,
                first,
                second,
            } => write!(
                formatter,
                "logical column name collision for '{normalized_name}': '{first}' and '{second}'"
            ),
            Self::InvalidSourceBinding {
                logical_name,
                source_id,
                slot,
                reason,
            } => write!(
                formatter,
                "invalid source binding for logical column '{logical_name}' (source_id={source_id:?}, slot={slot}): {reason}"
            ),
            Self::ColumnNotVisible { logical_name } => write!(
                formatter,
                "logical column '{logical_name}' is not visible and cannot be written"
            ),
            Self::AutomaticColumnReadOnly { logical_name } => write!(
                formatter,
                "automatic logical column '{logical_name}' is read-only"
            ),
            Self::MissingOwnedValue {
                logical_name,
                normalized_name,
            } => write!(
                formatter,
                "owned value for logical column '{logical_name}' is missing (binding={normalized_name})"
            ),
            Self::MissingAutomaticValue {
                logical_name,
                normalized_name,
            } => write!(
                formatter,
                "automatic value for logical column '{logical_name}' is missing (binding={normalized_name})"
            ),
        }
    }
}

impl std::error::Error for LogicalRowError {}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ResolvedValue<'a> {
    SourceScalar(SourceValueRef<'a>),
    SourceCell(SourceCellRef),
    Owned(&'a OwnedValue),
    Automatic(&'a OwnedValue),
}

pub(crate) struct VisibleColumn<'a> {
    logical_name: &'a str,
    binding: &'a ColumnBinding,
    source_field: Option<&'a Field>,
}

impl<'a> VisibleColumn<'a> {
    pub(crate) fn logical_name(&self) -> &'a str {
        self.logical_name
    }

    pub(crate) fn binding(&self) -> &'a ColumnBinding {
        self.binding
    }

    pub(crate) fn source_field(&self) -> Option<&'a Field> {
        self.source_field
    }
}

pub(crate) trait LogicalRowAccess {
    fn resolve(&self, logical_name: &str) -> Result<Option<ResolvedValue<'_>>, LogicalRowError>;
    fn visible_columns(&self) -> Result<Vec<VisibleColumn<'_>>, LogicalRowError>;
    fn write_owned(&mut self, logical_name: &str, value: OwnedValue)
        -> Result<(), LogicalRowError>;
}

#[derive(Clone)]
pub(crate) struct LogicalRowView {
    row_index: usize,
    columns: Vec<(String, ColumnBinding)>,
    column_index: HashMap<String, usize>,
    source_fields: Vec<Option<Field>>,
    source_registry: Rc<SourceRegistry>,
    source_cursors: HashMap<SourceId, Rc<NativeArrowRowCursor>>,
    owned_values: HashMap<String, OwnedValue>,
    automatic_values: HashMap<String, OwnedValue>,
    projection_applied: bool,
}

/// Native-owned logical row contract used by all row consumers.
pub(crate) type NativeLogicalRowView = LogicalRowView;

impl LogicalRowView {
    pub(crate) fn new(
        row_index: usize,
        source_registry: Rc<SourceRegistry>,
        columns: Vec<(String, ColumnBinding)>,
    ) -> Result<Self, LogicalRowError> {
        Self::with_values(
            row_index,
            source_registry,
            columns,
            HashMap::new(),
            HashMap::new(),
        )
    }

    pub(crate) fn with_values(
        row_index: usize,
        source_registry: Rc<SourceRegistry>,
        columns: Vec<(String, ColumnBinding)>,
        owned_values: HashMap<String, OwnedValue>,
        automatic_values: HashMap<String, OwnedValue>,
    ) -> Result<Self, LogicalRowError> {
        Self::with_values_internal(
            row_index,
            source_registry,
            columns,
            owned_values,
            automatic_values,
            true,
        )
    }

    pub(crate) fn from_schema(
        source_registry: Rc<SourceRegistry>,
        columns: Vec<(String, ColumnBinding)>,
    ) -> Result<Self, LogicalRowError> {
        Self::with_values_internal(
            0,
            source_registry,
            columns,
            HashMap::new(),
            HashMap::new(),
            false,
        )
    }

    fn with_values_internal(
        row_index: usize,
        source_registry: Rc<SourceRegistry>,
        columns: Vec<(String, ColumnBinding)>,
        owned_values: HashMap<String, OwnedValue>,
        automatic_values: HashMap<String, OwnedValue>,
        validate_source_row: bool,
    ) -> Result<Self, LogicalRowError> {
        let column_index = validate_columns(&columns)?;
        let mut owned_values = normalize_value_map(owned_values, "owned")?;
        let automatic_values = normalize_value_map(automatic_values, "automatic")?;
        let mut source_fields = Vec::with_capacity(columns.len());
        let mut source_cursors = HashMap::new();

        for (logical_name, binding) in &columns {
            match binding {
                ColumnBinding::Source { source_id, slot } => {
                    let cursor = if let Some(cursor) = source_cursors.get(source_id) {
                        Rc::clone(cursor)
                    } else {
                        let cursor = source_registry
                            .lookup(*source_id)
                            .map_err(|error| {
                                invalid_source(logical_name, *source_id, *slot, error.to_string())
                            })?
                            .cursor();
                        cursor.assign_source_id(*source_id).map_err(|error| {
                            invalid_source(logical_name, *source_id, *slot, error)
                        })?;
                        cursor
                            .assign_source_registry(Rc::downgrade(&source_registry))
                            .map_err(|error| {
                                invalid_source(logical_name, *source_id, *slot, error)
                            })?;
                        source_cursors.insert(*source_id, Rc::clone(&cursor));
                        cursor
                    };

                    if validate_source_row {
                        cursor
                            .validate_source_cell(row_index, *slot)
                            .map_err(|error| invalid_source(logical_name, *source_id, *slot, error))?;
                    }
                    let field = cursor.field_at(*slot).ok_or_else(|| {
                        invalid_source(
                            logical_name,
                            *source_id,
                            *slot,
                            "source field is not available".to_string(),
                        )
                    })?;
                    source_fields.push(Some(field.as_ref().clone()));
                }
                ColumnBinding::Owned { normalized_name } => {
                    validate_binding_name(logical_name, normalized_name)?;
                    owned_values
                        .entry(normalized_name.clone())
                        .or_insert_with(OwnedValue::null);
                    source_fields.push(None);
                }
                ColumnBinding::Automatic { normalized_name } => {
                    validate_binding_name(logical_name, normalized_name)?;
                    source_fields.push(None);
                }
            }
        }

        Ok(Self {
            row_index,
            columns,
            column_index,
            source_fields,
            source_registry,
            source_cursors,
            owned_values,
            automatic_values,
            projection_applied: false,
        })
    }

    pub(crate) fn row_index(&self) -> usize {
        self.row_index
    }

    pub(crate) fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Advance an unprojected source view without rebuilding its fixed schema.
    /// The runtime bounds the row index with its source cursor and resets row
    /// values separately. Columns introduced only in the previous row expire.
    pub(crate) fn reset_row_layout(&mut self, row_index: usize, schema_column_count: usize) {
        debug_assert!(!self.projection_applied);
        debug_assert!(schema_column_count <= self.columns.len());
        if self.columns.len() > schema_column_count {
            for (name, _) in self.columns.drain(schema_column_count..) {
                self.column_index.remove(&column_key(&name));
            }
            self.source_fields.truncate(schema_column_count);
        }
        self.row_index = row_index;
    }

    pub(crate) fn row_count(&self) -> usize {
        self.source_cursors
            .values()
            .map(|cursor| cursor.row_count())
            .min()
            .unwrap_or(0)
    }

    pub(crate) fn for_row_index(
        &self,
        row_index: usize,
    ) -> Result<Self, LogicalRowError> {
        let mut view = self.clone();
        view.row_index = row_index;
        // A row snapshot used for BY adjacency compares source bindings.  Do
        // not carry current-row owned/automatic mutations into the neighboring
        // source row.
        view.owned_values.clear();
        view.automatic_values.clear();
        for (logical_name, binding) in &view.columns {
            if let ColumnBinding::Source { source_id, slot } = binding {
                let cursor = view.source_cursors.get(source_id).ok_or_else(|| {
                    invalid_source(
                        logical_name,
                        *source_id,
                        *slot,
                        "source cursor is not available".to_string(),
                    )
                })?;
                cursor
                    .validate_source_cell(row_index, *slot)
                    .map_err(|error| invalid_source(logical_name, *source_id, *slot, error))?;
            }
        }
        Ok(view)
    }

    pub(crate) fn set_owned_value(
        &mut self,
        logical_name: &str,
        value: OwnedValue,
    ) -> Result<(), LogicalRowError> {
        let normalized_name = normalize_logical_name(logical_name)?;
        if !self.column_index.contains_key(&normalized_name) {
            if self.projection_applied {
                return Err(LogicalRowError::ColumnNotVisible {
                    logical_name: logical_name.to_string(),
                });
            }
            self.columns.push((
                logical_name.trim().to_string(),
                ColumnBinding::Owned {
                    normalized_name: normalized_name.clone(),
                },
            ));
            self.source_fields.push(None);
            self.column_index.insert(normalized_name.clone(), self.columns.len() - 1);
        }
        self.owned_values.insert(normalized_name, value);
        Ok(())
    }

    pub(crate) fn ensure_owned_column(
        &mut self,
        logical_name: &str,
    ) -> Result<(), LogicalRowError> {
        let normalized_name = normalize_logical_name(logical_name)?;
        if self.column_index.contains_key(&normalized_name) {
            return Ok(());
        }
        if self.projection_applied {
            return Err(LogicalRowError::ColumnNotVisible {
                logical_name: logical_name.to_string(),
            });
        }
        self.columns.push((
            logical_name.trim().to_string(),
            ColumnBinding::Owned {
                normalized_name: normalized_name.clone(),
            },
        ));
        self.source_fields.push(None);
        self.column_index
            .insert(normalized_name, self.columns.len() - 1);
        Ok(())
    }

    pub(crate) fn project_columns(
        &mut self,
        keep: &[String],
        drop: &[String],
        rename: &HashMap<String, String>,
    ) -> Result<(), LogicalRowError> {
        if keep.is_empty() && drop.is_empty() && rename.is_empty() {
            // An empty projection only freezes visibility. The validated
            // bindings, field ownership, and value keys remain unchanged.
            self.projection_applied = true;
            return Ok(());
        }
        let mut projected = Vec::new();
        let mut projected_fields = Vec::new();
        let mut projected_owned_values = Vec::new();
        for (index, (logical_name, binding)) in self.columns.iter().enumerate() {
            if !keep.is_empty()
                && !keep
                    .iter()
                    .any(|name| name.eq_ignore_ascii_case(logical_name))
            {
                continue;
            }
            if drop
                .iter()
                .any(|name| name.eq_ignore_ascii_case(logical_name))
            {
                continue;
            }
            let projected_name = rename
                .iter()
                .find(|(source, _)| source.eq_ignore_ascii_case(logical_name))
                .map(|(_, target)| target.clone())
                .unwrap_or_else(|| logical_name.clone());
            let projected_normalized_name = column_key(&projected_name);
            let source_normalized_name = column_key(logical_name);
            if let Some(value) = self
                .owned_values
                .get(&source_normalized_name)
                .or_else(|| self.owned_values.get(&projected_normalized_name))
            {
                projected_owned_values.push((projected_normalized_name, value.clone()));
            }
            projected.push((projected_name, binding.clone()));
            projected_fields.push(self.source_fields[index].clone());
        }

        let projected_index = validate_columns(&projected)?;
        self.columns = projected;
        self.source_fields = projected_fields;
        self.column_index = projected_index;
        for (name, value) in projected_owned_values {
            self.owned_values.insert(name, value);
        }
        self.projection_applied = true;
        Ok(())
    }

    pub(crate) fn set_automatic_value(
        &mut self,
        logical_name: &str,
        value: OwnedValue,
    ) -> Result<(), LogicalRowError> {
        let normalized_name = normalize_logical_name(logical_name)?;
        self.automatic_values.insert(normalized_name, value);
        Ok(())
    }

    pub(crate) fn clear_automatic_values(&mut self) {
        self.automatic_values.clear();
    }

    pub(crate) fn clear_owned_values(&mut self) {
        self.owned_values.clear();
        for (_, binding) in &self.columns {
            if let ColumnBinding::Owned { normalized_name } = binding {
                self.owned_values
                    .insert(normalized_name.clone(), OwnedValue::null());
            }
        }
    }

    fn resolve_source(
        &self,
        logical_name: &str,
        source_id: SourceId,
        slot: usize,
    ) -> Result<Option<ResolvedValue<'_>>, LogicalRowError> {
        let registered_cursor = self
            .source_registry
            .lookup(source_id)
            .map_err(|error| invalid_source(logical_name, source_id, slot, error.to_string()))?
            .cursor();
        let cursor = self.source_cursors.get(&source_id).ok_or_else(|| {
            invalid_source(
                logical_name,
                source_id,
                slot,
                "source cursor is not available".to_string(),
            )
        })?;
        if !Rc::ptr_eq(&registered_cursor, cursor) {
            return Err(invalid_source(
                logical_name,
                source_id,
                slot,
                "source cursor owner does not match the execution registry".to_string(),
            ));
        }

        let value = cursor
            .source_value_at_slot(self.row_index, slot)
            .map_err(|error| invalid_source(logical_name, source_id, slot, error))?;
        match value {
            Some(SourceValueRef::Scalar(value)) => Ok(Some(ResolvedValue::SourceScalar(
                SourceValueRef::Scalar(value),
            ))),
            Some(SourceValueRef::Cell(cell)) => Ok(Some(ResolvedValue::SourceCell(cell))),
            None => Ok(None),
        }
    }
}

impl LogicalRowAccess for LogicalRowView {
    fn resolve(&self, logical_name: &str) -> Result<Option<ResolvedValue<'_>>, LogicalRowError> {
        let normalized_name = normalize_logical_name(logical_name)?;
        if let Some(value) = self.automatic_values.get(&normalized_name) {
            return Ok(Some(ResolvedValue::Automatic(value)));
        }
        let Some(column_index) = self.column_index.get(&normalized_name).copied() else {
            return Ok(None);
        };
        let (visible_name, binding) = &self.columns[column_index];

        match binding {
            ColumnBinding::Source { source_id, slot } => {
                if let Some(value) = self.owned_values.get(&normalized_name) {
                    return Ok(Some(ResolvedValue::Owned(value)));
                }
                self.resolve_source(visible_name, *source_id, *slot)
            }
            ColumnBinding::Owned {
                normalized_name: binding_name,
            } => {
                let value = self
                    .owned_values
                    .get(binding_name)
                    .or_else(|| self.owned_values.get(&normalized_name))
                    .ok_or_else(|| LogicalRowError::MissingOwnedValue {
                        logical_name: visible_name.clone(),
                        normalized_name: binding_name.clone(),
                    })?;
                Ok(Some(ResolvedValue::Owned(value)))
            }
            ColumnBinding::Automatic {
                normalized_name: binding_name,
            } => {
                let value = self
                    .automatic_values
                    .get(binding_name)
                    .or_else(|| self.automatic_values.get(&normalized_name))
                    .ok_or_else(|| LogicalRowError::MissingAutomaticValue {
                        logical_name: visible_name.clone(),
                        normalized_name: binding_name.clone(),
                    })?;
                Ok(Some(ResolvedValue::Automatic(value)))
            }
        }
    }

    fn visible_columns(&self) -> Result<Vec<VisibleColumn<'_>>, LogicalRowError> {
        Ok(self
            .columns
            .iter()
            .enumerate()
            .map(|(index, (logical_name, binding))| VisibleColumn {
                logical_name,
                binding,
                source_field: self.source_fields[index].as_ref(),
            })
            .collect())
    }

    fn write_owned(
        &mut self,
        logical_name: &str,
        value: OwnedValue,
    ) -> Result<(), LogicalRowError> {
        let normalized_name = normalize_logical_name(logical_name)?;
        let Some(column_index) = self.column_index.get(&normalized_name).copied() else {
            return Err(LogicalRowError::ColumnNotVisible {
                logical_name: logical_name.to_string(),
            });
        };

        let (_, binding) = &self.columns[column_index];
        match binding {
            ColumnBinding::Source { .. } => {
                self.owned_values.insert(normalized_name, value);
            }
            ColumnBinding::Owned { normalized_name } => {
                self.owned_values.insert(normalized_name.clone(), value);
            }
            ColumnBinding::Automatic { .. } => {
                return Err(LogicalRowError::AutomaticColumnReadOnly {
                    logical_name: logical_name.to_string(),
                });
            }
        }
        Ok(())
    }
}

fn validate_columns(
    columns: &[(String, ColumnBinding)],
) -> Result<HashMap<String, usize>, LogicalRowError> {
    let mut column_index = HashMap::with_capacity(columns.len());
    for (index, (logical_name, binding)) in columns.iter().enumerate() {
        let normalized_name = normalize_logical_name(logical_name)?;
        if let Some(previous_index) = column_index.insert(normalized_name.clone(), index) {
            return Err(LogicalRowError::NameCollision {
                normalized_name,
                first: columns[previous_index].0.clone(),
                second: logical_name.clone(),
            });
        }
        if let ColumnBinding::Owned { normalized_name }
        | ColumnBinding::Automatic { normalized_name } = binding
        {
            validate_binding_name(logical_name, normalized_name)?;
        }
    }
    Ok(column_index)
}

fn validate_binding_name(logical_name: &str, normalized_name: &str) -> Result<(), LogicalRowError> {
    let expected = column_key(normalized_name);
    if expected.is_empty() {
        return Err(LogicalRowError::InvalidBindingName {
            logical_name: logical_name.to_string(),
            normalized_name: normalized_name.to_string(),
            reason: "binding name must not be empty".to_string(),
        });
    }
    if expected != normalized_name {
        return Err(LogicalRowError::InvalidBindingName {
            logical_name: logical_name.to_string(),
            normalized_name: normalized_name.to_string(),
            reason: "binding name must already be normalized".to_string(),
        });
    }
    Ok(())
}

fn normalize_logical_name(name: &str) -> Result<String, LogicalRowError> {
    let normalized = column_key(name);
    if normalized.is_empty() {
        return Err(LogicalRowError::InvalidAlias {
            name: name.to_string(),
            reason: "logical name must not be empty".to_string(),
        });
    }
    Ok(normalized)
}

fn normalize_value_map(
    values: HashMap<String, OwnedValue>,
    owner: &str,
) -> Result<HashMap<String, OwnedValue>, LogicalRowError> {
    let mut normalized_values = HashMap::with_capacity(values.len());
    let mut original_names = HashMap::with_capacity(values.len());
    for (name, value) in values {
        let normalized_name = normalize_logical_name(&name)?;
        if let Some(first) = original_names.insert(normalized_name.clone(), name.clone()) {
            return Err(LogicalRowError::NameCollision {
                normalized_name,
                first: format!("{owner}:{first}"),
                second: format!("{owner}:{name}"),
            });
        }
        normalized_values.insert(normalized_name, value);
    }
    Ok(normalized_values)
}

fn invalid_source(
    logical_name: &str,
    source_id: SourceId,
    slot: usize,
    reason: String,
) -> LogicalRowError {
    LogicalRowError::InvalidSourceBinding {
        logical_name: logical_name.to_string(),
        source_id,
        slot,
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::{ColumnBinding, LogicalRowAccess, LogicalRowView, ResolvedValue};
    use crate::expressions::ScalarValue;
    use crate::row_cursor::NativeArrowRowCursor;
    use crate::source_registry::{ExecutionSourceRegistry, SourceId};
    use crate::value_ref::{OwnedValue, SourceValueRef};
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyList};
    use std::collections::HashMap;
    use std::rc::Rc;

    fn empty_registry() -> Rc<ExecutionSourceRegistry<NativeArrowRowCursor>> {
        Rc::new(ExecutionSourceRegistry::new())
    }

    #[test]
    fn resolves_owned_and_automatic_values_case_insensitively() {
        let mut owned_values = HashMap::new();
        owned_values.insert(
            "result".to_string(),
            OwnedValue::from(ScalarValue::Integer(7)),
        );
        let mut automatic_values = HashMap::new();
        automatic_values.insert(
            "_n_".to_string(),
            OwnedValue::from(ScalarValue::Number(3.0)),
        );

        let view = LogicalRowView::with_values(
            0,
            empty_registry(),
            vec![
                (
                    "Result".to_string(),
                    ColumnBinding::Owned {
                        normalized_name: "RESULT".to_string(),
                    },
                ),
                (
                    "_N_".to_string(),
                    ColumnBinding::Automatic {
                        normalized_name: "_N_".to_string(),
                    },
                ),
            ],
            owned_values,
            automatic_values,
        )
        .expect("logical row view");

        match view.resolve(" result ").expect("owned resolution") {
            Some(ResolvedValue::Owned(OwnedValue::Scalar(ScalarValue::Integer(value)))) => {
                assert_eq!(*value, 7)
            }
            other => panic!("unexpected owned value: {other:?}"),
        }
        match view.resolve("_n_").expect("automatic resolution") {
            Some(ResolvedValue::Automatic(OwnedValue::Scalar(ScalarValue::Number(value)))) => {
                assert_eq!(*value, 3.0)
            }
            other => panic!("unexpected automatic value: {other:?}"),
        }
    }

    #[test]
    fn write_owned_overrides_source_binding_without_changing_source() {
        Python::attach(|py| {
            let cursor = source_cursor(py);
            let mut registry = ExecutionSourceRegistry::new();
            let source_id = registry.register(Rc::clone(&cursor));
            let registry = Rc::new(registry);
            let mut view = LogicalRowView::new(
                0,
                registry,
                vec![(
                    "Value".to_string(),
                    ColumnBinding::Source { source_id, slot: 0 },
                )],
            )
            .expect("logical row view");

            match view.resolve("value").expect("source resolution") {
                Some(ResolvedValue::SourceScalar(SourceValueRef::Scalar(
                    ScalarValue::Integer(value),
                ))) => assert_eq!(*value, 1),
                other => panic!("unexpected source value: {other:?}"),
            }

            view.write_owned("value", OwnedValue::from(ScalarValue::Unsigned(9)))
                .expect("owned write");

            match view.resolve("VALUE").expect("owned resolution") {
                Some(ResolvedValue::Owned(OwnedValue::Scalar(ScalarValue::Unsigned(value)))) => {
                    assert_eq!(*value, 9)
                }
                other => panic!("unexpected value after write: {other:?}"),
            }
            match cursor
                .source_value_at_slot(0, 0)
                .expect("source remains accessible")
            {
                Some(SourceValueRef::Scalar(ScalarValue::Integer(value))) => assert_eq!(*value, 1),
                other => panic!("unexpected source value after write: {other:?}"),
            }
        });
    }

    #[test]
    fn reused_row_layout_discards_transient_columns_and_source_overrides() {
        Python::attach(|py| {
            let cursor = source_cursor(py);
            let mut registry = ExecutionSourceRegistry::new();
            let source_id = registry.register(cursor);
            let mut view = LogicalRowView::from_schema(
                Rc::new(registry),
                vec![("ID".to_string(), ColumnBinding::Source { source_id, slot: 0 })],
            )
            .expect("source schema");
            view.ensure_owned_column("Result").expect("declared result");
            let schema_column_count = view.column_count();
            view.set_owned_value("id", OwnedValue::from(ScalarValue::Integer(99)))
                .expect("source override");
            view.set_owned_value("result", OwnedValue::from(ScalarValue::Integer(7)))
                .expect("row result");
            view.set_owned_value("Transient", OwnedValue::from(ScalarValue::Integer(8)))
                .expect("transient column");
            view.set_automatic_value("_N_", OwnedValue::from(ScalarValue::Integer(1)))
                .expect("automatic value");

            view.reset_row_layout(1, schema_column_count);
            view.clear_owned_values();
            view.clear_automatic_values();

            assert_eq!(view.column_count(), schema_column_count);
            assert!(view.resolve("transient").expect("transient resolution").is_none());
            assert!(view.resolve("_N_").expect("automatic resolution").is_none());
            assert!(matches!(view.resolve("result").expect("reset result"),
                Some(ResolvedValue::Owned(OwnedValue::Scalar(ScalarValue::Null)))));
            assert!(matches!(view.resolve("id").expect("next source row"),
                Some(ResolvedValue::SourceScalar(SourceValueRef::Scalar(ScalarValue::Integer(2))))));
            view.set_owned_value("Transient", OwnedValue::null())
                .expect("next row can introduce its own transient column");
        });
    }

    #[test]
    fn empty_projection_preserves_bindings_and_freezes_visibility() {
        Python::attach(|py| {
            let cursor = source_cursor(py);
            let mut registry = ExecutionSourceRegistry::new();
            let source_id = registry.register(cursor);
            let mut view = LogicalRowView::new(
                0,
                Rc::new(registry),
                vec![
                    ("ID".to_string(), ColumnBinding::Source { source_id, slot: 0 }),
                    ("Nested".to_string(), ColumnBinding::Source { source_id, slot: 1 }),
                ],
            )
            .expect("logical row view");
            view.ensure_owned_column("Result").expect("declared result");
            view.write_owned("id", OwnedValue::from(ScalarValue::Integer(9)))
                .expect("source override");
            let before: Vec<_> = view.visible_columns().expect("visible columns")
                .iter()
                .map(|column| (column.logical_name().to_string(), column.source_field().cloned()))
                .collect();

            view.project_columns(&[], &[], &HashMap::new()).expect("empty projection");

            let after: Vec<_> = view.visible_columns().expect("visible columns")
                .iter()
                .map(|column| (column.logical_name().to_string(), column.source_field().cloned()))
                .collect();
            assert_eq!(after, before);
            assert!(matches!(view.resolve("ID").expect("source override"),
                Some(ResolvedValue::Owned(OwnedValue::Scalar(ScalarValue::Integer(9))))));
            assert!(matches!(view.resolve("nested").expect("nested source"),
                Some(ResolvedValue::SourceCell(_))));
            view.set_owned_value("result", OwnedValue::from(ScalarValue::Integer(7)))
                .expect("existing column remains writable");
            assert!(view.set_owned_value("new", OwnedValue::null()).is_err());
            assert!(view.ensure_owned_column("new").is_err());
        });
    }

    #[test]
    fn projection_rename_keeps_owned_source_override_visible_under_alias() {
        Python::attach(|py| {
            let cursor = source_cursor(py);
            let mut registry = ExecutionSourceRegistry::new();
            let source_id = registry.register(Rc::clone(&cursor));
            let registry = Rc::new(registry);
            let mut view = LogicalRowView::new(
                0,
                registry,
                vec![(
                    "Value".to_string(),
                    ColumnBinding::Source { source_id, slot: 0 },
                )],
            )
            .expect("logical row view");

            view.write_owned("value", OwnedValue::from(ScalarValue::Integer(9)))
                .expect("owned write");
            view.project_columns(
                &[],
                &[],
                &HashMap::from([("value".to_string(), "alias".to_string())]),
            )
            .expect("projection");

            match view.resolve("alias").expect("alias resolution") {
                Some(ResolvedValue::Owned(OwnedValue::Scalar(ScalarValue::Integer(value)))) => {
                    assert_eq!(*value, 9)
                }
                other => panic!("unexpected aliased value: {other:?}"),
            }
        });
    }

    #[test]
    fn rejects_dropped_or_colliding_names_without_reexposing_them() {
        let mut view = LogicalRowView::new(
            0,
            empty_registry(),
            vec![(
                "Kept".to_string(),
                ColumnBinding::Owned {
                    normalized_name: "KEPT".to_string(),
                },
            )],
        )
        .expect("logical row view");

        assert!(view
            .resolve("dropped")
            .expect("dropped resolution")
            .is_none());
        assert!(view.write_owned("dropped", OwnedValue::null()).is_err());

        let collision = LogicalRowView::new(
            0,
            empty_registry(),
            vec![
                (
                    "Name".to_string(),
                    ColumnBinding::Owned {
                        normalized_name: "NAME".to_string(),
                    },
                ),
                (
                    "name".to_string(),
                    ColumnBinding::Owned {
                        normalized_name: "OTHER".to_string(),
                    },
                ),
            ],
        );
        assert!(collision.is_err());
    }

    #[test]
    fn visible_columns_keep_order_and_resolve_source_alias_without_reparsing() {
        Python::attach(|py| {
            let cursor = source_cursor(py);
            let mut registry = ExecutionSourceRegistry::new();
            let source_id = registry.register(Rc::clone(&cursor));
            let registry = Rc::new(registry);

            let view = LogicalRowView::new(
                0,
                Rc::clone(&registry),
                vec![
                    (
                        "renamed_id".to_string(),
                        ColumnBinding::Source { source_id, slot: 0 },
                    ),
                    (
                        "nested_alias".to_string(),
                        ColumnBinding::Source { source_id, slot: 1 },
                    ),
                ],
            )
            .expect("logical row view");

            let columns = view.visible_columns().expect("visible columns");
            assert_eq!(
                columns
                    .iter()
                    .map(|column| column.logical_name())
                    .collect::<Vec<_>>(),
                vec!["renamed_id", "nested_alias"]
            );
            assert_eq!(
                columns[0].source_field().expect("source field").name(),
                "ID"
            );
            assert_eq!(
                columns[1].source_field().expect("source field").name(),
                "Nested"
            );

            match view.resolve("RENAMED_ID").expect("source resolution") {
                Some(ResolvedValue::SourceScalar(SourceValueRef::Scalar(
                    ScalarValue::Integer(value),
                ))) => assert_eq!(*value, 1),
                other => panic!("unexpected source value: {other:?}"),
            }
            match view.resolve("nested_alias").expect("nested resolution") {
                Some(ResolvedValue::SourceCell(cell)) => {
                    assert_eq!(cell.source_id(), source_id);
                    assert_eq!(cell.global_row_index(), 0);
                    assert_eq!(cell.slot_index(), 1);
                }
                other => panic!("unexpected nested value: {other:?}"),
            }
            assert!(view
                .resolve("ID")
                .expect("original alias resolution")
                .is_none());
        });
    }

    fn source_cursor(py: Python<'_>) -> Rc<NativeArrowRowCursor> {
        let pyarrow = py.import("pyarrow").expect("pyarrow");
        let id_values = PyList::new(py, [1_i64, 2_i64]).expect("id values");
        let nested_values = PyList::empty(py);
        for value in [10_i64, 20_i64] {
            nested_values
                .append(PyList::new(py, [value]).expect("nested value"))
                .expect("nested row");
        }
        let columns = PyDict::new(py);
        columns.set_item("ID", id_values).expect("id column");
        columns
            .set_item("Nested", nested_values)
            .expect("nested column");
        let table = pyarrow
            .getattr("table")
            .expect("table function")
            .call1((columns,))
            .expect("table");
        let stream = table
            .call_method0("__arrow_c_stream__")
            .expect("arrow stream");
        Rc::new(
            NativeArrowRowCursor::from_arrow_stream(py, &stream).expect("native arrow row cursor"),
        )
    }
}
