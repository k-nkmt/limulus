use crate::expressions::{column_key, ScalarValue};
use crate::source_registry::{ExecutionSourceRegistry, SourceId};
use crate::source_view::NativeSourceView;
use crate::value_ref::{OwnedValue, SourceCellRef, SourceValueRef};
use arrow::datatypes::DataType;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::{Rc, Weak};

pub(crate) struct NativeArrowRowCursor {
    slot_order: Vec<String>,
    row_count: usize,
    source_view: Rc<NativeSourceView>,
    source_id: Cell<Option<SourceId>>,
    source_registry: RefCell<
        Option<Weak<ExecutionSourceRegistry<NativeArrowRowCursor>>>,
    >,
}

impl NativeArrowRowCursor {
    pub(crate) fn from_arrow_stream(
        _py: Python<'_>,
        stream_obj: &Bound<'_, PyAny>,
    ) -> Result<Self, String> {
        NativeSourceView::from_arrow_stream(stream_obj).map(Self::from_source_view)
    }

    pub(crate) fn from_source_view(source_view: NativeSourceView) -> Self {
        let slot_order = source_view.slot_order().to_vec();
        let row_count = source_view.row_count();
        Self {
            slot_order,
            row_count,
            source_view: Rc::new(source_view),
            source_id: Cell::new(None),
            source_registry: RefCell::new(None),
        }
    }

    pub(crate) fn assign_source_id(&self, source_id: SourceId) -> Result<(), String> {
        if let Some(existing) = self.source_id.get() {
            if existing != source_id {
                return Err(format!(
                    "source cursor already belongs to source id {existing:?}"
                ));
            }
            return Ok(());
        }
        self.source_id.set(Some(source_id));
        Ok(())
    }

    pub(crate) fn assign_source_registry(
        &self,
        registry: Weak<ExecutionSourceRegistry<NativeArrowRowCursor>>,
    ) -> Result<(), String> {
        if registry.upgrade().is_none() {
            return Err("source registry was already released".to_string());
        }
        *self.source_registry.borrow_mut() = Some(registry);
        Ok(())
    }

    pub(crate) fn row_count(&self) -> usize {
        self.row_count
    }

    pub(crate) fn source_id(&self) -> Result<SourceId, String> {
        self.source_id
            .get()
            .ok_or_else(|| "source cursor is not registered in an execution registry".to_string())
    }

    pub(crate) fn slot_order(&self) -> &[String] {
        &self.slot_order
    }

    pub(crate) fn source_value_at_slot(
        &self,
        row_index: usize,
        slot_index: usize,
    ) -> Result<Option<SourceValueRef<'_>>, String> {
        if slot_index >= self.slot_order.len() {
            return Err(format!("source slot index out of range: {slot_index}"));
        }
        if row_index >= self.row_count {
            return Err(format!("row index out of range: {row_index}"));
        }
        let source_id = self
            .source_id
            .get()
            .ok_or_else(|| "source cursor is not registered in an execution registry".to_string())?;
        let array = self.source_view.array_at(row_index, slot_index)?;
        if is_source_backed_type(array.data_type()) {
            let registry = self
                .source_registry
                .borrow()
                .clone()
                .ok_or_else(|| "source cursor has no execution registry handle".to_string())?;
            return SourceCellRef::new(
                registry,
                source_id,
                row_index,
                slot_index,
            )
            .map(SourceValueRef::Cell)
            .map(Some);
        }
        self.source_view.value_at(row_index, slot_index)
    }

    pub(crate) fn scalar_at_slot(
        &self,
        row_index: usize,
        slot_index: usize,
    ) -> Result<Option<ScalarValue>, String> {
        self.source_value_at_slot(row_index, slot_index)
            .and_then(|value| {
                value
                    .map(|value_ref| value_ref.to_owned_value().and_then(OwnedValue::into_scalar_value))
                    .transpose()
            })
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

    pub(crate) fn validate_source_cell(
        &self,
        row_index: usize,
        slot_index: usize,
    ) -> Result<(), String> {
        self.source_view.validate_cell_location(row_index, slot_index)
    }

    pub(crate) fn array_at(
        &self,
        row_index: usize,
        slot_index: usize,
    ) -> Result<arrow::array::ArrayRef, String> {
        self.source_view.array_at(row_index, slot_index)
    }

    pub(crate) fn field_at(&self, slot_index: usize) -> Option<arrow::datatypes::FieldRef> {
        self.source_view.field_at(slot_index)
    }

    pub(crate) fn record_batch_at(&self, batch_index: usize) -> Option<arrow::record_batch::RecordBatch> {
        self.source_view.record_batch_at(batch_index)
    }

    pub(crate) fn batch_row_location(&self, row_index: usize) -> Result<(usize, usize), String> {
        self.source_view.batch_row_location(row_index)
    }
}

fn is_source_backed_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::List(_)
            | DataType::LargeList(_)
            | DataType::FixedSizeList(_, _)
            | DataType::Struct(_)
    )
}

#[cfg(test)]
mod tests {
    use super::NativeArrowRowCursor;
    use crate::source_registry::{ExecutionSourceRegistry, SourceId};
    use crate::source_view::NativeSourceView;
    use crate::value_ref::SourceValueRef;
    use arrow::array::{ArrayRef, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    fn nested_cursor() -> NativeArrowRowCursor {
        let nested_field = Field::new("item", DataType::Int64, true);
        let schema = Arc::new(Schema::new_with_metadata(
            vec![
                Field::new("id", DataType::Int64, false),
                Field::new("nested", DataType::List(Arc::new(nested_field)), true)
                    .with_metadata(
                        [("column-role".to_string(), "nested".to_string())]
                            .into_iter()
                            .collect(),
                    ),
            ],
            [("dataset-role".to_string(), "source-backed".to_string())]
                .into_iter()
                .collect(),
        ));
        let first = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(vec![Some(1)])) as ArrayRef,
                Arc::new(arrow::array::ListArray::from_iter_primitive::<
                    arrow::datatypes::Int64Type,
                    _,
                    _,
                >(vec![Some(vec![Some(10_i64)])])) as ArrayRef,
            ],
        )
        .expect("first batch");
        let second = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![Some(2)])) as ArrayRef,
                Arc::new(arrow::array::ListArray::from_iter_primitive::<
                    arrow::datatypes::Int64Type,
                    _,
                    _,
                >(vec![Option::<Vec<Option<i64>>>::None])) as ArrayRef,
            ],
        )
        .expect("second batch");
        let view = NativeSourceView::from_record_batches(vec![first, second])
            .expect("native source view");
        NativeArrowRowCursor::from_source_view(view)
    }

    fn registered_nested_cursor(
    ) -> (
        std::rc::Rc<NativeArrowRowCursor>,
        std::rc::Rc<ExecutionSourceRegistry<NativeArrowRowCursor>>,
    ) {
        let cursor = std::rc::Rc::new(nested_cursor());
        let mut registry = ExecutionSourceRegistry::new();
        let source_id = registry.register(std::rc::Rc::clone(&cursor));
        let registry = std::rc::Rc::new(registry);
        cursor
            .assign_source_id(source_id)
            .expect("assign source id");
        cursor
            .assign_source_registry(std::rc::Rc::downgrade(&registry))
            .expect("assign source registry");
        (cursor, registry)
    }

    fn registered_nested_family_cursor(
    ) -> (
        std::rc::Rc<NativeArrowRowCursor>,
        std::rc::Rc<ExecutionSourceRegistry<NativeArrowRowCursor>>,
    ) {
        let list = arrow::array::ListArray::from_iter_primitive::<
            arrow::datatypes::Int32Type,
            _,
            _,
        >(vec![Some(vec![Some(1_i32)])]);
        let large_list = arrow::array::LargeListArray::from_iter_primitive::<
            arrow::datatypes::Int32Type,
            _,
            _,
        >(vec![Some(vec![Some(2_i32)])]);
        let fixed_size_list = arrow::array::FixedSizeListArray::from_iter_primitive::<
            arrow::datatypes::Int32Type,
            _,
            _,
        >(vec![Some(vec![Some(3_i32)])], 1);
        let struct_array = arrow::array::StructArray::from(vec![
            (
                Arc::new(Field::new("value", DataType::Int32, true)),
                Arc::new(arrow::array::Int32Array::from(vec![Some(4_i32)])) as ArrayRef,
            ),
        ]);
        let batch = RecordBatch::try_from_iter(vec![
            ("list", Arc::new(list) as ArrayRef),
            ("large_list", Arc::new(large_list) as ArrayRef),
            ("fixed_size_list", Arc::new(fixed_size_list) as ArrayRef),
            ("struct", Arc::new(struct_array) as ArrayRef),
        ])
        .expect("nested family batch");
        let view = NativeSourceView::from_record_batches(vec![batch])
            .expect("native nested family view");
        let cursor = std::rc::Rc::new(NativeArrowRowCursor::from_source_view(view));
        let mut registry = ExecutionSourceRegistry::new();
        let source_id = registry.register(std::rc::Rc::clone(&cursor));
        let registry = std::rc::Rc::new(registry);
        cursor
            .assign_source_id(source_id)
            .expect("assign nested family source id");
        cursor
            .assign_source_registry(std::rc::Rc::downgrade(&registry))
            .expect("assign nested family registry");
        (cursor, registry)
    }

    #[test]
    fn nested_source_cell_resolves_batch_boundary_field_and_owner() {
        let (cursor, _registry) = registered_nested_cursor();

        let value = cursor
            .source_value_at_slot(1, 1)
            .expect("cell lookup")
            .expect("cell value");
        let SourceValueRef::Cell(cell) = value else {
            panic!("nested value must remain a source cell: {value:?}");
        };

        assert_eq!(cell.source_id(), SourceId::new(0));
        assert_eq!(cell.global_row_index(), 1);
        assert_eq!(cell.slot_index(), 1);
        assert_eq!(cell.batch_index().expect("batch index"), 1);
        assert_eq!(cell.batch_local_row().expect("batch row"), 0);
        assert_eq!(cell.field().expect("field").name(), "nested");
        assert_eq!(
            cell.field()
                .expect("field metadata")
                .metadata()
                .get("column-role")
                .map(String::as_str),
            Some("nested")
        );
        assert_eq!(cell.array().expect("array").data_type(), &DataType::List(
            Arc::new(Field::new("item", DataType::Int64, true))
        ));
        assert!(cell.is_null().expect("null check"));
        assert_eq!(cell.record_batch().expect("record batch").num_rows(), 1);
    }

    #[test]
    fn source_cell_rejects_invalid_row_and_slot_without_dangling() {
        let (cursor, registry) = registered_nested_cursor();

        let invalid_row = crate::value_ref::SourceCellRef::new(
            std::rc::Rc::downgrade(&registry),
            SourceId::new(0),
            2,
            1,
        )
        .expect_err("row after the final batch must fail");
        assert!(invalid_row.contains("row index out of range"));

        let invalid_slot = crate::value_ref::SourceCellRef::new(
            std::rc::Rc::downgrade(&registry),
            SourceId::new(0),
            0,
            2,
        )
        .expect_err("slot after the schema must fail");
        assert!(invalid_slot.contains("slot index out of range"));

        let first = cursor
            .source_value_at_slot(0, 1)
            .expect("first cell lookup")
            .expect("first cell");
        let second = cursor
            .source_value_at_slot(1, 1)
            .expect("second cell lookup")
            .expect("second cell");
        assert_ne!(first, second);
        assert!(first
            .clone()
            .as_scalar()
            .unwrap_err()
            .contains("nested source cell"));

        drop(registry);
        assert!(
            first
                .clone()
                .into_cell()
                .expect("cell variant")
                .array()
                .expect_err("registry release must invalidate the cell")
                .contains("registry lookup")
        );
    }

    #[test]
    fn every_nested_family_uses_registry_cell_and_invalid_source_id_fails_closed() {
        let (cursor, registry) = registered_nested_family_cursor();
        let expected_types = [
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            DataType::LargeList(Arc::new(Field::new("item", DataType::Int32, true))),
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int32, true)), 1),
            DataType::Struct(
                vec![Arc::new(Field::new("value", DataType::Int32, true))]
                    .into(),
            ),
        ];

        for (slot_index, (name, expected_type)) in
            ["list", "large_list", "fixed_size_list", "struct"]
                .into_iter()
                .zip(expected_types)
                .enumerate()
        {
            let value = cursor
                .source_value_at_slot(0, slot_index)
                .expect("nested family lookup")
                .expect("nested family value");
            let SourceValueRef::Cell(cell) = value else {
                panic!("{name} must remain a source cell: {value:?}");
            };
            assert_eq!(cell.source_id(), SourceId::new(0));
            assert_eq!(cell.field().expect("nested family field").name(), name);
            assert_eq!(cell.array().expect("nested family array").data_type(), &expected_type);
        }

        let invalid_source = crate::value_ref::SourceCellRef::new(
            std::rc::Rc::downgrade(&registry),
            SourceId::new(99),
            0,
            0,
        )
        .expect_err("unknown source id must fail closed");
        assert!(invalid_source.contains("invalid source id"));
    }
}
