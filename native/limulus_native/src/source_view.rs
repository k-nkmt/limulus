use crate::expressions::{column_key, ScalarValue};
use crate::value_ref::SourceValueRef;
use arrow::array::{
    Array,
    BooleanArray,
    Date32Array,
    Date64Array,
    Decimal128Array,
    FixedSizeListArray,
    Float32Array,
    Float64Array,
    Int16Array,
    Int32Array,
    Int64Array,
    Int8Array,
    LargeListArray,
    LargeStringArray,
    ListArray,
    StringArray,
    StructArray,
    Time32MillisecondArray,
    Time32SecondArray,
    Time64MicrosecondArray,
    Time64NanosecondArray,
    TimestampMicrosecondArray,
    TimestampMillisecondArray,
    TimestampNanosecondArray,
    TimestampSecondArray,
    UInt16Array,
    UInt32Array,
    UInt64Array,
    UInt8Array,
};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Schema, TimeUnit};
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow::record_batch::{RecordBatch, RecordBatchReader};
use arrow::temporal_conversions::{
    date32_to_datetime,
    date64_to_datetime,
    time32ms_to_time,
    time32s_to_time,
    time64ns_to_time,
    time64us_to_time,
    timestamp_ms_to_datetime,
    timestamp_ns_to_datetime,
    timestamp_s_to_datetime,
    timestamp_us_to_datetime,
};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyCapsule, PyCapsuleMethods};
use std::cell::OnceCell;
use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::Arc;

pub(crate) struct NativeSourceView {
    schema: Arc<Schema>,
    batches: Vec<RecordBatch>,
    batch_row_offsets: Vec<usize>,
    row_count: usize,
    slot_order: Vec<String>,
    #[cfg_attr(not(test), allow(dead_code))]
    normalized_slot_index: HashMap<String, usize>,
    decoded_slot_batches: Vec<Vec<OnceCell<Result<Vec<ScalarValue>, String>>>>,
}

impl NativeSourceView {
    pub(crate) fn from_arrow_stream(stream_obj: &Bound<'_, PyAny>) -> Result<Self, String> {
        let mut reader = import_arrow_stream_reader(stream_obj)?;
        let schema = reader.schema();
        let mut batches = Vec::new();

        for batch_result in &mut reader {
            batches.push(
                batch_result
                    .map_err(|error| format!("failed to read next record batch: {error}"))?,
            );
        }

        Self::from_record_batches_with_schema(schema, batches)
    }

    fn from_record_batches_with_schema(
        schema: Arc<Schema>,
        batches: Vec<RecordBatch>,
    ) -> Result<Self, String> {
        let batches: Vec<RecordBatch> = batches
            .into_iter()
            .filter(|batch| batch.num_rows() > 0)
            .collect();

        let mut slot_order: Vec<String> = schema
            .fields()
            .iter()
            .map(|field| field.name().to_string())
            .collect();
        if slot_order.is_empty() {
            if let Some(batch) = batches.first() {
                slot_order = schema_field_names_from_batch(batch);
            }
        }

        let mut batch_row_offsets = Vec::with_capacity(batches.len());
        let mut row_count = 0;
        for batch in &batches {
            batch_row_offsets.push(row_count);
            row_count += batch.num_rows();
        }

        let slot_count = slot_order.len();
        let decoded_slot_batches = batches
            .iter()
            .map(|_| {
                (0..slot_count)
                    .map(|_| OnceCell::<Result<Vec<ScalarValue>, String>>::new())
                    .collect::<Vec<_>>()
            })
            .collect();

        let mut normalized_slot_index = HashMap::new();
        for (slot_index, slot_name) in slot_order.iter().enumerate() {
            normalized_slot_index.insert(column_key(slot_name), slot_index);
        }

        Ok(Self {
            schema,
            batches,
            batch_row_offsets,
            row_count,
            slot_order,
            normalized_slot_index,
            decoded_slot_batches,
        })
    }

    #[cfg(test)]
    pub(crate) fn from_record_batches(batches: Vec<RecordBatch>) -> Result<Self, String> {
        let schema = batches
            .first()
            .map(RecordBatch::schema)
            .unwrap_or_else(|| Arc::new(Schema::empty()));
        Self::from_record_batches_with_schema(schema, batches)
    }

    pub(crate) fn row_count(&self) -> usize {
        self.row_count
    }

    pub(crate) fn slot_order(&self) -> &[String] {
        let _ = &self.schema;
        &self.slot_order
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn value_by_name(
        &self,
        row_index: usize,
        name: &str,
    ) -> Result<Option<SourceValueRef<'_>>, String> {
        let normalized = column_key(name);
        self.value_by_normalized_name(row_index, &normalized)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn value_by_normalized_name(
        &self,
        row_index: usize,
        normalized_name: &str,
    ) -> Result<Option<SourceValueRef<'_>>, String> {
        let Some(slot_index) = self.normalized_slot_index.get(normalized_name).copied() else {
            return Ok(None);
        };
        self.value_at(row_index, slot_index)
    }

    pub(crate) fn value_at(
        &self,
        row_index: usize,
        slot_index: usize,
    ) -> Result<Option<SourceValueRef<'_>>, String> {
        if slot_index >= self.slot_order.len() {
            return Ok(None);
        }
        let (batch_index, batch_row_index) = self.batch_row_location(row_index)?;
        let batch = self
            .batches
            .get(batch_index)
            .ok_or_else(|| format!("batch index out of range: {batch_index}"))?;
        let decoded_slot_values = self
            .decoded_slot_batches
            .get(batch_index)
            .and_then(|batch_cells| batch_cells.get(slot_index))
            .ok_or_else(|| format!("slot index out of range: {slot_index}"))?
            .get_or_init(|| {
                let array = batch
                    .columns()
                    .get(slot_index)
                    .ok_or_else(|| format!("slot index out of range: {slot_index}"))?;
                decode_slot_values_from_array(array.as_ref())
            })
            .as_ref()
            .map_err(|error| error.clone())?;
        let value = decoded_slot_values
            .get(batch_row_index)
            .ok_or_else(|| format!("row index out of range: {row_index}"))?;
        Ok(Some(SourceValueRef::Scalar(value)))
    }

    fn batch_row_location(&self, row_index: usize) -> Result<(usize, usize), String> {
        if row_index >= self.row_count {
            return Err(format!("row index out of range: {row_index}"));
        }

        let batch_index = self
            .batch_row_offsets
            .partition_point(|offset| *offset <= row_index)
            .saturating_sub(1);
        let batch_row_offset = self
            .batch_row_offsets
            .get(batch_index)
            .copied()
            .ok_or_else(|| format!("batch index out of range: {batch_index}"))?;
        Ok((batch_index, row_index - batch_row_offset))
    }

    #[cfg(test)]
    fn decoded_slot_batch_count(&self) -> usize {
        self.decoded_slot_batches
            .iter()
            .flat_map(|batch_cells| batch_cells.iter())
            .filter(|cell| cell.get().is_some())
            .count()
    }
}

fn import_arrow_stream_reader(stream_obj: &Bound<'_, PyAny>) -> Result<ArrowArrayStreamReader, String> {
    let capsule = stream_obj
        .cast::<PyCapsule>()
        .map_err(|error| format!("Arrow C stream payload is not a capsule: {error}"))?;
    let capsule_name = arrow_stream_capsule_name();
    let pointer = capsule
        .pointer_checked(Some(capsule_name))
        .map_err(|error| format!("failed to access Arrow C stream capsule: {error}"))?;
    unsafe { ArrowArrayStreamReader::from_raw(pointer.cast::<FFI_ArrowArrayStream>().as_ptr()) }
        .map_err(|error| format!("failed to import Arrow C stream reader: {error}"))
}

fn arrow_stream_capsule_name() -> &'static CStr {
    unsafe { CStr::from_bytes_with_nul_unchecked(b"arrow_array_stream\0") }
}

fn schema_field_names_from_batch(batch: &RecordBatch) -> Vec<String> {
    batch
        .schema()
        .fields()
        .iter()
        .map(|field| field.name().to_string())
        .collect()
}

fn decode_slot_values_from_array(array: &dyn Array) -> Result<Vec<ScalarValue>, String> {
    let mut slot_values = Vec::with_capacity(array.len());
    extend_slot_values_from_array(&mut slot_values, array)?;
    Ok(slot_values)
}

fn extend_slot_values_from_array(slot_values: &mut Vec<ScalarValue>, array: &dyn Array) -> Result<(), String> {
    match array.data_type() {
        DataType::Null => {
            for _ in 0..array.len() {
                slot_values.push(ScalarValue::Null);
            }
            Ok(())
        }
        DataType::Float64 => extend_nullable_values(slot_values, downcast_array::<Float64Array>(array, "Float64Array")?, |values, row| {
            Ok(ScalarValue::Number(values.value(row)))
        }),
        DataType::Float32 => extend_nullable_values(slot_values, downcast_array::<Float32Array>(array, "Float32Array")?, |values, row| {
            Ok(ScalarValue::Number(values.value(row) as f64))
        }),
        DataType::Int64 => extend_nullable_values(slot_values, downcast_array::<Int64Array>(array, "Int64Array")?, |values, row| {
            Ok(ScalarValue::Number(values.value(row) as f64))
        }),
        DataType::Int32 => extend_nullable_values(slot_values, downcast_array::<Int32Array>(array, "Int32Array")?, |values, row| {
            Ok(ScalarValue::Number(values.value(row) as f64))
        }),
        DataType::Int16 => extend_nullable_values(slot_values, downcast_array::<Int16Array>(array, "Int16Array")?, |values, row| {
            Ok(ScalarValue::Number(values.value(row) as f64))
        }),
        DataType::Int8 => extend_nullable_values(slot_values, downcast_array::<Int8Array>(array, "Int8Array")?, |values, row| {
            Ok(ScalarValue::Number(values.value(row) as f64))
        }),
        DataType::UInt64 => extend_nullable_values(slot_values, downcast_array::<UInt64Array>(array, "UInt64Array")?, |values, row| {
            Ok(ScalarValue::Number(values.value(row) as f64))
        }),
        DataType::UInt32 => extend_nullable_values(slot_values, downcast_array::<UInt32Array>(array, "UInt32Array")?, |values, row| {
            Ok(ScalarValue::Number(values.value(row) as f64))
        }),
        DataType::UInt16 => extend_nullable_values(slot_values, downcast_array::<UInt16Array>(array, "UInt16Array")?, |values, row| {
            Ok(ScalarValue::Number(values.value(row) as f64))
        }),
        DataType::UInt8 => extend_nullable_values(slot_values, downcast_array::<UInt8Array>(array, "UInt8Array")?, |values, row| {
            Ok(ScalarValue::Number(values.value(row) as f64))
        }),
        DataType::Decimal128(precision, scale) => extend_nullable_values(
            slot_values,
            downcast_array::<Decimal128Array>(array, "Decimal128Array")?,
            |values, row| decimal_scalar(values.value(row), *precision, *scale),
        ),
        DataType::Boolean => extend_nullable_values(slot_values, downcast_array::<BooleanArray>(array, "BooleanArray")?, |values, row| {
            Ok(ScalarValue::Bool(values.value(row)))
        }),
        DataType::Utf8 => extend_nullable_values(slot_values, downcast_array::<StringArray>(array, "StringArray")?, |values, row| {
            Ok(ScalarValue::Text(values.value(row).to_string()))
        }),
        DataType::LargeUtf8 => extend_nullable_values(slot_values, downcast_array::<LargeStringArray>(array, "LargeStringArray")?, |values, row| {
            Ok(ScalarValue::Text(values.value(row).to_string()))
        }),
        DataType::List(_) => extend_nullable_values(slot_values, downcast_array::<ListArray>(array, "ListArray")?, |values, row| {
            list_scalar_from_array(values.value(row).as_ref())
        }),
        DataType::LargeList(_) => extend_nullable_values(
            slot_values,
            downcast_array::<LargeListArray>(array, "LargeListArray")?,
            |values, row| list_scalar_from_array(values.value(row).as_ref()),
        ),
        DataType::FixedSizeList(_, _) => extend_nullable_values(
            slot_values,
            downcast_array::<FixedSizeListArray>(array, "FixedSizeListArray")?,
            |values, row| list_scalar_from_array(values.value(row).as_ref()),
        ),
        DataType::Struct(_) => extend_nullable_values(slot_values, downcast_array::<StructArray>(array, "StructArray")?, |values, row| {
            struct_scalar_from_array(values, row)
        }),
        DataType::Date32 => extend_nullable_values(slot_values, downcast_array::<Date32Array>(array, "Date32Array")?, |values, row| {
            let datetime = date32_to_datetime(values.value(row))
                .ok_or_else(|| format!("failed to decode date32 value at row {row}"))?;
            Ok(ScalarValue::Date(datetime.date()))
        }),
        DataType::Date64 => extend_nullable_values(slot_values, downcast_array::<Date64Array>(array, "Date64Array")?, |values, row| {
            let datetime = date64_to_datetime(values.value(row))
                .ok_or_else(|| format!("failed to decode date64 value at row {row}"))?;
            Ok(ScalarValue::Date(datetime.date()))
        }),
        DataType::Timestamp(TimeUnit::Second, _) => extend_nullable_values(
            slot_values,
            downcast_array::<TimestampSecondArray>(array, "TimestampSecondArray")?,
            |values, row| temporal_datetime_scalar(timestamp_s_to_datetime(values.value(row)), row, "timestamp[s]"),
        ),
        DataType::Timestamp(TimeUnit::Millisecond, _) => extend_nullable_values(
            slot_values,
            downcast_array::<TimestampMillisecondArray>(array, "TimestampMillisecondArray")?,
            |values, row| temporal_datetime_scalar(timestamp_ms_to_datetime(values.value(row)), row, "timestamp[ms]"),
        ),
        DataType::Timestamp(TimeUnit::Microsecond, _) => extend_nullable_values(
            slot_values,
            downcast_array::<TimestampMicrosecondArray>(array, "TimestampMicrosecondArray")?,
            |values, row| temporal_datetime_scalar(timestamp_us_to_datetime(values.value(row)), row, "timestamp[us]"),
        ),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => extend_nullable_values(
            slot_values,
            downcast_array::<TimestampNanosecondArray>(array, "TimestampNanosecondArray")?,
            |values, row| temporal_datetime_scalar(timestamp_ns_to_datetime(values.value(row)), row, "timestamp[ns]"),
        ),
        DataType::Time32(TimeUnit::Second) => extend_nullable_values(
            slot_values,
            downcast_array::<Time32SecondArray>(array, "Time32SecondArray")?,
            |values, row| temporal_time_scalar(time32s_to_time(values.value(row)), row, "time32[s]"),
        ),
        DataType::Time32(TimeUnit::Millisecond) => extend_nullable_values(
            slot_values,
            downcast_array::<Time32MillisecondArray>(array, "Time32MillisecondArray")?,
            |values, row| temporal_time_scalar(time32ms_to_time(values.value(row)), row, "time32[ms]"),
        ),
        DataType::Time64(TimeUnit::Microsecond) => extend_nullable_values(
            slot_values,
            downcast_array::<Time64MicrosecondArray>(array, "Time64MicrosecondArray")?,
            |values, row| temporal_time_scalar(time64us_to_time(values.value(row)), row, "time64[us]"),
        ),
        DataType::Time64(TimeUnit::Nanosecond) => extend_nullable_values(
            slot_values,
            downcast_array::<Time64NanosecondArray>(array, "Time64NanosecondArray")?,
            |values, row| temporal_time_scalar(time64ns_to_time(values.value(row)), row, "time64[ns]"),
        ),
        DataType::Dictionary(_, value_type) => {
            let decoded = cast(array, value_type.as_ref())
                .map_err(|error| format!("failed to decode dictionary Arrow source column: {error}"))?;
            extend_slot_values_from_array(slot_values, decoded.as_ref())
        }
        data_type => Err(format!("unsupported Arrow source column type for native cursor: {data_type:?}")),
    }
}

fn downcast_array<'a, T: 'static>(array: &'a dyn Array, expected: &str) -> Result<&'a T, String> {
    array
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| format!("failed to downcast Arrow column to {expected}"))
}

fn extend_nullable_values<T, F>(slot_values: &mut Vec<ScalarValue>, values: &T, mut value_at: F) -> Result<(), String>
where
    T: Array,
    F: FnMut(&T, usize) -> Result<ScalarValue, String>,
{
    for row_index in 0..values.len() {
        if values.is_null(row_index) {
            slot_values.push(ScalarValue::Null);
            continue;
        }
        slot_values.push(value_at(values, row_index)?);
    }
    Ok(())
}

fn decimal_scalar(value: i128, precision: u8, scale: i8) -> Result<ScalarValue, String> {
    if scale < 0 {
        return Err(format!(
            "unsupported Arrow source column type for native cursor: Decimal128({precision}, {scale})"
        ));
    }
    Ok(ScalarValue::Decimal(decimal128_to_string(value, scale as usize)))
}

fn decimal128_to_string(value: i128, scale: usize) -> String {
    let negative = value < 0;
    let digits = value.abs().to_string();
    let rendered = if scale == 0 {
        digits
    } else if digits.len() <= scale {
        format!("0.{}{}", "0".repeat(scale - digits.len()), digits)
    } else {
        let split = digits.len() - scale;
        format!("{}.{}", &digits[..split], &digits[split..])
    };
    if negative {
        format!("-{rendered}")
    } else {
        rendered
    }
}

fn list_scalar_from_array(array: &dyn Array) -> Result<ScalarValue, String> {
    let mut values = Vec::with_capacity(array.len());
    for row_index in 0..array.len() {
        values.push(scalar_value_from_array_row(array, row_index)?);
    }
    Ok(ScalarValue::List(values))
}

fn struct_scalar_from_array(values: &StructArray, row_index: usize) -> Result<ScalarValue, String> {
    let DataType::Struct(fields) = values.data_type() else {
        return Err("failed to resolve struct Arrow source column metadata".to_string());
    };

    let mut items = Vec::with_capacity(fields.len());
    for (field_index, field) in fields.iter().enumerate() {
        items.push((
            field.name().to_string(),
            scalar_value_from_array_row(values.column(field_index).as_ref(), row_index)?,
        ));
    }
    Ok(ScalarValue::Struct(items))
}

fn scalar_value_from_array_row(array: &dyn Array, row_index: usize) -> Result<ScalarValue, String> {
    if row_index >= array.len() {
        return Err(format!("Arrow row index out of bounds: {row_index}"));
    }
    if array.is_null(row_index) {
        return Ok(ScalarValue::Null);
    }

    let slice = array.slice(row_index, 1);
    let mut values = Vec::with_capacity(1);
    extend_slot_values_from_array(&mut values, slice.as_ref())?;
    values
        .pop()
        .ok_or_else(|| format!("failed to decode Arrow value at row {row_index}"))
}

fn temporal_datetime_scalar(
    value: Option<chrono::NaiveDateTime>,
    row_index: usize,
    type_name: &str,
) -> Result<ScalarValue, String> {
    let datetime = value.ok_or_else(|| format!("failed to decode {type_name} value at row {row_index}"))?;
    Ok(ScalarValue::DateTime(datetime))
}

fn temporal_time_scalar(
    value: Option<chrono::NaiveTime>,
    row_index: usize,
    type_name: &str,
) -> Result<ScalarValue, String> {
    let time = value.ok_or_else(|| format!("failed to decode {type_name} value at row {row_index}"))?;
    Ok(ScalarValue::Time(time))
}

#[cfg(test)]
mod tests {
    use super::NativeSourceView;
    use crate::expressions::ScalarValue;
    use crate::value_ref::SourceValueRef;
    use arrow::array::{
        new_null_array,
        ArrayRef,
        BooleanArray,
        Float64Array,
        Int64Array,
        ListArray,
        StringArray,
        StringDictionaryBuilder,
        StructArray,
    };
    use arrow::datatypes::{DataType, Field, Fields, Int64Type, Int8Type, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    #[test]
    fn native_source_view_reads_rows_across_batch_boundaries() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Float64, true),
            Field::new("name", DataType::Utf8, true),
        ]));
        let batch_one = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Float64Array::from(vec![Some(1.0), Some(2.0)])),
                Arc::new(StringArray::from(vec![Some("alpha"), None])),
            ],
        )
        .expect("batch one");
        let batch_two = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![Some(3.0)])),
                Arc::new(StringArray::from(vec![Some("gamma")])),
            ],
        )
        .expect("batch two");

        let view = NativeSourceView::from_record_batches(vec![batch_one, batch_two]).expect("source view");

        assert_eq!(view.row_count(), 3);
        assert_eq!(view.slot_order(), &["id".to_string(), "name".to_string()]);

        match view.value_at(2, 0).expect("row access").expect("value") {
            SourceValueRef::Scalar(ScalarValue::Number(value)) => assert_eq!(*value, 3.0),
            other => panic!("unexpected value ref: {other:?}"),
        }
        match view.value_by_name(1, "name").expect("name access").expect("value") {
            SourceValueRef::Scalar(ScalarValue::Null) => {}
            other => panic!("unexpected name ref: {other:?}"),
        }
    }

    #[test]
    fn native_source_view_decodes_only_requested_batch_slots_lazily() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Float64, true),
            Field::new("name", DataType::Utf8, true),
        ]));
        let batch_one = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Float64Array::from(vec![Some(1.0), Some(2.0)])),
                Arc::new(StringArray::from(vec![Some("alpha"), Some("beta")])),
            ],
        )
        .expect("batch one");
        let batch_two = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![Some(3.0)])),
                Arc::new(StringArray::from(vec![Some("gamma")])),
            ],
        )
        .expect("batch two");

        let view = NativeSourceView::from_record_batches(vec![batch_one, batch_two]).expect("source view");

        assert_eq!(view.decoded_slot_batch_count(), 0);
        view.value_at(0, 0).expect("first batch id");
        assert_eq!(view.decoded_slot_batch_count(), 1);
        view.value_at(1, 0).expect("same slot same batch");
        assert_eq!(view.decoded_slot_batch_count(), 1);
        view.value_at(2, 0).expect("same slot next batch");
        assert_eq!(view.decoded_slot_batch_count(), 2);
        view.value_at(2, 1).expect("next slot next batch");
        assert_eq!(view.decoded_slot_batch_count(), 3);
    }

    #[test]
    fn native_source_view_reads_supported_success_path_types_by_name() {
        let tags = ListArray::from_iter_primitive::<Int64Type, _, _>(vec![
            Some(vec![Some(1_i64), Some(2_i64)]),
            None,
        ]);
        let meta = StructArray::from(vec![
            (
                Arc::new(Field::new("score", DataType::Int64, true)),
                Arc::new(Int64Array::from(vec![Some(10_i64), None])) as ArrayRef,
            ),
            (
                Arc::new(Field::new("flag", DataType::Boolean, true)),
                Arc::new(BooleanArray::from(vec![Some(true), Some(false)])) as ArrayRef,
            ),
        ]);
        let mut category_builder = StringDictionaryBuilder::<Int8Type>::new();
        category_builder.append("alpha").expect("append first dictionary value");
        category_builder.append_null();
        let category = category_builder.finish();
        let batch = RecordBatch::try_from_iter(vec![
            ("Id", Arc::new(Float64Array::from(vec![Some(1.0), Some(2.0)])) as ArrayRef),
            ("Tags", Arc::new(tags) as ArrayRef),
            ("Meta", Arc::new(meta) as ArrayRef),
            ("Category", Arc::new(category) as ArrayRef),
        ])
        .expect("record batch");

        let view = NativeSourceView::from_record_batches(vec![batch]).expect("source view");

        match view.value_by_name(0, "tags").expect("tags access").expect("tags value") {
            SourceValueRef::Scalar(ScalarValue::List(values)) => {
                assert_eq!(values, &vec![ScalarValue::Number(1.0), ScalarValue::Number(2.0)]);
            }
            other => panic!("unexpected tags value ref: {other:?}"),
        }

        match view.value_by_name(0, "META").expect("meta access").expect("meta value") {
            SourceValueRef::Scalar(ScalarValue::Struct(values)) => {
                assert_eq!(
                    values,
                    &vec![
                        ("score".to_string(), ScalarValue::Number(10.0)),
                        ("flag".to_string(), ScalarValue::Bool(true)),
                    ]
                );
            }
            other => panic!("unexpected meta value ref: {other:?}"),
        }

        match view
            .value_by_name(0, "category")
            .expect("dictionary access")
            .expect("dictionary value")
        {
            SourceValueRef::Scalar(ScalarValue::Text(value)) => assert_eq!(value, "alpha"),
            other => panic!("unexpected dictionary value ref: {other:?}"),
        }

        match view
            .value_by_name(1, "CATEGORY")
            .expect("dictionary null access")
            .expect("dictionary null value")
        {
            SourceValueRef::Scalar(ScalarValue::Null) => {}
            other => panic!("unexpected dictionary null value ref: {other:?}"),
        }
    }

    #[test]
    fn native_source_view_reads_decimal128_values_on_standard_success_path() {
        let batch = RecordBatch::try_from_iter(vec![(
            "price",
            Arc::new(arrow::array::Decimal128Array::from(vec![1234_i128, 950_i128])
                .with_precision_and_scale(10, 2)
                .expect("decimal array")) as ArrayRef,
        )])
        .expect("decimal batch");

        let view = NativeSourceView::from_record_batches(vec![batch]).expect("source view");

        match view.value_by_name(0, "price").expect("decimal access").expect("decimal value") {
            SourceValueRef::Scalar(ScalarValue::Decimal(value)) => assert_eq!(value, "12.34"),
            other => panic!("unexpected decimal value ref: {other:?}"),
        }
        match view.value_by_name(1, "PRICE").expect("decimal access").expect("decimal value") {
            SourceValueRef::Scalar(ScalarValue::Decimal(value)) => assert_eq!(value, "9.50"),
            other => panic!("unexpected decimal value ref: {other:?}"),
        }
    }

    #[test]
    fn native_source_view_returns_explicit_unsupported_errors_for_binary_decimal_and_map() {
        let map_entries = Arc::new(Field::new(
            "entries",
            DataType::Struct(Fields::from(vec![
                Field::new("keys", DataType::Utf8, false),
                Field::new("values", DataType::Int64, true),
            ])),
            false,
        ));
        let cases = [
            ("binary", DataType::Binary),
            ("map", DataType::Map(map_entries, false)),
        ];

        for (label, data_type) in cases {
            let rendered_type = format!("{data_type:?}");
            let batch = RecordBatch::try_from_iter(vec![(
                "value",
                new_null_array(&data_type, 1),
            )])
            .expect("unsupported-type batch");
            let view = NativeSourceView::from_record_batches(vec![batch]).expect("source view");

            let error = view.value_at(0, 0).expect_err("unsupported type should return an explicit error");
            assert!(
                error.contains("unsupported Arrow source column type for native cursor"),
                "{label}: unexpected error message {error}"
            );
            assert!(
                error.contains(&rendered_type),
                "{label}: error should include rendered type {rendered_type}, got {error}"
            );
        }
    }

    #[test]
    fn native_source_view_reads_nullable_general_case_for_supported_types() {
        let mut category_builder = StringDictionaryBuilder::<Int8Type>::new();
        category_builder.append_null();
        category_builder.append("beta").expect("append dictionary value");
        let category = category_builder.finish();
        let batch = RecordBatch::try_from_iter(vec![
            (
                "Amount",
                Arc::new(Float64Array::from(vec![None, Some(2.5)])) as ArrayRef,
            ),
            (
                "Name",
                Arc::new(StringArray::from(vec![None, Some("beta")])) as ArrayRef,
            ),
            (
                "Tags",
                Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>(vec![
                    None,
                    Some(vec![Some(3_i64), None]),
                ])) as ArrayRef,
            ),
            ("Category", Arc::new(category) as ArrayRef),
        ])
        .expect("nullable-general-case batch");

        let view = NativeSourceView::from_record_batches(vec![batch]).expect("source view");

        match view.value_by_name(0, "amount").expect("amount access").expect("amount value") {
            SourceValueRef::Scalar(ScalarValue::Null) => {}
            other => panic!("unexpected nullable amount value ref: {other:?}"),
        }
        match view.value_by_name(0, "NAME").expect("name access").expect("name value") {
            SourceValueRef::Scalar(ScalarValue::Null) => {}
            other => panic!("unexpected nullable name value ref: {other:?}"),
        }
        match view.value_by_name(0, "category").expect("category access").expect("category value") {
            SourceValueRef::Scalar(ScalarValue::Null) => {}
            other => panic!("unexpected nullable category value ref: {other:?}"),
        }

        match view.value_by_name(1, "tags").expect("tags access").expect("tags value") {
            SourceValueRef::Scalar(ScalarValue::List(values)) => {
                assert_eq!(values, &vec![ScalarValue::Number(3.0), ScalarValue::Null]);
            }
            other => panic!("unexpected nullable tags value ref: {other:?}"),
        }
        match view.value_by_name(1, "CATEGORY").expect("category value access").expect("category row value") {
            SourceValueRef::Scalar(ScalarValue::Text(value)) => assert_eq!(value, "beta"),
            other => panic!("unexpected category row value ref: {other:?}"),
        }
    }
}
