use crate::expressions::ScalarValue;
use crate::source_registry::SourceId;
use arrow::array::ArrayRef;
use arrow::datatypes::TimeUnit;
use arrow::datatypes::Field;
use arrow::record_batch::RecordBatch;
use std::fmt;
use std::rc::{Rc, Weak};

#[derive(Clone)]
pub(crate) struct SourceCellRef {
    source_id: SourceId,
    global_row_index: usize,
    slot_index: usize,
    registry: Weak<
        crate::source_registry::ExecutionSourceRegistry<crate::row_cursor::NativeArrowRowCursor>,
    >,
}

impl fmt::Debug for SourceCellRef {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SourceCellRef")
            .field("source_id", &self.source_id)
            .field("global_row_index", &self.global_row_index)
            .field("slot_index", &self.slot_index)
            .finish_non_exhaustive()
    }
}

impl PartialEq for SourceCellRef {
    fn eq(&self, other: &Self) -> bool {
        self.source_id == other.source_id
            && self.global_row_index == other.global_row_index
            && self.slot_index == other.slot_index
    }
}

impl Eq for SourceCellRef {}

impl SourceCellRef {
    pub(crate) fn new(
        registry: Weak<
            crate::source_registry::ExecutionSourceRegistry<crate::row_cursor::NativeArrowRowCursor>,
        >,
        source_id: SourceId,
        global_row_index: usize,
        slot_index: usize,
    ) -> Result<Self, String> {
        let registry_owner = registry
            .upgrade()
            .ok_or_else(|| "source registry was released before cell creation".to_string())?;
        let cursor = registry_owner
            .lookup(source_id)
            .map_err(|error| error.to_string())?
            .cursor();
        cursor.validate_source_cell(global_row_index, slot_index)?;
        Ok(Self {
            source_id,
            global_row_index,
            slot_index,
            registry,
        })
    }

    pub(crate) fn source_id(&self) -> SourceId {
        self.source_id
    }

    pub(crate) fn global_row_index(&self) -> usize {
        self.global_row_index
    }

    pub(crate) fn slot_index(&self) -> usize {
        self.slot_index
    }

    pub(crate) fn batch_index(&self) -> Result<usize, String> {
        self.cursor()?
            .batch_row_location(self.global_row_index)
            .map(|(batch_index, _)| batch_index)
    }

    pub(crate) fn batch_local_row(&self) -> Result<usize, String> {
        self.cursor()?
            .batch_row_location(self.global_row_index)
            .map(|(_, row_index)| row_index)
    }

    pub(crate) fn field(&self) -> Result<std::sync::Arc<Field>, String> {
        self.cursor()?
            .field_at(self.slot_index)
            .ok_or_else(|| format!("source slot index out of range: {}", self.slot_index))
    }

    pub(crate) fn record_batch(&self) -> Result<RecordBatch, String> {
        let batch_index = self.batch_index()?;
        self.cursor()?
            .record_batch_at(batch_index)
            .ok_or_else(|| format!("source batch index out of range: {batch_index}"))
    }

    pub(crate) fn array(&self) -> Result<ArrayRef, String> {
        self.cursor()?
            .array_at(self.global_row_index, self.slot_index)
    }

    pub(crate) fn is_null(&self) -> Result<bool, String> {
        let array = self.array()?;
        Ok(array.is_null(self.batch_local_row()?))
    }

    fn cursor(&self) -> Result<Rc<crate::row_cursor::NativeArrowRowCursor>, String> {
        let registry = self
            .registry
            .upgrade()
            .ok_or_else(|| self.access_error("registry lookup"))?;
        registry
            .lookup(self.source_id)
            .map_err(|error| format!("{}: {error}", self.access_error("registry lookup")))
            .map(|handle| handle.cursor())
    }

    pub(crate) fn access_error(&self, operation: &str) -> String {
        format!(
            "nested source cell access is unsupported for {operation} (source_id={:?}, row={}, slot={})",
            self.source_id, self.global_row_index, self.slot_index
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum OwnedValue {
    Scalar(ScalarValue),
    SourceCell(SourceCellRef),
}

impl OwnedValue {
    pub(crate) fn null() -> Self {
        Self::Scalar(ScalarValue::Null)
    }

    pub(crate) fn as_scalar(&self) -> Result<&ScalarValue, String> {
        match self {
            Self::Scalar(value) => Ok(value),
            Self::SourceCell(cell) => Err(cell.access_error("scalar conversion")),
        }
    }

    pub(crate) fn as_integer(&self) -> Option<i64> {
        self.as_scalar().ok().and_then(ScalarValue::as_integer)
    }

    pub(crate) fn as_unsigned(&self) -> Option<u64> {
        self.as_scalar().ok().and_then(ScalarValue::as_unsigned)
    }

    pub(crate) fn as_raw_timestamp(&self) -> Option<(i64, TimeUnit, Option<&str>)> {
        self.as_scalar().ok().and_then(ScalarValue::as_raw_timestamp)
    }

    pub(crate) fn as_raw_time(&self) -> Option<(i64, TimeUnit)> {
        self.as_scalar().ok().and_then(ScalarValue::as_raw_time)
    }

    pub(crate) fn into_scalar_value(self) -> Result<ScalarValue, String> {
        match self {
            Self::Scalar(value) => Ok(value),
            Self::SourceCell(cell) => Err(cell.access_error("scalar conversion")),
        }
    }
}

impl From<ScalarValue> for OwnedValue {
    fn from(value: ScalarValue) -> Self {
        Self::Scalar(value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SourceValueRef<'a> {
    Scalar(&'a ScalarValue),
    Cell(SourceCellRef),
}

impl<'a> SourceValueRef<'a> {
    pub(crate) fn as_scalar(&self) -> Result<&'a ScalarValue, String> {
        match self {
            Self::Scalar(value) => Ok(value),
            Self::Cell(cell) => Err(cell.access_error("scalar conversion")),
        }
    }

    pub(crate) fn as_integer(&self) -> Option<i64> {
        self.as_scalar().ok().and_then(ScalarValue::as_integer)
    }

    pub(crate) fn as_unsigned(&self) -> Option<u64> {
        self.as_scalar().ok().and_then(ScalarValue::as_unsigned)
    }

    pub(crate) fn as_raw_timestamp(&self) -> Option<(i64, TimeUnit, Option<&'a str>)> {
        self.as_scalar().ok().and_then(ScalarValue::as_raw_timestamp)
    }

    pub(crate) fn as_raw_time(&self) -> Option<(i64, TimeUnit)> {
        self.as_scalar().ok().and_then(ScalarValue::as_raw_time)
    }

    pub(crate) fn to_owned_value(self) -> Result<OwnedValue, String> {
        match self {
            Self::Scalar(value) => Ok(OwnedValue::from(value.clone())),
            Self::Cell(cell) => Ok(OwnedValue::SourceCell(cell)),
        }
    }

    pub(crate) fn into_cell(self) -> Option<SourceCellRef> {
        match self {
            Self::Cell(cell) => Some(cell),
            Self::Scalar(_) => None,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum ResolvedValue<'a> {
    Source(SourceValueRef<'a>),
    Owned(&'a OwnedValue),
}

impl<'a> ResolvedValue<'a> {
    #[allow(dead_code)]
    pub(crate) fn as_scalar(&self) -> Result<&ScalarValue, String> {
        match self {
            ResolvedValue::Source(value) => value.as_scalar(),
            ResolvedValue::Owned(value) => value.as_scalar(),
        }
    }

    pub(crate) fn into_owned_value(self) -> Result<OwnedValue, String> {
        match self {
            ResolvedValue::Source(value) => value.to_owned_value(),
            ResolvedValue::Owned(value) => Ok(value.clone()),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum AppendValue<'a> {
    Source(SourceValueRef<'a>),
    Owned(&'a OwnedValue),
}

impl<'a> AppendValue<'a> {
    pub(crate) fn into_owned_value(self) -> Result<OwnedValue, String> {
        match self {
            AppendValue::Source(value) => value.to_owned_value(),
            AppendValue::Owned(value) => Ok(value.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{OwnedValue, SourceValueRef};
    use crate::expressions::ScalarValue;
    use arrow::datatypes::TimeUnit;

    #[test]
    fn value_references_preserve_exact_scalar_domain() {
        let value = ScalarValue::Unsigned(u64::MAX);
        let source = SourceValueRef::Scalar(&value);

        assert_eq!(source.as_unsigned(), Some(u64::MAX));
        assert_eq!(source.as_integer(), None);
        assert_eq!(source.to_owned_value().unwrap().as_scalar().unwrap(), &value);

        let owned = OwnedValue::from(value);
        assert_eq!(owned.as_unsigned(), Some(u64::MAX));
        assert_eq!(owned.as_integer(), None);
    }

    #[test]
    fn value_references_preserve_raw_timestamp_identity() {
        let value = ScalarValue::RawTimestamp {
            value: -1,
            unit: TimeUnit::Nanosecond,
            timezone: Some("Asia/Tokyo".to_string()),
        };
        let source = SourceValueRef::Scalar(&value);
        let owned = source.clone().to_owned_value().unwrap();

        assert_eq!(
            source.as_raw_timestamp(),
            Some((-1, TimeUnit::Nanosecond, Some("Asia/Tokyo")))
        );
        assert_eq!(
            owned.as_raw_timestamp(),
            Some((-1, TimeUnit::Nanosecond, Some("Asia/Tokyo")))
        );
        assert_eq!(source.as_raw_time(), None);
    }
}
