use crate::expressions::ScalarValue;

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum OwnedValue {
    Scalar(ScalarValue),
}

impl OwnedValue {
    pub(crate) fn null() -> Self {
        Self::Scalar(ScalarValue::Null)
    }

    pub(crate) fn as_scalar(&self) -> &ScalarValue {
        match self {
            Self::Scalar(value) => value,
        }
    }

    pub(crate) fn into_scalar_value(self) -> ScalarValue {
        match self {
            Self::Scalar(value) => value,
        }
    }
}

impl From<ScalarValue> for OwnedValue {
    fn from(value: ScalarValue) -> Self {
        Self::Scalar(value)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum SourceValueRef<'a> {
    Scalar(&'a ScalarValue),
}

impl<'a> SourceValueRef<'a> {
    pub(crate) fn as_scalar(self) -> &'a ScalarValue {
        match self {
            Self::Scalar(value) => value,
        }
    }

    pub(crate) fn to_owned_value(self) -> OwnedValue {
        OwnedValue::from(self.as_scalar().clone())
    }
}

#[derive(Debug, Clone)]
pub(crate) enum ResolvedValue<'a> {
    Source(SourceValueRef<'a>),
    Owned(&'a OwnedValue),
}

impl<'a> ResolvedValue<'a> {
    #[allow(dead_code)]
    pub(crate) fn as_scalar(&self) -> &ScalarValue {
        match self {
            ResolvedValue::Source(value) => value.as_scalar(),
            ResolvedValue::Owned(value) => value.as_scalar(),
        }
    }

    pub(crate) fn into_owned_value(self) -> OwnedValue {
        match self {
            ResolvedValue::Source(value) => value.to_owned_value(),
            ResolvedValue::Owned(value) => value.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum AppendValue<'a> {
    Source(SourceValueRef<'a>),
    Owned(&'a OwnedValue),
}

impl<'a> AppendValue<'a> {
    pub(crate) fn into_owned_value(self) -> OwnedValue {
        match self {
            AppendValue::Source(value) => value.to_owned_value(),
            AppendValue::Owned(value) => value.clone(),
        }
    }
}
