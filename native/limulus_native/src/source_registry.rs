use std::fmt;
use std::rc::Rc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SourceId(usize);

impl SourceId {
    pub(crate) fn new(value: usize) -> Self {
        Self(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SourceRegistryError {
    InvalidSourceId(SourceId),
    PreparedMergeAlreadyRegistered,
}

impl fmt::Display for SourceRegistryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSourceId(source_id) => {
                write!(formatter, "invalid source id: {source_id:?}")
            }
            Self::PreparedMergeAlreadyRegistered => {
                write!(formatter, "prepared MERGE source is already registered")
            }
        }
    }
}

pub(crate) struct SourceHandle<T> {
    source_id: SourceId,
    cursor: Rc<T>,
}

impl<T> Clone for SourceHandle<T> {
    fn clone(&self) -> Self {
        Self {
            source_id: self.source_id,
            cursor: Rc::clone(&self.cursor),
        }
    }
}

impl<T> SourceHandle<T> {
    pub(crate) fn source_id(&self) -> SourceId {
        self.source_id
    }

    pub(crate) fn cursor(&self) -> Rc<T> {
        Rc::clone(&self.cursor)
    }
}

pub(crate) struct ExecutionSourceRegistry<T> {
    sources: Vec<SourceHandle<T>>,
    prepared_merge_source_id: Option<SourceId>,
}

impl<T> Default for ExecutionSourceRegistry<T> {
    fn default() -> Self {
        Self {
            sources: Vec::new(),
            prepared_merge_source_id: None,
        }
    }
}

impl<T> ExecutionSourceRegistry<T> {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn register(&mut self, cursor: Rc<T>) -> SourceId {
        let source_id = SourceId(self.sources.len());
        self.sources.push(SourceHandle { source_id, cursor });
        source_id
    }

    pub(crate) fn register_prepared_merge(
        &mut self,
        cursor: Rc<T>,
    ) -> Result<SourceId, SourceRegistryError> {
        if self.prepared_merge_source_id.is_some() {
            return Err(SourceRegistryError::PreparedMergeAlreadyRegistered);
        }
        let source_id = self.register(cursor);
        self.prepared_merge_source_id = Some(source_id);
        Ok(source_id)
    }

    pub(crate) fn lookup(
        &self,
        source_id: SourceId,
    ) -> Result<SourceHandle<T>, SourceRegistryError> {
        self.sources
            .get(source_id.0)
            .cloned()
            .ok_or(SourceRegistryError::InvalidSourceId(source_id))
    }

    pub(crate) fn len(&self) -> usize {
        self.sources.len()
    }
}

#[cfg(test)]
mod tests {
    use super::{ExecutionSourceRegistry, SourceId, SourceRegistryError};
    use std::cell::Cell;
    use std::rc::Rc;

    #[derive(Debug, PartialEq, Eq)]
    struct TestCursor(u8);

    #[test]
    fn assigns_stable_ids_in_load_order_and_registers_prepared_merge_once() {
        let first_cursor = Rc::new(TestCursor(1));
        let prepared_merge_cursor = Rc::new(TestCursor(2));
        let mut registry = ExecutionSourceRegistry::new();

        let first_id = registry.register(Rc::clone(&first_cursor));
        let prepared_merge_id = registry
            .register_prepared_merge(Rc::clone(&prepared_merge_cursor))
            .expect("prepared MERGE carrier must register once");

        assert_eq!(first_id, SourceId(0));
        assert_eq!(prepared_merge_id, SourceId(1));
        assert_eq!(registry.len(), 2);

        let first_handle = registry.lookup(first_id).expect("first source must resolve");
        let first_lookup = first_handle.cursor();
        assert_eq!(first_handle.source_id(), first_id);
        assert!(Rc::ptr_eq(&first_cursor, &first_lookup));

        let prepared_handle = registry
            .lookup(prepared_merge_id)
            .expect("prepared MERGE source must resolve");
        let prepared_lookup = prepared_handle.cursor();
        assert_eq!(prepared_handle.source_id(), prepared_merge_id);
        assert!(Rc::ptr_eq(&prepared_merge_cursor, &prepared_lookup));
    }

    #[test]
    fn invalid_source_id_lookup_returns_explicit_error() {
        let mut registry = ExecutionSourceRegistry::new();
        let source_id = registry.register(Rc::new(TestCursor(1)));

        assert!(matches!(
            registry.lookup(SourceId(9)),
            Err(SourceRegistryError::InvalidSourceId(invalid_id)) if invalid_id != source_id
        ));
    }

    #[test]
    fn rejects_duplicate_prepared_merge_registration_without_allocating_a_source() {
        let mut registry = ExecutionSourceRegistry::new();

        let first_id = registry
            .register_prepared_merge(Rc::new(TestCursor(1)))
            .expect("the first prepared MERGE carrier must register");
        let duplicate = registry.register_prepared_merge(Rc::new(TestCursor(2)));

        assert_eq!(first_id, SourceId(0));
        assert_eq!(duplicate, Err(SourceRegistryError::PreparedMergeAlreadyRegistered));
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn registry_handle_can_be_shared_by_state_and_output_builder() {
        let registry = Rc::new(ExecutionSourceRegistry::<TestCursor>::new());
        let state_registry = Rc::clone(&registry);
        let builder_registry = Rc::clone(&registry);

        assert!(Rc::ptr_eq(&registry, &state_registry));
        assert!(Rc::ptr_eq(&registry, &builder_registry));
        assert_eq!(Rc::strong_count(&registry), 3);
    }

    #[test]
    fn registry_keeps_cursor_alive_until_registry_drop() {
        struct DropProbe(Rc<Cell<bool>>);

        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.set(true);
            }
        }

        let dropped = Rc::new(Cell::new(false));
        let cursor = Rc::new(DropProbe(Rc::clone(&dropped)));
        let mut registry = ExecutionSourceRegistry::new();
        let source_id = registry.register(Rc::clone(&cursor));
        let handle = registry.lookup(source_id).expect("source must resolve");
        let lookup_cursor = handle.cursor();

        drop(cursor);
        drop(handle);
        drop(lookup_cursor);
        assert!(!dropped.get(), "registry owns the execution-scoped cursor");

        drop(registry);
        assert!(dropped.get(), "cursor is released with the registry");
    }
}
