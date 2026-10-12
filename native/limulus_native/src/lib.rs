//! Native limulus runtime entry points.

mod ast;
mod diagnostics;
mod expressions;
mod output;
mod row_cursor;
mod row_view;
mod runtime;
mod source_registry;
mod source_view;
mod typed_ast;
mod typed_lowering;
mod value_ref;

use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
fn execute_block(py: Python<'_>, payload: &Bound<'_, PyDict>) -> PyResult<Py<PyDict>> {
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
