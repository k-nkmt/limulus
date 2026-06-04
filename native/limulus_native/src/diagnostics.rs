use std::ops::Range;

use ariadne::{Label, Report, ReportKind, Source};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};

#[derive(Debug, Clone)]
pub(crate) struct RenderPayload {
    pub(crate) source_id: String,
    pub(crate) source_text: Option<String>,
    pub(crate) diagnostics: Vec<RenderDiagnostic>,
}

#[derive(Debug, Clone)]
pub(crate) struct RenderDiagnostic {
    pub(crate) code: String,
    pub(crate) severity: String,
    pub(crate) message: String,
    pub(crate) location: String,
    pub(crate) stage: String,
    pub(crate) span: Option<RenderSpan>,
    pub(crate) labels: Vec<RenderLabel>,
    pub(crate) notes: Vec<String>,
    pub(crate) source_text: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct RenderLabel {
    pub(crate) span: Option<RenderSpan>,
    pub(crate) message: String,
    #[allow(dead_code)]
    pub(crate) kind: String,
}

#[derive(Debug, Clone)]
pub(crate) struct RenderSpan {
    pub(crate) start: usize,
    pub(crate) end: usize,
    #[allow(dead_code)]
    pub(crate) line: usize,
    #[allow(dead_code)]
    pub(crate) column: usize,
    #[allow(dead_code)]
    pub(crate) end_line: Option<usize>,
    #[allow(dead_code)]
    pub(crate) end_column: Option<usize>,
    pub(crate) source_id: String,
}

pub(crate) fn render_diagnostics_ariadne(payload: &Bound<'_, PyDict>) -> PyResult<String> {
    let render_payload = extract_render_payload(payload)?;
    let mut rendered: Vec<String> = Vec::new();
    for diagnostic in &render_payload.diagnostics {
        match render_ariadne_diagnostic(diagnostic, &render_payload) {
            Ok(output) => rendered.push(output),
            Err(_) => rendered.push(render_simple_diagnostic(diagnostic)),
        }
    }
    Ok(rendered.join("\n"))
}

fn extract_render_payload(payload: &Bound<'_, PyDict>) -> PyResult<RenderPayload> {
    let source_id = py_dict_string(payload, "source_id", "<dsl>")?;
    let source_text = py_dict_optional_string(payload, "source_text")?;
    let diagnostics_any = payload
        .get_item("diagnostics")?
        .ok_or_else(|| PyValueError::new_err("payload.diagnostics is required"))?;
    let diagnostics_list = diagnostics_any.cast::<PyList>()?;
    let mut diagnostics: Vec<RenderDiagnostic> = Vec::new();
    for item in diagnostics_list.iter() {
        let diagnostic_dict = item.cast::<PyDict>()?;
        diagnostics.push(extract_render_diagnostic(&diagnostic_dict)?);
    }
    Ok(RenderPayload {
        source_id,
        source_text,
        diagnostics,
    })
}

fn extract_render_diagnostic(payload: &Bound<'_, PyDict>) -> PyResult<RenderDiagnostic> {
    let labels_any = payload.get_item("labels")?;
    let mut labels: Vec<RenderLabel> = Vec::new();
    if let Some(labels_value) = labels_any {
        let labels_list = labels_value.cast::<PyList>()?;
        for item in labels_list.iter() {
            let label_dict = item.cast::<PyDict>()?;
            labels.push(extract_render_label(&label_dict)?);
        }
    }

    let notes_any = payload.get_item("notes")?;
    let mut notes: Vec<String> = Vec::new();
    if let Some(notes_value) = notes_any {
        let notes_list = notes_value.cast::<PyList>()?;
        for item in notes_list.iter() {
            notes.push(item.extract::<String>()?);
        }
    }

    Ok(RenderDiagnostic {
        code: py_dict_string(payload, "code", "UNKNOWN")?,
        severity: py_dict_string(payload, "severity", "error")?,
        message: py_dict_string(payload, "message", "")?,
        location: py_dict_string(payload, "location", "")?,
        stage: py_dict_string(payload, "stage", "")?,
        span: extract_optional_render_span(payload.get_item("span")?)?,
        labels,
        notes,
        source_text: py_dict_optional_string(payload, "source_text")?,
    })
}

fn extract_render_label(payload: &Bound<'_, PyDict>) -> PyResult<RenderLabel> {
    Ok(RenderLabel {
        span: extract_optional_render_span(payload.get_item("span")?)?,
        message: py_dict_string(payload, "message", "")?,
        kind: py_dict_string(payload, "kind", "primary")?,
    })
}

fn extract_optional_render_span(value: Option<Bound<'_, PyAny>>) -> PyResult<Option<RenderSpan>> {
    let Some(span_value) = value else {
        return Ok(None);
    };
    if span_value.is_none() {
        return Ok(None);
    }
    let span_dict = span_value.cast::<PyDict>()?;
    Ok(Some(RenderSpan {
        start: py_dict_usize(&span_dict, "start", 0)?,
        end: py_dict_usize(&span_dict, "end", 1)?,
        line: py_dict_usize(&span_dict, "line", 1)?,
        column: py_dict_usize(&span_dict, "column", 1)?,
        end_line: py_dict_optional_usize(&span_dict, "end_line")?,
        end_column: py_dict_optional_usize(&span_dict, "end_column")?,
        source_id: py_dict_string(&span_dict, "source_id", "<dsl>")?,
    }))
}

fn py_dict_string(payload: &Bound<'_, PyDict>, key: &str, default: &str) -> PyResult<String> {
    let Some(value) = payload.get_item(key)? else {
        return Ok(default.to_string());
    };
    if value.is_none() {
        return Ok(default.to_string());
    }
    value.extract::<String>()
}

fn py_dict_optional_string(payload: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    let Some(value) = payload.get_item(key)? else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    Ok(Some(value.extract::<String>()?))
}

fn py_dict_usize(payload: &Bound<'_, PyDict>, key: &str, default: usize) -> PyResult<usize> {
    let Some(value) = payload.get_item(key)? else {
        return Ok(default);
    };
    if value.is_none() {
        return Ok(default);
    }
    value.extract::<usize>()
}

fn py_dict_optional_usize(payload: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<usize>> {
    let Some(value) = payload.get_item(key)? else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    Ok(Some(value.extract::<usize>()?))
}

fn normalize_render_range(span: &RenderSpan, source_text: &str) -> Range<usize> {
    let start = span.start.min(source_text.len());
    let end = span.end.max(start + 1).min(source_text.len().max(start + 1));
    start..end
}

fn render_report_kind(severity: &str) -> ReportKind<'static> {
    match severity.to_lowercase().as_str() {
        "warning" => ReportKind::Warning,
        "info" | "note" => ReportKind::Advice,
        _ => ReportKind::Error,
    }
}

fn render_header_message(diagnostic: &RenderDiagnostic) -> String {
    if diagnostic.stage.is_empty() {
        diagnostic.message.clone()
    } else {
        format!("[stage: {}] {}", diagnostic.stage, diagnostic.message)
    }
}

fn render_simple_diagnostic(diagnostic: &RenderDiagnostic) -> String {
    let mut header = format!("{}[{}]", diagnostic.severity.to_lowercase(), diagnostic.code);
    if !diagnostic.stage.is_empty() {
        header.push_str(&format!(" [stage: {}]", diagnostic.stage));
    }
    header.push_str(&format!(": {}", diagnostic.message));
    if !diagnostic.location.is_empty() {
        header.push_str(&format!(" ({})", diagnostic.location));
    }
    let mut lines = vec![header];
    for note in &diagnostic.notes {
        lines.push(format!("note: {}", note));
    }
    lines.join("\n")
}

fn render_ariadne_diagnostic(diagnostic: &RenderDiagnostic, payload: &RenderPayload) -> Result<String, String> {
    let source_text = diagnostic
        .source_text
        .as_deref()
        .or(payload.source_text.as_deref());
    let span = diagnostic.span.as_ref();
    let Some(source_text) = source_text else {
        return Ok(render_simple_diagnostic(diagnostic));
    };
    let Some(span) = span else {
        return Ok(render_simple_diagnostic(diagnostic));
    };

    let source_id = if span.source_id.is_empty() {
        payload.source_id.clone()
    } else {
        span.source_id.clone()
    };
    let report_range = normalize_render_range(span, source_text);
    let mut builder = Report::build(
        render_report_kind(&diagnostic.severity),
        (source_id.clone(), report_range.clone()),
    )
    .with_code(diagnostic.code.clone())
    .with_message(render_header_message(diagnostic));

    let mut has_label = false;
    for label in &diagnostic.labels {
        let Some(label_span) = label.span.as_ref() else {
            continue;
        };
        let range = normalize_render_range(label_span, source_text);
        let mut ariadne_label = Label::new((source_id.clone(), range));
        if !label.message.is_empty() {
            ariadne_label = ariadne_label.with_message(label.message.clone());
        }
        builder = builder.with_label(ariadne_label);
        has_label = true;
    }

    if !has_label {
        let range = normalize_render_range(span, source_text);
        builder = builder.with_label(Label::new((source_id.clone(), range)));
    }

    for note in &diagnostic.notes {
        builder = builder.with_note(note.clone());
    }

    let report = builder.finish();
    let mut buffer: Vec<u8> = Vec::new();
    if let Err(error) = report.write((source_id, Source::from(source_text)), &mut buffer) {
        return Err(error.to_string());
    }
    String::from_utf8(buffer).map_err(|error| error.to_string())
}
