use crate::ast::AstDatasetRefOptions;
use crate::expressions::{
    self, is_expression_parse_error, optional_logical_values_equal, py_to_scalar,
    resolve_array_variable_name, resolve_value, scalar_to_bool, scalar_to_number, ScalarValue,
};
use crate::output::{
    append_projected_row_to_target, ensure_output_target_shape_from_view,
    export_output_streams, initialize_output_builders,
    resolve_declared_target_name,
};
use crate::row_cursor::NativeArrowRowCursor;
use crate::row_view::{ColumnBinding, LogicalRowAccess, NativeLogicalRowView};
use crate::source_registry::ExecutionSourceRegistry;
use crate::typed_ast;
use crate::typed_lowering::{
    self, LoweredAssignmentTarget, LoweredExpression, LoweredStatement, LoweredStatementPayload,
};
use crate::value_ref::{OwnedValue, ResolvedValue};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::time::Instant;

mod state;
use self::state::{
    get_stateful_owned_value, mark_automatic_error, reset_automatic_scalars, set_runtime_flag,
    set_runtime_text, store_runtime_scalar,
};

#[derive(Default)]
pub(crate) struct EvalRuntimeState {
    pub(crate) lag_queues: HashMap<usize, Vec<ScalarValue>>,
    pub(crate) sum_totals: HashMap<String, f64>,
    pub(crate) retain_values: HashMap<String, OwnedValue>,
    pub(crate) automatic_values: HashMap<String, OwnedValue>,
    pub(crate) mutable_values: HashMap<String, OwnedValue>,
    pub(crate) materialize_mutable_values: bool,
    pub(crate) array_defs: HashMap<String, Vec<String>>,
    pub(crate) dataset_option_survivor_counts: HashMap<String, usize>,
    pub(crate) row_index: isize,
    pub(crate) source_registry: Rc<ExecutionSourceRegistry<NativeArrowRowCursor>>,
    pub(crate) source_cursor: Option<Rc<NativeArrowRowCursor>>,
    pub(crate) logical_row: Option<NativeLogicalRowView>,
    pub(crate) apply_registry: HashMap<String, Py<PyAny>>,
    pub(crate) numeric_format_catalogs: HashMap<String, Vec<(ScalarValue, String)>>,
    pub(crate) character_format_catalogs: HashMap<String, Vec<(ScalarValue, String)>>,
    pub(crate) informat_catalogs: HashMap<String, Vec<(ScalarValue, f64)>>,
}

fn condition_error_details(
    display: &str,
    cause: &str,
    syntax_summary: impl FnOnce() -> Result<Option<String>, String>,
) -> Result<(&'static str, String), String> {
    if is_expression_parse_error(cause) {
        let summary = syntax_summary()?
            .filter(|summary| !summary.is_empty())
            .unwrap_or_else(|| cause.to_string());
        Ok((
            "RUNTIME_OPERATOR_NOT_SUPPORTED",
            format!("Unsupported operator in expression '{display}': {summary}"),
        ))
    } else {
        Ok((
            "RUNTIME_EXPRESSION_EVALUATION_ERROR",
            format!("Rust WHERE evaluation failed: {cause}"),
        ))
    }
}

/// Format a typed condition failure from its saved syntax summary. This path
/// deliberately never compiles the display text; the callback seam keeps the
/// diagnostic classifier shared with its focused compatibility tests.
pub(crate) fn typed_condition_error_diagnostic(
    display: &str,
    cause: &str,
    where_syntax_summary: &str,
) -> (&'static str, String) {
    condition_error_details(display, cause, || {
        Ok(Some(where_syntax_summary.to_string()))
    })
    .expect("saved typed condition summaries are already available")
}

#[derive(Default)]
pub(crate) struct RowExecutionOutcome {
    pub(crate) deleted: bool,
    pub(crate) stopped: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PreparationKind {
    None,
    PreparedSet,
    PreparedMerge,
}

enum TypedWhereClause {
    Expression {
        condition: LoweredExpression,
        statement: LoweredStatement,
    },
    Deferred(typed_lowering::DeferredStatementDescriptor),
}

fn parse_preparation_kind(payload: &Bound<'_, PyDict>) -> PyResult<PreparationKind> {
    let value = payload
        .get_item("preparation_kind")?
        .ok_or_else(|| PyValueError::new_err("payload.preparation_kind is required"))?;
    let value = value.extract::<String>()?;
    match value.as_str() {
        "none" => Ok(PreparationKind::None),
        "prepared_set" => Ok(PreparationKind::PreparedSet),
        "prepared_merge" => Ok(PreparationKind::PreparedMerge),
        value => Err(PyValueError::new_err(format!(
            "payload.preparation_kind is invalid: {value}"
        ))),
    }
}

fn build_logical_row_schema_view(
    source_registry: &Rc<ExecutionSourceRegistry<NativeArrowRowCursor>>,
    source_cursor: &Rc<NativeArrowRowCursor>,
) -> Result<NativeLogicalRowView, String> {
    let source_id = source_cursor.source_id()?;
    let columns = source_cursor
        .slot_order()
        .iter()
        .enumerate()
        .map(|(slot, name)| (name.clone(), ColumnBinding::Source { source_id, slot }))
        .collect();
    NativeLogicalRowView::from_schema(Rc::clone(source_registry), columns)
        .map_err(|error| error.to_string())
}

fn parse_text_catalog_group(
    catalogs_any: Option<&Bound<'_, PyAny>>,
) -> PyResult<HashMap<String, Vec<(ScalarValue, String)>>> {
    let Some(catalogs_any) = catalogs_any else {
        return Ok(HashMap::new());
    };
    let catalogs_dict = catalogs_any
        .cast::<PyDict>()
        .map_err(|_| PyValueError::new_err("format catalog group must be a dict"))?;
    let mut resolved = HashMap::new();
    for (name_any, entries_any) in catalogs_dict.iter() {
        let name = name_any.extract::<String>()?;
        let entries_dict = entries_any
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err("format catalog entries must be a dict"))?;
        let mut entries = Vec::new();
        for (key_any, value_any) in entries_dict.iter() {
            let key = py_to_scalar(&key_any)?;
            let value = value_any.extract::<String>()?;
            entries.push((key, value));
        }
        resolved.insert(name, entries);
    }
    Ok(resolved)
}

fn parse_numeric_catalog_group(
    catalogs_any: Option<&Bound<'_, PyAny>>,
) -> PyResult<HashMap<String, Vec<(ScalarValue, f64)>>> {
    let Some(catalogs_any) = catalogs_any else {
        return Ok(HashMap::new());
    };
    let catalogs_dict = catalogs_any
        .cast::<PyDict>()
        .map_err(|_| PyValueError::new_err("informat catalog group must be a dict"))?;
    let mut resolved = HashMap::new();
    for (name_any, entries_any) in catalogs_dict.iter() {
        let name = name_any.extract::<String>()?;
        let entries_dict = entries_any
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err("informat catalog entries must be a dict"))?;
        let mut entries = Vec::new();
        for (key_any, value_any) in entries_dict.iter() {
            let key = py_to_scalar(&key_any)?;
            let value = value_any.extract::<f64>()?;
            entries.push((key, value));
        }
        resolved.insert(name, entries);
    }
    Ok(resolved)
}

fn diag(py: Python<'_>, code: &str, location: &str, message: &str) -> PyResult<Py<PyDict>> {
    let item = PyDict::new(py);
    item.set_item("code", code)?;
    item.set_item("severity", "error")?;
    item.set_item("location", location)?;
    item.set_item("message", message)?;
    Ok(item.unbind())
}

// Absent SUM targets reuse the prior accumulator or the 0 base, addressed by
// the canonical key.
fn absent_sum_base(state: &EvalRuntimeState, name: &str) -> f64 {
    state
        .sum_totals
        .get(&expressions::column_key(name))
        .copied()
        .unwrap_or(0.0)
}

// Single owner for every SUM statement state update (inline and normal arms
// call this and nothing else writes `sum_totals`).
pub(crate) fn apply_sum_update(
    row: &Bound<'_, PyDict>,
    state: &mut EvalRuntimeState,
    name: &str,
    increment: ScalarValue,
) -> Result<(), String> {
    // Increment classification first: NULL adds 0 without touching the base;
    // other values use the checked numeric coercion and fail closed with the
    // target variable identified.
    let increment_value = if matches!(increment, ScalarValue::Null) {
        0.0
    } else {
        scalar_to_number(&increment).map_err(|cause| {
            format!("SUM statement target '{name}' requires a numeric increment: {cause}")
        })?
    };

    // Target classification distinguishes Absent from an explicit NULL target
    // without collapsing the ResolvedValue variant first: a seeded owned NULL
    // binding that was never written this row is Absent,
    // while a written NULL (tracked by the per-row mutable ledger) or a source
    // NULL cell is an explicit NULL that resets the base.
    let base = match resolve_value(row, name, state)? {
        Some(ResolvedValue::Source(value)) => {
            let scalar = value
                .to_owned_value()
                .map_err(|error| {
                    format!(
                        "complex access is unsupported for column '{name}' during SUM statement: {error}"
                    )
                })?
                .into_scalar_value()
                .map_err(|error| {
                    format!(
                        "complex access is unsupported for column '{name}' during SUM statement: {error}"
                    )
                })?;
            if matches!(scalar, ScalarValue::Null) {
                // Explicit source NULL: the prior accumulator is not revived; base resets to 0.
                0.0
            } else {
                scalar_to_number(&scalar).map_err(|cause| {
                    format!(
                        "SUM statement target '{name}' requires a numeric target value: {cause}"
                    )
                })?
            }
        }
        Some(ResolvedValue::Owned(value)) => {
            let scalar = value.clone().into_scalar_value().map_err(|error| {
                format!(
                    "complex access is unsupported for column '{name}' during SUM statement: {error}"
                )
            })?;
            if !matches!(scalar, ScalarValue::Null) {
                scalar_to_number(&scalar).map_err(|cause| {
                    format!(
                        "SUM statement target '{name}' requires a numeric target value: {cause}"
                    )
                })?
            } else if state
                .mutable_values
                .contains_key(&expressions::column_key(name))
            {
                // Written NULL this row: the prior accumulator is not revived; base resets to 0.
                0.0
            } else {
                // Seeded absent binding (never written this row): the row value
                // does not exist, so the prior accumulator or 0 applies.
                absent_sum_base(state, name)
            }
        }
        // No binding at all: absent; prior accumulator or 0.
        None => absent_sum_base(state, name),
    };

    // Accumulator and row value share the same canonical key.
    let total = base + increment_value;
    state
        .sum_totals
        .insert(expressions::column_key(name), total);
    store_runtime_scalar(row, state, name, &ScalarValue::Number(total))
}

fn dataset_option_error_message(scope_key: &str, error: &str) -> String {
    let (subject_kind, subject_name) = if let Some(name) = scope_key.strip_prefix("input:") {
        ("source", name)
    } else if let Some(name) = scope_key.strip_prefix("output:") {
        ("output", name)
    } else {
        ("dataset", scope_key)
    };

    if let Some(detail) = error.strip_prefix("dataset option WHERE= evaluation failed: ") {
        return format!(
            "Dataset option WHERE= evaluation failed for {subject_kind} '{subject_name}': {detail}"
        );
    }
    format!("Invalid dataset option for {subject_kind} '{subject_name}': {error}")
}

fn dataset_ref_options_are_passthrough(options: &AstDatasetRefOptions) -> bool {
    options.keep_vars.is_empty()
        && options.drop_vars.is_empty()
        && options.where_expr.is_none()
        && options.rename_map.is_empty()
        && options.firstobs.is_none()
        && options.obs.is_none()
}

fn output_target_passes_options(
    target: &str,
    options: &AstDatasetRefOptions,
    state: &mut EvalRuntimeState,
    global_keep: &[String],
    global_drop: &[String],
    global_rename: &HashMap<String, String>,
) -> Result<bool, String> {
    if let Some(firstobs) = options.firstobs {
        if firstobs <= 0 {
            return Err("dataset option FIRSTOBS= must be positive".to_string());
        }
    }
    if let Some(obs) = options.obs {
        if obs < 0 {
            return Err("dataset option OBS= must be non-negative".to_string());
        }
    }

    if let Some(expression) = options.where_expr.as_ref() {
        let mut target_view = state
            .logical_row
            .as_ref()
            .ok_or_else(|| "logical row view is required for output WHERE= evaluation".to_string())?
            .clone();
        target_view
            .project_columns(global_keep, global_drop, global_rename)
            .and_then(|_| {
                target_view.project_columns(
                    &options.keep_vars,
                    &options.drop_vars,
                    &options.rename_map,
                )
            })
            .map_err(|error| error.to_string())?;
        if !expressions::evaluate_structured_where_on_view(expression, &target_view)? {
            return Ok(false);
        }
    }

    if options.firstobs.is_none() && options.obs.is_none() {
        return Ok(true);
    }

    let scope_key = format!("output:{}", target.to_lowercase());
    let survivor_count = state
        .dataset_option_survivor_counts
        .entry(scope_key)
        .or_insert(0);
    *survivor_count += 1;
    let position = *survivor_count as i64;
    let firstobs = options.firstobs.unwrap_or(1);
    if position < firstobs {
        return Ok(false);
    }
    if let Some(obs) = options.obs {
        if position >= firstobs + obs {
            return Ok(false);
        }
    }
    Ok(true)
}

fn get_plan_item<'py>(
    owner: &Bound<'py, PyAny>,
    name: &str,
) -> Result<Option<Bound<'py, PyAny>>, String> {
    if let Ok(dict) = owner.cast::<PyDict>() {
        return dict
            .get_item(name)
            .map_err(|error| format!("failed to read plan key '{name}': {error}"));
    }
    match owner.getattr(name) {
        Ok(value) => Ok(Some(value)),
        Err(error) => {
            if error.is_instance_of::<pyo3::exceptions::PyAttributeError>(owner.py()) {
                Ok(None)
            } else {
                Err(format!("failed to read plan attr '{name}': {error}"))
            }
        }
    }
}

fn get_plan_text(owner: &Bound<'_, PyAny>, name: &str) -> Result<Option<String>, String> {
    let Some(value) = get_plan_item(owner, name)? else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    value
        .extract::<String>()
        .map(Some)
        .map_err(|error| format!("failed to parse plan field '{name}' as text: {error}"))
}

fn get_plan_text_list(owner: &Bound<'_, PyAny>, name: &str) -> Result<Vec<String>, String> {
    let Some(value) = get_plan_item(owner, name)? else {
        return Ok(Vec::new());
    };
    if value.is_none() {
        return Ok(Vec::new());
    }
    value
        .extract::<Vec<String>>()
        .map_err(|error| format!("failed to parse plan field '{name}' as list[str]: {error}"))
}

pub(crate) fn execute_block(py: Python<'_>, payload: &Bound<'_, PyDict>) -> PyResult<Py<PyDict>> {
    let result = PyDict::new(py);
    let diagnostics = PyList::empty(py);
    let input_import_start = Instant::now();
    let native_program_json = match payload.get_item("native_program_json")? {
        Some(value) => match value.extract::<String>() {
            Ok(json) => json,
            Err(error) => {
                return invalid_ast_payload_result(
                    py,
                    &result,
                    &diagnostics,
                    &format!("payload.native_program_json must be a string: {error}"),
                );
            }
        },
        None => {
            return invalid_ast_payload_result(
                py,
                &result,
                &diagnostics,
                "payload.native_program_json is required",
            );
        }
    };
    let typed_payload = match typed_ast::parse_typed_program(&native_program_json) {
        Ok(parsed) => parsed,
        Err(error) => return invalid_ast_payload_result(py, &result, &diagnostics, &error),
    };
    let typed_program = match typed_lowering::lower_program(&typed_payload) {
        Ok(program) => program,
        Err(error) => return invalid_ast_payload_result(py, &result, &diagnostics, &error),
    };

    // Frontend SKIPPED markers intentionally retain the established early
    // unsupported-statement diagnostic. Other typed variants are dispatched
    // structurally below; there is no staged statement allowlist or text path.
    if let Some((index, statement)) = typed_program
        .statements
        .iter()
        .enumerate()
        .find(|(_, statement)| matches!(statement.payload, LoweredStatementPayload::Skipped))
    {
        diagnostics.append(diag(
            py,
            "RUNTIME_RUST_STATEMENT_UNSUPPORTED",
            &format!("statement:{}", index + 1),
            &format!(
                "Rust native runtime does not support statement kind: {}",
                statement.kind
            ),
        )?)?;
        result.set_item("diagnostics", diagnostics)?;
        return Ok(result.unbind());
    }

    let mut set_input_names: Vec<String> = Vec::new();
    let mut typed_where: Option<TypedWhereClause> = None;
    let mut keep_vars: Vec<String> = Vec::new();
    let mut drop_vars: Vec<String> = Vec::new();
    let mut rename_map: HashMap<String, String> = HashMap::new();
    let mut by_vars: Vec<String> = Vec::new();
    let mut retain_vars: Vec<String> = Vec::new();
    let mut set_in_var_by_dataset: HashMap<String, String> = HashMap::new();
    let mut set_options_by_dataset: HashMap<String, AstDatasetRefOptions> = HashMap::new();
    let mut output_options_by_target: HashMap<String, AstDatasetRefOptions> = HashMap::new();
    let mut all_in_vars: Vec<String> = Vec::new();
    let mut indsname_var: Option<String> = None;
    let mut end_var: Option<String> = None;

    for (target, options) in &typed_program.effective_output_options {
        output_options_by_target.insert(target.to_lowercase(), options.clone());
    }
    for statement in &typed_program.statements {
        match &statement.payload {
            LoweredStatementPayload::Source { refs, options } if statement.kind == "SET" => {
                for dataset_ref in refs {
                    set_input_names.push(dataset_ref.name.clone());
                    set_options_by_dataset
                        .insert(dataset_ref.name.to_lowercase(), dataset_ref.options.clone());
                    if let Some(in_var_name) = &dataset_ref.options.in_var {
                        set_in_var_by_dataset
                            .insert(dataset_ref.name.to_lowercase(), in_var_name.clone());
                        all_in_vars.push(in_var_name.clone());
                    }
                }
                indsname_var = options.indsname_var.clone();
                end_var = options.end_var.clone();
            }
            LoweredStatementPayload::Where { condition } if statement.kind == "WHERE" => {
                typed_where = Some(TypedWhereClause::Expression {
                    condition: condition.clone(),
                    statement: statement.clone(),
                });
            }
            LoweredStatementPayload::NameList { names } => match statement.kind.as_str() {
                "KEEP" => keep_vars.extend(names.iter().map(|name| name.name.clone())),
                "DROP" => drop_vars.extend(names.iter().map(|name| name.name.clone())),
                "BY" => by_vars.extend(names.iter().map(|name| name.name.clone())),
                "RETAIN" => retain_vars.extend(names.iter().map(|name| name.name.clone())),
                _ => {}
            },
            LoweredStatementPayload::Rename {
                rename_map: typed_rename_map,
            } if statement.kind == "RENAME" => {
                rename_map = typed_rename_map.clone();
            }
            LoweredStatementPayload::Deferred { descriptor } if statement.kind == "WHERE" => {
                typed_where = Some(TypedWhereClause::Deferred(descriptor.clone()));
            }
            LoweredStatementPayload::Data { .. }
            | LoweredStatementPayload::Run
            | LoweredStatementPayload::Assign { .. }
            | LoweredStatementPayload::Sum { .. }
            | LoweredStatementPayload::Label
            | LoweredStatementPayload::Output { .. }
            | LoweredStatementPayload::Deferred { .. } => {}
            _ => {}
        }
    }

    if set_input_names.is_empty() {
        diagnostics.append(diag(
            py,
            "RUNTIME_SET_DATASET_NOT_FOUND",
            "statement:set",
            "SET statement requires at least one input dataset.",
        )?)?;
        result.set_item("diagnostics", diagnostics)?;
        return Ok(result.unbind());
    }

    all_in_vars.sort();
    all_in_vars.dedup();

    let mut excluded_output_vars: HashSet<String> = HashSet::new();
    for in_var in &all_in_vars {
        excluded_output_vars.insert(in_var.clone());
    }
    if let Some(name) = &indsname_var {
        excluded_output_vars.insert(name.clone());
    }
    if let Some(name) = &end_var {
        excluded_output_vars.insert(name.clone());
    }
    for by_var in &by_vars {
        excluded_output_vars.insert(format!("first.{}", by_var));
        excluded_output_vars.insert(format!("FIRST.{}", by_var));
        excluded_output_vars.insert(format!("last.{}", by_var));
        excluded_output_vars.insert(format!("LAST.{}", by_var));
    }

    let targets = typed_program.output_targets.clone();

    let preparation_kind = parse_preparation_kind(payload)?;

    let execution_plan = payload.get_item("execution_plan")?;
    let mut planned_source_slot_order: Vec<String> = Vec::new();
    if preparation_kind != PreparationKind::PreparedMerge {
        if let Some(plan_any) = execution_plan.as_ref() {
            if let Some(row_loop_plan_any) =
                get_plan_item(plan_any, "row_loop_plan").map_err(PyValueError::new_err)?
            {
                planned_source_slot_order =
                    get_plan_text_list(&row_loop_plan_any, "source_slot_order")
                        .map_err(PyValueError::new_err)?;
                if let Some(reason_code) = get_plan_text(&row_loop_plan_any, "unsupported_reason")
                    .map_err(PyValueError::new_err)?
                {
                    if !reason_code.trim().is_empty() {
                        diagnostics.append(diag(
                        py,
                        &reason_code,
                        "row_loop_plan",
                        "Execution plan indicates row-loop unsupported condition for native runtime.",
                    )?)?;
                        result.set_item("diagnostics", diagnostics)?;
                        return Ok(result.unbind());
                    }
                }

                let mode =
                    get_plan_text(&row_loop_plan_any, "mode").map_err(PyValueError::new_err)?;
                let requested_mode = get_plan_text(&row_loop_plan_any, "requested_mode")
                    .map_err(PyValueError::new_err)?;
                let strategy =
                    get_plan_text(&row_loop_plan_any, "strategy").map_err(PyValueError::new_err)?;
                let engine =
                    get_plan_text(&row_loop_plan_any, "engine").map_err(PyValueError::new_err)?;
                let reason_codes = get_plan_text_list(&row_loop_plan_any, "reason_codes")
                    .map_err(PyValueError::new_err)?;
                let where_mode = get_plan_text(&row_loop_plan_any, "where_mode")
                    .map_err(PyValueError::new_err)?;
                let rewrite_mode = get_plan_text(&row_loop_plan_any, "rewrite_mode")
                    .map_err(PyValueError::new_err)?;
                let cursor_kind = get_plan_text(&row_loop_plan_any, "cursor_kind")
                    .map_err(PyValueError::new_err)?;
                let builder_mode = get_plan_text(&row_loop_plan_any, "builder_mode")
                    .map_err(PyValueError::new_err)?;
                let output_mode = get_plan_text(&row_loop_plan_any, "output_mode")
                    .map_err(PyValueError::new_err)?;
                let materialization_policy =
                    get_plan_text(&row_loop_plan_any, "materialization_policy")
                        .map_err(PyValueError::new_err)?;

                let expected_reason = match requested_mode.as_deref() {
                    Some("auto") => Some("ROW_ONLY_STRATEGY"),
                    Some("rust") => Some("EXPLICIT_RUST_ROW"),
                    Some("python") => Some("LEGACY_PYTHON_BACKEND_ALIAS"),
                    _ => None,
                };
                let reason_codes_valid = expected_reason
                    .map(|reason| reason_codes == vec![reason.to_string()])
                    .unwrap_or(false);
                if mode.as_deref() != Some("arrow_row_loop")
                    || expected_reason.is_none()
                    || strategy.as_deref() != Some("row")
                    || engine.as_deref() != Some("rust")
                    || !reason_codes_valid
                    || !matches!(
                        where_mode.as_deref(),
                        Some("none") | Some("shared_pre_row_filter")
                    )
                    || !matches!(
                        rewrite_mode.as_deref(),
                        Some("planner_owned_pending") | Some("planner_owned_active")
                    )
                    || cursor_kind.as_deref() != Some("arrow_row_cursor")
                    || builder_mode.as_deref() != Some("targeted_output_handoff")
                    || output_mode.as_deref() != Some("targeted_output_handoff")
                    || materialization_policy.as_deref() != Some("arrow_cursor")
                {
                    diagnostics.append(diag(
                    py,
                    "ROW_LOOP_UNSUPPORTED_PLAN",
                    "row_loop_plan",
                    "Native runtime requires the strategy-aware arrow_row_loop/targeted_output_handoff contract.",
                )?)?;
                    result.set_item("diagnostics", diagnostics)?;
                    return Ok(result.unbind());
                }
            }
        }
    }

    let input_streams: Py<PyDict> = payload
        .get_item("input_streams")?
        .ok_or_else(|| PyValueError::new_err("payload.input_streams is required"))?
        .cast::<PyDict>()
        .map_err(|_| PyValueError::new_err("payload.input_streams must be a dict"))?
        .clone()
        .unbind();
    let apply_registry: HashMap<String, Py<PyAny>> =
        if let Some(apply_registry_any) = payload.get_item("apply_registry")? {
            let apply_registry_dict = apply_registry_any
                .cast::<PyDict>()
                .map_err(|_| PyValueError::new_err("payload.apply_registry must be a dict"))?;
            let mut resolved = HashMap::new();
            for (key_any, value_any) in apply_registry_dict.iter() {
                let key = key_any.extract::<String>()?;
                resolved.insert(key, value_any.unbind());
            }
            resolved
        } else {
            HashMap::new()
        };
    let (numeric_format_catalogs, character_format_catalogs, informat_catalogs) =
        if let Some(format_catalog_payload_any) = payload.get_item("format_catalog_payload")? {
            let format_catalog_payload =
                format_catalog_payload_any.cast::<PyDict>().map_err(|_| {
                    PyValueError::new_err("payload.format_catalog_payload must be a dict")
                })?;
            let informats_any = format_catalog_payload.get_item("informats")?;
            let numeric_formats_any = if let Some(formats_any) =
                format_catalog_payload.get_item("formats")?
            {
                let formats_dict = formats_any.cast::<PyDict>().map_err(|_| {
                    PyValueError::new_err("payload.format_catalog_payload.formats must be a dict")
                })?;
                formats_dict.get_item("numeric")?
            } else {
                None
            };
            let character_formats_any = if let Some(formats_any) =
                format_catalog_payload.get_item("formats")?
            {
                let formats_dict = formats_any.cast::<PyDict>().map_err(|_| {
                    PyValueError::new_err("payload.format_catalog_payload.formats must be a dict")
                })?;
                formats_dict.get_item("character")?
            } else {
                None
            };
            (
                parse_text_catalog_group(numeric_formats_any.as_ref())?,
                parse_text_catalog_group(character_formats_any.as_ref())?,
                parse_numeric_catalog_group(informats_any.as_ref())?,
            )
        } else {
            (HashMap::new(), HashMap::new(), HashMap::new())
        };
    let load_rows_for_input =
        |input_name: &str| -> Result<Option<Rc<NativeArrowRowCursor>>, String> {
            let streams = input_streams.bind(py);
            let Some(stream_any) = streams
                .get_item(input_name)
                .map_err(|error| format!("input_streams lookup failed: {error}"))?
            else {
                return Ok(None);
            };
            NativeArrowRowCursor::from_arrow_stream(py, &stream_any)
                .map(Rc::new)
                .map(Some)
        };

    // A prepared SET carrier is a single Arrow stream under the first source
    // name; Python has already concatenated the remaining logical sources into
    // that stream. Every other preparation kind must carry every SET stream.
    // Validate the contract before row execution so a malformed payload cannot
    // silently produce a partial output.
    let required_set_input_names: Vec<String> = match preparation_kind {
        PreparationKind::PreparedSet => set_input_names.first().cloned().into_iter().collect(),
        PreparationKind::None | PreparationKind::PreparedMerge => set_input_names.clone(),
    };
    {
        let streams = input_streams.bind(py);
        for input_name in &required_set_input_names {
            if streams.get_item(input_name)?.is_none() {
                diagnostics.append(diag(
                    py,
                    "RUNTIME_SET_INPUT_STREAM_MISSING",
                    &format!("dataset:{input_name}"),
                    &format!(
                        "Rust native runtime requires an input stream for SET source '{}'.",
                        input_name
                    ),
                )?)?;
                result.set_item("diagnostics", diagnostics)?;
                return Ok(result.unbind());
            }
        }
    }

    let mut loaded_set_inputs: Vec<(String, Rc<NativeArrowRowCursor>)> = Vec::new();
    for input_name in &set_input_names {
        let row_cursor = match load_rows_for_input(input_name) {
            Ok(Some(rows)) => rows,
            Ok(None) => {
                // Prepared SET intentionally omits suffix streams, which were
                // folded into the first stream above. Any other omission is a
                // malformed transport payload and must fail closed.
                if preparation_kind == PreparationKind::PreparedSet
                    && required_set_input_names.first() != Some(input_name)
                {
                    continue;
                }
                diagnostics.append(diag(
                    py,
                    "RUNTIME_SET_INPUT_STREAM_MISSING",
                    &format!("dataset:{input_name}"),
                    &format!(
                        "Rust native runtime requires an input stream for SET source '{}'.",
                        input_name
                    ),
                )?)?;
                result.set_item("diagnostics", diagnostics)?;
                return Ok(result.unbind());
            }
            Err(error) => {
                diagnostics.append(diag(
                    py,
                    "RUNTIME_RUST_BRIDGE_ARROW_EXPORT_FAILED",
                    &format!("dataset:{}", input_name),
                    &format!("Rust native runtime failed to read input rows: {}", error),
                )?)?;
                result.set_item("diagnostics", diagnostics)?;
                return Ok(result.unbind());
            }
        };
        if let Err(error) = row_cursor.validate_source_slot_order(&planned_source_slot_order) {
            diagnostics.append(diag(
                py,
                "ROW_LOOP_UNSUPPORTED_PLAN",
                &format!("dataset:{}", input_name),
                &format!(
                    "Rust row cursor failed source slot order validation: {}",
                    error
                ),
            )?)?;
            result.set_item("diagnostics", diagnostics)?;
            return Ok(result.unbind());
        }
        loaded_set_inputs.push((input_name.clone(), row_cursor));
    }

    if preparation_kind == PreparationKind::PreparedMerge && loaded_set_inputs.len() > 1 {
        diagnostics.append(diag(
            py,
            "RUNTIME_PREPARED_MERGE_MULTIPLE_SOURCES",
            "statement:set",
            "Prepared MERGE requires exactly one loaded SET source carrier.",
        )?)?;
        result.set_item("diagnostics", diagnostics)?;
        return Ok(result.unbind());
    }

    // Keep the source registry alive through output export while preserving the
    // existing state/output cursor interface. Prepared MERGE is one carrier
    // stream and therefore receives one registry source entry.
    let mut source_registry = ExecutionSourceRegistry::new();
    let mut registered_set_inputs = Vec::with_capacity(loaded_set_inputs.len());
    for (input_name, row_cursor) in &loaded_set_inputs {
        let source_id = if preparation_kind == PreparationKind::PreparedMerge {
            match source_registry.register_prepared_merge(Rc::clone(row_cursor)) {
                Ok(source_id) => source_id,
                Err(error) => {
                    diagnostics.append(diag(
                        py,
                        "RUNTIME_SOURCE_REGISTRY_ERROR",
                        &format!("dataset:{input_name}"),
                        &format!("Rust source registry rejected prepared MERGE: {error}"),
                    )?)?;
                    result.set_item("diagnostics", diagnostics)?;
                    return Ok(result.unbind());
                }
            }
        } else {
            source_registry.register(Rc::clone(row_cursor))
        };
        if let Err(error) = row_cursor.assign_source_id(source_id) {
            diagnostics.append(diag(
                py,
                "RUNTIME_SOURCE_REGISTRY_ERROR",
                &format!("dataset:{input_name}"),
                &format!("Rust source registry could not assign source id: {error}"),
            )?)?;
            result.set_item("diagnostics", diagnostics)?;
            return Ok(result.unbind());
        }
        let registered_cursor = match source_registry.lookup(source_id) {
            Ok(handle) => handle.cursor(),
            Err(error) => {
                diagnostics.append(diag(
                    py,
                    "RUNTIME_SOURCE_REGISTRY_ERROR",
                    &format!("dataset:{input_name}"),
                    &format!("Rust source registry lookup failed: {error}"),
                )?)?;
                result.set_item("diagnostics", diagnostics)?;
                return Ok(result.unbind());
            }
        };
        registered_set_inputs.push((input_name.clone(), registered_cursor));
    }
    let source_registry = Rc::new(source_registry);
    for (input_name, row_cursor) in &registered_set_inputs {
        if let Err(error) = row_cursor.assign_source_registry(Rc::downgrade(&source_registry)) {
            diagnostics.append(diag(
                py,
                "RUNTIME_SOURCE_REGISTRY_ERROR",
                &format!("dataset:{input_name}"),
                &format!("Rust source registry handle assignment failed: {error}"),
            )?)?;
            result.set_item("diagnostics", diagnostics)?;
            return Ok(result.unbind());
        }
    }

    let mut output_builders = match initialize_output_builders(
        execution_plan.as_ref(),
        &targets,
        Rc::clone(&source_registry),
    ) {
        Ok(value) => value,
        Err(error) => {
            diagnostics.append(diag(
                py,
                "ROW_LOOP_UNSUPPORTED_PLAN",
                "output_handoff_plan",
                &format!("Rust output handoff plan resolution failed: {}", error),
            )?)?;
            result.set_item("diagnostics", diagnostics)?;
            return Ok(result.unbind());
        }
    };

    let has_explicit_output_statement = typed_program.suppress_implicit_output;
    let default_target = typed_program.default_target.clone();

    let mut eval_state = EvalRuntimeState {
        source_registry: Rc::clone(&source_registry),
        apply_registry,
        numeric_format_catalogs,
        character_format_catalogs,
        informat_catalogs,
        ..Default::default()
    };
    let mutable_column_names = collect_typed_mutable_column_names(&typed_program.statements);
    let mut global_row_index: i64 = 0;
    let mut stop_execution = false;

    let set_options_are_passthrough = registered_set_inputs.iter().all(|(input_name, _)| {
        set_options_by_dataset
            .get(&input_name.to_lowercase())
            .map(dataset_ref_options_are_passthrough)
            .unwrap_or(true)
    });
    if !set_options_are_passthrough {
        diagnostics.append(diag(
            py,
            "RUNTIME_PREPARED_INPUT_OPTION_REMAINING",
            "statement:set",
            "Input dataset options must be consumed by Arrow preparation before native row execution.",
        )?)?;
        result.set_item("diagnostics", diagnostics)?;
        return Ok(result.unbind());
    }

    // Check every source and target once, including streams that emit no rows.
    // Output events handle any later runtime-owned columns during append.
    for (_, row_cursor) in &registered_set_inputs {
        let mut schema_view = build_logical_row_schema_view(&source_registry, row_cursor)
            .map_err(PyValueError::new_err)?;
        for name in &mutable_column_names {
            schema_view
                .ensure_owned_column(name)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
        }
        for target in &targets {
            let empty_target_rename = HashMap::new();
            let target_options = output_options_by_target.get(&target.to_lowercase());
            let (target_keep, target_drop, target_rename) = target_options
                .map(|options| {
                    (
                        options.keep_vars.as_slice(),
                        options.drop_vars.as_slice(),
                        &options.rename_map,
                    )
                })
                .unwrap_or((&[], &[], &empty_target_rename));
            ensure_output_target_shape_from_view(
                &mut output_builders,
                target,
                &schema_view,
                &keep_vars,
                &drop_vars,
                &rename_map,
                target_keep,
                target_drop,
                target_rename,
                &excluded_output_vars,
            )
            .map_err(PyValueError::new_err)?;
        }
    }

    let input_import_ms = input_import_start.elapsed().as_secs_f64() * 1000.0;
    let row_execution_start = Instant::now();

    let total_surviving_set_rows = if end_var.is_some() {
        Some(
            registered_set_inputs
                .iter()
                .map(|(_, row_cursor)| row_cursor.row_count())
                .sum::<usize>(),
        )
    } else {
        None
    };

    for (input_name, row_cursor) in &registered_set_inputs {
        let row_count = row_cursor.row_count();
        eval_state.source_cursor = Some(Rc::clone(row_cursor));
        eval_state.materialize_mutable_values = false;

        let mut logical_row = build_logical_row_schema_view(&source_registry, row_cursor)
            .map_err(PyValueError::new_err)?;
        for name in &mutable_column_names {
            logical_row
                .ensure_owned_column(name)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
        }
        let schema_column_count = logical_row.column_count();
        eval_state.logical_row = Some(logical_row);

        for row_index in 0..row_count {
            eval_state.row_index = row_index as isize;
            eval_state.logical_row.as_mut()
                .expect("source-local logical row is initialized")
                .reset_row_layout(row_index, schema_column_count);
            let working_row_obj = PyDict::new(py).unbind();
            let working_row = working_row_obj.bind(py);

            global_row_index += 1;
            reset_automatic_scalars(&mut eval_state, global_row_index)
                .map_err(PyValueError::new_err)?;

            for in_var in &all_in_vars {
                set_runtime_flag(&mut eval_state, in_var, false).map_err(PyValueError::new_err)?;
            }
            if let Some(in_var_name) = set_in_var_by_dataset.get(&input_name.to_lowercase()) {
                set_runtime_flag(&mut eval_state, in_var_name, true)
                    .map_err(PyValueError::new_err)?;
            }
            if let Some(name) = &indsname_var {
                set_runtime_text(&mut eval_state, name, input_name.clone())
                    .map_err(PyValueError::new_err)?;
            }
            if let Some(name) = &end_var {
                let is_last = total_surviving_set_rows
                    .map(|total| total > 0 && global_row_index as usize == total)
                    .unwrap_or(false);
                set_runtime_flag(&mut eval_state, name, is_last).map_err(PyValueError::new_err)?;
            }

            for retained_name in &retain_vars {
                let current_value = get_stateful_owned_value(&eval_state, retained_name)
                    .map_err(PyValueError::new_err)?;
                if current_value.is_none()
                    || matches!(
                        current_value.as_ref(),
                        Some(OwnedValue::Scalar(ScalarValue::Null))
                    )
                {
                    if let Some(value) = eval_state.retain_values.get(retained_name).cloned() {
                        let scalar = value.as_scalar().map_err(|error| {
                            PyValueError::new_err(format!(
                                "RETAIN value for column '{retained_name}' is not scalar: {error}"
                            ))
                        })?;
                        store_runtime_scalar(&working_row, &mut eval_state, retained_name, scalar)
                            .map_err(PyValueError::new_err)?;
                    }
                }
            }

            if !by_vars.is_empty() {
                let prev_index = if row_index > 0 {
                    Some(row_index - 1)
                } else {
                    None
                };
                let next_index = if row_index + 1 < row_count {
                    Some(row_index + 1)
                } else {
                    None
                };
                for by_var in &by_vars {
                    let (is_first, is_last) = {
                        let logical_row = eval_state.logical_row.as_ref().ok_or_else(|| {
                            PyValueError::new_err("logical row view is required for BY evaluation")
                        })?;
                        let current_value = logical_row
                            .resolve(by_var)
                            .map_err(|error| PyValueError::new_err(error.to_string()))?;
                        let is_first = if let Some(prev_idx) = prev_index {
                            let previous = logical_row
                                .for_row_index(prev_idx)
                                .map_err(|error| PyValueError::new_err(error.to_string()))?;
                            let previous_value = previous
                                .resolve(by_var)
                                .map_err(|error| PyValueError::new_err(error.to_string()))?;
                            !optional_logical_values_equal(
                                current_value.clone(),
                                previous_value,
                                by_var,
                                "BY",
                            )
                            .map_err(PyValueError::new_err)?
                        } else {
                            true
                        };
                        let is_last = if let Some(next_idx) = next_index {
                            let next = logical_row
                                .for_row_index(next_idx)
                                .map_err(|error| PyValueError::new_err(error.to_string()))?;
                            let next_value = next
                                .resolve(by_var)
                                .map_err(|error| PyValueError::new_err(error.to_string()))?;
                            !optional_logical_values_equal(current_value, next_value, by_var, "BY")
                                .map_err(PyValueError::new_err)?
                        } else {
                            true
                        };
                        (is_first, is_last)
                    };
                    set_runtime_flag(&mut eval_state, &format!("first.{}", by_var), is_first)
                        .map_err(PyValueError::new_err)?;
                    set_runtime_flag(&mut eval_state, &format!("FIRST.{}", by_var), is_first)
                        .map_err(PyValueError::new_err)?;
                    set_runtime_flag(&mut eval_state, &format!("last.{}", by_var), is_last)
                        .map_err(PyValueError::new_err)?;
                    set_runtime_flag(&mut eval_state, &format!("LAST.{}", by_var), is_last)
                        .map_err(PyValueError::new_err)?;
                }
            }

            let typed_where_pass = match &typed_where {
                Some(TypedWhereClause::Expression {
                    condition,
                    statement,
                }) => {
                    let pass = match expressions::evaluate_typed_condition(
                        &working_row,
                        &condition.expression,
                        &mut eval_state,
                    ) {
                        Ok(value) => value,
                        Err(error) => {
                            mark_automatic_error(&mut eval_state).map_err(PyValueError::new_err)?;
                            let display = typed_expression_display(condition, statement);
                            let summary = match &condition.source {
                                typed_ast::TypedExpression::Deferred {
                                    where_syntax_summary,
                                    ..
                                } => where_syntax_summary.as_str(),
                                _ => "",
                            };
                            let (code, message) =
                                typed_condition_error_diagnostic(&display, &error, summary);
                            diagnostics.append(diag(py, code, "where", &message)?)?;
                            result.set_item("diagnostics", diagnostics)?;
                            return Ok(result.unbind());
                        }
                    };
                    Some(pass)
                }
                Some(TypedWhereClause::Deferred(descriptor)) => {
                    mark_automatic_error(&mut eval_state).map_err(PyValueError::new_err)?;
                    let (code, message) = typed_condition_error_diagnostic(
                        &descriptor.display_text,
                        &descriptor.native_cause,
                        &descriptor.where_syntax_summary,
                    );
                    diagnostics.append(diag(py, code, "where", &message)?)?;
                    result.set_item("diagnostics", diagnostics)?;
                    return Ok(result.unbind());
                }
                None => None,
            };
            if typed_where_pass == Some(false) {
                continue;
            }

            let mut output_router = |_source_row: &Bound<'_, PyDict>,
                                     target_opt: Option<&str>,
                                     state: &mut EvalRuntimeState|
             -> Result<(), String> {
                let requested_target = match target_opt {
                    Some(target) => target,
                    None => default_target
                        .as_deref()
                        .ok_or_else(|| "No default output target is declared".to_string())?,
                };
                let resolved_target = resolve_declared_target_name(requested_target, &targets);
                if !targets.contains(&resolved_target) {
                    return Err(format!("Output target is not declared: {requested_target}"));
                }

                let empty_target_rename = HashMap::new();
                let target_options = output_options_by_target.get(&resolved_target.to_lowercase());
                if let Some(options) = target_options {
                    if !output_target_passes_options(
                        &resolved_target,
                        options,
                        state,
                        &keep_vars,
                        &drop_vars,
                        &rename_map,
                    )
                    .map_err(|error| {
                        dataset_option_error_message(
                            &format!("output:{}", resolved_target.to_lowercase()),
                            &error,
                        )
                    })? {
                        return Ok(());
                    }
                }
                let (target_keep, target_drop, target_rename) = target_options
                    .map(|options| {
                        (
                            options.keep_vars.as_slice(),
                            options.drop_vars.as_slice(),
                            &options.rename_map,
                        )
                    })
                    .unwrap_or((&[], &[], &empty_target_rename));
                append_projected_row_to_target(
                    &mut output_builders,
                    &resolved_target,
                    state,
                    &keep_vars,
                    &drop_vars,
                    &rename_map,
                    target_keep,
                    target_drop,
                    target_rename,
                    &excluded_output_vars,
                )?;

                Ok(())
            };

            let execution_result = execute_typed_statement_block(
                &typed_program.statements,
                &working_row,
                &mut eval_state,
                &mut output_router,
            );
            let outcome = match execution_result {
                Ok(outcome) => outcome,
                Err(error) => {
                    mark_automatic_error(&mut eval_state).map_err(PyValueError::new_err)?;
                    diagnostics.append(diag(
                        py,
                        "RUNTIME_EXPRESSION_EVALUATION_ERROR",
                        "statement",
                        &format!("Rust statement execution failed: {}", error),
                    )?)?;
                    result.set_item("diagnostics", diagnostics)?;
                    return Ok(result.unbind());
                }
            };

            for retained_name in &retain_vars {
                if let Some(value) = get_stateful_owned_value(&eval_state, retained_name)
                    .map_err(PyValueError::new_err)?
                {
                    if value.as_scalar().is_err() {
                        return Err(PyValueError::new_err(format!(
                            "RETAIN value for column '{retained_name}' must be scalar"
                        )));
                    }
                    eval_state
                        .retain_values
                        .insert(retained_name.clone(), value);
                }
            }

            if !outcome.deleted && !has_explicit_output_statement {
                if let Err(error) =
                    output_router(&working_row, default_target.as_deref(), &mut eval_state)
                {
                    diagnostics.append(diag(
                        py,
                        "RUNTIME_EXPRESSION_EVALUATION_ERROR",
                        "output",
                        &format!("Rust output projection failed: {}", error),
                    )?)?;
                    result.set_item("diagnostics", diagnostics)?;
                    return Ok(result.unbind());
                }
            }

            if outcome.stopped {
                stop_execution = true;
                break;
            }
        }

        if stop_execution {
            break;
        }
    }

    let row_execution_ms = row_execution_start.elapsed().as_secs_f64() * 1000.0;

    let output_export_start = Instant::now();
    let output_streams = match export_output_streams(py, &output_builders) {
        Ok(value) => value,
        Err(error) => {
            diagnostics.append(diag(
                py,
                "RUNTIME_RUST_NATIVE_EXECUTION_FAILED",
                "output_streams",
                &format!(
                    "Rust native runtime failed to export output streams: {}",
                    error
                ),
            )?)?;
            result.set_item("diagnostics", diagnostics)?;
            return Ok(result.unbind());
        }
    };
    let output_export_ms = output_export_start.elapsed().as_secs_f64() * 1000.0;

    let phase_metrics = PyDict::new(py);
    phase_metrics.set_item("input_import_ms", input_import_ms)?;
    phase_metrics.set_item("row_execution_ms", row_execution_ms)?;
    phase_metrics.set_item("output_export_ms", output_export_ms)?;
    result.set_item("phase_metrics", phase_metrics)?;
    result.set_item("output_streams", output_streams)?;
    result.set_item("diagnostics", diagnostics)?;
    Ok(result.unbind())
}

fn invalid_ast_payload_result(
    py: Python<'_>,
    result: &Bound<'_, PyDict>,
    diagnostics: &Bound<'_, PyList>,
    error: &str,
) -> PyResult<Py<PyDict>> {
    diagnostics.append(diag(
        py,
        "PARSE_RUST_AST_DESERIALIZE_FAILED",
        "ast",
        &format!("Rust native parser/runtime failed to deserialize AST payload: {error}"),
    )?)?;
    result.set_item("diagnostics", diagnostics)?;
    Ok(result.clone().unbind())
}

fn collect_typed_mutable_column_names(statements: &[LoweredStatement]) -> Vec<String> {
    fn collect(
        statements: &[LoweredStatement],
        names: &mut Vec<String>,
        seen: &mut HashSet<String>,
    ) {
        for statement in statements {
            match &statement.payload {
                LoweredStatementPayload::Assign { target, .. }
                | LoweredStatementPayload::Sum { target, .. } => {
                    if let LoweredAssignmentTarget::Name { name, .. } = target {
                        let trimmed = name.trim();
                        if !trimmed.is_empty() && seen.insert(expressions::column_key(trimmed)) {
                            names.push(trimmed.to_string());
                        }
                    }
                }
                LoweredStatementPayload::NameList {
                    names: retained_names,
                } if statement.kind == "RETAIN" => {
                    for name in retained_names {
                        let trimmed = name.name.trim();
                        if !trimmed.is_empty() && seen.insert(expressions::column_key(trimmed)) {
                            names.push(trimmed.to_string());
                        }
                    }
                }
                LoweredStatementPayload::Do {
                    loop_var: Some(name),
                    ..
                } => {
                    let trimmed = name.trim();
                    if !trimmed.is_empty() && seen.insert(expressions::column_key(trimmed)) {
                        names.push(trimmed.to_string());
                    }
                }
                LoweredStatementPayload::Array { declaration } if statement.kind == "ARRAY" => {
                    for name in &declaration.variables {
                        let trimmed = name.trim();
                        if !trimmed.is_empty() && seen.insert(expressions::column_key(trimmed)) {
                            names.push(trimmed.to_string());
                        }
                    }
                }
                LoweredStatementPayload::If { then_action, .. } => {
                    if let Some(typed_lowering::LoweredInlineAction::Statement(nested)) =
                        then_action
                    {
                        collect(std::slice::from_ref(nested), names, seen);
                    }
                }
                LoweredStatementPayload::Else { action } => {
                    if let Some(typed_lowering::LoweredInlineAction::Statement(nested)) = action {
                        collect(std::slice::from_ref(nested), names, seen);
                    }
                }
                _ => {}
            }
        }
    }

    let mut names = Vec::new();
    let mut seen = HashSet::new();
    collect(statements, &mut names, &mut seen);
    names
}

fn execute_typed_statement_block(
    statements: &[LoweredStatement],
    row: &Bound<'_, PyDict>,
    state: &mut EvalRuntimeState,
    output_router: &mut impl FnMut(
        &Bound<'_, PyDict>,
        Option<&str>,
        &mut EvalRuntimeState,
    ) -> Result<(), String>,
) -> Result<RowExecutionOutcome, String> {
    execute_typed_statement_range(statements, 0, statements.len(), row, state, output_router)
        .map(|(_, outcome)| outcome)
}

fn execute_typed_statement_range(
    statements: &[LoweredStatement],
    start: usize,
    stop: usize,
    row: &Bound<'_, PyDict>,
    state: &mut EvalRuntimeState,
    output_router: &mut impl FnMut(
        &Bound<'_, PyDict>,
        Option<&str>,
        &mut EvalRuntimeState,
    ) -> Result<(), String>,
) -> Result<(usize, RowExecutionOutcome), String> {
    let mut cursor = start;
    let mut outcome = RowExecutionOutcome::default();
    while cursor < stop {
        let statement = &statements[cursor];
        match &statement.payload {
            LoweredStatementPayload::Data { .. }
            | LoweredStatementPayload::Source { .. }
            | LoweredStatementPayload::Where { .. }
            | LoweredStatementPayload::NameList { .. }
            | LoweredStatementPayload::Label
            | LoweredStatementPayload::Rename { .. }
            | LoweredStatementPayload::Run => {}
            LoweredStatementPayload::Skipped if statement.kind == "SKIPPED" => {}
            LoweredStatementPayload::Assign { target, value } if statement.kind == "ASSIGN" => {
                execute_typed_assignment(statement, target, value, row, state, true)?;
            }
            LoweredStatementPayload::Sum { target, increment } if statement.kind == "SUM" => {
                execute_typed_sum(target, increment, row, state)?;
            }
            LoweredStatementPayload::Deferred { descriptor } if statement.kind == "ASSIGN" => {
                return Err(format!(
                    "assignment expression failed: '{}': {}",
                    descriptor.display_text, descriptor.native_cause
                ));
            }
            LoweredStatementPayload::Deferred { descriptor } if statement.kind == "SUM" => {
                return Err(descriptor.native_cause.clone());
            }
            LoweredStatementPayload::Deferred { descriptor } if statement.kind == "IF" => {
                return Err(descriptor.native_cause.clone());
            }
            LoweredStatementPayload::Deferred { descriptor } if statement.kind == "DO" => {
                find_typed_matching_end(statements, cursor, stop)
                    .ok_or_else(|| "DO statement missing matching END".to_string())?;
                return Err(descriptor.native_cause.clone());
            }
            LoweredStatementPayload::Deferred { descriptor } if statement.kind == "ARRAY" => {
                return Err(descriptor.native_cause.clone());
            }
            LoweredStatementPayload::Array { declaration } if statement.kind == "ARRAY" => {
                state.array_defs.insert(
                    declaration.array_name.clone(),
                    declaration.variables.clone(),
                );
            }
            LoweredStatementPayload::Do {
                loop_var,
                start,
                end: do_end,
            } if statement.kind == "DO" => {
                let end_index = find_typed_matching_end(statements, cursor, stop)
                    .ok_or_else(|| "DO statement missing matching END".to_string())?;
                let loop_spec = (loop_var.as_deref(), start.as_ref(), do_end.as_ref());
                let nested = match loop_spec {
                    (Some(loop_var), Some(start), Some(do_end)) => execute_typed_do_range(
                        loop_var,
                        start,
                        do_end,
                        statements,
                        cursor + 1,
                        end_index,
                        row,
                        state,
                        output_router,
                    )?,
                    (None, None, None) => {
                        return Err(format!(
                            "unsupported DO statement: {}",
                            statement.display_text
                        ));
                    }
                    _ => return Err("invalid typed DO loop specification".to_string()),
                };
                merge_typed_outcome(&mut outcome, nested);
                // The shared range tail advances once after each statement.
                // Leave the cursor on END so the next iteration resumes at
                // the first statement after the matched block.
                cursor = end_index;
            }
            LoweredStatementPayload::If {
                condition,
                then_action,
                is_subset,
                is_then_do,
            } if matches!(statement.kind.as_str(), "IF" | "ELSE IF") => {
                if *is_subset && then_action.is_none() && !*is_then_do {
                    let matched = evaluate_typed_condition_value(row, condition, state)?;
                    cursor += 1;
                    if !matched {
                        outcome.deleted = true;
                        return Ok((cursor, outcome));
                    }
                    continue;
                }

                let matched = evaluate_typed_condition_value(row, condition, state)?;
                let mut chain_matched = false;
                if matched {
                    chain_matched = true;
                    if *is_then_do {
                        let end = find_typed_matching_end(statements, cursor, stop)
                            .ok_or_else(|| "IF THEN DO block missing matching END".to_string())?;
                        let nested = execute_typed_do_action(
                            then_action.as_ref(),
                            statements,
                            cursor + 1,
                            end,
                            row,
                            state,
                            output_router,
                        )?;
                        merge_typed_outcome(&mut outcome, nested);
                        cursor = end + 1;
                    } else {
                        if let Some(action) = then_action {
                            execute_typed_inline_action(
                                action,
                                row,
                                state,
                                &mut outcome,
                                output_router,
                            )?;
                        }
                        cursor += 1;
                    }
                } else {
                    cursor += 1;
                    if *is_then_do {
                        let end = find_typed_matching_end(statements, cursor - 1, stop)
                            .ok_or_else(|| "IF THEN DO block missing matching END".to_string())?;
                        cursor = end + 1;
                    }
                }

                while cursor < stop
                    && matches!(statements[cursor].kind.as_str(), "ELSE IF" | "ELSE")
                {
                    let branch = &statements[cursor];
                    match &branch.payload {
                        LoweredStatementPayload::If {
                            condition,
                            then_action,
                            is_then_do,
                            ..
                        } if branch.kind == "ELSE IF" => {
                            if !chain_matched
                                && evaluate_typed_condition_value(row, condition, state)?
                            {
                                chain_matched = true;
                                if *is_then_do {
                                    let end = find_typed_matching_end(statements, cursor, stop)
                                        .ok_or_else(|| {
                                            "ELSE IF THEN DO block missing matching END".to_string()
                                        })?;
                                    let nested = execute_typed_do_action(
                                        then_action.as_ref(),
                                        statements,
                                        cursor + 1,
                                        end,
                                        row,
                                        state,
                                        output_router,
                                    )?;
                                    merge_typed_outcome(&mut outcome, nested);
                                    cursor = end + 1;
                                } else {
                                    if let Some(action) = then_action {
                                        execute_typed_inline_action(
                                            action,
                                            row,
                                            state,
                                            &mut outcome,
                                            output_router,
                                        )?;
                                    }
                                    cursor += 1;
                                }
                            } else {
                                cursor += 1;
                                if *is_then_do {
                                    let end = find_typed_matching_end(statements, cursor - 1, stop)
                                        .ok_or_else(|| {
                                            "ELSE IF THEN DO block missing matching END".to_string()
                                        })?;
                                    cursor = end + 1;
                                }
                            }
                        }
                        LoweredStatementPayload::Deferred { descriptor }
                            if branch.kind == "ELSE IF" =>
                        {
                            if !chain_matched {
                                return Err(descriptor.native_cause.clone());
                            }
                            cursor += 1;
                        }
                        LoweredStatementPayload::Else { action } if branch.kind == "ELSE" => {
                            let is_else_do = action.as_ref().is_some_and(|action| {
                                matches!(action,
                                    typed_lowering::LoweredInlineAction::Statement(nested)
                                        if nested.kind == "DO")
                            });
                            if !chain_matched {
                                if is_else_do {
                                    let end = find_typed_matching_end(statements, cursor, stop)
                                        .ok_or_else(|| {
                                            "ELSE DO block missing matching END".to_string()
                                        })?;
                                    let nested = execute_typed_do_action(
                                        action.as_ref(),
                                        statements,
                                        cursor + 1,
                                        end,
                                        row,
                                        state,
                                        output_router,
                                    )?;
                                    merge_typed_outcome(&mut outcome, nested);
                                    cursor = end + 1;
                                } else if let Some(action) = action {
                                    execute_typed_inline_action(
                                        action,
                                        row,
                                        state,
                                        &mut outcome,
                                        output_router,
                                    )?;
                                    cursor += 1;
                                } else {
                                    cursor += 1;
                                }
                            } else {
                                cursor += 1;
                                if is_else_do {
                                    let end = find_typed_matching_end(statements, cursor - 1, stop)
                                        .ok_or_else(|| {
                                            "ELSE DO block missing matching END".to_string()
                                        })?;
                                    cursor = end + 1;
                                }
                            }
                            break;
                        }
                        LoweredStatementPayload::Deferred { descriptor }
                            if branch.kind == "ELSE" =>
                        {
                            if !chain_matched {
                                return Err(descriptor.native_cause.clone());
                            }
                            cursor += 1;
                            break;
                        }
                        _ => break,
                    }
                    if outcome.deleted || outcome.stopped {
                        return Ok((cursor, outcome));
                    }
                }
                if outcome.deleted || outcome.stopped {
                    return Ok((cursor, outcome));
                }
                continue;
            }
            LoweredStatementPayload::Else { .. } if statement.kind == "ELSE" => {
                cursor += 1;
            }
            LoweredStatementPayload::If { .. } if statement.kind == "ELSE IF" => {
                cursor += 1;
            }
            LoweredStatementPayload::Deferred { descriptor } if statement.kind == "OUTPUT" => {
                return Err(descriptor.native_cause.clone());
            }
            LoweredStatementPayload::Output { targets } if statement.kind == "OUTPUT" => {
                route_typed_output(targets, row, state, output_router)?;
            }
            LoweredStatementPayload::Delete if statement.kind == "DELETE" => {
                outcome.deleted = true;
                return Ok((cursor + 1, outcome));
            }
            LoweredStatementPayload::Stop if statement.kind == "STOP" => {
                outcome.stopped = true;
                return Ok((cursor + 1, outcome));
            }
            LoweredStatementPayload::End if statement.kind == "END" => {
                return Ok((cursor + 1, outcome));
            }
            LoweredStatementPayload::Deferred { descriptor } => {
                return Err(descriptor.native_cause.clone());
            }
            _ => unreachable!("typed payload lowering validates statement kind and payload"),
        }
        if outcome.deleted || outcome.stopped {
            return Ok((cursor + 1, outcome));
        }
        cursor += 1;
    }
    Ok((cursor, outcome))
}

fn execute_typed_do_action(
    action: Option<&typed_lowering::LoweredInlineAction>,
    statements: &[LoweredStatement],
    body_start: usize,
    body_end: usize,
    row: &Bound<'_, PyDict>,
    state: &mut EvalRuntimeState,
    output_router: &mut impl FnMut(
        &Bound<'_, PyDict>,
        Option<&str>,
        &mut EvalRuntimeState,
    ) -> Result<(), String>,
) -> Result<RowExecutionOutcome, String> {
    let do_statement = match action {
        Some(typed_lowering::LoweredInlineAction::Statement(statement))
            if statement.kind == "DO" =>
        {
            statement.as_ref()
        }
        Some(typed_lowering::LoweredInlineAction::Deferred(descriptor)) => {
            return Err(descriptor.native_cause.clone());
        }
        Some(_) | None => {
            return execute_typed_statement_range(
                statements,
                body_start,
                body_end,
                row,
                state,
                output_router,
            )
            .map(|(_, outcome)| outcome);
        }
    };

    match &do_statement.payload {
        LoweredStatementPayload::Do {
            loop_var: Some(loop_var),
            start: Some(start),
            end: Some(end),
        } => execute_typed_do_range(
            loop_var,
            start,
            end,
            statements,
            body_start,
            body_end,
            row,
            state,
            output_router,
        ),
        LoweredStatementPayload::Do {
            loop_var: None,
            start: None,
            end: None,
        } => execute_typed_statement_range(
            statements,
            body_start,
            body_end,
            row,
            state,
            output_router,
        )
        .map(|(_, outcome)| outcome),
        LoweredStatementPayload::Deferred { descriptor } => Err(descriptor.native_cause.clone()),
        _ => Err("invalid typed DO action".to_string()),
    }
}

fn execute_typed_do_range(
    loop_var: &str,
    start: &LoweredExpression,
    end: &LoweredExpression,
    statements: &[LoweredStatement],
    body_start: usize,
    body_end: usize,
    row: &Bound<'_, PyDict>,
    state: &mut EvalRuntimeState,
    output_router: &mut impl FnMut(
        &Bound<'_, PyDict>,
        Option<&str>,
        &mut EvalRuntimeState,
    ) -> Result<(), String>,
) -> Result<RowExecutionOutcome, String> {
    let start_value = scalar_to_number(&expressions::evaluate_typed_expression(
        row,
        &start.expression,
        state,
    )?)? as i64;
    let end_value = scalar_to_number(&expressions::evaluate_typed_expression(
        row,
        &end.expression,
        state,
    )?)? as i64;
    let mut iteration = start_value;
    while iteration <= end_value {
        store_runtime_scalar(row, state, loop_var, &ScalarValue::Number(iteration as f64))?;
        let (_, nested) = execute_typed_statement_range(
            statements,
            body_start,
            body_end,
            row,
            state,
            output_router,
        )?;
        if nested.deleted || nested.stopped {
            return Ok(nested);
        }
        iteration += 1;
    }
    Ok(RowExecutionOutcome::default())
}

fn evaluate_typed_condition_value(
    row: &Bound<'_, PyDict>,
    condition: &LoweredExpression,
    state: &mut EvalRuntimeState,
) -> Result<bool, String> {
    let value = expressions::evaluate_typed_expression(row, &condition.expression, state)?;
    scalar_to_bool(&value)
}

fn merge_typed_outcome(target: &mut RowExecutionOutcome, source: RowExecutionOutcome) {
    target.deleted |= source.deleted;
    target.stopped |= source.stopped;
}

fn find_typed_matching_end(
    statements: &[LoweredStatement],
    start: usize,
    stop: usize,
) -> Option<usize> {
    let mut depth = 1usize;
    let mut cursor = start + 1;
    while cursor < stop {
        let statement = &statements[cursor];
        let opens_block = matches!(
            &statement.payload,
            LoweredStatementPayload::If {
                is_then_do: true,
                ..
            } if matches!(statement.kind.as_str(), "IF" | "ELSE IF")
        ) || matches!(
            &statement.payload,
            LoweredStatementPayload::Else {
                action: Some(typed_lowering::LoweredInlineAction::Statement(nested))
            } if nested.kind == "DO"
        ) || matches!(
            &statement.payload,
            LoweredStatementPayload::Do { .. } | LoweredStatementPayload::Deferred { .. }
                if statement.kind == "DO"
        );
        if opens_block {
            depth += 1;
        } else if matches!(statement.payload, LoweredStatementPayload::End)
            && statement.kind == "END"
        {
            depth -= 1;
            if depth == 0 {
                return Some(cursor);
            }
        }
        cursor += 1;
    }
    None
}

fn execute_typed_assignment(
    statement: &LoweredStatement,
    target: &LoweredAssignmentTarget,
    value: &LoweredExpression,
    row: &Bound<'_, PyDict>,
    state: &mut EvalRuntimeState,
    wrap_error: bool,
) -> Result<(), String> {
    let value =
        expressions::evaluate_typed_expression(row, &value.expression, state).map_err(|cause| {
            if wrap_error {
                format!(
                    "assignment expression failed: '{}': {cause}",
                    typed_expression_display(value, statement)
                )
            } else {
                cause
            }
        })?;
    match target {
        LoweredAssignmentTarget::Name { name, .. } => {
            store_runtime_scalar(row, state, name, &value)
        }
        LoweredAssignmentTarget::ArrayElement { name, index, .. } => {
            let index_value = scalar_to_number(&expressions::evaluate_typed_expression(
                row,
                &index.expression,
                state,
            )?)?;
            let variable_name = resolve_array_variable_name(state, name, index_value)?;
            store_runtime_scalar(row, state, &variable_name, &value)
        }
    }
}

fn execute_typed_sum(
    target: &LoweredAssignmentTarget,
    increment: &LoweredExpression,
    row: &Bound<'_, PyDict>,
    state: &mut EvalRuntimeState,
) -> Result<(), String> {
    let name = match target {
        LoweredAssignmentTarget::Name { name, .. } => name,
        LoweredAssignmentTarget::ArrayElement { .. } => {
            unreachable!("typed payload validation restricts SUM targets to names")
        }
    };
    let increment = expressions::evaluate_typed_expression(row, &increment.expression, state)?;
    apply_sum_update(row, state, name, increment)
}

fn execute_typed_inline_action(
    action: &typed_lowering::LoweredInlineAction,
    row: &Bound<'_, PyDict>,
    state: &mut EvalRuntimeState,
    outcome: &mut RowExecutionOutcome,
    output_router: &mut impl FnMut(
        &Bound<'_, PyDict>,
        Option<&str>,
        &mut EvalRuntimeState,
    ) -> Result<(), String>,
) -> Result<(), String> {
    match action {
        typed_lowering::LoweredInlineAction::Statement(statement) => match &statement.payload {
            LoweredStatementPayload::Assign { target, value } if statement.kind == "ASSIGN" => {
                execute_typed_assignment(statement, target, value, row, state, false)
            }
            LoweredStatementPayload::Sum { target, increment } if statement.kind == "SUM" => {
                execute_typed_sum(target, increment, row, state)
            }
            LoweredStatementPayload::Deferred { descriptor }
                if matches!(statement.kind.as_str(), "ASSIGN" | "SUM") =>
            {
                Err(descriptor.native_cause.clone())
            }
            LoweredStatementPayload::Output { targets } if statement.kind == "OUTPUT" => {
                route_typed_output(targets, row, state, output_router)
            }
            LoweredStatementPayload::Delete if statement.kind == "DELETE" => {
                outcome.deleted = true;
                Ok(())
            }
            LoweredStatementPayload::Stop if statement.kind == "STOP" => {
                outcome.stopped = true;
                Ok(())
            }
            LoweredStatementPayload::Deferred { descriptor } => {
                Err(descriptor.native_cause.clone())
            }
            _ => Err(format!(
                "unsupported typed inline action kind: {}",
                statement.kind
            )),
        },
        typed_lowering::LoweredInlineAction::Deferred(descriptor) => {
            Err(descriptor.native_cause.clone())
        }
    }
}

fn route_typed_output(
    targets: &[typed_lowering::LoweredDatasetReference],
    row: &Bound<'_, PyDict>,
    state: &mut EvalRuntimeState,
    output_router: &mut impl FnMut(
        &Bound<'_, PyDict>,
        Option<&str>,
        &mut EvalRuntimeState,
    ) -> Result<(), String>,
) -> Result<(), String> {
    if targets.is_empty() {
        return output_router(row, None, state);
    }
    let mut seen = HashSet::new();
    for target in targets {
        let normalized = target.name.trim().to_uppercase();
        let routed_target = normalized.strip_prefix("WORK.").unwrap_or(&normalized);
        if seen.insert(routed_target.to_string()) {
            output_router(row, Some(routed_target), state)?;
        }
    }
    Ok(())
}

fn typed_expression_display(
    expression: &LoweredExpression,
    statement: &LoweredStatement,
) -> String {
    use crate::typed_ast::TypedExpression;

    let (start, end, saved_display) = match &expression.source {
        TypedExpression::Literal { span, .. }
        | TypedExpression::Name { span, .. }
        | TypedExpression::Unary { span, .. }
        | TypedExpression::Binary { span, .. }
        | TypedExpression::Call { span, .. }
        | TypedExpression::Subscript { span, .. }
        | TypedExpression::Concat { span, .. } => (span.start, span.end, None),
        TypedExpression::Deferred {
            span, display_text, ..
        } => (span.start, span.end, Some(display_text.as_str())),
    };
    if let Some(display) = saved_display {
        return display.to_string();
    }
    let start = start.saturating_sub(statement.span.start);
    let end = end.saturating_sub(statement.span.start);
    let Some(start_byte) = codepoint_to_byte(&statement.display_text, start) else {
        return statement.display_text.clone();
    };
    let Some(end_byte) = codepoint_to_byte(&statement.display_text, end) else {
        return statement.display_text.clone();
    };
    statement
        .display_text
        .get(start_byte..end_byte)
        .unwrap_or(&statement.display_text)
        .to_string()
}

fn codepoint_to_byte(text: &str, offset: usize) -> Option<usize> {
    if offset == text.chars().count() {
        return Some(text.len());
    }
    text.char_indices().nth(offset).map(|(byte, _)| byte)
}

#[cfg(test)]
mod tests {
    use super::{
        apply_sum_update, collect_typed_mutable_column_names, execute_typed_assignment,
        execute_typed_inline_action, execute_typed_statement_block,
        typed_condition_error_diagnostic, EvalRuntimeState, NativeLogicalRowView,
        RowExecutionOutcome,
    };
    use crate::expressions::{
        column_key, get_stateful_scalar_value, resolve_value, Expr, ScalarValue,
    };
    use crate::row_view::{ColumnBinding, LogicalRowView};
    use crate::source_registry::ExecutionSourceRegistry;
    use crate::typed_ast::{parse_typed_program, TypedExpression, TypedLiteralKind, TypedSpan};
    use crate::typed_lowering::{
        lower_program, LoweredAssignmentTarget, LoweredExpression, LoweredInlineAction,
        LoweredStatement, LoweredStatementPayload,
    };
    use crate::value_ref::OwnedValue;
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyList};
    use serde_json::{json, Value};
    use std::collections::HashMap;
    use std::rc::Rc;

    #[test]
    fn typed_condition_diagnostic_uses_saved_summary_without_compiling_display() {
        let display = "definitely not valid (";
        let (code, message) = typed_condition_error_diagnostic(
            display,
            "unexpected token: **",
            "saved WHERE syntax summary",
        );
        assert_eq!(code, "RUNTIME_OPERATOR_NOT_SUPPORTED");
        assert_eq!(
            message,
            "Unsupported operator in expression 'definitely not valid (': saved WHERE syntax summary"
        );

        let (empty_summary_code, empty_summary_message) =
            typed_condition_error_diagnostic(display, "unexpected token: **", "");
        assert_eq!(empty_summary_code, "RUNTIME_OPERATOR_NOT_SUPPORTED");
        assert_eq!(
            empty_summary_message,
            "Unsupported operator in expression 'definitely not valid (': unexpected token: **"
        );

        for cause in [
            "invalid numeric literal '1e999'",
            "unsupported function: mystery",
            "numeric value required, got 'bad'",
        ] {
            let (evaluation_code, evaluation_message) =
                typed_condition_error_diagnostic(display, cause, "must not replace a scalar cause");
            assert_eq!(evaluation_code, "RUNTIME_EXPRESSION_EVALUATION_ERROR");
            assert_eq!(
                evaluation_message,
                format!("Rust WHERE evaluation failed: {cause}")
            );
        }

        let syntax_summary_calls = std::cell::Cell::new(0);
        let (legacy_code, legacy_message) =
            super::condition_error_details(display, "unsupported function: mystery", || {
                syntax_summary_calls.set(syntax_summary_calls.get() + 1);
                Ok(Some("should stay lazy".to_string()))
            })
            .expect("diagnostic formatting succeeds");
        assert_eq!(legacy_code, "RUNTIME_EXPRESSION_EVALUATION_ERROR");
        assert_eq!(
            legacy_message,
            "Rust WHERE evaluation failed: unsupported function: mystery"
        );
        assert_eq!(syntax_summary_calls.get(), 0);

        let (legacy_parse_code, legacy_parse_message) =
            super::condition_error_details(display, "unexpected token: **", || {
                syntax_summary_calls.set(syntax_summary_calls.get() + 1);
                Ok(Some("legacy SyntaxError summary".to_string()))
            })
            .expect("parse failure diagnostic formatting succeeds");
        assert_eq!(legacy_parse_code, "RUNTIME_OPERATOR_NOT_SUPPORTED");
        assert_eq!(
            legacy_parse_message,
            "Unsupported operator in expression 'definitely not valid (': legacy SyntaxError summary"
        );
        assert_eq!(syntax_summary_calls.get(), 1);
    }

    #[test]
    fn typed_inline_assignment_uses_structural_target_and_keeps_raw_fault_cause() {
        let span = TypedSpan {
            start: 0,
            end: 10,
            line: 1,
            column: 1,
            end_line: Some(1),
            end_column: Some(11),
            source_id: "typed-inline-test".to_string(),
        };
        let assigned_value = LoweredExpression {
            expression: Expr::Literal(ScalarValue::Number(7.0)),
            source: TypedExpression::Literal {
                span: span.clone(),
                origin: None,
                kind: TypedLiteralKind::Numeric,
                lexeme: "7".to_string(),
            },
        };
        let simple_statement = LoweredStatement {
            kind: "ASSIGN".to_string(),
            display_text: "result = 7".to_string(),
            span: span.clone(),
            origin: None,
            apply_references: Vec::new(),
            payload: LoweredStatementPayload::Assign {
                target: LoweredAssignmentTarget::Name {
                    name: "result".to_string(),
                    span: span.clone(),
                },
                value: assigned_value,
            },
        };
        with_sum_row(vec![], |row, state| {
            let mut output_router =
                |_: &Bound<'_, PyDict>, _: Option<&str>, _: &mut EvalRuntimeState| Ok(());
            let mut outcome = RowExecutionOutcome::default();
            execute_typed_inline_action(
                &LoweredInlineAction::Statement(Box::new(simple_statement)),
                row,
                state,
                &mut outcome,
                &mut output_router,
            )
            .expect("inline assignment succeeds from its typed target");
            assert_eq!(
                get_stateful_scalar_value(row, "result", state).expect("inline value"),
                Some(ScalarValue::Number(7.0))
            );
        });

        let expression = LoweredExpression {
            expression: Expr::DeferredError {
                cause: "saved deferred cause".to_string(),
            },
            source: TypedExpression::Deferred {
                span: span.clone(),
                origin: None,
                error_kind: "unsupported_expression".to_string(),
                native_cause: "saved deferred cause".to_string(),
                where_syntax_summary: "saved summary".to_string(),
                display_text: "invalid +".to_string(),
                apply_references: Vec::new(),
            },
        };
        let statement = LoweredStatement {
            kind: "ASSIGN".to_string(),
            display_text: "result = invalid +".to_string(),
            span: span.clone(),
            origin: None,
            apply_references: Vec::new(),
            payload: LoweredStatementPayload::Assign {
                target: LoweredAssignmentTarget::Name {
                    name: "result".to_string(),
                    span,
                },
                value: expression.clone(),
            },
        };

        let normal_error = Python::attach(|py| {
            let row = PyDict::new(py);
            let mut state = EvalRuntimeState::default();
            execute_typed_assignment(
                &statement,
                match &statement.payload {
                    LoweredStatementPayload::Assign { target, .. } => target,
                    _ => unreachable!(),
                },
                &expression,
                &row,
                &mut state,
                true,
            )
            .expect_err("deferred normal assignment stays a fault")
        });
        assert_eq!(
            normal_error,
            "assignment expression failed: 'invalid +': saved deferred cause"
        );

        let inline_error = Python::attach(|py| {
            let row = PyDict::new(py);
            let mut state = EvalRuntimeState::default();
            let mut output_router =
                |_: &Bound<'_, PyDict>, _: Option<&str>, _: &mut EvalRuntimeState| Ok(());
            let mut outcome = RowExecutionOutcome::default();
            execute_typed_inline_action(
                &LoweredInlineAction::Statement(Box::new(statement)),
                &row,
                &mut state,
                &mut outcome,
                &mut output_router,
            )
            .expect_err("deferred inline assignment remains a fault")
        });
        assert_eq!(inline_error, "saved deferred cause");
    }

    fn typed_fixture_program() -> Value {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/typed_payload_v1.json"
        )))
        .expect("shared typed payload fixture is valid JSON");
        fixture["valid_program"].clone()
    }

    fn typed_sum_value_from_fixture(target_name: &str) -> Value {
        let program = typed_fixture_program();
        let mut statement = program["statements"]
            .as_array()
            .expect("fixture statements are an array")
            .iter()
            .find(|statement| statement["kind"] == "SUM")
            .expect("fixture has a SUM statement")
            .clone();
        let span = statement["span"].clone();
        statement["payload"]["target"] = json!({
            "tag": "name",
            "name": target_name,
            "span": span
        });
        let increment_span = statement["payload"]["increment"]["span"].clone();
        statement["payload"]["increment"] = typed_literal(&increment_span, "numeric", "1");
        statement
    }

    fn typed_if_value_from_fixture(condition_name: &str, then_do: bool) -> Value {
        let program = typed_fixture_program();
        let mut statement = program["statements"]
            .as_array()
            .expect("fixture statements are an array")
            .iter()
            .find(|statement| statement["kind"] == "IF" && statement["payload"]["tag"] == "if")
            .expect("fixture has an IF statement")
            .clone();
        let span = statement["payload"]["condition"]["span"].clone();
        statement["payload"]["condition"] = json!({
            "tag": "name",
            "span": span,
            "origin": null,
            "name": condition_name
        });
        statement["payload"]["then_action"] = Value::Null;
        statement["payload"]["is_subset"] = json!(false);
        statement["payload"]["is_then_do"] = json!(then_do);
        statement
    }

    fn typed_else_sum_value_from_fixture(target_name: &str, as_do: bool) -> Value {
        let program = typed_fixture_program();
        let mut statement = program["statements"]
            .as_array()
            .expect("fixture statements are an array")
            .iter()
            .find(|statement| statement["kind"] == "ELSE")
            .expect("fixture has an ELSE statement")
            .clone();
        if as_do {
            let mut do_statement = program["statements"]
                .as_array()
                .expect("fixture statements are an array")
                .iter()
                .find(|item| item["kind"] == "DO")
                .expect("fixture has a DO statement")
                .clone();
            do_statement["payload"] = json!({
                "tag": "do",
                "loop_var": null,
                "start": null,
                "end": null
            });
            statement["payload"]["action"] = json!({
                "tag": "statement",
                "statement": do_statement
            });
        } else {
            let mut sum = typed_sum_value_from_fixture(target_name);
            sum["payload"]["target"]["name"] = json!(target_name);
            statement["payload"]["action"] = json!({
                "tag": "statement",
                "statement": sum
            });
        }
        statement
    }

    fn typed_sum_sequence_statements(
        condition_name: &str,
        do_blocks: bool,
    ) -> Vec<LoweredStatement> {
        let mut program = typed_fixture_program();
        let mut statements = vec![typed_if_value_from_fixture(condition_name, do_blocks)];
        if do_blocks {
            let end = program["statements"]
                .as_array()
                .expect("fixture statements are an array")
                .iter()
                .find(|statement| statement["kind"] == "END")
                .expect("fixture has an END marker")
                .clone();
            statements.push(typed_sum_value_from_fixture("seq"));
            statements.push(end);
            statements.push(typed_else_sum_value_from_fixture("seq", true));
            statements.push(typed_sum_value_from_fixture("seq"));
            let end = program["statements"]
                .as_array()
                .expect("fixture statements are an array")
                .iter()
                .find(|statement| statement["kind"] == "END")
                .expect("fixture has an END marker")
                .clone();
            statements.push(end);
        } else {
            let sum = typed_sum_value_from_fixture("seq");
            statements[0]["payload"]["then_action"] = json!({
                "tag": "statement",
                "statement": sum
            });
            statements.push(typed_else_sum_value_from_fixture("seq", false));
        }
        program["statements"] = json!(statements);
        let parsed = parse_typed_program(&program.to_string()).expect("IF/SUM fixture is valid");
        lower_program(&parsed)
            .expect("IF/SUM fixture lowers")
            .statements
    }

    fn typed_fixture_statement(
        kind: &str,
        payload_tag: &str,
        mutate: impl FnOnce(&mut Value),
    ) -> LoweredStatement {
        let mut program = typed_fixture_program();
        let statements = program["statements"]
            .as_array()
            .expect("fixture statements are an array");
        let index = statements
            .iter()
            .position(|statement| {
                statement["kind"] == kind && statement["payload"]["tag"] == payload_tag
            })
            .unwrap_or_else(|| panic!("fixture has {kind}/{payload_tag} statement"));
        mutate(&mut program["statements"][index]);

        let parsed = parse_typed_program(&program.to_string()).expect("fixture remains valid");
        let lowered = lower_program(&parsed).expect("typed fixture lowers");
        lowered
            .statements
            .into_iter()
            .find(|statement| {
                statement.kind == kind
                    && match (&statement.payload, payload_tag) {
                        (LoweredStatementPayload::Assign { .. }, "assign") => true,
                        (LoweredStatementPayload::Sum { .. }, "sum") => true,
                        _ => false,
                    }
            })
            .unwrap_or_else(|| panic!("lowered fixture has {kind}/{payload_tag} statement"))
    }

    fn typed_literal(span: &Value, kind: &str, lexeme: &str) -> Value {
        json!({
            "tag": "literal",
            "span": span,
            "origin": null,
            "kind": kind,
            "lexeme": lexeme
        })
    }

    fn typed_sum_fixture_statement(
        target_name: &str,
        increment_kind: &str,
        increment_lexeme: &str,
    ) -> LoweredStatement {
        typed_fixture_statement("SUM", "sum", |statement| {
            let span = statement["span"].clone();
            statement["payload"]["target"] = json!({
                "tag": "name",
                "name": target_name,
                "span": span.clone()
            });
            let increment_span = statement["payload"]["increment"]["span"].clone();
            statement["payload"]["increment"] =
                typed_literal(&increment_span, increment_kind, increment_lexeme);
        })
    }

    fn typed_assignment_fixture_statement(
        target_name: &str,
        value: impl FnOnce(&Value) -> Value,
    ) -> LoweredStatement {
        typed_fixture_statement("ASSIGN", "assign", |statement| {
            let span = statement["span"].clone();
            statement["payload"]["target"] = json!({
                "tag": "name",
                "name": target_name,
                "span": span
            });
            let value_span = statement["payload"]["value"]["span"].clone();
            statement["payload"]["value"] = value(&value_span);
        })
    }

    fn typed_name_plus_one(span: &Value, name: &str) -> Value {
        json!({
            "tag": "binary",
            "span": span,
            "origin": null,
            "operator": "+",
            "left": {
                "tag": "name",
                "span": span,
                "origin": null,
                "name": name
            },
            "right": typed_literal(span, "numeric", "1")
        })
    }

    fn run_typed_statement_for_row(
        statement: &LoweredStatement,
        row: &Bound<'_, PyDict>,
        state: &mut EvalRuntimeState,
    ) -> Result<(), String> {
        let mut output_router =
            |_: &Bound<'_, PyDict>, _: Option<&str>, _: &mut EvalRuntimeState| Ok(());
        execute_typed_statement_block(
            std::slice::from_ref(statement),
            row,
            state,
            &mut output_router,
        )
        .map(|_| ())
    }

    fn typed_sum_statement_totals(
        statements: &[LoweredStatement],
        rows: Vec<Vec<(&str, ScalarValue)>>,
    ) -> Vec<Result<f64, String>> {
        Python::attach(|py| {
            let mut state = EvalRuntimeState::default();
            let mut totals = Vec::new();
            for bindings in rows {
                let row = PyDict::new(py);
                state.mutable_values = bindings
                    .iter()
                    .map(|(name, value)| ((*name).to_uppercase(), OwnedValue::from(value.clone())))
                    .collect();
                state.logical_row = Some(build_row_view(bindings));
                let mut output_router =
                    |_: &Bound<'_, PyDict>, _: Option<&str>, _: &mut EvalRuntimeState| Ok(());
                let result =
                    execute_typed_statement_block(statements, &row, &mut state, &mut output_router)
                        .map(|_| state.sum_totals.get("SEQ").copied().unwrap_or(f64::NAN));
                totals.push(result);
            }
            totals
        })
    }

    fn deferred_expression(span: Value, cause: &str, summary: &str, display: &str) -> Value {
        json!({
            "tag": "deferred",
            "span": span.clone(),
            "origin": null,
            "error_kind": "unsupported_expression",
            "native_cause": cause,
            "where_syntax_summary": summary,
            "display_text": display,
            "apply_references": [{
                "target": "lookup",
                "call_span": span.clone(),
                "owner_span": span
            }]
        })
    }

    fn assert_deferred_source(
        expression: &LoweredExpression,
        expected_cause: &str,
        expected_summary: &str,
        expected_display: &str,
    ) {
        match &expression.source {
            TypedExpression::Deferred {
                span,
                origin,
                error_kind,
                native_cause,
                where_syntax_summary,
                display_text,
                apply_references,
            } => {
                assert_eq!(error_kind, "unsupported_expression");
                assert_eq!(native_cause, expected_cause);
                assert_eq!(where_syntax_summary, expected_summary);
                assert_eq!(display_text, expected_display);
                assert_eq!(span.source_id, "typed-payload-fixture");
                assert!(origin.is_none());
                assert_eq!(apply_references.len(), 1);
                assert_eq!(apply_references[0].target, "lookup");
                assert_eq!(apply_references[0].call_span.source_id, span.source_id);
                assert_eq!(apply_references[0].owner_span.source_id, span.source_id);
            }
            other => panic!("expected retained deferred source metadata, got {other:?}"),
        }
    }

    #[test]
    fn typed_where_fault_uses_saved_metadata_without_evaluating_apply() {
        let mut program = typed_fixture_program();
        let statements = program["statements"]
            .as_array()
            .expect("fixture statements are an array");
        let where_index = statements
            .iter()
            .position(|statement| statement["kind"] == "WHERE")
            .expect("fixture has a WHERE statement");
        program["statements"][where_index]["payload"]["condition"]["native_cause"] =
            json!("unexpected token: **");
        program["statements"][where_index]["payload"]["condition"]["where_syntax_summary"] =
            json!("saved WHERE summary");
        program["statements"][where_index]["payload"]["condition"]["display_text"] =
            json!("definitely not valid (");

        let parsed = parse_typed_program(&program.to_string()).expect("fixture remains valid");
        let lowered = lower_program(&parsed).expect("typed fixture lowers");
        let statement = lowered
            .statements
            .iter()
            .find(|statement| statement.kind == "WHERE")
            .expect("lowered program retains WHERE");
        let (expression, summary, display) = match &statement.payload {
            LoweredStatementPayload::Where { condition } => {
                assert_deferred_source(
                    condition,
                    "unexpected token: **",
                    "saved WHERE summary",
                    "definitely not valid (",
                );
                let TypedExpression::Deferred {
                    where_syntax_summary,
                    display_text,
                    ..
                } = &condition.source
                else {
                    panic!("expected deferred WHERE source");
                };
                (
                    condition.expression.clone(),
                    where_syntax_summary.clone(),
                    display_text.clone(),
                )
            }
            other => panic!("expected lowered WHERE, got {other:?}"),
        };

        Python::attach(|py| {
            let calls = PyList::empty(py);
            let globals = PyDict::new(py);
            globals.set_item("calls", &calls).unwrap();
            let callback = py
                .eval(c"lambda *args: calls.append(1)", Some(&globals), None)
                .expect("test callback is callable");
            let row = PyDict::new(py);
            let mut state = EvalRuntimeState::default();
            state
                .apply_registry
                .insert("lookup".to_string(), callback.unbind());

            let cause = crate::expressions::evaluate_typed_condition(&row, &expression, &mut state)
                .unwrap_err();
            assert_eq!(cause, "unexpected token: **");
            let (code, message) =
                super::typed_condition_error_diagnostic(&display, &cause, &summary);
            assert_eq!(code, "RUNTIME_OPERATOR_NOT_SUPPORTED");
            assert_eq!(
                message,
                "Unsupported operator in expression 'definitely not valid (': saved WHERE summary"
            );
            assert_eq!(calls.len(), 0);
            assert_eq!(state.apply_registry.len(), 1);
        });
    }

    #[test]
    fn deferred_expression_metadata_survives_normal_and_inline_assignment_lowering() {
        let mut program = typed_fixture_program();
        let statements = program["statements"]
            .as_array()
            .expect("fixture statements are an array");
        let normal_index = statements
            .iter()
            .position(|statement| {
                statement["kind"] == "ASSIGN" && statement["payload"]["tag"] == "assign"
            })
            .expect("fixture has a normal assignment");
        let normal_span = program["statements"][normal_index]["payload"]["value"]["span"].clone();
        program["statements"][normal_index]["payload"]["value"] = deferred_expression(
            normal_span,
            "normal saved cause",
            "normal saved summary",
            "invalid normal assignment display (",
        );

        let statements = program["statements"]
            .as_array()
            .expect("fixture statements are an array");
        let inline_index = statements
            .iter()
            .position(|statement| {
                statement["payload"]["tag"] == "if"
                    && statement["payload"]["then_action"]["tag"] == "statement"
                    && statement["payload"]["then_action"]["statement"]["kind"] == "ASSIGN"
            })
            .expect("fixture has an inline assignment");
        let inline_payload = &mut program["statements"][inline_index]["payload"]["then_action"]
            ["statement"]["payload"];
        let inline_span = inline_payload["value"]["span"].clone();
        inline_payload["value"] = deferred_expression(
            inline_span,
            "inline saved cause",
            "inline saved summary",
            "invalid inline assignment display (",
        );

        let parsed = parse_typed_program(&program.to_string()).expect("fixture remains valid");
        let lowered = lower_program(&parsed).expect("typed fixture lowers");
        let normal = lowered
            .statements
            .iter()
            .find(|statement| matches!(statement.payload, LoweredStatementPayload::Assign { .. }))
            .expect("normal assignment remains lowered");
        assert_eq!(normal.display_text, "payload");
        match &normal.payload {
            LoweredStatementPayload::Assign { value, .. } => assert_deferred_source(
                value,
                "normal saved cause",
                "normal saved summary",
                "invalid normal assignment display (",
            ),
            other => panic!("expected normal assignment, got {other:?}"),
        }

        let inline = lowered
            .statements
            .iter()
            .find_map(|statement| match &statement.payload {
                LoweredStatementPayload::If {
                    then_action: Some(LoweredInlineAction::Statement(inner)),
                    ..
                } if matches!(inner.payload, LoweredStatementPayload::Assign { .. }) => Some(inner),
                _ => None,
            })
            .expect("inline assignment remains lowered");
        match &inline.payload {
            LoweredStatementPayload::Assign { value, .. } => assert_deferred_source(
                value,
                "inline saved cause",
                "inline saved summary",
                "invalid inline assignment display (",
            ),
            other => panic!("expected inline assignment, got {other:?}"),
        }
    }

    fn with_sum_row(
        owned_values: Vec<(&str, ScalarValue)>,
        run: impl FnOnce(&Bound<'_, PyDict>, &mut EvalRuntimeState),
    ) {
        with_seeded_sum_row(owned_values, &[], run);
    }

    // Mirrors the public row loop: `seeded_names` become owned bindings whose
    // values are re-seeded as NULL (like `clear_owned_values` does for ensured
    // columns), while `owned_values` are written values mirrored into
    // `state.mutable_values` exactly like `set_mutable_scalar` does, so the SUM
    // classification can distinguish written NULL from the seeded absent
    // binding.
    fn with_seeded_sum_row(
        owned_values: Vec<(&str, ScalarValue)>,
        seeded_names: &[&str],
        run: impl FnOnce(&Bound<'_, PyDict>, &mut EvalRuntimeState),
    ) {
        Python::attach(|py| {
            let row = PyDict::new(py);
            let mut values = HashMap::new();
            let mut written_values = HashMap::new();
            let mut columns = Vec::new();
            for (name, value) in owned_values {
                values.insert(name.to_string(), OwnedValue::from(value.clone()));
                written_values.insert(name.to_uppercase(), OwnedValue::from(value));
                columns.push((
                    name.to_string(),
                    ColumnBinding::Owned {
                        normalized_name: name.to_uppercase(),
                    },
                ));
            }
            for name in seeded_names {
                columns.push((
                    name.to_string(),
                    ColumnBinding::Owned {
                        normalized_name: name.to_uppercase(),
                    },
                ));
            }
            let view = LogicalRowView::with_values(
                0,
                Rc::new(ExecutionSourceRegistry::new()),
                columns,
                values,
                HashMap::new(),
            )
            .expect("logical row view");
            let mut state = EvalRuntimeState::default();
            state.logical_row = Some(view);
            state.mutable_values = written_values;
            run(&row, &mut state);
        });
    }

    fn expect_row_number(
        row: &Bound<'_, PyDict>,
        state: &EvalRuntimeState,
        name: &str,
        expected: f64,
    ) {
        match get_stateful_scalar_value(row, name, state).expect("row value") {
            Some(ScalarValue::Number(value)) => assert_eq!(value, expected),
            other => panic!("row value for '{name}' is {other:?}, expected Number({expected})"),
        }
    }

    #[test]
    fn first_sum_value_starts_from_zero_and_accumulates_absent_rows() {
        with_sum_row(vec![], |row, state| {
            apply_sum_update(row, state, "seq", ScalarValue::Number(1.0)).expect("first sum");
            assert_eq!(state.sum_totals.get("SEQ"), Some(&1.0));
            expect_row_number(row, state, "seq", 1.0);

            apply_sum_update(row, state, "seq", ScalarValue::Number(1.0)).expect("second sum");
            assert_eq!(state.sum_totals.get("SEQ"), Some(&2.0));

            apply_sum_update(row, state, "SEQ", ScalarValue::Number(1.0)).expect("third sum");
            assert_eq!(state.sum_totals.get("SEQ"), Some(&3.0));
        });
    }

    #[test]
    fn sum_seeded_absent_binding_accumulates_via_prior_total() {
        // Mirrors the public execute_block path: the row loop ensures an owned
        // binding for every mutable SUM target and re-seeds it with a NULL owned
        // value, and the binding is never written for that row. The seeded
        // binding must classify as Absent so the SUM statement reuses the prior
        // accumulator — the owner-level mirror of the public [1.0, 2.0, 3.0]
        // accumulation sequence.
        with_seeded_sum_row(vec![], &["seq"], |row, state| {
            apply_sum_update(row, state, "seq", ScalarValue::Number(1.0)).expect("first sum");
            assert_eq!(state.sum_totals.get("SEQ"), Some(&1.0));
            expect_row_number(row, state, "seq", 1.0);
        });

        with_seeded_sum_row(vec![], &["seq"], |row, state| {
            state.sum_totals.insert("SEQ".to_string(), 1.0);
            apply_sum_update(row, state, "seq", ScalarValue::Number(1.0)).expect("second sum");
            assert_eq!(state.sum_totals.get("SEQ"), Some(&2.0));
            expect_row_number(row, state, "seq", 2.0);
        });
    }

    #[test]
    fn sum_case_variations_share_one_canonical_accumulator() {
        with_sum_row(vec![], |row, state| {
            apply_sum_update(row, state, "Seq", ScalarValue::Number(1.0)).expect("Seq");
            apply_sum_update(row, state, "seq", ScalarValue::Number(1.0)).expect("seq");
            apply_sum_update(row, state, "SEQ", ScalarValue::Number(1.0)).expect("SEQ");

            assert_eq!(state.sum_totals.get("SEQ"), Some(&3.0));
            assert_eq!(state.sum_totals.get("seq"), None);
            expect_row_number(row, state, "seq", 3.0);
            assert_eq!(column_key("seq"), "SEQ");
        });
    }

    #[test]
    fn sum_explicit_null_resets_accumulator_without_reviving_it() {
        // Prior absent rows accumulated seq to 4.
        with_sum_row(vec![], |row, state| {
            apply_sum_update(row, state, "seq", ScalarValue::Number(2.0)).expect("first");
            apply_sum_update(row, state, "seq", ScalarValue::Number(2.0)).expect("second");
            assert_eq!(state.sum_totals.get("SEQ"), Some(&4.0));
        });

        // Explicit NULL row: base resets to 0, the stored accumulator is not revived.
        with_sum_row(vec![("seq", ScalarValue::Null)], |row, state| {
            state.sum_totals.insert("SEQ".to_string(), 4.0);
            apply_sum_update(row, state, "seq", ScalarValue::Number(1.0)).expect("null row");
            assert_eq!(state.sum_totals.get("SEQ"), Some(&1.0));
            expect_row_number(row, state, "seq", 1.0);
        });

        // The next absent row continues from the post-reset accumulator (1), not 4.
        with_sum_row(vec![], |row, state| {
            state.sum_totals.insert("SEQ".to_string(), 1.0);
            apply_sum_update(row, state, "seq", ScalarValue::Number(1.0)).expect("next row");
            assert_eq!(state.sum_totals.get("SEQ"), Some(&2.0));
        });
    }

    #[test]
    fn sum_null_increment_adds_zero_without_changing_base() {
        with_sum_row(vec![("seq", ScalarValue::Number(3.0))], |row, state| {
            apply_sum_update(row, state, "seq", ScalarValue::Null).expect("null increment");
            assert_eq!(state.sum_totals.get("SEQ"), Some(&3.0));
            expect_row_number(row, state, "seq", 3.0);
        });
    }

    #[test]
    fn sum_non_numeric_target_value_fails_closed_without_silent_zero() {
        with_sum_row(
            vec![("seq", ScalarValue::Text("abc".to_string()))],
            |row, state| {
                let error =
                    apply_sum_update(row, state, "seq", ScalarValue::Number(1.0)).unwrap_err();
                assert!(error.contains("seq"), "{error}");
                assert!(
                    error.contains("abc") || error.contains("numeric"),
                    "{error}"
                );
                assert!(state.sum_totals.get("SEQ").is_none());
            },
        );
    }

    #[test]
    fn sum_non_numeric_increment_fails_closed_with_target_identified() {
        with_sum_row(vec![], |row, state| {
            let error = apply_sum_update(row, state, "seq", ScalarValue::Text("xyz".to_string()))
                .unwrap_err();
            assert!(error.contains("seq"), "{error}");
            assert!(
                error.contains("numeric") || error.contains("xyz"),
                "{error}"
            );
            assert!(state.sum_totals.get("SEQ").is_none());
        });
    }

    #[test]
    fn sum_numeric_target_coercions_keep_checked_rules() {
        with_sum_row(
            vec![("seq", ScalarValue::Decimal("1.5".to_string()))],
            |row, state| {
                apply_sum_update(row, state, "seq", ScalarValue::Number(1.0))
                    .expect("decimal base");
                assert_eq!(state.sum_totals.get("SEQ"), Some(&2.5));
            },
        );

        with_sum_row(vec![("seq", ScalarValue::Bool(true))], |row, state| {
            apply_sum_update(row, state, "seq", ScalarValue::Number(1.0)).expect("bool base");
            assert_eq!(state.sum_totals.get("SEQ"), Some(&2.0));
        });

        with_sum_row(vec![("seq", ScalarValue::Integer(7))], |row, state| {
            apply_sum_update(row, state, "seq", ScalarValue::Number(1.0)).expect("integer base");
            assert_eq!(state.sum_totals.get("SEQ"), Some(&8.0));
        });
    }

    fn build_row_view(bindings: Vec<(&str, ScalarValue)>) -> NativeLogicalRowView {
        let mut values = HashMap::new();
        let mut columns = Vec::new();
        for (name, value) in bindings {
            values.insert(name.to_string(), OwnedValue::from(value));
            columns.push((
                name.to_string(),
                ColumnBinding::Owned {
                    normalized_name: name.to_uppercase(),
                },
            ));
        }
        LogicalRowView::with_values(
            0,
            Rc::new(ExecutionSourceRegistry::new()),
            columns,
            values,
            HashMap::new(),
        )
        .expect("logical row view")
    }

    #[test]
    fn typed_sum_undefined_target_accumulates_from_fixture() {
        let statement = typed_sum_fixture_statement("seq", "numeric", "1");
        let totals = typed_sum_statement_totals(
            std::slice::from_ref(&statement),
            vec![vec![], vec![], vec![]],
        );
        assert_eq!(totals, vec![Ok(1.0), Ok(2.0), Ok(3.0)]);
    }

    #[test]
    fn typed_if_inline_and_else_sum_actions_share_sequence_from_fixture() {
        let statements = typed_sum_sequence_statements("flag", false);
        let totals = typed_sum_statement_totals(
            &statements,
            vec![
                vec![("flag", ScalarValue::Bool(true))],
                vec![("flag", ScalarValue::Bool(false))],
                vec![("flag", ScalarValue::Bool(true))],
            ],
        );
        assert_eq!(totals, vec![Ok(1.0), Ok(2.0), Ok(3.0)]);
    }

    #[test]
    fn typed_if_then_do_and_else_do_share_sequence_from_fixture() {
        let statements = typed_sum_sequence_statements("flag", true);
        let totals = typed_sum_statement_totals(
            &statements,
            vec![
                vec![("flag", ScalarValue::Bool(true))],
                vec![("flag", ScalarValue::Bool(false))],
                vec![("flag", ScalarValue::Bool(true))],
            ],
        );
        assert_eq!(totals, vec![Ok(1.0), Ok(2.0), Ok(3.0)]);
    }

    #[test]
    fn typed_mutable_column_collection_reads_inline_actions_from_fixture() {
        let statements = typed_sum_sequence_statements("flag", false);
        assert_eq!(collect_typed_mutable_column_names(&statements), vec!["seq"]);
        let totals = typed_sum_statement_totals(
            &statements,
            vec![
                vec![("flag", ScalarValue::Number(0.0))],
                vec![("flag", ScalarValue::Number(0.0))],
                vec![("flag", ScalarValue::Number(0.0))],
            ],
        );
        assert_eq!(totals, vec![Ok(1.0), Ok(2.0), Ok(3.0)]);
    }

    #[test]
    fn typed_sum_explicit_null_row_resets_sequence_without_revival() {
        let statement = typed_sum_fixture_statement("seq", "numeric", "1");
        let rows = vec![vec![], vec![], vec![("seq", ScalarValue::Null)], vec![]];
        let totals = typed_sum_statement_totals(std::slice::from_ref(&statement), rows);
        // Reset mid-sequence: 2 accumulated, explicit NULL row starts from 0 again,
        // and the following absent row continues from the post-reset value.
        assert_eq!(totals, vec![Ok(1.0), Ok(2.0), Ok(1.0), Ok(2.0)]);
    }

    #[test]
    fn typed_sum_null_increment_keeps_base() {
        let statement = typed_sum_fixture_statement("seq", "missing", ".");
        let rows = vec![vec![("seq", ScalarValue::Number(3.0))], vec![]];
        let totals = typed_sum_statement_totals(std::slice::from_ref(&statement), rows);
        assert_eq!(totals, vec![Ok(3.0), Ok(3.0)]);
    }

    #[test]
    fn typed_sum_case_variations_share_one_accumulator() {
        let statements =
            ["Seq", "seq", "SEQ"].map(|name| typed_sum_fixture_statement(name, "numeric", "1"));
        Python::attach(|py| {
            let mut state = EvalRuntimeState::default();
            for statement in &statements {
                let row = PyDict::new(py);
                state.logical_row = Some(build_row_view(vec![]));
                run_typed_statement_for_row(statement, &row, &mut state).expect("typed SUM");
            }
            assert_eq!(state.sum_totals.get("SEQ"), Some(&3.0));
            assert_eq!(state.sum_totals.get("seq"), None);
        });
    }

    #[test]
    fn typed_sum_failure_propagates_without_partial_row_write() {
        let statement = typed_sum_fixture_statement("seq", "numeric", "1");
        Python::attach(|py| {
            let mut state = EvalRuntimeState::default();
            state.materialize_mutable_values = true;

            let first_row = PyDict::new(py);
            state.logical_row = Some(build_row_view(vec![]));
            run_typed_statement_for_row(&statement, &first_row, &mut state)
                .expect("first typed SUM");
            assert_eq!(state.sum_totals.get("SEQ"), Some(&1.0));
            assert!(first_row.get_item("seq").expect("row item").is_some());

            let failing_row = PyDict::new(py);
            state.logical_row = Some(build_row_view(vec![(
                "seq",
                ScalarValue::Text("abc".to_string()),
            )]));
            let error =
                run_typed_statement_for_row(&statement, &failing_row, &mut state).unwrap_err();
            assert!(error.contains("seq"), "{error}");
            assert!(failing_row.get_item("seq").expect("row item").is_none());
            assert_eq!(state.sum_totals.get("SEQ"), Some(&1.0));
        });
    }

    #[test]
    fn typed_sum_rules_stay_separate_from_assignment_null_propagation() {
        Python::attach(|py| {
            let sum_statement = typed_sum_fixture_statement("seq", "numeric", "1");
            let mut sum_state = EvalRuntimeState::default();
            let sum_row = PyDict::new(py);
            sum_state.logical_row = Some(build_row_view(vec![]));
            run_typed_statement_for_row(&sum_statement, &sum_row, &mut sum_state)
                .expect("typed SUM first row");
            assert_eq!(sum_state.sum_totals.get("SEQ"), Some(&1.0));

            let assign_statement =
                typed_assignment_fixture_statement("seq", |span| typed_name_plus_one(span, "seq"));
            let mut assign_state = EvalRuntimeState::default();
            let assign_row = PyDict::new(py);
            assign_state.logical_row = Some(build_row_view(vec![("seq", ScalarValue::Null)]));
            run_typed_statement_for_row(&assign_statement, &assign_row, &mut assign_state)
                .expect("typed assignment");
            let resolved = resolve_value(&assign_row, "seq", &assign_state)
                .expect("resolve assignment target")
                .expect("assignment target present");
            let scalar = resolved
                .into_owned_value()
                .expect("owned value")
                .into_scalar_value()
                .expect("scalar value");
            assert!(matches!(scalar, ScalarValue::Null));
            assert!(assign_state.sum_totals.is_empty());
        });
    }
}
