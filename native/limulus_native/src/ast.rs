use std::collections::HashMap;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct AstPayload {
    statements: Vec<AstStatement>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AstStatement {
    pub(crate) kind: String,
    pub(crate) text: String,
    #[serde(default)]
    pub(crate) dataset_refs: Vec<AstDatasetRef>,
    #[serde(default)]
    pub(crate) statement_options: AstStatementOptions,
    #[serde(default)]
    pub(crate) rename_map: HashMap<String, String>,
    #[serde(default)]
    pub(crate) if_spec: Option<AstIfSpec>,
    #[serde(default)]
    pub(crate) do_spec: Option<AstDoSpec>,
    #[serde(default)]
    pub(crate) array_spec: Option<AstArraySpec>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct AstIfSpec {
    pub(crate) condition: String,
    pub(crate) then_action: Option<String>,
    pub(crate) is_subset: bool,
    pub(crate) is_then_do: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct AstDoSpec {
    pub(crate) loop_var: String,
    pub(crate) start_expr: String,
    pub(crate) end_expr: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct AstArraySpec {
    pub(crate) array_name: String,
    #[serde(default)]
    pub(crate) variables: Vec<String>,
    #[allow(dead_code)]
    pub(crate) declared_size: Option<usize>,
    #[allow(dead_code)]
    pub(crate) wildcard_size: bool,
    #[allow(dead_code)]
    pub(crate) character_array: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct AstDatasetRef {
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) options: AstDatasetRefOptions,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct AstDatasetRefOptions {
    pub(crate) in_var: Option<String>,
    #[serde(default)]
    pub(crate) keep_vars: Vec<String>,
    #[serde(default)]
    pub(crate) drop_vars: Vec<String>,
    pub(crate) where_expr: Option<String>,
    #[serde(default)]
    pub(crate) rename_map: HashMap<String, String>,
    pub(crate) firstobs: Option<i64>,
    pub(crate) obs: Option<i64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct AstStatementOptions {
    pub(crate) indsname_var: Option<String>,
    pub(crate) end_var: Option<String>,
}

pub(crate) fn parse_runtime_statements(ast_json: &str) -> Result<Vec<AstStatement>, serde_json::Error> {
    serde_json::from_str::<AstPayload>(ast_json).map(|parsed| parsed.statements)
}
