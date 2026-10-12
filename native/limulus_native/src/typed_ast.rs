//! Strict serde input contract for the versioned typed native program.
//!
//! This is the only native syntax input. Parsing this module never falls back
//! to a DSL parser or evaluates syntax; option runtime structures are built
//! directly by the typed lowering layer.

use std::collections::{BTreeMap, BTreeSet};

use crate::expressions::{BinaryOp, Expr, ScalarValue};
use serde::{Deserialize, Deserializer};
use serde_json::Value;

const TYPED_PAYLOAD_VERSION: u32 = 1;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TypedAstPayload {
    pub(crate) version: u32,
    pub(crate) output_targets: Vec<String>,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub(crate) default_target: Option<String>,
    pub(crate) suppress_implicit_output: bool,
    pub(crate) effective_output_options: BTreeMap<String, TypedDatasetOptions>,
    pub(crate) statements: Vec<TypedStatement>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TypedSpan {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) line: usize,
    pub(crate) column: usize,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub(crate) end_line: Option<usize>,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub(crate) end_column: Option<usize>,
    pub(crate) source_id: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TypedApplyReference {
    pub(crate) target: String,
    pub(crate) call_span: TypedSpan,
    pub(crate) owner_span: TypedSpan,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(crate) enum TypedInteger {
    Signed(i64),
    Unsigned(u64),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TypedDatasetOptions {
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub(crate) in_var: Option<String>,
    pub(crate) keep_vars: Vec<String>,
    pub(crate) drop_vars: Vec<String>,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub(crate) where_expr: Option<TypedWhereExpression>,
    pub(crate) rename_map: BTreeMap<String, String>,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub(crate) firstobs: Option<TypedInteger>,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub(crate) obs: Option<TypedInteger>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
pub(crate) enum TypedWhereExpression {
    #[serde(rename = "comparison")]
    Comparison {
        column: String,
        operator: TypedWhereComparisonOperator,
        value: Value,
    },
    #[serde(rename = "null")]
    Null { column: String, negated: bool },
    #[serde(rename = "logical")]
    Logical {
        operator: TypedWhereLogicalOperator,
        operands: Vec<TypedWhereExpression>,
    },
    #[serde(rename = "not")]
    Not { operand: Box<TypedWhereExpression> },
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub(crate) enum TypedWhereComparisonOperator {
    #[serde(rename = "=")]
    Eq,
    #[serde(rename = "==")]
    Equal,
    #[serde(rename = "!=")]
    NotEqual,
    #[serde(rename = ">")]
    Greater,
    #[serde(rename = ">=")]
    GreaterEqual,
    #[serde(rename = "<")]
    Less,
    #[serde(rename = "<=")]
    LessEqual,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub(crate) enum TypedWhereLogicalOperator {
    #[serde(rename = "and")]
    And,
    #[serde(rename = "or")]
    Or,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "tag", deny_unknown_fields)]
pub(crate) enum TypedExpression {
    #[serde(rename = "literal")]
    Literal {
        span: TypedSpan,
        #[serde(deserialize_with = "deserialize_required_nullable")]
        origin: Option<TypedSpan>,
        kind: TypedLiteralKind,
        // Numeric values intentionally remain source lexemes at this boundary.
        lexeme: String,
    },
    #[serde(rename = "name")]
    Name {
        span: TypedSpan,
        #[serde(deserialize_with = "deserialize_required_nullable")]
        origin: Option<TypedSpan>,
        name: String,
    },
    #[serde(rename = "unary")]
    Unary {
        span: TypedSpan,
        #[serde(deserialize_with = "deserialize_required_nullable")]
        origin: Option<TypedSpan>,
        operator: TypedUnaryOperator,
        operand: Box<TypedExpression>,
    },
    #[serde(rename = "binary")]
    Binary {
        span: TypedSpan,
        #[serde(deserialize_with = "deserialize_required_nullable")]
        origin: Option<TypedSpan>,
        operator: TypedBinaryOperator,
        left: Box<TypedExpression>,
        right: Box<TypedExpression>,
    },
    #[serde(rename = "call")]
    Call {
        span: TypedSpan,
        #[serde(deserialize_with = "deserialize_required_nullable")]
        origin: Option<TypedSpan>,
        name: String,
        arguments: Vec<TypedExpression>,
    },
    #[serde(rename = "subscript")]
    Subscript {
        span: TypedSpan,
        #[serde(deserialize_with = "deserialize_required_nullable")]
        origin: Option<TypedSpan>,
        base: Box<TypedExpression>,
        index: Box<TypedExpression>,
    },
    #[serde(rename = "concat")]
    Concat {
        span: TypedSpan,
        #[serde(deserialize_with = "deserialize_required_nullable")]
        origin: Option<TypedSpan>,
        parts: Vec<TypedExpression>,
    },
    #[serde(rename = "deferred")]
    Deferred {
        span: TypedSpan,
        #[serde(deserialize_with = "deserialize_required_nullable")]
        origin: Option<TypedSpan>,
        error_kind: String,
        native_cause: String,
        where_syntax_summary: String,
        display_text: String,
        apply_references: Vec<TypedApplyReference>,
    },
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub(crate) enum TypedLiteralKind {
    #[serde(rename = "missing")]
    Missing,
    #[serde(rename = "string")]
    String,
    #[serde(rename = "numeric")]
    Numeric,
    #[serde(rename = "format")]
    Format,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub(crate) enum TypedUnaryOperator {
    #[serde(rename = "not")]
    Not,
    #[serde(rename = "-")]
    Negate,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub(crate) enum TypedBinaryOperator {
    #[serde(rename = "or")]
    Or,
    #[serde(rename = "and")]
    And,
    #[serde(rename = "==")]
    Equal,
    #[serde(rename = "!=")]
    NotEqual,
    #[serde(rename = ">")]
    Greater,
    #[serde(rename = ">=")]
    GreaterEqual,
    #[serde(rename = "<")]
    Less,
    #[serde(rename = "<=")]
    LessEqual,
    #[serde(rename = "+")]
    Add,
    #[serde(rename = "-")]
    Subtract,
    #[serde(rename = "*")]
    Multiply,
    #[serde(rename = "/")]
    Divide,
    #[serde(rename = "**")]
    Power,
    #[serde(rename = "mod")]
    Modulo,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "tag", deny_unknown_fields)]
pub(crate) enum TypedAssignmentTarget {
    #[serde(rename = "name")]
    Name { name: String, span: TypedSpan },
    #[serde(rename = "array_element")]
    ArrayElement {
        name: String,
        delimiter: TypedArrayDelimiter,
        index: TypedExpression,
        span: TypedSpan,
    },
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub(crate) enum TypedArrayDelimiter {
    #[serde(rename = "(")]
    Parentheses,
    #[serde(rename = "[")]
    Brackets,
    #[serde(rename = "{")]
    Braces,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TypedDatasetReference {
    pub(crate) name: String,
    pub(crate) options: TypedDatasetOptions,
    pub(crate) span: TypedSpan,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TypedStatementOptions {
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub(crate) indsname_var: Option<String>,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub(crate) end_var: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "tag", deny_unknown_fields)]
pub(crate) enum TypedInlineAction {
    #[serde(rename = "statement")]
    Statement { statement: Box<TypedStatement> },
    #[serde(rename = "deferred")]
    Deferred {
        error_kind: String,
        native_cause: String,
        where_syntax_summary: String,
        span: TypedSpan,
        display_text: String,
    },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "tag", deny_unknown_fields)]
pub(crate) enum TypedStatementPayload {
    #[serde(rename = "data")]
    Data { outputs: Vec<TypedDatasetReference> },
    #[serde(rename = "source")]
    Source {
        refs: Vec<TypedDatasetReference>,
        options: TypedStatementOptions,
    },
    #[serde(rename = "where")]
    Where { condition: TypedExpression },
    #[serde(rename = "assign")]
    Assign {
        target: TypedAssignmentTarget,
        value: TypedExpression,
    },
    #[serde(rename = "sum")]
    Sum {
        target: TypedAssignmentTarget,
        increment: TypedExpression,
    },
    #[serde(rename = "name_list")]
    NameList { names: Vec<TypedExpression> },
    #[serde(rename = "label")]
    Label {},
    #[serde(rename = "rename")]
    Rename {
        rename_map: BTreeMap<String, String>,
    },
    #[serde(rename = "output")]
    Output { targets: Vec<TypedDatasetReference> },
    #[serde(rename = "if")]
    If {
        condition: TypedExpression,
        #[serde(deserialize_with = "deserialize_required_nullable")]
        then_action: Option<TypedInlineAction>,
        is_subset: bool,
        is_then_do: bool,
    },
    #[serde(rename = "else")]
    Else {
        #[serde(deserialize_with = "deserialize_required_nullable")]
        action: Option<TypedInlineAction>,
    },
    #[serde(rename = "do")]
    Do {
        #[serde(deserialize_with = "deserialize_required_nullable")]
        loop_var: Option<String>,
        #[serde(deserialize_with = "deserialize_required_nullable")]
        start: Option<TypedExpression>,
        #[serde(deserialize_with = "deserialize_required_nullable")]
        end: Option<TypedExpression>,
    },
    #[serde(rename = "array")]
    Array {
        array_name: String,
        variables: Vec<String>,
        span: TypedSpan,
        #[serde(deserialize_with = "deserialize_required_nullable")]
        declared_size: Option<TypedInteger>,
        wildcard_size: bool,
        character_array: bool,
    },
    #[serde(rename = "delete")]
    Delete {},
    #[serde(rename = "stop")]
    Stop {},
    #[serde(rename = "end")]
    End {},
    #[serde(rename = "run")]
    Run {},
    #[serde(rename = "skipped")]
    Skipped {},
    #[serde(rename = "deferred")]
    Deferred {
        error_kind: String,
        native_cause: String,
        where_syntax_summary: String,
        span: TypedSpan,
        display_text: String,
    },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TypedStatement {
    pub(crate) kind: String,
    pub(crate) payload: TypedStatementPayload,
    pub(crate) display_text: String,
    pub(crate) span: TypedSpan,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub(crate) origin: Option<TypedSpan>,
    pub(crate) apply_references: Vec<TypedApplyReference>,
}

fn deserialize_required_nullable<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

impl TypedAstPayload {
    fn validate(&self) -> Result<(), String> {
        if self.version != TYPED_PAYLOAD_VERSION {
            return Err(format!(
                "unsupported typed payload version {}; expected {TYPED_PAYLOAD_VERSION}",
                self.version
            ));
        }

        let output_targets = self
            .output_targets
            .iter()
            .map(|target| dataset_key(target))
            .collect::<BTreeSet<_>>();
        let mut option_keys = BTreeSet::new();
        for key in self.effective_output_options.keys() {
            let normalized = dataset_key(key);
            if !option_keys.insert(normalized) {
                return Err(format!(
                    "effective output option keys contain a normalized duplicate: {key:?}"
                ));
            }
        }
        if option_keys != output_targets {
            return Err(
                "normalized output option keys must match normalized output targets".to_string(),
            );
        }

        for (key, options) in &self.effective_output_options {
            options.validate(&format!("effective_output_options[{key:?}]"))?;
        }
        for (index, statement) in self.statements.iter().enumerate() {
            statement.validate(&format!("statements[{index}]"))?;
        }
        Ok(())
    }
}

impl TypedSpan {
    fn validate(&self, path: &str) -> Result<(), String> {
        if self.end < self.start {
            return Err(format!("{path}.end must not precede {path}.start"));
        }
        if self.line == 0 || self.column == 0 {
            return Err(format!("{path}.line and {path}.column are 1-based"));
        }
        if self.end_line == Some(0) || self.end_column == Some(0) {
            return Err(format!("{path}.end_line and {path}.end_column are 1-based"));
        }
        Ok(())
    }
}

impl TypedApplyReference {
    fn validate(&self, path: &str) -> Result<(), String> {
        self.call_span.validate(&format!("{path}.call_span"))?;
        self.owner_span.validate(&format!("{path}.owner_span"))
    }
}

impl TypedDatasetOptions {
    fn validate(&self, path: &str) -> Result<(), String> {
        if let Some(where_expr) = &self.where_expr {
            where_expr.validate(&format!("{path}.where_expr"))?;
        }
        Ok(())
    }
}

impl TypedWhereExpression {
    fn validate(&self, path: &str) -> Result<(), String> {
        match self {
            Self::Comparison { value, .. } => {
                if !matches!(value, Value::String(_) | Value::Bool(_) | Value::Number(_)) {
                    return Err(format!("{path}.value must be a JSON scalar"));
                }
            }
            Self::Null { .. } => {}
            Self::Logical { operands, .. } => {
                if operands.is_empty() {
                    return Err(format!(
                        "{path}.operands must not be empty for a logical WHERE expression"
                    ));
                }
                for (index, operand) in operands.iter().enumerate() {
                    operand.validate(&format!("{path}.operands[{index}]"))?;
                }
            }
            Self::Not { operand } => operand.validate(&format!("{path}.operand"))?,
        }
        Ok(())
    }
}

impl TypedExpression {
    fn validate(&self, path: &str) -> Result<(), String> {
        match self {
            Self::Literal { span, origin, .. } | Self::Name { span, origin, .. } => {
                validate_expression_spans(span, origin.as_ref(), path)?;
            }
            Self::Unary {
                span,
                origin,
                operand,
                ..
            } => {
                validate_expression_spans(span, origin.as_ref(), path)?;
                operand.validate(&format!("{path}.operand"))?;
            }
            Self::Binary {
                span,
                origin,
                left,
                right,
                ..
            } => {
                validate_expression_spans(span, origin.as_ref(), path)?;
                left.validate(&format!("{path}.left"))?;
                right.validate(&format!("{path}.right"))?;
            }
            Self::Call {
                span,
                origin,
                arguments,
                ..
            } => {
                validate_expression_spans(span, origin.as_ref(), path)?;
                for (index, argument) in arguments.iter().enumerate() {
                    argument.validate(&format!("{path}.arguments[{index}]"))?;
                }
            }
            Self::Subscript {
                span,
                origin,
                base,
                index,
            } => {
                validate_expression_spans(span, origin.as_ref(), path)?;
                base.validate(&format!("{path}.base"))?;
                index.validate(&format!("{path}.index"))?;
            }
            Self::Concat {
                span,
                origin,
                parts,
            } => {
                validate_expression_spans(span, origin.as_ref(), path)?;
                if parts.len() < 2 {
                    return Err(format!(
                        "{path}.parts must contain at least two expressions"
                    ));
                }
                for (index, part) in parts.iter().enumerate() {
                    part.validate(&format!("{path}.parts[{index}]"))?;
                }
            }
            Self::Deferred {
                span,
                origin,
                apply_references,
                ..
            } => {
                validate_expression_spans(span, origin.as_ref(), path)?;
                for (index, reference) in apply_references.iter().enumerate() {
                    reference.validate(&format!("{path}.apply_references[{index}]"))?;
                }
            }
        }
        Ok(())
    }
}

fn validate_expression_spans(
    span: &TypedSpan,
    origin: Option<&TypedSpan>,
    path: &str,
) -> Result<(), String> {
    span.validate(&format!("{path}.span"))?;
    if let Some(origin) = origin {
        origin.validate(&format!("{path}.origin"))?;
    }
    Ok(())
}

impl TypedAssignmentTarget {
    fn validate(&self, path: &str) -> Result<(), String> {
        match self {
            Self::Name { span, .. } => span.validate(&format!("{path}.span")),
            Self::ArrayElement { index, span, .. } => {
                index.validate(&format!("{path}.index"))?;
                span.validate(&format!("{path}.span"))
            }
        }
    }

    fn is_name(&self) -> bool {
        matches!(self, Self::Name { .. })
    }
}

impl TypedInlineAction {
    fn validate(&self, path: &str) -> Result<(), String> {
        match self {
            Self::Statement { statement } => statement.validate(&format!("{path}.statement")),
            Self::Deferred { span, .. } => span.validate(&format!("{path}.span")),
        }
    }
}

impl TypedStatementPayload {
    fn validate_for_kind(&self, kind: &str, path: &str) -> Result<(), String> {
        let expected = match kind {
            "DATA" => "data",
            "SET" | "MERGE" => "source",
            "WHERE" => "where",
            "ASSIGN" => "assign",
            "SUM" => "sum",
            "KEEP" | "DROP" | "BY" | "RETAIN" => "name_list",
            "LABEL" => "label",
            "RENAME" => "rename",
            "OUTPUT" => "output",
            "IF" | "ELSE IF" => "if",
            "ELSE" => "else",
            "DO" => "do",
            "END" => "end",
            "ARRAY" => "array",
            "DELETE" => "delete",
            "STOP" => "stop",
            "RUN" => "run",
            "SKIPPED" => "skipped",
            _ => return Err(format!("{path}.kind has unknown statement kind {kind:?}")),
        };
        if let Self::Deferred { span, .. } = self {
            return span.validate(&format!("{path}.payload.span"));
        }
        if self.tag() != expected {
            return Err(format!(
                "{path}.payload.tag is {:?}; statement kind {kind:?} requires {expected:?}",
                self.tag()
            ));
        }

        match self {
            Self::Data { outputs } | Self::Output { targets: outputs } => {
                validate_dataset_references(outputs, &format!("{path}.payload"))?;
            }
            Self::Source { refs, .. } => {
                validate_dataset_references(refs, &format!("{path}.payload.refs"))?;
            }
            Self::Where { condition } => {
                condition.validate(&format!("{path}.payload.condition"))?
            }
            Self::Assign { target, value } => {
                target.validate(&format!("{path}.payload.target"))?;
                value.validate(&format!("{path}.payload.value"))?;
            }
            Self::Sum { target, increment } => {
                target.validate(&format!("{path}.payload.target"))?;
                if !target.is_name() {
                    return Err(format!(
                        "{path}.payload.target must use the name variant for SUM"
                    ));
                }
                increment.validate(&format!("{path}.payload.increment"))?;
            }
            Self::NameList { names } => {
                for (index, name) in names.iter().enumerate() {
                    if !matches!(name, TypedExpression::Name { .. }) {
                        return Err(format!(
                            "{path}.payload.names[{index}] must use the name expression variant"
                        ));
                    }
                    name.validate(&format!("{path}.payload.names[{index}]"))?;
                }
            }
            Self::Label {}
            | Self::Delete {}
            | Self::Stop {}
            | Self::End {}
            | Self::Run {}
            | Self::Skipped {} => {}
            Self::Rename { .. } => {}
            Self::If {
                condition,
                then_action,
                ..
            } => {
                condition.validate(&format!("{path}.payload.condition"))?;
                if let Some(action) = then_action {
                    action.validate(&format!("{path}.payload.then_action"))?;
                }
            }
            Self::Else { action } => {
                if let Some(action) = action {
                    action.validate(&format!("{path}.payload.action"))?;
                }
            }
            Self::Do {
                loop_var,
                start,
                end,
            } => {
                if loop_var.is_some() != start.is_some() || start.is_some() != end.is_some() {
                    return Err(format!(
                        "{path}.payload DO loop variable and bounds must be all present or all null"
                    ));
                }
                if let Some(start) = start {
                    start.validate(&format!("{path}.payload.start"))?;
                }
                if let Some(end) = end {
                    end.validate(&format!("{path}.payload.end"))?;
                }
            }
            Self::Array { span, .. } => span.validate(&format!("{path}.payload.span"))?,
            Self::Deferred { .. } => unreachable!("deferred payload returned above"),
        }
        Ok(())
    }

    fn tag(&self) -> &'static str {
        match self {
            Self::Data { .. } => "data",
            Self::Source { .. } => "source",
            Self::Where { .. } => "where",
            Self::Assign { .. } => "assign",
            Self::Sum { .. } => "sum",
            Self::NameList { .. } => "name_list",
            Self::Label {} => "label",
            Self::Rename { .. } => "rename",
            Self::Output { .. } => "output",
            Self::If { .. } => "if",
            Self::Else { .. } => "else",
            Self::Do { .. } => "do",
            Self::Array { .. } => "array",
            Self::Delete {} => "delete",
            Self::Stop {} => "stop",
            Self::End {} => "end",
            Self::Run {} => "run",
            Self::Skipped {} => "skipped",
            Self::Deferred { .. } => "deferred",
        }
    }
}

fn validate_dataset_references(
    references: &[TypedDatasetReference],
    path: &str,
) -> Result<(), String> {
    for (index, reference) in references.iter().enumerate() {
        reference
            .options
            .validate(&format!("{path}[{index}].options"))?;
        reference.span.validate(&format!("{path}[{index}].span"))?;
    }
    Ok(())
}

impl TypedStatement {
    fn validate(&self, path: &str) -> Result<(), String> {
        self.span.validate(&format!("{path}.span"))?;
        if let Some(origin) = &self.origin {
            origin.validate(&format!("{path}.origin"))?;
        }
        for (index, reference) in self.apply_references.iter().enumerate() {
            reference.validate(&format!("{path}.apply_references[{index}]"))?;
        }
        self.payload.validate_for_kind(&self.kind, path)
    }
}

fn dataset_key(value: &str) -> String {
    let normalized = value.trim().to_uppercase();
    normalized
        .strip_prefix("WORK.")
        .unwrap_or(&normalized)
        .to_string()
}

/// Deserialize and validate the internal version-one typed program.
///
/// This is a pure transport boundary: it validates typed fields and limited
/// structural relationships without parsing display text, applying dataset
/// policy, or accessing runtime state.
pub(crate) fn parse_typed_program(payload: &str) -> Result<TypedAstPayload, String> {
    let payload: TypedAstPayload =
        serde_json::from_str(payload).map_err(|error| error.to_string())?;
    payload.validate()?;
    Ok(payload)
}

/// Lower typed expressions without consulting display text or execution state.
pub(crate) fn lower_expression(expression: &TypedExpression) -> Result<Expr, String> {
    match expression {
        TypedExpression::Literal { kind, lexeme, .. } => match kind {
            TypedLiteralKind::Format => Ok(lower_format_literal(lexeme)),
            _ => lower_literal(*kind, lexeme).map(Expr::Literal),
        },
        TypedExpression::Name { name, .. } => match name.to_lowercase().as_str() {
            "true" => Ok(Expr::Literal(ScalarValue::Bool(true))),
            "false" => Ok(Expr::Literal(ScalarValue::Bool(false))),
            "none" | "null" => Ok(Expr::Literal(ScalarValue::Null)),
            _ => Ok(Expr::Variable(name.clone())),
        },
        TypedExpression::Unary {
            operator, operand, ..
        } => {
            let operand = lower_expression(operand)?;
            match operator {
                TypedUnaryOperator::Not => Ok(Expr::UnaryNot(Box::new(operand))),
                TypedUnaryOperator::Negate => Ok(lower_negation(operand)),
            }
        }
        TypedExpression::Binary {
            operator,
            left,
            right,
            ..
        } => Ok(Expr::Binary {
            left: Box::new(lower_expression(left)?),
            op: lower_binary_operator(*operator),
            right: Box::new(lower_expression(right)?),
        }),
        TypedExpression::Deferred { native_cause, .. } => Ok(Expr::DeferredError {
            cause: native_cause.clone(),
        }),
        TypedExpression::Call {
            name, arguments, ..
        } => Ok(Expr::FunctionCall {
            name: name.clone(),
            args: arguments
                .iter()
                .map(lower_expression)
                .collect::<Result<Vec<_>, _>>()?,
        }),
        TypedExpression::Subscript { base, index, .. } => {
            let name = match lower_expression(base)? {
                Expr::Variable(name) => name,
                _ => return Err("typed subscript base must be a name".to_string()),
            };
            Ok(Expr::ArrayRef {
                name,
                index: Box::new(lower_expression(index)?),
            })
        }
        TypedExpression::Concat { parts, .. } => {
            if parts.len() < 2 {
                return Err("typed concat requires at least two parts".to_string());
            }
            let mut lowered_parts = parts.iter().map(lower_expression);
            let mut result = lowered_parts.next().expect("concat length was checked")?;
            for part in lowered_parts {
                result = Expr::FunctionCall {
                    name: "__dsl_concat__".to_string(),
                    args: vec![result, part?],
                };
            }
            Ok(result)
        }
    }
}

fn lower_literal(kind: TypedLiteralKind, lexeme: &str) -> Result<ScalarValue, String> {
    match kind {
        TypedLiteralKind::Missing if lexeme == "." => Ok(ScalarValue::Null),
        TypedLiteralKind::Missing => Err(format!("invalid missing-value lexeme: {lexeme:?}")),
        TypedLiteralKind::String => lower_string_literal(lexeme).map(ScalarValue::Text),
        TypedLiteralKind::Numeric => lower_numeric_literal(lexeme),
        TypedLiteralKind::Format => Err("format literal must use expression lowering".to_string()),
    }
}

fn lower_format_literal(lexeme: &str) -> Expr {
    if lexeme
        .as_bytes()
        .first()
        .is_some_and(|first| first.is_ascii_alphabetic() || matches!(first, b'_' | b'$'))
    {
        Expr::Variable(lexeme.to_string())
    } else {
        Expr::Literal(ScalarValue::Text(lexeme.to_string()))
    }
}

fn lower_numeric_literal(lexeme: &str) -> Result<ScalarValue, String> {
    let integer_digits = lexeme.strip_prefix('-').unwrap_or(lexeme);
    if !integer_digits.is_empty() && integer_digits.bytes().all(|byte| byte.is_ascii_digit()) {
        if let Ok(value) = lexeme.parse::<i64>() {
            return Ok(ScalarValue::Integer(value));
        }
        if !lexeme.starts_with('-') {
            if let Ok(value) = lexeme.parse::<u64>() {
                return Ok(ScalarValue::Unsigned(value));
            }
        }
        return Err(format!("invalid integer literal: {lexeme}"));
    }

    let decimal_lexeme = lexeme.strip_prefix('-').unwrap_or(lexeme);
    let is_decimal = decimal_lexeme.contains('.')
        && decimal_lexeme.bytes().any(|byte| byte.is_ascii_digit())
        && decimal_lexeme
            .bytes()
            .all(|byte| byte.is_ascii_digit() || byte == b'.');
    if is_decimal {
        return lexeme
            .parse::<f64>()
            .map(ScalarValue::Number)
            .map_err(|_| format!("invalid number literal: {lexeme}"));
    }
    Err(format!("invalid numeric literal: {lexeme}"))
}

fn lower_string_literal(lexeme: &str) -> Result<String, String> {
    if is_native_numdot_text(lexeme) {
        return Ok(lexeme.to_string());
    }

    let bytes = lexeme.as_bytes();
    if bytes.len() < 2 || !matches!(bytes[0], b'\'' | b'"') || bytes[bytes.len() - 1] != bytes[0] {
        return Err(format!("invalid quoted string literal: {lexeme:?}"));
    }

    Ok(lexeme[1..bytes.len() - 1].to_string())
}

fn is_native_numdot_text(lexeme: &str) -> bool {
    let bytes = lexeme.as_bytes();
    if bytes.len() < 2 || bytes[bytes.len() - 1] != b'.' {
        return false;
    }
    if !bytes
        .iter()
        .all(|byte| byte.is_ascii_digit() || *byte == b'.')
    {
        return false;
    }

    bytes[0].is_ascii_digit() || (bytes[0] == b'.' && bytes[1].is_ascii_digit())
}

fn lower_negation(operand: Expr) -> Expr {
    match operand {
        Expr::Literal(ScalarValue::Integer(value)) => match value.checked_neg() {
            Some(negated) => Expr::Literal(ScalarValue::Integer(negated)),
            None => Expr::UnaryNeg(Box::new(Expr::Literal(ScalarValue::Integer(value)))),
        },
        Expr::Literal(ScalarValue::Unsigned(value))
            if value <= (i64::MAX as u64).saturating_add(1) =>
        {
            Expr::Literal(ScalarValue::Integer(-(value as i128) as i64))
        }
        other => Expr::UnaryNeg(Box::new(other)),
    }
}

fn lower_binary_operator(operator: TypedBinaryOperator) -> BinaryOp {
    match operator {
        TypedBinaryOperator::Or => BinaryOp::Or,
        TypedBinaryOperator::And => BinaryOp::And,
        TypedBinaryOperator::Equal => BinaryOp::Eq,
        TypedBinaryOperator::NotEqual => BinaryOp::Ne,
        TypedBinaryOperator::Greater => BinaryOp::Gt,
        TypedBinaryOperator::GreaterEqual => BinaryOp::Ge,
        TypedBinaryOperator::Less => BinaryOp::Lt,
        TypedBinaryOperator::LessEqual => BinaryOp::Le,
        TypedBinaryOperator::Add => BinaryOp::Add,
        TypedBinaryOperator::Subtract => BinaryOp::Sub,
        TypedBinaryOperator::Multiply => BinaryOp::Mul,
        TypedBinaryOperator::Divide => BinaryOp::Div,
        TypedBinaryOperator::Power => BinaryOp::Pow,
        TypedBinaryOperator::Modulo => BinaryOp::Mod,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        lower_expression, parse_typed_program, TypedExpression, TypedInlineAction,
        TypedStatementPayload,
    };
    use crate::expressions::{BinaryOp, Expr, ScalarValue};
    use serde_json::{json, Value};

    fn typed_payload_fixture() -> Value {
        serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/typed_payload_v1.json"
        )))
        .expect("shared typed payload fixture is valid JSON")
    }

    fn valid_program() -> Value {
        typed_payload_fixture()["valid_program"].clone()
    }

    fn expression_span() -> Value {
        json!({
            "start": 0,
            "end": 1,
            "line": 1,
            "column": 1,
            "end_line": 1,
            "end_column": 2,
            "source_id": "typed-lowering-test"
        })
    }

    fn literal(kind: &str, lexeme: &str) -> Value {
        json!({
            "tag": "literal",
            "span": expression_span(),
            "origin": null,
            "kind": kind,
            "lexeme": lexeme
        })
    }

    fn name(name: &str) -> Value {
        json!({
            "tag": "name",
            "span": expression_span(),
            "origin": null,
            "name": name
        })
    }

    fn unary(operator: &str, operand: Value) -> Value {
        json!({
            "tag": "unary",
            "span": expression_span(),
            "origin": null,
            "operator": operator,
            "operand": operand
        })
    }

    fn binary(operator: &str, left: Value, right: Value) -> Value {
        json!({
            "tag": "binary",
            "span": expression_span(),
            "origin": null,
            "operator": operator,
            "left": left,
            "right": right
        })
    }

    fn call(name: &str, arguments: Vec<Value>) -> Value {
        json!({
            "tag": "call",
            "span": expression_span(),
            "origin": null,
            "name": name,
            "arguments": arguments
        })
    }

    fn subscript(base: Value, index: Value) -> Value {
        json!({
            "tag": "subscript",
            "span": expression_span(),
            "origin": null,
            "base": base,
            "index": index
        })
    }

    fn concat(parts: Vec<Value>) -> Value {
        json!({
            "tag": "concat",
            "span": expression_span(),
            "origin": null,
            "parts": parts
        })
    }

    fn lower_fixture_expression(expression: Value) -> Result<Expr, String> {
        let mut program = valid_program();
        program["statements"][4]["payload"]["value"] = expression;
        let parsed = parse_typed_program(&program.to_string())?;
        match &parsed.statements[4].payload {
            TypedStatementPayload::Assign { value, .. } => lower_expression(value),
            _ => Err("fixture statement is not ASSIGN".to_string()),
        }
    }

    fn binary_op_name(operator: &BinaryOp) -> &'static str {
        match operator {
            BinaryOp::Or => "or",
            BinaryOp::And => "and",
            BinaryOp::Eq => "==",
            BinaryOp::Ne => "!=",
            BinaryOp::Gt => ">",
            BinaryOp::Ge => ">=",
            BinaryOp::Lt => "<",
            BinaryOp::Le => "<=",
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
            BinaryOp::Pow => "**",
            BinaryOp::Mod => "mod",
        }
    }

    fn deferred(cause: &str) -> Value {
        json!({
            "tag": "deferred",
            "span": expression_span(),
            "origin": null,
            "error_kind": "unsupported_expression",
            "native_cause": cause,
            "where_syntax_summary": "saved summary",
            "display_text": "display only",
            "apply_references": []
        })
    }

    #[test]
    fn typed_lower_preserves_right_associative_power_structure() {
        let expression = binary(
            "**",
            literal("numeric", "2"),
            binary("**", literal("numeric", "2"), literal("numeric", "3")),
        );
        let lowered = lower_fixture_expression(expression).unwrap();

        match lowered {
            Expr::Binary { left, op, right } => {
                assert_eq!(binary_op_name(&op), "**");
                assert!(matches!(*left, Expr::Literal(ScalarValue::Integer(2))));
                match *right {
                    Expr::Binary { left, op, right } => {
                        assert_eq!(binary_op_name(&op), "**");
                        assert!(matches!(*left, Expr::Literal(ScalarValue::Integer(2))));
                        assert!(matches!(*right, Expr::Literal(ScalarValue::Integer(3))));
                    }
                    other => panic!("expected nested right-hand power, got {other:?}"),
                }
            }
            other => panic!("expected binary power, got {other:?}"),
        }
    }

    #[test]
    fn typed_lower_maps_reserved_boolean_and_missing_names_like_legacy() {
        for keyword in ["true", "TRUE", "TrUe"] {
            assert!(matches!(
                lower_fixture_expression(name(keyword)).unwrap(),
                Expr::Literal(ScalarValue::Bool(true))
            ));
        }
        for keyword in ["false", "FALSE", "FaLsE"] {
            assert!(matches!(
                lower_fixture_expression(name(keyword)).unwrap(),
                Expr::Literal(ScalarValue::Bool(false))
            ));
        }
        for keyword in ["none", "None", "NULL", "null"] {
            assert!(matches!(
                lower_fixture_expression(name(keyword)).unwrap(),
                Expr::Literal(ScalarValue::Null)
            ));
        }
        assert!(matches!(
            lower_fixture_expression(name("truthy")).unwrap(),
            Expr::Variable(value) if value == "truthy"
        ));
    }

    #[test]
    fn typed_lower_preserves_integer_boundaries_and_negative_minimum() {
        assert!(matches!(
            lower_fixture_expression(literal("numeric", "9223372036854775807")).unwrap(),
            Expr::Literal(ScalarValue::Integer(i64::MAX))
        ));
        assert!(matches!(
            lower_fixture_expression(literal("numeric", "9223372036854775808")).unwrap(),
            Expr::Literal(ScalarValue::Unsigned(9_223_372_036_854_775_808))
        ));
        assert!(matches!(
            lower_fixture_expression(literal("numeric", "18446744073709551615")).unwrap(),
            Expr::Literal(ScalarValue::Unsigned(u64::MAX))
        ));
        assert!(matches!(
            lower_fixture_expression(unary("-", literal("numeric", "9223372036854775808")))
                .unwrap(),
            Expr::Literal(ScalarValue::Integer(i64::MIN))
        ));
        assert!(lower_fixture_expression(literal("numeric", "18446744073709551616")).is_err());
        assert!(lower_fixture_expression(literal("numeric", "1e3")).is_err());
        assert!(lower_fixture_expression(literal("numeric", "1.2.3")).is_err());

        assert!(matches!(
            lower_fixture_expression(literal("numeric", "1.25")).unwrap(),
            Expr::Literal(ScalarValue::Number(value)) if value == 1.25
        ));
    }

    #[test]
    fn typed_lower_preserves_string_lexeme_contents_without_reparsing_source() {
        assert!(matches!(
            lower_fixture_expression(literal("string", "'東京 🦑'")).unwrap(),
            Expr::Literal(ScalarValue::Text(value)) if value == "東京 🦑"
        ));
        assert!(matches!(
            lower_fixture_expression(literal("string", r#"'C:\\tmp'"#)).unwrap(),
            Expr::Literal(ScalarValue::Text(value)) if value == r"C:\\tmp"
        ));
        assert!(matches!(
            lower_fixture_expression(literal("missing", ".")).unwrap(),
            Expr::Literal(ScalarValue::Null)
        ));
        assert!(matches!(
            lower_fixture_expression(literal("string", "'.'")).unwrap(),
            Expr::Literal(ScalarValue::Text(value)) if value == "."
        ));
        assert!(matches!(
            lower_fixture_expression(name("column")).unwrap(),
            Expr::Variable(value) if value == "column"
        ));
        assert!(lower_fixture_expression(literal("missing", "not-missing")).is_err());
    }

    #[test]
    fn typed_lower_preserves_numdot_text_as_string() {
        assert!(matches!(
            lower_fixture_expression(literal("string", "5.")).unwrap(),
            Expr::Literal(ScalarValue::Text(value)) if value == "5."
        ));
    }

    #[test]
    fn typed_lower_calls_preserve_runtime_names_and_nested_arguments() {
        let lowered = lower_fixture_expression(call(
            "UnknownFn",
            vec![name("column"), call("inner", vec![literal("missing", ".")])],
        ))
        .unwrap();
        assert!(matches!(
            lowered,
            Expr::FunctionCall { name, args }
                if name == "UnknownFn"
                    && matches!(args.as_slice(),
                        [Expr::Variable(column), Expr::FunctionCall { name: inner, args: inner_args }]
                            if column == "column"
                                && inner == "inner"
                                && matches!(inner_args.as_slice(), [Expr::Literal(ScalarValue::Null)]))
        ));

        let array_call =
            lower_fixture_expression(call("items", vec![literal("numeric", "2")])).unwrap();
        assert!(matches!(
            array_call,
            Expr::FunctionCall { name, args }
                if name == "items"
                    && matches!(args.as_slice(), [Expr::Literal(ScalarValue::Integer(2))])
        ));
    }

    #[test]
    fn typed_lower_subscript_builds_named_array_reference() {
        let lowered = lower_fixture_expression(subscript(
            name("items"),
            binary("+", literal("numeric", "1"), literal("numeric", "2")),
        ))
        .unwrap();
        assert!(matches!(
            lowered,
            Expr::ArrayRef { name, index }
                if name == "items"
                    && matches!(*index, Expr::Binary { op: BinaryOp::Add, .. })
        ));
        assert!(lower_fixture_expression(subscript(
            call("array_factory", Vec::new()),
            literal("numeric", "1")
        ))
        .is_err());
    }

    #[test]
    fn typed_lower_format_token_matches_legacy_token_shape() {
        assert!(matches!(
            lower_fixture_expression(literal("format", "z8.2.")).unwrap(),
            Expr::Variable(format) if format == "z8.2."
        ));
        assert!(matches!(
            lower_fixture_expression(literal("format", "8.")).unwrap(),
            Expr::Literal(ScalarValue::Text(format)) if format == "8."
        ));
    }

    #[test]
    fn typed_lower_concat_left_folds_parts_without_reinterpreting_them() {
        let lowered = lower_fixture_expression(concat(vec![
            literal("string", "'left'"),
            literal("missing", "."),
            literal("string", "'right || + **'"),
        ]))
        .unwrap();
        assert!(matches!(
            lowered,
            Expr::FunctionCall { name, args }
                if name == "__dsl_concat__"
                    && matches!(args.as_slice(), [
                        Expr::FunctionCall { name: first_helper, args: first_args },
                        Expr::Literal(ScalarValue::Text(right))
                    ] if first_helper == "__dsl_concat__"
                        && right == "right || + **"
                        && matches!(first_args.as_slice(), [
                            Expr::Literal(ScalarValue::Text(left)),
                            Expr::Literal(ScalarValue::Null)
                        ] if left == "left"))
        ));
    }

    #[test]
    fn typed_lower_maps_existing_unary_and_binary_operators() {
        let operators = [
            ("or", "or"),
            ("and", "and"),
            ("==", "=="),
            ("!=", "!="),
            (">", ">"),
            (">=", ">="),
            ("<", "<"),
            ("<=", "<="),
            ("+", "+"),
            ("-", "-"),
            ("*", "*"),
            ("/", "/"),
            ("**", "**"),
            ("mod", "mod"),
        ];
        for (wire_operator, expected_operator) in operators {
            match lower_fixture_expression(binary(wire_operator, name("left"), name("right")))
                .unwrap()
            {
                Expr::Binary { op, .. } => {
                    assert_eq!(binary_op_name(&op), expected_operator, "{wire_operator}");
                }
                other => panic!("expected binary expression, got {other:?}"),
            }
        }

        assert!(matches!(
            lower_fixture_expression(unary("not", name("flag"))).unwrap(),
            Expr::UnaryNot(inner) if matches!(*inner, Expr::Variable(ref value) if value == "flag")
        ));
        assert!(matches!(
            lower_fixture_expression(unary("-", name("value"))).unwrap(),
            Expr::UnaryNeg(inner) if matches!(*inner, Expr::Variable(ref value) if value == "value")
        ));
    }

    #[test]
    fn typed_lower_keeps_deferred_expression_opaque_until_evaluation() {
        let cause = "unsupported expression: saved cause";
        assert!(matches!(
            lower_fixture_expression(deferred(cause)).unwrap(),
            Expr::DeferredError { cause: saved } if saved == cause
        ));
    }

    fn value_at_path_mut<'a>(value: &'a mut Value, path: &[Value]) -> &'a mut Value {
        let mut current = value;
        for segment in path {
            current = match segment {
                Value::String(key) => current
                    .as_object_mut()
                    .and_then(|object| object.get_mut(key))
                    .expect("fixture mutation object path exists"),
                Value::Number(index) => current
                    .as_array_mut()
                    .and_then(|items| {
                        items.get_mut(index.as_u64().expect("integer index") as usize)
                    })
                    .expect("fixture mutation array path exists"),
                _ => panic!("fixture mutation path must contain only keys and indices"),
            };
        }
        current
    }

    fn apply_mutation(program: &mut Value, mutation: &Value) {
        let operation = mutation["operation"].as_str().expect("mutation operation");
        let path = mutation["path"].as_array().expect("mutation path");
        let (last, parents) = path.split_last().expect("non-empty mutation path");
        let parent = value_at_path_mut(program, parents);
        match operation {
            "replace" => {
                *value_at_path_mut(program, path) = mutation["value"].clone();
            }
            "remove" => match (parent, last) {
                (Value::Object(object), Value::String(key)) => {
                    object.remove(key).expect("fixture object key exists");
                }
                (Value::Array(items), Value::Number(index)) => {
                    items.remove(index.as_u64().expect("integer index") as usize);
                }
                _ => panic!("fixture remove path has a compatible parent"),
            },
            "insert" => match (parent, last) {
                (Value::Object(object), Value::String(key)) => {
                    object.insert(key.clone(), mutation["value"].clone());
                }
                _ => panic!("fixture insert path targets an object"),
            },
            other => panic!("unsupported fixture mutation operation {other:?}"),
        }
    }

    fn remove_path(value: &mut Value, path: &[Value]) {
        let (last, parents) = path.split_last().expect("non-empty null field path");
        let parent = value_at_path_mut(value, parents);
        let Value::Object(object) = parent else {
            panic!("required nullable fields occur as object properties")
        };
        let Value::String(key) = last else {
            panic!("required nullable field path ends with an object key")
        };
        object.remove(key).expect("null field key exists");
    }

    fn collect_null_field_paths(
        value: &Value,
        path: &mut Vec<Value>,
        result: &mut Vec<Vec<Value>>,
    ) {
        match value {
            Value::Object(object) => {
                for (key, child) in object {
                    path.push(Value::String(key.clone()));
                    if child.is_null() {
                        result.push(path.clone());
                    } else {
                        collect_null_field_paths(child, path, result);
                    }
                    path.pop();
                }
            }
            Value::Array(items) => {
                for (index, child) in items.iter().enumerate() {
                    path.push(Value::from(index));
                    collect_null_field_paths(child, path, result);
                    path.pop();
                }
            }
            _ => {}
        }
    }

    fn expression_has_numeric_lexeme(expression: &TypedExpression, expected: &str) -> bool {
        match expression {
            TypedExpression::Literal {
                kind: super::TypedLiteralKind::Numeric,
                lexeme,
                ..
            } => lexeme == expected,
            TypedExpression::Unary { operand, .. } => {
                expression_has_numeric_lexeme(operand, expected)
            }
            TypedExpression::Binary { left, right, .. } => {
                expression_has_numeric_lexeme(left, expected)
                    || expression_has_numeric_lexeme(right, expected)
            }
            TypedExpression::Call { arguments, .. } => arguments
                .iter()
                .any(|argument| expression_has_numeric_lexeme(argument, expected)),
            TypedExpression::Subscript { base, index, .. } => {
                expression_has_numeric_lexeme(base, expected)
                    || expression_has_numeric_lexeme(index, expected)
            }
            TypedExpression::Concat { parts, .. } => parts
                .iter()
                .any(|part| expression_has_numeric_lexeme(part, expected)),
            TypedExpression::Literal { .. }
            | TypedExpression::Name { .. }
            | TypedExpression::Deferred { .. } => false,
        }
    }

    #[test]
    fn typed_payload_accepts_shared_valid_fixture_and_all_variants() {
        let program = valid_program();
        let parsed = parse_typed_program(&program.to_string())
            .expect("the shared version-one fixture must be accepted");

        assert_eq!(parsed.version, 1);
        assert_eq!(parsed.output_targets.len(), 3);
        assert_eq!(
            parsed.default_target.as_deref(),
            Some("explicit-unregistered-default")
        );
        assert_eq!(parsed.statements.len(), 28);
        assert_eq!(parsed.statements[0].display_text, "payload");
        assert_eq!(parsed.statements[0].span.source_id, "typed-payload-fixture");
        assert!(parsed.statements[0].origin.is_none());
        assert!(parsed.statements[4].origin.is_some());
        assert_eq!(parsed.statements[4].apply_references[0].target, "lookup");
        assert!(parsed
            .statements
            .iter()
            .any(|statement| matches!(statement.payload, TypedStatementPayload::Deferred { .. })));
        assert!(parsed.statements.iter().any(|statement| matches!(
            statement.payload,
            TypedStatementPayload::If {
                then_action: Some(TypedInlineAction::Statement { .. }),
                ..
            }
        )));
        assert!(parsed.statements.iter().any(|statement| matches!(
            statement.payload,
            TypedStatementPayload::If {
                then_action: Some(TypedInlineAction::Deferred { .. }),
                ..
            }
        )));
        assert!(parsed.statements.iter().any(|statement| matches!(
            statement.payload,
            TypedStatementPayload::If {
                then_action: None,
                ..
            }
        )));
        assert!(parsed.statements.iter().any(|statement| matches!(
            statement.payload,
            TypedStatementPayload::Else {
                action: Some(TypedInlineAction::Deferred { .. }),
            }
        )));
        let has_concat = parsed
            .statements
            .iter()
            .any(|statement| match &statement.payload {
                TypedStatementPayload::Assign { value, .. } => {
                    matches!(value, TypedExpression::Concat { .. })
                }
                _ => false,
            });
        assert!(has_concat);
        match &parsed.statements[4].payload {
            TypedStatementPayload::Assign {
                value: TypedExpression::Concat { parts, .. },
                ..
            } => {
                assert_eq!(parts.len(), 3);
                assert!(matches!(
                    &parts[0],
                    TypedExpression::Literal {
                        kind: super::TypedLiteralKind::String,
                        lexeme,
                        ..
                    } if lexeme == "'東京 🦑'"
                ));
                assert!(matches!(
                    &parts[1],
                    TypedExpression::Name { name, .. } if name == "value"
                ));
                assert!(matches!(
                    &parts[2],
                    TypedExpression::Literal {
                        kind: super::TypedLiteralKind::Missing,
                        lexeme,
                        ..
                    } if lexeme == "."
                ));
            }
            _ => panic!("fixture assignment must preserve its typed concat parts"),
        }
        match &parsed.statements[20].payload {
            TypedStatementPayload::Do {
                start: Some(expression),
                ..
            } => assert!(expression_has_numeric_lexeme(
                expression,
                "9223372036854775808"
            )),
            _ => panic!("fixture DO statement must preserve typed bounds"),
        }
    }

    #[test]
    fn typed_payload_rejects_shared_invalid_fixtures() {
        let fixture = typed_payload_fixture();
        let invalid_cases = fixture["invalid_cases"]
            .as_array()
            .expect("invalid fixture recipes");
        for case in invalid_cases {
            let name = case["name"].as_str().expect("invalid case name");
            let mut program = valid_program();
            apply_mutation(&mut program, &case["mutation"]);
            assert!(
                parse_typed_program(&program.to_string()).is_err(),
                "invalid fixture case {name:?} must be rejected"
            );
        }
    }

    #[test]
    fn typed_payload_accepts_explicit_null_and_rejects_missing_nullable_fields() {
        let mut program = valid_program();
        program["default_target"] = Value::Null;
        assert!(parse_typed_program(&program.to_string()).is_ok());

        let mut nullable_paths = Vec::new();
        collect_null_field_paths(&valid_program(), &mut Vec::new(), &mut nullable_paths);
        assert!(!nullable_paths.is_empty());
        for path in nullable_paths {
            let mut without_field = valid_program();
            remove_path(&mut without_field, &path);
            assert!(
                parse_typed_program(&without_field.to_string()).is_err(),
                "omitted required nullable field {path:?} must be rejected"
            );
        }

        for field in ["end_line", "end_column"] {
            let mut with_null = valid_program();
            with_null["statements"][0]["span"][field] = Value::Null;
            assert!(parse_typed_program(&with_null.to_string()).is_ok());

            let mut missing = valid_program();
            remove_path(
                &mut missing,
                &[
                    Value::from("statements"),
                    Value::from(0),
                    Value::from("span"),
                    Value::from(field),
                ],
            );
            assert!(parse_typed_program(&missing.to_string()).is_err());
        }
    }

    #[test]
    fn typed_payload_rejects_invalid_span_geometry_and_partial_do_bounds() {
        let mut program = valid_program();
        program["statements"][0]["span"]["start"] = Value::from(5);
        program["statements"][0]["span"]["end"] = Value::from(4);
        assert!(parse_typed_program(&program.to_string()).is_err());

        let mut program = valid_program();
        program["statements"][0]["span"]["line"] = Value::from(0);
        assert!(parse_typed_program(&program.to_string()).is_err());

        let mut program = valid_program();
        program["statements"][20]["payload"]["start"] = Value::Null;
        assert!(parse_typed_program(&program.to_string()).is_err());
    }

    #[test]
    fn typed_payload_rejects_invalid_deferred_statement_span_geometry() {
        let mut program = valid_program();
        program["statements"][5]["payload"]["span"]["start"] = Value::from(5);
        program["statements"][5]["payload"]["span"]["end"] = Value::from(4);
        assert!(parse_typed_program(&program.to_string()).is_err());

        for field in ["line", "column"] {
            let mut program = valid_program();
            program["statements"][5]["payload"]["span"][field] = Value::from(0);
            assert!(parse_typed_program(&program.to_string()).is_err());
        }
    }

    #[test]
    fn typed_payload_rejects_mismatched_statement_variant_and_short_concat() {
        let mut program = valid_program();
        program["statements"][0]["kind"] = Value::from("ASSIGN");
        assert!(parse_typed_program(&program.to_string()).is_err());

        let mut program = valid_program();
        program["statements"][5]["kind"] = Value::from("UNKNOWN");
        assert!(parse_typed_program(&program.to_string()).is_err());

        let mut program = valid_program();
        program["statements"][4]["payload"]["value"]["parts"] = Value::Array(Vec::new());
        assert!(parse_typed_program(&program.to_string()).is_err());
    }
}
