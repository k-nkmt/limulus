//! Pure lowering from validated versioned syntax into native execution adapters.
//!
//! Runtime option carriers are reused where their contracts match; statement
//! execution itself consumes only the lowered typed variants below.

use crate::ast::{
    AstDatasetRefOptions, AstStatementOptions, AstWhereComparisonOperator, AstWhereExpression,
    AstWhereLogicalOperator, AstWhereScalar,
};
use crate::expressions::Expr;
use crate::typed_ast::{
    TypedAssignmentTarget, TypedAstPayload, TypedDatasetOptions, TypedDatasetReference,
    TypedExpression, TypedInteger, TypedStatement, TypedStatementPayload,
    TypedWhereComparisonOperator, TypedWhereExpression, TypedWhereLogicalOperator,
};

#[derive(Debug, Clone)]
pub(crate) struct LoweredProgram {
    pub(crate) version: u32,
    pub(crate) output_targets: Vec<String>,
    pub(crate) default_target: Option<String>,
    pub(crate) suppress_implicit_output: bool,
    pub(crate) effective_output_options: std::collections::BTreeMap<String, AstDatasetRefOptions>,
    pub(crate) statements: Vec<LoweredStatement>,
}

#[derive(Debug, Clone)]
pub(crate) struct LoweredStatement {
    pub(crate) kind: String,
    pub(crate) display_text: String,
    pub(crate) span: crate::typed_ast::TypedSpan,
    pub(crate) origin: Option<crate::typed_ast::TypedSpan>,
    pub(crate) apply_references: Vec<crate::typed_ast::TypedApplyReference>,
    pub(crate) payload: LoweredStatementPayload,
}

#[derive(Debug, Clone)]
pub(crate) enum LoweredStatementPayload {
    Data {
        outputs: Vec<LoweredDatasetReference>,
    },
    Source {
        refs: Vec<LoweredDatasetReference>,
        options: AstStatementOptions,
    },
    Where {
        condition: LoweredExpression,
    },
    Assign {
        target: LoweredAssignmentTarget,
        value: LoweredExpression,
    },
    Sum {
        target: LoweredAssignmentTarget,
        increment: LoweredExpression,
    },
    NameList {
        names: Vec<LoweredName>,
    },
    Label,
    Rename {
        rename_map: std::collections::HashMap<String, String>,
    },
    Output {
        targets: Vec<LoweredDatasetReference>,
    },
    If {
        condition: LoweredExpression,
        then_action: Option<LoweredInlineAction>,
        is_subset: bool,
        is_then_do: bool,
    },
    Else {
        action: Option<LoweredInlineAction>,
    },
    Do {
        loop_var: Option<String>,
        start: Option<LoweredExpression>,
        end: Option<LoweredExpression>,
    },
    Array {
        declaration: LoweredArrayDeclaration,
    },
    Delete,
    Stop,
    End,
    Run,
    Skipped,
    Deferred {
        descriptor: DeferredStatementDescriptor,
    },
}

#[derive(Debug, Clone)]
pub(crate) enum LoweredInlineAction {
    Statement(Box<LoweredStatement>),
    Deferred(DeferredStatementDescriptor),
}

#[derive(Debug, Clone)]
pub(crate) struct LoweredDatasetReference {
    pub(crate) name: String,
    pub(crate) options: AstDatasetRefOptions,
    pub(crate) span: crate::typed_ast::TypedSpan,
}

#[derive(Debug, Clone)]
pub(crate) struct LoweredArrayDeclaration {
    pub(crate) array_name: String,
    pub(crate) variables: Vec<String>,
    pub(crate) span: crate::typed_ast::TypedSpan,
    pub(crate) declared_size: Option<TypedInteger>,
    pub(crate) wildcard_size: bool,
    pub(crate) character_array: bool,
}

#[derive(Debug, Clone)]
pub(crate) enum LoweredAssignmentTarget {
    Name {
        name: String,
        span: crate::typed_ast::TypedSpan,
    },
    ArrayElement {
        name: String,
        delimiter: crate::typed_ast::TypedArrayDelimiter,
        index: LoweredExpression,
        span: crate::typed_ast::TypedSpan,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct LoweredExpression {
    pub(crate) expression: Expr,
    // Retaining the validated source tree keeps deferred diagnostics complete
    // without using display text as executable syntax.
    pub(crate) source: crate::typed_ast::TypedExpression,
}

#[derive(Debug, Clone)]
pub(crate) struct LoweredName {
    pub(crate) name: String,
    pub(crate) span: crate::typed_ast::TypedSpan,
    pub(crate) origin: Option<crate::typed_ast::TypedSpan>,
}

#[derive(Debug, Clone)]
pub(crate) struct DeferredStatementDescriptor {
    pub(crate) error_kind: String,
    pub(crate) native_cause: String,
    pub(crate) where_syntax_summary: String,
    pub(crate) span: crate::typed_ast::TypedSpan,
    pub(crate) display_text: String,
}

pub(crate) fn lower_program(payload: &TypedAstPayload) -> Result<LoweredProgram, String> {
    Ok(LoweredProgram {
        version: payload.version,
        // These values were selected by the output-plan owner. Lowering copies
        // them verbatim and never selects a default from the target list.
        output_targets: payload.output_targets.clone(),
        default_target: payload.default_target.clone(),
        suppress_implicit_output: payload.suppress_implicit_output,
        effective_output_options: payload
            .effective_output_options
            .iter()
            .map(|(target, options)| Ok((target.clone(), lower_dataset_options(options)?)))
            .collect::<Result<_, String>>()?,
        statements: payload
            .statements
            .iter()
            .map(lower_statement)
            .collect::<Result<_, _>>()?,
    })
}

fn lower_statement(statement: &TypedStatement) -> Result<LoweredStatement, String> {
    let payload = match &statement.payload {
        TypedStatementPayload::Data { outputs } if statement.kind == "DATA" => {
            LoweredStatementPayload::Data {
                outputs: outputs
                    .iter()
                    .map(lower_dataset_reference)
                    .collect::<Result<_, _>>()?,
            }
        }
        TypedStatementPayload::Source { refs, options }
            if matches!(statement.kind.as_str(), "SET" | "MERGE") =>
        {
            LoweredStatementPayload::Source {
                refs: refs
                    .iter()
                    .map(lower_dataset_reference)
                    .collect::<Result<_, _>>()?,
                options: AstStatementOptions {
                    indsname_var: options.indsname_var.clone(),
                    end_var: options.end_var.clone(),
                },
            }
        }
        TypedStatementPayload::Where { condition } if statement.kind == "WHERE" => {
            LoweredStatementPayload::Where {
                condition: lower_typed_expression(condition)?,
            }
        }
        TypedStatementPayload::Assign { target, value } if statement.kind == "ASSIGN" => {
            LoweredStatementPayload::Assign {
                target: lower_assignment_target(target)?,
                value: lower_typed_expression(value)?,
            }
        }
        TypedStatementPayload::Sum { target, increment } if statement.kind == "SUM" => {
            LoweredStatementPayload::Sum {
                target: lower_assignment_target(target)?,
                increment: lower_typed_expression(increment)?,
            }
        }
        TypedStatementPayload::NameList { names }
            if matches!(statement.kind.as_str(), "KEEP" | "DROP" | "BY" | "RETAIN") =>
        {
            LoweredStatementPayload::NameList {
                names: names.iter().map(lower_name).collect::<Result<_, _>>()?,
            }
        }
        TypedStatementPayload::Label {} if statement.kind == "LABEL" => {
            LoweredStatementPayload::Label
        }
        TypedStatementPayload::Rename { rename_map } if statement.kind == "RENAME" => {
            LoweredStatementPayload::Rename {
                rename_map: rename_map
                    .iter()
                    .map(|(source, target)| (source.clone(), target.clone()))
                    .collect(),
            }
        }
        TypedStatementPayload::Output { targets } if statement.kind == "OUTPUT" => {
            LoweredStatementPayload::Output {
                targets: targets
                    .iter()
                    .map(lower_dataset_reference)
                    .collect::<Result<_, _>>()?,
            }
        }
        TypedStatementPayload::If {
            condition,
            then_action,
            is_subset,
            is_then_do,
        } if matches!(statement.kind.as_str(), "IF" | "ELSE IF") => LoweredStatementPayload::If {
            condition: lower_typed_expression(condition)?,
            then_action: then_action.as_ref().map(lower_inline_action).transpose()?,
            is_subset: *is_subset,
            is_then_do: *is_then_do,
        },
        TypedStatementPayload::Else { action } if statement.kind == "ELSE" => {
            LoweredStatementPayload::Else {
                action: action.as_ref().map(lower_inline_action).transpose()?,
            }
        }
        TypedStatementPayload::Do {
            loop_var,
            start,
            end,
        } if statement.kind == "DO" => lower_do(loop_var, start, end)?,
        TypedStatementPayload::Array {
            array_name,
            variables,
            span,
            declared_size,
            wildcard_size,
            character_array,
        } if statement.kind == "ARRAY" => LoweredStatementPayload::Array {
            declaration: LoweredArrayDeclaration {
                array_name: array_name.clone(),
                variables: variables.clone(),
                span: span.clone(),
                declared_size: declared_size.clone(),
                wildcard_size: *wildcard_size,
                character_array: *character_array,
            },
        },
        TypedStatementPayload::Delete {} if statement.kind == "DELETE" => {
            LoweredStatementPayload::Delete
        }
        TypedStatementPayload::Stop {} if statement.kind == "STOP" => LoweredStatementPayload::Stop,
        TypedStatementPayload::End {} if statement.kind == "END" => LoweredStatementPayload::End,
        TypedStatementPayload::Run {} if statement.kind == "RUN" => LoweredStatementPayload::Run,
        TypedStatementPayload::Skipped {} if statement.kind == "SKIPPED" => {
            LoweredStatementPayload::Skipped
        }
        TypedStatementPayload::Deferred {
            error_kind,
            native_cause,
            where_syntax_summary,
            span,
            display_text,
        } => LoweredStatementPayload::Deferred {
            descriptor: DeferredStatementDescriptor {
                error_kind: error_kind.clone(),
                native_cause: native_cause.clone(),
                where_syntax_summary: where_syntax_summary.clone(),
                span: span.clone(),
                display_text: display_text.clone(),
            },
        },
        _ => {
            return Err(format!(
                "typed statement kind {:?} does not match its payload",
                statement.kind
            ));
        }
    };

    Ok(LoweredStatement {
        kind: statement.kind.clone(),
        display_text: statement.display_text.clone(),
        span: statement.span.clone(),
        origin: statement.origin.clone(),
        apply_references: statement.apply_references.clone(),
        payload,
    })
}

fn lower_inline_action(
    action: &crate::typed_ast::TypedInlineAction,
) -> Result<LoweredInlineAction, String> {
    Ok(match action {
        crate::typed_ast::TypedInlineAction::Statement { statement } => {
            LoweredInlineAction::Statement(Box::new(lower_statement(statement)?))
        }
        crate::typed_ast::TypedInlineAction::Deferred {
            error_kind,
            native_cause,
            where_syntax_summary,
            span,
            display_text,
        } => LoweredInlineAction::Deferred(DeferredStatementDescriptor {
            error_kind: error_kind.clone(),
            native_cause: native_cause.clone(),
            where_syntax_summary: where_syntax_summary.clone(),
            span: span.clone(),
            display_text: display_text.clone(),
        }),
    })
}

fn lower_do(
    loop_var: &Option<String>,
    start: &Option<TypedExpression>,
    end: &Option<TypedExpression>,
) -> Result<LoweredStatementPayload, String> {
    Ok(LoweredStatementPayload::Do {
        loop_var: loop_var.clone(),
        start: start.as_ref().map(lower_typed_expression).transpose()?,
        end: end.as_ref().map(lower_typed_expression).transpose()?,
    })
}

fn lower_dataset_reference(
    reference: &TypedDatasetReference,
) -> Result<LoweredDatasetReference, String> {
    Ok(LoweredDatasetReference {
        name: reference.name.clone(),
        options: lower_dataset_options(&reference.options)?,
        span: reference.span.clone(),
    })
}

fn lower_dataset_options(options: &TypedDatasetOptions) -> Result<AstDatasetRefOptions, String> {
    Ok(AstDatasetRefOptions {
        in_var: options.in_var.clone(),
        keep_vars: options.keep_vars.clone(),
        drop_vars: options.drop_vars.clone(),
        where_expr: options
            .where_expr
            .as_ref()
            .map(lower_where_expression)
            .transpose()?,
        rename_map: options
            .rename_map
            .iter()
            .map(|(source, target)| (source.clone(), target.clone()))
            .collect(),
        firstobs: lower_native_count(options.firstobs.as_ref(), "FIRSTOBS")?,
        obs: lower_native_count(options.obs.as_ref(), "OBS")?,
    })
}

fn lower_native_count(value: Option<&TypedInteger>, field: &str) -> Result<Option<i64>, String> {
    match value {
        None => Ok(None),
        Some(TypedInteger::Signed(value)) => Ok(Some(*value)),
        Some(TypedInteger::Unsigned(value)) => i64::try_from(*value)
            .map(Some)
            .map_err(|_| format!("{field} value {value} is outside the native i64 range")),
    }
}

fn lower_where_expression(expression: &TypedWhereExpression) -> Result<AstWhereExpression, String> {
    Ok(match expression {
        TypedWhereExpression::Comparison {
            column,
            operator,
            value,
        } => AstWhereExpression::Comparison {
            column: column.clone(),
            operator: match operator {
                TypedWhereComparisonOperator::Eq => AstWhereComparisonOperator::Eq,
                TypedWhereComparisonOperator::Equal => AstWhereComparisonOperator::Equal,
                TypedWhereComparisonOperator::NotEqual => AstWhereComparisonOperator::NotEqual,
                TypedWhereComparisonOperator::Greater => AstWhereComparisonOperator::Greater,
                TypedWhereComparisonOperator::GreaterEqual => {
                    AstWhereComparisonOperator::GreaterEqual
                }
                TypedWhereComparisonOperator::Less => AstWhereComparisonOperator::Less,
                TypedWhereComparisonOperator::LessEqual => AstWhereComparisonOperator::LessEqual,
            },
            value: lower_where_scalar(value)?,
        },
        TypedWhereExpression::Null { column, negated } => AstWhereExpression::Null {
            column: column.clone(),
            negated: *negated,
        },
        TypedWhereExpression::Logical { operator, operands } => AstWhereExpression::Logical {
            operator: match operator {
                TypedWhereLogicalOperator::And => AstWhereLogicalOperator::And,
                TypedWhereLogicalOperator::Or => AstWhereLogicalOperator::Or,
            },
            operands: operands
                .iter()
                .map(lower_where_expression)
                .collect::<Result<_, _>>()?,
        },
        TypedWhereExpression::Not { operand } => AstWhereExpression::Not {
            operand: Box::new(lower_where_expression(operand)?),
        },
    })
}

fn lower_where_scalar(value: &serde_json::Value) -> Result<AstWhereScalar, String> {
    match value {
        serde_json::Value::String(value) => Ok(AstWhereScalar::Text(value.clone())),
        serde_json::Value::Bool(value) => Ok(AstWhereScalar::Bool(*value)),
        serde_json::Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                Ok(AstWhereScalar::Integer(value))
            } else if let Some(value) = value.as_u64() {
                Ok(AstWhereScalar::Unsigned(value))
            } else if let Some(value) = value.as_f64() {
                Ok(AstWhereScalar::Number(value))
            } else {
                Err("typed WHERE number is not representable by the native scalar carrier".into())
            }
        }
        _ => Err("typed WHERE value must be a JSON scalar".into()),
    }
}

fn lower_assignment_target(
    target: &TypedAssignmentTarget,
) -> Result<LoweredAssignmentTarget, String> {
    Ok(match target {
        TypedAssignmentTarget::Name { name, span } => LoweredAssignmentTarget::Name {
            name: name.clone(),
            span: span.clone(),
        },
        TypedAssignmentTarget::ArrayElement {
            name,
            delimiter,
            index,
            span,
        } => LoweredAssignmentTarget::ArrayElement {
            name: name.clone(),
            delimiter: *delimiter,
            index: lower_typed_expression(index)?,
            span: span.clone(),
        },
    })
}

fn lower_typed_expression(expression: &TypedExpression) -> Result<LoweredExpression, String> {
    Ok(LoweredExpression {
        expression: crate::typed_ast::lower_expression(expression)?,
        source: expression.clone(),
    })
}

fn lower_name(expression: &TypedExpression) -> Result<LoweredName, String> {
    match expression {
        TypedExpression::Name {
            name, span, origin, ..
        } => Ok(LoweredName {
            name: name.clone(),
            span: span.clone(),
            origin: origin.clone(),
        }),
        _ => Err("name-list lowering requires typed name expressions".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        lower_program, LoweredAssignmentTarget, LoweredInlineAction, LoweredStatementPayload,
    };
    use crate::ast::{
        AstWhereComparisonOperator, AstWhereExpression, AstWhereLogicalOperator, AstWhereScalar,
    };
    use crate::expressions::{BinaryOp, Expr, ScalarValue};
    use crate::typed_ast::{parse_typed_program, TypedExpression};
    use serde_json::{json, Value};

    fn fixture_program() -> Value {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/typed_payload_v1.json"
        )))
        .expect("shared typed payload fixture is valid JSON");
        fixture["valid_program"].clone()
    }

    fn parse_kinds(kinds: &[&str]) -> crate::typed_ast::TypedAstPayload {
        parse_selected(fixture_program(), |statement| {
            statement["kind"]
                .as_str()
                .is_some_and(|kind| kinds.contains(&kind))
        })
    }

    fn parse_selected(
        mut value: Value,
        predicate: impl Fn(&Value) -> bool,
    ) -> crate::typed_ast::TypedAstPayload {
        let statements = value["statements"]
            .as_array()
            .expect("fixture statements are an array")
            .iter()
            .filter(|statement| predicate(statement))
            .cloned()
            .collect::<Vec<_>>();
        value["statements"] = json!(statements);
        parse_program(value)
    }

    fn parse_program(value: Value) -> crate::typed_ast::TypedAstPayload {
        parse_typed_program(&value.to_string()).expect("selected fixture statements are valid")
    }

    fn statement_index(value: &Value, kind: &str, occurrence: usize) -> usize {
        value["statements"]
            .as_array()
            .expect("fixture statements are an array")
            .iter()
            .enumerate()
            .filter(|(_, statement)| statement["kind"] == kind)
            .nth(occurrence)
            .map(|(index, _)| index)
            .expect("fixture has the requested statement")
    }

    fn statement_value(value: &Value, kind: &str, occurrence: usize) -> Value {
        value["statements"][statement_index(value, kind, occurrence)].clone()
    }

    fn statement_value_mut<'a>(
        value: &'a mut Value,
        kind: &str,
        occurrence: usize,
    ) -> &'a mut Value {
        let index = statement_index(value, kind, occurrence);
        &mut value["statements"][index]
    }

    #[test]
    fn typed_statement_lower_maps_assign_sum_refs_names_and_output_plan() {
        let parsed = parse_kinds(&[
            "DATA", "SET", "MERGE", "ASSIGN", "SUM", "KEEP", "DROP", "BY", "RETAIN", "LABEL",
            "RENAME", "OUTPUT",
        ]);
        let lowered = lower_program(&parsed).expect("supported typed statements lower");

        assert_eq!(
            lowered.output_targets,
            ["WORK.OUT", "work.out", "work.empty"]
        );
        assert_eq!(
            lowered.default_target.as_deref(),
            Some("explicit-unregistered-default")
        );
        assert!(lowered.suppress_implicit_output);
        assert_eq!(lowered.effective_output_options.len(), 2);
        assert!(lowered.effective_output_options.contains_key("work.out"));
        assert!(lowered.effective_output_options.contains_key("WORK.EMPTY"));
        assert_eq!(lowered.statements.len(), 13);
        assert_eq!(lowered.version, 1);
        assert_eq!(lowered.statements[0].kind, "DATA");
        assert_eq!(lowered.statements[0].display_text, "payload");
        assert_eq!(
            lowered.statements[0].span.source_id,
            "typed-payload-fixture"
        );
        assert_eq!(
            lowered.statements[3].origin.as_ref().unwrap().source_id,
            "typed-origin"
        );
        assert_eq!(lowered.statements[3].apply_references.len(), 1);
        assert_eq!(lowered.statements[3].apply_references[0].target, "lookup");
        assert!(matches!(
            lowered.statements[3].payload,
            LoweredStatementPayload::Assign { .. }
        ));
        assert!(matches!(
            lowered.statements[4].payload,
            LoweredStatementPayload::Deferred { .. }
        ));
        assert!(matches!(
            lowered.statements[5].payload,
            LoweredStatementPayload::Sum { .. }
        ));

        match &lowered.statements[0].payload {
            LoweredStatementPayload::Data { outputs } => {
                assert_eq!(outputs[0].name, "out");
                assert_eq!(outputs[0].options.keep_vars, ["id"]);
            }
            other => panic!("expected DATA refs, got {other:?}"),
        }
        match &lowered.statements[1].payload {
            LoweredStatementPayload::Source { refs, options } => {
                assert_eq!(refs[0].name, "input");
                assert_eq!(refs[0].options.in_var.as_deref(), Some("seen"));
                assert_eq!(refs[0].options.keep_vars, ["id"]);
                assert_eq!(refs[0].options.drop_vars, ["raw"]);
                assert_eq!(refs[0].options.firstobs, Some(2));
                assert_eq!(refs[0].options.obs, Some(5));
                assert_eq!(
                    refs[0].options.rename_map.get("id").map(String::as_str),
                    Some("key")
                );
                assert_eq!(options.indsname_var.as_deref(), Some("source"));
                assert_eq!(options.end_var.as_deref(), Some("last"));
                match refs[0].options.where_expr.as_ref().unwrap() {
                    AstWhereExpression::Logical {
                        operator: AstWhereLogicalOperator::And,
                        operands,
                    } => {
                        assert_eq!(operands.len(), 2);
                        assert!(matches!(
                            &operands[0],
                            AstWhereExpression::Comparison {
                                column,
                                operator: AstWhereComparisonOperator::GreaterEqual,
                                value: AstWhereScalar::Integer(1),
                            } if column == "id"
                        ));
                    }
                    other => panic!("expected restrictive logical WHERE, got {other:?}"),
                }
            }
            other => panic!("expected SET refs, got {other:?}"),
        }
        match &lowered.statements[3].payload {
            LoweredStatementPayload::Assign { target, value } => {
                match target {
                    LoweredAssignmentTarget::ArrayElement {
                        name,
                        delimiter: crate::typed_ast::TypedArrayDelimiter::Brackets,
                        index,
                        ..
                    } => {
                        assert_eq!(name, "items");
                        assert!(matches!(
                            &index.expression,
                            Expr::ArrayRef { name, index }
                                if name == "items"
                                    && matches!(index.as_ref(), Expr::Literal(ScalarValue::Unsigned(9_223_372_036_854_775_808)))
                        ));
                        assert!(matches!(&index.source, TypedExpression::Subscript { .. }));
                    }
                    other => panic!("expected array-element target, got {other:?}"),
                }
                assert!(
                    matches!(&value.expression, Expr::FunctionCall { name, .. } if name == "__dsl_concat__")
                );
                assert!(matches!(&value.source, TypedExpression::Concat { .. }));
            }
            other => panic!("expected ASSIGN, got {other:?}"),
        }
        match &lowered.statements[4].payload {
            LoweredStatementPayload::Deferred { descriptor } => {
                assert_eq!(descriptor.error_kind, "unsupported_inline_action");
                assert_eq!(
                    descriptor.native_cause,
                    "unsupported inline action: opaque action"
                );
                assert_eq!(descriptor.where_syntax_summary, "saved action summary");
                assert_eq!(descriptor.display_text, "opaque action");
                assert_eq!(descriptor.span.source_id, "typed-payload-fixture");
            }
            other => panic!("expected deferred descriptor, got {other:?}"),
        }
        match &lowered.statements[5].payload {
            LoweredStatementPayload::Sum { target, .. } => {
                assert!(
                    matches!(target, LoweredAssignmentTarget::Name { name, .. } if name == "total")
                );
            }
            other => panic!("expected SUM target, got {other:?}"),
        }
        match &lowered.statements[9].payload {
            LoweredStatementPayload::NameList { names } => {
                assert_eq!(
                    names
                        .iter()
                        .map(|name| name.name.as_str())
                        .collect::<Vec<_>>(),
                    ["total"]
                );
                assert_eq!(names[0].span.source_id, "typed-payload-fixture");
            }
            other => panic!("expected RETAIN names, got {other:?}"),
        }
        match &lowered.statements[12].payload {
            LoweredStatementPayload::Output { targets } => {
                assert_eq!(targets[0].name, "out");
                assert_eq!(targets[0].options.keep_vars, ["id"]);
            }
            other => panic!("expected OUTPUT refs, got {other:?}"),
        }
        let effective = &lowered.effective_output_options["work.out"];
        assert_eq!(effective.keep_vars, ["id"]);
        assert_eq!(effective.drop_vars, ["raw"]);
        assert_eq!(effective.firstobs, Some(2));
        assert_eq!(effective.obs, Some(5));
        assert!(effective.in_var.is_none());
    }

    #[test]
    fn typed_statement_lower_retains_where_deferred_metadata_without_source_parsing() {
        let parsed = parse_kinds(&["WHERE"]);
        let lowered = lower_program(&parsed).expect("WHERE expression lowers from structure");
        let statement = &lowered.statements[0];
        assert_eq!(statement.display_text, "payload");
        assert_eq!(statement.apply_references.len(), 1);
        match &statement.payload {
            LoweredStatementPayload::Where { condition } => {
                assert!(matches!(
                    &condition.expression,
                    Expr::DeferredError { ref cause } if cause == "unsupported expression syntax"
                ));
                match &condition.source {
                    TypedExpression::Deferred {
                        error_kind,
                        native_cause,
                        where_syntax_summary,
                        display_text,
                        apply_references,
                        ..
                    } => {
                        assert_eq!(error_kind, "unsupported_expression");
                        assert_eq!(native_cause, "unsupported expression syntax");
                        assert_eq!(where_syntax_summary, "saved where summary");
                        assert_eq!(display_text, "display-only expression");
                        assert_eq!(apply_references.len(), 1);
                    }
                    other => panic!("expected retained typed deferred source, got {other:?}"),
                }
            }
            other => panic!("expected WHERE statement, got {other:?}"),
        }
    }

    #[test]
    fn typed_statement_lower_preserves_name_tokens_and_rejects_count_truncation() {
        let mut value = fixture_program();
        let statements = value["statements"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|statement| statement["kind"] == "RETAIN")
            .cloned()
            .collect::<Vec<_>>();
        value["statements"] = json!(statements);
        let retain = value["statements"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|statement| statement["kind"] == "RETAIN")
            .unwrap();
        let mut first = retain["payload"]["names"][0].clone();
        first["name"] = json!(".");
        let mut second = first.clone();
        second["name"] = json!("1e3");
        retain["payload"]["names"] = json!([first, second]);
        let parsed = parse_typed_program(&value.to_string()).unwrap();
        let lowered = lower_program(&parsed).unwrap();
        let retain = lowered
            .statements
            .iter()
            .find(|statement| statement.kind == "RETAIN")
            .unwrap();
        assert!(matches!(
            &retain.payload,
            LoweredStatementPayload::NameList { names }
                if names.iter().map(|name| name.name.as_str()).collect::<Vec<_>>() == [".", "1e3"]
        ));

        let mut out_of_range = fixture_program();
        out_of_range["effective_output_options"]["work.out"]["firstobs"] =
            json!(9_223_372_036_854_775_808_u64);
        let parsed = parse_typed_program(&out_of_range.to_string()).unwrap();
        assert!(lower_program(&parsed)
            .unwrap_err()
            .contains("FIRSTOBS value 9223372036854775808"));
    }

    #[test]
    fn typed_statement_lower_preserves_bounded_do_expressions_and_deferred_specs() {
        let mut fixture = fixture_program();
        let mut numeric_do = statement_value(&fixture, "DO", 0);
        let text_null_do = statement_value(&fixture, "DO", 0);
        let deferred_bound_do = statement_value(&fixture, "DO", 0);
        let bare_do = statement_value(&fixture, "DO", 1);
        let end_statement = statement_value(&fixture, "END", 0);

        let outer_span = json!({
            "start": 40,
            "end": 48,
            "line": 4,
            "column": 1,
            "end_line": 4,
            "end_column": 9,
            "source_id": "do-unicode-source"
        });
        let outer_origin = json!({
            "start": 41,
            "end": 43,
            "line": 4,
            "column": 2,
            "end_line": 4,
            "end_column": 4,
            "source_id": "do-origin"
        });
        numeric_do["display_text"] = json!("do 🦑");
        numeric_do["span"] = outer_span.clone();
        numeric_do["origin"] = outer_origin.clone();
        numeric_do["payload"]["start"]["span"] = outer_span.clone();
        numeric_do["payload"]["start"]["origin"] = outer_origin.clone();

        let text_span = json!({
            "start": 60,
            "end": 74,
            "line": 5,
            "column": 1,
            "end_line": 5,
            "end_column": 15,
            "source_id": "do-text-source"
        });
        let text_origin = json!({
            "start": 61,
            "end": 65,
            "line": 5,
            "column": 2,
            "end_line": 5,
            "end_column": 6,
            "source_id": "do-text-origin"
        });
        let mut text_null_do = text_null_do;
        text_null_do["payload"]["loop_var"] = json!("CountCase");
        text_null_do["payload"]["start"] = json!({
            "tag": "literal",
            "span": text_span.clone(),
            "origin": text_origin.clone(),
            "kind": "string",
            "lexeme": "'not numeric'"
        });
        text_null_do["payload"]["end"] = json!({
            "tag": "literal",
            "span": text_span.clone(),
            "origin": text_origin.clone(),
            "kind": "missing",
            "lexeme": "."
        });

        let mut deferred_bound_do = deferred_bound_do;
        deferred_bound_do["payload"]["loop_var"] = json!("DeferredIndex");
        deferred_bound_do["payload"]["start"] = json!({
            "tag": "deferred",
            "span": text_span.clone(),
            "origin": text_origin.clone(),
            "error_kind": "unsupported_expression",
            "native_cause": "unsupported DO boundary syntax",
            "where_syntax_summary": "saved DO boundary summary",
            "display_text": "display-only bound",
            "apply_references": [{
                "target": "deferred.host",
                "call_span": text_span.clone(),
                "owner_span": outer_span.clone()
            }]
        });

        let mut whole_deferred_do = numeric_do.clone();
        whole_deferred_do["display_text"] = json!("do unsupported spec");
        whole_deferred_do["apply_references"] = json!([{
            "target": "whole.spec.host",
            "call_span": text_span.clone(),
            "owner_span": outer_span.clone()
        }]);
        whole_deferred_do["payload"] = json!({
            "tag": "deferred",
            "error_kind": "unsupported_do_spec",
            "native_cause": "unsupported DO statement: unsupported spec",
            "where_syntax_summary": "saved whole DO spec summary",
            "span": text_span.clone(),
            "display_text": "display-only whole DO spec"
        });

        fixture["statements"] = json!([
            numeric_do,
            text_null_do,
            deferred_bound_do,
            whole_deferred_do,
            bare_do,
            end_statement
        ]);

        let lowered = lower_program(&parse_program(fixture))
            .expect("typed DO boundaries and deferred specs lower without execution");
        assert_eq!(
            lowered
                .statements
                .iter()
                .map(|statement| statement.kind.as_str())
                .collect::<Vec<_>>(),
            ["DO", "DO", "DO", "DO", "DO", "END"]
        );
        assert_eq!(lowered.statements[0].display_text, "do 🦑");
        assert_eq!(lowered.statements[0].span.start, 40);
        assert_eq!(lowered.statements[0].span.end, 48);
        assert_eq!(
            lowered.statements[0].origin.as_ref().unwrap().source_id,
            "do-origin"
        );

        match &lowered.statements[0].payload {
            LoweredStatementPayload::Do {
                loop_var: Some(loop_var),
                start: Some(start),
                end: Some(end),
            } => {
                assert_eq!(loop_var, "index");
                assert!(matches!(
                    &start.source,
                    TypedExpression::Binary {
                        left,
                        right,
                        ..
                    } if matches!(left.as_ref(), TypedExpression::Name { name, .. } if name == "value")
                        && matches!(right.as_ref(), TypedExpression::Binary { .. })
                ));
                assert!(matches!(
                    &start.expression,
                    Expr::Binary {
                        op: BinaryOp::Pow,
                        right,
                        ..
                    } if matches!(right.as_ref(), Expr::Binary { op: BinaryOp::Pow, .. })
                ));
                assert!(matches!(
                    &end.source,
                    TypedExpression::Literal {
                        kind: crate::typed_ast::TypedLiteralKind::Numeric,
                        lexeme,
                        origin: Some(origin),
                        ..
                    } if lexeme == "9223372036854775808" && origin.source_id == "typed-origin"
                ));
            }
            other => panic!("expected bounded numeric DO payload, got {other:?}"),
        }

        match &lowered.statements[1].payload {
            LoweredStatementPayload::Do {
                loop_var: Some(loop_var),
                start: Some(start),
                end: Some(end),
            } => {
                assert_eq!(loop_var, "CountCase");
                assert!(matches!(
                    &start.source,
                    TypedExpression::Literal { lexeme, .. } if lexeme == "'not numeric'"
                ));
                assert!(matches!(
                    &start.expression,
                    Expr::Literal(ScalarValue::Text(value)) if value == "not numeric"
                ));
                assert!(matches!(
                    &end.source,
                    TypedExpression::Literal {
                        kind: crate::typed_ast::TypedLiteralKind::Missing,
                        lexeme,
                        ..
                    } if lexeme == "."
                ));
                assert!(matches!(&end.expression, Expr::Literal(ScalarValue::Null)));
            }
            other => panic!("expected text/null DO boundaries, got {other:?}"),
        }

        match &lowered.statements[2].payload {
            LoweredStatementPayload::Do {
                loop_var: Some(loop_var),
                start: Some(start),
                end: Some(_),
            } => {
                assert_eq!(loop_var, "DeferredIndex");
                assert!(matches!(
                    &start.expression,
                    Expr::DeferredError { cause } if cause == "unsupported DO boundary syntax"
                ));
                assert!(matches!(
                    &start.source,
                    TypedExpression::Deferred {
                        error_kind,
                        native_cause,
                        where_syntax_summary,
                        display_text,
                        span,
                        origin,
                        apply_references,
                    } if where_syntax_summary == "saved DO boundary summary"
                        && error_kind == "unsupported_expression"
                        && native_cause == "unsupported DO boundary syntax"
                        && display_text == "display-only bound"
                        && span.source_id == "do-text-source"
                        && origin.as_ref().is_some_and(|span| span.source_id == "do-text-origin")
                        && apply_references.len() == 1
                        && apply_references[0].target == "deferred.host"
                ));
            }
            other => panic!("expected deferred DO boundary, got {other:?}"),
        }

        match &lowered.statements[3].payload {
            LoweredStatementPayload::Deferred { descriptor } => {
                assert_eq!(descriptor.error_kind, "unsupported_do_spec");
                assert_eq!(
                    descriptor.native_cause,
                    "unsupported DO statement: unsupported spec"
                );
                assert_eq!(
                    descriptor.where_syntax_summary,
                    "saved whole DO spec summary"
                );
                assert_eq!(descriptor.display_text, "display-only whole DO spec");
                assert_eq!(descriptor.span.source_id, "do-text-source");
                assert_eq!(
                    lowered.statements[3].origin.as_ref().unwrap().source_id,
                    "do-origin"
                );
                assert_eq!(
                    lowered.statements[3].apply_references[0].target,
                    "whole.spec.host"
                );
            }
            other => panic!("expected whole DO deferred descriptor, got {other:?}"),
        }

        assert!(matches!(
            &lowered.statements[4].payload,
            LoweredStatementPayload::Do {
                loop_var: None,
                start: None,
                end: None
            }
        ));
        assert!(matches!(
            &lowered.statements[5].payload,
            LoweredStatementPayload::End
        ));
    }

    #[test]
    fn typed_statement_lower_accepts_array_declaration_and_array_expression() {
        let mut program = fixture_program();
        let mut explicit_array = statement_value(&program, "ARRAY", 0);
        let end = statement_value(&program, "END", 0);
        let mut assign = statement_value(&program, "ASSIGN", 0);
        let argument = assign["payload"]["value"].clone();
        assign["payload"]["value"] = json!({
            "tag": "call",
            "span": argument["span"].clone(),
            "origin": argument["origin"].clone(),
            "name": "items",
            "arguments": [argument]
        });

        let declaration_span = json!({
            "start": 30,
            "end": 45,
            "line": 3,
            "column": 1,
            "end_line": 3,
            "end_column": 16,
            "source_id": "array-declaration-source"
        });
        let statement_span = json!({
            "start": 28,
            "end": 46,
            "line": 3,
            "column": 1,
            "end_line": 3,
            "end_column": 19,
            "source_id": "array-statement-source"
        });
        let statement_origin = json!({
            "start": 29,
            "end": 34,
            "line": 3,
            "column": 2,
            "end_line": 3,
            "end_column": 7,
            "source_id": "array-origin"
        });
        explicit_array["display_text"] = json!("array 🦑");
        explicit_array["span"] = statement_span.clone();
        explicit_array["origin"] = statement_origin.clone();
        explicit_array["payload"]["array_name"] = json!("items");
        explicit_array["payload"]["variables"] = json!(["First", "Second", "id3"]);
        explicit_array["payload"]["span"] = declaration_span.clone();
        explicit_array["payload"]["declared_size"] = json!(u64::MAX);
        explicit_array["payload"]["wildcard_size"] = json!(false);
        explicit_array["payload"]["character_array"] = json!(true);

        let mut wildcard_array = explicit_array.clone();
        wildcard_array["display_text"] = json!("array *");
        wildcard_array["payload"]["array_name"] = json!("WideArray");
        wildcard_array["payload"]["variables"] = json!(["left", "RIGHT"]);
        wildcard_array["payload"]["declared_size"] = Value::Null;
        wildcard_array["payload"]["wildcard_size"] = json!(true);
        wildcard_array["payload"]["character_array"] = json!(false);

        let mut deferred_array = explicit_array.clone();
        deferred_array["display_text"] = json!("array unsupported declaration");
        deferred_array["apply_references"] = json!([{
            "target": "array.deferred.host",
            "call_span": declaration_span.clone(),
            "owner_span": statement_span.clone()
        }]);
        deferred_array["payload"] = json!({
            "tag": "deferred",
            "error_kind": "invalid_array_declaration",
            "native_cause": "unsupported ARRAY declaration: saved cause",
            "where_syntax_summary": "saved ARRAY declaration summary",
            "span": declaration_span.clone(),
            "display_text": "display-only ARRAY declaration"
        });

        program["statements"] =
            json!([assign, explicit_array, wildcard_array, deferred_array, end]);
        let parsed = parse_program(program);
        let lowered = lower_program(&parsed)
            .expect("ARRAY declarations lower without looking up runtime array bindings");
        assert_eq!(
            lowered
                .statements
                .iter()
                .map(|statement| statement.kind.as_str())
                .collect::<Vec<_>>(),
            ["ASSIGN", "ARRAY", "ARRAY", "ARRAY", "END"]
        );

        match &lowered.statements[0].payload {
            LoweredStatementPayload::Assign { target, value } => {
                assert!(matches!(
                    target,
                    LoweredAssignmentTarget::ArrayElement {
                        name,
                        index,
                        ..
                    } if name == "items"
                        && matches!(&index.expression, Expr::ArrayRef { name, .. } if name == "items")
                        && matches!(&index.source, TypedExpression::Subscript { .. })
                ));
                assert!(matches!(
                    &value.expression,
                    Expr::FunctionCall { name, args }
                        if name == "items"
                            && matches!(args.as_slice(), [Expr::FunctionCall { name, .. }] if name == "__dsl_concat__")
                ));
                assert!(
                    matches!(&value.source, TypedExpression::Call { name, .. } if name == "items")
                );
            }
            other => panic!("expected array-element assignment, got {other:?}"),
        }

        match &lowered.statements[1].payload {
            LoweredStatementPayload::Array { declaration } => {
                assert_eq!(declaration.array_name, "items");
                assert_eq!(declaration.variables, ["First", "Second", "id3"]);
                assert!(matches!(
                    declaration.declared_size.as_ref(),
                    Some(crate::typed_ast::TypedInteger::Unsigned(size)) if *size == u64::MAX
                ));
                assert!(!declaration.wildcard_size);
                assert!(declaration.character_array);
                assert_eq!(declaration.span.source_id, "array-declaration-source");
                assert_eq!(declaration.span.start, 30);
                assert_eq!(
                    lowered.statements[1].span.source_id,
                    "array-statement-source"
                );
                assert_eq!(
                    lowered.statements[1].origin.as_ref().unwrap().source_id,
                    "array-origin"
                );
                assert_eq!(lowered.statements[1].display_text, "array 🦑");
            }
            other => panic!("expected explicit ARRAY declaration, got {other:?}"),
        }

        match &lowered.statements[2].payload {
            LoweredStatementPayload::Array { declaration } => {
                assert_eq!(declaration.array_name, "WideArray");
                assert_eq!(declaration.variables, ["left", "RIGHT"]);
                assert!(declaration.declared_size.is_none());
                assert!(declaration.wildcard_size);
                assert!(!declaration.character_array);
            }
            other => panic!("expected wildcard ARRAY declaration, got {other:?}"),
        }

        match &lowered.statements[3].payload {
            LoweredStatementPayload::Deferred { descriptor } => {
                assert_eq!(descriptor.error_kind, "invalid_array_declaration");
                assert_eq!(
                    descriptor.native_cause,
                    "unsupported ARRAY declaration: saved cause"
                );
                assert_eq!(
                    descriptor.where_syntax_summary,
                    "saved ARRAY declaration summary"
                );
                assert_eq!(descriptor.display_text, "display-only ARRAY declaration");
                assert_eq!(descriptor.span.source_id, "array-declaration-source");
                assert_eq!(
                    lowered.statements[3].apply_references[0].target,
                    "array.deferred.host"
                );
            }
            other => panic!("expected whole ARRAY deferred descriptor, got {other:?}"),
        }
    }

    #[test]
    fn typed_statement_lower_accepts_every_statement_in_shared_fixture() {
        let parsed = parse_program(fixture_program());
        let lowered = lower_program(&parsed)
            .expect("every producer-supported statement in the shared fixture lowers");

        assert_eq!(lowered.version, parsed.version);
        assert_eq!(lowered.output_targets, parsed.output_targets);
        assert_eq!(lowered.default_target, parsed.default_target);
        assert_eq!(
            lowered.suppress_implicit_output,
            parsed.suppress_implicit_output
        );
        assert_eq!(lowered.statements.len(), parsed.statements.len());
        assert_eq!(
            lowered
                .statements
                .iter()
                .map(|statement| statement.kind.as_str())
                .collect::<Vec<_>>(),
            parsed
                .statements
                .iter()
                .map(|statement| statement.kind.as_str())
                .collect::<Vec<_>>()
        );

        let tags = lowered
            .statements
            .iter()
            .map(|statement| match &statement.payload {
                LoweredStatementPayload::Data { .. } => "data",
                LoweredStatementPayload::Source { .. } => "source",
                LoweredStatementPayload::Where { .. } => "where",
                LoweredStatementPayload::Assign { .. } => "assign",
                LoweredStatementPayload::Sum { .. } => "sum",
                LoweredStatementPayload::NameList { .. } => "name_list",
                LoweredStatementPayload::Label => "label",
                LoweredStatementPayload::Rename { .. } => "rename",
                LoweredStatementPayload::Output { .. } => "output",
                LoweredStatementPayload::If { .. } => "if",
                LoweredStatementPayload::Else { .. } => "else",
                LoweredStatementPayload::Do { .. } => "do",
                LoweredStatementPayload::Array { .. } => "array",
                LoweredStatementPayload::Delete => "delete",
                LoweredStatementPayload::Stop => "stop",
                LoweredStatementPayload::End => "end",
                LoweredStatementPayload::Run => "run",
                LoweredStatementPayload::Skipped => "skipped",
                LoweredStatementPayload::Deferred { .. } => "deferred",
            })
            .collect::<std::collections::BTreeSet<_>>();
        let expected = [
            "data",
            "source",
            "where",
            "assign",
            "sum",
            "name_list",
            "label",
            "rename",
            "output",
            "if",
            "else",
            "do",
            "array",
            "delete",
            "stop",
            "end",
            "run",
            "skipped",
            "deferred",
        ]
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(tags, expected);
    }

    #[test]
    fn typed_statement_lower_accepts_if_else_and_flat_action_markers() {
        let mut value = fixture_program();
        let second_if = statement_value_mut(&mut value, "IF", 1);
        second_if["kind"] = json!("ELSE IF");
        second_if["payload"]["is_subset"] = json!(true);
        let first_if = statement_value_mut(&mut value, "IF", 0);
        first_if["payload"]["is_subset"] = json!(true);
        let parsed = parse_selected(value, |statement| {
            matches!(
                statement["kind"].as_str(),
                Some("IF" | "ELSE IF" | "ELSE" | "END" | "DELETE" | "STOP" | "RUN" | "SKIPPED")
            )
        });
        let lowered = lower_program(&parsed).expect("IF/ELSE and flat markers lower structurally");
        assert_eq!(
            lowered
                .statements
                .iter()
                .map(|statement| statement.kind.as_str())
                .collect::<Vec<_>>(),
            [
                "IF", "ELSE IF", "IF", "ELSE", "ELSE", "ELSE", "END", "DELETE", "STOP", "RUN",
                "SKIPPED"
            ]
        );
        assert!(lowered.suppress_implicit_output);

        match &lowered.statements[0].payload {
            LoweredStatementPayload::If {
                condition,
                then_action: Some(LoweredInlineAction::Statement(statement)),
                is_subset,
                is_then_do,
            } => {
                assert!(*is_subset);
                assert!(!*is_then_do);
                assert!(matches!(
                    &condition.expression,
                    Expr::FunctionCall { name, .. } if name == "apply.custom"
                ));
                assert_eq!(statement.kind, "ASSIGN");
                assert!(matches!(
                    statement.payload,
                    LoweredStatementPayload::Assign { .. }
                ));
            }
            other => panic!("expected structured IF with inline ASSIGN, got {other:?}"),
        }
        match &lowered.statements[1].payload {
            LoweredStatementPayload::If {
                then_action: Some(LoweredInlineAction::Deferred(descriptor)),
                is_subset,
                ..
            } => {
                assert!(*is_subset);
                assert_eq!(
                    descriptor.native_cause,
                    "unsupported inline action: opaque action"
                );
                assert_eq!(descriptor.where_syntax_summary, "saved action summary");
            }
            other => panic!("expected ELSE IF with deferred action, got {other:?}"),
        }
        assert_eq!(lowered.statements[1].apply_references[0].target, "lookup");
        assert!(matches!(
            &lowered.statements[2].payload,
            LoweredStatementPayload::If {
                then_action: None,
                is_subset: false,
                is_then_do: false,
                ..
            }
        ));

        assert!(matches!(
            &lowered.statements[3].payload,
            LoweredStatementPayload::Else {
                action: Some(LoweredInlineAction::Statement(statement))
            } if statement.kind == "ASSIGN"
        ));
        assert!(matches!(
            &lowered.statements[4].payload,
            LoweredStatementPayload::Else {
                action: Some(LoweredInlineAction::Deferred(descriptor))
            } if descriptor.native_cause == "unsupported inline action: opaque action"
        ));
        assert!(matches!(
            &lowered.statements[5].payload,
            LoweredStatementPayload::Else { action: None }
        ));
        assert!(matches!(
            lowered.statements[6].payload,
            LoweredStatementPayload::End
        ));
        assert!(matches!(
            lowered.statements[7].payload,
            LoweredStatementPayload::Delete
        ));
        assert!(matches!(
            lowered.statements[8].payload,
            LoweredStatementPayload::Stop
        ));
        assert!(matches!(
            lowered.statements[9].payload,
            LoweredStatementPayload::Run
        ));
        assert!(matches!(
            lowered.statements[10].payload,
            LoweredStatementPayload::Skipped
        ));
    }

    #[test]
    fn typed_statement_lower_distinguishes_inner_action_and_outer_if_faults() {
        let mut fixture = fixture_program();
        let mut inner = statement_value(&fixture, "IF", 1);
        inner["payload"]["condition"] =
            statement_value(&fixture, "IF", 0)["payload"]["condition"].clone();
        let first_reference = inner["apply_references"][0].clone();
        let mut second_reference = first_reference.clone();
        second_reference["target"] = json!("later");
        inner["apply_references"] = json!([first_reference, second_reference]);

        let mut outer = inner.clone();
        outer["payload"] = statement_value(&fixture, "ASSIGN", 1)["payload"].clone();
        outer["payload"]["native_cause"] = json!("outer IF split fault");
        outer["payload"]["error_kind"] = json!("if_split_error");
        outer["payload"]["where_syntax_summary"] = json!("outer summary");
        outer["payload"]["display_text"] = json!("outer display");
        fixture["statements"] = json!([inner, outer]);

        let parsed = parse_program(fixture);
        let lowered = lower_program(&parsed).expect("lowering preserves both deferred layers");
        assert_eq!(lowered.statements.len(), 2);
        assert_eq!(
            lowered.statements[0]
                .apply_references
                .iter()
                .map(|reference| reference.target.as_str())
                .collect::<Vec<_>>(),
            ["lookup", "later"]
        );
        match &lowered.statements[0].payload {
            LoweredStatementPayload::If {
                condition,
                then_action: Some(LoweredInlineAction::Deferred(descriptor)),
                ..
            } => {
                assert!(matches!(
                    &condition.expression,
                    Expr::FunctionCall { name, .. } if name == "apply.custom"
                ));
                assert_eq!(
                    descriptor.native_cause,
                    "unsupported inline action: opaque action"
                );
                assert_eq!(descriptor.span.source_id, "typed-payload-fixture");
            }
            other => panic!("expected inner deferred action, got {other:?}"),
        }
        match &lowered.statements[1].payload {
            LoweredStatementPayload::Deferred { descriptor } => {
                assert_eq!(descriptor.error_kind, "if_split_error");
                assert_eq!(descriptor.native_cause, "outer IF split fault");
                assert_eq!(descriptor.where_syntax_summary, "outer summary");
                assert_eq!(descriptor.display_text, "outer display");
            }
            other => panic!("expected outer IF split fault, got {other:?}"),
        }
    }

    #[test]
    fn typed_statement_lower_preserves_inline_output_and_planned_suppression() {
        let mut value = fixture_program();
        let mut if_statement = statement_value(&value, "IF", 0);
        let output = statement_value(&value, "OUTPUT", 0);
        if_statement["payload"]["condition"] = json!({
            "tag": "literal",
            "span": if_statement["span"].clone(),
            "origin": null,
            "kind": "string",
            "lexeme": "' output '"
        });
        if_statement["payload"]["then_action"] = json!({
            "tag": "statement",
            "statement": output
        });
        value["statements"] = json!([if_statement]);
        value["suppress_implicit_output"] = json!(true);
        let parsed = parse_program(value);
        let lowered = lower_program(&parsed).unwrap();
        assert!(lowered.suppress_implicit_output);
        assert_eq!(
            lowered.output_targets,
            ["WORK.OUT", "work.out", "work.empty"]
        );
        match &lowered.statements[0].payload {
            LoweredStatementPayload::If {
                condition,
                then_action: Some(LoweredInlineAction::Statement(statement)),
                ..
            } => {
                assert!(matches!(
                    &condition.expression,
                    Expr::Literal(ScalarValue::Text(value)) if value == " output "
                ));
                match &statement.payload {
                    LoweredStatementPayload::Output { targets } => {
                        assert_eq!(targets[0].name, "out");
                        assert_eq!(targets[0].options.keep_vars, ["id"]);
                    }
                    other => panic!("expected inline OUTPUT, got {other:?}"),
                }
            }
            other => panic!("expected IF with inline OUTPUT, got {other:?}"),
        }

        let mut literal_only = fixture_program();
        let mut no_action_if = statement_value(&literal_only, "IF", 2);
        no_action_if["payload"]["condition"] = json!({
            "tag": "literal",
            "span": no_action_if["span"].clone(),
            "origin": null,
            "kind": "string",
            "lexeme": "' output'"
        });
        literal_only["statements"] = json!([no_action_if]);
        literal_only["suppress_implicit_output"] = json!(false);
        let lowered = lower_program(&parse_program(literal_only)).unwrap();
        assert!(!lowered.suppress_implicit_output);
        assert!(matches!(
            &lowered.statements[0].payload,
            LoweredStatementPayload::If {
                then_action: None,
                ..
            }
        ));
    }

    #[test]
    fn typed_statement_lower_preserves_bare_then_else_do_and_unicode_spans() {
        let mut fixture = fixture_program();
        let mut if_statement = statement_value(&fixture, "IF", 0);
        let mut else_statement = statement_value(&fixture, "ELSE", 0);
        let mut bare_do = statement_value(&fixture, "DO", 1);
        let outer_span = json!({
            "start": 10,
            "end": 12,
            "line": 3,
            "column": 1,
            "end_line": 3,
            "end_column": 3,
            "source_id": "unicode-source"
        });
        let outer_origin = json!({
            "start": 10,
            "end": 11,
            "line": 3,
            "column": 1,
            "end_line": 3,
            "end_column": 2,
            "source_id": "unicode-origin"
        });
        if_statement["display_text"] = json!("é🦑");
        if_statement["span"] = outer_span.clone();
        if_statement["origin"] = outer_origin.clone();
        if_statement["payload"]["condition"]["span"] = outer_span.clone();
        if_statement["payload"]["condition"]["origin"] = outer_origin.clone();
        if_statement["payload"]["then_action"] = json!({
            "tag": "statement",
            "statement": bare_do.clone()
        });
        if_statement["payload"]["is_then_do"] = json!(true);

        bare_do["display_text"] = json!("do");
        bare_do["span"] = json!({
            "start": 12,
            "end": 14,
            "line": 3,
            "column": 3,
            "end_line": 3,
            "end_column": 5,
            "source_id": "unicode-source"
        });
        bare_do["origin"] = json!({
            "start": 12,
            "end": 13,
            "line": 3,
            "column": 3,
            "end_line": 3,
            "end_column": 4,
            "source_id": "unicode-origin"
        });
        if_statement["payload"]["then_action"]["statement"] = bare_do.clone();
        else_statement["payload"]["action"] = json!({
            "tag": "statement",
            "statement": bare_do
        });
        fixture["statements"] = json!([if_statement, else_statement]);

        let lowered = lower_program(&parse_program(fixture)).unwrap();
        assert_eq!(lowered.statements[0].display_text.chars().count(), 2);
        assert_eq!(lowered.statements[0].span.start, 10);
        assert_eq!(lowered.statements[0].span.end, 12);
        assert_eq!(
            lowered.statements[0].origin.as_ref().unwrap().source_id,
            "unicode-origin"
        );
        match &lowered.statements[0].payload {
            LoweredStatementPayload::If {
                condition,
                then_action: Some(LoweredInlineAction::Statement(statement)),
                is_then_do: true,
                ..
            } => {
                match &condition.source {
                    TypedExpression::Call { span, origin, .. } => {
                        assert_eq!(span.source_id, "unicode-source");
                        assert_eq!(origin.as_ref().unwrap().source_id, "unicode-origin");
                    }
                    other => panic!("expected typed call condition, got {other:?}"),
                }
                assert_eq!(statement.kind, "DO");
                assert_eq!(statement.span.start, 12);
                assert_eq!(statement.span.end, 14);
                assert_eq!(
                    statement.origin.as_ref().unwrap().source_id,
                    "unicode-origin"
                );
                assert!(matches!(
                    &statement.payload,
                    LoweredStatementPayload::Do {
                        loop_var: None,
                        start: None,
                        end: None
                    }
                ));
            }
            other => panic!("expected THEN DO structural action, got {other:?}"),
        }
        assert!(matches!(
            &lowered.statements[1].payload,
            LoweredStatementPayload::Else {
                action: Some(LoweredInlineAction::Statement(statement))
            } if statement.kind == "DO"
                && matches!(statement.payload, LoweredStatementPayload::Do { loop_var: None, start: None, end: None })
        ));
    }
}
