use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub(crate) struct AstDatasetRefOptions {
    pub(crate) in_var: Option<String>,
    pub(crate) keep_vars: Vec<String>,
    pub(crate) drop_vars: Vec<String>,
    pub(crate) where_expr: Option<AstWhereExpression>,
    pub(crate) rename_map: HashMap<String, String>,
    pub(crate) firstobs: Option<i64>,
    pub(crate) obs: Option<i64>,
}

#[derive(Debug, Clone)]
pub(crate) enum AstWhereExpression {
    Comparison {
        column: String,
        operator: AstWhereComparisonOperator,
        value: AstWhereScalar,
    },
    Null {
        column: String,
        negated: bool,
    },
    Logical {
        operator: AstWhereLogicalOperator,
        operands: Vec<AstWhereExpression>,
    },
    Not {
        operand: Box<AstWhereExpression>,
    },
}

#[derive(Debug, Clone)]
pub(crate) enum AstWhereScalar {
    Text(String),
    Integer(i64),
    Unsigned(u64),
    Number(f64),
    Bool(bool),
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum AstWhereComparisonOperator {
    Eq,
    Equal,
    NotEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum AstWhereLogicalOperator {
    And,
    Or,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct AstStatementOptions {
    pub(crate) indsname_var: Option<String>,
    pub(crate) end_var: Option<String>,
}
