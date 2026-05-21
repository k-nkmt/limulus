import unittest
import re
from types import SimpleNamespace
from unittest.mock import patch

import datetime as dt

import pyarrow as pa
import pytest

from limulus import Session
from limulus.native_bridge import load_native_module
from limulus.runtime import DataStepExecutor
from limulus.io_adapters import (
    DataFrameAdapterPandas,
    DataInputAdapterArrow,
    DataOutputAdapterArrow,
    InputSpec,
    OutputSpec,
)
from limulus.models import DataSetRef, DiagnosticLabel, DiagnosticSpan, ExecuteRequest, ExecuteResponse, LogEntry, SubmitResult


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)

SESSION_SCENARIOS = {
    "load_submit_dataset_access": {
        "overview": "Session.load + submit exposes work datasets and filters rows by IF condition",
        "inputs": {"inp": {"id": [1, 2], "amount": [10, -5]}},
        "dsl": """
        data out;
        set inp;
        if amount >= 0 then output out;
        run;
        """,
        "expected_output": [{"id": 1, "amount": 10}],
    },
    "to_arrow_to_pandas": {
        "overview": "Session.run output can be retrieved as Arrow and pandas with same row content",
        "inputs": {"inp": [{"id": 1, "amount": 3}]},
        "dsl": """
        data out;
        set inp;
        output out;
        run;
        """,
        "expected_rows": [{"id": 1, "amount": 3}],
    },
    "if_then_do_else_do": {
        "overview": "IF/ELSE DO routes rows exclusively into age1/age2 outputs",
        "inputs": {
            "students": {
                "name": ["Adam", "Alice", "Bob", "Catherine"],
                "age": [14, 12, 14, 13],
            }
        },
        "dsl": """
        data age1 age2;
            set students end = eof;
            seq = _n_;
            if age > 13 then do;
                output age1;
            end;
            else do;
                output age2;
            end;
        run;
        """,
        "expected_age1_names": ["Adam", "Bob"],
        "expected_age2_names": ["Alice", "Catherine"],
    },
    "log_returns_entries": {
        "overview": "Invalid blank submit populates session log with error severity",
        "dsl": " ",
        "expected_success": False,
        "expected_first_severity": "error",
    },
    "backend_preferences_init": {
        "overview": "Session initialization accepts runtime/parser backend preferences",
        "runtime_backend": "rust",
        "parser_backend": "rust",
        "inputs": {"inp": [{"id": 1}]},
        "dsl": """
        data out;
        set inp;
        output out;
        run;
        """,
        "expected_output": [{"id": 1}],
    },
    "auto_backend_prefers_rust": {
        "overview": "Auto backend selects rust path for eligible WHERE-only workload",
        "backend": "auto",
        "inputs": {"inp": {"id": [1, 2], "amount": [3, -1]}},
        "dsl": """
        data out;
        set inp;
        where amount >= 0;
        output out;
        run;
        """,
        "expected_output": [{"id": 1, "amount": 3}],
        "expected_runtime_backend": "rust",
    },
    "acceptance_e2e_multi_output": {
        "overview": "Executor end-to-end run routes filtered rows into out_a/out_b based on IF condition",
        "dsl": (
            "data out_a out_b; "
            "set in; "
            "where amount >= 0; "
            "if amount > 10 then output out_a; "
            "else output out_b; "
            "run;"
        ),
        "inputs": {
            "in": [{"id": 1, "amount": 20}, {"id": 2, "amount": 5}, {"id": 3, "amount": -1}],
        },
        "output_targets": ["out_a", "out_b"],
        "expected_out_a": [{"id": 1, "amount": 20}],
        "expected_out_b": [{"id": 2, "amount": 5}],
    },
    "acceptance_io_interop": {
        "overview": "Arrow/Pandas adapters interoperate and preserve expected canonical rows",
        "arrow_input": {"id": [1], "value": [10]},
        "arrow_expected": [{"id": 1, "value": 10}],
        "pandas_rows": [{"id": 11, "value": None}],
        "pandas_expected": [{"id": 11, "value": None}],
    },
    "acceptance_merge_option_chain": {
        "overview": "SET dataset options, INDSNAME/END and keep chain retain only final qualifying row",
        "dsl": (
            "data out; set in(keep=id amount tmp drop=tmp where=(amount >= 10) rename=(amount=amt) in=in_flag) "
            "indsname=src end=last; if in_flag and last then output out; keep id amt src last; run;"
        ),
        "inputs": [{"id": 1, "amount": 5, "tmp": "a"}, {"id": 2, "amount": 10, "tmp": "b"}],
        "expected_output": [{"id": 2, "amt": 10}],
    },
    "acceptance_output_options_multi_target": {
        "overview": "Output dataset options apply independently across multi-target OUTPUT routing",
        "inputs": [{"id": 1, "name": "Alice", "x": 1, "tmp": "a"}, {"id": 2, "name": "Bob", "x": 0, "tmp": "b"}],
        "dsl": """
        data a(keep=id) b(keep=id name rename=(name=full_name));
        set ds;
        if x > 0 then output a;
        else output b;
        run;
        """,
        "expected_a": [{"id": 1}],
        "expected_b": [{"id": 2, "full_name": "Bob"}],
    },
    "acceptance_output_options_python_runtime": {
        "overview": "DATA output options are honored under explicit python runtime preference",
        "runtime_backend": "python",
        "inputs": {"id": [1, 2], "name": ["Alice", "Bob"], "tmp": ["a", "b"]},
        "dsl": """
        data out(keep=id name drop=name rename=(id=subject_id));
        set inp;
        output out;
        run;
        """,
        "expected_output": [{"subject_id": 1}, {"subject_id": 2}],
    },
    "acceptance_catalog_resolution": {
        "overview": "Catalog-resolved input works and explicit request inputs override registered tables",
        "catalog_payload": [{"id": 1}, {"id": 2}],
        "catalog_dsl": "data out; set in; where id = 2; output out; run;",
        "catalog_expected": [{"id": 2}],
        "explicit_dsl": "data out; set in; output out; run;",
        "explicit_payload": [{"id": 9}],
        "explicit_expected": [{"id": 9}],
    },
    "acceptance_register_tables_bulk": {
        "overview": "register_tables accepts mixed table refs and resolves SET inputs in declared order",
        "dsl": "data out; set in_a in_b; output out; run;",
        "expected_output": [{"id": 1}, {"id": 2}],
    },
    "acceptance_resolve_registered_set_input": {
        "overview": "Registered table is resolved as SET input when request inputs are omitted",
        "dsl": "data out; set in; output out; run;",
        "registered_payload": [{"id": 1, "amount": 10}],
        "expected_output": [{"id": 1, "amount": 10}],
    },
    "acceptance_previous_outputs": {
        "overview": "Outputs from a prior execute call can be consumed as SET inputs in a subsequent call",
        "first_dsl": "data tmp; set in; output tmp; run;",
        "first_inputs": [{"id": 1}],
        "second_dsl": "data out; set tmp; output out; run;",
        "expected_output": [{"id": 1}],
    },
    "acceptance_multi_block_chain": {
        "overview": "Multi-block request chains intermediate outputs between blocks",
        "dsl": "data stage1; set in; output stage1; run; data stage2; set stage1; if id = 2 then output stage2; run;",
        "inputs": [{"id": 1}, {"id": 2}],
        "expected_stage1": [{"id": 1}, {"id": 2}],
        "expected_stage2": [{"id": 2}],
    },
    "acceptance_multi_block_failure": {
        "overview": "Failure in later block reports diagnostic with block location",
        "dsl": "data stage1; set in; output stage1; run; data stage2; set missing_in; output stage2; run;",
        "inputs": [{"id": 1}],
        "expected_code": "RUNTIME_SET_DATASET_NOT_FOUND",
        "expected_location": "block:2",
    },
    "acceptance_set_multi_input_order": {
        "overview": "SET with multiple inputs preserves declared concatenation order",
        "dsl": "data out; set a b c; output out; run;",
        "inputs_a": [{"id": 1}],
        "inputs_b": [{"id": 2}, {"id": 3}],
        "inputs_c": [{"id": 4}],
        "expected_output": [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}],
    },
    "acceptance_set_by_interleave_numeric": {
        "overview": "SET BY with multiple inputs produces interleaved (merge-sorted) rows by numeric BY key",
        "dsl": "data out; set a b; by id; output out; keep id src; run;",
        "inputs_a": [{"id": 1, "src": "a"}, {"id": 3, "src": "a"}, {"id": 5, "src": "a"}],
        "inputs_b": [{"id": 2, "src": "b"}, {"id": 4, "src": "b"}],
        "expected_output": [
            {"id": 1, "src": "a"},
            {"id": 2, "src": "b"},
            {"id": 3, "src": "a"},
            {"id": 4, "src": "b"},
            {"id": 5, "src": "a"},
        ],
    },
    "acceptance_set_by_interleave_string": {
        "overview": "SET BY with multiple inputs produces interleaved rows by character BY key",
        "dsl": "data out; set a b; by name; output out; keep name grp; run;",
        "inputs_a": [{"name": "Alice", "grp": "a"}, {"name": "Catherine", "grp": "a"}],
        "inputs_b": [{"name": "Bob", "grp": "b"}, {"name": "Dave", "grp": "b"}],
        "expected_output": [
            {"name": "Alice", "grp": "a"},
            {"name": "Bob", "grp": "b"},
            {"name": "Catherine", "grp": "a"},
            {"name": "Dave", "grp": "b"},
        ],
    },
    "acceptance_set_in_flag_subset": {
        "overview": "IN= flags can drive subset logic over multiple SET inputs",
        "dsl": "data out; set a(in=in_a) b(in=in_b); if in_a then output out; keep id; run;",
        "inputs_a": [{"id": 1}, {"id": 2}],
        "inputs_b": [{"id": 3}],
        "expected_output": [{"id": 1}, {"id": 2}],
    },
    "acceptance_skipped_unsupported_logs": {
        "overview": "Unsupported statements are skipped and logged as info notices",
        "dsl": """
        data out;
        length id 8;
        attrib id length=8;
        format id 8.;
        label id = "Identifier";
        informat id 8.;
        set inp;
        output out;
        run;
        """,
        "expected_skip_count": 4,
    },
    "acceptance_dataset_options_order": {
        "overview": "SET options apply in keep/drop, where, rename, obs/firstobs order and preserve final row shape",
        "dsl": "data out; set in(keep=id amount tmp drop=tmp where=(amount > 10) rename=(amount=amt) firstobs=2 obs=1); output out; run;",
        "inputs": [
            {"id": 1, "amount": 5, "tmp": "x"},
            {"id": 2, "amount": 20, "tmp": "y"},
            {"id": 3, "amount": 30, "tmp": "z"},
        ],
        "expected_output": [{"id": 3, "amt": 30}],
    },
    "acceptance_set_statement_options": {
        "overview": "SET statement-level INDSNAME/END options are evaluated across concatenated inputs",
        "dsl": "data out; set a b indsname=src end=last; if last then output out; keep id src last; run;",
        "inputs_a": [{"id": 1}],
        "inputs_b": [{"id": 2}, {"id": 3}],
        "expected_output": [{"id": 3}],
    },
    "acceptance_merge_end_statement_option": {
        "overview": "MERGE statement END= is accepted and marks the final merged row",
        "dsl": "data out; merge a b end=eof; by id; if eof then output out; keep id eof; run;",
        "inputs_a": [{"id": 1}, {"id": 2}],
        "inputs_b": [{"id": 2}, {"id": 3}],
        "expected_output": [{"id": 3}],
    },
    "label_metadata": {
        "overview": "Dataset and column labels are preserved in Arrow metadata",
        "dsl": '''
        data dm(label="DM");
        set inp;
        label id = "Identifier" amount = "Amount";
        output dm;
        run;
        ''',
        "inputs": {"id": [1], "amount": [10]},
    },
    "astype": {
        "overview": "DatasetView.astype converts columns through polars",
        "inputs": {"id": [1, 2], "amount": ["10", "20"]},
    },
    "sql": {
        "overview": "Session.sql returns Arrow output and can save into the session catalog via CREATE TABLE",
        "inputs": {"id": [1, 2, 3], "amount": [5, 20, 30]},
    },
    "acceptance_set_by_rename_internal_excluded": {
        "overview": "BY + rename output excludes internal helper variables and keeps renamed value",
        "dsl": (
            "data out; set in(in=in_flag) end=eof; by grp; if last.grp then output out; "
            "keep grp amount; rename amount = value; run;"
        ),
        "inputs": [{"grp": "A", "amount": 1}, {"grp": "A", "amount": 2}, {"grp": "B", "amount": 3}],
        "expected_output": [{"grp": "A", "value": 2}, {"grp": "B", "value": 3}],
    },
    "acceptance_subset_if": {
        "overview": "Subset IF keeps only rows where condition is true, equivalent to IF NOT(...) THEN DELETE",
        "subset_dsl": "data out; set in; if amount > 0; output out; run;",
        "delete_equiv_dsl": "data out; set in; if not(amount > 0) then delete; output out; run;",
        "inputs": [{"id": 1, "amount": -1}, {"id": 2, "amount": 1}, {"id": 3, "amount": 10}],
        "expected_output": [{"id": 2, "amount": 1}, {"id": 3, "amount": 10}],
    },
    "acceptance_stop_statement": {
        "overview": "STOP terminates the DATA step and preserves rows explicitly output before STOP",
        "dsl": "data out; set in; output out; if id = 2 then stop; run;",
        "inputs": [{"id": 1}, {"id": 2}, {"id": 3}],
        "expected_output": [{"id": 1}, {"id": 2}],
    },
    "acceptance_missing_assignment": {
        "overview": "Missing assignment supports dot and None while preserving empty string separately",
        "dsl": "data out; set in; a = .; b = \"\"; c = None; output out; keep id a b c; run;",
        "inputs": [{"id": 1}, {"id": 2}],
        "expected_output": [
            {"id": 1, "a": None, "b": "", "c": None},
            {"id": 2, "a": None, "b": "", "c": None},
        ],
    },
    "acceptance_sum_statement": {
        "overview": "Sum statement accumulates value row-by-row in execution order",
        "dsl": "data out; set in; seq + 1; output out; keep id seq; run;",
        "inputs": [{"id": 10}, {"id": 20}, {"id": 30}],
        "expected_output": [{"id": 10, "seq": 1}, {"id": 20, "seq": 2}, {"id": 30, "seq": 3}],
    },
    "acceptance_if_do_sum_retain": {
        "overview": "IF DO with sum-style retain behavior keeps state through END and emits on last row",
        "dsl": (
            "data out; set in end=last; if id = 1 then do; flag + 1; end; if last = 1 then output out; keep id flag; run;"
        ),
        "inputs": [{"id": 1}, {"id": 2}],
        "expected_output": [{"id": 2, "flag": 1}],
    },
    "acceptance_do_end_loop": {
        "overview": "DO loop emits output for each iteration with loop index value",
        "dsl": "data out; set in; do times = 1 to 3; output out; end; keep id times; run;",
        "inputs": [{"id": 1}],
        "expected_output": [{"id": 1, "times": 1}, {"id": 1, "times": 2}, {"id": 1, "times": 3}],
    },
    "acceptance_if_do_nesting_limit": {
        "overview": "Nested IF/DO beyond runtime limit returns identifiable nesting diagnostic",
        "nested_if_count": 11,
        "expected_code": "RUNTIME_LOOP_NESTING_LIMIT_EXCEEDED",
    },
    "acceptance_concat_and_n": {
        "overview": "String concatenation and _n_ automatic variable behave as expected",
        "dsl": (
            "data out; set in; joined = prefix || id || suffix; rowno = _n_; "
            "if joined = 'x1' and rowno = 1 then output out; keep joined rowno; run;"
        ),
        "inputs": [{"prefix": "x", "id": 1, "suffix": None}, {"prefix": None, "id": 2, "suffix": "z"}],
        "expected_output": [{"joined": "x1", "rowno": 1}],
    },
    "acceptance_array_assignment_dim_vname": {
        "overview": "Array assignment variants and dim/vname return expected transformed values",
        "dsl": (
            "data out; set in; array chars $ c1 c2; array vals a b c; vals(1) = vals(1) * 10; "
            "vals[2] = vals[2] * 10; vals{3} = vals{3} * 10; arr_size = dim(vals); name2 = vname(vals[2]); "
            "output out; keep a b c arr_size name2 c1 c2; run;"
        ),
        "inputs": [{"a": 1, "b": 2, "c": 3, "c1": "x", "c2": "y"}],
        "expected_output": [{"a": 10, "b": 20, "c": 30, "arr_size": 3, "name2": "b", "c1": "x", "c2": "y"}],
    },
    "acceptance_array_star_case_and_by_case": {
        "overview": "Array[*], _N_/_ERROR_ case-insensitivity, and FIRST./LAST. case-insensitive references are supported",
        "array_inputs": {"a": [10], "b": [20], "c": [30]},
        "array_dsl": """
        data out;
        set inp;
        array vars [*] a b c;
        dim_vars = dim(vars);
        x = vars[2];
        output out;
        keep dim_vars x;
        run;
        """,
        "array_expected": [{"dim_vars": 3, "x": 20}],
        "auto_inputs": {"id": [101, 102]},
        "auto_dsl": """
        data out;
        set inp;
        seq_l = _n_;
        seq_u = _N_;
        err_l = _error_;
        err_u = _ERROR_;
        output out;
        keep seq_l seq_u err_l err_u;
        run;
        """,
        "auto_expected": [
            {"seq_l": 1, "seq_u": 1, "err_l": 0, "err_u": 0},
            {"seq_l": 2, "seq_u": 2, "err_l": 0, "err_u": 0},
        ],
        "left_inputs": {"subj": ["A", "A", "B"], "x": [1, 2, 3]},
        "right_inputs": {"subj": ["A", "B"], "y": [10, 20]},
        "by_dsl": """
        data out;
        merge left right;
        by subj;
        if first.subj = FIRST.subj and last.subj = LAST.subj then output out;
        keep subj;
        run;
        """,
        "by_expected": [{"subj": "A"}, {"subj": "A"}, {"subj": "B"}],
    },
}

EXECUTE_API_SCENARIOS = {
    "valid_request": {
        "overview": "Valid minimal request executes without diagnostics/notices",
        "dsl": "data out; set in; run;",
        "output_targets": ["out"],
    },
    "compat_notice": {
        "overview": "Missing-value comparison emits compatibility notice",
        "dsl": "data out; set in; if amount = . then output out; run;",
        "inputs": [{"amount": None}],
        "output_targets": ["out"],
        "notice_id": "COMPAT_MISSING_VALUE_SEMANTICS",
    },
    "reject_empty_dsl": {
        "overview": "Blank DSL is rejected with request validation diagnostic",
        "dsl": "   ",
        "output_targets": ["out"],
        "expected_code": "REQ_EMPTY_DSL",
    },
    "reject_invalid_inputs": {
        "overview": "Invalid input mapping key is rejected",
        "dsl": "data out; set in; run;",
        "output_targets": ["out"],
        "expected_code": "REQ_INVALID_INPUTS",
    },
    "reject_invalid_output_targets": {
        "overview": "Invalid output target names are rejected",
        "dsl": "data out; set in; run;",
        "output_targets": ["", "out"],
        "expected_code": "REQ_INVALID_OUTPUT_TARGETS",
    },
    "resolve_targets_data_statement": {
        "overview": "When targets omitted, DATA statement targets are inferred",
        "dsl": "data out; set in; output out; run;",
        "inputs": [{"id": 1}],
        "output_targets": [],
        "expected_output": [{"id": 1}],
    },
    "outputs_arrow_reuse": {
        "overview": "outputs_arrow view can be reused as input in subsequent execution",
        "first_dsl": "data out; set in; output out; run;",
        "first_inputs": [{"id": 1}, {"id": 2}],
        "second_dsl": "data out2; set out; if id = 2 then output out2; run;",
        "first_expected": [{"id": 1}, {"id": 2}],
        "second_expected": [{"id": 2}],
    },
    "convert_outputs_arrow_pylist": {
        "overview": "convert_outputs returns arrow and pylist only when explicitly requested",
        "dsl": "data out; set in; output out; run;",
        "inputs": [{"id": 1}],
        "expected_output": [{"id": 1}],
    },
    "outputs_arrow_lazy_materialize": {
        "overview": "outputs_arrow materialization is lazy until accessed",
        "dsl": "data out; set in; output out; run;",
        "inputs": [{"id": 1}],
    },
    "simple_set_reuse_rows": {
        "overview": "Simple set/output path reuses row objects without extra copies",
        "dsl": "data out; set in; output out; run;",
        "inputs": [{"id": 1, "amount": 10}, {"id": 2, "amount": 20}],
    },
    "pylist_only_explicit": {
        "overview": "to_pylist is only invoked for explicit pylist conversion",
        "rows": [{"id": 1}],
    },
    "rust_runtime_fallback": {
        "overview": "Rust runtime preference falls back to python when workload unsupported",
        "dsl": "data out; set in; output out; run;",
        "inputs": [{"id": 1}],
        "expected_runtime_backend": "python",
    },
    "rust_runtime_fallback_apply": {
        "overview": "Rust runtime preference falls back to python when apply() is used",
        "dsl": "data out; set in; doubled = apply('double', amount); output out; keep id doubled; run;",
        "inputs": [{"id": 1, "amount": 4}],
        "expected_output": [{"id": 1, "doubled": 8}],
        "expected_runtime_backend": "python",
    },
    "rust_parser_preference": {
        "overview": "Rust parser preference is used for supported subset",
        "dsl": "data out; set in; output out; run;",
        "inputs": [{"id": 1}],
        "expected_parser_backend": "rust",
    },
    "convert_unsupported_format": {
        "overview": "Unsupported convert format returns identifiable diagnostic",
        "dsl": "data out; set in; output out; run;",
        "inputs": [{"id": 1}],
        "format": "json",
        "expected_code": "CONVERT_OUTPUT_FORMAT_UNSUPPORTED",
    },
    "convert_failed_dataset": {
        "overview": "Unconvertible dataset payload returns convert failure diagnostic",
        "expected_code": "CONVERT_OUTPUT_FAILED",
    },
    "resolve_targets_data_and_output": {
        "overview": "Output targets are inferred from DATA and OUTPUT statements",
        "dsl": "data out_a out_b; set in; if id = 1 then output out_a; else output out_b; run;",
        "inputs": [{"id": 1}, {"id": 2}],
        "output_targets": [],
        "expected_out_a": [{"id": 1}],
        "expected_out_b": [{"id": 2}],
    },
    "setless_do_output": {
        "overview": "SET-less DATA step with DO loop emits expected rows",
        "dsl": "data out; text = 'abc'; do i = 1 to 3; output out; end; run;",
        "expected_output": [{"text": "abc", "i": 1}, {"text": "abc", "i": 2}, {"text": "abc", "i": 3}],
    },
    "set_missing_dataset": {
        "overview": "Missing SET dataset returns runtime not-found diagnostic",
        "dsl": "data out; set missing; output out; run;",
        "output_targets": ["out"],
        "expected_code": "RUNTIME_SET_DATASET_NOT_FOUND",
    },
    "invalid_dataset_option_expr": {
        "overview": "Invalid dataset option expression returns identifiable diagnostic",
        "dsl": "data out; set in(where=(unknown_col > 0)); output out; run;",
        "inputs": [{"id": 1}],
        "output_targets": ["out"],
        "expected_code": "RUNTIME_DATASET_OPTION_INVALID",
    },
    "reserved_auto_var_option": {
        "overview": "Reserved auto variable in dataset option returns diagnostic",
        "dsl": "data out; set in(in=_N_); output out; run;",
        "inputs": [{"id": 1}],
        "output_targets": ["out"],
        "expected_code": "RUNTIME_DATASET_OPTION_INVALID",
    },
    "rename_statement_success": {
        "overview": "RENAME statement rewrites output column names",
        "dsl": "data out; set in; rename amount=amt; output out; run;",
        "inputs": [{"amount": 10}],
        "output_targets": ["out"],
        "expected_output": [{"amt": 10}],
    },
    "rename_statement_invalid": {
        "overview": "Invalid RENAME mapping returns identifiable diagnostic",
        "dsl": "data out; set in; rename missing=amt; output out; run;",
        "inputs": [{"amount": 10}],
        "output_targets": ["out"],
        "expected_code": "RUNTIME_RENAME_STATEMENT_INVALID",
    },
    "merge_first_last_flags": {
        "overview": "MERGE/BY generates expected joined rows while hiding internal FIRST/LAST flags",
        "dsl": "data out; merge a b; by id; output out; keep id x y FIRST.id LAST.id; run;",
        "inputs_a": [{"id": 1, "x": 10}, {"id": 2, "x": 20}],
        "inputs_b": [{"id": 1, "y": 100}, {"id": 3, "y": 300}],
        "output_targets": ["out"],
        "expected_output": [{"id": 1, "x": 10, "y": 100}, {"id": 2, "x": 20}, {"id": 3, "y": 300}],
    },
    "merge_in_flag_filter": {
        "overview": "MERGE IN= flags can filter left-side rows",
        "dsl": "data out; merge a(in=in_left) b(in=in_right); by id; if in_left then output out; keep id in_left in_right; run;",
        "inputs_a": [{"id": 1}, {"id": 2}],
        "inputs_b": [{"id": 1}, {"id": 3}],
        "output_targets": ["out"],
        "expected_output": [{"id": 1}, {"id": 2}],
    },
    "internal_vars_excluded_keep": {
        "overview": "Internal reference vars remain excluded even when KEEP requests them",
        "dsl": "data out; set a(in=in_a) b(in=in_b) indsname=src end=last; if in_a or in_b then output out; keep id in_a in_b src last; run;",
        "inputs_a": [{"id": 1}],
        "inputs_b": [{"id": 2}],
        "output_targets": ["out"],
        "expected_output": [{"id": 1}, {"id": 2}],
    },
    "internal_vars_excluded_after_rename": {
        "overview": "Renaming internal vars does not expose them in final output",
        "dsl": "data out; set in(in=in_flag); rename in_flag=joined; output out; keep id in_flag; run;",
        "inputs": [{"id": 1}],
        "output_targets": ["out"],
        "expected_output": [{"id": 1}],
    },
    "collision_internal_var": {
        "overview": "Input column collision with internal reference var returns diagnostic",
        "dsl": "data out; set in(in=in_flag); output out; run;",
        "inputs": [{"id": 1, "in_flag": False}],
        "output_targets": ["out"],
        "expected_code": "RUNTIME_INTERNAL_VAR_NAME_COLLISION",
        "expected_message_fragment": "in_flag",
    },
    "collision_first_last_var": {
        "overview": "Input column collision with FIRST/LAST helper var returns diagnostic",
        "dsl": "data out; merge a b; by id; output out; run;",
        "inputs_a": [{"id": 1, "FIRST.id": True}],
        "inputs_b": [{"id": 1, "y": 10}],
        "output_targets": ["out"],
        "expected_code": "RUNTIME_INTERNAL_VAR_NAME_COLLISION",
        "expected_message_fragment": "FIRST.id",
    },
    "not_equal_aliases": {
        "overview": "Not-equal operator aliases (^=, ~=, ¬=) are supported",
        "dsl": "data out; set in; where amount ^= 2 and amount ~= 3 and amount ¬= 4; output out; run;",
        "inputs": [{"id": 1, "amount": 1}, {"id": 2, "amount": 2}],
        "output_targets": ["out"],
        "expected_output": [{"id": 1, "amount": 1}],
    },
    "in_operator": {
        "overview": "IN operator is supported in WHERE expressions",
        "dsl": "data out; set in; where id in (1, 3); output out; run;",
        "inputs": [{"id": 1}, {"id": 2}, {"id": 3}],
        "output_targets": ["out"],
        "expected_output": [{"id": 1}, {"id": 3}],
    },
    "invalid_operator": {
        "overview": "Unsupported operator returns identifiable diagnostic",
        "dsl": "data out; set in; where amount >< 1; output out; run;",
        "inputs": [{"amount": 1}],
        "output_targets": ["out"],
        "expected_code": "RUNTIME_OPERATOR_NOT_SUPPORTED",
    },
    "prxmatch": {
        "overview": "PRXMATCH function works in WHERE clause",
        "dsl": "data out; set in; where prxmatch('/^A.+e$/', name) > 0; output out; run;",
        "inputs": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
        "output_targets": ["out"],
        "expected_output": [{"id": 1, "name": "Alice"}],
    },
    "prxchange": {
        "overview": "PRXCHANGE function works in IF condition",
        "dsl": "data out; set in; if prxchange('s/-//', -1, phone) = '0901234' then output out; run;",
        "inputs": [{"id": 1, "phone": "090-1234"}, {"id": 2, "phone": "03-9999"}],
        "output_targets": ["out"],
        "expected_output": [{"id": 1, "phone": "090-1234"}],
    },
    "invalid_prx_pattern": {
        "overview": "Invalid regex pattern returns function argument diagnostic",
        "dsl": "data out; set in; where prxmatch('/[/', name) > 0; output out; run;",
        "inputs": [{"name": "Alice"}],
        "output_targets": ["out"],
        "expected_code": "RUNTIME_FUNCTION_ARGUMENT_INVALID",
    },
    "internal_vars_numeric": {
        "overview": "Internal IN= flags behave as numeric 0/1 in expressions",
        "dsl": "data out; merge a(in=in_left) b(in=in_right); by id; if in_left = 1 and in_right = 0 then output out; run;",
        "inputs_a": [{"id": 1}, {"id": 2}],
        "inputs_b": [{"id": 1}],
        "expected_output": [{"id": 2}],
    },
    "sum_accepts_first_last": {
        "overview": "Sum statement accepts FIRST./LAST. numeric flags",
        "dsl": "data out; merge a b; by id; seq + FIRST.id; output out; keep id seq FIRST.id; run;",
        "inputs_a": [{"id": 1}, {"id": 2}],
        "inputs_b": [{"id": 1}, {"id": 3}],
        "expected_first_seq": 1,
    },
    "array_dim_and_eof_default_output": {
        "overview": "Array dimension syntax and EOF default output behavior work together",
        "dsl": (
            "data out; set in end=eof; array outv [4] v1 v2 v3 v4; array inv [2] amount region; retain v1 v2 v3 v4; "
            "do i = 1 to dim(inv); if _n_ = 1 then outv[i] = inv[i]; else outv[i+2] = inv[i]; end; "
            "if eof then output; keep v1 v2 v3 v4; run;"
        ),
        "inputs": [{"amount": 10.2, "region": "East"}, {"amount": -2.0, "region": "West"}],
        "expected_output": [{"v1": 10.2, "v2": "East", "v3": -2.0, "v4": "West"}],
    },
    "expanded_string_functions": {
        "overview": "Expanded string function set is supported in WHERE expression",
        "dsl": (
            "data out; set in; where trim(name) = 'alice' and upcase(name) = 'ALICE' and lowcase('AbC') = 'abc' and "
            "propcase('john DOE') = 'John Doe' and catx('-', ' A ', 'B') = 'A-B' and cat('x', 'y') = 'xy' and "
            "catt('a  ', 'b  ') = 'ab' and index(name, 'li') = 2 and find(name, 'AL', 1, 'i') = 1 and "
            "tranwrd('aa-bb', '-', '_') = 'aa_bb' and translate('abc', 'xyz', 'ab') = 'xyc' and length(name) = 5 and "
            "lengthn('') = 0 and strip('  x  ') = 'x' and reverse(code) = '1X' and repeat('a', 3) = 'aaa' and "
            "countw('a,b,c', ',') = 3 and compress('a b c') = 'abc'; output out; run;"
        ),
        "inputs": [{"id": 1, "name": "alice", "code": "X1"}, {"id": 2, "name": "bob", "code": "Y2"}],
        "output_targets": ["out"],
        "expected_output": [{"id": 1, "name": "alice", "code": "X1"}],
    },
    "expanded_numeric_functions": {
        "overview": "Expanded numeric function set is supported in WHERE expression",
        "dsl": (
            "data out; set in; where ceil(value) = 4 and floor(value) = 3 and abs(delta) = 2 and mod(num, 3) = 1 and "
            "max(1, num, 2) = num and min(num, 5) = 4 and sum(num, 1, .) = 5 and mean(2, 4, .) = 3 and "
            "sqrt(9) = 3 and round(log(exp(1)), 1e-9) = 1 and sign(delta) = -1; output out; run;"
        ),
        "inputs": [{"id": 1, "value": 3.2, "delta": -2, "num": 4}],
        "output_targets": ["out"],
        "expected_output": [{"id": 1, "value": 3.2, "delta": -2, "num": 4}],
    },
    "common_funcs_lead_apply": {
        "overview": "Common functions, lag/lead, and apply() work with runtime function registry",
        "dsl": (
            "data out; set in; if missing(note) and nmiss(score, .) = 2 and cmiss(note, '') = 2 and "
            "lag(id) = . and lead('id') = 2 and apply('double', amount) = 20 then output out; run;"
        ),
        "inputs": [{"id": 1, "note": None, "score": None, "amount": 10}, {"id": 2, "note": "ok", "score": 1, "amount": 5}],
        "output_targets": ["out"],
        "expected_output": [{"id": 1, "note": None, "score": None, "amount": 10}],
    },
    "apply_python_names": {
        "overview": "apply() can call builtins and dotted module functions without registration",
        "dsl": (
            "data out; set in; "
            "if apply('len', note) = 0 and apply('math.sqrt', num) = 2 then output out; run;"
        ),
        "inputs": [{"id": 1, "note": "", "num": 4}, {"id": 2, "note": "x", "num": 9}],
        "output_targets": ["out"],
        "expected_output": [{"id": 1, "note": "", "num": 4}],
    },
    "apply_user_defined": {
        "overview": "apply() resolves local Python functions by name without explicit registration",
        "dsl": (
            "data out; set in; "
            "if apply('myfunc', amount) = 30 then output out; run;"
        ),
        "inputs": [{"id": 1, "amount": 10}, {"id": 2, "amount": 15}, {"id": 3, "amount": 20}],
        "output_targets": ["out"],
        "expected_output": [{"id": 3, "amount": 20}],
    },
    "assignment_dispatch_runtime_registry": {
        "overview": "Assignment function dispatch resolves Python names via apply helper",
        "dsl": "data out; set in; doubled = apply('double', amount); output out; keep id doubled; run;",
        "inputs": [{"id": 1, "amount": 4}],
        "output_targets": ["out"],
        "expected_output": [{"id": 1, "doubled": 8}],
    },
    "merge_by_precondition_failed": {
        "overview": "MERGE BY fails when a BY key is missing in one input",
        "dsl": "data out; merge a b; by id; output out; run;",
        "inputs_a": [{"id": 1}],
        "inputs_b": [{"x": 2}],
        "output_targets": ["out"],
        "expected_code": "RUNTIME_BY_PRECONDITION_FAILED",
    },
    "merge_duplicate_column": {
        "overview": "MERGE BY fails on duplicate non-key columns",
        "dsl": "data out; merge a b; by id; output out; run;",
        "inputs_a": [{"id": 1, "value": 10}],
        "inputs_b": [{"id": 1, "value": 20}],
        "output_targets": ["out"],
        "expected_code": "RUNTIME_MERGE_DUPLICATE_COLUMN",
    },
    "conditional_delete": {
        "overview": "Conditional DELETE excludes only matching rows",
        "dsl": "data out; set in; if amount < 0 then delete; output out; run;",
        "inputs": [{"id": 1, "amount": 10}, {"id": 2, "amount": -1}],
        "output_targets": ["out"],
        "expected_output": [{"id": 1, "amount": 10}],
    },
    "unconditional_delete": {
        "overview": "Unconditional DELETE excludes all rows from output",
        "dsl": "data out; set in; delete; output out; run;",
        "inputs": [{"id": 1}, {"id": 2}],
        "output_targets": ["out"],
        "expected_output": [],
    },
    "where_string_funcs": {
        "overview": "String functions in WHERE expression are supported",
        "dsl": "data out; set in; where substr(name, 1, 1) = 'A' and cats(code, suffix) = 'X1'; output out; run;",
        "inputs": [{"id": 1, "name": "Alice", "code": "X", "suffix": "1"}, {"id": 2, "name": "Bob", "code": "Y", "suffix": "2"}],
        "output_targets": ["out"],
        "expected_output": [{"id": 1, "name": "Alice", "code": "X", "suffix": "1"}],
    },
    "if_numeric_funcs": {
        "overview": "Numeric functions in IF expression are supported",
        "dsl": "data out; set in; if round(sum(amount, 0.4), 1) >= 5 then output out; run;",
        "inputs": [{"id": 1, "amount": 4.8}, {"id": 2, "amount": 2.1}],
        "output_targets": ["out"],
        "expected_output": [{"id": 1, "amount": 4.8}],
    },
    "where_date_funcs": {
        "overview": "Date functions in WHERE expression are supported",
        "dsl": (
            "data out; set in; where year(mdy(month, day, year_num)) = 2024 and "
            "intck('day', mdy(1, 1, 2024), mdy(1, 3, 2024)) = 2; output out; run;"
        ),
        "inputs": [{"id": 1, "month": 2, "day": 10, "year_num": 2024}],
        "output_targets": ["out"],
        "expected_output": [{"id": 1, "month": 2, "day": 10, "year_num": 2024}],
    },
    "unsupported_function": {
        "overview": "Unsupported custom function returns identifiable runtime diagnostic",
        "dsl": "data out; set in; where custom_not_supported(amount) > 0; output out; run;",
        "inputs": [{"amount": 1}],
        "output_targets": ["out"],
        "expected_code": "RUNTIME_UNSUPPORTED_FUNCTION",
        "expected_message_fragment": "custom_not_supported",
    },
}


class _CountingToPylistTable:
    def __init__(self, rows):
        self._rows = [dict(row) for row in rows]
        self.to_pylist_calls = 0

    def to_pylist(self):
        self.to_pylist_calls += 1
        return [dict(row) for row in self._rows]


def test_session_load_submit_and_dataset_access() -> None:
    scenario = SESSION_SCENARIOS["load_submit_dataset_access"]
    session = Session()
    table = pa.table(scenario["inputs"]["inp"])
    session.load("inp", table)

    result = session.submit(scenario["dsl"])

    assert result.success is True
    assert "out" in session.datasets
    assert session.work is session.datasets
    assert session["out"].to_pylist() == scenario["expected_output"]


def test_session_to_arrow_and_to_pandas() -> None:
    scenario = SESSION_SCENARIOS["to_arrow_to_pandas"]
    session = Session()
    session.load("inp", scenario["inputs"]["inp"])

    session.run(scenario["dsl"])

    out_arrow = session.to_arrow("out")
    out_pandas = session.to_pandas("out")

    assert out_arrow.to_pylist() == scenario["expected_rows"]
    assert out_pandas.to_dict(orient="records") == scenario["expected_rows"]


def test_session_submit_supports_put_input_and_hour_functions() -> None:
    session = Session()
    session.load(
        "in",
        pa.table(
            {
                "id": [7],
                "amount": [12345.6],
                "best_text": ["12345.6"],
                "date_text": ["2024-02-03"],
                "timestamp_text": ["2024-02-03T16:24:43"],
                "clock_text": ["11:30"],
            }
        ),
    )

    result = session.submit(
        """
        data out;
        set in;
        code = put(id, z5.);
        fixed_text = put(amount, 8.1.);
        rounded_text = put(amount, 8.);
        comma_text = put(amount, comma8.1.);
        zero_scaled = put(amount, z8.1.);
        best_rendered = put(amount, best.);
        best_value = input(best_text, best.);
        visit_date = input(date_text, yymmdd10.);
        visit_iso = put(visit_date, e8601da.);
        timestamp_value = input(timestamp_text, e8601dt.);
        timestamp_iso = put(timestamp_value, e8601dt.);
        clock_value = input(clock_text, time.);
        clock_iso = put(clock_value, time.);
        clock_hour = hour(clock_text);
        output out;
        run;
        """
    )

    assert result.success is True
    assert session["out"].to_pylist() == [
        {
            "id": 7,
            "amount": 12345.6,
            "best_text": "12345.6",
            "date_text": "2024-02-03",
            "timestamp_text": "2024-02-03T16:24:43",
            "clock_text": "11:30",
            "code": "00007",
            "fixed_text": "12345.6",
            "rounded_text": "12346",
            "comma_text": "12,345.6",
            "zero_scaled": "012345.6",
            "best_rendered": "12345.6",
            "best_value": 12345.6,
            "visit_date": dt.date(2024, 2, 3),
            "visit_iso": "2024-02-03",
            "timestamp_value": dt.datetime(2024, 2, 3, 16, 24, 43),
            "timestamp_iso": "2024-02-03T16:24:43",
            "clock_value": dt.time(11, 30),
            "clock_iso": "11:30:00",
            "clock_hour": 11.5,
        }
    ]


def test_if_then_do_else_do_routes_rows_exclusively() -> None:
    scenario = SESSION_SCENARIOS["if_then_do_else_do"]
    session = Session()
    session.load("students", pa.table(scenario["inputs"]["students"]))

    result = session.submit(scenario["dsl"])

    assert result.success is True
    assert [row["name"] for row in session["age1"].to_pylist()] == scenario["expected_age1_names"]
    assert [row["name"] for row in session["age2"].to_pylist()] == scenario["expected_age2_names"]


def test_session_log_returns_entries() -> None:
    scenario = SESSION_SCENARIOS["log_returns_entries"]
    session = Session()
    result = session.submit(scenario["dsl"])

    assert result.success is scenario["expected_success"]
    assert len(session.log.log) >= 1
    assert session.log.log[0].severity == scenario["expected_first_severity"]
    assert session.get_log() == session.log


def test_session_submit_does_not_print_log_on_success(capsys: pytest.CaptureFixture[str]) -> None:
    scenario = SESSION_SCENARIOS["load_submit_dataset_access"]
    session = Session()
    session.load("inp", pa.table(scenario["inputs"]["inp"]))

    result = session.submit(scenario["dsl"])
    captured = capsys.readouterr()

    assert result.success is True
    assert captured.out == ""


def test_session_submit_prints_log_on_failure(capsys: pytest.CaptureFixture[str]) -> None:
    scenario = SESSION_SCENARIOS["log_returns_entries"]
    session = Session()

    result = session.submit(scenario["dsl"])
    captured = capsys.readouterr()

    assert result.success is False
    assert "Error" in captured.out


def test_session_submit_parse_failure_prints_source_excerpt(capsys: pytest.CaptureFixture[str]) -> None:
    session = Session()

    result = session.submit("data out;\n    invalid syntax;\nrun;")
    captured = capsys.readouterr()
    rendered = _strip_ansi(captured.out)

    assert result.success is False
    assert "parse_unsupported_statement" in rendered.lower()
    assert "<dsl>:2:5" in rendered
    assert "invalid syntax;" in rendered
    assert "syntax error" in rendered.lower()


def test_session_submit_ignores_leading_comment_statement() -> None:
    session = Session()
    session.load("inp", pa.table({"id": [1, 2]}))

    result = session.submit("* comment;\ndata out;\n  set inp;\nrun;")

    assert result.success is True
    assert session["out"].to_pylist() == [{"id": 1}, {"id": 2}]


def test_session_submit_ignores_block_comment_with_semicolon() -> None:
    session = Session()
    session.load("inp", pa.table({"id": [1, 2]}))

    result = session.submit("data out;\n  /* comment ; still comment */\n  set inp;\nrun;")

    assert result.success is True
    assert session["out"].to_pylist() == [{"id": 1}, {"id": 2}]


def test_session_submit_validation_error_shows_category_label() -> None:
    session = Session()

    result = session.submit("data out;\n  set dummy;\nrun;")

    rendered = _strip_ansi(result.format_log())
    assert result.success is False
    assert "runtime_set_dataset_not_found" in rendered.lower()
    assert "missing dataset" in rendered.lower()
    assert "set dummy;" in rendered.lower()


def test_session_submit_execute_error_uses_statement_excerpt() -> None:
    session = Session()
    session.load("inp", pa.table({"amount": [10]}))

    result = session.submit("data out;\n  set inp;\n  total = unknown_func(amount);\nrun;")

    rendered = _strip_ansi(result.format_log())
    assert result.success is False
    assert "runtime_expression_evaluation_error" in rendered.lower()
    assert "[stage: execute]" in rendered.lower()
    assert "total = unknown_func(amount)" in rendered
    assert "expression error" in rendered.lower()
    assert "^" in rendered


def test_session_submit_execute_where_error_uses_statement_excerpt() -> None:
    session = Session()
    session.load("inp", pa.table({"amount": [10]}))

    result = session.submit("data out;\n  set inp;\n  where amount >< 1;\nrun;")

    rendered = _strip_ansi(result.format_log())
    assert result.success is False
    assert "runtime_operator_not_supported" in rendered.lower()
    assert "[stage: execute]" in rendered.lower()
    assert "where amount >< 1" in rendered.lower()
    assert "operator error" in rendered.lower()
    assert "^" in rendered


def test_session_tracks_last_submit_result() -> None:
    scenario = SESSION_SCENARIOS["load_submit_dataset_access"]
    session = Session()
    assert session.log is None

    session.load("inp", pa.table(scenario["inputs"]["inp"]))
    result = session.submit(scenario["dsl"])

    assert result.success is True
    assert session.get_log() is result
   
def test_submit_result_print_log_on_success_shows_status_and_elapsed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    scenario = SESSION_SCENARIOS["load_submit_dataset_access"]
    session = Session()
    session.load("inp", pa.table(scenario["inputs"]["inp"]))

    result = session.submit(scenario["dsl"])
    _ = capsys.readouterr()

    result.print_log()
    captured = capsys.readouterr()

    assert result.success is True
    assert "success: True" in captured.out
    assert "seconds elapsed" in captured.out
    assert "(no log entries)" in captured.out


def test_submit_result_print_log_on_failure_shows_status_elapsed_and_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    scenario = SESSION_SCENARIOS["log_returns_entries"]
    session = Session()

    result = session.submit(scenario["dsl"])
    _ = capsys.readouterr()

    result.print_log()
    captured = capsys.readouterr()

    assert result.success is False
    assert "success: False" in captured.out
    assert "seconds elapsed" in captured.out
    assert "Error" in captured.out


def _failed_submit_result_for_renderer() -> SubmitResult:
    span = DiagnosticSpan(start=10, end=17, line=1, column=11, end_line=1, end_column=18)
    return SubmitResult(
        success=False,
        log=(
            LogEntry(
                code="TEST_RENDER",
                severity="error",
                message="renderer test failure",
                stage="parse",
                span=span,
                labels=(DiagnosticLabel(span=span, message="problem"),),
                source_text="data out; invalid syntax; run;",
            ),
        ),
        elapsed_seq=0.01,
    )


def test_submit_result_print_log_uses_native_renderer_when_available(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = _failed_submit_result_for_renderer()
    fake_module = SimpleNamespace(render_diagnostics_ariadne=lambda payload: "native-rendered")

    with patch("limulus.renderer.load_native_module", return_value=(fake_module, None)):
        result.print_log()
    captured = capsys.readouterr()

    assert "native-rendered" in captured.out


def test_submit_result_print_log_falls_back_when_native_renderer_raises(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = _failed_submit_result_for_renderer()

    def _raise_renderer(payload: object) -> str:
        raise RuntimeError("render failed")

    fake_module = SimpleNamespace(render_diagnostics_ariadne=_raise_renderer)

    with patch("limulus.renderer.load_native_module", return_value=(fake_module, None)):
        result.print_log()
    captured = capsys.readouterr()

    assert "error[test_render]" in captured.out.lower()
    assert "renderer test failure" in captured.out
    assert "^" in captured.out


def test_session_submit_invalid_prx_pattern_renders_source_aware_excerpt() -> None:
    session = Session()
    session.load("in", pa.table({"name": ["Alice"]}))

    result = session.submit("data out; set in; where prxmatch('/[/', name) > 0; output out; run;")

    rendered = _strip_ansi(result.format_log())
    assert result.success is False
    assert "runtime_function_argument_invalid" in rendered.lower()
    assert "<prxmatch-pattern>" in rendered
    assert "/[/" in rendered
    assert "regex syntax" in rendered.lower()
    assert "^" in rendered


def test_session_submit_invalid_prx_flag_renders_source_aware_excerpt() -> None:
    session = Session()
    session.load("in", pa.table({"name": ["Alice"]}))

    result = session.submit("data out; set in; where prxmatch('/foo/z', name) > 0; output out; run;")

    rendered = _strip_ansi(result.format_log())
    assert result.success is False
    assert "runtime_function_argument_invalid" in rendered.lower()
    assert "<prxmatch-pattern>" in rendered
    assert "/foo/z" in rendered
    assert "unsupported prx flag" in rendered.lower()
    assert "^" in rendered


def test_native_ariadne_renderer_renders_multiple_diagnostics_and_notes() -> None:
    native_module, error = load_native_module()

    assert native_module is not None, error
    if not callable(getattr(native_module, "render_diagnostics_ariadne", None)):
        pytest.skip("installed native module does not expose render_diagnostics_ariadne in this test environment")

    rendered = native_module.render_diagnostics_ariadne(
        {
            "source_id": "<dsl>",
            "source_text": "data out; set in; keep id missing; run;",
            "diagnostics": [
                {
                    "code": "VALIDATE_COLUMN_NOT_FOUND",
                    "severity": "error",
                    "message": "KEEP statement references unknown variable: missing",
                    "stage": "validate",
                    "span": {
                        "start": 18,
                        "end": 33,
                        "line": 1,
                        "column": 19,
                        "end_line": 1,
                        "end_column": 34,
                        "source_id": "<dsl>",
                    },
                    "labels": [
                        {
                            "span": {
                                "start": 23,
                                "end": 30,
                                "line": 1,
                                "column": 24,
                                "end_line": 1,
                                "end_column": 31,
                                "source_id": "<dsl>",
                            },
                            "message": "unknown variable",
                            "kind": "primary",
                        },
                        {
                            "span": {
                                "start": 18,
                                "end": 22,
                                "line": 1,
                                "column": 19,
                                "end_line": 1,
                                "end_column": 23,
                                "source_id": "<dsl>",
                            },
                            "message": "KEEP clause",
                            "kind": "secondary",
                        },
                    ],
                    "notes": ["validate checks only statically derivable identifiers"],
                },
                {
                    "code": "PARSE_UNSUPPORTED_STATEMENT",
                    "severity": "error",
                    "message": "Unsupported statement syntax",
                    "stage": "parse",
                    "span": {
                        "start": 10,
                        "end": 17,
                        "line": 1,
                        "column": 11,
                        "end_line": 1,
                        "end_column": 18,
                        "source_id": "<dsl>",
                    },
                    "labels": [
                        {
                            "span": {
                                "start": 10,
                                "end": 17,
                                "line": 1,
                                "column": 11,
                                "end_line": 1,
                                "end_column": 18,
                                "source_id": "<dsl>",
                            },
                            "message": "syntax error",
                            "kind": "primary",
                        }
                    ],
                    "notes": ["second diagnostic"],
                    "source_text": "data out; invalid syntax; run;",
                },
            ],
        }
    )
    stripped = _strip_ansi(rendered)

    assert "VALIDATE_COLUMN_NOT_FOUND" in stripped
    assert "PARSE_UNSUPPORTED_STATEMENT" in stripped
    assert "KEEP clause" in stripped
    assert "unknown variable" in stripped
    assert "syntax error" in stripped
    assert "validate checks only statically derivable identifiers" in stripped
    assert "second diagnostic" in stripped


def test_session_backend_preferences_can_be_provided_at_init() -> None:
    scenario = SESSION_SCENARIOS["backend_preferences_init"]
    session = Session(runtime_backend=scenario["runtime_backend"], parser_backend=scenario["parser_backend"])
    session.load("inp", scenario["inputs"]["inp"])

    result = session.submit(scenario["dsl"])

    assert result.success is True
    assert result.datasets["out"].to_pylist() == scenario["expected_output"]


def test_session_auto_backend_prefers_rust_when_eligible() -> None:
    scenario = SESSION_SCENARIOS["auto_backend_prefers_rust"]
    session = Session(backend=scenario["backend"])
    table = pa.table(scenario["inputs"]["inp"])
    session.load("inp", table)

    result = session.submit(scenario["dsl"])

    assert result.success is True
    assert session["out"].to_pylist() == scenario["expected_output"]
    assert session._executor.last_runtime_backend == scenario["expected_runtime_backend"]


def test_session_acceptance_e2e_multi_output() -> None:
    scenario = SESSION_SCENARIOS["acceptance_e2e_multi_output"]
    executor = DataStepExecutor()
    request = ExecuteRequest(
        dsl_text=scenario["dsl"],
        inputs={
            "in": DataSetRef(
                kind="memory",
                location="dataset://in",
                payload=scenario["inputs"]["in"],
            )
        },
        output_targets=scenario["output_targets"],
    )

    response = executor.execute(request)

    assert response.has_errors is False
    assert response.outputs["out_a"].payload == scenario["expected_out_a"]
    assert response.outputs["out_b"].payload == scenario["expected_out_b"]


def test_session_acceptance_io_interop_arrow_pandas(tmp_path) -> None:
    scenario = SESSION_SCENARIOS["acceptance_io_interop"]

    pyarrow = pytest.importorskip("pyarrow")
    arrow_input = DataInputAdapterArrow()
    arrow_output = DataOutputAdapterArrow()

    arrow_table_rows = arrow_input.load(
        InputSpec(format="arrow_table", payload=pyarrow.table(scenario["arrow_input"]))
    )
    assert arrow_table_rows == scenario["arrow_expected"]

    stored = arrow_output.store(
        scenario["arrow_expected"],
        OutputSpec(format="arrow_table", location="memory://roundtrip"),
    )
    roundtrip_rows = arrow_input.load(InputSpec(format="arrow_table", payload=stored.payload))
    assert roundtrip_rows == scenario["arrow_expected"]

    pandas = pytest.importorskip("pandas")
    pandas_adapter = DataFrameAdapterPandas()
    frame = pandas.DataFrame(scenario["pandas_rows"])
    pandas_rows = pandas_adapter.load(InputSpec(format="pandas", payload=frame))
    assert pandas_rows == scenario["pandas_expected"]


def test_session_acceptance_merge_and_dataset_option_chain() -> None:
    scenario = SESSION_SCENARIOS["acceptance_merge_option_chain"]
    executor = DataStepExecutor()
    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="memory",
                    location="dataset://in",
                    payload=scenario["inputs"],
                )
            },
            output_targets=["out"],
        )
    )

    assert response.has_errors is False
    assert response.outputs["out"].payload == scenario["expected_output"]


def test_session_acceptance_output_dataset_options_multi_target() -> None:
    scenario = SESSION_SCENARIOS["acceptance_output_options_multi_target"]
    session = Session()
    session.load("ds", scenario["inputs"])

    result = session.submit(scenario["dsl"])

    assert result.success is True
    assert session["a"].to_pylist() == scenario["expected_a"]
    assert session["b"].to_pylist() == scenario["expected_b"]


def test_session_acceptance_output_dataset_options_with_python_runtime() -> None:
    scenario = SESSION_SCENARIOS["acceptance_output_options_python_runtime"]
    session = Session(runtime_backend=scenario["runtime_backend"])
    session.load("inp", pa.table(scenario["inputs"]))

    result = session.submit(scenario["dsl"])

    assert result.success is True
    assert session["out"].to_pylist() == scenario["expected_output"]


def test_session_acceptance_catalog_resolution_and_override() -> None:
    scenario = SESSION_SCENARIOS["acceptance_catalog_resolution"]
    executor = DataStepExecutor()
    executor.register_table(
        "in",
        DataSetRef(kind="memory", location="dataset://in", payload=scenario["catalog_payload"]),
    )

    from_catalog = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["catalog_dsl"],
            inputs={},
            output_targets=[],
        )
    )
    assert from_catalog.has_errors is False
    assert from_catalog.outputs["out"].payload == scenario["catalog_expected"]

    explicit = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["explicit_dsl"],
            inputs={"in": DataSetRef(kind="memory", location="dataset://in-explicit", payload=scenario["explicit_payload"])},
            output_targets=[],
        )
    )
    assert explicit.has_errors is False
    assert explicit.outputs["out"].payload == scenario["explicit_expected"]


def test_session_acceptance_register_tables_bulk_and_registered_resolution() -> None:
    scenario = SESSION_SCENARIOS["acceptance_register_tables_bulk"]
    executor = DataStepExecutor()
    executor.register_tables(
        {
            "in_a": pa.table({"id": [1]}),
            "in_b": DataSetRef(kind="memory", location="dataset://in_b", payload=[{"id": 2}]),
        }
    )

    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={},
            output_targets=["out"],
        )
    )

    assert response.has_errors is False
    assert response.outputs["out"].payload == scenario["expected_output"]


def test_session_acceptance_resolves_set_input_from_registered_table() -> None:
    scenario = SESSION_SCENARIOS["acceptance_resolve_registered_set_input"]
    executor = DataStepExecutor()
    executor.register_table(
        "in",
        DataSetRef(
            kind="arrow_table",
            location="dataset://in",
            payload=scenario["registered_payload"],
        ),
    )

    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={},
            output_targets=["out"],
        )
    )

    assert response.has_errors is False
    assert response.outputs["out"].payload == scenario["expected_output"]


def test_session_acceptance_previous_outputs_are_available_for_next_request() -> None:
    scenario = SESSION_SCENARIOS["acceptance_previous_outputs"]
    executor = DataStepExecutor()

    first = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["first_dsl"],
            inputs={"in": DataSetRef(kind="memory", location="dataset://in", payload=scenario["first_inputs"])},
        )
    )
    assert first.has_errors is False

    second = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["second_dsl"],
        )
    )

    assert second.has_errors is False
    assert second.outputs["out"].payload == scenario["expected_output"]


def test_session_acceptance_multi_block_request_chains_outputs() -> None:
    scenario = SESSION_SCENARIOS["acceptance_multi_block_chain"]
    executor = DataStepExecutor()
    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="memory",
                    location="dataset://in",
                    payload=scenario["inputs"],
                )
            },
        )
    )

    assert response.has_errors is False
    assert response.outputs["stage1"].payload == scenario["expected_stage1"]
    assert response.outputs["stage2"].payload == scenario["expected_stage2"]


def test_session_acceptance_multi_block_failure_contains_block_location() -> None:
    scenario = SESSION_SCENARIOS["acceptance_multi_block_failure"]
    executor = DataStepExecutor()
    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": DataSetRef(kind="memory", location="dataset://in", payload=scenario["inputs"])},
        )
    )

    assert response.has_errors is True
    assert response.diagnostics[0].code == scenario["expected_code"]
    assert response.diagnostics[0].location == scenario["expected_location"]


def test_session_acceptance_set_multiple_inputs_preserve_declared_order() -> None:
    scenario = SESSION_SCENARIOS["acceptance_set_multi_input_order"]
    executor = DataStepExecutor()
    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "a": DataSetRef(kind="memory", location="dataset://a", payload=scenario["inputs_a"]),
                "b": DataSetRef(kind="memory", location="dataset://b", payload=scenario["inputs_b"]),
                "c": DataSetRef(kind="memory", location="dataset://c", payload=scenario["inputs_c"]),
            },
            output_targets=["out"],
        )
    )

    assert response.has_errors is False
    assert response.outputs["out"].payload == scenario["expected_output"]


def test_session_acceptance_set_by_interleave_numeric() -> None:
    scenario = SESSION_SCENARIOS["acceptance_set_by_interleave_numeric"]
    executor = DataStepExecutor()
    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "a": DataSetRef(kind="memory", location="dataset://a", payload=scenario["inputs_a"]),
                "b": DataSetRef(kind="memory", location="dataset://b", payload=scenario["inputs_b"]),
            },
            output_targets=["out"],
        )
    )

    assert response.has_errors is False
    assert response.outputs["out"].payload == scenario["expected_output"]


def test_session_acceptance_set_by_interleave_string() -> None:
    scenario = SESSION_SCENARIOS["acceptance_set_by_interleave_string"]
    executor = DataStepExecutor()
    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "a": DataSetRef(kind="memory", location="dataset://a", payload=scenario["inputs_a"]),
                "b": DataSetRef(kind="memory", location="dataset://b", payload=scenario["inputs_b"]),
            },
            output_targets=["out"],
        )
    )

    assert response.has_errors is False
    assert response.outputs["out"].payload == scenario["expected_output"]


def test_session_acceptance_set_in_flag_can_drive_subset_logic() -> None:
    scenario = SESSION_SCENARIOS["acceptance_set_in_flag_subset"]
    executor = DataStepExecutor()
    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "a": DataSetRef(kind="memory", location="dataset://a", payload=scenario["inputs_a"]),
                "b": DataSetRef(kind="memory", location="dataset://b", payload=scenario["inputs_b"]),
            },
            output_targets=["out"],
        )
    )

    assert response.has_errors is False
    assert response.outputs["out"].payload == scenario["expected_output"]


def test_session_logs_info_for_skipped_unsupported_statements() -> None:
    scenario = SESSION_SCENARIOS["acceptance_skipped_unsupported_logs"]
    session = Session(runtime_backend="python", parser_backend="python")
    session.load("inp", pa.table({"id": [1]}))

    result = session.submit(scenario["dsl"])

    assert result.success is True
    skipped_logs = [entry for entry in result.log if entry.severity == "info" and "unsupported statement skipped" in entry.message]
    assert len(skipped_logs) == scenario["expected_skip_count"]


def test_session_acceptance_dataset_options_apply_keep_drop_rename_where_order() -> None:
    scenario = SESSION_SCENARIOS["acceptance_dataset_options_order"]
    executor = DataStepExecutor()
    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="memory",
                    location="dataset://in",
                    payload=scenario["inputs"],
                )
            },
            output_targets=["out"],
        )
    )

    assert response.has_errors is False
    assert response.outputs["out"].payload == scenario["expected_output"]


def test_session_acceptance_set_statement_options_indsname_and_end() -> None:
    scenario = SESSION_SCENARIOS["acceptance_set_statement_options"]
    executor = DataStepExecutor()
    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "a": DataSetRef(kind="memory", location="dataset://a", payload=scenario["inputs_a"]),
                "b": DataSetRef(kind="memory", location="dataset://b", payload=scenario["inputs_b"]),
            },
            output_targets=["out"],
        )
    )

    assert response.has_errors is False
    assert response.outputs["out"].payload == scenario["expected_output"]


def test_session_acceptance_merge_statement_end_option() -> None:
    scenario = SESSION_SCENARIOS["acceptance_merge_end_statement_option"]
    executor = DataStepExecutor(runtime_backend="python", parser_backend="python")
    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "a": DataSetRef(kind="memory", location="dataset://a", payload=scenario["inputs_a"]),
                "b": DataSetRef(kind="memory", location="dataset://b", payload=scenario["inputs_b"]),
            },
            output_targets=["out"],
        )
    )

    assert response.has_errors is False
    assert response.outputs["out"].payload == scenario["expected_output"]


def test_session_acceptance_set_by_rename_excludes_internal_variables() -> None:
    scenario = SESSION_SCENARIOS["acceptance_set_by_rename_internal_excluded"]
    executor = DataStepExecutor(runtime_backend="python", parser_backend="python")
    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="memory",
                    location="dataset://in",
                    payload=scenario["inputs"],
                )
            },
            output_targets=["out"],
        )
    )

    assert response.has_errors is False
    assert response.outputs["out"].payload == scenario["expected_output"]


def test_session_preserves_dataset_and_column_labels_in_arrow_metadata() -> None:
    scenario = SESSION_SCENARIOS["label_metadata"]
    session = Session(runtime_backend="python", parser_backend="python")
    session.load("inp", pa.table(scenario["inputs"]))

    result = session.submit(scenario["dsl"])

    assert result.success is True
    table = session.to_arrow("dm")
    assert table.schema.metadata[b"memlabel"] == b"DM"
    assert table.schema.field("id").metadata[b"label"] == b"Identifier"
    assert table.schema.field("amount").metadata[b"label"] == b"Amount"


def test_session_label_metadata_is_preserved_with_rust_runtime() -> None:
    scenario = SESSION_SCENARIOS["label_metadata"]
    session = Session(runtime_backend="rust", parser_backend="python")
    session.load("inp", pa.table(scenario["inputs"]))

    result = session.submit(scenario["dsl"])

    assert result.success is True
    assert session._executor.last_runtime_backend == "rust"
    table = session.to_arrow("dm")
    assert table.schema.metadata[b"memlabel"] == b"DM"
    assert table.schema.field("id").metadata[b"label"] == b"Identifier"
    assert table.schema.field("amount").metadata[b"label"] == b"Amount"


def test_session_acceptance_subset_if_matches_if_not_then_delete() -> None:
    scenario = SESSION_SCENARIOS["acceptance_subset_if"]

    for backend in ("python", "rust"):
        session_subset = Session(runtime_backend=backend, parser_backend="python")
        session_subset.load("in", scenario["inputs"])
        subset_result = session_subset.submit(scenario["subset_dsl"])

        session_delete = Session(runtime_backend=backend, parser_backend="python")
        session_delete.load("in", scenario["inputs"])
        delete_result = session_delete.submit(scenario["delete_equiv_dsl"])

        assert subset_result.success is True
        assert delete_result.success is True
        assert session_subset["out"].to_pylist() == session_delete["out"].to_pylist()
        assert session_subset["out"].to_pylist() == scenario["expected_output"]


def test_session_acceptance_stop_terminates_data_step() -> None:
    scenario = SESSION_SCENARIOS["acceptance_stop_statement"]

    for backend in ("python", "rust"):
        session = Session(runtime_backend=backend, parser_backend="python")
        session.load("in", scenario["inputs"])
        result = session.submit(scenario["dsl"])

        assert result.success is True
        assert session["out"].to_pylist() == scenario["expected_output"]


def test_session_acceptance_missing_assignment_dot_none_and_empty_string() -> None:
    scenario = SESSION_SCENARIOS["acceptance_missing_assignment"]

    for backend in ("python", "rust", "auto"):
        session = Session(runtime_backend=backend, parser_backend="python")
        session.load("in", scenario["inputs"])
        result = session.submit(scenario["dsl"])

        assert result.success is True
        assert session["out"].to_pylist() == scenario["expected_output"]


def test_session_acceptance_sum_statement_accumulates_across_rows() -> None:
    scenario = SESSION_SCENARIOS["acceptance_sum_statement"]
    executor = DataStepExecutor()
    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="memory",
                    location="dataset://in",
                    payload=scenario["inputs"],
                )
            },
        )
    )

    assert response.has_errors is False
    assert response.outputs["out"].payload == scenario["expected_output"]


def test_session_acceptance_if_then_do_and_sum_retain_behavior() -> None:
    scenario = SESSION_SCENARIOS["acceptance_if_do_sum_retain"]
    executor = DataStepExecutor()
    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": DataSetRef(kind="memory", location="dataset://in", payload=scenario["inputs"])},
        )
    )

    assert response.has_errors is False
    assert response.outputs["out"].payload == scenario["expected_output"]


def test_session_acceptance_do_end_executes_for_each_iteration() -> None:
    scenario = SESSION_SCENARIOS["acceptance_do_end_loop"]
    executor = DataStepExecutor()
    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": DataSetRef(kind="memory", location="dataset://in", payload=scenario["inputs"])},
        )
    )

    assert response.has_errors is False
    assert response.outputs["out"].payload == scenario["expected_output"]


def test_session_acceptance_if_then_do_nesting_limit() -> None:
    scenario = SESSION_SCENARIOS["acceptance_if_do_nesting_limit"]
    executor = DataStepExecutor()
    nested_if_count = scenario["nested_if_count"]
    statements = ["data out", "set in"]
    statements.extend("if id = 1 then do" for _ in range(nested_if_count))
    statements.extend(["output out"])
    statements.extend("end" for _ in range(nested_if_count))
    statements.extend(["run"])
    dsl_text = "; ".join(statements) + ";"

    response = executor.execute(
        ExecuteRequest(
            dsl_text=dsl_text,
            inputs={"in": DataSetRef(kind="memory", location="dataset://in", payload=[{"id": 1}])},
        )
    )

    assert response.has_errors is True
    assert response.diagnostics[0].code == scenario["expected_code"]


def test_session_acceptance_concat_and_lowercase_n_auto_variable() -> None:
    scenario = SESSION_SCENARIOS["acceptance_concat_and_n"]
    executor = DataStepExecutor()
    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="memory",
                    location="dataset://in",
                    payload=scenario["inputs"],
                )
            },
        )
    )

    assert response.has_errors is False
    assert response.outputs["out"].payload == scenario["expected_output"]


def test_session_acceptance_array_assignment_and_dim_vname() -> None:
    scenario = SESSION_SCENARIOS["acceptance_array_assignment_dim_vname"]
    executor = DataStepExecutor()
    response = executor.execute(
        ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="memory",
                    location="dataset://in",
                    payload=scenario["inputs"],
                )
            },
        )
    )

    assert response.has_errors is False
    assert response.outputs["out"].payload == scenario["expected_output"]


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_array_star_declaration_resolves_dimension_and_index_access(backend: str) -> None:
    scenario = SESSION_SCENARIOS["acceptance_array_star_case_and_by_case"]
    session = Session(runtime_backend=backend, parser_backend="python")
    session.load("inp", pa.table(scenario["array_inputs"]))

    result = session.submit(scenario["array_dsl"])

    assert result.success is True
    assert session["out"].to_pylist() == scenario["array_expected"]


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_auto_variables_are_case_insensitive_for_n_and_error(backend: str) -> None:
    scenario = SESSION_SCENARIOS["acceptance_array_star_case_and_by_case"]
    session = Session(runtime_backend=backend, parser_backend="python")
    session.load("inp", pa.table(scenario["auto_inputs"]))

    result = session.submit(scenario["auto_dsl"])

    assert result.success is True
    assert session["out"].to_pylist() == scenario["auto_expected"]


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_by_group_first_last_prefix_is_case_insensitive(backend: str) -> None:
    scenario = SESSION_SCENARIOS["acceptance_array_star_case_and_by_case"]
    session = Session(runtime_backend=backend, parser_backend="python")
    session.load("left", pa.table(scenario["left_inputs"]))
    session.load("right", pa.table(scenario["right_inputs"]))

    result = session.submit(scenario["by_dsl"])

    assert result.success is True
    assert session["out"].to_pylist() == scenario["by_expected"]


class ExecuteApiValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executor = DataStepExecutor()

    def test_accepts_valid_request(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["valid_request"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": DataSetRef(kind="memory", location="dataset://in")},
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(response.diagnostics, ())
        self.assertEqual(response.notices, ())

    def test_attaches_compatibility_notices_to_execution_response(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["compat_notice"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="memory",
                    location="dataset://in",
                    payload=scenario["inputs"],
                )
            },
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(len(response.notices), 1)
        self.assertEqual(response.notices[0].id, scenario["notice_id"])

    def test_rejects_empty_dsl(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["reject_empty_dsl"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": DataSetRef(kind="memory", location="dataset://in")},
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertTrue(response.has_errors)
        self.assertEqual(response.diagnostics[0].code, scenario["expected_code"])

    def test_rejects_invalid_input_mapping(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["reject_invalid_inputs"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"": DataSetRef(kind="memory", location="dataset://in")},
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertTrue(response.has_errors)
        self.assertEqual(response.diagnostics[0].code, scenario["expected_code"])

    def test_rejects_invalid_output_targets(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["reject_invalid_output_targets"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": DataSetRef(kind="memory", location="dataset://in")},
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertTrue(response.has_errors)
        self.assertEqual(response.diagnostics[0].code, scenario["expected_code"])

    def test_resolves_output_targets_from_data_statement_when_request_omits_them(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["resolve_targets_data_statement"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": DataSetRef(kind="memory", location="dataset://in", payload=scenario["inputs"])},
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].payload, scenario["expected_output"])

    def test_exposes_outputs_arrow_view_for_result_reuse(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["outputs_arrow_reuse"]
        response = self.executor.execute(
            ExecuteRequest(
                dsl_text=scenario["first_dsl"],
                inputs={"in": DataSetRef(kind="memory", location="dataset://in", payload=scenario["first_inputs"])},
            )
        )

        self.assertFalse(response.has_errors)
        self.assertIn("out", response.outputs_arrow)
        self.assertEqual(response.outputs_arrow["out"].to_pylist(), scenario["first_expected"])

        reused = self.executor.execute(
            ExecuteRequest(
                dsl_text=scenario["second_dsl"],
                inputs={"out": DataSetRef(kind="arrow_table", location="dataset://out", payload=response.outputs_arrow["out"])},
            )
        )

        self.assertFalse(reused.has_errors)
        self.assertEqual(reused.outputs["out2"].payload, scenario["second_expected"])

    def test_convert_outputs_returns_arrow_and_pylist_by_explicit_request(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["convert_outputs_arrow_pylist"]
        response = self.executor.execute(
            ExecuteRequest(
                dsl_text=scenario["dsl"],
                inputs={"in": DataSetRef(kind="memory", location="dataset://in", payload=scenario["inputs"])},
            )
        )

        arrow_converted = self.executor.convert_outputs(response, "arrow_table")
        pylist_converted = self.executor.convert_outputs(response, "pylist")

        self.assertFalse(arrow_converted.has_errors)
        self.assertFalse(pylist_converted.has_errors)
        self.assertEqual(arrow_converted.outputs["out"].to_pylist(), scenario["expected_output"])
        self.assertEqual(pylist_converted.outputs["out"], scenario["expected_output"])

    def test_outputs_arrow_is_materialized_lazily(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["outputs_arrow_lazy_materialize"]
        call_count = {"store": 0}
        original_store = self.executor._arrow_output.store

        def counting_store(*args, **kwargs):
            call_count["store"] += 1
            return original_store(*args, **kwargs)

        self.executor._arrow_output.store = counting_store
        response = self.executor.execute(
            ExecuteRequest(
                dsl_text=scenario["dsl"],
                inputs={"in": DataSetRef(kind="memory", location="dataset://in", payload=scenario["inputs"])},
            )
        )

        self.assertFalse(response.has_errors)
        self.assertEqual(call_count["store"], 0)

        arrow_table = response.outputs_arrow.get("out")
        self.assertIsNotNone(arrow_table)
        self.assertEqual(call_count["store"], 1)

    def test_simple_set_output_path_reuses_row_objects_without_extra_materialize(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["simple_set_reuse_rows"]
        input_rows = scenario["inputs"]
        response = self.executor.execute(
            ExecuteRequest(
                dsl_text=scenario["dsl"],
                inputs={"in": DataSetRef(kind="memory", location="dataset://in", payload=input_rows)},
            )
        )

        self.assertFalse(response.has_errors)
        self.assertIs(response.outputs["out"].payload[0], input_rows[0])
        self.assertIs(response.outputs["out"].payload[1], input_rows[1])

    def test_pylist_conversion_is_only_performed_by_explicit_convert_api(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["pylist_only_explicit"]
        table = _CountingToPylistTable(scenario["rows"])
        response = ExecuteResponse(
            outputs={"out": DataSetRef(kind="arrow_table", location="dataset://out", payload=table)},
            outputs_arrow={"out": table},
        )

        arrow_converted = self.executor.convert_outputs(response, "arrow_table")
        self.assertFalse(arrow_converted.has_errors)
        self.assertEqual(table.to_pylist_calls, 0)

        pylist_converted = self.executor.convert_outputs(response, "pylist")
        self.assertFalse(pylist_converted.has_errors)
        self.assertEqual(pylist_converted.outputs["out"], scenario["rows"])
        self.assertEqual(table.to_pylist_calls, 1)

    def test_rust_backend_preference_falls_back_to_python_backend(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["rust_runtime_fallback"]
        rust_executor = DataStepExecutor(runtime_backend="rust")
        response = rust_executor.execute(
            ExecuteRequest(
                dsl_text=scenario["dsl"],
                inputs={"in": DataSetRef(kind="memory", location="dataset://in", payload=scenario["inputs"])},
            )
        )

        self.assertFalse(response.has_errors)
        self.assertEqual(rust_executor.last_runtime_backend, scenario["expected_runtime_backend"])

    def test_rust_backend_preference_falls_back_to_python_when_apply_is_present(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["rust_runtime_fallback_apply"]

        def double(value):
            return value * 2

        globals()["double"] = double
        rust_executor = DataStepExecutor(runtime_backend="rust")
        response = rust_executor.execute(
            ExecuteRequest(
                dsl_text=scenario["dsl"],
                inputs={"in": DataSetRef(kind="memory", location="dataset://in", payload=scenario["inputs"])},
            )
        )

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].payload, scenario["expected_output"])
        self.assertEqual(rust_executor.last_runtime_backend, scenario["expected_runtime_backend"])

    def test_rust_parser_preference_uses_rust_parser_for_supported_subset(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["rust_parser_preference"]
        rust_parser_executor = DataStepExecutor(parser_backend="rust")
        response = rust_parser_executor.execute(
            ExecuteRequest(
                dsl_text=scenario["dsl"],
                inputs={"in": DataSetRef(kind="memory", location="dataset://in", payload=scenario["inputs"])},
            )
        )

        self.assertFalse(response.has_errors)
        self.assertEqual(rust_parser_executor.last_parser_backend, scenario["expected_parser_backend"])

    def test_convert_outputs_reports_diagnostic_for_unsupported_format(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["convert_unsupported_format"]
        response = self.executor.execute(
            ExecuteRequest(
                dsl_text=scenario["dsl"],
                inputs={"in": DataSetRef(kind="memory", location="dataset://in", payload=scenario["inputs"])},
            )
        )

        converted = self.executor.convert_outputs(response, scenario["format"])

        self.assertTrue(converted.has_errors)
        self.assertEqual(converted.diagnostics[0].code, scenario["expected_code"])

    def test_convert_outputs_reports_diagnostic_when_dataset_cannot_be_converted(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["convert_failed_dataset"]
        response = ExecuteResponse(
            outputs={"out": DataSetRef(kind="memory", location="dataset://out", payload=object())}
        )

        converted = self.executor.convert_outputs(response, "arrow_table")

        self.assertTrue(converted.has_errors)
        self.assertEqual(converted.diagnostics[0].code, scenario["expected_code"])

    def test_resolves_output_targets_from_data_and_output_statements(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["resolve_targets_data_and_output"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": DataSetRef(kind="memory", location="dataset://in", payload=scenario["inputs"])},
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out_a"].payload, scenario["expected_out_a"])
        self.assertEqual(response.outputs["out_b"].payload, scenario["expected_out_b"])

    def test_supports_set_less_data_step_with_do_and_output(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["setless_do_output"]
        response = self.executor.execute(
            ExecuteRequest(
                dsl_text=scenario["dsl"]
            )
        )

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].payload, scenario["expected_output"])

    def test_returns_runtime_error_when_set_dataset_cannot_be_resolved(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["set_missing_dataset"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={},
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertTrue(response.has_errors)
        self.assertEqual(response.diagnostics[0].code, scenario["expected_code"])

    def test_invalid_dataset_option_expression_returns_identifiable_diagnostic(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["invalid_dataset_option_expr"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": DataSetRef(kind="memory", location="dataset://in", payload=scenario["inputs"])},
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertTrue(response.has_errors)
        self.assertEqual(response.diagnostics[0].code, scenario["expected_code"])

    def test_reserved_auto_variable_name_in_options_returns_diagnostic(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["reserved_auto_var_option"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": DataSetRef(kind="memory", location="dataset://in", payload=scenario["inputs"])},
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertTrue(response.has_errors)
        self.assertEqual(response.diagnostics[0].code, scenario["expected_code"])

    def test_rename_statement_renames_output_columns(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["rename_statement_success"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": DataSetRef(kind="memory", location="dataset://in", payload=scenario["inputs"])},
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].payload, scenario["expected_output"])

    def test_invalid_rename_statement_mapping_returns_diagnostic(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["rename_statement_invalid"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": DataSetRef(kind="memory", location="dataset://in", payload=scenario["inputs"])},
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertTrue(response.has_errors)
        self.assertEqual(response.diagnostics[0].code, scenario["expected_code"])

    def test_merge_by_generates_first_last_flags(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["merge_first_last_flags"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "a": DataSetRef(
                    kind="memory",
                    location="dataset://a",
                    payload=scenario["inputs_a"],
                ),
                "b": DataSetRef(
                    kind="memory",
                    location="dataset://b",
                    payload=scenario["inputs_b"],
                ),
            },
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].payload, scenario["expected_output"])

    def test_merge_by_with_in_flag_can_filter_left_rows(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["merge_in_flag_filter"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "a": DataSetRef(kind="memory", location="dataset://a", payload=scenario["inputs_a"]),
                "b": DataSetRef(kind="memory", location="dataset://b", payload=scenario["inputs_b"]),
            },
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].payload, scenario["expected_output"])

    def test_internal_reference_variables_are_excluded_even_when_keep_specifies_them(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["internal_vars_excluded_keep"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "a": DataSetRef(kind="memory", location="dataset://a", payload=scenario["inputs_a"]),
                "b": DataSetRef(kind="memory", location="dataset://b", payload=scenario["inputs_b"]),
            },
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].payload, scenario["expected_output"])

    def test_internal_reference_variables_remain_excluded_after_rename(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["internal_vars_excluded_after_rename"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": DataSetRef(kind="memory", location="dataset://in", payload=scenario["inputs"])},
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].payload, scenario["expected_output"])

    def test_collision_with_input_column_and_internal_reference_variable_returns_diagnostic(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["collision_internal_var"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="memory",
                    location="dataset://in",
                    payload=scenario["inputs"],
                )
            },
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertTrue(response.has_errors)
        self.assertEqual(response.diagnostics[0].code, scenario["expected_code"])
        self.assertIn(scenario["expected_message_fragment"], response.diagnostics[0].message)

    def test_collision_with_input_column_and_first_last_variable_returns_diagnostic(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["collision_first_last_var"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "a": DataSetRef(kind="memory", location="dataset://a", payload=scenario["inputs_a"]),
                "b": DataSetRef(kind="memory", location="dataset://b", payload=scenario["inputs_b"]),
            },
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertTrue(response.has_errors)
        self.assertEqual(response.diagnostics[0].code, scenario["expected_code"])
        self.assertIn(scenario["expected_message_fragment"], response.diagnostics[0].message)

    def test_not_equal_operator_aliases_are_supported(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["not_equal_aliases"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="memory",
                    location="dataset://in",
                    payload=scenario["inputs"],
                )
            },
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].payload, scenario["expected_output"])

    def test_in_operator_is_supported_in_where_expression(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["in_operator"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="memory",
                    location="dataset://in",
                    payload=scenario["inputs"],
                )
            },
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].payload, scenario["expected_output"])

    def test_invalid_operator_returns_identifiable_diagnostic(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["invalid_operator"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": DataSetRef(kind="memory", location="dataset://in", payload=scenario["inputs"])},
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertTrue(response.has_errors)
        self.assertEqual(response.diagnostics[0].code, scenario["expected_code"])

    def test_prxmatch_is_supported(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["prxmatch"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="memory",
                    location="dataset://in",
                    payload=scenario["inputs"],
                )
            },
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].payload, scenario["expected_output"])

    def test_prxchange_is_supported(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["prxchange"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="memory",
                    location="dataset://in",
                    payload=scenario["inputs"],
                )
            },
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].payload, scenario["expected_output"])

    def test_invalid_prx_pattern_returns_identifiable_diagnostic(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["invalid_prx_pattern"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": DataSetRef(kind="memory", location="dataset://in", payload=scenario["inputs"])},
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertTrue(response.has_errors)
        self.assertEqual(response.diagnostics[0].code, scenario["expected_code"])

    def test_internal_reference_variables_are_numeric_zero_or_one(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["internal_vars_numeric"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "a": DataSetRef(kind="memory", location="dataset://a", payload=scenario["inputs_a"]),
                "b": DataSetRef(kind="memory", location="dataset://b", payload=scenario["inputs_b"]),
            },
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].payload, scenario["expected_output"])

    def test_sum_statement_accepts_first_last_numeric_flags(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["sum_accepts_first_last"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "a": DataSetRef(kind="memory", location="dataset://a", payload=scenario["inputs_a"]),
                "b": DataSetRef(kind="memory", location="dataset://b", payload=scenario["inputs_b"]),
            },
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].payload[0]["seq"], scenario["expected_first_seq"])

    def test_array_dimension_syntax_and_if_eof_default_output_work_together(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["array_dim_and_eof_default_output"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="memory",
                    location="dataset://in",
                    payload=scenario["inputs"],
                )
            },
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].payload, scenario["expected_output"])

    def test_expanded_string_functions_are_supported(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["expanded_string_functions"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="memory",
                    location="dataset://in",
                    payload=scenario["inputs"],
                )
            },
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].payload, scenario["expected_output"])

    def test_expanded_numeric_functions_are_supported(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["expanded_numeric_functions"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="memory",
                    location="dataset://in",
                    payload=scenario["inputs"],
                )
            },
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].payload, scenario["expected_output"])

    def test_common_functions_and_lead_apply_are_supported(self) -> None:
        # ensure the helper function is available in the call stack rather than
        # relying on a registry
        def double(value):
            return value * 2
        globals()["double"] = double

        scenario = EXECUTE_API_SCENARIOS["common_funcs_lead_apply"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="memory",
                    location="dataset://in",
                    payload=scenario["inputs"],
                )
            },
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].payload, scenario["expected_output"])

    def test_apply_can_use_python_builtin_and_module_names_without_registration(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["apply_python_names"]

        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="memory",
                    location="dataset://in",
                    payload=scenario["inputs"],
                )
            },
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].payload, scenario["expected_output"])

    def test_apply_looks_up_user_defined_function_without_registration(self) -> None:
        # define a helper in the local/global namespace prior to execution
        def myfunc(v):
            # returns 10 more than the input; only the last row (20) yields 30
            return v + 10

        # inject into globals so _apply can find it via stack inspection
        globals()["myfunc"] = myfunc

        scenario = EXECUTE_API_SCENARIOS["apply_user_defined"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="memory",
                    location="dataset://in",
                    payload=scenario["inputs"],
                )
            },
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].payload, scenario["expected_output"])

    def test_assignment_function_dispatch_uses_runtime_registry_entrypoint(self) -> None:
        # defining the helper in globals is sufficient
        def double(value):
            return value * 2
        globals()["double"] = double

        scenario = EXECUTE_API_SCENARIOS["assignment_dispatch_runtime_registry"]
        response = self.executor.execute(
            ExecuteRequest(
                dsl_text=scenario["dsl"],
                inputs={
                    "in": DataSetRef(
                        kind="memory",
                        location="dataset://in",
                        payload=scenario["inputs"],
                    )
                },
                output_targets=scenario["output_targets"],
            )
        )

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].payload, scenario["expected_output"])

    def test_merge_by_reports_precondition_failure_when_by_key_missing(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["merge_by_precondition_failed"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "a": DataSetRef(kind="memory", location="dataset://a", payload=scenario["inputs_a"]),
                "b": DataSetRef(kind="memory", location="dataset://b", payload=scenario["inputs_b"]),
            },
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertTrue(response.has_errors)
        self.assertEqual(response.diagnostics[0].code, scenario["expected_code"])

    def test_merge_by_reports_duplicate_non_key_columns(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["merge_duplicate_column"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "a": DataSetRef(kind="memory", location="dataset://a", payload=scenario["inputs_a"]),
                "b": DataSetRef(kind="memory", location="dataset://b", payload=scenario["inputs_b"]),
            },
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertTrue(response.has_errors)
        self.assertEqual(response.diagnostics[0].code, scenario["expected_code"])

    def test_conditional_delete_excludes_row_from_output(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["conditional_delete"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="memory",
                    location="dataset://in",
                    payload=scenario["inputs"],
                )
            },
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].payload, scenario["expected_output"])

    def test_unconditional_delete_excludes_all_rows(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["unconditional_delete"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="memory",
                    location="dataset://in",
                    payload=scenario["inputs"],
                )
            },
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].payload, scenario["expected_output"])

    def test_string_functions_are_supported_in_where_expression(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["where_string_funcs"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="memory",
                    location="dataset://in",
                    payload=scenario["inputs"],
                )
            },
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].payload, scenario["expected_output"])

    def test_numeric_functions_are_supported_in_if_expression(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["if_numeric_funcs"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="memory",
                    location="dataset://in",
                    payload=scenario["inputs"],
                )
            },
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].payload, scenario["expected_output"])

    def test_date_functions_are_supported_in_where_expression(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["where_date_funcs"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={
                "in": DataSetRef(
                    kind="memory",
                    location="dataset://in",
                    payload=scenario["inputs"],
                )
            },
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertFalse(response.has_errors)
        self.assertEqual(response.outputs["out"].payload, scenario["expected_output"])

    def test_unsupported_function_returns_identifiable_error(self) -> None:
        scenario = EXECUTE_API_SCENARIOS["unsupported_function"]
        request = ExecuteRequest(
            dsl_text=scenario["dsl"],
            inputs={"in": DataSetRef(kind="memory", location="dataset://in", payload=scenario["inputs"])},
            output_targets=scenario["output_targets"],
        )

        response = self.executor.execute(request)

        self.assertTrue(response.has_errors)
        self.assertEqual(response.diagnostics[0].code, scenario["expected_code"])
        self.assertIn(scenario["expected_message_fragment"], response.diagnostics[0].message)
