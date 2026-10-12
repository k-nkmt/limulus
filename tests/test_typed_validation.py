"""Typed validation entry point contracts."""

from __future__ import annotations

import ast as ast_module
import builtins
from unittest.mock import patch

import pyarrow as pa

from limulus.execution import DataStepExecutor
from limulus.execution.contracts import (
    TypedValidationRequest,
    ValidationResult,
)
from limulus.execution.validation import ExecutionValidationService
from limulus.models import DataSetRef, Diagnostic
from limulus.syntax import compat as syntax_compat
from limulus.syntax import parse_program
from limulus.syntax.models import SourceDocument
from limulus.models import ExecuteRequest


def _source_ref(columns: tuple[str, ...] = ("id", "amount", "grp")) -> DataSetRef:
    return DataSetRef(
        kind="arrow_table",
        location="dataset://source",
        payload=pa.table({name: [1] for name in columns}),
    )


def _typed_request(
    dsl_text: str,
    *,
    explicit_inputs: dict[str, DataSetRef] | None = None,
    explicit_output_targets: tuple[str, ...] = (),
    registered_tables: dict[str, DataSetRef] | None = None,
) -> TypedValidationRequest:
    parsed = parse_program(dsl_text)
    assert parsed.program is not None
    assert parsed.has_errors is False
    return TypedValidationRequest(
        block=parsed.program.blocks[0],
        source=SourceDocument(dsl_text),
        explicit_inputs=explicit_inputs or {},
        available_inputs={},
        registered_tables=registered_tables or {},
        explicit_output_targets=explicit_output_targets,
    )


def _codes(result: ValidationResult) -> list[str]:
    return [diagnostic.code for diagnostic in result.diagnostics]


# --------------------------------------------------------------------------
# central contract: the typed entry point exists and mirrors the legacy
# reserved -> input -> column rejection order without touching the legacy path
# --------------------------------------------------------------------------


def test_typed_validation_entry_exists_and_returns_validation_result() -> None:
    service = ExecutionValidationService()
    request = _typed_request(
        "data out; set s; run;",
        explicit_inputs={"s": _source_ref()},
    )

    result = service.validate_typed(request)

    assert isinstance(result, ValidationResult)
    assert result.diagnostics == ()


def test_typed_validation_rejection_order() -> None:
    """The typed entry keeps the reserved -> input -> column rejection order."""

    service = ExecutionValidationService()
    cases = [
        # reserved DATA target is reported before the missing SET input
        (
            "data dictionary.tables; set missing; run;",
            ["VALIDATE_RESERVED_OUTPUT_TARGET", "RUNTIME_SET_DATASET_NOT_FOUND"],
        ),
        # a missing SET input suppresses the downstream column check
        ("data out; set missing; keep nope; run;", ["RUNTIME_SET_DATASET_NOT_FOUND"]),
        ("data out; set s; keep nope; run;", ["VALIDATE_COLUMN_NOT_FOUND"]),
        ("data out; set s; run;", []),
        ("data out; set s; if undef then y = 1; run;", []),
        ("data out; set s; rename a = b c = b; run;", ["RUNTIME_RENAME_STATEMENT_INVALID"]),
        # the inlined 'by grp' fragment is carried as extra SET refs, each
        # reported dataset-not-found
        (
            "data out; set s(in=flag) end=e indsname=d by grp; keep nope; run;",
            ["RUNTIME_SET_DATASET_NOT_FOUND", "RUNTIME_SET_DATASET_NOT_FOUND"],
        ),
    ]
    for dsl_text, expected_codes in cases:
        result = service.validate_typed(
            _typed_request(dsl_text, explicit_inputs={"s": _source_ref()})
        )
        assert _codes(result) == expected_codes, dsl_text


def test_typed_validation_duplicate_diagnostic_contract() -> None:
    """The DATA duplicate rejection is a typed-entry-only diagnostic with
    full error metadata."""

    service = ExecutionValidationService()
    dsl_text = "data a a; set s; run;"
    typed_result = service.validate_typed(
        _typed_request(dsl_text, explicit_inputs={"s": _source_ref()})
    )

    assert _codes(typed_result) == ["VALIDATE_DUPLICATE_DATA_TARGET"]
    diagnostic = typed_result.diagnostics[0]
    assert isinstance(diagnostic, Diagnostic)
    assert diagnostic.severity == "error"
    assert diagnostic.stage == "validate"
    assert diagnostic.source_text == dsl_text


# --------------------------------------------------------------------------
# DATA duplicate target: dataset-key normalized, option-independent
# --------------------------------------------------------------------------


def test_typed_validation_rejects_duplicate_data_targets() -> None:
    service = ExecutionValidationService()
    variants = [
        "data a a; set s; run;",
        "data a a(keep=id); set s; run;",
        "data a A; set s; run;",
        "data a work.a; set s; run;",
        "data a a(keep=id) work.a(obs=1); set s; run;",
    ]
    for dsl_text in variants:
        result = service.validate_typed(
            _typed_request(dsl_text, explicit_inputs={"s": _source_ref()})
        )
        assert _codes(result) == ["VALIDATE_DUPLICATE_DATA_TARGET"], dsl_text
        diagnostic = result.diagnostics[0]
        assert diagnostic.severity == "error"
        assert diagnostic.stage == "validate"
        assert "a" in diagnostic.message
        # The diagnostic span is the later duplicate reference occurrence.
        assert diagnostic.span is not None
        assert diagnostic.span.start > 0
        assert diagnostic.source_text == dsl_text


def test_typed_validation_duplicate_survives_explicit_targets_and_zero_rows() -> None:
    service = ExecutionValidationService()
    dsl_text = "data a a; set s; run;"
    explicit_result = service.validate_typed(
        _typed_request(
            dsl_text,
            explicit_inputs={"s": _source_ref()},
            explicit_output_targets=("z",),
        )
    )
    assert _codes(explicit_result) == ["VALIDATE_DUPLICATE_DATA_TARGET"]

    empty_ref = DataSetRef(
        kind="arrow_table",
        location="dataset://empty",
        payload=pa.table({"id": pa.array([], type=pa.int64())}),
    )
    zero_row_result = service.validate_typed(
        _typed_request(dsl_text, explicit_inputs={"s": empty_ref})
    )
    assert _codes(zero_row_result) == ["VALIDATE_DUPLICATE_DATA_TARGET"]


def test_typed_validation_duplicate_span_is_the_later_reference() -> None:
    service = ExecutionValidationService()
    dsl_text = "data a a; set s; run;"
    parsed = parse_program(dsl_text)
    block = parsed.program.blocks[0]
    data_statement = block.statements[0]
    first_ref, second_ref = data_statement.payload.outputs
    result = service.validate_typed(
        TypedValidationRequest(
            block=block,
            source=SourceDocument(dsl_text),
            explicit_inputs={"s": _source_ref()},
        )
    )

    diagnostic = result.diagnostics[0]
    duplicate_span = diagnostic.span
    assert duplicate_span is not None
    assert duplicate_span.start == second_ref.span.start
    assert duplicate_span.end == second_ref.span.end
    assert duplicate_span.start != first_ref.span.start
    # source-aware raw text: the span slices the duplicate reference name
    assert duplicate_span.source_id == "<dsl>"


def test_typed_validation_allows_output_writes_and_separate_blocks() -> None:
    service = ExecutionValidationService()
    source_inputs = {"s": _source_ref()}

    ok_cases = [
        # multiple OUTPUT statements writing to the same declared target
        "data out; set s; output out; output out; run;",
        # top-level and inline OUTPUT to the same target is not a DATA duplicate
        "data out; set s; if 0 then do; output out; end; output out; run;",
        # inline single-statement OUTPUT target
        "data out; set s; if 1 then output out; run;",
    ]
    for dsl_text in ok_cases:
        result = service.validate_typed(_typed_request(dsl_text, explicit_inputs=source_inputs))
        assert result.diagnostics == (), dsl_text

    # different blocks are separate requests: the same name is allowed again
    first = service.validate_typed(_typed_request("data out; set s; run;", explicit_inputs=source_inputs))
    second = service.validate_typed(_typed_request("data out; set s; run;", explicit_inputs=source_inputs))
    assert first.diagnostics == () and second.diagnostics == ()


# --------------------------------------------------------------------------
# schema binding through the existing owner: source options and name columns
# --------------------------------------------------------------------------


def test_typed_validation_applies_source_options_to_schema_binding() -> None:
    service = ExecutionValidationService()
    unknown_cases = [
        "data out; set s(keep=id); keep amount; run;",
        "data out; set s(drop=id); keep id; run;",
        "data out; set s(rename=(id=key)); keep id; run;",
    ]
    for dsl_text in unknown_cases:
        result = service.validate_typed(
            _typed_request(dsl_text, explicit_inputs={"s": _source_ref()})
        )
        assert _codes(result) == ["VALIDATE_COLUMN_NOT_FOUND"], dsl_text

    ok_cases = [
        "data out; set s(rename=(id=key)); keep key amount; run;",
        "data out; set s(in=flag) end=e indsname=d; keep flag e d amount; run;",
        "data out; set s; by grp; keep first.grp last.grp amount; run;",
        "data out; set s; x = 1; total + 1; keep x total amount; run;",
        "data out; set s; arr[1] = 1; keep arr amount; run;",
        # the ARRAY name itself is not a defined column (legacy contract);
        # only the declared member variables are
        "data out; set s; array arr[2] a b; keep a b amount; run;",
        "data out; set s; retain x y 5; keep x y amount; run;",
        "data out; set s; if x then y = 1; keep y amount; run;",
    ]
    for dsl_text in ok_cases:
        result = service.validate_typed(
            _typed_request(dsl_text, explicit_inputs={"s": _source_ref()})
        )
        assert result.diagnostics == (), dsl_text

    # the array name itself stays undefined for column checking (legacy parity)
    array_name = service.validate_typed(
        _typed_request(
            "data out; set s; array arr[2] a b; keep arr; run;",
            explicit_inputs={"s": _source_ref()},
        )
    )
    assert _codes(array_name) == ["VALIDATE_COLUMN_NOT_FOUND"]


def test_typed_validation_by_and_rename_contracts_match_legacy() -> None:
    service = ExecutionValidationService()
    inputs = {"s": _source_ref()}

    by_result = service.validate_typed(
        _typed_request("data out; set s; by absent; run;", explicit_inputs=inputs)
    )
    assert _codes(by_result) == ["RUNTIME_BY_PRECONDITION_FAILED"]

    rename_result = service.validate_typed(
        _typed_request("data out; set s; rename a = b c = b; run;", explicit_inputs=inputs)
    )
    assert _codes(rename_result) == ["RUNTIME_RENAME_STATEMENT_INVALID"]

    # ELSE inline actions keep the legacy defined-column contract: the ELSE
    # action target is not collected (no reachability judgment either).
    else_result = service.validate_typed(
        _typed_request(
            "data out; set s; if x then y = 1; else z = 2; keep z; run;",
            explicit_inputs=inputs,
        )
    )
    assert _codes(else_result) == ["VALIDATE_COLUMN_NOT_FOUND"]


def test_typed_validation_zero_row_schema_binding() -> None:
    service = ExecutionValidationService()
    empty_ref = DataSetRef(
        kind="arrow_table",
        location="dataset://empty",
        payload=pa.table({"id": pa.array([], type=pa.int64())}),
    )

    ok = service.validate_typed(
        _typed_request("data out; set s; keep id; run;", explicit_inputs={"s": empty_ref})
    )
    assert ok.diagnostics == ()

    missing = service.validate_typed(
        _typed_request("data out; set s; keep amount; run;", explicit_inputs={"s": empty_ref})
    )
    assert _codes(missing) == ["VALIDATE_COLUMN_NOT_FOUND"]


# --------------------------------------------------------------------------
# output options are preserved, not bulk-rejected
# --------------------------------------------------------------------------


def test_typed_validation_keeps_data_and_output_options() -> None:
    service = ExecutionValidationService()
    inputs = {"s": _source_ref()}
    ok_cases = [
        "data out(keep=id rename=(id=key)); set s; output out; run;",
        "data out; set s; output out(keep=id rename=(id=key)); run;",
        "data out(obs=1); set s; output out(firstobs=1); run;",
    ]
    for dsl_text in ok_cases:
        result = service.validate_typed(_typed_request(dsl_text, explicit_inputs=inputs))
        assert result.diagnostics == (), dsl_text


# --------------------------------------------------------------------------
# source-aware spans and quoted option parity
# --------------------------------------------------------------------------


def test_typed_validation_source_aware_unicode_spans() -> None:
    service = ExecutionValidationService()
    dsl_text = "data out(label='ラベル'); set missing; /* 日本語コメント */ run;"
    result = service.validate_typed(
        _typed_request(dsl_text, explicit_inputs={"s": _source_ref()})
    )

    assert _codes(result) == ["RUNTIME_SET_DATASET_NOT_FOUND"]
    diagnostic = result.diagnostics[0]
    assert diagnostic.span is not None
    assert diagnostic.source_text == dsl_text
    raw = dsl_text[diagnostic.span.start : diagnostic.span.end]
    assert raw == "set missing"


def test_typed_validation_quoted_keep_option_column_check() -> None:
    service = ExecutionValidationService()
    inputs = {"s": _source_ref()}
    dsl_text = "data out; set s(keep='my col'); keep x; run;"

    typed_result = service.validate_typed(_typed_request(dsl_text, explicit_inputs=inputs))

    assert _codes(typed_result) == ["VALIDATE_COLUMN_NOT_FOUND"]


# --------------------------------------------------------------------------
# typed request contract: snapshots and span/source agreement
# --------------------------------------------------------------------------


def test_typed_request_snapshots_catalog_inputs_and_is_immutable() -> None:
    source = _source_ref()
    inputs = {"s": source}
    registered = {"s": source}
    request = _typed_request(
        "data out; set s; run;",
        explicit_inputs=inputs,
        registered_tables=registered,
    )

    inputs.clear()
    registered.clear()

    assert tuple(request.explicit_inputs) == ("s",)
    assert tuple(request.registered_tables) == ("s",)
    for mapping in (request.explicit_inputs, request.registered_tables):
        try:
            mapping["other"] = source  # ty: ignore[invalid-assignment]
        except TypeError:
            pass
        else:
            raise AssertionError("typed validation snapshots must be immutable")


def test_typed_request_requires_matching_block_span_source() -> None:
    dsl_text = "data out; set s; run;"
    parsed = parse_program(dsl_text)
    block = parsed.program.blocks[0]

    try:
        TypedValidationRequest(
            block=block,
            source=SourceDocument(dsl_text, source_id="other"),
            explicit_inputs={"s": _source_ref()},
        )
    except ValueError:
        pass
    else:
        raise AssertionError("block span must belong to the request source document")


# --------------------------------------------------------------------------
# guard: no legacy scanner / DTO projection / reparse in the typed entry
# --------------------------------------------------------------------------


def test_typed_validation_does_not_reach_legacy_scanners_or_reparse() -> None:
    service = ExecutionValidationService()
    # Representative block exercising every typed structure the entry reads.
    dsl_text = (
        "data out; set s(rename=(id=key)) end=e indsname=d;"
        " by grp; x = 1; total + 1; arr[1] = 1; array arr2[2] a b;"
        " retain r1 r2 5; if x then y = 1; else z = 2;"
        " output out(keep=id); run;"
    )
    request = _typed_request(dsl_text, explicit_inputs={"s": _source_ref()})

    def _forbidden(name: str):
        def _raise(*args, **kwargs):
            raise AssertionError(f"typed validation reached legacy {name}")

        return _raise

    with (
        patch.object(
            service._io_service,
            "extract_data_targets",
            side_effect=AssertionError("legacy DATA target scanner"),
        ),
        patch.object(
            service._io_service,
            "extract_output_targets",
            side_effect=AssertionError("legacy OUTPUT scanner"),
        ),
        patch.object(syntax_compat, "project_result", _forbidden("legacy DTO projection")),
        patch.object(syntax_compat, "project_program", _forbidden("legacy DTO projection")),
        patch.object(syntax_compat, "project_block", _forbidden("legacy DTO projection")),
        patch.object(syntax_compat, "project_statement", _forbidden("legacy DTO projection")),
        patch.object(builtins, "compile", _forbidden("compile")),
        patch.object(ast_module, "parse", _forbidden("ast.parse")),
    ):
        result = service.validate_typed(request)

    assert result.diagnostics == (), _codes(result)


# --------------------------------------------------------------------------
# legacy path stays untouched
# --------------------------------------------------------------------------


def test_public_path_validation_diagnostics_unchanged() -> None:
    executor = DataStepExecutor()
    dsl_text = "data out; set source; rename missing = amount; run;"
    response = executor.execute(
        ExecuteRequest(
            dsl_text=dsl_text, inputs={"source": _source_ref(("id", "amount"))}
        )
    )
    public = response.diagnostics[0]

    assert public.code == "RUNTIME_RENAME_STATEMENT_INVALID"
    assert public.stage == "validate"
