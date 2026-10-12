"""Focused tests for the pure parser import boundary.

Cold subprocesses install a meta-path hook denying heavy dependencies
before importing the package."""

from __future__ import annotations

import inspect
import subprocess
import sys
import textwrap
from pathlib import Path

import limulus
import limulus.models as limulus_models
import limulus.parser as parser_module
import limulus.syntax as syntax_package
import limulus.syntax.legacy as legacy_module
from limulus.dataset_option_expressions import where_expression_to_dict as legacy_where_to_dict
from limulus.syntax.legacy import where_expression_to_dict as pure_where_to_dict

REPO_ROOT = Path(__file__).resolve().parents[1]

_DENIED_ROOTS = ("pyarrow", "polars", "limulus_native", "limulus.session")

_HOOK = textwrap.dedent(
    """
    import sys

    _DENIED = ("pyarrow", "polars", "limulus_native", "limulus.session")


    class _DeniedDependencyGate:
        def find_spec(self, fullname, path=None, target=None):
            for root in _DENIED:
                if fullname == root or fullname.startswith(root + "."):
                    raise ImportError(
                        f"module {fullname!r} is denied by the parser isolation test hook"
                    )
            return None


    sys.meta_path.insert(0, _DeniedDependencyGate())
    """
)


def _run_isolated(snippet: str) -> None:
    script = _HOOK + "\n" + textwrap.dedent(snippet)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"isolated import failed:\n{result.stdout}\n{result.stderr}"


def test_pure_syntax_import_does_not_load_session_or_data_libraries() -> None:
    _run_isolated(
        """
        import sys

        import limulus
        import limulus.syntax
        import limulus.syntax.legacy

        for denied in ("pyarrow", "polars", "limulus_native", "limulus.session"):
            assert denied not in sys.modules, denied
        """
    )


def test_legacy_dto_constructors_and_serialization_work_in_isolation() -> None:
    _run_isolated(
        """
        from limulus.syntax.legacy import (
            ArrayStatementSpec,
            DataStepAst,
            DatasetReference,
            DatasetReferenceOptionSpec,
            DoStatementSpec,
            IfStatementSpec,
            ParseResult,
            ParsedStatement,
            SetStatementOptionSpec,
            ast_to_dict,
            ast_statements_to_dict,
            statement_to_dict,
        )
        from limulus.syntax.models import WhereComparison

        options = DatasetReferenceOptionSpec(
            in_var="flag",
            keep_vars=("x", "y"),
            drop_vars=(),
            where_expr=WhereComparison(column="x", operator="==", value=1),
            rename_map={"a": "b"},
            firstobs=2,
            obs=10,
            label="demo",
        )
        reference = DatasetReference(name="in", options=options)
        statement = ParsedStatement(
            kind="SET",
            text="set in(where=(x == 1));",
            dataset_refs=(reference,),
            statement_options=SetStatementOptionSpec(indsname_var="src"),
            rename_map={"old": "new"},
            label_map={"id": "identifier"},
            if_spec=IfStatementSpec(condition="x == 1", is_subset=True),
            do_spec=DoStatementSpec(loop_var="i", start_expr="1", end_expr="10"),
            array_spec=ArrayStatementSpec(array_name="arr", variables=("x",), wildcard_size=True),
        )
        serialized = statement_to_dict(statement)
        assert serialized["dataset_refs"][0]["options"]["where_expr"] == {
            "kind": "comparison",
            "column": "x",
            "operator": "==",
            "value": 1,
        }
        assert serialized["statement_options"] == {"indsname_var": "src", "end_var": None}
        assert serialized["if_spec"] == {
            "condition": "x == 1",
            "then_action": None,
            "is_subset": True,
            "is_then_do": False,
        }
        assert serialized["do_spec"] == {"loop_var": "i", "start_expr": "1", "end_expr": "10"}
        assert serialized["array_spec"] == {
            "array_name": "arr",
            "variables": ["x"],
            "declared_size": None,
            "wildcard_size": True,
            "character_array": False,
        }
        assert ast_to_dict(DataStepAst(statements=(statement,))) == {"statements": [serialized]}
        assert ast_statements_to_dict((statement,)) == {"statements": [serialized]}

        result = ParseResult()
        assert result.has_errors is False
        """
    )


def test_parser_service_import_uses_only_lark_and_stdlib() -> None:
    _run_isolated(
        """
        import sys

        import limulus.parser as parser_module

        for denied in ("pyarrow", "polars", "limulus_native", "limulus.session"):
            assert denied not in sys.modules, denied

        service = parser_module.ParserService()
        result = service.parse("data out; set in(where=(x == 1)); run;")
        assert not result.has_errors
        assert [statement.kind for statement in result.ast.statements] == ["DATA", "SET", "RUN"]
        """
    )


def test_legacy_parser_names_reexport_the_same_class_objects() -> None:
    from limulus.parser import DataStepAst, ParseResult, ParsedStatement, ParserService
    from limulus.syntax.legacy import (
        ArrayStatementSpec,
        DataStepAst as legacy_DataStepAst,
        DatasetReference,
        DatasetReferenceOptionSpec,
        DoStatementSpec,
        IfStatementSpec,
        ParseResult as legacy_ParseResult,
        ParsedStatement as legacy_ParsedStatement,
        SetStatementOptionSpec,
        StatementRegion,
    )

    assert DataStepAst is legacy_DataStepAst
    assert ParseResult is legacy_ParseResult
    assert ParsedStatement is legacy_ParsedStatement
    for name in (
        "ArrayStatementSpec",
        "DatasetReference",
        "DatasetReferenceOptionSpec",
        "DoStatementSpec",
        "IfStatementSpec",
        "SetStatementOptionSpec",
        "StatementRegion",
    ):
        assert getattr(parser_module, name) is getattr(legacy_module, name), name
    assert ParserService is parser_module.LarkParserFrontend


def test_package_lazy_exports_resolve_to_the_same_objects() -> None:
    assert limulus.ParsedStatement is parser_module.ParsedStatement
    assert limulus.ParseResult is parser_module.ParseResult
    assert limulus.DataStepAst is parser_module.DataStepAst
    assert limulus.ParserService is parser_module.ParserService
    assert limulus.Diagnostic is limulus_models.Diagnostic
    assert limulus.SubmitResult is limulus_models.SubmitResult


def test_syntax_package_exposes_the_legacy_parser_view() -> None:
    assert syntax_package.ParseResult is legacy_module.ParseResult
    assert syntax_package.ParsedStatement is legacy_module.ParsedStatement
    assert syntax_package.DatasetReference is legacy_module.DatasetReference


def test_package_all_list_is_unchanged() -> None:
    assert limulus.__all__ == [
        "Session",
        "submit",
        "run",
        "SubmitResult",
        "LogEntry",
        "DatasetCatalog",
        "DataStepExecutor",
        "FormatSupportResult",
        "RuntimeRequirements",
        "DataAdapterError",
        "DataFrameAdapterPandas",
        "DataInputAdapterArrow",
        "DataOutputAdapterArrow",
        "InputSpec",
        "OutputSpec",
        "CompatibilityNotice",
        "DataSetRef",
        "Diagnostic",
        "ExecuteRequest",
        "ExecuteResponse",
        "OutputConversionResult",
        "DataStepAst",
        "ParseResult",
        "ParsedStatement",
        "ParserService",
    ]


def test_submit_and_run_stay_module_level_callables_with_same_signatures() -> None:
    assert callable(limulus.submit)
    assert callable(limulus.run)

    submit_signature = inspect.signature(limulus.submit)
    assert list(submit_signature.parameters) == ["code", "backend", "functions", "datasets"]
    assert submit_signature.parameters["backend"].default == "auto"

    run_signature = inspect.signature(limulus.run)
    assert list(run_signature.parameters) == ["code", "backend", "functions", "datasets"]
    assert run_signature.parameters["backend"].default == "auto"


def test_lazy_export_failure_is_transparent_without_eager_import() -> None:
    _run_isolated(
        """
        import sys

        import limulus

        assert "limulus.session" not in sys.modules
        try:
            limulus.Session
        except ImportError:
            pass
        else:
            raise AssertionError("Session resolution should have imported the denied runtime")
        assert "limulus.session" not in sys.modules
        """
    )


def test_where_serializer_is_shared_not_cloned() -> None:
    assert pure_where_to_dict is legacy_where_to_dict


def test_legacy_result_uses_the_existing_diagnostic_type() -> None:
    from limulus.syntax.legacy import ParseResult

    diagnostic = limulus_models.Diagnostic(
        code="PARSE_UNSUPPORTED_STATEMENT",
        severity="error",
        location="statement:1",
        stage="parse",
        message="boom",
    )
    result = ParseResult(diagnostics=(diagnostic,))

    assert result.has_errors is True
    assert type(result.diagnostics[0]) is limulus_models.Diagnostic


# --- public import / constructor / attribute / capture ---------------------


_EAGER_MODEL_NAMES = (
    "SubmitResult",
    "LogEntry",
    "DatasetCatalog",
    "CompatibilityNotice",
    "DataSetRef",
    "Diagnostic",
    "ExecuteRequest",
    "ExecuteResponse",
    "OutputConversionResult",
)

_LAZY_FAMILY_MODULES = {
    "Session": "limulus.session",
    "DataStepExecutor": "limulus.execution",
    "FormatSupportResult": "limulus.execution",
    "RuntimeRequirements": "limulus.execution",
    "DataAdapterError": "limulus.io_adapters",
    "DataFrameAdapterPandas": "limulus.io_adapters",
    "DataInputAdapterArrow": "limulus.io_adapters",
    "DataOutputAdapterArrow": "limulus.io_adapters",
    "InputSpec": "limulus.io_adapters",
    "OutputSpec": "limulus.io_adapters",
    "DataStepAst": "limulus.parser",
    "ParseResult": "limulus.parser",
    "ParsedStatement": "limulus.parser",
    "ParserService": "limulus.parser",
}

_COLD_APPLY_SCRIPT = textwrap.dedent(
    """
    import sys

    import limulus


    def scale(value):
        return value * 10  # explicit registry mode


    def global_double(value):
        return value * 2  # caller-globals mode


    def _run_submit():
        def local_inc(value):
            return value + 1  # caller-locals mode

        import pyarrow as pa

        return limulus.submit(
            "data out; set inp; "
            "scaled = apply('scale', x); "
            "doubled = apply('global_double', x); "
            "incremented = apply('local_inc', x); "
            "output out; run;",
            functions={"scale": scale},
            inp=pa.table({"x": [1, 2]}),
        )


    result = _run_submit()
    assert result.success is True, result.format_log()
    assert result.datasets["out"].to_pylist() == [
        {"x": 1, "scaled": 10, "doubled": 2, "incremented": 2},
        {"x": 2, "scaled": 20, "doubled": 4, "incremented": 3},
    ]

    # One-shot ``run`` in the same cold process: caller-globals capture with
    # the real runtime, no ``functions=`` registry argument at all.
    def global_triple(value):
        return value * 3

    import pyarrow as pa

    run_result = limulus.run(
        "data out2; set inp; tripled = apply('global_triple', x); output out2; run;",
        inp=pa.table({"x": [3]}),
    )
    assert run_result.success is True, run_result.format_log()
    assert run_result.datasets["out2"].to_pylist() == [{"x": 3, "tripled": 9}]

    # The native extension is a runtime concern: it loaded only when the
    # real row runtime ran, never during package import or parse.
    assert "limulus_native" in sys.modules
    """
)


def test_star_import_resolves_every_public_name() -> None:
    namespace: dict[str, object] = {}
    exec("from limulus import *", namespace)  # noqa: S102 - intentional star import

    assert sorted(namespace) == sorted([*limulus.__all__, "__builtins__"])
    for name in limulus.__all__:
        assert namespace[name] is getattr(limulus, name), name


def test_lazy_and_eager_exports_match_their_owning_modules() -> None:
    import importlib

    for name, module_name in _LAZY_FAMILY_MODULES.items():
        assert getattr(limulus, name) is getattr(importlib.import_module(module_name), name), name
    for name in _EAGER_MODEL_NAMES:
        assert getattr(limulus, name) is getattr(limulus_models, name), name
    assert limulus.submit.__module__ == "limulus"
    assert limulus.run.__module__ == "limulus"


def test_cold_package_init_resolves_pure_names_first_and_defers_data_exports() -> None:
    _run_isolated(
        """
        import sys

        import limulus

        assert len(limulus.__all__) == 25

        for name in (
            "SubmitResult", "LogEntry", "DatasetCatalog", "CompatibilityNotice",
            "DataSetRef", "Diagnostic", "ExecuteRequest", "ExecuteResponse",
            "OutputConversionResult", "submit", "run",
        ):
            assert getattr(limulus, name) is not None, name

        import limulus.parser as parser_module

        for name in ("DataStepAst", "ParseResult", "ParsedStatement", "ParserService"):
            assert getattr(limulus, name) is getattr(parser_module, name), name

        # Session and execution are genuinely deferred: resolving them pulls
        # the denied data stack and fails transparently, leaving no residue.
        for name in ("Session", "DataStepExecutor", "FormatSupportResult", "RuntimeRequirements"):
            try:
                getattr(limulus, name)
            except ImportError:
                pass
            else:
                raise AssertionError(f"{name} should require the denied data stack")
            for denied in ("pyarrow", "polars", "limulus_native", "limulus.session"):
                assert denied not in sys.modules, f"{name}: {denied}"

        # io adapters are data-processing exports too: not imported at init,
        # loaded at first use, and still data-library free at module scope.
        assert "limulus.io_adapters" not in sys.modules
        assert limulus.DataAdapterError is not None
        assert "limulus.io_adapters" in sys.modules
        for denied in ("pyarrow", "polars", "limulus_native", "limulus.session"):
            assert denied not in sys.modules, f"io_adapters: {denied}"
        """
    )


def test_cold_subprocess_top_level_submit_and_run_apply_user_functions() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _COLD_APPLY_SCRIPT],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"cold submit failed:\n{result.stdout}\n{result.stderr}"


# --- isolated whole-text parse and legacy view --------------------------------
#
# The pure whole-text parse path (``parse_program`` plus the legacy facade
# projection) runs with ONLY Lark and the standard library: a cold
# subprocess installs the heavy-import gate (Arrow, Polars, native,
# session, execution) and exercises frontend import, real parses, the
# typed projection facts, the legacy DTO/serializer surface, and the
# block splitter display — the independent-usage contract.
# Closed quoted semicolons keep the old parse
# stop (no fabricated success), and the new model exports resolve on the
# pure syntax package.


_DENY_HOOK_72 = textwrap.dedent(
    """
    import sys

    _DENIED = (
        "pyarrow",
        "polars",
        "limulus_native",
        "limulus.session",
        "limulus.execution",
    )


    class _DeniedDependencyGate:
        def find_spec(self, fullname, path=None, target=None):
            for root in _DENIED:
                if fullname == root or fullname.startswith(root + "."):
                    raise ImportError(
                        f"module {fullname!r} is denied by the parser isolation hook"
                    )
            return None


    sys.meta_path.insert(0, _DeniedDependencyGate())
    """
)


def _run_isolated_72(snippet: str) -> None:
    script = _DENY_HOOK_72 + "\n" + textwrap.dedent(snippet)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"isolated parser run failed:\n{result.stdout}\n{result.stderr}"


def test_cold_pure_syntax_import_and_parse_run_without_heavy_imports() -> None:
    """Cold subprocess acceptance: importing ``limulus.syntax`` alone never
    loads Lark; a real ``parse_program`` (Unicode, original tabs, quoted
    comma/paren operands, IF / DO general expressions, multi-block) plus
    the failing multi-WHERE stop run pure, and the same subprocess
    exercises the legacy facade view (alias, classes, serializer) and the
    new pure model exports."""

    _run_isolated_72(
        """
        import sys

        import limulus.syntax

        # 1) the pure syntax package imports without Lark or any data stack.
        for module in ("lark", "pyarrow", "polars", "limulus_native", "limulus.session"):
            assert module not in sys.modules, module

        # 2) real parses: valid multi-block text and the invalid stop.
        from limulus.syntax import (
            ParseDiagnosticDisplay,
            ProgramProjectionFacts,
            parse_program,
        )

        result = parse_program(
            "data out;\\tset in; if name = '日本語(1), ok' then y = 2;"
            " do i = 'a, (1)' to 'z'; w = i; end; run;\\n"
            "data out2; set in; run;"
        )
        assert result.program is not None, result.diagnostics
        assert result.program.source.source_id == "<dsl>"
        assert [
            (span.start, span.end, terminated)
            for span, terminated in result.projection.block_display_spans
        ] == [(0, 93, True), (94, 117, True)]
        assert result.projection.skip_regions == ()
        assert result.projection.diagnostic_displays == ()

        stop = parse_program("data out; set i(where=(a=1)) j(where=(a+b)); run;")
        assert stop.program is None
        assert len(stop.diagnostics) == 1
        assert stop.diagnostics[0].source_text == "data out; set i(where=(a=1)) j(where=(a+b)); run;"
        assert stop.projection.diagnostic_displays == (ParseDiagnosticDisplay(source_text="a+b"),)

        # 3) the legacy facade view in the SAME cold subprocess: alias,
        # re-exported classes, serializer output, and the diagnostic view.
        from limulus.parser import (
            DataStepAst,
            LarkParserFrontend,
            ParseResult,
            ParserService,
            ParsedStatement,
            ast_to_dict,
        )
        from limulus.syntax.legacy import DataStepAst as LegacyDataStepAst

        assert ParserService is LarkParserFrontend
        assert DataStepAst is LegacyDataStepAst

        legacy_stop = ParserService().parse("data out; set i(where=(a=1)) j(where=(a+b)); run;")
        assert legacy_stop.diagnostics[0].source_text == "a+b"

        legacy_ok = ParserService().parse(
            "data out;\\tset in; if name = '日本語(1), ok' then y = 2; run;"
        )
        assert [st.kind for st in legacy_ok.ast.statements] == ["DATA", "SET", "IF", "RUN"]
        assert isinstance(legacy_ok, ParseResult)
        assert all(isinstance(st, ParsedStatement) for st in legacy_ok.ast.statements)
        serialized = ast_to_dict(legacy_ok.ast)
        if_statement = next(item for item in serialized["statements"] if item["kind"] == "IF")
        assert if_statement["if_spec"]["condition"] == "name == '日本語(1), ok'"

        # Closed quoted semicolons keep the OLD parse stop: no fabricated
        # success for a shape the legacy surface refused.
        quoted_semicolon = ParserService().parse("data out; x = 'a; b'; run;")
        assert quoted_semicolon.has_errors is True
        assert quoted_semicolon.ast.statements == ()

        # New pure model exports resolve on the cold syntax package.
        import limulus.syntax as syntax_package

        assert syntax_package.ParseDiagnosticDisplay is ParseDiagnosticDisplay
        assert syntax_package.ProgramProjectionFacts is ProgramProjectionFacts

        # Lark itself is the only third-party module this path loaded.
        assert "lark" in sys.modules
        for module in ("pyarrow", "polars", "limulus_native", "limulus.session"):
            assert module not in sys.modules, module
        """
    )


def test_cold_block_splitter_displays_come_from_the_initial_facts() -> None:
    """The block splitter runs cold and heavy-free: success displays come
    from the typed blocks and a stop displays the SAME initial call's
    region facts — no second region pass and no data stack."""

    _run_isolated_72(
        """
        import sys

        from limulus.block_splitter import DataStepBlockSplitter
        from limulus.syntax import parse_program

        splitter = DataStepBlockSplitter()
        multi = "data a; x=1; run;\\ndata b; y=2; run;"
        assert splitter.split(multi) == ("data a; x=1; run;", "data b; y=2; run;")

        stopped = "data bad;\\n  ???;\\nrun;\\ndata good;\\n  set inp;\\nrun;\\n"
        assert splitter.split(stopped) == (
            "data bad;\\n  ???;\\nrun;",
            "data good;\\n  set inp;\\nrun;",
        )
        typed = parse_program(stopped)
        assert typed.program is None
        assert len(typed.projection.block_display_spans) == 2

        # A region-refusing input keeps the single raw pseudo-block.
        assert splitter.split("* only comment;") == ("* only comment;",)

        assert "lark" in sys.modules
        for module in ("pyarrow", "polars", "limulus_native", "limulus.session"):
            assert module not in sys.modules, module
        """
    )
