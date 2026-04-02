"""代码分析功能单元测试

覆盖 BL-CA-01 (CodeAnalysisResult dataclass) 和 BL-CA-03 (build_code_symbols)。

运行方式：
    uv run pytest tests/test_code_analysis.py -v
"""

import pytest

from wrapper.src.utils.code_analyzer import (
    CodeAnalysisResult,
    build_code_symbols,
)


# ==================== BL-CA-01: CodeAnalysisResult dataclass ====================


class TestCodeAnalysisResult:
    def test_default_fields(self):
        result = CodeAnalysisResult(
            content="x = 1",
            language="python",
            functions=[],
            classes=[],
            imports=[],
            exports=[],
            comments=[],
            docstrings=[],
            dependencies=[],
            complexity_metrics={"lines_of_code": 1},
        )
        assert result.analyzer == "tree-sitter"
        assert result.interfaces == []
        assert result.errors == []
        assert result.warnings == []

    def test_analyzer_regex(self):
        result = CodeAnalysisResult(
            content="",
            language="python",
            functions=[],
            classes=[],
            imports=[],
            exports=[],
            comments=[],
            docstrings=[],
            dependencies=[],
            complexity_metrics={},
            analyzer="regex",
        )
        assert result.analyzer == "regex"

    def test_to_metadata_dict_includes_new_fields(self):
        result = CodeAnalysisResult(
            content="def foo(): pass",
            language="python",
            functions=[{"name": "foo", "start_line": 0}],
            classes=[],
            imports=[],
            exports=[],
            comments=[],
            docstrings=[],
            dependencies=[],
            complexity_metrics={"cyclomatic_complexity": 1},
            analyzed_at="2026-03-31T00:00:00Z",
            analyzer="tree-sitter",
            interfaces=[{"name": "IBar"}],
            errors=[{"line": "5", "msg": "syntax error"}],
            warnings=[],
        )
        d = result.to_metadata_dict()
        assert d["analyzer"] == "tree-sitter"
        assert d["interfaces"] == [{"name": "IBar"}]
        assert d["errors"] == [{"line": "5", "msg": "syntax error"}]
        assert d["warnings"] == []
        assert "content" not in d

    def test_to_metadata_dict_backward_compatible(self):
        result = CodeAnalysisResult(
            content="pass",
            language="python",
            functions=[],
            classes=[],
            imports=[],
            exports=[],
            comments=[],
            docstrings=[],
            dependencies=[],
            complexity_metrics={},
        )
        d = result.to_metadata_dict()
        assert "language" in d
        assert "functions" in d
        assert "complexity" in d
        assert "analyzer_version" in d


# ==================== BL-CA-03: build_code_symbols ====================


class TestBuildCodeSymbols:
    def test_empty_input(self):
        assert build_code_symbols({}) == ""

    def test_functions_only(self):
        code_analysis = {
            "functions": [
                {"name": "foo", "start_line": 1},
                {"name": "bar", "start_line": 5},
            ]
        }
        assert build_code_symbols(code_analysis) == "foo bar"

    def test_classes_only(self):
        code_analysis = {
            "classes": [
                {"name": "MyClass", "start_line": 0},
            ]
        }
        assert build_code_symbols(code_analysis) == "MyClass"

    def test_mixed_symbols(self):
        code_analysis = {
            "functions": [{"name": "foo"}],
            "classes": [{"name": "Bar"}],
            "interfaces": [{"name": "IBaz"}],
            "exports": [{"name": "qux"}],
        }
        assert build_code_symbols(code_analysis) == "foo Bar IBaz qux"

    def test_exports_old_format_str_list(self):
        code_analysis = {
            "exports": ["export_a", "export_b"],
        }
        assert build_code_symbols(code_analysis) == "export_a export_b"

    def test_exports_new_format_dict_list(self):
        code_analysis = {
            "exports": [{"name": "export_a"}, {"name": "export_b"}],
        }
        assert build_code_symbols(code_analysis) == "export_a export_b"

    def test_skips_empty_names(self):
        code_analysis = {
            "functions": [{"name": ""}, {"name": "valid"}, {"name": None}],
        }
        assert build_code_symbols(code_analysis) == "valid"
