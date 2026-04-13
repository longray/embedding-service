"""Tests for CodeParser"""

import pytest
from unittest.mock import MagicMock, patch

from wrapper.src.services.code_parser import CodeParser


class TestCodeParserInitialization:
    """Test CodeParser initialization"""

    def test_initialization(self):
        """Test basic initialization"""
        parser = CodeParser()
        assert parser._logger is not None
        assert isinstance(parser._parsers, dict)

    def test_extension_map(self):
        """Test extension map contains expected languages"""
        assert ".py" in CodeParser.EXTENSION_MAP
        assert ".js" in CodeParser.EXTENSION_MAP
        assert ".ts" in CodeParser.EXTENSION_MAP
        assert ".jsx" in CodeParser.EXTENSION_MAP
        assert ".tsx" in CodeParser.EXTENSION_MAP

        assert CodeParser.EXTENSION_MAP[".py"] == "python"
        assert CodeParser.EXTENSION_MAP[".js"] == "javascript"
        assert CodeParser.EXTENSION_MAP[".ts"] == "typescript"


class TestCodeParserLanguageDetection:
    """Test CodeParser language detection"""

    def test_get_language_python(self):
        """Test detecting Python files"""
        parser = CodeParser()
        assert parser.get_language("test.py") == "python"
        assert parser.get_language("/path/to/file.py") == "python"
        assert parser.get_language("file.PY") == "python"

    def test_get_language_javascript(self):
        """Test detecting JavaScript files"""
        parser = CodeParser()
        assert parser.get_language("test.js") == "javascript"
        assert parser.get_language("/path/to/file.js") == "javascript"

    def test_get_language_typescript(self):
        """Test detecting TypeScript files"""
        parser = CodeParser()
        assert parser.get_language("test.ts") == "typescript"
        assert parser.get_language("test.tsx") == "typescript"

    def test_get_language_unsupported(self):
        """Test detecting unsupported files"""
        parser = CodeParser()
        assert parser.get_language("test.java") is None
        assert parser.get_language("test.cpp") is None
        assert parser.get_language("test.txt") is None
        assert parser.get_language("test") is None


class TestCodeParserSupport:
    """Test CodeParser support checking"""

    def test_is_supported_with_loaded_parser(self):
        """Test is_supported when parser is loaded"""
        parser = CodeParser()
        # Mock that python parser is loaded
        parser._parsers["python"] = MagicMock()
        assert parser.is_supported("test.py") is True

    def test_is_supported_without_loaded_parser(self):
        """Test is_supported when parser is not loaded"""
        parser = CodeParser()
        # Ensure python parser is not loaded
        parser._parsers = {}
        assert parser.is_supported("test.py") is False

    def test_is_supported_unsupported_extension(self):
        """Test is_supported for unsupported extension"""
        parser = CodeParser()
        assert parser.is_supported("test.java") is False


class TestCodeParserParse:
    """Test CodeParser parse functionality"""

    def test_parse_without_parser(self):
        """Test parse when parser is not available"""
        parser = CodeParser()
        parser._parsers = {}  # No parsers loaded
        result = parser.parse("def test(): pass", "python")
        assert result is None

    def test_parse_unsupported_language(self):
        """Test parse with unsupported language"""
        parser = CodeParser()
        result = parser.parse("content", "java")
        assert result is None


class TestCodeParserGetSupportedLanguages:
    """Test CodeParser get_supported_languages"""

    def test_get_supported_languages_empty(self):
        """Test getting supported languages when none loaded"""
        parser = CodeParser()
        parser._parsers = {}
        languages = parser.get_supported_languages()
        assert languages == []

    def test_get_supported_languages_with_parsers(self):
        """Test getting supported languages with loaded parsers"""
        parser = CodeParser()
        parser._parsers = {"python": MagicMock(), "javascript": MagicMock()}
        languages = parser.get_supported_languages()
        assert "python" in languages
        assert "javascript" in languages
        assert len(languages) == 2


class TestCodeParserExtractFunction:
    """Test CodeParser _extract_function"""

    def test_extract_function_with_name(self):
        """Test extracting function with name"""
        parser = CodeParser()

        # Create a mock node
        mock_node = MagicMock()
        mock_node.start_point = (0, 0)
        mock_identifier = MagicMock(type="identifier", start_byte=4, end_byte=17)
        mock_node.children = [mock_identifier]

        content = "def test_function(): pass"
        result = parser._extract_function(mock_node, content, "function")

        assert result["type"] == "function"
        # The name is extracted from content[start_byte:end_byte]
        assert result["name"] == content[4:17]  # "test_function"
        assert result["line"] == 1
        assert result["column"] == 0


class TestCodeParserExtractClass:
    """Test CodeParser _extract_class"""

    def test_extract_class_with_name(self):
        """Test extracting class with name"""
        parser = CodeParser()

        # Create a mock node
        mock_node = MagicMock()
        mock_node.start_point = (0, 0)
        mock_node.children = [
            MagicMock(type="identifier", start_byte=6, end_byte=15),
        ]

        content = "class TestClass: pass"
        result = parser._extract_class(mock_node, content)

        assert result["type"] == "class"
        assert result["name"] == "TestClass"
        assert result["line"] == 1
        assert result["column"] == 0
