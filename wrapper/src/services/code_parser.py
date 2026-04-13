"""代码解析器

集成 tree-sitter 进行代码解析：
- 支持 Python/JavaScript/TypeScript
- 提取函数和类定义
- 支持符号分析
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CodeParser:
    """代码解析器

    使用 tree-sitter 解析代码，提取符号信息。

    Attributes:
        languages: 支持的语言映射
    """

    # 文件扩展名到语言的映射
    EXTENSION_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
    }

    def __init__(self):
        """初始化代码解析器"""
        self._logger = logging.getLogger(__name__)
        self._parsers: Dict[str, any] = {}

        self._init_parsers()

        self._logger.debug("[CodeParser] 初始化完成")

    def _init_parsers(self) -> None:
        """初始化 tree-sitter 解析器"""
        try:
            from tree_sitter import Language, Parser

            # 尝试加载语言（如果已安装）
            self._try_load_language("python", "tree_sitter_python")
            self._try_load_language("javascript", "tree_sitter_javascript")
            self._try_load_language("typescript", "tree_sitter_typescript")

        except ImportError as e:
            self._logger.warning("[CodeParser] tree-sitter 未安装: %s", e)

    def _try_load_language(self, language: str, package: str) -> None:
        """尝试加载语言"""
        try:
            import importlib

            lang_module = importlib.import_module(package)
            self._parsers[language] = lang_module
            self._logger.debug("[CodeParser] 加载语言: %s", language)
        except ImportError:
            self._logger.debug("[CodeParser] 语言未安装: %s", language)

    def get_language(self, file_path: str) -> Optional[str]:
        """根据文件路径获取语言

        Args:
            file_path: 文件路径

        Returns:
            语言名称，如果不支持返回 None
        """
        ext = Path(file_path).suffix.lower()
        return self.EXTENSION_MAP.get(ext)

    def is_supported(self, file_path: str) -> bool:
        """检查文件是否支持

        Args:
            file_path: 文件路径

        Returns:
            是否支持
        """
        language = self.get_language(file_path)
        return language is not None and language in self._parsers

    def parse(self, content: str, language: str) -> Optional[Dict]:
        """解析代码

        Args:
            content: 代码内容
            language: 语言名称

        Returns:
            解析结果，包含 AST 和符号信息
        """
        if language not in self._parsers:
            self._logger.warning("[CodeParser] 不支持的语言: %s", language)
            return None

        try:
            from tree_sitter import Parser

            parser = Parser()
            parser.set_language(self._parsers[language])

            tree = parser.parse(bytes(content, "utf8"))

            symbols = self._extract_symbols(tree, content)

            return {
                "language": language,
                "symbols": symbols,
                "root_node": tree.root_node,
            }

        except Exception as e:
            self._logger.error("[CodeParser] 解析失败: %s", e)
            return None

    def _extract_symbols(self, tree, content: str) -> List[Dict]:
        """提取符号

        从 AST 中提取函数和类定义。

        Args:
            tree: tree-sitter 解析树
            content: 原始代码内容

        Returns:
            符号列表
        """
        symbols = []
        root_node = tree.root_node

        for child in root_node.children:
            symbol = self._process_node(child, content)
            if symbol:
                symbols.append(symbol)

        return symbols

    def _process_node(self, node, content: str) -> Optional[Dict]:
        """处理 AST 节点

        Args:
            node: AST 节点
            content: 原始代码内容

        Returns:
            符号信息，如果不是符号返回 None
        """
        node_type = node.type

        # Python 函数定义
        if node_type == "function_definition":
            return self._extract_function(node, content, "function")

        # Python 类定义
        if node_type == "class_definition":
            return self._extract_class(node, content)

        # JavaScript/TypeScript 函数定义
        if node_type in ("function_declaration", "method_definition"):
            return self._extract_function(node, content, "function")

        # JavaScript/TypeScript 类定义
        if node_type == "class_declaration":
            return self._extract_class(node, content)

        # 箭头函数（JS/TS）
        if node_type == "variable_declaration":
            return self._extract_arrow_function(node, content)

        return None

    def _extract_function(self, node, content: str, kind: str) -> Dict:
        """提取函数信息"""
        name = ""
        for child in node.children:
            if child.type in ("identifier", "property_identifier"):
                name = content[child.start_byte : child.end_byte]
                break

        return {
            "type": kind,
            "name": name,
            "line": node.start_point[0] + 1,
            "column": node.start_point[1],
        }

    def _extract_class(self, node, content: str) -> Dict:
        """提取类信息"""
        name = ""
        for child in node.children:
            if child.type == "identifier":
                name = content[child.start_byte : child.end_byte]
                break

        return {
            "type": "class",
            "name": name,
            "line": node.start_point[0] + 1,
            "column": node.start_point[1],
        }

    def _extract_arrow_function(self, node, content: str) -> Optional[Dict]:
        """提取箭头函数信息"""
        # 简化处理，检查是否是箭头函数赋值
        for child in node.children:
            if child.type == "variable_declarator":
                for subchild in child.children:
                    if subchild.type == "identifier":
                        name = content[subchild.start_byte : subchild.end_byte]
                        return {
                            "type": "function",
                            "name": name,
                            "line": node.start_point[0] + 1,
                            "column": node.start_point[1],
                        }
        return None

    def get_supported_languages(self) -> List[str]:
        """获取支持的语言列表

        Returns:
            语言列表
        """
        return list(self._parsers.keys())
