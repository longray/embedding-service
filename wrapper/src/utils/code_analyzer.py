"""代码分析器模块

实现代码解析、注释提取、AST分析等代码开发相关功能。
基于Tree-sitter或其他解析器来深入理解代码结构。
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import tree_sitter  # type: ignore
    import tree_sitter_languages  # type: ignore

    HAS_TREE_SITTER = True
except ImportError:
    HAS_TREE_SITTER = False
    logger.warning("tree-sitter-languages not available, code analysis features will be limited")


@dataclass
class CodeAnalysisResult:
    """代码分析结果"""

    content: str
    language: str
    functions: List[Dict[str, Any]]
    classes: List[Dict[str, Any]]
    imports: List[str]
    exports: List[str]
    comments: List[Dict[str, str]]  # 包含注释位置和内容
    docstrings: List[Dict[str, str]]  # 包含文档字符串
    dependencies: List[str]
    complexity_metrics: Dict[str, int]

    # B-028: 持久化新增字段
    analyzed_at: str = ""  # ISO 8601 时间戳
    analyzer_version: str = "1.0.0"

    def to_metadata_dict(self) -> dict[str, Any]:
        """将分析结果转换为 metadata.code_analysis 字典（不含原始 content）"""
        return {
            "language": self.language,
            "functions": self.functions,
            "classes": self.classes,
            "imports": self.imports,
            "exports": self.exports,
            "comments_count": len(self.comments),
            "docstrings_count": len(self.docstrings),
            "dependencies": self.dependencies,
            "complexity": self.complexity_metrics,
            "analyzed_at": self.analyzed_at,
            "analyzer_version": self.analyzer_version,
        }


class CodeAnalyzer:
    """代码分析器，用于提取代码注释、解析代码结构等"""

    def __init__(self):
        self.parser_cache = {}

    async def analyze_code(self, content: str, language: str = "python") -> CodeAnalysisResult:
        """分析代码内容并返回结构化信息"""
        if HAS_TREE_SITTER:
            return await self._analyze_with_tree_sitter(content, language)
        else:
            return await self._analyze_with_regex(content, language)

    async def _analyze_with_tree_sitter(self, content: str, language: str) -> CodeAnalysisResult:
        """使用Tree-sitter分析代码"""
        try:
            parser = self._get_parser(language)
            if not parser:
                return await self._analyze_with_regex(content, language)

            tree = parser.parse(bytes(content, "utf8"))

            # 提取各种代码元素
            functions = self._extract_functions(tree.root_node, content)
            classes = self._extract_classes(tree.root_node, content)
            imports = self._extract_imports(tree.root_node, content)
            exports = self._extract_exports(tree.root_node, content)
            comments = self._extract_comments(tree.root_node, content)
            docstrings = self._extract_docstrings(tree.root_node, content)

            # 分析依赖和复杂度
            dependencies = await self._extract_dependencies(content)
            complexity_metrics = self._calculate_complexity(tree.root_node)

            from datetime import datetime, timezone

            return CodeAnalysisResult(
                content=content,
                language=language,
                functions=functions,
                classes=classes,
                imports=imports,
                exports=exports,
                comments=comments,
                docstrings=docstrings,
                dependencies=dependencies,
                complexity_metrics=complexity_metrics,
                analyzed_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as e:
            logger.warning(f"Tree-sitter analysis failed: {e}, falling back to regex")
            return await self._analyze_with_regex(content, language)

    def _get_parser(self, language: str):
        """获取指定语言的解析器"""
        if language in self.parser_cache:
            return self.parser_cache[language]

        try:
            if language == "python":
                parser = tree_sitter_languages.get_parser("python")
            elif language == "javascript":
                parser = tree_sitter_languages.get_parser("javascript")
            elif language == "typescript":
                parser = tree_sitter_languages.get_parser("typescript")
            elif language == "java":
                parser = tree_sitter_languages.get_parser("java")
            elif language == "go":
                parser = tree_sitter_languages.get_parser("go")
            elif language == "rust":
                parser = tree_sitter_languages.get_parser("rust")
            elif language == "c":
                parser = tree_sitter_languages.get_parser("c")
            elif language == "cpp":
                parser = tree_sitter_languages.get_parser("cpp")
            elif language == "html":
                parser = tree_sitter_languages.get_parser("html")
            elif language == "css":
                parser = tree_sitter_languages.get_parser("css")
            elif language == "sql":
                parser = tree_sitter_languages.get_parser("sql")
            else:
                # 尝试泛型解析
                parser = tree_sitter_languages.get_parser(language)

            self.parser_cache[language] = parser
            return parser
        except Exception:
            logger.warning(f"Unsupported language: {language}")
            return None

    def _extract_functions(self, node, content: str) -> List[Dict[str, Any]]:
        """提取函数定义"""
        functions = []

        def traverse(node):
            if node.type in ["function_definition", "method_definition"]:
                func_name = None
                # 找到函数名
                for child in node.children:
                    if child.type in ["identifier", "function_name"]:
                        func_name = content[child.start_byte : child.end_byte]
                        break

                # 提取参数
                params = []
                body_start = None
                for child in node.children:
                    if child.type == "parameters":
                        param_nodes = [c for c in child.children if c.type == "identifier"]
                        params = [content[p.start_byte : p.end_byte] for p in param_nodes]
                    elif child.type in ["block", "function_body"]:
                        body_start = child.start_point[0]

                functions.append(
                    {
                        "name": func_name,
                        "start_line": node.start_point[0],
                        "end_line": node.end_point[0],
                        "parameters": params,
                        "body_start_line": body_start,
                    }
                )

            for child in node.children:
                traverse(child)

        traverse(node)
        return functions

    def _extract_classes(self, node, content: str) -> List[Dict[str, Any]]:
        """提取类定义"""
        classes = []

        def traverse(node):
            if node.type == "class_declaration":
                class_name = None
                for child in node.children:
                    if child.type == "type_identifier":
                        class_name = content[child.start_byte : child.end_byte]
                        break

                classes.append({"name": class_name, "start_line": node.start_point[0], "end_line": node.end_point[0]})

            for child in node.children:
                traverse(child)

        traverse(node)
        return classes

    def _extract_imports(self, node, content: str) -> List[str]:
        """提取导入语句"""
        imports = []

        def traverse(node):
            if node.type in ["import_statement", "import_from_statement"]:
                import_text = content[node.start_byte : node.end_byte]
                imports.append(import_text)

            for child in node.children:
                traverse(child)

        traverse(node)
        return imports

    def _extract_exports(self, node, content: str) -> List[str]:
        """提取导出语句"""
        exports = []

        def traverse(node):
            if node.type == "export_statement":
                export_text = content[node.start_byte : node.end_byte]
                exports.append(export_text)

            for child in node.children:
                traverse(child)

        traverse(node)
        return exports

    def _extract_comments(self, node, content: str) -> List[Dict[str, str]]:
        """提取注释"""
        comments = []

        def traverse(node):
            if node.type in ["comment", "line_comment", "block_comment"]:
                comment_text = content[node.start_byte : node.end_byte]
                comments.append(
                    {
                        "text": comment_text,
                        "start_line": node.start_point[0],
                        "end_line": node.end_point[0],
                        "type": node.type,
                    }
                )

            for child in node.children:
                traverse(child)

        traverse(node)
        return comments

    def _extract_docstrings(self, node, content: str) -> List[Dict[str, str]]:
        """提取文档字符串"""
        docstrings = []

        def traverse(node):
            if node.type in ["expression_statement"]:
                for child in node.children:
                    if child.type in ["string", "string_literal"]:
                        text = content[child.start_byte : child.end_byte]
                        # 检查是否看起来像文档字符串（通常在函数或类开始处）
                        if len(text) > 2 and ("'''" in text or '"""' in text):
                            docstrings.append(
                                {"text": text, "start_line": child.start_point[0], "end_line": child.end_point[0]}
                            )

            for child in node.children:
                traverse(child)

        traverse(node)
        return docstrings

    def _calculate_complexity(self, node) -> Dict[str, int]:
        """计算代码复杂度指标"""
        metrics = {
            "lines_of_code": 0,
            "function_count": 0,
            "class_count": 0,
            "nesting_depth": 0,
            "cyclomatic_complexity": 1,  # 从1开始，每个if/for/while都增加
        }

        def traverse(node, depth=0):
            metrics["nesting_depth"] = max(metrics["nesting_depth"], depth)

            # 计算行数
            if hasattr(node, "start_point") and hasattr(node, "end_point"):
                metrics["lines_of_code"] += node.end_point[0] - node.start_point[0] + 1

            # 计算复杂度节点
            if node.type in [
                "if_statement",
                "for_statement",
                "while_statement",
                "do_statement",
                "case_statement",
                "catch_clause",
            ]:
                metrics["cyclomatic_complexity"] += 1
            elif node.type in ["function_definition", "method_definition"]:
                metrics["function_count"] += 1
            elif node.type == "class_declaration":
                metrics["class_count"] += 1

            for child in node.children:
                traverse(child, depth + 1 if node.type in ["block", "compound_statement"] else depth)

        traverse(node)
        return metrics

    async def _extract_dependencies(self, content: str) -> List[str]:
        """提取代码依赖项"""
        deps = set()

        # 不同语言的导入模式
        import_patterns = {
            "python": [
                r"^import\s+(\w+)",
                r"^from\s+(\w+)\s+import",
                r"^from\s+(\w+\.\w+)\s+import",
            ],
            "javascript": [
                r'import\s+.*\s+from\s+[\'"`](.*?)[\'"`]',
                r'require\s*\(\s*[\'"`](.*?)[\'"`]\s*\)',
            ],
            "java": [r"import\s+(?:static\s+)?(.+?\.[\w\*]+);"],
            "go": [r'import\s+[\'"`](.*?)[\'"`]', r'import\s+\(\s*[\'"`](.*?)[\'"`]'],
        }

        # 通用的依赖查找
        import_pattern = (
            r"(?:^|\s)(?:require|import|from)\s+"
            r"(?:\w+\s+)?"
            r'[\'"`]([^\'"`]+)[\'"`]'
        )

        import_lines = [
            line.strip()
            for line in content.split("\n")
            if any(kw in line.lower() for kw in ["import", "require", "from"])
        ]

        for line in import_lines:
            # 查找常见的包名
            if "import" in line or "require" in line:
                import_parts = (
                    line.replace("import", "").replace("require", "").replace("from", "").replace('"', "'").split("'")
                )
                for part in import_parts:
                    if part.strip() and "." not in part and len(part.strip()) > 1:
                        pkg_name = part.strip().split()[0].split(".")[0]
                        if pkg_name and not pkg_name.startswith("//") and not pkg_name.startswith("#"):
                            deps.add(pkg_name)

        return list(deps)

    async def _analyze_with_regex(self, content: str, language: str) -> CodeAnalysisResult:
        """使用正则表达式进行基本分析（备用方法）"""
        # 基本统计
        lines = content.split("\n")
        imports = []
        functions = []
        classes = []
        comments = []
        docstrings = []

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            # 检测导入语句
            if any(lang_imp in line for lang_imp in ["import ", "from ", "require"]):
                imports.append(line_stripped)

            # 检测函数定义
            if any(func_kw in line for func_kw in ["def ", "function ", "func ", "async def"]):
                func_name = ""
                if "def " in line:
                    func_name = line.split("def ")[1].split("(")[0].strip()
                elif "function " in line:
                    func_name = line.split("function ")[1].split("(")[0].strip()
                elif "func " in line and language == "go":
                    func_name = line.split("func ")[1].split("(")[0].strip()

                functions.append({"name": func_name, "start_line": i, "parameters": []})

            # 检测类定义
            if any(class_kw in line for class_kw in ["class ", "interface ", "struct "]):
                class_name = ""
                if "class " in line:
                    class_name = line.split("class ")[1].split(":")[0].strip()
                elif "interface " in line:
                    class_name = line.split("interface ")[1].split("{")[0].strip()
                elif "struct " in line and language == "go":
                    class_name = line.split("struct ")[1].split("{")[0].strip()

                classes.append({"name": class_name, "start_line": i})

            # 检测注释
            if line_stripped.startswith("#") or line_stripped.startswith("//") or line_stripped.startswith("/*"):
                comments.append({"text": line_stripped, "start_line": i, "type": "line_comment"})

        # 基本复杂度指标
        complexity = {
            "lines_of_code": len(lines),
            "function_count": len(functions),
            "class_count": len(classes),
            "nesting_depth": min(3, len([l for l in lines if l.startswith("    ")]) // 10),  # 简单估计
            "cyclomatic_complexity": len(
                [l for l in lines if any(kw in l for kw in ["if ", "for ", "while ", "elif "])]
            ),
        }

        return CodeAnalysisResult(
            content=content,
            language=language,
            functions=functions,
            classes=classes,
            imports=imports,
            exports=[],
            comments=comments,
            docstrings=docstrings,
            dependencies=await self._extract_dependencies(content),
            complexity_metrics=complexity,
            analyzed_at=datetime.now(timezone.utc).isoformat(),
            analyzer_version="1.0.0",
        )

    def extract_comment_and_docstring_content(self, analysis_result: CodeAnalysisResult) -> str:
        """提取注释和文档字符串内容，用于增强搜索"""
        comment_texts = [c["text"] for c in analysis_result.comments]
        docstring_texts = [d["text"] for d in analysis_result.docstrings]

        all_comment_content = "\n".join(comment_texts + docstring_texts)
        return all_comment_content


# 全局代码分析器实例
code_analyzer = CodeAnalyzer()


async def get_code_analysis(content: str, language: str = "python") -> CodeAnalysisResult:
    """便捷函数：获取代码分析结果"""
    return await code_analyzer.analyze_code(content, language)
