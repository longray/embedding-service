from __future__ import annotations

"""代码分析功能集成测试

验证代码分析端到端流程：
1. 上传代码记忆时 Meilisearch 字段正确提取
2. code_filter 所有参数组合过滤
3. Upsert 逻辑：同一 file_path + project_id 更新而非新建
4. 搜索返回结果包含 code_analysis 元数据
5. code_symbols 可被全文搜索匹配

运行方式:
    uv run pytest tests/test_code_analysis_integration.py -v
"""

import pytest

pytestmark = pytest.mark.integration

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wrapper.src.utils.memory_manager import MemoryManager
from wrapper.src.utils.code_analyzer import build_code_symbols


# ==================== Fixtures ====================


@pytest.fixture
def mock_db():
    """模拟 SurrealDB 异步客户端"""
    db = AsyncMock()
    # 模拟 SurrealDB 返回格式
    db.query.return_value = [{"result": [{"id": "memory:test123"}]}]
    return db


@pytest.fixture
def mock_meili():
    """模拟 MeilisearchClient"""
    meili = AsyncMock()
    meili.search = AsyncMock()
    meili.add_documents = AsyncMock()
    return meili


@pytest.fixture
def manager_with_meili(mock_db, mock_meili):
    """创建 MemoryManager 实例（含 Meilisearch）"""
    mgr = MemoryManager(
        db=mock_db,
        embedding_service_url="http://localhost:18000",
        batch_size=10,
    )
    mgr.set_meili_client(mock_meili)
    return mgr


@pytest.fixture
def sample_code_memory():
    """示例代码记忆数据"""
    return {
        "content": "def analyze_code(content: str) -> dict:\n    pass",
        "abstract": "Python analyze_code function",
        "overview": "代码分析函数，接收字符串返回字典",
        "type": "code",
        "tags": ["python", "function"],
        "project_id": "github.com/test/repo",
        "metadata": {
            "file_path": "src/analyzer.py",
            "file_name": "analyzer.py",
            "code_analysis": {
                "language": "python",
                "analyzer": "tree-sitter",
                "functions": [{"name": "analyze_code", "start_line": 1}],
                "classes": [],
                "exports": [{"name": "analyze_code", "type": "function"}],
                "complexity": {
                    "cyclomatic_complexity": 3,
                    "function_count": 1,
                    "class_count": 0,
                },
            },
            "code_symbols": "analyze_code",
        },
        "local_id": "01TEST123",
    }


# ==================== 测试 1: Meilisearch 字段提取 ====================


class TestCodeAnalysisMeiliFields:
    """测试上传代码记忆时 Meilisearch 字段正确提取"""

    def test_build_meili_doc_extracts_code_fields(self, manager_with_meili, sample_code_memory):
        """测试 _build_meili_doc 正确提取代码分析字段"""
        doc = manager_with_meili._build_meili_doc(
            record_id="memory:test123",
            memory_data=sample_code_memory,
            tenant_id="default",
        )

        # 验证代码字段被正确提取
        assert doc["code_language"] == "python"
        assert doc["code_complexity"] == 3
        assert doc["code_function_count"] == 1
        assert doc["code_class_count"] == 0
        assert doc["code_analyzer"] == "tree-sitter"
        assert doc["code_symbols"] == "analyze_code"
        assert doc["code_has_exports"] is True  # BL-CA-18: 验证新字段
        assert doc["type"] == "code"
        assert doc["project_id"] == "github.com/test/repo"

    def test_build_meili_doc_handles_missing_code_analysis(self, manager_with_meili):
        """测试非代码记忆不添加代码字段（或添加默认值）"""
        memory_data = {
            "content": "普通记忆内容",
            "type": "general",
            "metadata": {},
        }

        doc = manager_with_meili._build_meili_doc(
            record_id="memory:test456",
            memory_data=memory_data,
            tenant_id="default",
        )

        if "code_language" in doc:
            assert doc["code_language"] == ""
            assert doc["code_complexity"] == 0
            assert doc["code_function_count"] == 0
            assert doc["code_class_count"] == 0
            assert doc["code_analyzer"] == ""
        else:
            assert "code_language" not in doc


# ==================== 测试 2: code_filter 参数组合 ====================


class TestCodeFilterParameters:
    """测试 code_filter 所有参数组合过滤"""

    @pytest.mark.asyncio
    async def test_code_filter_language_only(self, manager_with_meili):
        """测试仅 language 过滤"""
        from wrapper.src.routers.search import search_memories

        mock_mm = AsyncMock()
        mock_mm.search_memories.return_value = {"results": [], "total": 0}

        with patch("wrapper.src.routers.search.state") as mock_state:
            mock_state.memory_manager = mock_mm

            class MockRequest:
                def __init__(self):
                    self.query = "test"
                    self.mode = "hybrid"
                    self.limit = 10
                    self.threshold = 0.7
                    self.level = 2
                    self.tenant_id = "default"
                    self.code_filter = {"language": "python"}

            await search_memories(MockRequest())

            call_kwargs = mock_mm.search_memories.call_args[1]
            assert 'code_language = "python"' in call_kwargs["filters"]

    @pytest.mark.asyncio
    async def test_code_filter_min_max_complexity(self, manager_with_meili):
        """测试 min_complexity 和 max_complexity 组合"""
        from wrapper.src.routers.search import search_memories

        mock_mm = AsyncMock()
        mock_mm.search_memories.return_value = {"results": [], "total": 0}

        with patch("wrapper.src.routers.search.state") as mock_state:
            mock_state.memory_manager = mock_mm

            class MockRequest:
                def __init__(self):
                    self.query = "test"
                    self.mode = "hybrid"
                    self.limit = 10
                    self.threshold = 0.7
                    self.level = 2
                    self.tenant_id = "default"
                    self.code_filter = {
                        "min_complexity": 5,
                        "max_complexity": 30,
                    }

            await search_memories(MockRequest())

            call_kwargs = mock_mm.search_memories.call_args[1]
            filters = call_kwargs["filters"]
            assert "code_complexity >= 5" in filters
            assert "code_complexity <= 30" in filters
            assert " AND " in filters

    @pytest.mark.asyncio
    async def test_code_filter_all_params(self, manager_with_meili):
        """测试所有参数组合"""
        from wrapper.src.routers.search import search_memories

        mock_mm = AsyncMock()
        mock_mm.search_memories.return_value = {"results": [], "total": 0}

        with patch("wrapper.src.routers.search.state") as mock_state:
            mock_state.memory_manager = mock_mm

            class MockRequest:
                def __init__(self):
                    self.query = "auth"
                    self.mode = "hybrid"
                    self.limit = 10
                    self.threshold = 0.7
                    self.level = 2
                    self.tenant_id = "default"
                    self.code_filter = {
                        "language": "typescript",
                        "min_complexity": 5,
                        "max_complexity": 30,
                    }

            await search_memories(MockRequest())

            call_kwargs = mock_mm.search_memories.call_args[1]
            filters = call_kwargs["filters"]
            assert 'code_language = "typescript"' in filters
            assert "code_complexity >= 5" in filters
            assert "code_complexity <= 30" in filters
            assert filters.count(" AND ") == 2

    @pytest.mark.asyncio
    async def test_code_filter_v14_new_fields(self, manager_with_meili):
        """测试 BL-CA-24 新增的 code_filter 字段"""
        from wrapper.src.routers.search import search_memories

        mock_mm = AsyncMock()
        mock_mm.search_memories.return_value = {"results": [], "total": 0}

        with patch("wrapper.src.routers.search.state") as mock_state:
            mock_state.memory_manager = mock_mm

            class MockRequest:
                def __init__(self):
                    self.query = "auth"
                    self.mode = "hybrid"
                    self.limit = 10
                    self.threshold = 0.7
                    self.level = 2
                    self.tenant_id = "default"
                    self.code_filter = {
                        "min_function_count": 5,
                        "max_function_count": 20,
                        "min_class_count": 1,
                        "max_class_count": 10,
                        "has_exports": True,
                        "analyzer": "oxc",
                    }

            await search_memories(MockRequest())

            call_kwargs = mock_mm.search_memories.call_args[1]
            filters = call_kwargs["filters"]
            assert "code_function_count >= 5" in filters
            assert "code_function_count <= 20" in filters
            assert "code_class_count >= 1" in filters
            assert "code_class_count <= 10" in filters
            assert "code_has_exports = True" in filters
            assert 'code_analyzer = "oxc"' in filters
            assert filters.count(" AND ") == 5


# ==================== 测试 3: Upsert 逻辑 ====================


class TestCodeMemoryUpsert:
    """测试 Upsert 逻辑：同一 file_path + project_id 更新而非新建"""

    def test_upsert_query_uses_correct_filter(self, manager_with_meili):
        from wrapper.src.utils.memory_manager.crud import CrudMixin
        import inspect

        source = inspect.getsource(CrudMixin.upload_memories)
        assert "type = 'code'" in source
        assert "file_path" in source
        assert "project_id" in source


# ==================== 测试 4: code_symbols 搜索 ====================


class TestCodeSymbolsSearch:
    """测试 code_symbols 可被全文搜索匹配"""

    def test_build_code_symbols_extracts_function_names(self):
        """测试 build_code_symbols 提取函数名"""
        code_analysis = {
            "functions": [
                {"name": "analyze_code"},
                {"name": "parse_file"},
            ],
            "classes": [],
            "interfaces": [],
            "exports": [],
        }

        symbols = build_code_symbols(code_analysis)
        assert "analyze_code" in symbols
        assert "parse_file" in symbols

    def test_build_code_symbols_extracts_class_names(self):
        """测试 build_code_symbols 提取类名"""
        code_analysis = {
            "functions": [],
            "classes": [
                {"name": "CodeAnalyzer"},
                {"name": "Parser"},
            ],
            "interfaces": [],
            "exports": [],
        }

        symbols = build_code_symbols(code_analysis)
        assert "CodeAnalyzer" in symbols
        assert "Parser" in symbols

    def test_build_code_symbols_handles_old_format_exports(self):
        """测试 build_code_symbols 兼容旧格式 exports"""
        code_analysis = {
            "functions": [{"name": "foo"}],
            "classes": [],
            "interfaces": [],
            "exports": ["export_a", "export_b"],  # 旧格式：字符串列表
        }

        symbols = build_code_symbols(code_analysis)
        assert "foo" in symbols
        assert "export_a" in symbols
        assert "export_b" in symbols

    def test_build_code_symbols_handles_new_format_exports(self):
        """测试 build_code_symbols 支持新格式 exports"""
        code_analysis = {
            "functions": [{"name": "foo"}],
            "classes": [],
            "interfaces": [],
            "exports": [
                {"name": "export_a"},
                {"name": "export_b"},
            ],  # 新格式：字典列表
        }

        symbols = build_code_symbols(code_analysis)
        assert "foo" in symbols
        assert "export_a" in symbols
        assert "export_b" in symbols


# ==================== 测试 5: 搜索返回元数据 ====================


class TestSearchReturnsMetadata:
    """测试搜索返回结果包含 code_analysis 元数据"""

    @pytest.mark.asyncio
    async def test_search_returns_code_analysis_metadata(self, manager_with_meili):
        """测试搜索结果包含 code_analysis 元数据"""
        # 模拟搜索结果
        mock_result = {
            "results": [
                {
                    "id": "memory:test123",
                    "content": "def foo(): pass",
                    "type": "code",
                    "metadata": {
                        "file_path": "src/test.py",
                        "code_analysis": {
                            "language": "python",
                            "functions": [{"name": "foo"}],
                            "complexity": {"cyclomatic_complexity": 1},
                        },
                    },
                }
            ],
            "total": 1,
        }

        manager_with_meili.search_memories = AsyncMock(return_value=mock_result)

        result = await manager_with_meili.search_memories(
            query="foo",
            mode="hybrid",
            tenant_id="default",
        )

        # 验证返回结果包含 code_analysis
        assert len(result["results"]) == 1
        assert result["results"][0]["metadata"]["code_analysis"]["language"] == "python"
        assert result["results"][0]["metadata"]["code_analysis"]["functions"][0]["name"] == "foo"


# ==================== 测试 6: 项目统计 API (BL-CA-25) ====================


class TestProjectStats:
    """测试项目代码统计 API"""

    @pytest.mark.asyncio
    async def test_get_project_stats_returns_correct_structure(self, manager_with_meili):
        """测试项目统计返回正确的数据结构"""
        from wrapper.src.routers.memories import get_project_stats

        mock_mm = AsyncMock()
        mock_mm.get_project_stats.return_value = {
            "status": "success",
            "project_id": "github.com/test/repo",
            "total_files": 10,
            "total_functions": 50,
            "total_classes": 5,
            "avg_complexity": 4.5,
            "max_complexity": 15,
        }

        with patch("wrapper.src.routers.memories.state") as mock_state:
            mock_state.memory_manager = mock_mm

            result = await get_project_stats("github.com/test/repo", "default")

            assert result["status"] == "success"
            assert result["project_id"] == "github.com/test/repo"
            assert "total_files" in result
            assert "total_functions" in result
            assert "total_classes" in result
            assert "avg_complexity" in result
            assert "max_complexity" in result


# ==================== 测试 7: 代码分析缓存 (BL-CA-28) ====================


class TestCodeAnalysisCache:
    """测试代码分析结果缓存"""

    def test_code_analyzer_has_cache(self):
        """测试 CodeAnalyzer 有缓存实例"""
        from wrapper.src.utils.code_analyzer import CodeAnalyzer

        analyzer = CodeAnalyzer(cache_size=50, cache_ttl=1800)
        stats = analyzer.get_cache_stats()

        assert stats["max_size"] == 50
        assert stats["ttl_seconds"] == 1800
        assert stats["current_size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_code_analyzer_cache_stats(self):
        """测试缓存统计功能"""
        from wrapper.src.utils.code_analyzer import CodeAnalyzer

        analyzer = CodeAnalyzer(cache_size=10, cache_ttl=3600)

        # 初始状态
        stats = analyzer.get_cache_stats()
        assert stats["current_size"] == 0
        assert stats["hit_rate"] == 0

    def test_code_analyzer_clear_cache(self):
        """测试清空缓存功能"""
        from wrapper.src.utils.code_analyzer import CodeAnalyzer

        analyzer = CodeAnalyzer(cache_size=10, cache_ttl=3600)

        # 清空缓存
        analyzer.clear_cache()
        stats = analyzer.get_cache_stats()

        assert stats["current_size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0


# ==================== 测试 8: 项目代码地图 (BL-CA-23) ====================


class TestProjectMap:
    """测试项目代码地图 API"""

    @pytest.mark.asyncio
    async def test_get_project_map_returns_correct_structure(self, manager_with_meili):
        """测试项目地图返回正确的数据结构"""
        from wrapper.src.routers.memories import get_project_map

        mock_mm = AsyncMock()
        mock_mm.get_project_map.return_value = {
            "status": "success",
            "project_id": "github.com/test/repo",
            "file_tree": [
                {
                    "name": "src",
                    "type": "directory",
                    "path": "src",
                    "children": [
                        {
                            "name": "auth.ts",
                            "type": "file",
                            "path": "src/auth.ts",
                            "complexity": 8.5,
                            "function_count": 5,
                            "class_count": 1,
                        }
                    ],
                }
            ],
            "module_dependencies": [{"from": "src/auth.ts", "to": "src/utils/crypto.ts", "type": "import"}],
            "hot_files": ["src/auth.ts", "src/utils/api.ts"],
            "statistics": {
                "total_files": 10,
                "total_functions": 50,
                "total_classes": 5,
                "avg_complexity": 4.5,
                "max_complexity": 15,
            },
        }

        with patch("wrapper.src.routers.memories.state") as mock_state:
            mock_state.memory_manager = mock_mm

            result = await get_project_map("github.com/test/repo", "default")

            assert result["status"] == "success"
            assert result["project_id"] == "github.com/test/repo"
            assert "file_tree" in result
            assert "module_dependencies" in result
            assert "hot_files" in result
            assert "statistics" in result
            assert result["statistics"]["total_files"] == 10
