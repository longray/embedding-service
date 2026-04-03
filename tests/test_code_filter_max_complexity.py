"""测试 code_filter max_complexity 支持

验证 BL-CA-05: code_filter 添加 max_complexity 支持
"""

import pytest

pytestmark = pytest.mark.unit
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestCodeFilterMaxComplexity:
    """测试 code_filter max_complexity 参数"""

    @pytest.fixture
    def mock_request(self):
        """创建 mock request 对象"""

        class MockRequest:
            def __init__(self, code_filter=None):
                self.query = "test"
                self.mode = "hybrid"
                self.limit = 10
                self.threshold = 0.7
                self.level = 2
                self.tenant_id = "default"
                self.code_filter = code_filter

        return MockRequest

    @pytest.mark.asyncio
    async def test_max_complexity_filter_generation(self, mock_request):
        """测试 max_complexity 生成正确的过滤条件"""
        from wrapper.src.routers.search import search_memories

        # Mock state.memory_manager
        mock_mm = AsyncMock()
        mock_mm.search_memories.return_value = {"results": [], "total": 0}

        with patch("wrapper.src.routers.search.state") as mock_state:
            mock_state.memory_manager = mock_mm

            # 测试只有 max_complexity
            request = mock_request(code_filter={"max_complexity": 30})
            await search_memories(request)

            call_kwargs = mock_mm.search_memories.call_args[1]
            assert call_kwargs["filters"] == "code_complexity <= 30"

    @pytest.mark.asyncio
    async def test_min_max_complexity_combined(self, mock_request):
        """测试 min_complexity 和 max_complexity 组合"""
        from wrapper.src.routers.search import search_memories

        mock_mm = AsyncMock()
        mock_mm.search_memories.return_value = {"results": [], "total": 0}

        with patch("wrapper.src.routers.search.state") as mock_state:
            mock_state.memory_manager = mock_mm

            request = mock_request(code_filter={"min_complexity": 5, "max_complexity": 30})
            await search_memories(request)

            call_kwargs = mock_mm.search_memories.call_args[1]
            assert "code_complexity >= 5" in call_kwargs["filters"]
            assert "code_complexity <= 30" in call_kwargs["filters"]
            assert " AND " in call_kwargs["filters"]

    @pytest.mark.asyncio
    async def test_all_code_filter_params_combined(self, mock_request):
        """测试所有 code_filter 参数组合"""
        from wrapper.src.routers.search import search_memories

        mock_mm = AsyncMock()
        mock_mm.search_memories.return_value = {"results": [], "total": 0}

        with patch("wrapper.src.routers.search.state") as mock_state:
            mock_state.memory_manager = mock_mm

            request = mock_request(code_filter={"language": "python", "min_complexity": 5, "max_complexity": 30})
            await search_memories(request)

            call_kwargs = mock_mm.search_memories.call_args[1]
            filters = call_kwargs["filters"]
            assert 'code_language = "python"' in filters
            assert "code_complexity >= 5" in filters
            assert "code_complexity <= 30" in filters
            assert filters.count(" AND ") == 2  # 三个条件，两个 AND

    @pytest.mark.asyncio
    async def test_backward_compatibility_without_max_complexity(self, mock_request):
        """测试向后兼容：不含 max_complexity 时行为不变"""
        from wrapper.src.routers.search import search_memories

        mock_mm = AsyncMock()
        mock_mm.search_memories.return_value = {"results": [], "total": 0}

        with patch("wrapper.src.routers.search.state") as mock_state:
            mock_state.memory_manager = mock_mm

            # 只有 language 和 min_complexity（原有功能）
            request = mock_request(code_filter={"language": "typescript", "min_complexity": 10})
            await search_memories(request)

            call_kwargs = mock_mm.search_memories.call_args[1]
            filters = call_kwargs["filters"]
            assert 'code_language = "typescript"' in filters
            assert "code_complexity >= 10" in filters
            assert "code_complexity <=" not in filters  # 不含 max_complexity
