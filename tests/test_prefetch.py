"""预取功能端点测试

验证 BL-C-8: 预取功能端点 (相关记忆 + 热门记忆)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from wrapper.src.routers.prefetch import prefetch_related, prefetch_popular


class TestPrefetch:
    """测试预取功能"""

    @pytest.fixture
    def mock_memory_manager(self):
        """模拟 MemoryManager"""
        mm = MagicMock()
        mm.prefetch_related_memories = AsyncMock(
            return_value={
                "status": "success",
                "message": "成功预取 3 个相关记忆",
                "related_memories": [
                    {
                        "id": "memory:2",
                        "content": "Related content 1",
                        "type": "note",
                    },
                    {
                        "id": "memory:3",
                        "content": "Related content 2",
                        "type": "code",
                    },
                    {
                        "id": "memory:4",
                        "content": "Related content 3",
                        "type": "doc",
                    },
                ],
                "total_fetched": 3,
                "depth": 1,
                "memory_id": "memory:1",
                "tenant_id": "default",
            }
        )
        mm.prefetch_popular_queries = AsyncMock(
            return_value={
                "status": "success",
                "message": "成功预取 5 个热门记忆",
                "popular_memories": [
                    {
                        "id": "memory:10",
                        "content": "Popular content 1",
                        "type": "note",
                    },
                    {
                        "id": "memory:11",
                        "content": "Popular content 2",
                        "type": "code",
                    },
                ],
                "total_fetched": 2,
                "tenant_id": "default",
            }
        )
        return mm

    @pytest.mark.asyncio
    async def test_prefetch_related_success(self, mock_memory_manager):
        """测试预取相关记忆成功"""
        with patch("wrapper.src.routers.prefetch.state.memory_manager", mock_memory_manager):
            result = await prefetch_related(
                memory_id="memory:1",
                tenant_id="default",
                depth=1,
                limit=10,
            )

        assert result["status"] == "success"
        assert result["total_fetched"] == 3
        assert len(result["related_memories"]) == 3
        assert result["memory_id"] == "memory:1"
        assert result["tenant_id"] == "default"

        # 验证调用参数
        mock_memory_manager.prefetch_related_memories.assert_called_once_with(
            memory_id="memory:1",
            tenant_id="default",
            depth=1,
            limit=10,
        )

    @pytest.mark.asyncio
    async def test_prefetch_related_empty_result(self, mock_memory_manager):
        """测试没有相关记忆的场景"""
        mock_memory_manager.prefetch_related_memories = AsyncMock(
            return_value={
                "status": "success",
                "message": "没有找到相关记忆",
                "related_memories": [],
                "total_fetched": 0,
                "depth": 1,
                "memory_id": "memory:999",
                "tenant_id": "default",
            }
        )

        with patch("wrapper.src.routers.prefetch.state.memory_manager", mock_memory_manager):
            result = await prefetch_related(memory_id="memory:999")

        assert result["status"] == "success"
        assert result["total_fetched"] == 0
        assert result["related_memories"] == []

    @pytest.mark.asyncio
    async def test_prefetch_related_error(self, mock_memory_manager):
        """测试预取相关记忆失败"""
        mock_memory_manager.prefetch_related_memories = AsyncMock(
            return_value={
                "status": "error",
                "message": "数据库查询失败",
                "related_memories": [],
                "total_fetched": 0,
            }
        )

        with patch("wrapper.src.routers.prefetch.state.memory_manager", mock_memory_manager):
            with pytest.raises(HTTPException) as exc_info:
                await prefetch_related(memory_id="memory:1")

        assert exc_info.value.status_code == 500
        assert "数据库查询失败" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_prefetch_popular_success(self, mock_memory_manager):
        """测试预取热门记忆成功"""
        with patch("wrapper.src.routers.prefetch.state.memory_manager", mock_memory_manager):
            result = await prefetch_popular(
                tenant_id="default",
                top_n=20,
            )

        assert result["status"] == "success"
        assert result["total_fetched"] == 2
        assert len(result["popular_memories"]) == 2
        assert result["tenant_id"] == "default"

        # 验证调用参数
        mock_memory_manager.prefetch_popular_queries.assert_called_once_with(
            tenant_id="default",
            top_n=20,
        )

    @pytest.mark.asyncio
    async def test_prefetch_popular_empty_result(self, mock_memory_manager):
        """测试没有热门记忆的场景"""
        mock_memory_manager.prefetch_popular_queries = AsyncMock(
            return_value={
                "status": "success",
                "message": "没有找到热门记忆",
                "popular_memories": [],
                "total_fetched": 0,
                "tenant_id": "default",
            }
        )

        with patch("wrapper.src.routers.prefetch.state.memory_manager", mock_memory_manager):
            result = await prefetch_popular()

        assert result["status"] == "success"
        assert result["total_fetched"] == 0
        assert result["popular_memories"] == []

    @pytest.mark.asyncio
    async def test_prefetch_popular_error(self, mock_memory_manager):
        """测试预取热门记忆失败"""
        mock_memory_manager.prefetch_popular_queries = AsyncMock(
            return_value={
                "status": "error",
                "message": "数据库查询失败",
                "popular_memories": [],
                "total_fetched": 0,
            }
        )

        with patch("wrapper.src.routers.prefetch.state.memory_manager", mock_memory_manager):
            with pytest.raises(HTTPException) as exc_info:
                await prefetch_popular()

        assert exc_info.value.status_code == 500
        assert "数据库查询失败" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_prefetch_not_initialized(self):
        """测试 MemoryManager 未初始化"""
        with patch("wrapper.src.routers.prefetch.state.memory_manager", None):
            with pytest.raises(HTTPException) as exc_info:
                await prefetch_related(memory_id="memory:1")

        assert exc_info.value.status_code == 503
        assert "MemoryManager未初始化" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_prefetch_popular_not_initialized(self):
        """测试预取热门记忆时 MemoryManager 未初始化"""
        with patch("wrapper.src.routers.prefetch.state.memory_manager", None):
            with pytest.raises(HTTPException) as exc_info:
                await prefetch_popular()

        assert exc_info.value.status_code == 503
        assert "MemoryManager未初始化" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_prefetch_related_exception(self, mock_memory_manager):
        """测试预取相关记忆异常"""
        mock_memory_manager.prefetch_related_memories = AsyncMock(side_effect=Exception("Unexpected error"))

        with patch("wrapper.src.routers.prefetch.state.memory_manager", mock_memory_manager):
            with pytest.raises(HTTPException) as exc_info:
                await prefetch_related(memory_id="memory:1")

        assert exc_info.value.status_code == 500
        assert "预取失败" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_prefetch_popular_exception(self, mock_memory_manager):
        """测试预取热门记忆异常"""
        mock_memory_manager.prefetch_popular_queries = AsyncMock(side_effect=Exception("Unexpected error"))

        with patch("wrapper.src.routers.prefetch.state.memory_manager", mock_memory_manager):
            with pytest.raises(HTTPException) as exc_info:
                await prefetch_popular()

        assert exc_info.value.status_code == 500
        assert "预取失败" in exc_info.value.detail


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
