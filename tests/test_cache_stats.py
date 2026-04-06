"""测试缓存统计端点"""

import pytest
from unittest.mock import AsyncMock


class TestCacheStats:
    """测试缓存统计功能"""

    @pytest.mark.asyncio
    async def test_get_cache_stats_success(self):
        """测试成功获取缓存统计"""
        from wrapper.src.utils.memory_manager import MemoryManager

        mock_db = AsyncMock()

        manager = MemoryManager(
            db=mock_db,
            embedding_service_url="http://localhost:18000",
        )

        result = await manager.get_cache_stats()

        assert result["status"] == "success"
        assert "stats" in result
        assert "cache_enabled" in result["stats"]
        assert "cache_ttl_seconds" in result["stats"]

    @pytest.mark.asyncio
    async def test_get_cache_stats_with_cache_disabled(self):
        """测试缓存禁用状态"""
        from wrapper.src.utils.memory_manager import MemoryManager

        mock_db = AsyncMock()

        manager = MemoryManager(
            db=mock_db,
            embedding_service_url="http://localhost:18000",
        )

        # 禁用缓存
        manager._cache_enabled = False
        manager._vector_cache = None
        manager._keyword_cache = None

        result = await manager.get_cache_stats()

        assert result["status"] == "success"
        assert result["stats"]["cache_enabled"] is False
        assert result["stats"]["vector_cache_status"] == "not_initialized"
        assert result["stats"]["keyword_cache_status"] == "not_initialized"
