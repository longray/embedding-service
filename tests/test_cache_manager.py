"""缓存管理测试

验证 BL-C-5: 缓存管理端点
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from wrapper.src.routers.cache import get_cache_stats, clear_cache, warmup_cache


class TestCacheManager:
    """测试缓存管理"""

    @pytest.fixture
    def mock_memory_manager(self):
        """模拟 MemoryManager"""
        mm = MagicMock()
        mm.get_cache_stats = AsyncMock(
            return_value={
                "status": "success",
                "stats": {
                    "cache_enabled": True,
                    "vector_cache_status": "active",
                    "keyword_cache_status": "active",
                },
            }
        )
        mm.clear_embedding_cache = AsyncMock(
            return_value={
                "status": "success",
                "cleared_count": 2,
                "message": "缓存已清除",
            }
        )
        mm.warmup_embedding_cache = AsyncMock(
            return_value={
                "status": "success",
                "warmed_count": 50,
                "message": "已预热 50 个记忆的嵌入向量",
            }
        )
        return mm

    @pytest.mark.asyncio
    async def test_get_cache_stats_success(self, mock_memory_manager):
        """测试获取缓存统计成功"""
        with patch("wrapper.src.routers.cache.state.memory_manager", mock_memory_manager):
            result = await get_cache_stats()

            assert result["status"] == "success"
            assert "stats" in result
            assert result["stats"]["cache_enabled"] is True
            mock_memory_manager.get_cache_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_cache_success(self, mock_memory_manager):
        """测试清除缓存成功"""
        with patch("wrapper.src.routers.cache.state.memory_manager", mock_memory_manager):
            result = await clear_cache()

            assert result["status"] == "success"
            assert "result" in result
            assert result["result"]["cleared_count"] == 2
            mock_memory_manager.clear_embedding_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_warmup_cache_success(self, mock_memory_manager):
        """测试缓存预热成功"""
        with patch("wrapper.src.routers.cache.state.memory_manager", mock_memory_manager):
            result = await warmup_cache(tenant_id="default", limit=100)

            assert result["status"] == "success"
            assert "result" in result
            assert result["result"]["warmed_count"] == 50
            mock_memory_manager.warmup_embedding_cache.assert_called_once_with("default", 100)

    @pytest.mark.asyncio
    async def test_warmup_cache_custom_params(self, mock_memory_manager):
        """测试缓存预热（自定义参数）"""
        with patch("wrapper.src.routers.cache.state.memory_manager", mock_memory_manager):
            result = await warmup_cache(tenant_id="test", limit=50)

            assert result["status"] == "success"
            mock_memory_manager.warmup_embedding_cache.assert_called_once_with("test", 50)

    @pytest.mark.asyncio
    async def test_memory_manager_not_initialized(self):
        """测试 MemoryManager 未初始化"""
        with patch("wrapper.src.routers.cache.state.memory_manager", None):
            from fastapi import HTTPException

            with pytest.raises(HTTPException) as exc_info:
                await get_cache_stats()

            assert exc_info.value.status_code == 503
            assert "MemoryManager未初始化" in exc_info.value.detail


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
