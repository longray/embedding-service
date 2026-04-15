"""HNSW 索引管理测试

验证 BL-C-4: HNSW 索引管理端点
"""

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from wrapper.src.routers.hnsw import get_hnsw_stats, optimize_hnsw, rebuild_hnsw


class TestHNSWManager:
    """测试 HNSW 索引管理"""

    @pytest.fixture
    def mock_memory_manager(self):
        """模拟 MemoryManager"""
        mm = MagicMock()
        mm.get_memory_stats = AsyncMock(
            return_value={
                "status": "success",
                "index_name": "memory_embedding_hnsw",
                "tenant_id": "default",
            }
        )
        mm.optimize_hnsw = AsyncMock(
            return_value={
                "status": "success",
                "optimization_needed": False,
                "current_params": {"efConstruction": 128, "M": 16},
                "recommended_params": {"efConstruction": 128, "M": 16},
            }
        )
        mm.rebuild_hnsw_index = AsyncMock(
            return_value={
                "status": "success",
                "message": "索引重建完成",
                "params": {"efConstruction": 128, "M": 16},
            }
        )
        return mm

    @pytest.mark.asyncio
    async def test_get_hnsw_stats_success(self, mock_memory_manager):
        """测试获取 HNSW 统计信息成功"""
        with patch("wrapper.src.routers.hnsw.state.memory_manager", mock_memory_manager):
            result = await get_hnsw_stats(tenant_id="default")

            assert result["status"] == "success"
            assert result["index_name"] == "memory_embedding_hnsw"
            mock_memory_manager.get_memory_stats.assert_called_once_with("default")

    @pytest.mark.asyncio
    async def test_optimize_hnsw_success(self, mock_memory_manager):
        """测试优化 HNSW 成功"""
        with patch("wrapper.src.routers.hnsw.state.memory_manager", mock_memory_manager):
            result = await optimize_hnsw(tenant_id="default")

            assert result["status"] == "success"
            assert "result" in result
            assert result["result"]["optimization_needed"] is False
            mock_memory_manager.optimize_hnsw.assert_called_once_with("default")

    @pytest.mark.asyncio
    async def test_rebuild_hnsw_success(self, mock_memory_manager):
        """测试重建 HNSW 成功"""
        with patch("wrapper.src.routers.hnsw.state.memory_manager", mock_memory_manager):
            result = await rebuild_hnsw(tenant_id="default", force=False)

            assert result["status"] == "success"
            assert "result" in result
            assert result["result"]["message"] == "索引重建完成"
            mock_memory_manager.rebuild_hnsw_index.assert_called_once_with("default", False)

    @pytest.mark.asyncio
    async def test_rebuild_hnsw_with_force(self, mock_memory_manager):
        """测试强制重建 HNSW"""
        with patch("wrapper.src.routers.hnsw.state.memory_manager", mock_memory_manager):
            result = await rebuild_hnsw(tenant_id="default", force=True)

            assert result["status"] == "success"
            mock_memory_manager.rebuild_hnsw_index.assert_called_once_with("default", True)

    @pytest.mark.asyncio
    async def test_memory_manager_not_initialized(self):
        """测试 MemoryManager 未初始化"""
        with patch("wrapper.src.routers.hnsw.state.memory_manager", None):
            from fastapi import HTTPException

            with pytest.raises(HTTPException) as exc_info:
                await get_hnsw_stats()

            assert exc_info.value.status_code == 503
            assert "MemoryManager未初始化" in exc_info.value.detail


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
