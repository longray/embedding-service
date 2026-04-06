"""测试 HNSW 统计端点"""

import pytest
from unittest.mock import AsyncMock, patch


class TestHNSWStats:
    """测试 HNSW 统计功能"""

    @pytest.mark.asyncio
    async def test_get_memory_stats_success(self):
        """测试成功获取 HNSW 统计"""
        from wrapper.src.utils.memory_manager import MemoryManager

        # Mock 依赖
        mock_db = AsyncMock()
        mock_db.query = AsyncMock(
            return_value=[
                {
                    "result": [
                        {
                            "name": "memory_embedding_hnsw",
                            "type": "HNSW",
                            "m": 16,
                            "ef_construction": 64,
                        }
                    ]
                }
            ]
        )

        manager = MemoryManager(
            db=mock_db,
            embedding_service_url="http://localhost:18000",
        )

        result = await manager.get_memory_stats(tenant_id="default")

        assert result["status"] == "success"
        assert result["index_name"] == "memory_embedding_hnsw"
        assert result["index_type"] == "HNSW"
        assert result["tenant_id"] == "default"

    @pytest.mark.asyncio
    async def test_get_memory_stats_index_not_found(self):
        """测试 HNSW 索引不存在"""
        from wrapper.src.utils.memory_manager import MemoryManager

        mock_db = AsyncMock()
        mock_db.query = AsyncMock(return_value=[])  # 空结果表示索引不存在

        manager = MemoryManager(
            db=mock_db,
            embedding_service_url="http://localhost:18000",
        )

        result = await manager.get_memory_stats(tenant_id="default")

        assert result["status"] == "not_found"
        assert result["index_name"] == "memory_embedding_hnsw"

    @pytest.mark.asyncio
    async def test_get_memory_stats_error(self):
        """测试获取 HNSW 统计失败"""
        from wrapper.src.utils.memory_manager import MemoryManager

        mock_db = AsyncMock()
        mock_db.query = AsyncMock(side_effect=Exception("DB error"))

        manager = MemoryManager(
            db=mock_db,
            embedding_service_url="http://localhost:18000",
        )

        result = await manager.get_memory_stats(tenant_id="default")

        assert result["status"] == "error"
        assert "DB error" in result["message"]
