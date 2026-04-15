"""记忆聚类端点测试

验证 BL-C-7: 记忆聚类端点 (Leiden 算法)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from wrapper.src.routers.clustering import cluster_memories_leiden


class TestClustering:
    """测试记忆聚类功能"""

    @pytest.fixture
    def mock_memory_manager(self):
        """模拟 MemoryManager"""
        mm = MagicMock()
        mm.cluster_memories_leiden = AsyncMock(
            return_value={
                "status": "success",
                "message": "成功将 10 个记忆聚类为 3 个簇",
                "clusters": [
                    {
                        "cluster_id": 0,
                        "members": ["memory:1", "memory:2", "memory:3"],
                        "size": 3,
                        "representative": "memory:1",
                        "centroid": [0.1, 0.2, 0.3],
                    },
                    {
                        "cluster_id": 1,
                        "members": ["memory:4", "memory:5"],
                        "size": 2,
                        "representative": "memory:4",
                        "centroid": [0.4, 0.5, 0.6],
                    },
                    {
                        "cluster_id": 2,
                        "members": ["memory:6", "memory:7", "memory:8", "memory:9", "memory:10"],
                        "size": 5,
                        "representative": "memory:6",
                        "centroid": [0.7, 0.8, 0.9],
                    },
                ],
                "total_memories": 10,
                "num_clusters": 3,
                "similarity_threshold": 0.75,
                "tenant_id": "default",
            }
        )
        return mm

    @pytest.mark.asyncio
    async def test_cluster_memories_leiden_success(self, mock_memory_manager):
        """测试记忆聚类成功场景"""
        with patch("wrapper.src.routers.clustering.state.memory_manager", mock_memory_manager):
            result = await cluster_memories_leiden(
                tenant_id="default",
                content_threshold=0.75,
                max_clusters=20,
            )

        assert result["status"] == "success"
        assert result["total_memories"] == 10
        assert result["num_clusters"] == 3
        assert len(result["clusters"]) == 3
        assert result["similarity_threshold"] == 0.75
        assert result["tenant_id"] == "default"

        # 验证第一个簇
        cluster_0 = result["clusters"][0]
        assert cluster_0["cluster_id"] == 0
        assert cluster_0["size"] == 3
        assert len(cluster_0["members"]) == 3
        assert "centroid" in cluster_0

        # 验证调用参数
        mock_memory_manager.cluster_memories_leiden.assert_called_once_with(
            tenant_id="default",
            content_threshold=0.75,
            max_clusters=20,
        )

    @pytest.mark.asyncio
    async def test_cluster_memories_leiden_empty_result(self, mock_memory_manager):
        """测试没有记忆可聚类的场景"""
        mock_memory_manager.cluster_memories_leiden = AsyncMock(
            return_value={
                "status": "success",
                "message": "没有找到可聚类的记忆",
                "clusters": [],
                "total_memories": 0,
                "num_clusters": 0,
            }
        )

        with patch("wrapper.src.routers.clustering.state.memory_manager", mock_memory_manager):
            result = await cluster_memories_leiden()

        assert result["status"] == "success"
        assert result["total_memories"] == 0
        assert result["clusters"] == []

    @pytest.mark.asyncio
    async def test_cluster_memories_leiden_error(self, mock_memory_manager):
        """测试聚类失败场景"""
        mock_memory_manager.cluster_memories_leiden = AsyncMock(
            return_value={
                "status": "error",
                "message": "数据库查询失败",
                "clusters": [],
                "total_memories": 0,
                "num_clusters": 0,
            }
        )

        with patch("wrapper.src.routers.clustering.state.memory_manager", mock_memory_manager):
            with pytest.raises(HTTPException) as exc_info:
                await cluster_memories_leiden()

        assert exc_info.value.status_code == 500
        assert "数据库查询失败" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_cluster_memories_leiden_not_initialized(self):
        """测试 MemoryManager 未初始化场景"""
        with patch("wrapper.src.routers.clustering.state.memory_manager", None):
            with pytest.raises(HTTPException) as exc_info:
                await cluster_memories_leiden()

        assert exc_info.value.status_code == 503
        assert "MemoryManager未初始化" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_cluster_memories_leiden_custom_params(self, mock_memory_manager):
        """测试自定义参数"""
        with patch("wrapper.src.routers.clustering.state.memory_manager", mock_memory_manager):
            result = await cluster_memories_leiden(
                tenant_id="test_tenant",
                content_threshold=0.85,
                max_clusters=10,
            )

        assert result["status"] == "success"
        # 验证调用参数
        mock_memory_manager.cluster_memories_leiden.assert_called_once_with(
            tenant_id="test_tenant",
            content_threshold=0.85,
            max_clusters=10,
        )

    @pytest.mark.asyncio
    async def test_cluster_memories_leiden_exception(self, mock_memory_manager):
        """测试异常处理"""
        mock_memory_manager.cluster_memories_leiden = AsyncMock(side_effect=Exception("Unexpected error"))

        with patch("wrapper.src.routers.clustering.state.memory_manager", mock_memory_manager):
            with pytest.raises(HTTPException) as exc_info:
                await cluster_memories_leiden()

        assert exc_info.value.status_code == 500
        assert "聚类分析失败" in exc_info.value.detail


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
