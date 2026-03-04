"""
记忆管理API测试

注意：这些测试需要以下服务运行：
1. SurrealDB (ws://localhost:8000)
2. Embedding服务 (http://localhost:18000)
"""

import pytest
from httpx import AsyncClient

# 测试配置
BASE_URL = "http://localhost:3001"


@pytest.mark.asyncio
async def test_health_check_with_surrealdb():
    """测试健康检查端点（包含SurrealDB状态）"""
    async with AsyncClient(base_url=BASE_URL) as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "surrealdb" in data


@pytest.mark.asyncio
async def test_upload_memories():
    """测试批量上传记忆"""
    async with AsyncClient(base_url=BASE_URL) as client:
        payload = {
            "memories": [
                {
                    "content": "test memory",
                    "metadata": {"source": "test"},
                    "entities": [{"name": "test_entity", "type": "test"}],
                }
            ]
        }
        response = await client.post("/api/v1/memories", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "success" in data


@pytest.mark.asyncio
async def test_search_memories_vector():
    """测试向量搜索"""
    async with AsyncClient(base_url=BASE_URL) as client:
        payload = {"query": "test", "mode": "vector", "limit": 10}
        response = await client.post("/api/v1/memories/search", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data


@pytest.mark.asyncio
async def test_search_memories_hybrid():
    """测试混合搜索"""
    async with AsyncClient(base_url=BASE_URL) as client:
        payload = {"query": "test", "mode": "hybrid", "limit": 10}
        response = await client.post("/api/v1/memories/search", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
