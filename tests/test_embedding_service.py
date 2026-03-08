"""
Embedding服务接口测试
"""

import pytest
import httpx


@pytest.mark.asyncio
class TestEmbeddingService:
    """Embedding服务测试套件"""

    async def test_health_check(self, embedding_client: httpx.AsyncClient):
        """测试健康检查接口"""
        response = await embedding_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "device" in data
        assert "model_loaded" in data

    async def test_create_embedding_single(self, embedding_client: httpx.AsyncClient, sample_text: str):
        """测试单个文本嵌入"""
        response = await embedding_client.post(
            "/v1/embeddings",
            json={"input": sample_text, "model": "Qwen3-Embedding-0.6B"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert "embedding" in data["data"][0]
        assert isinstance(data["data"][0]["embedding"], list)
        assert len(data["data"][0]["embedding"]) > 0

    async def test_create_embedding_batch(self, embedding_client: httpx.AsyncClient, sample_texts: list[str]):
        """测试批量文本嵌入"""
        response = await embedding_client.post(
            "/v1/embeddings",
            json={"input": sample_texts, "model": "Qwen3-Embedding-0.6B"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == len(sample_texts)

    async def test_create_embedding_empty_input(self, embedding_client: httpx.AsyncClient):
        """测试空输入"""
        response = await embedding_client.post("/v1/embeddings", json={"input": "", "model": "Qwen3-Embedding-0.6B"})
        # 应该返回错误或空结果
        assert response.status_code in [200, 400, 422]

    async def test_get_models(self, embedding_client: httpx.AsyncClient):
        """测试获取模型列表"""
        response = await embedding_client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0

    async def test_get_stats(self, embedding_client: httpx.AsyncClient):
        """测试获取统计信息"""
        response = await embedding_client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "cache_stats" in data
