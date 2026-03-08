"""
包装层服务接口测试
"""

import pytest
import httpx


@pytest.mark.asyncio
class TestWrapperService:
    """包装层服务测试套件"""

    async def test_health_check(self, wrapper_client: httpx.AsyncClient):
        """测试健康检查接口"""
        response = await wrapper_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "cache_stats" in data
        assert "circuit_breakers" in data
        assert "embedding" in data["circuit_breakers"]
        assert "llm" in data["circuit_breakers"]

    async def test_embeddings_with_cache(self, wrapper_client: httpx.AsyncClient, sample_text: str):
        """测试嵌入接口（缓存功能）"""
        # 第一次请求（缓存未命中）
        response1 = await wrapper_client.post(
            "/v1/embeddings",
            json={"input": sample_text, "model": "Qwen3-Embedding-0.6B"},
        )
        assert response1.status_code == 200
        data1 = response1.json()

        # 第二次请求（缓存命中）
        response2 = await wrapper_client.post(
            "/v1/embeddings",
            json={"input": sample_text, "model": "Qwen3-Embedding-0.6B"},
        )
        assert response2.status_code == 200
        data2 = response2.json()

        # 验证结果一致
        assert data1["data"][0]["embedding"] == data2["data"][0]["embedding"]

    async def test_chat_completions(self, wrapper_client: httpx.AsyncClient, sample_messages: list[dict]):
        """测试聊天补全接口"""
        response = await wrapper_client.post(
            "/v1/chat/completions",
            json={"messages": sample_messages, "model": "MiniCPM4-0.5B"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data

    async def test_metrics(self, wrapper_client: httpx.AsyncClient):
        """测试Prometheus指标接口"""
        response = await wrapper_client.get("/metrics")
        assert response.status_code == 200
        content = response.text
        assert "wrapper_requests_total" in content
        assert "wrapper_cache_hits_total" in content
