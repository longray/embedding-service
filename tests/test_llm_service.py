"""
LLM服务接口测试
"""

import pytest
import httpx


@pytest.mark.asyncio
class TestLLMService:
    """LLM服务测试套件"""

    async def test_health_check(self, llm_client: httpx.AsyncClient):
        """测试健康检查接口"""
        response = await llm_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "device" in data
        assert "model_loaded" in data

    async def test_chat_completion(self, llm_client: httpx.AsyncClient, sample_messages: list[dict]):
        """测试聊天补全接口"""
        response = await llm_client.post(
            "/v1/chat/completions",
            json={
                "messages": sample_messages,
                "model": "MiniCPM4-0.5B",
                "temperature": 0.7,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]

    async def test_generate(self, llm_client: httpx.AsyncClient):
        """测试简单生成接口"""
        response = await llm_client.post("/generate", json={"prompt": "你好", "model": "MiniCPM4-0.5B"})
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert isinstance(data["response"], str)

    async def test_get_models(self, llm_client: httpx.AsyncClient):
        """测试获取模型列表"""
        response = await llm_client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data

    async def test_get_stats(self, llm_client: httpx.AsyncClient):
        """测试获取统计信息"""
        response = await llm_client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "cache_stats" in data
