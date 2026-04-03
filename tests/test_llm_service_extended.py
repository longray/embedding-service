"""
LLM服务测试 - 接口、边界条件、错误处理
"""

import pytest
import httpx

pytestmark = pytest.mark.e2e


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


@pytest.mark.asyncio
class TestLLMServiceBoundary:
    """边界条件测试"""

    async def test_very_long_message(self, llm_client: httpx.AsyncClient):
        """测试超长消息"""
        long_content = "这是一个很长的测试消息。" * 200
        response = await llm_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": long_content}],
                "model": "MiniCPM4-0.5B",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0

    async def test_empty_message_content(self, llm_client: httpx.AsyncClient):
        """测试空消息内容"""
        response = await llm_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": ""}],
                "model": "MiniCPM4-0.5B",
            },
        )
        # 空消息应该返回200或400
        assert response.status_code in [200, 400]

    async def test_special_characters_in_message(self, llm_client: httpx.AsyncClient):
        """测试消息中的特殊字符"""
        response = await llm_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello 👋 世界 🌍 <script>alert('test')</script>"}],
                "model": "MiniCPM4-0.5B",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data

    async def test_multiple_messages(self, llm_client: httpx.AsyncClient):
        """测试多轮对话"""
        response = await llm_client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
                    {"role": "user", "content": "介绍一下自己"},
                ],
                "model": "MiniCPM4-0.5B",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data

    async def test_max_tokens_parameter(self, llm_client: httpx.AsyncClient):
        """测试max_tokens参数"""
        response = await llm_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "介绍一下Python"}],
                "model": "MiniCPM4-0.5B",
                "max_tokens": 50,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data


@pytest.mark.asyncio
class TestLLMServiceErrors:
    """错误处理测试"""

    async def test_missing_messages_field(self, llm_client: httpx.AsyncClient):
        """测试缺失messages字段"""
        response = await llm_client.post(
            "/v1/chat/completions",
            json={"model": "MiniCPM4-0.5B"},
        )
        assert response.status_code == 422

    async def test_missing_model_field(self, llm_client: httpx.AsyncClient):
        """测试缺失model字段"""
        response = await llm_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "test"}]},
        )
        assert response.status_code == 422

    async def test_invalid_messages_type(self, llm_client: httpx.AsyncClient):
        """测试无效的messages类型"""
        response = await llm_client.post(
            "/v1/chat/completions",
            json={"messages": "invalid", "model": "MiniCPM4-0.5B"},
        )
        assert response.status_code == 422

    async def test_empty_messages_array(self, llm_client: httpx.AsyncClient):
        """测试空messages数组"""
        response = await llm_client.post(
            "/v1/chat/completions",
            json={"messages": [], "model": "MiniCPM4-0.5B"},
        )
        assert response.status_code in [400, 422]

    async def test_invalid_role(self, llm_client: httpx.AsyncClient):
        """测试无效的role"""
        response = await llm_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "invalid_role", "content": "test"}],
                "model": "MiniCPM4-0.5B",
            },
        )
        assert response.status_code in [400, 422]

    async def test_missing_content_field(self, llm_client: httpx.AsyncClient):
        """测试缺失content字段"""
        response = await llm_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user"}],
                "model": "MiniCPM4-0.5B",
            },
        )
        assert response.status_code == 422

    async def test_null_messages(self, llm_client: httpx.AsyncClient):
        """测试null messages"""
        response = await llm_client.post(
            "/v1/chat/completions",
            json={"messages": None, "model": "MiniCPM4-0.5B"},
        )
        assert response.status_code == 422

    async def test_invalid_max_tokens(self, llm_client: httpx.AsyncClient):
        """测试无效的max_tokens值"""
        response = await llm_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "model": "MiniCPM4-0.5B",
                "max_tokens": -1,
            },
        )
        assert response.status_code in [400, 422]
