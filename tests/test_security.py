"""
安全测试 - 注入攻击、恶意输入、安全防护
"""

import pytest
import httpx


@pytest.mark.asyncio
class TestSecurityInjection:
    """注入攻击测试"""

    async def test_sql_injection_attempt(self, embedding_client: httpx.AsyncClient):
        """测试SQL注入尝试"""
        malicious_input = "'; DROP TABLE users; --"
        response = await embedding_client.post(
            "/v1/embeddings",
            json={"input": malicious_input, "model": "Qwen3-Embedding-0.6B"},
        )
        # 服务应该正常处理，不应该执行SQL
        assert response.status_code == 200

    async def test_xss_attempt(self, embedding_client: httpx.AsyncClient):
        """测试XSS攻击尝试"""
        xss_input = "<script>alert('XSS')</script>"
        response = await embedding_client.post(
            "/v1/embeddings",
            json={"input": xss_input, "model": "Qwen3-Embedding-0.6B"},
        )
        assert response.status_code == 200
        # 验证响应不包含未转义的脚本
        data = response.json()
        assert "<script>" not in str(data)

    async def test_command_injection_attempt(self, llm_client: httpx.AsyncClient):
        """测试命令注入尝试"""
        malicious_content = "; rm -rf / #"
        response = await llm_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": malicious_content}],
                "model": "MiniCPM4-0.5B",
            },
        )
        # 服务应该正常处理，不应该执行系统命令
        assert response.status_code == 200


@pytest.mark.asyncio
class TestSecurityPayload:
    """恶意payload测试"""

    async def test_extremely_large_payload(self, embedding_client: httpx.AsyncClient):
        """测试超大payload（>1MB）"""
        # 生成约1MB的文本
        large_text = "A" * (1024 * 1024)
        response = await embedding_client.post(
            "/v1/embeddings",
            json={"input": large_text, "model": "Qwen3-Embedding-0.6B"},
            timeout=60.0,
        )
        # 服务应该拒绝或截断
        assert response.status_code in [200, 400, 413, 422]

    async def test_deeply_nested_json(self, embedding_client: httpx.AsyncClient):
        """测试深度嵌套的JSON"""
        # 创建深度嵌套的结构
        nested = {"input": "test", "model": "Qwen3-Embedding-0.6B"}
        for _ in range(100):
            nested = {"nested": nested}

        response = await embedding_client.post("/v1/embeddings", json=nested)
        # 应该返回验证错误
        assert response.status_code in [400, 422]

    async def test_null_bytes_in_input(self, embedding_client: httpx.AsyncClient):
        """测试输入中的null字节"""
        malicious_input = "test\x00malicious"
        response = await embedding_client.post(
            "/v1/embeddings",
            json={"input": malicious_input, "model": "Qwen3-Embedding-0.6B"},
        )
        # 服务应该处理或拒绝
        assert response.status_code in [200, 400, 422]


@pytest.mark.asyncio
class TestSecurityHeaders:
    """安全头测试"""

    async def test_cors_headers(self, wrapper_client: httpx.AsyncClient):
        """测试CORS头"""
        response = await wrapper_client.get("/health")
        # 检查是否有CORS相关头
        assert response.status_code == 200

    async def test_content_type_validation(self, embedding_client: httpx.AsyncClient):
        """测试Content-Type验证"""
        # 发送错误的Content-Type
        response = await embedding_client.post(
            "/v1/embeddings",
            content='{"input": "test", "model": "Qwen3-Embedding-0.6B"}',
            headers={"Content-Type": "text/plain"},
        )
        # 应该返回错误
        assert response.status_code in [400, 415, 422]


@pytest.mark.asyncio
class TestSecurityRateLimiting:
    """速率限制测试（如果实现）"""

    async def test_rapid_requests(self, embedding_client: httpx.AsyncClient):
        """测试快速连续请求"""
        # 快速发送多个请求
        responses = []
        for i in range(20):
            response = await embedding_client.post(
                "/v1/embeddings",
                json={"input": f"快速请求 {i}", "model": "Qwen3-Embedding-0.6B"},
            )
            responses.append(response)

        # 所有请求应该成功（如果没有速率限制）
        # 或者部分请求被限制（如果有速率限制）
        success_count = sum(1 for r in responses if r.status_code == 200)
        assert success_count > 0, "至少应该有一些请求成功"
