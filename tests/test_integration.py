"""
集成测试 - 测试服务之间的交互
"""

import pytest
import httpx

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
class TestIntegration:
    """集成测试套件"""

    async def test_end_to_end_embedding(
        self,
        embedding_client: httpx.AsyncClient,
        wrapper_client: httpx.AsyncClient,
        sample_text: str,
    ):
        """端到端测试：直接访问vs通过包装层"""
        # 直接访问后端
        response1 = await embedding_client.post(
            "/v1/embeddings",
            json={"input": sample_text, "model": "Qwen3-Embedding-0.6B"},
        )
        assert response1.status_code == 200

        # 通过包装层访问
        response2 = await wrapper_client.post(
            "/v1/embeddings",
            json={"input": sample_text, "model": "Qwen3-Embedding-0.6B"},
        )
        assert response2.status_code == 200

        # 验证结果一致
        data1 = response1.json()
        data2 = response2.json()
        assert len(data1["data"]) == len(data2["data"])

    async def test_service_dependency(self, wrapper_client: httpx.AsyncClient, sample_text: str):
        """测试服务依赖关系"""
        # 包装层依赖后端服务
        response = await wrapper_client.get("/health")
        assert response.status_code == 200
        data = response.json()

        # 检查熔断器状态
        assert "circuit_breakers" in data
        # 如果后端服务正常，熔断器应该是closed
        # 注意：这个测试假设后端服务正在运行
