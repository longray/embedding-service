"""
Embedding服务测试 - 接口、边界条件、错误处理
"""

import pytest
import httpx

pytestmark = pytest.mark.e2e


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
        assert "cache" in data


@pytest.mark.asyncio
class TestEmbeddingServiceBoundary:
    """边界条件测试"""

    async def test_very_long_text(self, embedding_client: httpx.AsyncClient):
        """测试超长文本（接近MAX_LENGTH限制）"""
        # 生成约8000字符的文本（接近MAX_LENGTH=8192）
        long_text = "测试文本 " * 1000
        response = await embedding_client.post(
            "/v1/embeddings",
            json={"input": long_text, "model": "Qwen3-Embedding-0.6B"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        assert len(data["data"][0]["embedding"]) > 0

    async def test_empty_string(self, embedding_client: httpx.AsyncClient):
        """测试空字符串"""
        response = await embedding_client.post(
            "/v1/embeddings",
            json={"input": "", "model": "Qwen3-Embedding-0.6B"},
        )
        # 空字符串应该返回200，但embedding可能为零向量
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1

    async def test_special_characters(self, embedding_client: httpx.AsyncClient):
        """测试特殊字符（emoji、Unicode）"""
        special_text = "Hello 👋 世界 🌍 测试 ñ é ü"
        response = await embedding_client.post(
            "/v1/embeddings",
            json={"input": special_text, "model": "Qwen3-Embedding-0.6B"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        assert len(data["data"][0]["embedding"]) > 0

    async def test_large_batch(self, embedding_client: httpx.AsyncClient):
        """测试大批量文本（50条）"""
        texts = [f"测试文本 {i}" for i in range(50)]
        response = await embedding_client.post(
            "/v1/embeddings",
            json={"input": texts, "model": "Qwen3-Embedding-0.6B"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 50

    async def test_whitespace_only(self, embedding_client: httpx.AsyncClient):
        """测试仅包含空白字符"""
        response = await embedding_client.post(
            "/v1/embeddings",
            json={"input": "   \n\t  ", "model": "Qwen3-Embedding-0.6B"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1


@pytest.mark.asyncio
class TestEmbeddingServiceErrors:
    """错误处理测试"""

    async def test_missing_input_field(self, embedding_client: httpx.AsyncClient):
        """测试缺失input字段"""
        response = await embedding_client.post(
            "/v1/embeddings",
            json={"model": "Qwen3-Embedding-0.6B"},
        )
        assert response.status_code == 422  # FastAPI validation error

    async def test_missing_model_field(self, embedding_client: httpx.AsyncClient):
        """测试缺失 model 字段 — 服务使用默认模型"""
        response = await embedding_client.post("/v1/embeddings", json={"input": "测试"})
        assert response.status_code == 200
        data = response.json()
        assert "data" in data

    async def test_invalid_input_type(self, embedding_client: httpx.AsyncClient):
        """测试无效的input类型（数字而非字符串）"""
        response = await embedding_client.post(
            "/v1/embeddings",
            json={"input": 12345, "model": "Qwen3-Embedding-0.6B"},
        )
        assert response.status_code == 422

    async def test_invalid_model_name(self, embedding_client: httpx.AsyncClient):
        """测试无效的模型名"""
        response = await embedding_client.post(
            "/v1/embeddings",
            json={"input": "test", "model": "invalid-model-name"},
        )
        # 服务可能返回400或继续处理（取决于实现）
        assert response.status_code in [200, 400, 422]

    async def test_empty_array_input(self, embedding_client: httpx.AsyncClient):
        """测试空数组输入"""
        response = await embedding_client.post(
            "/v1/embeddings",
            json={"input": [], "model": "Qwen3-Embedding-0.6B"},
        )
        # 空数组应该返回错误或空结果
        assert response.status_code in [200, 400, 422]

    async def test_null_input(self, embedding_client: httpx.AsyncClient):
        """测试null输入"""
        response = await embedding_client.post(
            "/v1/embeddings",
            json={"input": None, "model": "Qwen3-Embedding-0.6B"},
        )
        assert response.status_code == 422

    async def test_malformed_json(self, embedding_client: httpx.AsyncClient):
        """测试格式错误的JSON"""
        response = await embedding_client.post(
            "/v1/embeddings",
            content="{invalid json}",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422
