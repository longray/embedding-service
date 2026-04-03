"""
包装层服务扩展测试 - 熔断器和缓存完整测试
"""

import asyncio
import pytest
import httpx

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
class TestWrapperServiceCircuitBreaker:
    """熔断器测试"""

    async def test_circuit_breaker_status_in_health(self, wrapper_client: httpx.AsyncClient):
        """测试健康检查中的熔断器状态"""
        response = await wrapper_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "circuit_breakers" in data
        assert "embedding" in data["circuit_breakers"]
        assert "llm" in data["circuit_breakers"]

        # 验证熔断器状态字段
        embedding_cb = data["circuit_breakers"]["embedding"]
        assert "state" in embedding_cb
        assert "failure_count" in embedding_cb
        assert "last_failure_time" in embedding_cb

    async def test_circuit_breaker_closed_state(self, wrapper_client: httpx.AsyncClient):
        """测试熔断器关闭状态（正常）"""
        # 先检查健康状态
        health_response = await wrapper_client.get("/health")
        health_data = health_response.json()

        # 如果后端服务正常，熔断器应该是closed
        if health_data["status"] == "healthy":
            embedding_cb = health_data["circuit_breakers"]["embedding"]
            # 注意：如果之前有失败，状态可能不是closed
            assert embedding_cb["state"] in ["closed", "half_open"]

    async def test_successful_request_through_wrapper(self, wrapper_client: httpx.AsyncClient, sample_text: str):
        """测试通过包装层的成功请求"""
        response = await wrapper_client.post(
            "/v1/embeddings",
            json={"input": sample_text, "model": "Qwen3-Embedding-0.6B"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0


@pytest.mark.asyncio
class TestWrapperServiceCache:
    """缓存测试"""

    async def test_cache_hit_and_miss(self, wrapper_client: httpx.AsyncClient):
        """测试缓存命中和未命中"""
        test_input = "缓存测试文本 - unique_12345"

        # 第一次请求 - 缓存未命中
        response1 = await wrapper_client.post(
            "/v1/embeddings",
            json={"input": test_input, "model": "Qwen3-Embedding-0.6B"},
        )
        assert response1.status_code == 200
        data1 = response1.json()

        # 第二次请求 - 缓存命中
        response2 = await wrapper_client.post(
            "/v1/embeddings",
            json={"input": test_input, "model": "Qwen3-Embedding-0.6B"},
        )
        assert response2.status_code == 200
        data2 = response2.json()

        # 验证结果一致
        assert data1["data"][0]["embedding"] == data2["data"][0]["embedding"]

        # 检查缓存统计
        health_response = await wrapper_client.get("/health")
        health_data = health_response.json()
        assert "cache_stats" in health_data
        cache_stats = health_data["cache_stats"]
        assert cache_stats["hits"] > 0

    async def test_cache_isolation_different_inputs(self, wrapper_client: httpx.AsyncClient):
        """测试不同输入的缓存隔离"""
        input1 = "测试文本A"
        input2 = "测试文本B"

        # 请求不同的输入
        response1 = await wrapper_client.post(
            "/v1/embeddings",
            json={"input": input1, "model": "Qwen3-Embedding-0.6B"},
        )
        response2 = await wrapper_client.post(
            "/v1/embeddings",
            json={"input": input2, "model": "Qwen3-Embedding-0.6B"},
        )

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        # 验证不同输入产生不同的embedding
        assert data1["data"][0]["embedding"] != data2["data"][0]["embedding"]

    async def test_cache_with_batch_input(self, wrapper_client: httpx.AsyncClient):
        """测试批量输入的缓存"""
        batch_input = ["批量测试1", "批量测试2", "批量测试3"]

        # 第一次批量请求
        response1 = await wrapper_client.post(
            "/v1/embeddings",
            json={"input": batch_input, "model": "Qwen3-Embedding-0.6B"},
        )
        assert response1.status_code == 200
        data1 = response1.json()

        # 第二次相同批量请求
        response2 = await wrapper_client.post(
            "/v1/embeddings",
            json={"input": batch_input, "model": "Qwen3-Embedding-0.6B"},
        )
        assert response2.status_code == 200
        data2 = response2.json()

        # 验证结果一致
        assert len(data1["data"]) == len(data2["data"])
        for i in range(len(data1["data"])):
            assert data1["data"][i]["embedding"] == data2["data"][i]["embedding"]

    async def test_cache_stats_in_health(self, wrapper_client: httpx.AsyncClient):
        """测试健康检查中的缓存统计"""
        response = await wrapper_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "cache_stats" in data

        cache_stats = data["cache_stats"]
        assert "hits" in cache_stats
        assert "misses" in cache_stats
        assert "size" in cache_stats
        assert "max_size" in cache_stats

        # 验证统计数据类型
        assert isinstance(cache_stats["hits"], int)
        assert isinstance(cache_stats["misses"], int)
        assert isinstance(cache_stats["size"], int)


@pytest.mark.asyncio
class TestWrapperServiceMetrics:
    """Prometheus指标测试"""

    async def test_metrics_endpoint(self, wrapper_client: httpx.AsyncClient):
        """测试Prometheus指标端点"""
        response = await wrapper_client.get("/metrics")
        assert response.status_code == 200
        content = response.text

        # 验证关键指标存在
        assert "wrapper_requests_total" in content
        assert "wrapper_cache_hits_total" in content
        assert "wrapper_cache_misses_total" in content
        assert "wrapper_request_duration_seconds" in content

    async def test_metrics_after_requests(self, wrapper_client: httpx.AsyncClient, sample_text: str):
        """测试请求后的指标更新"""
        # 发送一些请求
        for _ in range(3):
            await wrapper_client.post(
                "/v1/embeddings",
                json={"input": sample_text, "model": "Qwen3-Embedding-0.6B"},
            )

        # 检查指标
        response = await wrapper_client.get("/metrics")
        assert response.status_code == 200
        content = response.text

        # 验证请求计数器增加
        assert "wrapper_requests_total" in content
