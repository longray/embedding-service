"""
性能测试 - 并发、响应时间、吞吐量
"""

import asyncio
import time
import pytest
import httpx

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
class TestEmbeddingServicePerformance:
    """Embedding服务性能测试"""

    async def test_single_request_response_time(self, embedding_client: httpx.AsyncClient, sample_text: str):
        """测试单个请求的响应时间（应<2秒）"""
        start_time = time.time()
        response = await embedding_client.post(
            "/v1/embeddings",
            json={"input": sample_text, "model": "Qwen3-Embedding-0.6B"},
        )
        elapsed_time = time.time() - start_time

        assert response.status_code == 200
        assert elapsed_time < 2.0, f"响应时间过长: {elapsed_time:.2f}秒"

    async def test_concurrent_requests(self, embedding_client: httpx.AsyncClient):
        """测试并发请求（10个并发）"""

        async def make_request(text: str):
            return await embedding_client.post(
                "/v1/embeddings",
                json={"input": text, "model": "Qwen3-Embedding-0.6B"},
            )

        # 创建10个并发请求
        tasks = [make_request(f"并发测试文本 {i}") for i in range(10)]

        start_time = time.time()
        responses = await asyncio.gather(*tasks)
        elapsed_time = time.time() - start_time

        # 验证所有请求成功
        for response in responses:
            assert response.status_code == 200

        # 验证总时间合理（应该比串行快）
        assert elapsed_time < 20.0, f"并发请求总时间过长: {elapsed_time:.2f}秒"

    async def test_batch_processing_performance(self, embedding_client: httpx.AsyncClient):
        """测试批量处理性能（100条文本）"""
        texts = [f"批量性能测试 {i}" for i in range(100)]

        start_time = time.time()
        response = await embedding_client.post(
            "/v1/embeddings",
            json={"input": texts, "model": "Qwen3-Embedding-0.6B"},
        )
        elapsed_time = time.time() - start_time

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 100

        # 验证批量处理时间合理
        assert elapsed_time < 30.0, f"批量处理时间过长: {elapsed_time:.2f}秒"


@pytest.mark.asyncio
class TestLLMServicePerformance:
    """LLM服务性能测试"""

    async def test_chat_response_time(self, llm_client: httpx.AsyncClient):
        """测试聊天响应时间"""
        start_time = time.time()
        response = await llm_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "你好"}],
                "model": "MiniCPM4-0.5B",
            },
        )
        elapsed_time = time.time() - start_time

        assert response.status_code == 200
        assert elapsed_time < 5.0, f"聊天响应时间过长: {elapsed_time:.2f}秒"

    async def test_concurrent_chat_requests(self, llm_client: httpx.AsyncClient):
        """测试并发聊天请求（5个并发）"""

        async def make_chat_request(content: str):
            return await llm_client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": content}],
                    "model": "MiniCPM4-0.5B",
                },
            )

        tasks = [make_chat_request(f"问题 {i}") for i in range(5)]

        start_time = time.time()
        responses = await asyncio.gather(*tasks)
        elapsed_time = time.time() - start_time

        for response in responses:
            assert response.status_code == 200

        assert elapsed_time < 25.0, f"并发聊天总时间过长: {elapsed_time:.2f}秒"


@pytest.mark.asyncio
class TestWrapperServicePerformance:
    """包装层服务性能测试"""

    async def test_cache_performance_improvement(self, wrapper_client: httpx.AsyncClient):
        """测试缓存对性能的提升"""
        test_input = "缓存性能测试文本"

        # 第一次请求（缓存未命中）
        start_time1 = time.time()
        response1 = await wrapper_client.post(
            "/v1/embeddings",
            json={"input": test_input, "model": "Qwen3-Embedding-0.6B"},
        )
        time1 = time.time() - start_time1

        # 第二次请求（缓存命中）
        start_time2 = time.time()
        response2 = await wrapper_client.post(
            "/v1/embeddings",
            json={"input": test_input, "model": "Qwen3-Embedding-0.6B"},
        )
        time2 = time.time() - start_time2

        assert response1.status_code == 200
        assert response2.status_code == 200

        # 缓存命中应该更快
        assert time2 < time1, f"缓存未提升性能: 第一次={time1:.3f}s, 第二次={time2:.3f}s"

    async def test_wrapper_overhead(
        self, embedding_client: httpx.AsyncClient, wrapper_client: httpx.AsyncClient, sample_text: str
    ):
        """测试包装层的开销"""
        # 直接访问后端
        start_time1 = time.time()
        response1 = await embedding_client.post(
            "/v1/embeddings",
            json={"input": sample_text, "model": "Qwen3-Embedding-0.6B"},
        )
        time_direct = time.time() - start_time1

        # 通过包装层访问（清除缓存影响，使用不同文本）
        unique_text = f"{sample_text} - unique_{time.time()}"
        start_time2 = time.time()
        response2 = await wrapper_client.post(
            "/v1/embeddings",
            json={"input": unique_text, "model": "Qwen3-Embedding-0.6B"},
        )
        time_wrapper = time.time() - start_time2

        assert response1.status_code == 200
        assert response2.status_code == 200

        # 包装层开销应该很小（<100ms）
        overhead = time_wrapper - time_direct
        assert overhead < 0.1, f"包装层开销过大: {overhead:.3f}秒"
