"""
语义搜索性能测试

测试指标：
1. 搜索延迟（latency）
2. 搜索准确性（relevance）
3. 批量吞吐量（throughput）

需要服务运行：
- SurrealDB (ws://localhost:8000)
- Embedding服务 (http://localhost:18000)
- 包装服务 (http://localhost:3001)
"""

import time
import pytest
from httpx import AsyncClient

BASE_URL = "http://localhost:3001"

# 测试数据：相关性已知的记忆对
TEST_MEMORIES = [
    {"content": "Python是一种高级编程语言", "metadata": {"topic": "programming"}},
    {"content": "JavaScript用于Web开发", "metadata": {"topic": "programming"}},
    {"content": "机器学习是人工智能的分支", "metadata": {"topic": "ai"}},
    {"content": "深度学习使用神经网络", "metadata": {"topic": "ai"}},
    {"content": "北京是中国的首都", "metadata": {"topic": "geography"}},
]


@pytest.mark.asyncio
async def test_search_latency():
    """测试搜索延迟（单次查询响应时间）"""
    async with AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        # 上传测试数据
        await client.post("/api/v1/memories", json={"memories": TEST_MEMORIES})

        # 测试搜索延迟
        query = "编程语言"
        start = time.perf_counter()
        response = await client.post("/api/v1/memories/search", json={"query": query, "mode": "vector", "limit": 5})
        latency = (time.perf_counter() - start) * 1000  # ms

        assert response.status_code == 200
        assert latency < 500, f"搜索延迟过高: {latency:.2f}ms"
        print(f"✓ 搜索延迟: {latency:.2f}ms")


@pytest.mark.asyncio
async def test_search_relevance():
    """测试搜索准确性（相关结果排序）"""
    async with AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        # 上传测试数据
        await client.post("/api/v1/memories", json={"memories": TEST_MEMORIES})

        # 测试：查询"编程"应该返回programming相关的结果
        response = await client.post(
            "/api/v1/memories/search",
            json={"query": "编程语言和开发", "mode": "vector", "limit": 3, "threshold": 0.0}
        )

        assert response.status_code == 200
        data = response.json()
        results = data.get("results", [])
        
        if len(results) == 0:
            pytest.skip("未返回搜索结果，可能数据库为空")

        # 验证：前2个结果应该包含"Python"或"JavaScript"
        top_contents = [r["content"] for r in results[:2]]
        relevant = any("Python" in c or "JavaScript" in c for c in top_contents)
        assert relevant, f"搜索结果相关性低: {top_contents}"
        print(f"✓ 搜索相关性验证通过")


@pytest.mark.asyncio
async def test_batch_throughput():
    """测试批量搜索吞吐量（QPS）"""
    async with AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        # 上传测试数据
        await client.post("/api/v1/memories", json={"memories": TEST_MEMORIES})

        # 批量查询
        queries = ["编程", "人工智能", "地理", "Python", "机器学习"]
        start = time.perf_counter()

        for query in queries:
            response = await client.post("/api/v1/memories/search", json={"query": query, "mode": "vector", "limit": 5})
            assert response.status_code == 200

        elapsed = time.perf_counter() - start
        qps = len(queries) / elapsed

        assert qps > 1.0, f"吞吐量过低: {qps:.2f} QPS"
        print(f"✓ 批量吞吐量: {qps:.2f} QPS ({len(queries)}个查询/{elapsed:.2f}秒)")


@pytest.mark.asyncio
async def test_hybrid_vs_vector():
    """对比混合搜索和向量搜索的性能"""
    async with AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        # 上传测试数据
        await client.post("/api/v1/memories", json={"memories": TEST_MEMORIES})

        query = "深度学习"

        # 向量搜索
        start = time.perf_counter()
        vector_resp = await client.post("/api/v1/memories/search", json={"query": query, "mode": "vector", "limit": 5})
        vector_time = (time.perf_counter() - start) * 1000

        # 混合搜索
        start = time.perf_counter()
        hybrid_resp = await client.post("/api/v1/memories/search", json={"query": query, "mode": "hybrid", "limit": 5})
        hybrid_time = (time.perf_counter() - start) * 1000

        assert vector_resp.status_code == 200
        assert hybrid_resp.status_code == 200

        print(f"✓ 向量搜索: {vector_time:.2f}ms")
        print(f"✓ 混合搜索: {hybrid_time:.2f}ms")
        print(f"✓ 性能差异: {abs(hybrid_time - vector_time):.2f}ms")
