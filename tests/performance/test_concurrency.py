"""并发压力测试

验证 BL-T-4: 并发压力测试

测试范围：
- 并发连接测试
- 并发写入测试
- 并发搜索测试
- 资源监控
- 性能基准

运行方式：
    uv run pytest tests/performance/test_concurrency.py -v

前置条件：
- Wrapper 服务运行在 http://localhost:18008
- 所有依赖服务正常运行
"""

import asyncio
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_PERF_TESTS") == "1", reason="Performance tests skipped (SKIP_PERF_TESTS=1)"
)

BASE_URL = "http://localhost:18008"


@dataclass
class PerformanceMetrics:
    """性能指标"""

    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    requests_per_second: float


async def run_concurrent_requests(
    client: httpx.AsyncClient,
    requests: list[dict],
    max_concurrent: int = 10,
) -> PerformanceMetrics:
    """执行并发请求并收集指标"""

    semaphore = asyncio.Semaphore(max_concurrent)
    response_times = []
    successful = 0
    failed = 0

    async def execute_request(req: dict) -> tuple[bool, float]:
        async with semaphore:
            start = time.time()
            try:
                method = req.get("method", "GET")
                url = req["url"]
                kwargs = {k: v for k, v in req.items() if k not in ["method", "url"]}

                response = await client.request(method, url, timeout=30.0, **kwargs)
                duration = time.time() - start

                success = 200 <= response.status_code < 300
                return success, duration
            except Exception:
                duration = time.time() - start
                return False, duration

    start_time = time.time()
    results = await asyncio.gather(*[execute_request(req) for req in requests])
    total_time = time.time() - start_time

    for success, duration in results:
        response_times.append(duration)
        if success:
            successful += 1
        else:
            failed += 1

    if response_times:
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
    else:
        avg_time = min_time = max_time = 0.0

    return PerformanceMetrics(
        total_requests=len(requests),
        successful_requests=successful,
        failed_requests=failed,
        total_time=total_time,
        avg_response_time=avg_time,
        min_response_time=min_time,
        max_response_time=max_time,
        requests_per_second=len(requests) / total_time if total_time > 0 else 0,
    )


class TestConcurrentConnections:
    """并发连接测试"""

    @pytest.mark.asyncio
    async def test_concurrent_health_checks_10(self):
        """测试 10 并发健康检查"""
        requests = [{"method": "GET", "url": f"{BASE_URL}/health"} for _ in range(10)]

        async with httpx.AsyncClient() as client:
            metrics = await run_concurrent_requests(client, requests, max_concurrent=10)

        assert metrics.failed_requests == 0, f"失败请求: {metrics.failed_requests}"
        assert metrics.avg_response_time < 1.0, f"平均响应时间过长: {metrics.avg_response_time:.2f}s"
        print(
            f"\n10 并发健康检查: {metrics.requests_per_second:.2f} req/s, 平均: {metrics.avg_response_time * 1000:.2f}ms"
        )

    @pytest.mark.asyncio
    async def test_concurrent_health_checks_50(self):
        """测试 50 并发健康检查"""
        requests = [{"method": "GET", "url": f"{BASE_URL}/health"} for _ in range(50)]

        async with httpx.AsyncClient() as client:
            metrics = await run_concurrent_requests(client, requests, max_concurrent=50)

        assert metrics.failed_requests == 0, f"失败请求: {metrics.failed_requests}"
        assert metrics.avg_response_time < 2.0, f"平均响应时间过长: {metrics.avg_response_time:.2f}s"
        print(
            f"\n50 并发健康检查: {metrics.requests_per_second:.2f} req/s, 平均: {metrics.avg_response_time * 1000:.2f}ms"
        )

    @pytest.mark.asyncio
    async def test_concurrent_health_checks_100(self):
        """测试 100 并发健康检查"""
        requests = [{"method": "GET", "url": f"{BASE_URL}/health"} for _ in range(100)]

        async with httpx.AsyncClient() as client:
            metrics = await run_concurrent_requests(client, requests, max_concurrent=100)

        # 允许少量失败（< 5%）
        max_allowed_failures = 5
        assert metrics.failed_requests <= max_allowed_failures, f"失败请求过多: {metrics.failed_requests}"
        assert metrics.avg_response_time < 5.0, f"平均响应时间过长: {metrics.avg_response_time:.2f}s"
        print(
            f"\n100 并发健康检查: {metrics.requests_per_second:.2f} req/s, 平均: {metrics.avg_response_time * 1000:.2f}ms, 失败: {metrics.failed_requests}"
        )


class TestConcurrentWrites:
    """并发写入测试"""

    @pytest.mark.asyncio
    async def test_concurrent_memory_uploads_10(self):
        """测试 10 并发记忆上传"""
        uid = str(uuid.uuid4())[:8]
        requests = [
            {
                "method": "POST",
                "url": f"{BASE_URL}/api/v1/memories",
                "json": {
                    "memories": [{"content": f"并发测试 {uid} {i}"}],
                    "tenant_id": "concurrent_test",
                },
            }
            for i in range(10)
        ]

        async with httpx.AsyncClient() as client:
            metrics = await run_concurrent_requests(client, requests, max_concurrent=10)

        assert metrics.failed_requests == 0, f"失败请求: {metrics.failed_requests}"
        assert metrics.avg_response_time < 5.0, f"平均响应时间过长: {metrics.avg_response_time:.2f}s"
        print(f"\n10 并发写入: {metrics.requests_per_second:.2f} req/s, 平均: {metrics.avg_response_time * 1000:.2f}ms")

    @pytest.mark.asyncio
    async def test_concurrent_memory_uploads_30(self):
        """测试 30 并发记忆上传"""
        uid = str(uuid.uuid4())[:8]
        requests = [
            {
                "method": "POST",
                "url": f"{BASE_URL}/api/v1/memories",
                "json": {
                    "memories": [{"content": f"并发测试 {uid} {i}"}],
                    "tenant_id": "concurrent_test",
                },
            }
            for i in range(30)
        ]

        async with httpx.AsyncClient() as client:
            metrics = await run_concurrent_requests(client, requests, max_concurrent=30)

        # 允许少量失败（< 10%）
        max_allowed_failures = 3
        assert metrics.failed_requests <= max_allowed_failures, f"失败请求过多: {metrics.failed_requests}"
        assert metrics.avg_response_time < 10.0, f"平均响应时间过长: {metrics.avg_response_time:.2f}s"
        print(
            f"\n30 并发写入: {metrics.requests_per_second:.2f} req/s, 平均: {metrics.avg_response_time * 1000:.2f}ms, 失败: {metrics.failed_requests}"
        )

    @pytest.mark.asyncio
    async def test_concurrent_memory_uploads_data_consistency(self):
        """测试并发写入数据一致性"""
        uid = str(uuid.uuid4())[:8]
        num_requests = 20

        requests = [
            {
                "method": "POST",
                "url": f"{BASE_URL}/api/v1/memories",
                "json": {
                    "memories": [{"content": f"一致性测试 {uid} {i}", "source_id": f"consistency_{uid}_{i}"}],
                    "tenant_id": "consistency_test",
                },
            }
            for i in range(num_requests)
        ]

        async with httpx.AsyncClient() as client:
            metrics = await run_concurrent_requests(client, requests, max_concurrent=20)

        # 验证所有请求都成功
        assert metrics.successful_requests == num_requests, (
            f"成功请求数不匹配: {metrics.successful_requests}/{num_requests}"
        )

        # 等待数据写入
        await asyncio.sleep(2.0)

        # 查询验证数据一致性
        response = await client.post(
            f"{BASE_URL}/api/v1/memories/search",
            json={"query": f"一致性测试 {uid}", "mode": "keyword", "tenant_id": "consistency_test"},
        )

        if response.status_code == 200:
            data = response.json()
            found_count = len(data.get("results", []))
            print(f"\n数据一致性验证: 写入 {num_requests} 条，找到 {found_count} 条")
            # 应该能找到大部分数据（> 80%）
            assert found_count >= num_requests * 0.8, f"数据丢失过多: {found_count}/{num_requests}"


class TestConcurrentSearches:
    """并发搜索测试"""

    @pytest.mark.asyncio
    async def test_concurrent_searches_10(self):
        """测试 10 并发搜索"""
        queries = ["Python", "JavaScript", "Web", "编程", "测试", "FastAPI", "Docker", "AI", "数据", "搜索"]
        requests = [
            {
                "method": "POST",
                "url": f"{BASE_URL}/api/v1/memories/search",
                "json": {"query": q, "mode": "keyword", "limit": 10},
            }
            for q in queries
        ]

        async with httpx.AsyncClient() as client:
            metrics = await run_concurrent_requests(client, requests, max_concurrent=10)

        assert metrics.failed_requests == 0, f"失败请求: {metrics.failed_requests}"
        assert metrics.avg_response_time < 3.0, f"平均响应时间过长: {metrics.avg_response_time:.2f}s"
        print(f"\n10 并发搜索: {metrics.requests_per_second:.2f} req/s, 平均: {metrics.avg_response_time * 1000:.2f}ms")

    @pytest.mark.asyncio
    async def test_concurrent_searches_30(self):
        """测试 30 并发搜索"""
        queries = [f"搜索测试 {i}" for i in range(30)]
        requests = [
            {
                "method": "POST",
                "url": f"{BASE_URL}/api/v1/memories/search",
                "json": {"query": q, "mode": "keyword", "limit": 5},
            }
            for q in queries
        ]

        async with httpx.AsyncClient() as client:
            metrics = await run_concurrent_requests(client, requests, max_concurrent=30)

        # 允许少量失败（< 10%）
        max_allowed_failures = 3
        assert metrics.failed_requests <= max_allowed_failures, f"失败请求过多: {metrics.failed_requests}"
        assert metrics.avg_response_time < 5.0, f"平均响应时间过长: {metrics.avg_response_time:.2f}s"
        print(
            f"\n30 并发搜索: {metrics.requests_per_second:.2f} req/s, 平均: {metrics.avg_response_time * 1000:.2f}ms, 失败: {metrics.failed_requests}"
        )

    @pytest.mark.asyncio
    async def test_mixed_read_write_workload(self):
        """测试混合读写负载（读 80% + 写 20%）"""
        uid = str(uuid.uuid4())[:8]
        requests = []

        # 80% 读请求
        for i in range(16):
            requests.append(
                {
                    "method": "POST",
                    "url": f"{BASE_URL}/api/v1/memories/search",
                    "json": {"query": f"测试 {i}", "mode": "keyword", "limit": 5},
                }
            )

        # 20% 写请求
        for i in range(4):
            requests.append(
                {
                    "method": "POST",
                    "url": f"{BASE_URL}/api/v1/memories",
                    "json": {
                        "memories": [{"content": f"混合负载测试 {uid} {i}"}],
                        "tenant_id": "mixed_workload",
                    },
                }
            )

        async with httpx.AsyncClient() as client:
            metrics = await run_concurrent_requests(client, requests, max_concurrent=20)

        # 允许少量失败（< 10%）
        max_allowed_failures = 2
        assert metrics.failed_requests <= max_allowed_failures, f"失败请求过多: {metrics.failed_requests}"
        print(
            f"\n混合负载 (读80%写20%): {metrics.requests_per_second:.2f} req/s, 平均: {metrics.avg_response_time * 1000:.2f}ms, 失败: {metrics.failed_requests}"
        )


class TestPerformanceBenchmarks:
    """性能基准测试"""

    @pytest.mark.asyncio
    async def test_single_request_latency(self):
        """测试单请求延迟 < 200ms"""
        async with httpx.AsyncClient() as client:
            start = time.time()
            response = await client.get(f"{BASE_URL}/health")
            duration = time.time() - start

        assert response.status_code == 200
        assert duration < 0.2, f"单请求延迟过高: {duration * 1000:.2f}ms"
        print(f"\n单请求延迟: {duration * 1000:.2f}ms")

    @pytest.mark.asyncio
    async def test_error_rate_under_load(self):
        """测试并发场景下错误率 < 5%"""
        requests = [{"method": "GET", "url": f"{BASE_URL}/health"} for _ in range(50)]

        async with httpx.AsyncClient() as client:
            metrics = await run_concurrent_requests(client, requests, max_concurrent=50)

        error_rate = metrics.failed_requests / metrics.total_requests * 100
        assert error_rate < 5.0, f"错误率过高: {error_rate:.2f}%"
        print(f"\n50 并发错误率: {error_rate:.2f}%")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
