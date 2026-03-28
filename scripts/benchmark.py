#!/usr/bin/env python3
"""
Embedding Service 性能基准测试

测试范围：
- Embedding 生成延迟（单条/批量/长文本）
- 记忆搜索延迟（keyword/vector/hybrid）
- 记忆上传延迟
- 同步预览延迟
- 端到端延迟分解

使用方式：
    uv run python scripts/benchmark.py
    uv run python scripts/benchmark.py --iterations 10
    uv run python scripts/benchmark.py --url http://localhost:17999
"""

import argparse
import asyncio
import json
import random
import string
import time
from dataclasses import dataclass, field

import httpx

BASE_URL = "http://localhost:17999"
TENANT_ID = "bench-tenant"
TIMEOUT = 60.0


@dataclass
class Metric:
    name: str
    values: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def avg_ms(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0

    @property
    def min_ms(self) -> float:
        return min(self.values) if self.values else 0

    @property
    def max_ms(self) -> float:
        return max(self.values) if self.values else 0

    @property
    def p50_ms(self) -> float:
        return self._percentile(50)

    @property
    def p95_ms(self) -> float:
        return self._percentile(95)

    @property
    def p99_ms(self) -> float:
        return self._percentile(99)

    def _percentile(self, p: int) -> float:
        if not self.values:
            return 0
        sorted_vals = sorted(self.values)
        idx = int(len(sorted_vals) * p / 100)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]


def generate_id() -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=12))


def generate_content(topic: str = "") -> str:
    topics = ["Python 性能优化", "FastAPI 异步编程", "SurrealDB 查询优化", "向量搜索原理", "Meilisearch 配置"]
    t = topic or random.choice(topics)
    return f"{t} - 这是一条基准测试数据，包含一些随机内容 {generate_id()}"


def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_metric(m: Metric):
    print(f"  {m.name}:")
    print(f"    请求次数: {m.count}")
    print(f"    平均延迟: {m.avg_ms:.1f}ms")
    print(f"    P50: {m.p50_ms:.1f}ms  P95: {m.p95_ms:.1f}ms  P99: {m.p99_ms:.1f}ms")
    print(f"    最小: {m.min_ms:.1f}ms  最大: {m.max_ms:.1f}ms")


async def bench_embeddings(client: httpx.AsyncClient, iterations: int) -> list[Metric]:
    print_section("Embedding 生成延迟")

    single = Metric("单文本 Embedding")
    batch_10 = Metric("批量 Embedding (10条)")
    long_text = Metric("长文本 Embedding (~10K字符)")

    for _ in range(iterations):
        # 单文本
        start = time.perf_counter()
        resp = await client.post(
            f"{BASE_URL}/v1/embeddings",
            json={"input": f"测试文本 {generate_id()}", "model": "Qwen3-Embedding-0.6B"},
            timeout=TIMEOUT,
        )
        single.values.append((time.perf_counter() - start) * 1000)
        assert resp.status_code == 200, f"单文本失败: {resp.status_code}"

        # 批量 10（逐条调用，wrapper 只支持单条输入）
        texts = [f"批量文本 {i} {generate_id()}" for i in range(10)]
        start = time.perf_counter()
        for text in texts:
            resp = await client.post(
                f"{BASE_URL}/v1/embeddings",
                json={"input": text, "model": "Qwen3-Embedding-0.6B"},
                timeout=TIMEOUT,
            )
        batch_10.values.append((time.perf_counter() - start) * 1000)

        # 长文本
        long = "这是一段长文本，用于测试长文本的 embedding 性能。" * 200
        start = time.perf_counter()
        resp = await client.post(
            f"{BASE_URL}/v1/embeddings",
            json={"input": long, "model": "Qwen3-Embedding-0.6B"},
            timeout=TIMEOUT,
        )
        long_text.values.append((time.perf_counter() - start) * 1000)
        if resp.status_code != 200:
            print(f"    ⚠️ 长文本 embedding 返回 {resp.status_code}")

    for m in [single, batch_10, long_text]:
        print_metric(m)

    return [single, batch_10, long_text]


async def bench_search(client: httpx.AsyncClient, iterations: int) -> list[Metric]:
    print_section("记忆搜索延迟")

    keyword = Metric("关键词搜索 (keyword)")
    vector = Metric("向量搜索 (vector)")
    hybrid = Metric("混合搜索 (hybrid)")

    queries = ["Python", "FastAPI", "向量搜索", "性能优化", "数据库配置"]

    for i in range(iterations):
        query = queries[i % len(queries)]

        # keyword
        start = time.perf_counter()
        resp = await client.post(
            f"{BASE_URL}/api/v1/memories/search",
            json={"query": query, "mode": "keyword", "limit": 10, "tenant_id": TENANT_ID},
            timeout=TIMEOUT,
        )
        keyword.values.append((time.perf_counter() - start) * 1000)

        # vector
        start = time.perf_counter()
        resp = await client.post(
            f"{BASE_URL}/api/v1/memories/search",
            json={"query": query, "mode": "vector", "limit": 10, "threshold": 0.7, "tenant_id": TENANT_ID},
            timeout=TIMEOUT,
        )
        vector.values.append((time.perf_counter() - start) * 1000)

        # hybrid
        start = time.perf_counter()
        resp = await client.post(
            f"{BASE_URL}/api/v1/memories/search",
            json={"query": query, "mode": "hybrid", "limit": 10, "tenant_id": TENANT_ID},
            timeout=TIMEOUT,
        )
        hybrid.values.append((time.perf_counter() - start) * 1000)

    for m in [keyword, vector, hybrid]:
        print_metric(m)

    return [keyword, vector, hybrid]


async def bench_upload(client: httpx.AsyncClient, iterations: int) -> list[Metric]:
    print_section("记忆上传延迟")

    single = Metric("单条上传")
    batch_5 = Metric("批量上传 (5条)")

    for _ in range(iterations):
        # 单条
        content = generate_content()
        start = time.perf_counter()
        resp = await client.post(
            f"{BASE_URL}/api/v1/memories",
            json={
                "memories": [
                    {
                        "content": content,
                        "content_abstract": content[:50],
                        "content_overview": content[:100],
                        "local_id": generate_id(),
                        "type": "benchmark",
                        "tags": ["bench"],
                    }
                ],
                "tenant_id": TENANT_ID,
            },
            timeout=TIMEOUT,
        )
        single.values.append((time.perf_counter() - start) * 1000)

        # 批量 5
        memories = []
        for _ in range(5):
            c = generate_content()
            memories.append(
                {
                    "content": c,
                    "content_abstract": c[:50],
                    "content_overview": c[:100],
                    "local_id": generate_id(),
                    "type": "benchmark",
                    "tags": ["bench", "batch"],
                }
            )
        start = time.perf_counter()
        resp = await client.post(
            f"{BASE_URL}/api/v1/memories",
            json={"memories": memories, "tenant_id": TENANT_ID},
            timeout=TIMEOUT,
        )
        batch_5.values.append((time.perf_counter() - start) * 1000)

    for m in [single, batch_5]:
        print_metric(m)

    return [single, batch_5]


async def bench_sync(client: httpx.AsyncClient, iterations: int) -> list[Metric]:
    print_section("同步操作延迟")

    fingerprints_get = Metric("获取指纹 (GET)")
    preview_10 = Metric("同步预览 (10条)")
    preview_100 = Metric("同步预览 (100条)")

    for _ in range(iterations):
        # GET fingerprints
        start = time.perf_counter()
        resp = await client.get(
            f"{BASE_URL}/api/v1/sync/fingerprints",
            params={"tenant_id": TENANT_ID},
            timeout=TIMEOUT,
        )
        fingerprints_get.values.append((time.perf_counter() - start) * 1000)

        # sync preview 10
        fps = [
            {
                "source_id": generate_id(),
                "hash": generate_id(),
                "mtime": int(time.time()),
                "path": f"bench/{generate_id()}.md",
            }
            for _ in range(10)
        ]
        start = time.perf_counter()
        resp = await client.post(
            f"{BASE_URL}/api/v1/sync/preview",
            json={"fingerprints": fps, "tenant_id": TENANT_ID},
            timeout=TIMEOUT,
        )
        preview_10.values.append((time.perf_counter() - start) * 1000)

        # sync preview 100
        fps100 = [
            {
                "source_id": generate_id(),
                "hash": generate_id(),
                "mtime": int(time.time()),
                "path": f"bench/{generate_id()}.md",
            }
            for _ in range(100)
        ]
        start = time.perf_counter()
        resp = await client.post(
            f"{BASE_URL}/api/v1/sync/preview",
            json={"fingerprints": fps100, "tenant_id": TENANT_ID},
            timeout=TIMEOUT,
        )
        preview_100.values.append((time.perf_counter() - start) * 1000)

    for m in [fingerprints_get, preview_10, preview_100]:
        print_metric(m)

    return [fingerprints_get, preview_10, preview_100]


async def bench_e2e(client: httpx.AsyncClient, iterations: int) -> list[Metric]:
    print_section("端到端延迟分解（上传 → 搜索 → 验证）")

    e2e_total = Metric("E2E 完整流程")

    for _ in range(iterations):
        start = time.perf_counter()

        # Step 1: Upload
        content = f"E2E基准测试 {generate_id()} - 包含唯一内容"
        resp = await client.post(
            f"{BASE_URL}/api/v1/memories",
            json={
                "memories": [
                    {
                        "content": content,
                        "content_abstract": content[:50],
                        "content_overview": content[:100],
                        "local_id": generate_id(),
                        "type": "benchmark",
                        "tags": ["e2e"],
                    }
                ],
                "tenant_id": TENANT_ID,
            },
            timeout=TIMEOUT,
        )
        upload_ms = (time.perf_counter() - start) * 1000

        # Step 2: Search
        search_start = time.perf_counter()
        resp = await client.post(
            f"{BASE_URL}/api/v1/memories/search",
            json={"query": content[:20], "mode": "hybrid", "limit": 5, "tenant_id": TENANT_ID},
            timeout=TIMEOUT,
        )
        search_ms = (time.perf_counter() - search_start) * 1000

        total_ms = (time.perf_counter() - start) * 1000
        e2e_total.values.append(total_ms)

    print_metric(e2e_total)

    return [e2e_total]


async def run_benchmark(url: str, iterations: int):
    global BASE_URL
    BASE_URL = url

    print("=" * 60)
    print(f"  Embedding Service 性能基准测试")
    print(f"  目标: {BASE_URL}")
    print(f"  迭代次数: {iterations}")
    print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_metrics: list[Metric] = []
    total_start = time.perf_counter()

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        # 检查服务可用性
        try:
            resp = await client.get(f"{BASE_URL}/health", timeout=5.0)
            if resp.status_code != 200:
                print(f"❌ 服务不可用: {resp.status_code}")
                return
            print(f"✅ 服务健康: {resp.json().get('status')}")
        except Exception as e:
            print(f"❌ 无法连接: {e}")
            return

        all_metrics.extend(await bench_embeddings(client, iterations))
        all_metrics.extend(await bench_search(client, iterations))
        all_metrics.extend(await bench_upload(client, iterations))
        all_metrics.extend(await bench_sync(client, iterations))
        all_metrics.extend(await bench_e2e(client, iterations))

    total_elapsed = (time.perf_counter() - total_start) * 1000

    # 汇总
    print_section("汇总")
    print(f"  总耗时: {total_elapsed / 1000:.1f}s")
    print(f"  测试项: {len(all_metrics)}")
    print()
    print(f"  {'指标':<30} {'平均':>8} {'P50':>8} {'P95':>8} {'P99':>8}")
    print(f"  {'-' * 30} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
    for m in all_metrics:
        print(f"  {m.name:<30} {m.avg_ms:>7.1f}ms {m.p50_ms:>7.1f}ms {m.p95_ms:>7.1f}ms {m.p99_ms:>7.1f}ms")

    # 输出 JSON 格式结果
    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "url": BASE_URL,
        "iterations": iterations,
        "total_time_ms": round(total_elapsed, 1),
        "metrics": {
            m.name: {
                "count": m.count,
                "avg_ms": round(m.avg_ms, 1),
                "p50_ms": round(m.p50_ms, 1),
                "p95_ms": round(m.p95_ms, 1),
                "p99_ms": round(m.p99_ms, 1),
                "min_ms": round(m.min_ms, 1),
                "max_ms": round(m.max_ms, 1),
            }
            for m in all_metrics
        },
    }

    output_file = f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Embedding Service 性能基准测试")
    parser.add_argument("--url", default="http://localhost:17999", help="服务地址")
    parser.add_argument("--iterations", type=int, default=5, help="每项测试迭代次数")
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.url, args.iterations))


if __name__ == "__main__":
    main()
