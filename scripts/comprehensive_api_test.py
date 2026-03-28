#!/usr/bin/env python3
"""
Wrapper Service v2.4.1 全面 API 测试
覆盖所有 API 端点，包含正常、边界、异常情况
"""

import asyncio
import hashlib
import json
import random
import string
import sys
import time
from typing import Any

import httpx

# 配置
BASE_URL = "http://localhost:17999"
TENANT_ID = "test-tenant-001"
TEST_TIMEOUT = 30.0

# 统计
stats = {
    "total": 0,
    "passed": 0,
    "failed": 0,
    "errors": [],
}


def generate_id() -> str:
    """生成随机 ID"""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=16))


def generate_content() -> str:
    """生成测试内容"""
    topics = ["Python", "FastAPI", "SurrealDB", "Meilisearch", "Docker", "机器学习"]
    return f"这是一个关于{random.choice(topics)}的测试内容，用于API测试。ID: {generate_id()}"


def log_test(name: str, status: str, detail: str = ""):
    """记录测试结果"""
    stats["total"] += 1
    if status == "PASS":
        stats["passed"] += 1
        print(f"✅ {name}")
    else:
        stats["failed"] += 1
        print(f"❌ {name}: {detail}")
        stats["errors"].append({"test": name, "status": status, "detail": detail})


async def test_health():
    """测试健康检查 API"""
    async with httpx.AsyncClient() as client:
        # 正常情况
        try:
            resp = await client.get(f"{BASE_URL}/health", timeout=TEST_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "healthy":
                    log_test("Health Check - Normal", "PASS")
                else:
                    log_test("Health Check - Normal", "FAIL", f"Unexpected status: {data}")
            else:
                log_test("Health Check - Normal", "FAIL", f"Status: {resp.status_code}")
        except Exception as e:
            log_test("Health Check - Normal", "ERROR", str(e))


async def test_embeddings():
    """测试 Embedding API"""
    async with httpx.AsyncClient() as client:
        # 正常情况 - 单文本
        try:
            resp = await client.post(
                f"{BASE_URL}/v1/embeddings",
                json={"input": "测试文本", "model": "Qwen3-Embedding-0.6B"},
                timeout=TEST_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                # Embedding is in data['data'][0]['embedding']
                if data.get("data") and len(data["data"]) > 0 and data["data"][0].get("embedding"):
                    log_test("Embedding - Single Text", "PASS")
                else:
                    log_test("Embedding - Single Text", "FAIL", "No embedding in response")
            else:
                log_test("Embedding - Single Text", "FAIL", f"Status: {resp.status_code}")
        except Exception as e:
            log_test("Embedding - Single Text", "ERROR", str(e))

        # 边界 - 空文本
        try:
            resp = await client.post(f"{BASE_URL}/v1/embeddings", json={"input": ""}, timeout=TEST_TIMEOUT)
            log_test(
                "Embedding - Empty Text",
                "PASS" if resp.status_code in [200, 400, 422] else "FAIL",
                f"Status: {resp.status_code}",
            )
        except Exception as e:
            log_test("Embedding - Empty Text", "ERROR", str(e))

        # 边界 - 超长文本
        try:
            long_text = "测试" * 5000
            resp = await client.post(f"{BASE_URL}/v1/embeddings", json={"input": long_text}, timeout=TEST_TIMEOUT)
            log_test(
                "Embedding - Long Text", "PASS" if resp.status_code == 200 else "FAIL", f"Status: {resp.status_code}"
            )
        except Exception as e:
            log_test("Embedding - Long Text", "ERROR", str(e))


async def test_memories_upload():
    """测试记忆上传 API"""
    async with httpx.AsyncClient() as client:
        # 正常情况 - 单条记忆
        try:
            content = generate_content()
            resp = await client.post(
                f"{BASE_URL}/api/v1/memories",
                json={
                    "memories": [
                        {
                            "content": content,
                            "content_abstract": content[:50],
                            "content_overview": content[:100],
                            "local_id": generate_id(),
                            "type": "test",
                            "tags": ["test", "api"],
                            "project_id": "test-project",
                        }
                    ],
                    "tenant_id": TENANT_ID,
                },
                timeout=TEST_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success", 0) > 0:
                    log_test("Memory Upload - Single", "PASS")
                    return data.get("memory_ids", [None])[0]
                else:
                    log_test("Memory Upload - Single", "FAIL", f"No success: {data}")
            else:
                log_test("Memory Upload - Single", "FAIL", f"Status: {resp.status_code}, Body: {resp.text[:200]}")
        except Exception as e:
            log_test("Memory Upload - Single", "ERROR", str(e))
        return None

        # 批量上传
        try:
            memories = []
            for i in range(5):
                content = generate_content()
                memories.append(
                    {
                        "content": content,
                        "content_abstract": content[:50],
                        "content_overview": content[:100],
                        "local_id": generate_id(),
                        "type": "test",
                        "tags": ["batch", "test"],
                    }
                )
            resp = await client.post(
                f"{BASE_URL}/api/v1/memories", json={"memories": memories, "tenant_id": TENANT_ID}, timeout=TEST_TIMEOUT
            )
            if resp.status_code == 200:
                data = resp.json()
                log_test(
                    "Memory Upload - Batch (5)",
                    "PASS" if data.get("success", 0) == 5 else "FAIL",
                    f"Success: {data.get('success', 0)}/5",
                )
            else:
                log_test("Memory Upload - Batch", "FAIL", f"Status: {resp.status_code}")
        except Exception as e:
            log_test("Memory Upload - Batch", "ERROR", str(e))

        # 错误 - 空内容
        try:
            resp = await client.post(
                f"{BASE_URL}/api/v1/memories",
                json={"memories": [{"content": ""}], "tenant_id": TENANT_ID},
                timeout=TEST_TIMEOUT,
            )
            log_test(
                "Memory Upload - Empty Content",
                "PASS" if resp.status_code in [400, 422] else "FAIL",
                f"Status: {resp.status_code}",
            )
        except Exception as e:
            log_test("Memory Upload - Empty Content", "ERROR", str(e))


async def test_memories_search():
    """测试记忆搜索 API"""
    async with httpx.AsyncClient() as client:
        # 关键词搜索
        try:
            resp = await client.post(
                f"{BASE_URL}/api/v1/memories/search",
                json={"query": "测试", "mode": "keyword", "limit": 10, "tenant_id": TENANT_ID},
                timeout=TEST_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                log_test("Memory Search - Keyword", "PASS", f"Found {len(data.get('results', []))} results")
            else:
                log_test("Memory Search - Keyword", "FAIL", f"Status: {resp.status_code}")
        except Exception as e:
            log_test("Memory Search - Keyword", "ERROR", str(e))

        # 向量搜索
        try:
            resp = await client.post(
                f"{BASE_URL}/api/v1/memories/search",
                json={"query": "测试文本", "mode": "vector", "limit": 5, "threshold": 0.7, "tenant_id": TENANT_ID},
                timeout=TEST_TIMEOUT,
            )
            if resp.status_code == 200:
                log_test("Memory Search - Vector", "PASS")
            else:
                log_test("Memory Search - Vector", "FAIL", f"Status: {resp.status_code}")
        except Exception as e:
            log_test("Memory Search - Vector", "ERROR", str(e))

        # 混合搜索
        try:
            resp = await client.post(
                f"{BASE_URL}/api/v1/memories/search",
                json={"query": "Python FastAPI", "mode": "hybrid", "limit": 10, "tenant_id": TENANT_ID},
                timeout=TEST_TIMEOUT,
            )
            if resp.status_code == 200:
                log_test("Memory Search - Hybrid", "PASS")
            else:
                log_test("Memory Search - Hybrid", "FAIL", f"Status: {resp.status_code}")
        except Exception as e:
            log_test("Memory Search - Hybrid", "ERROR", str(e))


async def test_memories_clear():
    """测试记忆清空 API"""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.delete(
                f"{BASE_URL}/api/v1/memories/clear", headers={"WRAPPER_MEILI_API_KEY": "test-key"}, timeout=TEST_TIMEOUT
            )
            log_test(
                "Memory Clear", "PASS" if resp.status_code in [200, 401, 403] else "FAIL", f"Status: {resp.status_code}"
            )
        except Exception as e:
            log_test("Memory Clear", "ERROR", str(e))


async def test_sync_apis():
    """测试同步 API"""
    async with httpx.AsyncClient() as client:
        # Sync Full
        try:
            memories = []
            for i in range(3):
                content = generate_content()
                memories.append(
                    {
                        "content": content,
                        "content_abstract": content[:50],
                        "content_overview": content[:100],
                        "local_id": generate_id(),
                        "type": "sync-test",
                    }
                )
            resp = await client.post(
                f"{BASE_URL}/api/v1/sync/full",
                json={"memories": memories, "tenant_id": TENANT_ID},
                timeout=TEST_TIMEOUT,
            )
            if resp.status_code == 200:
                log_test("Sync Full", "PASS")
            else:
                log_test("Sync Full", "FAIL", f"Status: {resp.status_code}")
        except Exception as e:
            log_test("Sync Full", "ERROR", str(e))

        # Get Fingerprints
        try:
            resp = await client.get(
                f"{BASE_URL}/api/v1/sync/fingerprints", params={"tenant_id": TENANT_ID}, timeout=TEST_TIMEOUT
            )
            if resp.status_code == 200:
                data = resp.json()
                log_test("Sync Fingerprints", "PASS", f"Count: {data.get('count', 0)}")
            else:
                log_test("Sync Fingerprints", "FAIL", f"Status: {resp.status_code}")
        except Exception as e:
            log_test("Sync Fingerprints", "ERROR", str(e))

        # Sync Preview
        try:
            resp = await client.post(
                f"{BASE_URL}/api/v1/sync/preview",
                json={
                    "fingerprints": [
                        {
                            "source_id": f"test-{i}",
                            "hash": generate_id(),
                            "mtime": int(time.time()),
                            "path": f"test-{i}.md",
                        }
                        for i in range(3)
                    ],
                    "tenant_id": TENANT_ID,
                },
                timeout=TEST_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                log_test(
                    "Sync Preview",
                    "PASS",
                    f"Upload: {len(data.get('to_upload', []))}, Conflicts: {len(data.get('conflicts', []))}",
                )
            else:
                log_test("Sync Preview", "FAIL", f"Status: {resp.status_code}")
        except Exception as e:
            log_test("Sync Preview", "ERROR", str(e))


async def test_cache_apis():
    """测试缓存 API"""
    async with httpx.AsyncClient() as client:
        # Cache Stats
        try:
            resp = await client.get(f"{BASE_URL}/api/v1/cache/stats", timeout=TEST_TIMEOUT)
            if resp.status_code == 200:
                log_test("Cache Stats", "PASS")
            else:
                log_test("Cache Stats", "FAIL", f"Status: {resp.status_code}")
        except Exception as e:
            log_test("Cache Stats", "ERROR", str(e))

        # Cache Warmup
        try:
            resp = await client.post(
                f"{BASE_URL}/api/v1/cache/warmup", json={"tenant_id": TENANT_ID, "limit": 10}, timeout=TEST_TIMEOUT
            )
            log_test("Cache Warmup", "PASS" if resp.status_code == 200 else "FAIL", f"Status: {resp.status_code}")
        except Exception as e:
            log_test("Cache Warmup", "ERROR", str(e))


async def test_hnsw_apis():
    """测试 HNSW API"""
    async with httpx.AsyncClient() as client:
        # HNSW Stats
        try:
            resp = await client.get(
                f"{BASE_URL}/api/v1/hnsw/stats", params={"tenant_id": TENANT_ID}, timeout=TEST_TIMEOUT
            )
            log_test("HNSW Stats", "PASS" if resp.status_code == 200 else "FAIL", f"Status: {resp.status_code}")
        except Exception as e:
            log_test("HNSW Stats", "ERROR", str(e))

        # HNSW Optimize
        try:
            resp = await client.post(
                f"{BASE_URL}/api/v1/hnsw/optimize",
                json={"tenant_id": TENANT_ID, "target_recall": 0.95},
                timeout=TEST_TIMEOUT,
            )
            log_test("HNSW Optimize", "PASS" if resp.status_code == 200 else "FAIL", f"Status: {resp.status_code}")
        except Exception as e:
            log_test("HNSW Optimize", "ERROR", str(e))


async def test_access_log():
    """测试访问日志 API"""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{BASE_URL}/api/v1/access-log",
                json={
                    "entries": [
                        {
                            "entry_id": generate_id(),
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "type": "read",
                        }
                    ],
                    "tenant_id": TENANT_ID,
                },
                timeout=TEST_TIMEOUT,
            )
            log_test("Access Log", "PASS" if resp.status_code == 200 else "FAIL", f"Status: {resp.status_code}")
        except Exception as e:
            log_test("Access Log", "ERROR", str(e))


async def test_graph_apis():
    """测试图关系 API"""
    async with httpx.AsyncClient() as client:
        # 先创建两条记忆用于测试关系
        memory_ids = []
        try:
            for i in range(2):
                # 使用唯一内容避免去重
                content = f"Graph test memory {i} at {time.time()} - {generate_id()}"
                resp = await client.post(
                    f"{BASE_URL}/api/v1/memories",
                    json={
                        "memories": [
                            {
                                "content": content,
                                "content_abstract": content[:50],
                                "content_overview": content[:100],
                                "local_id": f"graph-test-{i}-{int(time.time())}",
                                "type": "graph-test",
                                "tags": ["graph", "test"],
                            }
                        ],
                        "tenant_id": TENANT_ID,
                    },
                    timeout=TEST_TIMEOUT,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    # 检查是否有成功创建或跳过的记忆
                    if data.get("memory_ids"):
                        memory_ids.extend(data["memory_ids"])
                    elif data.get("skipped"):
                        # 如果被跳过，使用已存在的记忆ID
                        for skipped in data["skipped"]:
                            if skipped.get("existing_id"):
                                memory_ids.append(skipped["existing_id"])
        except Exception as e:
            log_test("Graph - Create Memories", "ERROR", str(e))
            return

        if len(memory_ids) < 2:
            log_test("Graph - Create Memories", "FAIL", f"Not enough memories created: {memory_ids}")
            return

        log_test("Graph - Create Memories", "PASS", f"Created {len(memory_ids)} memories")

        # 创建关系
        try:
            resp = await client.post(
                f"{BASE_URL}/api/v1/memories/relations",
                json={
                    "from_id": memory_ids[0],
                    "to_id": memory_ids[1],
                    "relationship_type": "related",
                    "weight": 0.8,
                    "tenant_id": TENANT_ID,
                },
                timeout=TEST_TIMEOUT,
            )
            log_test(
                "Graph - Create Relations", "PASS" if resp.status_code == 200 else "FAIL", f"Status: {resp.status_code}"
            )
        except Exception as e:
            log_test("Graph - Create Relations", "ERROR", str(e))

        # 查询图关系
        try:
            resp = await client.post(
                f"{BASE_URL}/api/v1/memories/{memory_ids[0]}/graph",
                json={
                    "depth": 1,
                    "tenant_id": TENANT_ID,
                },
                timeout=TEST_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                nodes = len(data.get("nodes", []))
                edges = len(data.get("edges", []))
                log_test("Graph - Query Graph", "PASS", f"Nodes: {nodes}, Edges: {edges}")
            else:
                log_test("Graph - Query Graph", "FAIL", f"Status: {resp.status_code}")
        except Exception as e:
            log_test("Graph - Query Graph", "ERROR", str(e))


async def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Wrapper Service v2.4.1 - 全面 API 测试")
    print("=" * 60)
    print()

    start_time = time.time()

    await test_health()
    await test_embeddings()
    await test_memories_upload()
    await test_memories_search()
    await test_memories_clear()
    await test_sync_apis()
    await test_cache_apis()
    await test_hnsw_apis()
    await test_graph_apis()
    await test_access_log()

    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print("测试完成")
    print("=" * 60)
    print(f"总测试数: {stats['total']}")
    print(f"通过: {stats['passed']}")
    print(f"失败: {stats['failed']}")
    print(f"成功率: {stats['passed'] / stats['total'] * 100:.1f}%")
    print(f"耗时: {elapsed:.2f}s")
    print()

    if stats["errors"]:
        print("错误详情:")
        for error in stats["errors"][:5]:
            print(f"  - {error['test']}: {error['detail'][:100]}")
        if len(stats["errors"]) > 5:
            print(f"  ... 还有 {len(stats['errors']) - 5} 个错误")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
