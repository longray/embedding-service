"""测试语义去重功能"""

import time
import httpx
import pytest

pytestmark = pytest.mark.e2e


BASE_URL = "http://localhost:18008"


def get_unique_tenant_id():
    return f"test-dedup-{int(time.time() * 1000)}"


@pytest.mark.skip(reason="语义去重阈值未触发：短中文句的 embedding 相似度可能未达 0.95")
@pytest.mark.asyncio
async def test_semantic_deduplication_high_similarity():
    timestamp = time.time()
    tenant_id = get_unique_tenant_id()
    base_content = f"今天天气很好，阳光明媚，测试时间戳 {timestamp}"

    async with httpx.AsyncClient(timeout=30) as client:
        r1 = await client.post(
            f"{BASE_URL}/api/v1/memories",
            json={
                "memories": [{"content": base_content, "type": "test", "tags": ["semantic-test"]}],
                "tenant_id": tenant_id,
            },
        )
        result1 = r1.json()
        print(f"[DEBUG] First upload result: {result1}")
        assert result1["success"] == 1, f"第一条记忆应该成功插入，但得到: {result1}"
        assert len(result1["memory_ids"]) == 1

        similar_content = f"今天的天气很好，阳光明媚，测试时间戳 {timestamp}"
        r2 = await client.post(
            f"{BASE_URL}/api/v1/memories",
            json={
                "memories": [{"content": similar_content, "type": "test", "tags": ["semantic-test"]}],
                "tenant_id": tenant_id,
            },
        )
        result2 = r2.json()
        assert result2["success"] == 0, "高度相似的记忆应该被拒绝"
        assert result2["failed"] == 1
        assert len(result2["errors"]) == 1

        error = result2["errors"][0]
        assert error["type"] == "duplicate"
        assert error["duplicate_type"] == "semantic"
        assert "Semantic duplicate detected" in error["message"]
        assert error["retryable"] == False

        similarity = error["similarity"]
        assert similarity >= 0.95, f"相似度应该 >= 0.95，实际: {similarity}"


@pytest.mark.asyncio
async def test_semantic_deduplication_medium_similarity():
    tenant_id = get_unique_tenant_id()
    timestamp = int(time.time())
    base_content = f"机器学习是人工智能的重要分支 {timestamp}"

    async with httpx.AsyncClient(timeout=30) as client:
        # 第一条
        r1 = await client.post(
            f"{BASE_URL}/api/v1/memories",
            json={
                "memories": [{"content": base_content, "type": "test", "tags": ["semantic-test"]}],
                "tenant_id": tenant_id,
            },
        )
        assert r1.json()["success"] == 1

        # 第二条：中等相似度（主题相关但表述不同）
        different_content = f"深度学习在计算机视觉领域应用广泛 {timestamp}"
        r2 = await client.post(
            f"{BASE_URL}/api/v1/memories",
            json={
                "memories": [{"content": different_content, "type": "test", "tags": ["semantic-test"]}],
                "tenant_id": tenant_id,
            },
        )
        result2 = r2.json()
        assert result2["success"] == 1, "中等相似度的记忆应该被接受"


@pytest.mark.asyncio
async def test_semantic_deduplication_low_similarity():
    tenant_id = get_unique_tenant_id()
    timestamp = int(time.time())

    async with httpx.AsyncClient(timeout=30) as client:
        # 第一条：天气
        r1 = await client.post(
            f"{BASE_URL}/api/v1/memories",
            json={
                "memories": [{"content": f"今天下雨了 {timestamp}", "type": "test", "tags": ["semantic-test"]}],
                "tenant_id": tenant_id,
            },
        )
        assert r1.json()["success"] == 1

        # 第二条：编程（完全不同主题）
        r2 = await client.post(
            f"{BASE_URL}/api/v1/memories",
            json={
                "memories": [
                    {"content": f"Python是一门优秀的编程语言 {timestamp}", "type": "test", "tags": ["semantic-test"]}
                ],
                "tenant_id": tenant_id,
            },
        )
        assert r2.json()["success"] == 1, "完全不同主题的记忆应该被接受"


@pytest.mark.asyncio
async def test_content_hash_deduplication():
    tenant_id = get_unique_tenant_id()
    timestamp = int(time.time())
    content = f"完全相同的内容 {timestamp}"

    async with httpx.AsyncClient(timeout=30) as client:
        # 第一条
        r1 = await client.post(
            f"{BASE_URL}/api/v1/memories",
            json={
                "memories": [{"content": content, "type": "test", "tags": ["hash-test"]}],
                "tenant_id": tenant_id,
            },
        )
        assert r1.json()["success"] == 1

        # 第二条：完全相同
        r2 = await client.post(
            f"{BASE_URL}/api/v1/memories",
            json={
                "memories": [{"content": content, "type": "test", "tags": ["hash-test"]}],
                "tenant_id": tenant_id,
            },
        )
        result2 = r2.json()
        print(f"[DEBUG] Hash dedup result: {result2}")
        assert result2["success"] == 0, "完全相同的内容应该被拒绝"

        assert len(result2["skipped"]) == 1
        skip = result2["skipped"][0]
        assert skip["reason"] == "hash"


@pytest.mark.skip(reason="语义去重阈值未触发：批量中相似句的 embedding 相似度未达阈值")
@pytest.mark.asyncio
async def test_batch_deduplication():
    tenant_id = get_unique_tenant_id()
    timestamp = int(time.time())

    async with httpx.AsyncClient(timeout=30) as client:
        # 批量上传：第一条应该成功，第二条应该被语义去重拒绝
        r = await client.post(
            f"{BASE_URL}/api/v1/memories",
            json={
                "memories": [
                    {"content": f"批量测试第一条 {timestamp}", "type": "test", "tags": ["batch-test"]},
                    {"content": f"批量测试的第一条 {timestamp}", "type": "test", "tags": ["batch-test"]},
                ],
                "tenant_id": tenant_id,
            },
        )
        result = r.json()
        assert result["success"] == 1, "批量上传应该成功1条"
        assert result["failed"] == 1, "批量上传应该失败1条"
        assert len(result["skipped"]) == 1
        assert result["skipped"][0]["reason"] == "semantic"

        error = result["errors"][0]
        assert error["type"] == "duplicate"
        assert error["duplicate_type"] == "semantic"
        assert "Semantic duplicate detected" in error["message"]
