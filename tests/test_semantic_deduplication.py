"""测试语义去重功能"""

import time
import httpx
import pytest


BASE_URL = "http://localhost:17999"
TENANT_ID = "test-semantic-dedup"


@pytest.mark.asyncio
async def test_semantic_deduplication_high_similarity():
    """测试高相似度去重（>= 0.95）"""
    timestamp = int(time.time())
    base_content = f"今天天气很好，阳光明媚 {timestamp}"

    async with httpx.AsyncClient(timeout=30) as client:
        # 第一条：应该成功
        r1 = await client.post(
            f"{BASE_URL}/api/v1/memories",
            json={
                "memories": [{"content": base_content, "type": "test", "tags": ["semantic-test"]}],
                "tenant_id": TENANT_ID,
            },
        )
        result1 = r1.json()
        assert result1["success"] == 1, "第一条记忆应该成功插入"
        assert len(result1["memory_ids"]) == 1

        # 第二条：高度相似（加了"的"），应该被拒绝
        similar_content = f"今天的天气很好，阳光明媚 {timestamp}"
        r2 = await client.post(
            f"{BASE_URL}/api/v1/memories",
            json={
                "memories": [{"content": similar_content, "type": "test", "tags": ["semantic-test"]}],
                "tenant_id": TENANT_ID,
            },
        )
        result2 = r2.json()
        assert result2["success"] == 0, "高度相似的记忆应该被拒绝"
        assert result2["failed"] == 1
        assert len(result2["errors"]) == 1
        assert "Semantic duplicate detected" in result2["errors"][0]
        assert "similarity:" in result2["errors"][0]

        # 提取相似度分数
        error_msg = result2["errors"][0]
        similarity = float(error_msg.split("similarity: ")[1].rstrip(")"))
        assert similarity >= 0.95, f"相似度应该 >= 0.95，实际: {similarity}"


@pytest.mark.asyncio
async def test_semantic_deduplication_medium_similarity():
    """测试中等相似度（< 0.95）应该被接受"""
    timestamp = int(time.time())
    base_content = f"机器学习是人工智能的重要分支 {timestamp}"

    async with httpx.AsyncClient(timeout=30) as client:
        # 第一条
        r1 = await client.post(
            f"{BASE_URL}/api/v1/memories",
            json={
                "memories": [{"content": base_content, "type": "test", "tags": ["semantic-test"]}],
                "tenant_id": TENANT_ID,
            },
        )
        assert r1.json()["success"] == 1

        # 第二条：中等相似度（主题相关但表述不同）
        different_content = f"深度学习在计算机视觉领域应用广泛 {timestamp}"
        r2 = await client.post(
            f"{BASE_URL}/api/v1/memories",
            json={
                "memories": [{"content": different_content, "type": "test", "tags": ["semantic-test"]}],
                "tenant_id": TENANT_ID,
            },
        )
        result2 = r2.json()
        assert result2["success"] == 1, "中等相似度的记忆应该被接受"


@pytest.mark.asyncio
async def test_semantic_deduplication_low_similarity():
    """测试低相似度（完全不同主题）应该被接受"""
    timestamp = int(time.time())

    async with httpx.AsyncClient(timeout=30) as client:
        # 第一条：天气
        r1 = await client.post(
            f"{BASE_URL}/api/v1/memories",
            json={
                "memories": [{"content": f"今天下雨了 {timestamp}", "type": "test", "tags": ["semantic-test"]}],
                "tenant_id": TENANT_ID,
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
                "tenant_id": TENANT_ID,
            },
        )
        assert r2.json()["success"] == 1, "完全不同主题的记忆应该被接受"


@pytest.mark.asyncio
async def test_content_hash_deduplication():
    """测试内容哈希去重（完全相同内容）"""
    timestamp = int(time.time())
    content = f"完全相同的内容 {timestamp}"

    async with httpx.AsyncClient(timeout=30) as client:
        # 第一条
        r1 = await client.post(
            f"{BASE_URL}/api/v1/memories",
            json={
                "memories": [{"content": content, "type": "test", "tags": ["hash-test"]}],
                "tenant_id": TENANT_ID,
            },
        )
        assert r1.json()["success"] == 1

        # 第二条：完全相同
        r2 = await client.post(
            f"{BASE_URL}/api/v1/memories",
            json={
                "memories": [{"content": content, "type": "test", "tags": ["hash-test"]}],
                "tenant_id": TENANT_ID,
            },
        )
        result2 = r2.json()
        assert result2["success"] == 0, "完全相同的内容应该被拒绝"
        assert "Content hash duplicate detected" in result2["errors"][0]


@pytest.mark.asyncio
async def test_batch_deduplication():
    """测试批量上传时的去重"""
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
                "tenant_id": TENANT_ID,
            },
        )
        result = r.json()
        assert result["success"] == 1, "批量上传应该成功1条"
        assert result["failed"] == 1, "批量上传应该失败1条"
        assert "Semantic duplicate detected" in result["errors"][0]
