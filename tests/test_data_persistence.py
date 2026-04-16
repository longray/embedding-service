"""
数据持久化验证测试

验证上传的数据确实写入数据库，且所有字段都正确存储。
这是为了防止 schema 字段缺失导致的数据丢失问题。
"""

import asyncio
import uuid

import httpx
import pytest

WRAPPER_URL = "http://localhost:18008"


@pytest.mark.asyncio
async def test_upload_and_immediate_query():
    """测试上传后立即查询，验证数据确实写入 SurrealDB"""
    uid = str(uuid.uuid4())[:8]

    async with httpx.AsyncClient() as client:
        # 1. 上传记忆
        upload_response = await client.post(
            f"{WRAPPER_URL}/api/v1/memories",
            json={
                "memories": [
                    {
                        "content": f"持久化测试 {uid}",
                        "abstract": f"摘要 {uid}",
                        "overview": f"概览 {uid}",
                        "type": "test",
                        "project_id": "test-project",
                        "metadata": {"test_id": uid},
                    }
                ]
            },
        )

        assert upload_response.status_code == 200
        result = upload_response.json()
        assert result["success"] == 1, f"上传失败: {result}"

        memory_id = result["memory_ids"][0]
        print(f"✅ 上传成功，memory_id: {memory_id}")

        # 2. 立即查询（验证 SurrealDB 写入）
        await asyncio.sleep(0.5)  # 等待写入完成

        query_response = await client.get(f"{WRAPPER_URL}/api/v1/memories/{memory_id}?tenant_id=default")

        assert query_response.status_code == 200, f"查询失败: {query_response.status_code}"

        data = query_response.json()
        assert data["status"] == "success"
        memory = data["memory"]

        # 3. 验证所有字段都存在
        assert memory["content"] == f"持久化测试 {uid}", "content 字段缺失或错误"
        assert memory["abstract"] == f"摘要 {uid}", "abstract 字段缺失或错误"
        assert memory["overview"] == f"概览 {uid}", "overview 字段缺失或错误"
        assert memory["type"] == "test", "type 字段缺失或错误"
        assert memory["project_id"] == "test-project", "project_id 字段缺失或错误"
        assert memory["metadata"]["test_id"] == uid, "metadata 字段缺失或错误"

        print(f"✅ 所有字段验证通过")


@pytest.mark.asyncio
async def test_code_upload_and_query():
    """测试代码分析数据上传后查询，验证代码类型特殊处理"""
    uid = str(uuid.uuid4())[:8]

    async with httpx.AsyncClient() as client:
        # 1. 上传代码分析数据
        upload_response = await client.post(
            f"{WRAPPER_URL}/api/v1/memories",
            json={
                "memories": [
                    {
                        "type": "code",
                        "content": f"// Test {uid}\nfunction test() {{ return 42; }}",
                        "abstract": f"代码摘要 {uid}",
                        "overview": f"代码概览 {uid}",
                        "project_id": "code-test-project",
                        "metadata": {
                            "file_path": f"src/test_{uid}.ts",
                            "language": "typescript",
                        },
                    }
                ]
            },
        )

        assert upload_response.status_code == 200
        result = upload_response.json()
        assert result["success"] == 1, f"上传失败: {result}"

        memory_id = result["memory_ids"][0]
        print(f"✅ 代码上传成功，memory_id: {memory_id}")

        # 2. 查询验证
        await asyncio.sleep(0.5)

        query_response = await client.get(f"{WRAPPER_URL}/api/v1/memories/{memory_id}?tenant_id=default")

        assert query_response.status_code == 200, f"查询失败: {query_response.status_code}"

        data = query_response.json()
        memory = data["memory"]

        # 3. 验证代码类型字段
        assert memory["type"] == "code"
        assert memory["abstract"] == f"代码摘要 {uid}"
        assert memory["overview"] == f"代码概览 {uid}"
        assert memory["project_id"] == "code-test-project"
        assert memory["metadata"]["file_path"] == f"src/test_{uid}.ts"

        print(f"✅ 代码数据验证通过")


@pytest.mark.asyncio
async def test_upload_with_minimal_fields():
    """测试最小字段上传，验证默认值填充"""
    uid = str(uuid.uuid4())[:8]

    async with httpx.AsyncClient() as client:
        # 只提供必需字段
        upload_response = await client.post(
            f"{WRAPPER_URL}/api/v1/memories",
            json={
                "memories": [
                    {
                        "content": f"最小字段测试 {uid}",
                        "abstract": "最小测试",
                    }
                ]
            },
        )

        assert upload_response.status_code == 200
        result = upload_response.json()
        assert result["success"] == 1

        memory_id = result["memory_ids"][0]

        # 查询验证默认值
        await asyncio.sleep(0.5)
        query_response = await client.get(f"{WRAPPER_URL}/api/v1/memories/{memory_id}?tenant_id=default")

        assert query_response.status_code == 200
        memory = query_response.json()["memory"]

        # 验证默认值
        assert memory["type"] == "general"
        assert memory["project_id"] == "global"

        print(f"✅ 最小字段上传验证通过")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
