"""Memory Lookup API 真实数据测试

创建测试数据并验证 lookup API 功能。
"""

import asyncio
import hashlib
import httpx
from datetime import datetime


BASE_URL = "http://localhost:18008"


async def create_test_memories():
    """创建测试记忆数据"""
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        print("=" * 60)
        print("创建测试记忆数据")
        print("=" * 60)

        test_memories = [
            {
                "content": "function add(a, b) { return a + b; }",
                "source_id": "01H1ABC1234567890ABCDEF123456",
                "file_path": "src/utils/math.ts",
                "project_id": "test-project-001",
                "type": "code",
            },
            {
                "content": "function subtract(a, b) { return a - b; }",
                "source_id": "01H2DEF2345678901BCDEF2345678",
                "file_path": "src/utils/math.ts",
                "project_id": "test-project-001",
                "type": "code",
            },
            {
                "content": "class User { constructor(name) { this.name = name; } }",
                "source_id": "01H3GHI3456789012CDEF34567890",
                "file_path": "src/models/user.ts",
                "project_id": "test-project-001",
                "type": "code",
            },
        ]

        created_memories = []

        for i, mem in enumerate(test_memories):
            content_hash = hashlib.md5(mem["content"].encode()).hexdigest()

            response = await client.post(
                "/api/v1/memories",
                json={
                    "memories": [
                        {
                            "content": mem["content"],
                            "type": mem["type"],
                            "project_id": mem["project_id"],
                            "source_id": mem["source_id"],
                            "metadata": {
                                "file_path": mem["file_path"],
                                "content_hash": content_hash,
                                "language": "javascript",
                            },
                            "tenant_id": "default",
                        }
                    ]
                },
            )

            if response.status_code == 200:
                data = response.json()
                memory_id = data.get("ids", [None])[0] if data.get("ids") else None
                print(f"✅ 创建记忆 {i + 1}: {memory_id}")
                print(f"   source_id: {mem['source_id']}")
                print(f"   content_hash: {content_hash}")
                print(f"   file_path: {mem['file_path']}")
                created_memories.append(
                    {
                        "memory_id": memory_id,
                        "source_id": mem["source_id"],
                        "content_hash": content_hash,
                        "file_path": mem["file_path"],
                        "project_id": mem["project_id"],
                    }
                )
            else:
                print(f"❌ 创建失败: {response.text}")

        print(f"\n共创建 {len(created_memories)} 条测试记忆")
        return created_memories


async def test_lookup_by_source_id(created_memories):
    """测试 source_id 查询"""
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        print("\n" + "=" * 60)
        print("测试 1: source_id 查询")
        print("=" * 60)

        if not created_memories:
            print("⚠️ 没有测试数据，跳过")
            return

        test_memory = created_memories[0]
        source_id = test_memory["source_id"]

        print(f"\n查询 source_id: {source_id}")

        response = await client.get(
            "/api/v1/memories/lookup",
            params={"source_id": source_id, "tenant_id": "default"},
        )

        print(f"状态码: {response.status_code}")
        data = response.json()

        if data.get("found"):
            print(f"✅ 找到记忆:")
            print(f"   memory_id: {data.get('memory_id')}")
            print(f"   source_id: {data.get('source_id')}")
            print(f"   file_path: {data.get('file_path')}")
            print(f"   project_id: {data.get('project_id')}")

            # 验证返回的数据是否正确
            if data.get("source_id") == source_id:
                print("✅ source_id 匹配正确")
            else:
                print(f"❌ source_id 不匹配: 期望 {source_id}, 实际 {data.get('source_id')}")
        else:
            print(f"❌ 未找到记忆: {data.get('message')}")


async def test_lookup_by_hash(created_memories):
    """测试 hash 查询"""
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        print("\n" + "=" * 60)
        print("测试 2: hash 查询")
        print("=" * 60)

        if not created_memories:
            print("⚠️ 没有测试数据，跳过")
            return

        test_memory = created_memories[1]
        content_hash = test_memory["content_hash"]

        print(f"\n查询 hash: {content_hash}")

        response = await client.get(
            "/api/v1/memories/lookup",
            params={"hash": content_hash, "tenant_id": "default"},
        )

        print(f"状态码: {response.status_code}")
        data = response.json()

        if data.get("found"):
            print(f"✅ 找到记忆:")
            print(f"   memory_id: {data.get('memory_id')}")
            print(f"   content_hash: {data.get('content_hash')}")

            if data.get("content_hash") == content_hash:
                print("✅ hash 匹配正确")
            else:
                print(f"❌ hash 不匹配")
        else:
            print(f"❌ 未找到记忆: {data.get('message')}")


async def test_lookup_by_file_path(created_memories):
    """测试 file_path + project_id 查询"""
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        print("\n" + "=" * 60)
        print("测试 3: file_path + project_id 查询")
        print("=" * 60)

        if not created_memories:
            print("⚠️ 没有测试数据，跳过")
            return

        test_memory = created_memories[0]
        file_path = test_memory["file_path"]
        project_id = test_memory["project_id"]

        print(f"\n查询 file_path: {file_path}")
        print(f"     project_id: {project_id}")

        response = await client.get(
            "/api/v1/memories/lookup",
            params={
                "file_path": file_path,
                "project_id": project_id,
                "tenant_id": "default",
            },
        )

        print(f"状态码: {response.status_code}")
        data = response.json()

        if data.get("found"):
            print(f"✅ 找到记忆:")
            print(f"   memory_id: {data.get('memory_id')}")
            print(f"   file_path: {data.get('file_path')}")
            print(f"   project_id: {data.get('project_id')}")

            if data.get("file_path") == file_path:
                print("✅ file_path 匹配正确")
            else:
                print(f"❌ file_path 不匹配")
        else:
            print(f"❌ 未找到记忆: {data.get('message')}")


async def test_lookup_multiple(created_memories):
    """测试返回多条记录"""
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        print("\n" + "=" * 60)
        print("测试 4: 返回多条记录")
        print("=" * 60)

        if len(created_memories) < 2:
            print("⚠️ 测试数据不足，跳过")
            return

        # 使用相同的 file_path 查询，应该返回多条
        file_path = "src/utils/math.ts"
        project_id = "test-project-001"

        print(f"\n查询 file_path: {file_path} (limit=10)")

        response = await client.get(
            "/api/v1/memories/lookup",
            params={
                "file_path": file_path,
                "project_id": project_id,
                "tenant_id": "default",
                "limit": 10,
            },
        )

        print(f"状态码: {response.status_code}")
        data = response.json()

        if data.get("found"):
            if "memories" in data:
                print(f"✅ 找到 {data.get('count', 0)} 条记忆:")
                for i, mem in enumerate(data["memories"][:3]):  # 只显示前3条
                    print(f"   {i + 1}. {mem.get('memory_id')} - {mem.get('file_path')}")

                if data.get("count", 0) >= 2:
                    print("✅ 正确返回多条记录")
                else:
                    print("⚠️ 返回记录数较少（可能数据未完全创建）")
            else:
                print("⚠️ 单条响应格式（可能只有一条匹配）")
        else:
            print(f"❌ 未找到记忆: {data.get('message')}")


async def cleanup_test_memories(created_memories):
    """清理测试数据"""
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        print("\n" + "=" * 60)
        print("清理测试数据")
        print("=" * 60)

        for mem in created_memories:
            memory_id_raw = mem.get("memory_id") or ""
            memory_id = memory_id_raw.replace("memory:", "") if memory_id_raw else ""
            if memory_id:
                try:
                    response = await client.delete(f"/api/v1/memories/{memory_id}")
                    if response.status_code == 200:
                        print(f"✅ 删除记忆: {memory_id}")
                    else:
                        print(f"⚠️ 删除失败: {memory_id} - {response.status_code}")
                except Exception as e:
                    print(f"⚠️ 删除异常: {memory_id} - {e}")


async def main():
    """主函数"""
    print("=" * 60)
    print("Memory Lookup API 真实数据测试")
    print("=" * 60)

    created_memories = []

    try:
        # 创建测试数据
        created_memories = await create_test_memories()

        # 等待数据索引（给 Meilisearch 一点时间）
        print("\n等待数据索引...")
        await asyncio.sleep(2)

        # 运行测试
        await test_lookup_by_source_id(created_memories)
        await test_lookup_by_hash(created_memories)
        await test_lookup_by_file_path(created_memories)
        await test_lookup_multiple(created_memories)

    finally:
        # 清理测试数据
        if created_memories:
            await cleanup_test_memories(created_memories)

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
