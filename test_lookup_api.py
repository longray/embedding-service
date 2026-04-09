"""测试 Memory Lookup API (BL-CA-34)"""

import asyncio
import httpx


BASE_URL = "http://localhost:17999"


async def test_lookup_api():
    """测试 Memory Lookup API"""
    print("=" * 60)
    print("测试 Memory Lookup API (BL-CA-34)")
    print("=" * 60)

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        # 1. 测试参数不足
        print("\n1. 测试参数不足...")
        try:
            response = await client.get("/api/v1/memories/lookup")
            print(f"   状态码: {response.status_code}")
            if response.status_code == 400:
                print("✅ 正确返回 400 错误")
            else:
                print(f"⚠️ 返回: {response.text}")
        except Exception as e:
            print(f"❌ 请求失败: {e}")

        # 2. 测试 source_id 查询（假设存在）
        print("\n2. 测试 source_id 查询...")
        try:
            response = await client.get(
                "/api/v1/memories/lookup",
                params={"source_id": "test-source-id-001", "tenant_id": "default"},
            )
            print(f"   状态码: {response.status_code}")
            data = response.json()
            if data.get("found"):
                print(f"✅ 找到记忆: {data.get('memory_id')}")
            else:
                print(f"✅ 未找到（符合预期，测试数据不存在）")
        except Exception as e:
            print(f"❌ 请求失败: {e}")

        # 3. 测试 hash 查询
        print("\n3. 测试 hash 查询...")
        try:
            response = await client.get(
                "/api/v1/memories/lookup",
                params={"hash": "d41d8cd98f00b204e9800998ecf8427e", "tenant_id": "default"},
            )
            print(f"   状态码: {response.status_code}")
            data = response.json()
            if data.get("found"):
                print(f"✅ 找到记忆: {data.get('memory_id')}")
            else:
                print(f"✅ 未找到（符合预期，测试数据不存在）")
        except Exception as e:
            print(f"❌ 请求失败: {e}")

        # 4. 测试 file_path + project_id 查询
        print("\n4. 测试 file_path + project_id 查询...")
        try:
            response = await client.get(
                "/api/v1/memories/lookup",
                params={
                    "file_path": "src/utils.ts",
                    "project_id": "test-project",
                    "tenant_id": "default",
                },
            )
            print(f"   状态码: {response.status_code}")
            data = response.json()
            if data.get("found"):
                print(f"✅ 找到记忆: {data.get('memory_id')}")
            else:
                print(f"✅ 未找到（符合预期，测试数据不存在）")
        except Exception as e:
            print(f"❌ 请求失败: {e}")

        # 5. 测试多条返回
        print("\n5. 测试多条返回 (limit=5)...")
        try:
            response = await client.get(
                "/api/v1/memories/lookup",
                params={"source_id": "test-source-id-001", "limit": 5},
            )
            print(f"   状态码: {response.status_code}")
            data = response.json()
            if data.get("found"):
                if "memories" in data:
                    print(f"✅ 返回 {data.get('count', 0)} 条记忆")
                else:
                    print(f"✅ 单条响应格式")
            else:
                print(f"✅ 未找到")
        except Exception as e:
            print(f"❌ 请求失败: {e}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_lookup_api())
