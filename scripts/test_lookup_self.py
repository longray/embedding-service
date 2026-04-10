"""Lookup API 自测脚本"""

import asyncio
import hashlib
import httpx

BASE_URL = "http://localhost:17999"
TEST_TENANT = "default"
TEST_PROJECT = "self-test-project"


async def test_upload_and_lookup():
    """测试上传和查询完整流程"""
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        # 1. 上传测试数据
        print("=" * 60)
        print("1. 上传测试数据")
        print("=" * 60)

        content = "function testLookup() { return 'hello'; }"
        content_hash = hashlib.md5(content.encode()).hexdigest()
        source_id = f"self-test-source-{int(asyncio.get_event_loop().time())}"
        file_path = "src/self-test.ts"

        upload_result = await client.post(
            "/api/v1/memories",
            json={
                "memories": [
                    {
                        "content": content,
                        "type": "code",
                        "source_id": source_id,
                        "project_id": TEST_PROJECT,
                        "tenant_id": TEST_TENANT,
                        "metadata": {
                            "file_path": file_path,
                            "content_hash": content_hash,
                        },
                    }
                ]
            },
        )

        print(f"上传状态: {upload_result.status_code}")
        upload_data = upload_result.json()
        print(f"上传结果: {upload_data}")

        if upload_result.status_code != 200:
            print("❌ 上传失败")
            return False

        # 等待索引
        await asyncio.sleep(1)

        # 2. 通过 source_id 查询
        print("\n" + "=" * 60)
        print("2. 通过 source_id 查询")
        print("=" * 60)

        lookup_result = await client.get(
            "/api/v1/memories/lookup",
            params={"source_id": source_id, "tenant_id": TEST_TENANT},
        )

        print(f"查询状态: {lookup_result.status_code}")
        lookup_data = lookup_result.json()
        print(f"查询结果: {lookup_data}")

        if not lookup_data.get("found"):
            print("❌ source_id 查询失败")
            return False

        if lookup_data.get("source_id") != source_id:
            print(f"❌ source_id 不匹配: 期望 {source_id}, 实际 {lookup_data.get('source_id')}")
            return False

        print("✅ source_id 查询成功")

        # 3. 通过 file_path + project_id 查询
        print("\n" + "=" * 60)
        print("3. 通过 file_path + project_id 查询")
        print("=" * 60)

        lookup_result = await client.get(
            "/api/v1/memories/lookup",
            params={
                "file_path": file_path,
                "project_id": TEST_PROJECT,
                "tenant_id": TEST_TENANT,
            },
        )

        print(f"查询状态: {lookup_result.status_code}")
        lookup_data = lookup_result.json()
        print(f"查询结果: {lookup_data}")

        if not lookup_data.get("found"):
            print("❌ file_path 查询失败")
            return False

        if lookup_data.get("file_path") != file_path:
            print(f"❌ file_path 不匹配: 期望 {file_path}, 实际 {lookup_data.get('file_path')}")
            return False

        print("✅ file_path 查询成功")

        # 4. 通过 hash 查询
        print("\n" + "=" * 60)
        print("4. 通过 hash 查询")
        print("=" * 60)

        lookup_result = await client.get(
            "/api/v1/memories/lookup",
            params={"hash": content_hash, "tenant_id": TEST_TENANT},
        )

        print(f"查询状态: {lookup_result.status_code}")
        lookup_data = lookup_result.json()
        print(f"查询结果: {lookup_data}")

        if not lookup_data.get("found"):
            print("❌ hash 查询失败")
            return False

        print("✅ hash 查询成功")

        # 5. 测试 tenant_id 隔离
        print("\n" + "=" * 60)
        print("5. 测试 tenant_id 隔离")
        print("=" * 60)

        lookup_result = await client.get(
            "/api/v1/memories/lookup",
            params={"source_id": source_id, "tenant_id": "other-tenant"},
        )

        lookup_data = lookup_result.json()
        if lookup_data.get("found"):
            print("❌ tenant_id 隔离失败：其他租户能查到数据")
            return False

        print("✅ tenant_id 隔离正常")

        # 6. 测试参数不足
        print("\n" + "=" * 60)
        print("6. 测试参数不足")
        print("=" * 60)

        lookup_result = await client.get(
            "/api/v1/memories/lookup",
            params={"tenant_id": TEST_TENANT},
        )

        print(f"查询状态: {lookup_result.status_code}")
        if lookup_result.status_code != 400:
            print("❌ 参数不足时应返回 400")
            return False

        print("✅ 参数验证正常")

        return True


async def main():
    print("Lookup API 自测开始")
    print("=" * 60)

    success = await test_upload_and_lookup()

    print("\n" + "=" * 60)
    if success:
        print("✅ 所有测试通过！可以通知插件端")
    else:
        print("❌ 测试失败，需要修复")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
