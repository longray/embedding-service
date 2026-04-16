#!/usr/bin/env python3
"""测试 SurrealDB 查询"""

import asyncio
import httpx

BASE_URL = "http://localhost:18008"


async def test():
    # 先上传一个代码文件
    payload = {
        "memories": [
            {
                "type": "code",
                "content": "// Test\nfunction test() {}",
                "abstract": "Test",
                "project_id": "test-query",
                "metadata": {"file_path": "src/query_test.ts", "language": "typescript"},
            }
        ],
        "tenant_id": "default",
    }

    async with httpx.AsyncClient() as client:
        # 上传
        response = await client.post(f"{BASE_URL}/api/v1/memories", json=payload)
        result = response.json()
        print(f"上传结果: {result}")

        await asyncio.sleep(1)

        # 再次上传相同的 file_path
        response2 = await client.post(f"{BASE_URL}/api/v1/memories", json=payload)
        result2 = response2.json()
        print(f"第二次上传: {result2}")

        if result2.get("updated") == 1:
            print("✅ 正确: 第二次上传触发了更新")
        elif result2.get("success") == 1 and not result2.get("skipped"):
            print("✅ 正确: 第二次上传创建了新的记录（跳过去重）")
        elif result2.get("skipped"):
            print("❌ 问题: 第二次上传被跳过")
        else:
            print(f"⚠️  其他结果: {result2}")


if __name__ == "__main__":
    asyncio.run(test())
