#!/usr/bin/env python3
"""测试部署状态"""

import asyncio
import time
import httpx

BASE_URL = "http://localhost:17999"


async def test():
    timestamp = int(time.time() * 1000)

    payload = {
        "memories": [
            {
                "type": "code",
                "content": f"// Test {timestamp}\nfunction test() {{ return 42; }}",
                "abstract": "Test function",
                "project_id": "test-deploy",
                "metadata": {"file_path": f"src/test_{timestamp}.ts", "language": "typescript"},
            }
        ],
        "tenant_id": "default",
    }

    async with httpx.AsyncClient() as client:
        print("=== 测试 HTTP API 上传 ===")
        print(f"Payload: {payload}")

        # 上传
        response = await client.post(f"{BASE_URL}/api/v1/memories", json=payload)
        print(f"\n上传响应: {response.status_code}")
        result = response.json()
        print(f"结果: {result}")

        if result.get("success") == 0:
            print("❌ 上传失败")
            return

        memory_id = result.get("memory_ids", [None])[0]
        if not memory_id:
            print("❌ 没有返回 memory_id")
            return

        print(f"✅ 上传成功，memory_id: {memory_id}")

        # 立即查询
        print(f"\n=== 立即查询 ===")
        await asyncio.sleep(0.5)
        query_response = await client.get(f"{BASE_URL}/api/v1/memories/{memory_id}?tenant_id=default")
        print(f"查询响应: {query_response.status_code}")

        if query_response.status_code == 404:
            print("❌ 查询不到记忆（数据未写入）")
        else:
            query_result = query_response.json()
            print(f"✅ 查询成功: {query_result.get('status')}")


if __name__ == "__main__":
    asyncio.run(test())
