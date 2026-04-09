#!/usr/bin/env python3
"""测试插件端格式"""

import asyncio
import time
import httpx

BASE_URL = "http://localhost:17999"


async def test():
    timestamp = int(time.time() * 1000)

    # 模拟插件端的请求格式
    payload = {
        "memories": [
            {
                "type": "code",
                "content": f"// Test file generated at {timestamp}\nexport function uniqueFunction{timestamp}() {{ return 'unique'; }}",
                "abstract": "Unique test function",
                "overview": "Test code for dedup verification",
                "project_id": "test-project",
                "metadata": {
                    "file_path": f"src/test_{timestamp}.ts",
                    "code_analysis": {"language": "typescript", "functions": [{"name": "uniqueFunction", "line": 3}]},
                },
            }
        ],
        "tenant_id": "default",
    }

    async with httpx.AsyncClient() as client:
        print("=== 测试插件端格式 ===")

        # 上传
        response = await client.post(f"{BASE_URL}/api/v1/memories", json=payload)
        print(f"上传响应: {response.status_code}")
        result = response.json()
        print(f"结果: {result}")

        if result.get("success") == 0 and not result.get("memory_ids"):
            print("❌ 上传失败")
            return

        memory_id = result.get("memory_ids", [None])[0]
        if not memory_id:
            print("❌ 没有返回 memory_id")
            return

        print(f"✅ 上传成功，memory_id: {memory_id}")

        # 等待 2 秒（和插件端一样）
        print("\n等待 2 秒...")
        await asyncio.sleep(2)

        # 查询
        print("\n=== 查询验证 ===")
        query_response = await client.get(f"{BASE_URL}/api/v1/memories/{memory_id}?tenant_id=default")
        print(f"查询响应: {query_response.status_code}")

        if query_response.status_code == 404:
            print("❌ 查询不到记忆（数据未写入）")
        else:
            query_result = query_response.json()
            print(f"✅ 查询成功!")
            print(f"Type: {query_result.get('memory', {}).get('type')}")
            print(f"Project: {query_result.get('memory', {}).get('project_id')}")


if __name__ == "__main__":
    asyncio.run(test())
