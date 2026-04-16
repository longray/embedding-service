#!/usr/bin/env python3
"""测试批量插入分批处理"""

import asyncio
import time
import httpx

BASE_URL = "http://localhost:18008"


async def test():
    async with httpx.AsyncClient() as client:
        # 测试 1: 小批量上传（< 100条）
        print("=== 测试 1: 小批量上传（50条）===")
        small_batch = []
        for i in range(50):
            small_batch.append(
                {
                    "type": "code",
                    "content": f"// Test file {i}\nfunction test{i}() {{ return {i}; }}",
                    "abstract": f"Test function {i}",
                    "project_id": "test-batch-small",
                    "metadata": {
                        "file_path": f"src/test_{i}.ts",
                        "language": "typescript",
                        "code_analysis": {
                            "complexity": {"cyclomatic_complexity": 1, "function_count": 1, "class_count": 0},
                            "imports": [],
                        },
                    },
                }
            )

        start_time = time.time()
        response = await client.post(
            f"{BASE_URL}/api/v1/memories", json={"memories": small_batch, "tenant_id": "default"}
        )
        elapsed = time.time() - start_time
        result = response.json()

        print(f"上传 50 条: success={result.get('success')}, time={elapsed:.2f}s")
        if result.get("success") == 50:
            print("✅ 小批量上传成功")
        else:
            print(f"❌ 小批量上传失败: {result}")

        # 测试 2: 大批量上传（> 100条，触发分批）
        print("\n=== 测试 2: 大批量上传（250条，分3批）===")
        large_batch = []
        for i in range(250):
            large_batch.append(
                {
                    "type": "code",
                    "content": f"// Large test file {i}\nfunction largeTest{i}() {{ return {i}; }}",
                    "abstract": f"Large test function {i}",
                    "project_id": "test-batch-large",
                    "metadata": {
                        "file_path": f"src/large_test_{i}.ts",
                        "language": "typescript",
                        "code_analysis": {
                            "complexity": {"cyclomatic_complexity": 1, "function_count": 1, "class_count": 0},
                            "imports": [],
                        },
                    },
                }
            )

        start_time = time.time()
        response = await client.post(
            f"{BASE_URL}/api/v1/memories", json={"memories": large_batch, "tenant_id": "default"}
        )
        elapsed = time.time() - start_time
        result = response.json()

        print(f"上传 250 条: success={result.get('success')}, failed={result.get('failed')}, time={elapsed:.2f}s")
        if result.get("success") == 250:
            print("✅ 大批量上传成功（已分批处理）")
        else:
            print(f"⚠️  大批量上传部分失败: {result}")

        print("\n✅ 批量插入分批处理测试完成")


if __name__ == "__main__":
    asyncio.run(test())
