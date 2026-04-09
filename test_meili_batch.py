#!/usr/bin/env python3
"""测试 Meilisearch 分批同步"""

import asyncio
import time
import httpx

BASE_URL = "http://localhost:17999"


async def test():
    async with httpx.AsyncClient(timeout=60.0) as client:  # 增加超时
        # 测试 120条数据（应该分成 3 批：100条 SurrealDB + 50条 Meili）
        print("=== 测试 120条数据上传 ===")
        batch = []
        for i in range(120):
            batch.append(
                {
                    "type": "code",
                    "content": f"// Test file {i}\nfunction test{i}() {{ return {i}; }}",
                    "abstract": f"Test function {i}",
                    "project_id": "test-meili-batch",
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
        try:
            response = await client.post(
                f"{BASE_URL}/api/v1/memories", json={"memories": batch, "tenant_id": "default"}
            )
            elapsed = time.time() - start_time
            result = response.json()

            print(f"上传 120 条: success={result.get('success')}, failed={result.get('failed')}, time={elapsed:.2f}s")
            if result.get("success") == 120:
                print("✅ 120条上传成功")
            else:
                print(f"⚠️  部分失败: {result}")
        except Exception as e:
            print(f"❌ 请求失败: {e}")

        print("\n✅ 测试完成")


if __name__ == "__main__":
    asyncio.run(test())
