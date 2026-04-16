#!/usr/bin/env python3
"""测试 SQL 注入防护"""

import asyncio
import httpx

BASE_URL = "http://localhost:18008"


async def test():
    async with httpx.AsyncClient() as client:
        # 1. 上传两个代码文件
        print("=== 上传代码文件 ===")
        upload_response = await client.post(
            f"{BASE_URL}/api/v1/memories",
            json={
                "memories": [
                    {
                        "type": "code",
                        "content": "export function main() { helper(); }",
                        "abstract": "Main file",
                        "project_id": "test-sql-injection",
                        "metadata": {
                            "file_path": "src/main.ts",
                            "language": "typescript",
                            "code_analysis": {
                                "complexity": {"cyclomatic_complexity": 2, "function_count": 1, "class_count": 0},
                                "imports": [],
                            },
                        },
                    },
                    {
                        "type": "code",
                        "content": "export function helper() { return 1; }",
                        "abstract": "Helper file",
                        "project_id": "test-sql-injection",
                        "metadata": {
                            "file_path": "src/helper.ts",
                            "language": "typescript",
                            "code_analysis": {
                                "complexity": {"cyclomatic_complexity": 1, "function_count": 1, "class_count": 0},
                                "imports": [],
                            },
                        },
                    },
                ]
            },
        )

        result = upload_response.json()
        print(f"上传结果: {result}")

        if result.get("success") != 2:
            print("❌ 上传失败")
            return

        memory_ids = result.get("memory_ids", [])
        print(f"✅ 上传成功，memory_ids: {memory_ids}")

        # 2. 测试特殊字符 description
        print("\n=== 测试特殊字符 description ===")
        special_descriptions = [
            "正常描述",
            "描述'包含单引号",
            '描述"包含双引号',
            "描述;包含分号",
            "描述--包含注释",
            "描述/*包含块注释*/",
            "描述\\包含反斜杠",
        ]

        for desc in special_descriptions:
            calls_response = await client.post(
                f"{BASE_URL}/api/v1/calls/batch",
                json={
                    "calls": [
                        {
                            "caller_memory_id": memory_ids[0],
                            "callee_memory_id": memory_ids[1],
                            "line": 1,
                            "description": desc,
                        }
                    ]
                },
            )

            calls_result = calls_response.json()
            if calls_result.get("created") == 1:
                print(f"✅ 通过: {desc[:20]}...")
            else:
                print(f"❌ 失败: {desc[:20]}... - {calls_result}")

        print("\n✅ SQL 注入防护测试完成")


if __name__ == "__main__":
    asyncio.run(test())
