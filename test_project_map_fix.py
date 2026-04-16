#!/usr/bin/env python3
"""测试项目地图查询修复"""

import asyncio
import httpx

BASE_URL = "http://localhost:18008"


async def test():
    async with httpx.AsyncClient() as client:
        project_id = "test-project-map-fix"

        # 1. 上传测试文件（不带 code_analysis）
        print("=== 上传测试文件 ===")
        upload_response = await client.post(
            f"{BASE_URL}/api/v1/memories",
            json={
                "memories": [
                    {
                        "type": "code",
                        "content": "export function helper() { return 1; }",
                        "abstract": "Helper function",
                        "project_id": project_id,
                        "metadata": {
                            "file_path": "src/helper.ts",
                            "language": "typescript",
                            # 注意：没有 code_analysis
                        },
                    },
                    {
                        "type": "code",
                        "content": "export function main() { helper(); }",
                        "abstract": "Main function",
                        "project_id": project_id,
                        "metadata": {
                            "file_path": "src/main.ts",
                            "language": "typescript",
                            # 注意：没有 code_analysis
                        },
                    },
                ],
                "tenant_id": "default",
            },
        )

        upload_result = upload_response.json()
        memory_ids = upload_result.get("memory_ids", [])

        if len(memory_ids) != 2:
            print(f"❌ 上传失败: {upload_result}")
            return

        print(f"✅ 上传成功，memory_ids: {memory_ids}")

        # 2. 创建调用关系
        print("\n=== 创建调用关系 ===")
        calls_response = await client.post(
            f"{BASE_URL}/api/v1/calls/batch",
            json={
                "calls": [
                    {
                        "caller_memory_id": memory_ids[1],  # main.ts
                        "callee_memory_id": memory_ids[0],  # helper.ts
                        "line": 1,
                    }
                ]
            },
        )

        calls_result = calls_response.json()
        print(f"调用关系创建: {calls_result}")

        # 3. 查询项目地图
        print("\n=== 查询项目地图 ===")
        await asyncio.sleep(1)  # 等待数据同步

        map_response = await client.get(f"{BASE_URL}/api/v1/projects/{project_id}/map?tenant_id=default")

        map_result = map_response.json()
        print(f"项目地图响应: {map_result}")

        # 验证
        file_tree = map_result.get("file_tree", {})
        module_deps = map_result.get("module_dependencies", [])
        stats = map_result.get("statistics", {})

        print(f"\n验证结果:")
        print(f"  file_tree: {file_tree}")
        print(f"  module_dependencies: {module_deps}")
        print(f"  statistics: {stats}")

        if file_tree and len(module_deps) > 0:
            print("\n✅ 项目地图查询成功！")
        else:
            print("\n❌ 项目地图查询返回空数据")


if __name__ == "__main__":
    asyncio.run(test())
