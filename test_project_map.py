#!/usr/bin/env python3
"""测试项目地图 API 的 module_dependencies"""

import asyncio
import httpx

BASE_URL = "http://localhost:17999"


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
                        "project_id": "test-map-project",
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
                        "project_id": "test-map-project",
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

        # 2. 创建调用关系
        print("\n=== 创建调用关系 ===")
        calls_response = await client.post(
            f"{BASE_URL}/api/v1/calls/batch",
            json={
                "calls": [
                    {
                        "caller_memory_id": memory_ids[0],
                        "callee_memory_id": memory_ids[1],
                        "line": 1,
                    }
                ]
            },
        )

        calls_result = calls_response.json()
        print(f"调用关系结果: {calls_result}")

        if calls_result.get("created") != 1:
            print("❌ 调用关系创建失败")
            return

        print("✅ 调用关系创建成功")

        # 3. 查询项目地图
        print("\n=== 查询项目地图 ===")
        await asyncio.sleep(1)  # 等待数据同步

        map_response = await client.get(f"{BASE_URL}/api/v1/projects/test-map-project/map?tenant_id=default")

        map_result = map_response.json()
        print(f"项目地图结果:")
        print(f"  Status: {map_result.get('status')}")
        print(f"  File tree: {len(map_result.get('file_tree', []))} items")
        print(f"  Module dependencies: {len(map_result.get('module_dependencies', []))} items")
        print(f"  Hot files: {len(map_result.get('hot_files', []))} items")
        print(f"  Statistics: {map_result.get('statistics', {})}")

        print("\n=== 调试：直接查询调用关系 ===")
        debug_response = await client.post(
            f"{BASE_URL}/api/v1/memories/search",
            json={"query": "", "filters": {"type": "code", "project_id": "test-map-project"}, "limit": 10},
        )
        debug_result = debug_response.json()
        print(f"搜索到的代码文件: {len(debug_result.get('results', []))}")
        for r in debug_result.get("results", [])[:2]:
            print(f"  - {r.get('id')}: {r.get('metadata', {}).get('file_path')}")

        deps = map_result.get("module_dependencies", [])
        if deps:
            print(f"\n✅ Module dependencies 不为空!")
            for dep in deps[:3]:
                print(f"  - {dep}")
        else:
            print(f"\n⚠️ Module dependencies 为空")


if __name__ == "__main__":
    asyncio.run(test())
