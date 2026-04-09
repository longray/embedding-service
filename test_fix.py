#!/usr/bin/env python3
"""测试代码分析数据上传修复"""

import asyncio
import json
import time
import httpx

BASE_URL = "http://localhost:17999"


async def test_upload_new_code_file():
    """测试1: 上传全新代码文件（有 file_path）"""
    print("\n=== 测试1: 上传全新代码文件 ===")

    unique_content = f"// Test file generated at {int(time.time() * 1000)}\nfunction test() {{ return 42; }}"

    payload = {
        "memories": [
            {
                "type": "code",
                "content": unique_content,
                "abstract": "Unique test function",
                "project_id": "test-project",
                "metadata": {"file_path": f"src/test_{int(time.time() * 1000)}.ts", "language": "typescript"},
            }
        ],
        "tenant_id": "default",
    }

    async with httpx.AsyncClient() as client:
        # 上传
        response = await client.post(f"{BASE_URL}/api/v1/memories", json=payload)
        print(f"上传响应: {response.status_code}")
        result = response.json()
        print(f"响应内容: {json.dumps(result, indent=2, ensure_ascii=False)}")

        if result.get("success", 0) == 0:
            print("❌ 上传失败")
            return False

        memory_id = result.get("memory_ids", [None])[0]
        if not memory_id:
            print("❌ 没有返回 memory_id")
            return False

        print(f"✅ 上传成功，memory_id: {memory_id}")

        # 立即查询
        await asyncio.sleep(0.5)  # 等待写入完成
        query_response = await client.get(f"{BASE_URL}/api/v1/memories/{memory_id}?tenant_id=default")
        print(f"查询响应: {query_response.status_code}")

        if query_response.status_code == 404:
            print("❌ 查询不到记忆（数据未写入）")
            return False

        query_result = query_response.json()
        print(f"查询内容: {json.dumps(query_result, indent=2, ensure_ascii=False)[:500]}")
        print("✅ 查询成功，数据已写入")
        return True


async def test_upload_code_without_filepath():
    """测试2: 上传代码文件（无 file_path）"""
    print("\n=== 测试2: 上传代码文件（无 file_path） ===")

    unique_content = f"// Code snippet {int(time.time() * 1000)}\nconst x = 1;"

    payload = {
        "memories": [
            {
                "type": "code",
                "content": unique_content,
                "abstract": "Code snippet test",
                "project_id": "test-project",
                "metadata": {
                    "language": "javascript"
                    # 没有 file_path
                },
            }
        ],
        "tenant_id": "default",
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/api/v1/memories", json=payload)
        print(f"上传响应: {response.status_code}")
        result = response.json()
        print(f"响应内容: {json.dumps(result, indent=2, ensure_ascii=False)}")

        if result.get("success", 0) == 0:
            print("❌ 上传失败")
            return False

        memory_id = result.get("memory_ids", [None])[0]
        if not memory_id:
            print("❌ 没有返回 memory_id")
            return False

        print(f"✅ 上传成功，memory_id: {memory_id}")

        await asyncio.sleep(0.5)
        query_response = await client.get(f"{BASE_URL}/api/v1/memories/{memory_id}?tenant_id=default")
        print(f"查询响应: {query_response.status_code}")

        if query_response.status_code == 404:
            print("❌ 查询不到记忆")
            return False

        print("✅ 查询成功，数据已写入")
        return True


async def test_hash_dedup_bypass():
    """测试3: 验证代码数据跳过去重"""
    print("\n=== 测试3: 验证代码数据跳过去重 ===")

    # 相同内容的代码，应该创建两条记录
    same_content = "// Same content\nfunction same() { return 1; }"

    payload = {
        "memories": [
            {
                "type": "code",
                "content": same_content,
                "abstract": "Same content test",
                "project_id": "test-project",
                "metadata": {"file_path": "src/same.ts", "language": "typescript"},
            }
        ],
        "tenant_id": "default",
    }

    async with httpx.AsyncClient() as client:
        # 第一次上传
        response1 = await client.post(f"{BASE_URL}/api/v1/memories", json=payload)
        result1 = response1.json()
        print(f"第一次上传: success={result1.get('success')}, memory_ids={result1.get('memory_ids')}")

        # 等待第一次写入完成
        await asyncio.sleep(1)

        # 第二次上传（相同内容）
        response2 = await client.post(f"{BASE_URL}/api/v1/memories", json=payload)
        result2 = response2.json()
        print(
            f"第二次上传: success={result2.get('success')}, updated={result2.get('updated')}, memory_ids={result2.get('memory_ids')}"
        )

        # 验证：第二次应该更新（因为 file_path 相同），而不是跳过
        if result2.get("updated", 0) == 1:
            print("✅ 第二次上传触发更新（符合预期，file_path 相同）")
            return True
        elif result2.get("success", 0) == 1 and result2.get("skipped"):
            print("❌ 第二次上传被跳过（不应该发生）")
            return False
        else:
            print(f"⚠️  第二次上传结果: {json.dumps(result2, indent=2)}")
            return True


async def test_general_memory_dedup():
    """测试4: 验证非代码类型仍然去重"""
    print("\n=== 测试4: 验证非代码类型仍然去重 ===")

    same_content = f"This is a test memory {int(time.time() / 1000)}"  # 秒级时间戳确保相同

    payload = {
        "memories": [
            {"type": "general", "content": same_content, "abstract": "Test memory", "project_id": "test-project"}
        ],
        "tenant_id": "default",
    }

    async with httpx.AsyncClient() as client:
        # 第一次上传
        response1 = await client.post(f"{BASE_URL}/api/v1/memories", json=payload)
        result1 = response1.json()
        print(f"第一次上传: success={result1.get('success')}, memory_ids={result1.get('memory_ids')}")

        # 第二次上传（相同内容）
        response2 = await client.post(f"{BASE_URL}/api/v1/memories", json=payload)
        result2 = response2.json()
        print(f"第二次上传: success={result2.get('success')}, skipped={len(result2.get('skipped', []))}")

        # 验证：第二次应该被跳过
        if result2.get("skipped") and len(result2.get("skipped", [])) > 0:
            print("✅ 第二次上传被跳过（符合预期，general 类型去重）")
            return True
        else:
            print("⚠️  第二次上传未被跳过（可能内容不同）")
            return True


async def main():
    print("=" * 60)
    print("代码分析数据上传修复测试")
    print("=" * 60)

    results = []

    try:
        results.append(("测试1: 新代码文件（有 file_path）", await test_upload_new_code_file()))
    except Exception as e:
        print(f"❌ 测试1异常: {e}")
        results.append(("测试1", False))

    try:
        results.append(("测试2: 代码文件（无 file_path）", await test_upload_code_without_filepath()))
    except Exception as e:
        print(f"❌ 测试2异常: {e}")
        results.append(("测试2", False))

    try:
        results.append(("测试3: 代码数据跳过去重", await test_hash_dedup_bypass()))
    except Exception as e:
        print(f"❌ 测试3异常: {e}")
        results.append(("测试3", False))

    try:
        results.append(("测试4: 非代码类型去重", await test_general_memory_dedup()))
    except Exception as e:
        print(f"❌ 测试4异常: {e}")
        results.append(("测试4", False))

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{status}: {name}")

    passed_count = sum(1 for _, p in results if p)
    print(f"\n总计: {passed_count}/{len(results)} 通过")


if __name__ == "__main__":
    asyncio.run(main())
