#!/usr/bin/env python3
"""测试 embedding 字段优化"""

import asyncio
import httpx
import json

BASE_URL = "http://localhost:17999"


async def test():
    async with httpx.AsyncClient() as client:
        # 先上传一个测试记忆
        print("=== 上传测试记忆 ===")
        upload_response = await client.post(
            f"{BASE_URL}/api/v1/memories",
            json={
                "memories": [
                    {
                        "type": "code",
                        "content": "export function test() { return 42; }",
                        "abstract": "Test function for embedding opt",
                        "project_id": "test-embedding-opt",
                        "metadata": {"file_path": "src/test_embedding.ts"},
                    }
                ],
                "tenant_id": "default",
            },
        )

        upload_result = upload_response.json()
        memory_id = upload_result.get("memory_ids", [None])[0]

        if not memory_id:
            print("❌ 上传失败")
            return

        print(f"✅ 上传成功，memory_id: {memory_id}")

        # 测试 1: 默认查询（不包含 embedding）
        print("\n=== 测试 1: 默认查询（不包含 embedding）===")
        response1 = await client.get(f"{BASE_URL}/api/v1/memories/{memory_id}?tenant_id=default")

        result1 = response1.json()
        memory1 = result1.get("memory", {})

        has_embedding_default = "embedding" in memory1
        response_size_default = len(json.dumps(result1))

        print(f"包含 embedding: {has_embedding_default}")
        print(f"响应大小: {response_size_default} bytes")

        if not has_embedding_default:
            print("✅ 默认查询不包含 embedding")
        else:
            print("❌ 默认查询包含 embedding（不符合预期）")

        # 测试 2: 显式包含 embedding
        print("\n=== 测试 2: 显式包含 embedding ===")
        response2 = await client.get(f"{BASE_URL}/api/v1/memories/{memory_id}?tenant_id=default&include_embedding=true")

        result2 = response2.json()
        memory2 = result2.get("memory", {})

        has_embedding_explicit = "embedding" in memory2
        response_size_explicit = len(json.dumps(result2))

        print(f"包含 embedding: {has_embedding_explicit}")
        print(f"响应大小: {response_size_explicit} bytes")

        if has_embedding_explicit:
            embedding_len = len(memory2.get("embedding", []))
            print(f"embedding 维度: {embedding_len}")
            print("✅ 显式查询包含 embedding")
        else:
            print("❌ 显式查询不包含 embedding（不符合预期）")

        # 测试 3: 响应体积对比
        print("\n=== 测试 3: 响应体积对比 ===")
        if response_size_explicit > 0:
            reduction = (1 - response_size_default / response_size_explicit) * 100
            print(f"默认查询: {response_size_default} bytes")
            print(f"包含 embedding: {response_size_explicit} bytes")
            print(f"体积减少: {reduction:.1f}%")

            if reduction > 80:
                print("✅ 体积减少超过 80%")
            else:
                print(f"⚠️  体积减少 {reduction:.1f}%（预期 > 80%）")

        print("\n✅ 所有测试完成")


if __name__ == "__main__":
    asyncio.run(test())
