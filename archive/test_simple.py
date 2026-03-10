"""
简化版 API 测试
快速验证核心功能
"""

import httpx
import asyncio
import sys


async def quick_test():
    """快速测试核心功能"""
    print("=" * 60)
    print("Embedding Service 快速功能测试")
    print("=" * 60)

    services = {
        "Embedding": "http://localhost:18000",
        "LLM": "http://localhost:18001",
        "Wrapper": "http://localhost:3001",
    }

    # 1. 健康检查
    print("\n1. 服务健康检查")
    print("-" * 60)
    all_healthy = True

    for name, url in services.items():
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{url}/health")
                if response.status_code == 200:
                    print(f"   ✅ {name}: 健康")
                else:
                    print(f"   ❌ {name}: 异常 ({response.status_code})")
                    all_healthy = False
        except Exception as e:
            print(f"   ❌ {name}: 无法连接 ({str(e)[:50]})")
            all_healthy = False

    if not all_healthy:
        print("\n❌ 部分服务未启动，请先运行: python start_services.py")
        return False

    # 2. Embedding 功能
    print("\n2. Embedding 功能测试")
    print("-" * 60)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:18000/v1/embeddings", json={"input": "你好，世界", "model": "Qwen3-Embedding-0.6B"}
            )
            if response.status_code == 200:
                data = response.json()
                embedding = data.get("data", [{}])[0].get("embedding", [])
                print(f"   ✅ 嵌入生成成功 (维度: {len(embedding)})")
            else:
                print(f"   ❌ 嵌入生成失败: {response.status_code}")
                return False
    except Exception as e:
        print(f"   ❌ 嵌入生成异常: {e}")
        return False

    # 3. LLM 功能
    print("\n3. LLM 功能测试")
    print("-" * 60)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:18001/generate", json={"prompt": "你好", "max_new_tokens": 20}
            )
            if response.status_code == 200:
                data = response.json()
                text = data.get("text", "")[:30]
                print(f"   ✅ 文本生成成功: {text}...")
            else:
                print(f"   ❌ 文本生成失败: {response.status_code}")
                return False
    except Exception as e:
        print(f"   ❌ 文本生成异常: {e}")
        return False

    # 4. Wrapper 代理功能
    print("\n4. Wrapper 代理功能测试")
    print("-" * 60)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:3001/v1/embeddings", json={"input": "测试文本", "model": "Qwen3-Embedding-0.6B"}
            )
            if response.status_code == 200:
                print(f"   ✅ 代理 Embedding 成功")
            else:
                print(f"   ❌ 代理 Embedding 失败: {response.status_code}")
                return False
    except Exception as e:
        print(f"   ❌ 代理 Embedding 异常: {e}")
        return False

    # 5. 记忆管理（可选）
    print("\n5. 记忆管理功能测试")
    print("-" * 60)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # 上传记忆
            response = await client.post(
                "http://localhost:3001/api/v1/memories",
                json={"memories": [{"content": "测试记忆内容", "metadata": {"test": True}}]},
            )
            if response.status_code == 200:
                print(f"   ✅ 记忆上传成功")
            else:
                print(f"   ⚠️  记忆上传跳过 (可能需要 SurrealDB)")

            # 搜索记忆
            response = await client.post(
                "http://localhost:3001/api/v1/memories/search", json={"query": "测试", "mode": "hybrid", "limit": 5}
            )
            if response.status_code == 200:
                print(f"   ✅ 记忆搜索成功")
            else:
                print(f"   ⚠️  记忆搜索跳过 (可能需要 SurrealDB)")
    except Exception as e:
        print(f"   ⚠️  记忆功能测试跳过: {str(e)[:50]}")

    print("\n" + "=" * 60)
    print("✅ 核心功能测试全部通过！")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        result = asyncio.run(quick_test())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\n测试被中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n测试出错: {e}")
        sys.exit(1)
