"""
最小化包装服务测试脚本

测试三个API端点的功能：
1. POST /v1/embeddings - 文本嵌入（带缓存）
2. POST /api/v1/memories - 批量上传记忆
3. POST /api/v1/memories/search - 搜索记忆

使用说明：
1. 确保embedding服务运行在 http://localhost:18000
2. 确保SurrealDB运行在 ws://localhost:8000
3. 启动wrapper服务: uv run python wrapper/src/main.py
4. 运行测试: uv run python tests/test_wrapper_minimal.py
"""

import asyncio
import httpx
import time
from typing import Any

BASE_URL = "http://localhost:17999"


async def test_health():
    """测试健康检查端点"""
    print("
=== 测试健康检查 ===")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Service: {data.get('service')}")
        print(f"Version: {data.get('version')}")
        print(f"Cache stats: {data.get('cache_stats')}")
        return response.status_code == 200


async def test_embeddings():
    """测试文本嵌入端点（带缓存）"""
    print("
=== 测试文本嵌入 ===")
    test_text = "这是一个测试文本"
    
    async with httpx.AsyncClient() as client:
        # 第一次请求（缓存未命中）
        print("
--- 第一次请求（缓存未命中）---")
        start = time.time()
        response1 = await client.post(
            f"{BASE_URL}/v1/embeddings",
            json={"input": test_text, "model": "Qwen3-Embedding-0.6B"}
        )
        duration1 = time.time() - start
        print(f"Status: {response1.status_code}")
        print(f"Duration: {duration1:.2f}s")
        
        if response1.status_code == 200:
            data1 = response1.json()
            embedding_count = len(data1.get("data", []))
            embedding_dim = len(data1["data"][0].get("embedding", [])) if data1.get("data") else 0
            print(f"Embeddings: {embedding_count}, Dimension: {embedding_dim}")
        
        # 第二次请求（缓存命中）
        print("
--- 第二次请求（缓存命中）---")
        start = time.time()
        response2 = await client.post(
            f"{BASE_URL}/v1/embeddings",
            json={"input": test_text, "model": "Qwen3-Embedding-0.6B"}
        )
        duration2 = time.time() - start
        print(f"Status: {response2.status_code}")
        print(f"Duration: {duration2:.2f}s")
        print(f"Cache hit speedup: {duration1/duration2:.1f}x" if duration2 > 0 else "N/A")
        
        return response1.status_code == 200 and response2.status_code == 200


async def test_upload_memories():
    """测试批量上传记忆"""
    print("
=== 测试批量上传记忆 ===")
    
    memories = [
        {
            "content": "Python是一种流行的编程语言",
            "metadata": {"source": "test", "category": "programming"},
        },
        {
            "content": "FastAPI是一个现代Web框架",
            "metadata": {"source": "test", "category": "web"},
        },
        {
            "content": "SurrealDB是一个多模型数据库",
            "metadata": {"source": "test", "category": "database"},
        },
    ]
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/v1/memories",
            json={"memories": memories}
        )
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Total: {data.get('total')}")
            print(f"Success: {data.get('success')}")
            print(f"Failed: {data.get('failed')}")
            print(f"Memory IDs: {data.get('memory_ids', [])[:3]}")  # 只显示前3个
            return data.get("success", 0) == len(memories)
        else:
            print(f"Error: {response.text}")
            return False


async def test_search_memories():
    """测试搜索记忆"""
    print("
=== 测试搜索记忆 ===")
    
    async with httpx.AsyncClient() as client:
        # 测试向量搜索
        print("
--- 向量搜索 ---")
        response1 = await client.post(
            f"{BASE_URL}/api/v1/memories/search",
            json={
                "query": "编程语言",
                "mode": "vector",
                "limit": 5,
                "threshold": 0.5
            }
        )
        print(f"Status: {response1.status_code}")
        if response1.status_code == 200:
            data = response1.json()
            print(f"Results: {data.get('total')}")
            print(f"Mode: {data.get('mode')}")
        
        # 测试混合搜索
        print("
--- 混合搜索 ---")
        response2 = await client.post(
            f"{BASE_URL}/api/v1/memories/search",
            json={
                "query": "web框架",
                "mode": "hybrid",
                "limit": 5,
                "threshold": 0.5
            }
        )
        print(f"Status: {response2.status_code}")
        if response2.status_code == 200:
            data = response2.json()
            print(f"Results: {data.get('total')}")
            print(f"Mode: {data.get('mode')}")
            for i, result in enumerate(data.get("results", [])[:3]):
                print(f"  [{i+1}] {result.get('content', '')[:50]}...")
        
        return response1.status_code == 200 and response2.status_code == 200


async def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("最小化包装服务测试套件")
    print("=" * 60)
    
    results = []
    
    # 测试1: 健康检查
    try:
        results.append(("Health Check", await test_health()))
    except Exception as e:
        print(f"Health Check failed: {e}")
        results.append(("Health Check", False))
    
    # 测试2: 文本嵌入（带缓存）
    try:
        results.append(("Embeddings", await test_embeddings()))
    except Exception as e:
        print(f"Embeddings test failed: {e}")
        results.append(("Embeddings", False))
    
    # 测试3: 批量上传记忆
    try:
        results.append(("Upload Memories", await test_upload_memories()))
    except Exception as e:
        print(f"Upload test failed: {e}")
        results.append(("Upload Memories", False))
    
    # 测试4: 搜索记忆
    try:
        results.append(("Search Memories", await test_search_memories()))
    except Exception as e:
        print(f"Search test failed: {e}")
        results.append(("Search Memories", False))
    
    # 汇总
    print("
" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name:.<40} {status}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    print(f"
总计: {passed_count}/{total_count} 通过")
    
    return passed_count == total_count


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
