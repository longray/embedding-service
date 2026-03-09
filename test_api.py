import httpx
import json
import asyncio

async def test_apis():
    base_url = "http://localhost:17999"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # 测试 /v1/embeddings
        print("=== 测试 /v1/embeddings 端点 ===")
        try:
            response = await client.post(
                f"{base_url}/v1/embeddings",
                json={"input": "测试文本嵌入", "model": "Qwen3-Embedding-0.6B"}
            )
            print(f"状态码: {response.status_code}")
            data = response.json()
            if "data" in data:
                print(f"✅ 成功 - 返回 {len(data['data'])} 个 embedding")
                print(f"   维度: {len(data['data'][0]['embedding'])}")
            else:
                print(f"响应: {json.dumps(data, ensure_ascii=False)[:200]}")
        except Exception as e:
            print(f"❌ 错误: {e}")
        
        # 测试缓存命中
        print("\n=== 再次测试 /v1/embeddings（缓存命中）===")
        try:
            response = await client.post(
                f"{base_url}/v1/embeddings",
                json={"input": "测试文本嵌入", "model": "Qwen3-Embedding-0.6B"}
            )
            print(f"状态码: {response.status_code}")
            data = response.json()
            if "data" in data:
                print(f"✅ 成功 - 缓存命中")
        except Exception as e:
            print(f"❌ 错误: {e}")
        
        # 测试 /api/v1/memories
        print("\n=== 测试 /api/v1/memories 端点 ===")
        try:
            response = await client.post(
                f"{base_url}/api/v1/memories",
                json={
                    "memories": [
                        {"content": "用户喜欢使用 Python 编程", "metadata": {"source": "test"}},
                        {"content": "系统使用 FastAPI 框架", "metadata": {"source": "test"}}
                    ]
                }
            )
            print(f"状态码: {response.status_code}")
            data = response.json()
            print(f"响应: {json.dumps(data, ensure_ascii=False)[:300]}")
        except Exception as e:
            print(f"❌ 错误: {e}")
        
        # 测试 /api/v1/memories/search
        print("\n=== 测试 /api/v1/memories/search 端点 ===")
        try:
            response = await client.post(
                f"{base_url}/api/v1/memories/search",
                json={"query": "Python 编程", "mode": "keyword", "limit": 5}
            )
            print(f"状态码: {response.status_code}")
            data = response.json()
            print(f"响应: {json.dumps(data, ensure_ascii=False)[:300]}")
        except Exception as e:
            print(f"❌ 错误: {e}")
        
        # 检查缓存统计
        print("\n=== 检查缓存统计 ===")
        response = await client.get(f"{base_url}/health")
        data = response.json()
        if "cache_stats" in data:
            stats = data["cache_stats"]
            print(f"缓存大小: {stats['current_size']}/{stats['max_size']}")
            print(f"命中: {stats['hits']}, 未命中: {stats['misses']}")
            print(f"命中率: {stats['hit_rate']}%")

asyncio.run(test_apis())
