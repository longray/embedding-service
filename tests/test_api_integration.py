"""
Embedding Service API 功能完整性测试

测试范围:
1. Embedding 服务 (端口 18000)
2. LLM 服务 (端口 18001)
3. Wrapper 服务 (端口 3001)
4. 记忆管理功能

使用方法:
    python test_api_integration.py

要求:
    - 服务已启动 (python start_services.py 或 docker-compose up)
    - Python 3.11+
    - httpx (pip install httpx)
"""

import asyncio
import httpx
import json
from datetime import datetime
from typing import Any
import sys

# 服务配置
BASE_URLS = {
    "embedding": "http://localhost:18000",
    "llm": "http://localhost:18001",
    "wrapper": "http://localhost:3001",
}

# 测试统计
test_results = {
    "total": 0,
    "passed": 0,
    "failed": 0,
    "errors": [],
}


def print_header(title: str):
    """打印测试标题"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_result(name: str, success: bool, message: str = ""):
    """打印测试结果"""
    status = "✅ PASS" if success else "❌ FAIL"
    test_results["total"] += 1
    if success:
        test_results["passed"] += 1
    else:
        test_results["failed"] += 1
        test_results["errors"].append(f"{name}: {message}")

    if message:
        print(f"  {status} - {name}: {message}")
    else:
        print(f"  {status} - {name}")


async def test_embedding_service():
    """测试 Embedding 服务"""
    print_header("测试 Embedding 服务 (端口 18000)")

    base_url = BASE_URLS["embedding"]

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test 1: 健康检查
        try:
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print_result("健康检查", True, f"状态: {data.get('status', 'unknown')}")
            else:
                print_result("健康检查", False, f"状态码: {response.status_code}")
        except Exception as e:
            print_result("健康检查", False, str(e))

        # Test 2: 获取模型列表
        try:
            response = await client.get(f"{base_url}/v1/models")
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                print_result("获取模型列表", True, f"找到 {len(models)} 个模型")
            else:
                print_result("获取模型列表", False, f"状态码: {response.status_code}")
        except Exception as e:
            print_result("获取模型列表", False, str(e))

        # Test 3: 生成文本嵌入
        try:
            response = await client.post(
                f"{base_url}/v1/embeddings", json={"input": "这是一个测试文本", "model": "Qwen3-Embedding-0.6B"}
            )
            if response.status_code == 200:
                data = response.json()
                embedding = data.get("data", [{}])[0].get("embedding", [])
                print_result("生成文本嵌入", True, f"维度: {len(embedding)}")
            else:
                print_result("生成文本嵌入", False, f"状态码: {response.status_code}, 响应: {response.text[:100]}")
        except Exception as e:
            print_result("生成文本嵌入", False, str(e))

        # Test 4: 批量嵌入
        try:
            response = await client.post(
                f"{base_url}/v1/embeddings",
                json={"input": ["文本1", "文本2", "文本3"], "model": "Qwen3-Embedding-0.6B"},
            )
            if response.status_code == 200:
                data = response.json()
                embeddings = data.get("data", [])
                print_result("批量嵌入", True, f"生成 {len(embeddings)} 个向量")
            else:
                print_result("批量嵌入", False, f"状态码: {response.status_code}")
        except Exception as e:
            print_result("批量嵌入", False, str(e))

        # Test 5: 获取统计信息
        try:
            response = await client.get(f"{base_url}/stats")
            if response.status_code == 200:
                data = response.json()
                print_result("获取统计信息", True, f"缓存命中: {data.get('cache_hit', 0)}")
            else:
                print_result("获取统计信息", False, f"状态码: {response.status_code}")
        except Exception as e:
            print_result("获取统计信息", False, str(e))


async def test_llm_service():
    """测试 LLM 服务"""
    print_header("测试 LLM 服务 (端口 18001)")

    base_url = BASE_URLS["llm"]

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test 1: 健康检查
        try:
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print_result("健康检查", True, f"状态: {data.get('status', 'unknown')}")
            else:
                print_result("健康检查", False, f"状态码: {response.status_code}")
        except Exception as e:
            print_result("健康检查", False, str(e))

        # Test 2: 获取模型列表
        try:
            response = await client.get(f"{base_url}/v1/models")
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                print_result("获取模型列表", True, f"找到 {len(models)} 个模型")
            else:
                print_result("获取模型列表", False, f"状态码: {response.status_code}")
        except Exception as e:
            print_result("获取模型列表", False, str(e))

        # Test 3: 简单生成
        try:
            response = await client.post(f"{base_url}/generate", json={"prompt": "你好", "max_new_tokens": 50})
            if response.status_code == 200:
                data = response.json()
                text = data.get("text", "")[:50]
                print_result("简单生成", True, f"生成文本: {text}...")
            else:
                print_result("简单生成", False, f"状态码: {response.status_code}")
        except Exception as e:
            print_result("简单生成", False, str(e))

        # Test 4: 对话补全
        try:
            response = await client.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "你好，请介绍一下自己"}],
                    "model": "MiniCPM4-0.5B",
                    "max_tokens": 100,
                    "temperature": 0.7,
                },
            )
            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")[:50]
                print_result("对话补全", True, f"回复: {content}...")
            else:
                print_result("对话补全", False, f"状态码: {response.status_code}")
        except Exception as e:
            print_result("对话补全", False, str(e))


async def test_wrapper_service():
    """测试 Wrapper 服务"""
    print_header("测试 Wrapper 服务 (端口 3001)")

    base_url = BASE_URLS["wrapper"]

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test 1: 健康检查
        try:
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                data = response.json()
                cache_stats = data.get("cache_stats", {})
                print_result("健康检查", True, f"缓存命中率: {cache_stats.get('hit_rate', 0):.2%}")
            else:
                print_result("健康检查", False, f"状态码: {response.status_code}")
        except Exception as e:
            print_result("健康检查", False, str(e))

        # Test 2: Prometheus 指标
        try:
            response = await client.get(f"{base_url}/metrics")
            if response.status_code == 200:
                metrics = response.text
                # 检查是否包含关键指标
                if "wrapper_requests_total" in metrics:
                    print_result("Prometheus 指标", True, "指标可用")
                else:
                    print_result("Prometheus 指标", True, "指标返回但可能不完整")
            else:
                print_result("Prometheus 指标", False, f"状态码: {response.status_code}")
        except Exception as e:
            print_result("Prometheus 指标", False, str(e))

        # Test 3: 包装层 Embedding
        try:
            response = await client.post(
                f"{base_url}/v1/embeddings", json={"input": "测试文本", "model": "Qwen3-Embedding-0.6B"}
            )
            if response.status_code == 200:
                data = response.json()
                embedding = data.get("data", [{}])[0].get("embedding", [])
                print_result("包装层 Embedding", True, f"维度: {len(embedding)}")
            else:
                print_result("包装层 Embedding", False, f"状态码: {response.status_code}")
        except Exception as e:
            print_result("包装层 Embedding", False, str(e))

        # Test 4: 包装层对话
        try:
            response = await client.post(
                f"{base_url}/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "你好"}], "model": "MiniCPM4-0.5B", "max_tokens": 50},
            )
            if response.status_code == 200:
                print_result("包装层对话", True, "对话成功")
            else:
                print_result("包装层对话", False, f"状态码: {response.status_code}")
        except Exception as e:
            print_result("包装层对话", False, str(e))


async def test_memory_management():
    """测试记忆管理功能"""
    print_header("测试记忆管理功能 (SurrealDB)")

    base_url = BASE_URLS["wrapper"]

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test 1: 上传记忆
        memory_id = None
        try:
            response = await client.post(
                f"{base_url}/api/v1/memories",
                json={
                    "memories": [
                        {
                            "content": "用户喜欢Python编程",
                            "metadata": {"source": "test", "category": "preference"},
                            "entities": [{"name": "Python", "type": "language"}, {"name": "编程", "type": "skill"}],
                        }
                    ]
                },
            )
            if response.status_code == 200:
                data = response.json()
                memory_ids = data.get("memory_ids", [])
                memory_id = memory_ids[0] if memory_ids else None
                print_result("上传记忆", True, f"ID: {memory_id}")
            else:
                print_result("上传记忆", False, f"状态码: {response.status_code}, 响应: {response.text[:100]}")
        except Exception as e:
            print_result("上传记忆", False, str(e))

        # Test 2: 搜索记忆
        try:
            response = await client.post(
                f"{base_url}/api/v1/memories/search", json={"query": "Python 编程", "mode": "hybrid", "limit": 5}
            )
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                print_result("搜索记忆", True, f"找到 {len(results)} 条结果")
            else:
                print_result("搜索记忆", False, f"状态码: {response.status_code}")
        except Exception as e:
            print_result("搜索记忆", False, str(e))

        # Test 3: 向量搜索
        try:
            response = await client.post(
                f"{base_url}/api/v1/memories/search", json={"query": "编程语言", "mode": "vector", "limit": 3}
            )
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                print_result("向量搜索", True, f"找到 {len(results)} 条结果")
            else:
                print_result("向量搜索", False, f"状态码: {response.status_code}")
        except Exception as e:
            print_result("向量搜索", False, str(e))

        # Test 4: 关键词搜索
        try:
            response = await client.post(
                f"{base_url}/api/v1/memories/search", json={"query": "Python", "mode": "keyword", "limit": 3}
            )
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                print_result("关键词搜索", True, f"找到 {len(results)} 条结果")
            else:
                print_result("关键词搜索", False, f"状态码: {response.status_code}")
        except Exception as e:
            print_result("关键词搜索", False, str(e))


async def test_service_dependencies():
    """测试服务依赖关系"""
    print_header("测试服务依赖关系")

    # 检查服务启动顺序
    services = ["embedding", "llm", "wrapper"]

    for service in services:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{BASE_URLS[service]}/health")
                if response.status_code == 200:
                    print(f"  ✅ {service}: 运行中")
                else:
                    print(f"  ❌ {service}: 异常 (状态码: {response.status_code})")
        except Exception as e:
            print(f"  ❌ {service}: 无法连接 ({str(e)})")


async def run_all_tests():
    """运行所有测试"""
    print(f"\n{'#' * 60}")
    print(f"#  Embedding Service API 功能完整性测试")
    print(f"#  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 60}")

    # 检查服务依赖
    await test_service_dependencies()

    # 运行各项测试
    await test_embedding_service()
    await test_llm_service()
    await test_wrapper_service()
    await test_memory_management()

    # 打印测试报告
    print_header("测试报告")
    print(f"  总测试数: {test_results['total']}")
    print(f"  ✅ 通过: {test_results['passed']}")
    print(f"  ❌ 失败: {test_results['failed']}")
    print(
        f"  通过率: {test_results['passed'] / test_results['total'] * 100:.1f}%"
        if test_results["total"] > 0
        else "  通过率: N/A"
    )

    if test_results["errors"]:
        print(f"\n  失败详情:")
        for error in test_results["errors"]:
            print(f"    - {error}")

    # 返回退出码
    if test_results["failed"] > 0:
        print(f"\n{'=' * 60}")
        print("  测试结果: 部分测试失败")
        print(f"{'=' * 60}")
        return 1
    else:
        print(f"\n{'=' * 60}")
        print("  测试结果: 全部通过 ✅")
        print(f"{'=' * 60}")
        return 0


if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print(__doc__)
        sys.exit(0)

    # 运行测试
    try:
        exit_code = asyncio.run(run_all_tests())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n测试执行出错: {e}")
        sys.exit(1)
