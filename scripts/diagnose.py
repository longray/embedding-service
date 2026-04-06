#!/usr/bin/env python3
"""Embedding Service 系统诊断工具

一键检查整个系统的健康状态，帮助快速定位问题。

用法:
    uv run python scripts/diagnose.py

环境变量:
    WRAPPER_URL          Wrapper 服务地址 (默认: http://localhost:17999)
    EMBEDDING_URL        Embedding 服务地址 (默认: http://localhost:18000)
    SURREALDB_URL        SurrealDB 地址 (默认: ws://localhost:18002)
    MEILISEARCH_URL      Meilisearch 地址 (默认: http://localhost:18003)

退出码:
    0 - 所有服务正常
    1 - 部分服务异常
"""

import asyncio
import sys
from typing import Optional

import httpx


# ANSI 颜色代码
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str) -> None:
    print(f"\n{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print("=" * 50)


def print_success(text: str) -> None:
    print(f"  {Colors.GREEN}✅{Colors.RESET} {text}")


def print_warning(text: str) -> None:
    print(f"  {Colors.YELLOW}⚠️{Colors.RESET}  {text}")


def print_error(text: str, suggestion: Optional[str] = None) -> None:
    print(f"  {Colors.RED}❌{Colors.RESET} {text}")
    if suggestion:
        print(f"     {Colors.YELLOW}💡 建议: {suggestion}{Colors.RESET}")


def print_info(text: str) -> None:
    print(f"  {Colors.BLUE}ℹ️{Colors.RESET}  {text}")


async def check_http_service(
    name: str,
    url: str,
    health_path: str = "/health",
    timeout: float = 5.0,
) -> tuple[bool, Optional[dict]]:
    """检查 HTTP 服务健康状态"""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{url}{health_path}")
            if response.status_code == 200:
                try:
                    data = response.json()
                    return True, data
                except Exception:
                    return True, None
            else:
                return False, {"status_code": response.status_code}
    except httpx.ConnectError:
        return False, {"error": "连接拒绝"}
    except httpx.TimeoutException:
        return False, {"error": "连接超时"}
    except Exception as e:
        return False, {"error": str(e)}


async def check_wrapper_service(base_url: str) -> bool:
    """检查 Wrapper 服务"""
    print_info("检查 Wrapper 服务...")

    ok, data = await check_http_service("Wrapper", base_url)

    if ok:
        print_success(f"Wrapper 服务 ({base_url}) - 健康")
        if data:
            if "embedding_service" in data:
                status = data["embedding_service"].get("status", "unknown")
                if status == "healthy":
                    print_success("  └─ Embedding 后端连接正常")
                else:
                    print_warning(f"  └─ Embedding 后端状态: {status}")
            if "surrealdb" in data:
                status = data["surrealdb"].get("status", "unknown")
                if status == "connected":
                    print_success("  └─ SurrealDB 连接正常")
                else:
                    print_warning(f"  └─ SurrealDB 状态: {status}")
        return True
    else:
        error = data.get("error", "未知错误") if data else "未知错误"
        print_error(
            f"Wrapper 服务 ({base_url}) - {error}",
            suggestion="检查 docker-compose 是否启动: docker-compose ps",
        )
        return False


async def check_embedding_service(base_url: str) -> bool:
    """检查 Embedding 服务"""
    print_info("检查 Embedding 服务...")

    ok, data = await check_http_service("Embedding", base_url)

    if ok:
        print_success(f"Embedding 服务 ({base_url}) - 健康")
        if data and "model_loaded" in data:
            if data["model_loaded"]:
                print_success("  └─ 模型已加载")
            else:
                print_warning("  └─ 模型未加载（可能正在初始化）")
        return True
    else:
        error = data.get("error", "未知错误") if data else "未知错误"
        print_error(
            f"Embedding 服务 ({base_url}) - {error}",
            suggestion="检查服务是否启动: uv run python start_services.py",
        )
        return False


async def check_meilisearch(base_url: str) -> bool:
    """检查 Meilisearch 服务"""
    print_info("检查 Meilisearch 服务...")

    # Meilisearch health 端点是 /health
    ok, data = await check_http_service("Meilisearch", base_url, health_path="/health")

    if ok:
        print_success(f"Meilisearch ({base_url}) - 健康")
        return True
    else:
        error = data.get("error", "未知错误") if data else "未知错误"
        print_error(
            f"Meilisearch ({base_url}) - {error}",
            suggestion="检查 docker-compose: docker-compose up -d meilisearch",
        )
        return False


async def check_hnsw_stats(wrapper_url: str) -> bool:
    """检查 HNSW 索引统计"""
    print_info("检查 HNSW 索引...")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{wrapper_url}/api/v1/hnsw/stats?tenant_id=default")
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    print_success("HNSW 索引 - 正常")
                    return True
                elif data.get("status") == "not_found":
                    print_warning("HNSW 索引 - 不存在（可能需要初始化数据库）")
                    return False
                else:
                    print_error(f"HNSW 索引 - {data.get('message', '未知错误')}")
                    return False
            else:
                print_error(f"HNSW 索引 - HTTP {response.status_code}")
                return False
    except Exception as e:
        print_error(f"HNSW 索引 - {e}")
        return False


async def check_cache_stats(wrapper_url: str) -> bool:
    """检查缓存统计"""
    print_info("检查缓存系统...")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{wrapper_url}/api/v1/cache/stats")
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    stats = data.get("stats", {})
                    enabled = stats.get("cache_enabled", False)
                    if enabled:
                        ttl = stats.get("cache_ttl_seconds", 0)
                        print_success(f"缓存系统 - 已启用 (TTL: {ttl}s)")
                    else:
                        print_warning("缓存系统 - 已禁用")
                    return True
                else:
                    print_error(f"缓存系统 - {data.get('message', '未知错误')}")
                    return False
            else:
                print_error(f"缓存系统 - HTTP {response.status_code}")
                return False
    except Exception as e:
        print_error(f"缓存系统 - {e}")
        return False


async def main():
    """主函数"""
    print_header("🔍 Embedding Service 系统诊断")

    # 获取配置
    wrapper_url = "http://localhost:17999"
    embedding_url = "http://localhost:18000"
    meilisearch_url = "http://localhost:18003"

    # 服务状态检查
    print(f"\n{Colors.BOLD}服务状态检查:{Colors.RESET}")

    results = []

    # 1. Embedding 服务
    results.append(("Embedding", await check_embedding_service(embedding_url)))

    # 2. Wrapper 服务
    results.append(("Wrapper", await check_wrapper_service(wrapper_url)))

    # 3. Meilisearch
    results.append(("Meilisearch", await check_meilisearch(meilisearch_url)))

    # 功能验证（仅当 Wrapper 正常时）
    if results[-2][1]:  # Wrapper 正常
        print(f"\n{Colors.BOLD}功能验证:{Colors.RESET}")

        # 4. HNSW 索引
        hnsw_ok = await check_hnsw_stats(wrapper_url)
        results.append(("HNSW", hnsw_ok))

        # 5. 缓存系统
        cache_ok = await check_cache_stats(wrapper_url)
        results.append(("Cache", cache_ok))
    else:
        print_warning("跳过功能验证（Wrapper 服务未运行）")

    # 总体状态
    print_header("诊断结果")

    total = len(results)
    passed = sum(1 for _, ok in results if ok)

    if passed == total:
        print(f"{Colors.GREEN}{Colors.BOLD}🟢 总体状态: 健康 ({passed}/{total} 检查通过){Colors.RESET}")
        print("\n所有服务运行正常！")
        return 0
    elif passed >= total // 2:
        print(f"{Colors.YELLOW}{Colors.BOLD}🟡 总体状态: 警告 ({passed}/{total} 检查通过){Colors.RESET}")
        print("\n部分服务异常，请查看上方详情。")
        return 1
    else:
        print(f"{Colors.RED}{Colors.BOLD}🔴 总体状态: 异常 ({passed}/{total} 检查通过){Colors.RESET}")
        print("\n多个服务异常，建议检查：")
        print("  1. docker-compose 是否启动: docker-compose up -d")
        print("  2. 端口是否被占用: netstat -an | grep 1800")
        print("  3. 日志查看: docker-compose logs -f")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
