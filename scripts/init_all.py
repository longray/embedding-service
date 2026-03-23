"""
一键初始化脚本

从零开始初始化完整的 Embedding Service 环境，包括：
- SurrealDB 数据库初始化
- Meilisearch 索引初始化
- 验证所有服务正常运行

用法:
    uv run python scripts/init_all.py

环境变量:
    # SurrealDB 配置
    SURREAL_URL          SurrealDB WebSocket URL (默认: ws://localhost:18002)
    SURREAL_NS           命名空间 (默认: memory_ns)
    SURREAL_DB           数据库 (默认: memory_db)
    SURREAL_USER         用户名 (默认: root)
    SURREAL_PASS         密码 (默认: root)

    # Meilisearch 配置
    WRAPPER_MEILI_URL         Meilisearch URL (默认: http://localhost:7700)
    WRAPPER_MEILI_API_KEY      Meilisearch API Key (默认: None)
    WRAPPER_MEILI_INDEX_NAME   索引名 (默认: memories)

示例:
    # 默认配置初始化
    uv run python scripts/init_all.py

    # 自定义配置初始化
    export SURREAL_URL=ws://localhost:18002
    export WRAPPER_MEILI_URL=http://localhost:7700
    uv run python scripts/init_all.py

    # 仅验证（不重新初始化）
    uv run python scripts/init_all.py --verify-only
"""

import argparse
import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("init_all")


async def check_surrealdb_health(url: str) -> bool:
    """检查 SurrealDB 健康状态"""
    try:
        import httpx

        # 尝试连接 SurrealDB
        response = await httpx.AsyncClient(timeout=5.0).get(
            url.replace("ws://", "http://").replace("ws://", "http://") + "/health"
        )
        return response.status_code == 200
    except Exception:
        return False


async def check_meilisearch_health(url: str) -> bool:
    """检查 Meilisearch 健康状态"""
    try:
        import httpx

        response = await httpx.AsyncClient(timeout=5.0).get(f"{url}/health")
        return response.status_code == 200
    except Exception:
        return False


async def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(description="一键初始化 Embedding Service 环境")
    parser.add_argument("--verify-only", action="store_true", help="仅验证环境，不重新初始化")
    parser.add_argument("--skip-db", action="store_true", help="跳过 SurrealDB 初始化")
    parser.add_argument("--skip-meili", action="store_true", help="跳过 Meilisearch 初始化")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Embedding Service 一键初始化")
    logger.info("=" * 60)
    logger.info("")

    # 从环境变量读取配置
    surreal_url = os.getenv("SURREAL_URL", "ws://localhost:18002")
    surreal_ns = os.getenv("SURREAL_NS", "memory_ns")
    surreal_db = os.getenv("SURREAL_DB", "memory_db")

    meili_url = os.getenv("WRAPPER_MEILI_URL", "http://localhost:7700")
    meili_api_key = os.getenv("WRAPPER_MEILI_API_KEY")
    meili_index = os.getenv("WRAPPER_MEILI_INDEX_NAME", "memories")

    logger.info("配置:")
    logger.info("  SurrealDB: %s (%s.%s)", surreal_url, surreal_ns, surreal_db)
    logger.info("  Meilisearch: %s (index: %s)", meili_url, meili_index)
    logger.info("")

    # 1. 检查服务健康状态
    logger.info("🔍 检查服务健康状态...")

    surreal_healthy = await check_surrealdb_health(surreal_url)
    if surreal_healthy:
        logger.info("  ✅ SurrealDB 运行正常")
    else:
        logger.warning("  ⚠️  SurrealDB 未运行或无法连接")
        logger.warning("  请先启动 SurrealDB:")
        logger.warning("    docker-compose up surrealdb")
        logger.warning("    或")
        logger.warning("    surreal start --log trace")

    meili_healthy = await check_meilisearch_health(meili_url)
    if meili_healthy:
        logger.info("  ✅ Meilisearch 运行正常")
    else:
        logger.warning("  ⚠️  Meilisearch 未运行或无法连接")
        logger.warning("  请先启动 Meilisearch:")
        logger.warning("    docker-compose up meilisearch")

    logger.info("")

    # 如果验证模式，仅检查健康状态
    if args.verify_only:
        logger.info("=" * 60)
        logger.info("✅ 验证完成!")
        logger.info("=" * 60)
        return

    # 2. 初始化 SurrealDB
    if not args.skip_db:
        if not surreal_healthy:
            logger.error("❌ SurrealDB 未运行，跳过数据库初始化")
            logger.error("请先启动 SurrealDB:")
            logger.error("  docker-compose up surrealdb")
            sys.exit(1)

        logger.info("=" * 60)
        logger.info("1️⃣  初始化 SurrealDB 数据库")
        logger.info("=" * 60)

        # 调用数据库初始化脚本
        script_path = Path(__file__).parent / "init_database.py"
        cmd = [sys.executable, "-m", "scripts.init_database"]
        env = os.environ.copy()

        try:
            result = subprocess.run(cmd, env=env, check=True, text=True)
            logger.info(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error("❌ SurrealDB 初始化失败:")
            logger.error(e.stderr)
            sys.exit(1)

        logger.info("")
    else:
        logger.info("⏭️  跳过 SurrealDB 初始化")

    # 3. 初始化 Meilisearch
    if not args.skip_meili:
        if not meili_healthy:
            logger.error("❌ Meilisearch 未运行，跳过索引初始化")
            logger.error("请先启动 Meilisearch:")
            logger.error("  docker-compose up meilisearch")
            sys.exit(1)

        logger.info("=" * 60)
        logger.info("2️⃣  初始化 Meilisearch 索引")
        logger.info("=" * 60)

        # 调用 Meilisearch 初始化脚本
        script_path = Path(__file__).parent / "init_meilisearch.py"
        cmd = [sys.executable, "-m", "scripts.init_meilisearch"]
        env = os.environ.copy()

        try:
            result = subprocess.run(cmd, env=env, check=True, text=True)
            logger.info(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error("❌ Meilisearch 初始化失败:")
            logger.error(e.stderr)
            sys.exit(1)

        logger.info("")
    else:
        logger.info("⏭️  跳过 Meilisearch 初始化")

    # 4. 完成
    logger.info("=" * 60)
    logger.info("✅ 初始化完成!")
    logger.info("=" * 60)
    logger.info("")
    logger.info("下一步:")
    logger.info("  1. 启动包装服务:")
    logger.info("     uv run python -m wrapper.src.main")
    logger.info("")
    logger.info("  2. 或者使用 docker-compose:")
    logger.info("     docker-compose up wrapper")
    logger.info("")
    logger.info("  3. 运行测试:")
    logger.info("     uv run pytest tests/ -v")
    logger.info("")


if __name__ == "__main__":
    asyncio.run(main())
