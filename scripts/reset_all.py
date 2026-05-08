"""重置所有数据 - 开发/测试环境一键重置

功能：
1. 清空 SurrealDB 所有数据（memory, atom, entity, reference, conflict）
2. 清空 Meilisearch 所有文档
3. 重新初始化 SurrealDB schema
4. 重新初始化 Meilisearch 索引

使用方式:
    uv run python scripts/reset_all.py
    uv run python scripts/reset_all.py --skip-db    # 跳过 SurrealDB
    uv run python scripts/reset_all.py --skip-meili # 跳过 Meilisearch
    uv run python scripts/reset_all.py --dry-run    # 仅预览，不执行

安全机制:
- 需要确认输入 "yes" 才能执行
- 生产环境建议先备份数据
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from wrapper.src.config import config
from wrapper.src.utils.meili_client import MeilisearchClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def clear_surrealdb():
    """清空 SurrealDB 所有数据"""
    from surrealdb import AsyncSurreal
    
    logger.info("[Reset] 连接 SurrealDB...")
    db = AsyncSurreal(config.surrealdb.url)
    
    try:
        await db.connect()
        await db.signin({
            "username": config.surrealdb.username,
            "password": config.surrealdb.password,
        })
        await db.use(config.surrealdb.namespace, config.surrealdb.database)
        
        # 获取统计
        tables = ["memory", "atom", "entity", "reference", "conflict"]
        counts = {}
        for table in tables:
            try:
                result = await db.query(f"SELECT count() AS total FROM {table} GROUP ALL")
                counts[table] = result[0].get("total", 0) if result else 0
            except Exception:
                counts[table] = 0
        
        total = sum(counts.values())
        logger.info(f"[Reset] 当前数据量: {counts}")
        
        if total == 0:
            logger.info("[Reset] SurrealDB 已经是空的")
            return
        
        # 清空数据
        logger.warning(f"[Reset] 正在清空 {total} 条记录...")
        for table in tables:
            if counts[table] > 0:
                await db.query(f"DELETE {table};")
                logger.info(f"[Reset] 已清空 {table}: {counts[table]} 条")
        
        logger.info("[Reset] SurrealDB 清空完成")
        
    finally:
        await db.close()


async def clear_meilisearch():
    """清空 Meilisearch 所有文档"""
    logger.info("[Reset] 连接 Meilisearch...")
    
    client = MeilisearchClient(
        url=config.meilisearch.url,
        api_key=config.meilisearch.api_key,
        index_name=config.meilisearch.index_name,
    )
    
    try:
        await client.connect()
        
        # 获取统计
        stats = await client.get_stats()
        doc_count = stats.get("numberOfDocuments", 0)
        
        if doc_count == 0:
            logger.info("[Reset] Meilisearch 已经是空的")
            return
        
        logger.warning(f"[Reset] 正在清空 {doc_count} 个文档...")
        await client.delete_all_documents()
        logger.info("[Reset] Meilisearch 清空完成")
        
    finally:
        await client.close()


async def init_surrealdb():
    """重新初始化 SurrealDB schema"""
    import subprocess
    
    logger.info("[Reset] 初始化 SurrealDB schema...")
    
    script_path = Path(__file__).parent / "init_database.py"
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        logger.error(f"[Reset] SurrealDB 初始化失败: {result.stderr}")
        raise RuntimeError("SurrealDB 初始化失败")
    
    logger.info("[Reset] SurrealDB schema 初始化完成")


async def init_meilisearch():
    """重新初始化 Meilisearch"""
    import subprocess
    
    logger.info("[Reset] 初始化 Meilisearch...")
    
    script_path = Path(__file__).parent / "init_meilisearch.py"
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        logger.error(f"[Reset] Meilisearch 初始化失败: {result.stderr}")
        raise RuntimeError("Meilisearch 初始化失败")
    
    logger.info("[Reset] Meilisearch 初始化完成")


async def main():
    parser = argparse.ArgumentParser(description="重置所有数据")
    parser.add_argument("--skip-db", action="store_true", help="跳过 SurrealDB")
    parser.add_argument("--skip-meili", action="store_true", help="跳过 Meilisearch")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不执行")
    parser.add_argument("--force", action="store_true", help="跳过确认提示")
    args = parser.parse_args()
    
    # 确认提示
    if not args.force and not args.dry_run:
        print("\n" + "=" * 60)
        print("⚠️  警告: 这将删除所有数据！")
        print("=" * 60)
        print("\n此操作将:")
        if not args.skip_db:
            print("  - 清空 SurrealDB: memory, atom, entity, reference, conflict")
            print("  - 重新初始化 SurrealDB schema")
        if not args.skip_meili:
            print("  - 清空 Meilisearch 所有文档")
            print("  - 重新初始化 Meilisearch 索引")
        print("\n数据将无法恢复！")
        print("=" * 60)
        
        confirm = input("\n输入 'yes' 确认重置: ")
        if confirm.lower() != "yes":
            print("已取消")
            return
    
    try:
        # 清空 SurrealDB
        if not args.skip_db:
            if args.dry_run:
                logger.info("[Dry Run] 将清空 SurrealDB")
            else:
                await clear_surrealdb()
                await init_surrealdb()
        
        # 清空 Meilisearch
        if not args.skip_meili:
            if args.dry_run:
                logger.info("[Dry Run] 将清空 Meilisearch")
            else:
                await clear_meilisearch()
                await init_meilisearch()
        
        if args.dry_run:
            logger.info("[Dry Run] 预览完成，未执行任何操作")
        else:
            logger.info("=" * 60)
            logger.info("✅ 重置完成！")
            logger.info("=" * 60)
            logger.info("所有数据已清空并重新初始化")
            logger.info("可以开始新的测试/开发工作")
            
    except Exception as e:
        logger.error(f"[Reset] 失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
