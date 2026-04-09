"""
迁移脚本: 将 metadata.file_path 复制到顶层 file_path 字段
用于修复 Lookup API 查询问题

用法:
    uv run python scripts/migrate_file_path.py

环境变量:
    SURREAL_URL          SurrealDB WebSocket URL (默认: ws://localhost:18002)
    SURREAL_NS           命名空间 (默认: memory_ns)
    SURREAL_DB           数据库 (默认: memory_db)
    SURREAL_USER         用户名 (默认: root)
    SURREAL_PASS         密码 (默认: root)
"""

import asyncio
import logging
import os
from typing import Any

from surrealdb import AsyncSurreal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("migrate_file_path")


async def migrate():
    """迁移 file_path 从 metadata 到顶层字段"""

    # 连接 SurrealDB
    url = os.getenv("SURREAL_URL", "ws://localhost:18002/rpc")
    ns = os.getenv("SURREAL_NS", "memory_ns")
    db_name = os.getenv("SURREAL_DB", "memory_db")
    user = os.getenv("SURREAL_USER") or "root"
    password = os.getenv("SURREAL_PASS") or "root"

    logger.info(f"连接 SurrealDB: {url}")

    db: Any = AsyncSurreal(url)
    await db.connect()
    await db.signin({"username": user, "password": password})
    await db.use(ns, db_name)

    logger.info("开始迁移 file_path...")

    # 查询所有有 metadata.file_path 但没有顶层 file_path 的记录
    result = await db.query(
        """
        SELECT id, metadata FROM memory
        WHERE metadata.file_path IS NOT NONE
        AND (file_path IS NONE OR file_path = "")
        """
    )

    records = result if result and isinstance(result, list) else []
    logger.info(f"找到 {len(records)} 条需要迁移的记录")

    updated = 0
    for record in records:
        if not isinstance(record, dict):
            continue

        memory_id = record.get("id")
        metadata = record.get("metadata", {})
        file_path = metadata.get("file_path")

        if file_path:
            await db.query("UPDATE $id SET file_path = $file_path", {"id": memory_id, "file_path": file_path})
            updated += 1
            if updated % 100 == 0:
                logger.info(f"已更新 {updated} 条记录...")

    logger.info(f"迁移完成: 共更新 {updated} 条记录")
    await db.close()


if __name__ == "__main__":
    asyncio.run(migrate())
