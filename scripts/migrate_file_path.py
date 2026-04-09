"""
迁移脚本: 将 metadata.file_path 复制到顶层 file_path 字段
用于修复 Lookup API 查询问题
"""

import asyncio
import os
from surrealdb import Surreal


async def migrate():
    """迁移 file_path 从 metadata 到顶层字段"""

    # 连接 SurrealDB
    db = Surreal(os.getenv("SURREAL_URL", "ws://localhost:18002/rpc"))
    await db.connect()
    await db.use(os.getenv("SURREAL_NS", "memory_ns"), os.getenv("SURREAL_DB", "memory_db"))

    print("开始迁移 file_path...")

    # 查询所有有 metadata.file_path 但没有顶层 file_path 的记录
    result = await db.query("""
        SELECT id, metadata FROM memory 
        WHERE metadata.file_path IS NOT NONE 
        AND (file_path IS NONE OR file_path = "")
    """)

    records = result[0] if result else []
    print(f"找到 {len(records)} 条需要迁移的记录")

    updated = 0
    for record in records:
        memory_id = record.get("id")
        metadata = record.get("metadata", {})
        file_path = metadata.get("file_path")

        if file_path:
            await db.query("UPDATE $id SET file_path = $file_path", {"id": memory_id, "file_path": file_path})
            updated += 1
            if updated % 100 == 0:
                print(f"已更新 {updated} 条记录...")

    print(f"迁移完成: 共更新 {updated} 条记录")
    await db.close()


if __name__ == "__main__":
    asyncio.run(migrate())
