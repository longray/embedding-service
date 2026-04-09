#!/usr/bin/env python3
"""测试 SurrealDB 查询"""

import asyncio
from surrealdb import AsyncSurreal

SURREAL_URL = "ws://localhost:18002/rpc"
SURREAL_NS = "memory_ns"
SURREAL_DB = "memory_db"
SURREAL_USER = "root"
SURREAL_PASS = "root"


async def test_query():
    db = AsyncSurreal(SURREAL_URL)
    await db.signin({"user": SURREAL_USER, "pass": SURREAL_PASS})
    await db.use(SURREAL_NS, SURREAL_DB)

    # 查询所有代码记忆
    result = await db.query("SELECT id, metadata FROM memory WHERE type = 'code' LIMIT 10")
    print("所有代码记忆:")
    print(result)

    # 查询特定 file_path
    result2 = await db.query(
        "SELECT id, metadata FROM memory WHERE type = 'code' AND project_id = $project_id AND metadata->file_path = $file_path LIMIT 1",
        {"project_id": "test-project", "file_path": "src/same.ts"},
    )
    print("\n特定 file_path 查询:")
    print(result2)

    await db.close()


if __name__ == "__main__":
    asyncio.run(test_query())
