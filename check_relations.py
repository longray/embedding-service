#!/usr/bin/env python3
"""检查调用关系"""

import asyncio
from surrealdb import AsyncSurreal


async def main():
    db = AsyncSurreal("ws://localhost:18002")
    await db.connect()
    await db.signin({"username": "root", "password": "root"})
    await db.use("memory_ns", "memory_db")

    # Check all calls relations
    print("=== 查询所有 calls 关系 ===")
    result = await db.query('SELECT * FROM memory_relation WHERE relationship_type = "calls" LIMIT 10')
    print(f"找到 {len(result)} 条 calls 关系")
    for r in result[:3]:
        print(f"  {r}")

    # Check memory records
    print("\n=== 查询 test-map-project 的代码文件 ===")
    result2 = await db.query(
        'SELECT id, metadata.file_path FROM memory WHERE type = "code" AND project_id = "test-map-project"'
    )
    print(f"找到 {len(result2)} 个代码文件")
    for r in result2:
        print(f"  {r.get('id')}: {r.get('metadata', {}).get('file_path')}")

    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
