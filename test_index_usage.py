#!/usr/bin/env python3
"""验证索引使用情况"""

import asyncio
from surrealdb import AsyncSurreal


async def main():
    db = AsyncSurreal("ws://localhost:18002")
    await db.connect()
    await db.signin({"username": "root", "password": "root"})
    await db.use("memory_ns", "memory_db")

    # 检查索引
    print("=== 检查索引 ===")
    result = await db.query("INFO FOR TABLE memory")
    if result and "indexes" in result[0]:
        indexes = result[0]["indexes"]
        print(f"找到 {len(indexes)} 个索引:")
        for name, defn in indexes.items():
            print(f"  - {name}")

    # EXPLAIN 分析查询
    print("\n=== EXPLAIN 分析项目地图查询 ===")
    explain_result = await db.query("""
        EXPLAIN SELECT id FROM memory 
        WHERE tenant_id = 'default'
          AND type = 'code'
          AND project_id = 'test-project'
        LIMIT 10
    """)
    print(f"EXPLAIN 结果: {explain_result}")

    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
