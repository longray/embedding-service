#!/usr/bin/env python3
"""检查 SurrealDB 中的 local_id 格式"""
import asyncio
from surrealdb import AsyncSurreal


async def main():
    db = AsyncSurreal("ws://localhost:18002/rpc")
    try:
        await db.connect()
        await db.signin({"username": "root", "password": "root"})
        await db.use("memory_ns", "memory_db")

        print("=== 检查 '错误处理模式' 的 local_id ===")
        result = await db.query("SELECT local_id, name, entity_id FROM atom WHERE name = '错误处理模式' LIMIT 5")
        print(result)

        print("\n=== 检查所有不同的 local_id 格式（前20个） ===")
        all_ids = await db.query("SELECT local_id FROM atom LIMIT 20")
        print(all_ids)

        print("\n=== 检查 entity_id 为空的 atom 数量 ===")
        empty_count = await db.query("SELECT count() FROM atom WHERE entity_id IS NONE GROUP BY count")
        print(f"entity_id 为空的 atom 数量: {empty_count}")

        print("\n=== 检查 entity_id 不为空的 atom 数量 ===")
        filled_count = await db.query("SELECT count() FROM atom WHERE entity_id IS NOT NONE GROUP BY count")
        print(f"entity_id 不为空的 atom 数量: {filled_count}")
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
