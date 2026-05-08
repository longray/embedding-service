#!/usr/bin/env python3
import asyncio
from surrealdb import Surreal


async def main():
    async with Surreal("ws://localhost:18002/rpc") as db:
        await db.signin({"user": "root", "pass": "root"})
        await db.use("memory_ns", "memory_db")

        print("=== Atom 表索引 ===")
        result = await db.query("INFO FOR TABLE atom")
        print(result)

        print("\n=== Atom 数量 ===")
        count = await db.query("SELECT count() FROM atom GROUP BY count")
        print(count)

        print("\n=== entity_id 为空的 Atom ===")
        empty_entity = await db.query("SELECT count() FROM atom WHERE entity_id IS NONE GROUP BY count")
        print(empty_entity)


if __name__ == "__main__":
    asyncio.run(main())
