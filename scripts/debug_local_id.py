#!/usr/bin/env python3
"""调试 local_id 问题"""

import asyncio
from surrealdb import AsyncSurreal


async def debug_local_id():
    """查询 SurrealDB 中的 local_id"""
    db = AsyncSurreal("ws://localhost:18002/rpc")

    try:
        await db.connect()
        await db.signin({"username": "root", "password": "root"})
        await db.use("memory_ns", "memory_db")

        # 查询最近的 atoms
        print("=== 查询最近的 10 个 atoms ===")
        result = await db.query("SELECT id, local_id, name, type FROM atom LIMIT 10")

        if result and len(result) > 0:
            records = result[0]
            print(f"Records type: {type(records)}")
            if isinstance(records, list):
                for record in records[:3]:
                    if isinstance(record, dict):
                        local_id = record.get('local_id')
                        name = record.get('name', 'N/A')
                        print(f"ID: {record.get('id')}")
                        print(f"  local_id: {local_id} (长度: {len(local_id) if local_id else 0})")
                        print(f"  name: {name}")
                        print()
            elif isinstance(records, dict):
                print(f"Single record: {records}")

        # 查询特定名称
        print("\n=== 查询 '第1章：Promise 基础' ===")
        result = await db.query(
            "SELECT id, local_id, name FROM atom WHERE name = '第1章：Promise 基础' LIMIT 1"
        )

        if result and len(result) > 0:
            records = result[0]
            for record in records:
                print(f"ID: {record.get('id')}")
                print(f"  local_id: {record.get('local_id')}")
                print(f"  name: {record.get('name')}")

    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(debug_local_id())
