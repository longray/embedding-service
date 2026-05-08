#!/usr/bin/env python3
"""检查 SurrealDB 中的 atom 表定义和数据"""
import asyncio
from surrealdb import AsyncSurreal


async def main():
    db = AsyncSurreal("ws://localhost:18002/rpc")
    await db.connect()
    await db.signin({"username": "root", "password": "root"})
    await db.use("memory_ns", "memory_db")
    
    # 检查 atom 表定义
    result = await db.query("INFO FOR TABLE atom")
    print("Atom table definition:")
    print(result)
    
    # 检查一个具体的 atom
    atom = await db.query("SELECT * FROM atom WHERE name = '错误处理模式' LIMIT 1")
    print("\n具体 atom 数据:")
    print(atom)
    
    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
