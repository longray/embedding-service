#!/usr/bin/env python3
"""SurrealDB 查询脚本模板 - 正确使用 AsyncSurreal"""
import asyncio
from surrealdb import AsyncSurreal


async def main():
    """查询 SurrealDB 中的 atom 数据"""
    # 1. 创建连接（不自动连接）
    db = AsyncSurreal("ws://localhost:18002/rpc")
    
    try:
        # 2. 显式连接
        await db.connect()
        
        # 3. 认证（使用 username/password，不是 user/pass）
        await db.signin({"username": "root", "password": "root"})
        
        # 4. 选择命名空间和数据库
        await db.use("memory_ns", "memory_db")
        
        # 5. 执行查询
        print("=== 查询 atom 表 ===")
        result = await db.query("SELECT * FROM atom LIMIT 5")
        print(result)
        
        print("\n=== 查询特定条件 ===")
        result = await db.query("SELECT local_id, name FROM atom WHERE name = '错误处理模式' LIMIT 1")
        print(result)
        
    finally:
        # 6. 关闭连接
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
