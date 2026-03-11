"""
简单的数据库连接和功能验证脚本
"""

import asyncio
from surrealdb import Surreal


async def test_database_connection():
    """测试数据库连接和基本操作"""
    print("=== 测试SurrealDB连接 ===")

    # 连接数据库
    db = Surreal("ws://localhost:18002/rpc")
    await db.connect()
    await db.signin({"user": "root", "pass": "root"})
    await db.use("memory_ns", "memory_db")

    print("✅ 数据库连接成功")

    # 测试创建记忆
    print("\n=== 测试创建记忆 ===")
    test_memory = {
        "content": "测试记忆内容",
        "embedding": [0.1] * 1024,  # 1024维向量
    }

    result = await db.create("memory", test_memory)
    print(f"创建结果: {result}")  # 调试：查看返回格式
    if result:
        print(f"✅ 创建记忆成功: {result if isinstance(result, str) else result}")

    # 测试查询记忆
    print("\n=== 测试查询记忆 ===")
    memories = await db.select("memory")
    print(f"✅ 查询成功，共 {len(memories)} 条记忆")

    # 清理测试数据
    print("\n=== 清理测试数据 ===")
    await db.delete("memory")
    print("✅ 清理完成")
    print("✅ 清理完成")

    await db.close()
    print("\n=== 所有测试通过 ===")


if __name__ == "__main__":
    asyncio.run(test_database_connection())
