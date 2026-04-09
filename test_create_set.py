import asyncio
from surrealdb import AsyncSurreal


async def test():
    db = AsyncSurreal("ws://localhost:18002/rpc")
    await db.connect()
    await db.signin({"username": "root", "password": "root"})
    await db.use("memory_ns", "memory_db")

    # Test CREATE SET with object
    try:
        result = await db.query(
            "CREATE audit_log SET action = 'test', details = $details", {"details": {"content_length": 100}}
        )
        print(f"CREATE SET result type: {type(result)}")
        print(f"CREATE SET result: {result}")
    except Exception as e:
        print(f"CREATE SET error: {e}")

    await db.close()


asyncio.run(test())
