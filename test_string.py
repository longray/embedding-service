import asyncio
from surrealdb import AsyncSurreal


async def test():
    db = AsyncSurreal("ws://localhost:18002/rpc")
    await db.connect()
    await db.signin({"username": "root", "password": "root"})
    await db.use("memory_ns", "memory_db")

    # Test with string
    result = await db.query("CREATE audit_log SET action = 'test', details = '{\"test\": 1}'")
    print(f"Result: {result}")

    await db.close()


asyncio.run(test())
