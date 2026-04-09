import asyncio
from surrealdb import AsyncSurreal


async def test():
    db = AsyncSurreal("ws://localhost:18002/rpc")
    await db.connect()
    await db.signin({"username": "root", "password": "root"})
    await db.use("memory_ns", "memory_db")

    # Test simple query
    result = await db.query(
        'SELECT count() AS total_files FROM memory WHERE type = "code" AND project_id = "test-project" AND tenant_id = "default" GROUP ALL'
    )
    print("Count result:", result)

    await db.close()


asyncio.run(test())
