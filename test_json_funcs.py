import asyncio
from surrealdb import AsyncSurreal


async def test():
    db = AsyncSurreal("ws://localhost:18002/rpc")
    await db.connect()
    await db.signin({"username": "root", "password": "root"})
    await db.use("memory_ns", "memory_db")

    # Test different JSON functions
    functions = [
        "object::from_json",
        "type::from_json",
        "json::parse",
        "parse::json",
        "from_json",
    ]

    for func in functions:
        try:
            result = await db.query(f'RETURN {func}("{{\\"test\\": 1}}")')
            print(f"{func}: {result}")
        except Exception as e:
            print(f"{func}: {str(e)[:50]}")

    await db.close()


asyncio.run(test())
