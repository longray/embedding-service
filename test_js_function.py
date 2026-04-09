import asyncio
from surrealdb import AsyncSurreal


async def test():
    db = AsyncSurreal("ws://localhost:18002/rpc")
    await db.connect()
    await db.signin({"username": "root", "password": "root"})
    await db.use("memory_ns", "memory_db")

    # Create JavaScript function to parse JSON
    try:
        await db.query("DEFINE FUNCTION fn::parse_json(str) { RETURN function() { return JSON.parse(str); }; };")
        print("Function created")
    except Exception as e:
        print(f"Function creation error: {e}")

    # Test the function
    try:
        result = await db.query('RETURN fn::parse_json("{\\"test\\": 1}")')
        print(f"Function result: {result}")
    except Exception as e:
        print(f"Function test error: {e}")

    await db.close()


asyncio.run(test())
