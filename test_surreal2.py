import asyncio
from surrealdb import AsyncSurreal


async def test():
    db = AsyncSurreal("ws://localhost:18002/rpc")
    await db.connect()
    await db.signin({"username": "root", "password": "root"})
    await db.use("memory_ns", "memory_db")

    # Test stats query
    stats_query = """
        SELECT
            metadata.code_analysis.complexity.function_count AS function_count,
            metadata.code_analysis.complexity.class_count AS class_count,
            metadata.code_analysis.complexity.cyclomatic_complexity AS complexity
        FROM memory
        WHERE type = "code"
            AND project_id = "test-project"
            AND tenant_id = "default"
            AND metadata.code_analysis IS NOT NONE
    """
    result = await db.query(stats_query)
    print("Stats result:", result)
    print("Type:", type(result))
    if result:
        print("First record:", result[0] if isinstance(result, list) else result)

    await db.close()


asyncio.run(test())
