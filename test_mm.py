import asyncio
import sys

sys.path.insert(0, "D:\\embedding_service")

from wrapper.src.utils.memory_manager import MemoryManager


async def test():
    # Create a minimal mock
    class MockDB:
        async def query(self, q, params):
            print(f"Query: {q[:50]}...")
            print(f"Params: {params}")
            return [{"total_files": 22}]

    mm = MemoryManager.__new__(MemoryManager)
    mm._db = MockDB()

    # Test the method
    result = await mm.get_project_stats("test-project", "default")
    print("Result:", result)


asyncio.run(test())
