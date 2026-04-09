"""Test SurrealDB object field with Python dict - Research validation"""

import asyncio
from surrealdb import AsyncSurreal


async def test_object_field():
    """Test different approaches to store Python dict in SurrealDB object field"""

    db = AsyncSurreal("ws://localhost:18002/rpc")
    await db.connect()
    await db.signin({"username": "root", "password": "root"})
    await db.use("memory_ns", "memory_db")

    print("=" * 60)
    print("Testing SurrealDB object field with Python dict")
    print("=" * 60)

    # Test 1: INSERT with raw SurrealQL (no parameter binding)
    print("\n1. Testing INSERT with raw SurrealQL...")
    try:
        result = await db.query("""
            INSERT INTO audit_log {
                timestamp: time::now(),
                action: "test_raw_insert",
                details: {content_length: 100, language: "python"},
                tenant_id: "test"
            }
        """)
        print(f"✅ Raw INSERT success: {result}")
    except Exception as e:
        print(f"❌ Raw INSERT failed: {e}")

    # Test 2: INSERT with parameter binding (the problematic case)
    print("\n2. Testing INSERT with parameter binding...")
    try:
        result = await db.query(
            """
            INSERT INTO audit_log {
                timestamp: time::now(),
                action: $action,
                details: $details,
                tenant_id: $tenant_id
            }
        """,
            {
                "action": "test_param_insert",
                "details": {"content_length": 200, "language": "javascript"},
                "tenant_id": "test",
            },
        )
        print(f"✅ Param INSERT success: {result}")
    except Exception as e:
        print(f"❌ Param INSERT failed: {e}")

    # Test 3: Using db.insert() method (recommended for objects)
    print("\n3. Testing db.insert() method...")
    try:
        result = await db.insert(
            "audit_log",
            {
                "timestamp": "2026-04-09T10:00:00Z",
                "action": "test_db_insert",
                "details": {"content_length": 300, "language": "rust"},
                "tenant_id": "test",
            },
        )
        print(f"✅ db.insert() success: {result}")
    except Exception as e:
        print(f"❌ db.insert() failed: {e}")

    # Test 4: Using db.create() method
    print("\n4. Testing db.create() method...")
    try:
        result = await db.create(
            "audit_log",
            {
                "timestamp": "2026-04-09T11:00:00Z",
                "action": "test_db_create",
                "details": {"content_length": 400, "language": "go"},
                "tenant_id": "test",
            },
        )
        print(f"✅ db.create() success: {result}")
    except Exception as e:
        print(f"❌ db.create() failed: {e}")

    # Verify all records
    print("\n5. Verifying all inserted records...")
    try:
        result = await db.query("SELECT * FROM audit_log WHERE tenant_id = 'test' ORDER BY timestamp DESC")
        records = result[0] if result else []
        print(f"Found {len(records)} test records:")
        for r in records:
            action = r.get("action", "unknown")
            details = r.get("details", {})
            print(f"  - {action}: details={details}")
    except Exception as e:
        print(f"❌ Query failed: {e}")

    # Cleanup
    print("\n6. Cleaning up test records...")
    try:
        await db.query("DELETE FROM audit_log WHERE tenant_id = 'test'")
        print("✅ Cleanup complete")
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")

    await db.close()

    print("\n" + "=" * 60)
    print("Test completed")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_object_field())
