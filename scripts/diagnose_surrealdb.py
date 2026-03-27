#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "wrapper" / "src"))

from surrealdb import AsyncSurreal


def fmt(v):
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if hasattr(v, "__class__") and v.__class__.__name__ == "RecordID":
        return f"RecordID({v})"
    if hasattr(v, "to_dict"):
        return v.to_dict()
    return str(type(v).__name__)


async def diagnose():
    url = "ws://localhost:18002/rpc"
    db = AsyncSurreal(url)
    await db.connect()
    await db.signin({"username": "root", "password": "root"})
    await db.use("memory_ns", "memory_db")
    print("✓ 连接成功\n")

    r = await db.query("SELECT id, source_id, content_hash, tenant_id FROM memory LIMIT 3")
    print(f"[Test 1] SELECT LIMIT 3")
    print(f"  type={type(r)}, len={len(r) if isinstance(r, list) else 'N/A'}")
    if r:
        print(f"  r[0] type={type(r[0])}")
        if isinstance(r[0], dict):
            for k, v in r[0].items():
                print(f"    {k}: {fmt(v)}")
        else:
            print(f"  raw[0]: {r[0]}")

    r2 = await db.query("SELECT count() as total FROM memory GROUP ALL")
    print(f"\n[Test 2] SELECT count()")
    print(f"  type={type(r2)}, value={r2}")

    r3 = await db.query(
        "SELECT source_id, content_hash FROM memory WHERE tenant_id=$tenant_id", {"tenant_id": "default"}
    )
    print(f"\n[Test 3] WHERE tenant_id='default' (param)")
    print(f"  type={type(r3)}, len={len(r3) if isinstance(r3, list) else 'N/A'}")
    if r3:
        print(f"  r3[0]: {r3[0]}")

    r4 = await db.query(
        "SELECT source_id, content_hash FROM memory WHERE tenant_id=$tenant_id", {"tenant_id": "longray"}
    )
    print(f"\n[Test 4] WHERE tenant_id='longray' (param)")
    print(f"  type={type(r4)}, len={len(r4) if isinstance(r4, list) else 'N/A'}")
    if r4:
        print(f"  r4[0]: {r4[0]}")

    await db.close()
    print("\n✓ 完成")


if __name__ == "__main__":
    asyncio.run(diagnose())
