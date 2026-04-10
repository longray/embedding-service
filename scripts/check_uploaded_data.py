"""检查刚上传的数据"""

import asyncio
import os
from surrealdb import AsyncSurreal


async def check():
    url = os.getenv("SURREAL_URL", "ws://localhost:18002/rpc")
    ns = os.getenv("SURREAL_NS", "memory_ns")
    db_name = os.getenv("SURREAL_DB", "memory_db")
    user = os.getenv("SURREAL_USER") or "root"
    password = os.getenv("SURREAL_PASS") or "root"

    db = AsyncSurreal(url)
    await db.connect()
    await db.signin({"username": user, "password": password})
    await db.use(ns, db_name)

    # 检查刚上传的数据
    result = await db.query(
        "SELECT id, source_id, file_path, project_id, tenant_id, metadata FROM memory WHERE source_id = 'self-test-source-001'"
    )

    print("=== 刚上传的数据 ===")
    records = result if result else []
    print(f"找到 {len(records)} 条记录")
    for r in records:
        if isinstance(r, dict):
            print(f"ID: {r.get('id')}")
            print(f"  source_id: {r.get('source_id')}")
            print(f"  file_path: {r.get('file_path')}")
            print(f"  project_id: {r.get('project_id')}")
            print(f"  tenant_id: {r.get('tenant_id')}")
            print(f"  metadata: {r.get('metadata')}")
            print()

    await db.close()


asyncio.run(check())
