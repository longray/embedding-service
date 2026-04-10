"""检查 Lookup API 相关数据状态"""

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

    # 检查 source_id 和 file_path 的分布
    print("=== 字段分布统计 ===")

    # 检查 source_id
    result = await db.query("SELECT count() as total, count(source_id) as has_source_id FROM memory")
    print(f"总记录数: {result[0].get('total', 0) if result else 0}")
    print(f"有 source_id: {result[0].get('has_source_id', 0) if result else 0}")

    # 检查 file_path
    result = await db.query("SELECT count() as total, count(file_path) as has_file_path FROM memory")
    print(f"有 file_path: {result[0].get('has_file_path', 0) if result else 0}")

    # 检查测试数据
    print("\n=== 测试数据检查 ===")
    result = await db.query(
        "SELECT id, source_id, file_path, project_id, tenant_id FROM memory WHERE source_id IS NOT NONE LIMIT 5"
    )
    records = result if result else []
    print(f"有 source_id 的记录: {len(records)} 条")
    for r in records:
        if isinstance(r, dict):
            print(f"  ID: {r.get('id')}")
            print(f"    source_id: {r.get('source_id')}")
            print(f"    file_path: {r.get('file_path')}")
            print(f"    project_id: {r.get('project_id')}")
            print(f"    tenant_id: {r.get('tenant_id')}")
            print()

    # 检查所有记录
    print("\n=== 所有记录样本 ===")
    result = await db.query("SELECT id, source_id, file_path, project_id, tenant_id FROM memory LIMIT 5")
    records = result if result else []
    print(f"总记录数: {len(records)}")
    for r in records:
        if isinstance(r, dict):
            print(f"  ID: {r.get('id')}")
            print(f"    source_id: {r.get('source_id')}")
            print(f"    file_path: {r.get('file_path')}")
            print(f"    project_id: {r.get('project_id')}")
            print(f"    tenant_id: {r.get('tenant_id')}")
            print()

    await db.close()


asyncio.run(check())
