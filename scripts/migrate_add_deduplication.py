import asyncio
import hashlib
from collections import defaultdict

from surrealdb import AsyncSurreal


async def main():
    db = AsyncSurreal("ws://localhost:18002/rpc")
    await db.connect()
    await db.signin({"username": "root", "password": "root"})
    await db.use("memory_ns", "memory_db")

    print("🔍 步骤1: 分析重复数据...")

    result = await db.query("SELECT id, content, tenant_id, created_at FROM memory;")

    groups = defaultdict(list)
    for record in result:
        content = record.get("content", "")
        tenant_id = record.get("tenant_id", "default")
        key = (tenant_id, content)
        groups[key].append(record)

    duplicates = {k: v for k, v in groups.items() if len(v) > 1}

    print(f"  - 总记录数: {len(result)}")
    print(f"  - 唯一内容: {len(groups)}")
    print(f"  - 重复组数: {len(duplicates)}")
    print(f"  - 重复记录: {sum(len(v) - 1 for v in duplicates.values())}")

    if not duplicates:
        print("\n✅ 无重复数据，跳过清理步骤")
    else:
        print(f"\n🗑️  步骤2: 清理 {len(duplicates)} 组重复数据...")
        deleted_count = 0

        for (tenant_id, content), records in duplicates.items():
            sorted_records = sorted(records, key=lambda r: r.get("created_at", ""))
            keep_id = sorted_records[0].get("id")

            for record in sorted_records[1:]:
                record_id = record.get("id")
                await db.query(f"DELETE {record_id};")
                deleted_count += 1

        print(f"  - 已删除: {deleted_count} 条重复记录")

    print("\n🔨 步骤3: 生成 content_hash...")
    await db.query("UPDATE memory SET content_hash = crypto::md5(content) WHERE content_hash = NONE;")
    print("  - 已为所有记录生成哈希")

    print("\n✅ 迁移完成！现在可以执行 add_deduplication.surql 应用 UNIQUE 索引")

    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
