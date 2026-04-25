"""
清空后端所有记忆数据（SurrealDB + Meilisearch）

用法:
    uv run python scripts/clear_all_data.py

环境变量:
    SURREAL_URL          SurrealDB WebSocket URL (默认: ws://localhost:18002)
    SURREAL_NS           命名空间 (默认: memory_ns)
    SURREAL_DB           数据库 (默认: memory_db)
    SURREAL_USER         用户名 (默认: root)
    SURREAL_PASS         密码 (默认: root)
    WRAPPER_MEILI_URL    Meilisearch URL (默认: http://localhost:18003)
    WRAPPER_MEILI_API_KEY Meilisearch API Key
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

from surrealdb import AsyncSurreal

sys.path.insert(0, str(Path(__file__).parent.parent / "wrapper" / "src"))
from utils.meili_client import MeilisearchClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


async def clear_surrealdb():
    config = {
        "url": os.getenv("SURREAL_URL", "ws://localhost:18002"),
        "ns": os.getenv("SURREAL_NS", "memory_ns"),
        "db": os.getenv("SURREAL_DB", "memory_db"),
        "user": os.getenv("SURREAL_USER", "root"),
        "pass": os.getenv("SURREAL_PASS", "root"),
    }

    print(f"🗑️  清空 SurrealDB...")
    print(f"   URL: {config['url']}")
    print(f"   Namespace: {config['ns']}")
    print(f"   Database: {config['db']}")

    db = AsyncSurreal(config["url"])

    try:
        await db.connect()
        await db.signin({"username": config["user"], "password": config["pass"]})
        await db.use(config["ns"], config["db"])

        memories = await db.select("memories")
        count = len(memories) if memories else 0
        print(f"📊 SurrealDB 中有 {count} 条记忆")

        if count > 0:
            await db.query("DELETE memories;")
            print(f"✅ SurrealDB 已清空 {count} 条记录")

        await db.query("DELETE reference;")
        print(f"✅ SurrealDB 已清空所有关系")

        await db.query("DELETE conflict;")
        print(f"✅ SurrealDB 已清空所有冲突")

        await db.close()
        return True
    except Exception as e:
        print(f"❌ SurrealDB 清空失败: {e}")
        return False


async def clear_meilisearch():
    config = {
        "url": os.getenv("WRAPPER_MEILI_URL", "http://localhost:18003"),
        "api_key": os.getenv("WRAPPER_MEILI_API_KEY"),
        "index_name": "memories",
    }

    print(f"\n🗑️  清空 Meilisearch...")
    print(f"   URL: {config['url']}")
    print(f"   Index: {config['index_name']}")

    client = MeilisearchClient(
        url=config["url"],
        api_key=config["api_key"],
        index_name=config["index_name"],
    )

    try:
        await client.connect()

        stats = await client.get_stats()
        if "error" not in stats:
            doc_count = stats.get("numberOfDocuments", 0)
            print(f"📊 Meilisearch 中有 {doc_count} 个文档")

            if doc_count > 0:
                print("ℹ️  Meilisearch 文档删除需要删除并重建索引...")
                await client.delete_documents_by_filter("")
                print(f"✅ Meilisearch 已清空所有文档")
        else:
            print("ℹ️  Meilisearch 统计获取失败，尝试直接清空")
            await client.delete_documents_by_filter("")
            print("✅ Meilisearch 已清空")

        await client.close()
        return True
    except Exception as e:
        print(f"❌ Meilisearch 清空失败: {e}")
        return False


async def main():
    print("=" * 60)
    print("清空后端所有记忆数据")
    print("=" * 60)
    print()

    success = True

    db_success = await clear_surrealdb()
    success = success and db_success

    meili_success = await clear_meilisearch()
    success = success and meili_success

    print()
    print("=" * 60)
    if success:
        print("✅ 所有数据已清空完成！")
    else:
        print("⚠️  清空过程中遇到错误，请检查日志")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
