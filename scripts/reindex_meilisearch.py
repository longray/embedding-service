#!/usr/bin/env python3
"""重新索引 Meilisearch - 解决 local_id 格式问题"""

import asyncio
import os
from surrealdb import AsyncSurreal
from meilisearch_python_sdk import AsyncClient


async def reindex_meilisearch():
    """清空 Meilisearch 并重新从 SurrealDB 索引数据"""
    
    # 连接 SurrealDB
    surreal_db = AsyncSurreal("ws://localhost:18002/rpc")
    await surreal_db.connect()
    await surreal_db.signin({"username": "root", "password": "root"})
    await surreal_db.use("memory_ns", "memory_db")
    
    # 连接 Meilisearch
    meili_url = os.getenv("WRAPPER_MEILI_URL", "http://localhost:7700")
    meili_api_key = os.getenv("WRAPPER_MEILI_API_KEY", "")
    meili_client = AsyncClient(meili_url, api_key=meili_api_key)
    
    try:
        # 获取索引
        index_name = os.getenv("WRAPPER_MEILI_INDEX_NAME", "memories")
        index = meili_client.index(index_name)
        
        # 清空索引
        print("=== 清空 Meilisearch 索引 ===")
        await index.delete_all_documents()
        print("✅ 索引已清空")
        
        # 从 SurrealDB 获取所有记忆
        print("\n=== 从 SurrealDB 获取数据 ===")
        result = await surreal_db.query(
            "SELECT * FROM memory WHERE tenant_id = 'default' LIMIT 10000"
        )
        
        if not result or len(result) == 0:
            print("❌ 没有数据需要索引")
            return
        
        records = result[0]
        print(f"找到 {len(records)} 条记录")
        
        # 准备 Meilisearch 文档
        documents = []
        for record in records:
            doc = {
                "id": str(record.get("id")),
                "content": record.get("content", ""),
                "abstract": record.get("abstract", ""),
                "overview": record.get("overview", {}),
                "local_id": record.get("local_id"),
                "file_path": record.get("file_path"),
                "metadata": record.get("metadata", {}),
                "type": record.get("type", "general"),
                "tags": record.get("tags", []),
                "project_id": record.get("project_id", "global"),
                "tenant_id": record.get("tenant_id", "default"),
            }
            documents.append(doc)
        
        # 批量添加到 Meilisearch
        print(f"\n=== 索引 {len(documents)} 条记录到 Meilisearch ===")
        await index.add_documents(documents)
        print("✅ 索引完成")
        
        # 验证索引
        print("\n=== 验证索引 ===")
        stats = await index.get_stats()
        print(f"索引文档数: {stats.number_of_documents}")
        
    finally:
        await surreal_db.close()
        await meili_client.aclose()


if __name__ == "__main__":
    asyncio.run(reindex_meilisearch())
