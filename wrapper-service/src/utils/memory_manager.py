"""
记忆管理器模块

封装记忆的批量上传、搜索等业务逻辑，协调embedding服务和SurrealDB。
"""

import asyncio
import logging
from typing import Any

import httpx

from .connection_pool import SurrealDBConnectionPool
from .surrealdb_client import SurrealDBClient

logger = logging.getLogger(__name__)


class MemoryManager:
    """记忆管理器，协调embedding服务和数据库操作"""

    def __init__(
        self,
        pool: SurrealDBConnectionPool,
        embedding_service_url: str,
        batch_size: int = 10,
    ) -> None:
        self._pool = pool
        self._embedding_service_url = embedding_service_url
        self._batch_size = batch_size
        self._http_client = httpx.AsyncClient(timeout=30.0)

    async def close(self) -> None:
        """关闭HTTP客户端"""
        await self._http_client.aclose()

    async def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """批量获取文本的embedding向量"""
        response = await self._http_client.post(
            f"{self._embedding_service_url}/v1/embeddings",
            json={"input": texts, "model": "Qwen3-Embedding-0.6B"},
        )
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]

    async def upload_memories(
        self,
        memories: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """批量上传记忆"""
        total = len(memories)
        success_count = 0
        failed_count = 0
        memory_ids = []

        # 提取所有文本内容
        texts = [m["content"] for m in memories]

        # 批量获取embeddings
        try:
            embeddings = await self._get_embeddings(texts)
        except Exception as e:
            logger.error("获取embeddings失败", extra={"error": str(e)})
            return {
                "total": total,
                "success": 0,
                "failed": total,
                "memory_ids": [],
                "error": str(e),
            }

        # 批量存储到数据库
        async with self._pool.acquire() as conn:
            client = SurrealDBClient(conn)

            for memory, embedding in zip(memories, embeddings):
                try:
                    # 处理实体
                    entity_ids = []
                    if "entities" in memory and memory["entities"]:
                        for entity in memory["entities"]:
                            entity_id = await client.process_entity(
                                name=entity["name"],
                                entity_type=entity["type"],
                                properties=entity.get("properties"),
                            )
                            entity_ids.append(entity_id)

                    # 创建记忆
                    memory_id = await client.create_memory(
                        content=memory["content"],
                        embedding=embedding,
                        metadata=memory.get("metadata"),
                        entities=entity_ids if entity_ids else None,
                    )
                    memory_ids.append(memory_id)
                    success_count += 1

                    # 处理关系
                    if "relations" in memory and memory["relations"]:
                        for relation in memory["relations"]:
                            await client.create_relation(
                                from_entity=relation["from"],
                                to_entity=relation["to"],
                                relation_type=relation["type"],
                                properties=relation.get("properties"),
                            )

                except Exception as e:
                    logger.error("存储记忆失败", extra={"error": str(e)})
                    failed_count += 1

        return {
            "total": total,
            "success": success_count,
            "failed": failed_count,
            "memory_ids": memory_ids,
        }

    async def search_memories(
        self,
        query: str,
        mode: str = "hybrid",
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """搜索记忆"""
        async with self._pool.acquire() as conn:
            client = SurrealDBClient(conn)

            if mode == "vector":
                # 向量搜索
                embeddings = await self._get_embeddings([query])
                return await client.search_by_vector(embeddings[0], limit, threshold)

            elif mode == "keyword":
                # 关键词搜索
                return await client.search_by_keyword(query, limit)

            elif mode == "hybrid":
                # 混合搜索
                embeddings = await self._get_embeddings([query])
                return await client.hybrid_search(query, embeddings[0], limit, threshold)

            else:
                raise ValueError(f"不支持的搜索模式: {mode}")
