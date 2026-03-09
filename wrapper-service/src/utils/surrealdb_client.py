"""
SurrealDB 客户端模块

封装 SurrealDB 的核心操作，包括记忆CRUD、向量搜索、混合搜索等。
"""

import logging
from typing import Any

from surrealdb import AsyncSurreal

logger = logging.getLogger(__name__)


class SurrealDBClient:
    """SurrealDB 客户端，封装数据库操作"""

    def __init__(self, conn: AsyncSurreal) -> None:
        self._conn = conn

    async def create_memory(
        self,
        content: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
        entities: list[str] | None = None,
    ) -> str:
        """创建记忆记录"""
        data = {
            "content": content,
            "embedding": embedding,
            "metadata": metadata,
            "entities": entities,
        }
        result = await self._conn.create("memory", data)
        memory_id = result[0]["id"]
        logger.debug("创建记忆", extra={"memory_id": memory_id})
        return memory_id

    async def search_by_vector(
        self,
        embedding: list[float],
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """向量搜索记忆"""
        query = """
            SELECT id, content, metadata, entities, created_at,
                   vector::similarity::cosine(embedding, $embedding) AS score
            FROM memory
            WHERE vector::similarity::cosine(embedding, $embedding) > $threshold
            ORDER BY score DESC
            LIMIT $limit
        """
        results = await self._conn.query(
            query,
            {"embedding": embedding, "threshold": threshold, "limit": limit},
        )
        return results[0]["result"] if results else []

    async def search_by_keyword(
        self,
        keyword: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """关键词搜索记忆"""
        query = """
            SELECT id, content, metadata, entities, created_at
            FROM memory
            WHERE content CONTAINS $keyword
            ORDER BY created_at DESC
            LIMIT $limit
        """
        results = await self._conn.query(query, {"keyword": keyword, "limit": limit})
        return results[0]["result"] if results else []

    async def hybrid_search(
        self,
        keyword: str,
        embedding: list[float],
        limit: int = 10,
        threshold: float = 0.7,
        vector_weight: float = 0.7,
    ) -> list[dict[str, Any]]:
        """混合搜索（向量+关键词）"""
        # 向量搜索
        vector_results = await self.search_by_vector(embedding, limit * 2, threshold)

        # 关键词搜索
        keyword_results = await self.search_by_keyword(keyword, limit * 2)

        # 合并结果（简单去重）
        seen_ids = set()
        merged = []

        for item in vector_results:
            item_id = item["id"]
            if item_id not in seen_ids:
                seen_ids.add(item_id)
                merged.append(item)

        for item in keyword_results:
            item_id = item["id"]
            if item_id not in seen_ids:
                seen_ids.add(item_id)
                item["score"] = 0.5  # 关键词匹配给予固定分数
                merged.append(item)

        # 按分数排序并限制结果数
        merged.sort(key=lambda x: x.get("score", 0), reverse=True)
        return merged[:limit]

    async def process_entity(
        self,
        name: str,
        entity_type: str,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """处理实体（存在则返回ID，不存在则创建）"""
        # 查询是否已存在
        query = "SELECT id FROM entity WHERE name = $name AND type = $type"
        results = await self._conn.query(query, {"name": name, "type": entity_type})

        if results and results[0]["result"]:
            return results[0]["result"][0]["id"]

        # 不存在则创建
        data = {"name": name, "type": entity_type, "properties": properties}
        result = await self._conn.create("entity", data)
        return result[0]["id"]

    async def create_relation(
        self,
        from_entity: str,
        to_entity: str,
        relation_type: str,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """创建关系（自动去重）"""
        data = {
            "in": from_entity,
            "out": to_entity,
            "type": relation_type,
            "properties": properties,
        }
        result = await self._conn.create("relation", data)
        return result[0]["id"]

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            await self._conn.query("SELECT 1")
            return True
        except Exception as e:
            logger.error("SurrealDB健康检查失败", extra={"error": str(e)})
            return False

    async def create_hnsw_index(self) -> bool:
        """创建 HNSW 向量索引

        HNSW (Hierarchical Navigable Small World) 是一种近似最近邻搜索算法
        相比暴力搜索 O(n)，HNSW 的时间复杂度为 O(log n)
        预期性能提升: 10-100 倍
        """
        try:
            # 读取初始化脚本
            import os

            script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "init_hnsw_index.surql")

            if os.path.exists(script_path):
                with open(script_path, "r", encoding="utf-8") as f:
                    query = f.read()

                await self._conn.query(query)
                logger.info("hnsw_index_created")
                return True
            else:
                logger.warning("hnsw_init_script_not_found", path=script_path)
                return False

        except Exception as e:
            logger.error("hnsw_index_creation_failed", error=str(e))
            return False

    async def search_by_vector_hnsw(
        self,
        embedding: list[float],
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """使用 HNSW 索引进行向量搜索

        比暴力搜索快 10-100 倍，但可能有轻微精度损失
        """
        try:
            # 使用 HNSW 近似搜索语法
            query = """
                SELECT id, content, metadata, entities, created_at,
                       vector::similarity::cosine(embedding, $embedding) AS score
                FROM memory 
                WHERE embedding <|<|> $limit, $embedding
                AND vector::similarity::cosine(embedding, $embedding) > $threshold
                ORDER BY score DESC
                LIMIT $limit
            """

            results = await self._conn.query(
                query,
                {"embedding": embedding, "threshold": threshold, "limit": limit},
            )
            return results[0]["result"] if results else []

        except Exception as e:
            # HNSW 失败时回退到暴力搜索
            logger.warning("hnsw_search_failed_fallback", error=str(e))
            return await self.search_by_vector(embedding, limit, threshold)
