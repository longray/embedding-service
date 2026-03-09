"""记忆管理器模块

封装记忆的批量上传、搜索等业务逻辑。
不使用熔断器，依赖HTTP连接池的重试机制。
"""

import asyncio
from typing import Any, Optional

from surrealdb import AsyncSurreal

from .http_pool import get_http_pool
from .exceptions import EmbeddingError, DatabaseError, ValidationError


class MemoryManager:
    """记忆管理器，协调embedding服务和数据库操作"""

    def __init__(
        self,
        db: AsyncSurreal,
        embedding_service_url: str,
        batch_size: int = 10,
    ) -> None:
        self._db = db
        self._embedding_service_url = embedding_service_url
        self._batch_size = batch_size
        self._http_pool: Optional[Any] = None

    async def _get_http_pool(self):
        """延迟初始化HTTP连接池"""
        if self._http_pool is None:
            self._http_pool = await get_http_pool()
        return self._http_pool

    async def close(self) -> None:
        """关闭资源"""
        pass

    async def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """批量获取文本的embedding向量"""
        try:
            http_pool = await self._get_http_pool()
            response = await http_pool.post(
                f"{self._embedding_service_url}/v1/embeddings",
                json={"input": texts, "model": "Qwen3-Embedding-0.6B"},
            )
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data["data"]]
        except Exception as e:
            raise EmbeddingError(f"Failed to get embeddings: {str(e)}")

    async def upload_memories(self, memories: list[dict[str, Any]]) -> dict[str, Any]:
        """批量上传记忆"""
        if not memories:
            raise ValidationError("Memories list cannot be empty")

        total = len(memories)
        success_count = 0
        failed_count = 0
        memory_ids = []
        errors = []

        texts = [m.get("content", "") for m in memories]

        try:
            embeddings = await self._get_embeddings(texts)
        except EmbeddingError as e:
            return {"total": total, "success": 0, "failed": total, "memory_ids": [], "errors": [str(e)]}

        for memory, embedding in zip(memories, embeddings):
            try:
                memory_data = {
                    "content": memory.get("content", ""),
                    "embedding": embedding,
                    "metadata": memory.get("metadata", {}),
                }
                result = await self._db.create("memory", memory_data)
                if result:
                    if isinstance(result, list) and len(result) > 0:
                        memory_ids.append(str(result[0].get("id", "")))
                    elif isinstance(result, dict):
                        memory_ids.append(str(result.get("id", "")))
                    success_count += 1
                else:
                    failed_count += 1
                    errors.append("Empty result from database")
            except Exception as e:
                failed_count += 1
                errors.append(f"{type(e).__name__}: {str(e)}")

        result = {"total": total, "success": success_count, "failed": failed_count, "memory_ids": memory_ids}
        if errors:
            result["errors"] = errors[:10]
        return result

    async def search_memories(
        self, query: str, mode: str = "hybrid", limit: int = 10, threshold: float = 0.7
    ) -> dict[str, Any]:
        """搜索记忆"""
        if mode not in ("vector", "keyword", "hybrid"):
            raise ValidationError(f"Invalid search mode: {mode}")

        try:
            if mode == "vector":
                embeddings = await self._get_embeddings([query])
                results = await self._search_by_vector(embeddings[0], limit, threshold)
            elif mode == "keyword":
                results = await self._search_by_keyword(query, limit)
            else:
                embeddings = await self._get_embeddings([query])
                results = await self._hybrid_search(query, embeddings[0], limit, threshold)

            return {"results": results, "total": len(results), "mode": mode, "query": query}
        except Exception as e:
            raise DatabaseError(f"Search failed: {str(e)}")

    async def _search_by_vector(self, embedding, limit, threshold):
        q = """
            SELECT id, content, metadata,
                   vector::similarity::cosine(embedding, $embedding) AS score
            FROM memory
            WHERE vector::similarity::cosine(embedding, $embedding) > $threshold
            ORDER BY score DESC LIMIT $limit
        """
        result = await self._db.query(q, {"embedding": embedding, "threshold": threshold, "limit": limit})
        return self._format_results(result)

    async def _search_by_keyword(self, query_text, limit):
        q = """
            SELECT id, content, metadata
            FROM memory WHERE content CONTAINS $query LIMIT $limit
        """
        result = await self._db.query(q, {"query": query_text, "limit": limit})
        return self._format_results(result, default_score=0.5)

    async def _hybrid_search(self, query_text, embedding, limit, threshold):
        tasks = [
            self._search_by_vector(embedding, limit * 2, threshold),
            self._search_by_keyword(query_text, limit * 2),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        vector_results = [] if isinstance(results[0], Exception) else results[0]
        keyword_results = [] if isinstance(results[1], Exception) else results[1]

        seen = set()
        merged = []
        for item in vector_results:
            if item.get("id") not in seen:
                seen.add(item["id"])
                merged.append(item)
        for item in keyword_results:
            if item.get("id") not in seen:
                seen.add(item["id"])
                item["score"] = 0.5
                merged.append(item)
        merged.sort(key=lambda x: x.get("score", 0), reverse=True)
        return merged[:limit]

    def _format_results(self, db_result, default_score=None):
        results = []
        if not db_result or not isinstance(db_result, list):
            return results
        for item in db_result:
            if isinstance(item, dict):
                formatted = {
                    "id": str(item.get("id", "")),
                    "content": item.get("content", ""),
                    "metadata": item.get("metadata", {}),
                }
                if "score" in item:
                    formatted["score"] = item["score"]
                elif default_score is not None:
                    formatted["score"] = default_score
                results.append(formatted)
            elif isinstance(item, list):
                for record in item:
                    if isinstance(record, dict):
                        formatted = {
                            "id": str(record.get("id", "")),
                            "content": record.get("content", ""),
                            "metadata": record.get("metadata", {}),
                        }
                        if "score" in record:
                            formatted["score"] = record["score"]
                        elif default_score is not None:
                            formatted["score"] = default_score
                        results.append(formatted)
        return results
