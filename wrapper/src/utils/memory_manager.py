"""记忆管理器模块

封装记忆的批量上传、搜索等业务逻辑。
不使用熔断器，依赖HTTP连接池的重试机制。
"""

import asyncio
import inspect
import re
from typing import Any, Optional, cast


from .http_pool import get_http_pool
from .exceptions import EmbeddingError, DatabaseError, ValidationError


class MemoryManager:
    """记忆管理器，协调embedding服务和数据库操作"""

    def __init__(
        self,
        db: Any,
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

    async def _db_call(self, method_name: str, *args, **kwargs):
        method = getattr(self._db, method_name)
        result = method(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

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
                result = await self._db_call("create", "memory", memory_data)
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
        self,
        query: str,
        mode: str = "hybrid",
        limit: int = 10,
        threshold: float = 0.7,
        metadata_filters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """搜索记忆"""
        if mode not in ("vector", "keyword", "hybrid"):
            raise ValidationError(f"Invalid search mode: {mode}")

        try:
            if mode == "vector":
                embeddings = await self._get_embeddings([query])
                results = await self._search_by_vector(embeddings[0], limit, threshold, metadata_filters)
            elif mode == "keyword":
                results = await self._search_by_keyword(query, limit, metadata_filters)
            else:
                embeddings = await self._get_embeddings([query])
                results = await self._hybrid_search(query, embeddings[0], limit, threshold, metadata_filters)

            results = self._apply_metadata_filters(results, metadata_filters)
            results = results[:limit]

            return {"results": results, "total": len(results), "mode": mode, "query": query}
        except Exception as e:
            raise DatabaseError(f"Search failed: {str(e)}")

    def _apply_metadata_filters(
        self,
        results: list[dict[str, Any]],
        metadata_filters: Optional[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not metadata_filters:
            return results

        filtered: list[dict[str, Any]] = []
        for record in results:
            metadata = record.get("metadata", {})
            if not isinstance(metadata, dict):
                continue
            if all(metadata.get(key) == value for key, value in metadata_filters.items()):
                filtered.append(record)
        return filtered

    def _build_metadata_filter_clause(self, metadata_filters: Optional[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        if not metadata_filters:
            return "", {}

        clauses: list[str] = []
        params: dict[str, Any] = {}
        for idx, (key, value) in enumerate(metadata_filters.items()):
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", str(key)):
                raise ValidationError(f"Invalid metadata filter key: {key}")
            value_param = f"meta_value_{idx}"
            clauses.append(f"metadata.{key} = ${value_param}")
            params[value_param] = value

        return " AND " + " AND ".join(clauses), params

    async def _search_by_vector(self, embedding, limit, threshold, metadata_filters=None):
        filter_clause, filter_params = self._build_metadata_filter_clause(metadata_filters)
        q = f"""  # nosec B608 - SurrealQL 参数化查询，非 SQL 注入风险
            SELECT id, content, metadata,
                   vector::similarity::cosine(embedding, $embedding) AS score
            FROM memory
            WHERE vector::similarity::cosine(embedding, $embedding) > $threshold{filter_clause}
            ORDER BY score DESC LIMIT $limit
        """
        params = {"embedding": embedding, "threshold": threshold, "limit": limit}
        params.update(filter_params)
        result = await self._db_call("query", q, params)
        return self._format_results(result)

    async def _search_by_keyword(self, query_text, limit, metadata_filters=None):
        filter_clause, filter_params = self._build_metadata_filter_clause(metadata_filters)
        q = f"""  # nosec B608 - SurrealQL 参数化查询，filter_clause 已验证
            SELECT id, content, metadata
            FROM memory WHERE content CONTAINS $query{filter_clause} LIMIT $limit
        """
        params = {"query": query_text, "limit": limit}
        params.update(filter_params)
        result = await self._db_call("query", q, params)
        return self._format_results(result, default_score=0.5)

    async def _hybrid_search(self, query_text, embedding, limit, threshold, metadata_filters=None):
        tasks = [
            self._search_by_vector(embedding, limit * 2, threshold, metadata_filters),
            self._search_by_keyword(query_text, limit * 2, metadata_filters),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        vector_raw = results[0]
        keyword_raw = results[1]
        vector_results = [] if isinstance(vector_raw, Exception) else cast(list[dict[str, Any]], vector_raw)
        keyword_results = [] if isinstance(keyword_raw, Exception) else cast(list[dict[str, Any]], keyword_raw)

        seen = set()
        merged = []
        for item in cast(list[dict[str, Any]], vector_results):
            if item.get("id") not in seen:
                seen.add(item["id"])
                merged.append(item)
        for item in cast(list[dict[str, Any]], keyword_results):
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
