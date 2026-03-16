"""记忆管理器模块

封装记忆的批量上传、搜索等业务逻辑。
支持 KNN 向量搜索（HNSW 索引）、Meilisearch 全文搜索和 RRF 混合搜索。
支持 SurrealDB RELATE 图关系操作（创建/查询/删除记忆间关联）。
多租户隔离通过 tenant_id 字段实现。
"""

import asyncio
import hashlib
import json
import logging
import re
from typing import Any

from .exceptions import DatabaseError, EmbeddingError, ValidationError
from .http_pool import get_http_pool
from .meili_client import MeilisearchClient
from .tracing import get_tracer

logger = logging.getLogger(__name__)


class MemoryManager:
    """记忆管理器，协调 embedding 服务和数据库操作

    支持多租户隔离和 SurrealDB 3.0 新查询语法：
    - 向量搜索: KNN <|K,EF|> 算子 + HNSW 索引
    - 关键词搜索: Meilisearch 全文搜索（CJK 分词）+ SurrealDB BM25（降级路径）
    - 混合搜索: RRF 融合算法（向量 from SurrealDB + 关键词 from Meilisearch）
    - 图关系: RELATE 语句实现记忆间关联（follow_up/related/elaboration 等）
    """

    def __init__(
        self,
        db: Any,  # AsyncSurreal SDK 返回联合类型，使用 Any 避免类型检查误报
        embedding_service_url: str,
        search_config: Any = None,
        batch_size: int = 10,
    ) -> None:
        self._db = db
        self._embedding_service_url = embedding_service_url
        self._batch_size = batch_size
        self._http_pool: Any | None = None
        self._meili: MeilisearchClient | None = None

        # 搜索配置（从 config.SearchConfig 传入，使用 getattr 保持向后兼容）
        self._rrf_k: int = getattr(search_config, "rrf_k", 60)
        self._rrf_vector_weight: float = getattr(search_config, "rrf_vector_weight", 0.7)
        self._rrf_keyword_weight: float = getattr(search_config, "rrf_keyword_weight", 0.3)
        self._hnsw_ef_search: int = getattr(search_config, "hnsw_ef_search", 50)
        self._default_tenant_id: str = getattr(search_config, "default_tenant_id", "default")

    async def _get_http_pool(self):
        """延迟初始化 HTTP 连接池"""
        if self._http_pool is None:
            self._http_pool = await get_http_pool()
        return self._http_pool

    def set_meili_client(self, client: MeilisearchClient) -> None:
        self._meili = client

    async def close(self) -> None:
        """关闭资源"""

    async def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """批量获取文本的 embedding 向量"""
        tracer = get_tracer()
        with tracer.start_as_current_span("embedding.get_batch") as span:
            span.set_attribute("embedding.text_count", len(texts))
            try:
                http_pool = await self._get_http_pool()
                response = await http_pool.post(
                    f"{self._embedding_service_url}/v1/embeddings",
                    json={"input": texts, "model": "Qwen3-Embedding-0.6B"},
                )
                response.raise_for_status()
                data = response.json()
                result = [item["embedding"] for item in data["data"]]
                span.set_attribute("embedding.dimension", len(result[0]) if result else 0)
                return result
            except Exception as e:
                span.record_exception(e)
                raise EmbeddingError(f"Failed to get embeddings: {e!s}") from e

    # ==================== 上传 ====================

    async def upload_memories(self, memories: list[dict[str, Any]], tenant_id: str | None = None) -> dict[str, Any]:
        """批量上传记忆"""
        if not memories:
            raise ValidationError("Memories list cannot be empty")

        effective_tenant_id = tenant_id or self._default_tenant_id
        tracer = get_tracer()

        with tracer.start_as_current_span("memory.upload") as span:
            span.set_attribute("memory.count", len(memories))
            span.set_attribute("tenant.id", effective_tenant_id)

            total = len(memories)
            success_count = 0
            failed_count = 0
            memory_ids: list[str] = []
            errors: list[str | dict[str, Any]] = []
            meili_docs: list[dict[str, Any]] = []

            # 批量获取 embeddings
            texts = [m.get("content", "") for m in memories]
            try:
                embeddings = await self._get_embeddings(texts)
            except EmbeddingError as e:
                span.record_exception(e)
                span.set_attribute("memory.upload.success", 0)
                span.set_attribute("memory.upload.failed", total)
                return {
                    "total": total,
                    "success": 0,
                    "failed": total,
                    "memory_ids": [],
                    "errors": [str(e)],
                }

            for memory, embedding in zip(memories, embeddings, strict=False):
                try:
                    content = memory.get("content", "")
                    memory_data: dict[str, Any] = {
                        "content": content,
                        "content_hash": hashlib.md5(content.encode("utf-8"), usedforsecurity=False).hexdigest(),
                        "embedding": embedding,
                        "tenant_id": effective_tenant_id,
                        "type": memory.get("type", "general"),
                        "tags": memory.get("tags", []),
                        "project_id": memory.get("project_id", "global"),
                        "source": memory.get("source", "api"),
                        "metadata": memory.get("metadata", {}),
                    }

                    if "source_id" in memory:
                        memory_data["source_id"] = memory["source_id"]
                    if "source_timestamp" in memory:
                        memory_data["source_timestamp"] = memory["source_timestamp"]
                    if "classification_confidence" in memory:
                        memory_data["classification_confidence"] = memory["classification_confidence"]

                    existing = await self._db.query(
                        "SELECT id FROM memory WHERE tenant_id = $tenant_id AND content_hash = $hash LIMIT 1",
                        {"tenant_id": effective_tenant_id, "hash": memory_data["content_hash"]},
                    )
                    logger.info(f"Content hash check: hash={memory_data['content_hash']}, existing={existing}")
                    existing_records = self._extract_records(existing)
                    logger.info(
                        f"Extracted records: type={type(existing_records)}, len={len(existing_records)}, records={existing_records}"
                    )
                    if existing_records:
                        existing_id = str(existing_records[0].get("id", ""))
                        failed_count += 1
                        errors.append(
                            {
                                "type": "duplicate",
                                "duplicate_type": "hash",
                                "message": "Content hash duplicate detected",
                                "existing_id": existing_id,
                                "retryable": False,
                            }
                        )
                        logger.info(f"Content hash duplicate detected, existing_id={existing_id}, skipping")
                        continue

                    similar = await self._search_by_vector(
                        embedding=embedding,
                        limit=1,
                        threshold=0.95,
                        tenant_id=effective_tenant_id,
                    )
                    print(f"[DEBUG] Semantic search result: found={len(similar)} items, threshold=0.95")
                    logger.info(f"Semantic search result: found={len(similar)} items, threshold=0.95")
                    if similar:
                        print(f"[DEBUG] Similar items: {similar}")
                        similarity_score = similar[0].get("score", 0)
                        existing_id = similar[0].get("id", "")
                        logger.info(
                            f"Semantic duplicate found: similarity={similarity_score:.3f}, existing_id={existing_id}"
                        )
                        failed_count += 1
                        errors.append(
                            {
                                "type": "duplicate",
                                "duplicate_type": "semantic",
                                "message": f"Semantic duplicate detected (similarity: {similarity_score:.3f})",
                                "existing_id": existing_id,
                                "similarity": similarity_score,
                                "retryable": False,
                            }
                        )
                        continue

                    result = await self._db.create("memory", memory_data)
                    record_id: str | None = None
                    if isinstance(result, list) and len(result) > 0:
                        record_id = str(result[0].get("id", "")) or None
                    elif isinstance(result, dict) and result.get("id"):
                        record_id = str(result["id"])

                    if not record_id:
                        logger.warning(
                            f"SurrealDB create returned no ID. Result type: {type(result)}, Result: {result}"
                        )

                    if record_id:
                        memory_ids.append(record_id)
                        success_count += 1
                        # 构建 Meilisearch 文档（不含 embedding 向量）
                        if self._meili:
                            meili_doc: dict[str, Any] = {
                                "id": self._to_meili_id(record_id),
                                "surreal_id": record_id,
                                "content": memory.get("content", ""),
                                "metadata": memory.get("metadata", {}),
                                "tenant_id": effective_tenant_id,
                                "type": memory.get("type", "general"),
                                "tags": memory.get("tags", []),
                                "project_id": memory.get("project_id", "global"),
                            }
                            # 额外字段，方便 Meilisearch 的过滤与字段级搜索
                            meili_doc["ip_address"] = memory.get("metadata", {}).get("ip_address") or memory.get(
                                "metadata", {}
                            ).get("ip")
                            meili_doc["email"] = memory.get("metadata", {}).get("email")
                            meili_doc["version"] = memory.get("metadata", {}).get("version")
                            meili_doc["code"] = memory.get("content", "")  # 代码搜索字段
                            if "source_id" in memory:
                                meili_doc["source_id"] = memory["source_id"]
                            if "source_timestamp" in memory:
                                meili_doc["date"] = memory["source_timestamp"]
                            meili_docs.append(meili_doc)
                    else:
                        failed_count += 1
                        errors.append("No record ID returned (possible UNIQUE constraint violation)")
                except Exception as e:
                    failed_count += 1
                    errors.append(f"{type(e).__name__}: {e!s}")

            # 同步到 Meilisearch（优雅降级：失败不影响主流��）
            if self._meili and meili_docs:
                try:
                    await self._meili.add_documents(meili_docs)
                    span.set_attribute("memory.upload.meili_synced", len(meili_docs))
                except Exception as meili_err:
                    logger.warning("[Meili sync] 同步失败（不影响 SurrealDB 数据）: %s", meili_err)
                    span.set_attribute("memory.upload.meili_error", str(meili_err))

            span.set_attribute("memory.upload.success", success_count)
            span.set_attribute("memory.upload.failed", failed_count)

            result_data: dict[str, Any] = {
                "total": total,
                "success": success_count,
                "failed": failed_count,
                "memory_ids": memory_ids,
            }
            if errors:
                result_data["errors"] = errors[:10]
            return result_data

    # ==================== 搜索 ====================

    async def search_memories(
        self,
        query: str,
        mode: str = "hybrid",
        limit: int = 10,
        threshold: float = 0.7,
        tenant_id: str | None = None,
        filters: str | None = None,
    ) -> dict[str, Any]:
        """搜索记忆"""
        if mode not in ("vector", "keyword", "hybrid"):
            raise ValidationError(f"Invalid search mode: {mode}")

        effective_tenant_id = tenant_id or self._default_tenant_id
        tracer = get_tracer()

        with tracer.start_as_current_span("memory.search") as span:
            span.set_attribute("search.mode", mode)
            span.set_attribute("search.limit", limit)
            span.set_attribute("search.threshold", threshold)
            span.set_attribute("tenant.id", effective_tenant_id)
            span.set_attribute("search.query_length", len(query))

            try:
                if mode == "vector":
                    embeddings = await self._get_embeddings([query])
                    results = await self._search_by_vector(embeddings[0], limit, threshold, effective_tenant_id)
                elif mode == "keyword":
                    results = await self._search_by_keyword(query, limit, effective_tenant_id, filter_expr=filters)
                else:
                    embeddings = await self._get_embeddings([query])
                    results = await self._hybrid_search(
                        query, embeddings[0], limit, threshold, effective_tenant_id, filters
                    )

                span.set_attribute("search.result_count", len(results))
                return {"results": results, "total": len(results), "mode": mode, "query": query}
            except Exception as e:
                span.record_exception(e)
                raise DatabaseError(f"Search failed: {e!s}") from e

    async def _search_by_vector(
        self,
        embedding: list[float],
        limit: int,
        threshold: float,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        """向量相似度搜索（利用 HNSW 索引）"""
        tracer = get_tracer()
        with tracer.start_as_current_span("search.vector") as span:
            span.set_attribute("search.vector.limit", limit)
            span.set_attribute("search.vector.threshold", threshold)

            q = (  # nosec B608
                "SELECT id, content, metadata, type, tags, project_id, "  # nosec B608
                "vector::similarity::cosine(embedding, $query_embedding) AS score "
                "FROM memory "
                "WHERE tenant_id = $tenant_id "
                "AND vector::similarity::cosine(embedding, $query_embedding) >= $threshold "
                "ORDER BY score DESC "
                f"LIMIT {int(limit)}"
            )
            result = await self._db.query(
                q, {"tenant_id": tenant_id, "query_embedding": embedding, "threshold": threshold}
            )

            results = self._format_similarity_results(result)
            span.set_attribute("search.vector.result_count", len(results))
            return results

    async def _search_by_keyword(
        self, query_text: str, limit: int, tenant_id: str, filter_expr: str | None = None
    ) -> list[dict[str, Any]]:
        """全文搜索：优先使用 Meilisearch，降级到 SurrealDB BM25"""
        if self._meili:
            return await self._search_by_keyword_meili(query_text, limit, tenant_id, filter_expr=filter_expr)
        return await self._search_by_keyword_surreal(query_text, limit, tenant_id)

    async def _search_by_keyword_meili(
        self,
        query_text: str,
        limit: int,
        tenant_id: str,
        filter_expr: str | None = None,
    ) -> list[dict[str, Any]]:
        """Meilisearch 全文搜索（支持 CJK 分词、日期精确匹配）"""
        assert self._meili is not None  # 由 _search_by_keyword 保证
        tracer = get_tracer()
        with tracer.start_as_current_span("search.keyword.meili") as span:
            span.set_attribute("search.keyword.engine", "meilisearch")
            span.set_attribute("search.keyword.query", query_text[:100])

            actual_filter = filter_expr or f"tenant_id = '{tenant_id}'"
            result = await self._meili.search(
                query_text,
                filter_expr=actual_filter,
                limit=limit,
            )
            results = self._format_meili_results(result)
            span.set_attribute("search.keyword.result_count", len(results))
            return results

    async def _search_by_keyword_surreal(
        self,
        query_text: str,
        limit: int,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        """SurrealDB BM25 全文搜索（Meilisearch 不可用时的降级路径）"""
        tracer = get_tracer()
        with tracer.start_as_current_span("search.keyword.surreal") as span:
            safe_query = self._sanitize_query(query_text)
            span.set_attribute("search.keyword.engine", "surrealdb")
            span.set_attribute("search.keyword.query", safe_query[:100])
            q = (  # nosec B608
                "SELECT id, content, metadata, type, tags, project_id, "
                "search::score(1) AS score "
                "FROM memory "
                "WHERE tenant_id = $tenant_id "
                f"AND content @1@ '{safe_query}' "  # nosec B608
                "ORDER BY score DESC "
                f"LIMIT {int(limit)}"  # nosec B608
            )
            result = await self._db.query(q, {"tenant_id": tenant_id})
            results = self._format_keyword_results(result)
            span.set_attribute("search.keyword.result_count", len(results))
            return results

    async def _hybrid_search(
        self,
        query_text: str,
        embedding: list[float],
        limit: int,
        threshold: float,
        tenant_id: str,
        filter_expr: str | None = None,
    ) -> list[dict[str, Any]]:
        """RRF 混合搜索：并行执行向量+关键词搜索，然后用 RRF 算法融合"""
        tracer = get_tracer()
        with tracer.start_as_current_span("search.hybrid") as span:
            tasks = [
                self._search_by_vector(embedding, limit * 2, threshold, tenant_id),
                self._search_by_keyword(query_text, limit * 2, tenant_id, filter_expr=filter_expr),
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            vector_results: list[dict[str, Any]] = [] if isinstance(results[0], BaseException) else results[0]
            keyword_results: list[dict[str, Any]] = [] if isinstance(results[1], BaseException) else results[1]

            span.set_attribute("search.hybrid.vector_count", len(vector_results))
            span.set_attribute("search.hybrid.keyword_count", len(keyword_results))

            merged = self._rrf_merge(vector_results, keyword_results)
            span.set_attribute("search.hybrid.merged_count", len(merged[:limit]))
            return merged[:limit]

    def _rrf_merge(
        self,
        vector_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """RRF (Reciprocal Rank Fusion) 混合搜索合并算法

        公式: RRF_score(d) = Σ (weight_i / (k + rank_i(d)))

        使用排名位置而非原始分数，天然归一化，不受 distance/score 尺度差异影响。
        k=60 来自原始论文（Cormack et al. 2009）推荐值。
        """
        k = self._rrf_k
        vector_weight = self._rrf_vector_weight
        keyword_weight = self._rrf_keyword_weight

        scores: dict[str, float] = {}
        items: dict[str, dict[str, Any]] = {}

        # 向量搜索贡献
        for rank, item in enumerate(vector_results):
            doc_id = item.get("id")
            if not doc_id:
                continue
            scores[doc_id] = vector_weight / (k + rank + 1)
            items[doc_id] = item

        # 关键词搜索贡献
        for rank, item in enumerate(keyword_results):
            doc_id = item.get("id")
            if not doc_id:
                continue
            scores.setdefault(doc_id, 0.0)
            scores[doc_id] += keyword_weight / (k + rank + 1)
            if doc_id not in items:
                items[doc_id] = item

        # 按 RRF 分数降序排列
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        results: list[dict[str, Any]] = []
        for doc_id in sorted_ids:
            item = items[doc_id].copy()
            item["score"] = round(scores[doc_id], 6)
            results.append(item)

        return results

    # ==================== 结果格式化 ====================

    def _format_vector_results(self, db_result: Any, max_distance: float) -> list[dict[str, Any]]:
        """格式化向量搜索结果：distance → similarity score + 阈值过滤"""
        raw_items = self._extract_records(db_result)
        results: list[dict[str, Any]] = []
        for item in raw_items:
            distance = item.get("distance")
            if distance is None:
                continue
            # 阈值过滤：distance > max_distance 表示相似度不够
            if float(distance) > max_distance:
                continue
            # distance → similarity score (cosine: similarity = 1 - distance)
            score = round(1.0 - float(distance), 6)
            results.append(self._build_result_item(item, score=score))
        return results

    def _format_similarity_results(self, db_result: Any) -> list[dict[str, Any]]:
        raw_items = self._extract_records(db_result)
        results: list[dict[str, Any]] = []
        for item in raw_items:
            score = item.get("score")
            if score is None:
                continue
            results.append(self._build_result_item(item, score=float(score)))
        return results

    def _format_keyword_results(self, db_result: Any) -> list[dict[str, Any]]:
        """格式化关键词搜索结果"""
        raw_items = self._extract_records(db_result)
        results: list[dict[str, Any]] = []
        for item in raw_items:
            score = item.get("score", 0.0)
            results.append(self._build_result_item(item, score=float(score)))
        return results

    def _format_meili_results(self, meili_result: dict[str, Any]) -> list[dict[str, Any]]:
        """格式化 Meilisearch 搜索结果为统一格式

        Meilisearch 返回结构:
        {"hits": [{"id": "...", "content": "...", "_rankingScore": 0.95, ...}], ...}
        """
        results: list[dict[str, Any]] = []
        for hit in meili_result.get("hits", []):
            # 使用 surreal_id 还原完整的 SurrealDB record ID
            doc_id = hit.get("surreal_id") or self._from_meili_id(str(hit.get("id", "")))
            score = hit.get("_rankingScore", 0.0)
            results.append(
                {
                    "id": str(doc_id),
                    "content": hit.get("content", ""),
                    "metadata": hit.get("metadata", {}),
                    "type": hit.get("type", "general"),
                    "tags": hit.get("tags", []),
                    "project_id": hit.get("project_id", "global"),
                    "score": round(float(score), 6),
                }
            )
        return results

    def _build_result_item(self, item: dict[str, Any], score: float) -> dict[str, Any]:
        """构建统一的搜索结果条目（包含新增的 type/tags/project_id 字段）"""
        return {
            "id": str(item.get("id", "")),
            "content": item.get("content", ""),
            "metadata": item.get("metadata", {}),
            "type": item.get("type", "general"),
            "tags": item.get("tags", []),
            "project_id": item.get("project_id", "global"),
            "score": score,
        }

    def _extract_records(self, db_result: Any) -> list[dict[str, Any]]:
        """从 SurrealDB query() 返回值中提取记录列表

        处理 SDK 返回的多种格式：
        - list[dict]: 直接的记录列表（单条 SELECT 语句）
        - list[list[dict]]: 嵌套结构（多语句结果或 query_raw）
        """
        records: list[dict[str, Any]] = []
        if not db_result or not isinstance(db_result, list):
            return records
        for item in db_result:
            if isinstance(item, dict):
                records.append(item)
            elif isinstance(item, list):
                for record in item:
                    if isinstance(record, dict):
                        records.append(record)
        return records

    @staticmethod
    def _sanitize_query(text: str) -> str:
        """清洗搜索查询文本，防止 SurrealQL 注入

        策略：移除 SurrealQL 特殊字符，保留字母数字和 CJK 字符。
        比简单转义更安全：直接移除潜在危险字符而非依赖转义正确性。
        """
        # 保留: 字母、数字、空格、CJK 统一表意文字（U+4E00-U+9FFF）
        # 移除: 引号、分号、反斜杠等 SQL/SurrealQL 特殊字符
        return re.sub(r"[^\w\s\u4e00-\u9fff\u3400-\u4dbf\uff00-\uffef-]", "", text).strip()[:500]

    # ==================== 图关系 (RELATE) ====================

    async def create_relation(
        self,
        from_id: str,
        to_id: str,
        relationship_type: str = "related",
        weight: float = 0.5,
        tenant_id: str | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """创建两条记忆之间的图关系"""
        effective_tenant_id = tenant_id or self._default_tenant_id

        valid_types = {"related", "follow_up", "elaboration", "contradiction", "reference", "derived_from"}
        if relationship_type not in valid_types:
            raise ValidationError(f"Invalid relationship_type: {relationship_type}. Must be one of {valid_types}")
        if not 0.0 <= weight <= 1.0:
            raise ValidationError(f"weight must be between 0.0 and 1.0, got {weight}")

        from_ref = self._normalize_memory_id(from_id)
        to_ref = self._normalize_memory_id(to_id)

        tracer = get_tracer()
        with tracer.start_as_current_span("graph.create_relation") as span:
            span.set_attribute("graph.from", from_ref)
            span.set_attribute("graph.to", to_ref)
            span.set_attribute("graph.type", relationship_type)
            span.set_attribute("tenant.id", effective_tenant_id)

            try:
                set_clauses = [
                    f"relationship_type = '{relationship_type}'",
                    f"weight = {float(weight)}",
                    "tenant_id = $tenant_id",
                ]
                if description:
                    safe_desc = self._sanitize_query(description)
                    set_clauses.append(f"description = '{safe_desc}'")
                if metadata:
                    set_clauses.append(f"metadata = {json.dumps(metadata)}")

                set_str = ", ".join(set_clauses)
                q = (  # nosec B608
                    f"RELATE {from_ref}->memory_relation->{to_ref} "  # nosec B608
                    f"SET {set_str}"  # nosec B608
                )
                result = await self._db.query(q, {"tenant_id": effective_tenant_id})

                records = self._extract_records(result)
                if records:
                    return {
                        "id": str(records[0].get("id", "")),
                        "from": from_ref,
                        "to": to_ref,
                        "relationship_type": relationship_type,
                        "weight": weight,
                    }
                return {"error": "No relation created"}
            except Exception as e:
                span.record_exception(e)
                raise DatabaseError(f"Failed to create relation: {e!s}") from e

    async def get_relations(
        self,
        memory_id: str,
        direction: str = "both",
        relationship_type: str | None = None,
        tenant_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """查询记忆的关联关系"""
        effective_tenant_id = tenant_id or self._default_tenant_id
        mem_ref = self._normalize_memory_id(memory_id)

        if direction not in ("outgoing", "incoming", "both"):
            raise ValidationError(f"Invalid direction: {direction}. Must be outgoing/incoming/both")

        tracer = get_tracer()
        with tracer.start_as_current_span("graph.get_relations") as span:
            span.set_attribute("graph.memory_id", mem_ref)
            span.set_attribute("graph.direction", direction)
            span.set_attribute("tenant.id", effective_tenant_id)

            try:
                results: list[dict[str, Any]] = []

                if direction in ("outgoing", "both"):
                    q = (
                        f"SELECT *, meta::id(id) AS relation_id, "  # nosec B608
                        f"meta::id(in) AS from_id, meta::id(out) AS to_id "
                        f"FROM memory_relation "
                        f"WHERE in = {mem_ref} AND tenant_id = $tenant_id "
                    )
                    if relationship_type:
                        safe_type = self._sanitize_query(relationship_type)
                        q += f"AND relationship_type = '{safe_type}' "  # nosec B608
                    q += f"LIMIT {int(limit)}"
                    r = await self._db.query(q, {"tenant_id": effective_tenant_id})
                    results.extend(self._format_relation_results(r, "outgoing"))

                if direction in ("incoming", "both"):
                    q = (
                        f"SELECT *, meta::id(id) AS relation_id, "  # nosec B608
                        f"meta::id(in) AS from_id, meta::id(out) AS to_id "
                        f"FROM memory_relation "
                        f"WHERE out = {mem_ref} AND tenant_id = $tenant_id "
                    )
                    if relationship_type:
                        safe_type = self._sanitize_query(relationship_type)
                        q += f"AND relationship_type = '{safe_type}' "  # nosec B608
                    q += f"LIMIT {int(limit)}"
                    r = await self._db.query(q, {"tenant_id": effective_tenant_id})
                    results.extend(self._format_relation_results(r, "incoming"))

                span.set_attribute("graph.relation_count", len(results))
                return results
            except Exception as e:
                span.record_exception(e)
                raise DatabaseError(f"Failed to get relations: {e!s}") from e

    async def delete_relation(
        self,
        relation_id: str,
        tenant_id: str | None = None,
    ) -> bool:
        """删除指定的关系

        Args:
            relation_id: 关系 ID（如 "memory_relation:abc123"）
            tenant_id: 租户 ID（安全检查，防止跨租户删除）
        """
        effective_tenant_id = tenant_id or self._default_tenant_id
        rel_ref = self._normalize_relation_id(relation_id)

        try:
            # 先验证关系存在且属于该租户
            q = f"SELECT id FROM {rel_ref} WHERE tenant_id = $tenant_id"  # nosec B608
            check = await self._db.query(q, {"tenant_id": effective_tenant_id})
            records = self._extract_records(check)
            if not records:
                return False

            await self._db.query(f"DELETE {rel_ref}")  # nosec B608
            return True
        except Exception as e:
            raise DatabaseError(f"Failed to delete relation: {e!s}") from e

    async def get_related_memories(
        self,
        memory_id: str,
        depth: int = 1,
        relationship_type: str | None = None,
        tenant_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """图遍历：获取关联的记忆内容（支持多层深度）"""
        effective_tenant_id = tenant_id or self._default_tenant_id
        mem_ref = self._normalize_memory_id(memory_id)
        depth = max(1, min(depth, 3))

        tracer = get_tracer()
        with tracer.start_as_current_span("graph.traverse") as span:
            span.set_attribute("graph.start", mem_ref)
            span.set_attribute("graph.depth", depth)
            span.set_attribute("tenant.id", effective_tenant_id)

            try:
                path = "->memory_relation->memory" * depth

                q = (
                    f"SELECT {path}.* AS related "  # nosec B608
                    f"FROM {mem_ref}"  # nosec B608
                )
                result = await self._db.query(q)

                seen: set[str] = set()
                memories: list[dict[str, Any]] = []
                records = self._extract_records(result)

                for record in records:
                    related_list = record.get("related", [])
                    if not isinstance(related_list, list):
                        related_list = [related_list]
                    for mem in related_list:
                        if not isinstance(mem, dict):
                            continue
                        mem_id = str(mem.get("id", ""))
                        if not mem_id or mem_id in seen:
                            continue
                        if mem.get("tenant_id") != effective_tenant_id:
                            continue
                        seen.add(mem_id)
                        memories.append(self._build_result_item(mem, score=0.0))
                        if len(memories) >= limit:
                            break
                    if len(memories) >= limit:
                        break

                span.set_attribute("graph.traverse.result_count", len(memories))
                return memories
            except Exception as e:
                span.record_exception(e)
                raise DatabaseError(f"Failed to get related memories: {e!s}") from e

    # ==================== ID 规范化 ====================

    @staticmethod
    def _normalize_memory_id(memory_id: str) -> str:
        """规范化记忆 ID 为 SurrealDB record ID 格式

        接受: "memory:abc123" 或 "abc123"
        返回: "memory:abc123"
        """
        mid = str(memory_id)
        if mid.startswith("memory:"):
            return mid
        return f"memory:{mid}"

    @staticmethod
    def _to_meili_id(surreal_id: str) -> str:
        """SurrealDB record ID → Meilisearch 主键（去掉表名前缀）

        'memory:abc123' → 'abc123'
        'abc123' → 'abc123'
        """
        sid = str(surreal_id)
        if ":" in sid:
            return sid.split(":", 1)[1]
        return sid

    @staticmethod
    def _from_meili_id(meili_id: str) -> str:
        """Meilisearch 主键 → SurrealDB record ID（补全表名前缀）

        'abc123' → 'memory:abc123'
        'memory:abc123' → 'memory:abc123'
        """
        mid = str(meili_id)
        if mid.startswith("memory:"):
            return mid
        return f"memory:{mid}"

    @staticmethod
    def _normalize_relation_id(relation_id: str) -> str:
        """规范化关系 ID 为 SurrealDB record ID 格式"""
        rid = str(relation_id)
        if rid.startswith("memory_relation:"):
            return rid
        return f"memory_relation:{rid}"

    def _format_relation_results(self, db_result: Any, direction: str) -> list[dict[str, Any]]:
        """格式化关系查询结果"""
        raw_items = self._extract_records(db_result)
        results: list[dict[str, Any]] = []
        for item in raw_items:
            results.append(
                {
                    "id": str(item.get("id", "")),
                    "from": str(item.get("in", "")),
                    "to": str(item.get("out", "")),
                    "direction": direction,
                    "relationship_type": item.get("relationship_type", "related"),
                    "weight": item.get("weight", 0.5),
                    "description": item.get("description"),
                    "metadata": item.get("metadata", {}),
                    "created_at": str(item.get("created_at", "")),
                }
            )
        return results
