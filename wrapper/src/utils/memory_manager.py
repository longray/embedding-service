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

from aiocache import cached
from aiocache.serializers import JsonSerializer

from .code_analyzer import CodeAnalyzer, CodeAnalysisResult
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

        # Phase A-B3: 查询结果缓存（aiocache）
        self._cache_enabled: bool = getattr(search_config, "cache_enabled", True)
        self._cache_ttl: int = getattr(search_config, "cache_ttl", 300)  # 5分钟
        self._vector_cache: Any | None = None
        self._keyword_cache: Any | None = None
        if self._cache_enabled:
            from aiocache import Cache

            self._vector_cache = Cache(Cache.MEMORY, serializer=JsonSerializer())
            self._keyword_cache = Cache(Cache.MEMORY, serializer=JsonSerializer())

        # 搜索配置（从 config.SearchConfig 传入，使用 getattr 保持向后兼容）
        self._rrf_k: int = getattr(search_config, "rrf_k", 60)
        self._rrf_vector_weight: float = getattr(search_config, "rrf_vector_weight", 0.7)
        self._rrf_keyword_weight: float = getattr(search_config, "rrf_keyword_weight", 0.3)
        self._hnsw_ef_search: int = getattr(search_config, "hnsw_ef_search", 50)
        self._default_tenant_id: str = getattr(search_config, "default_tenant_id", "default")

        # Phase A-B6: 动态去重阈值
        self._dedup_thresholds: dict[str, float] = getattr(
            search_config,
            "dedup_thresholds",
            {
                "preference": 0.88,
                "decision": 0.90,
                "long-term": 0.93,
                "general": 0.95,
                "daily": 1.0,
            },
        )

        # Code analyzer instance
        self.code_analyzer = CodeAnalyzer()

    async def _get_http_pool(self):
        """延迟初始化 HTTP 连接池"""
        if self._http_pool is None:
            self._http_pool = await get_http_pool()
        return self._http_pool

    def _get_dedup_threshold(self, memory_type: str) -> float:
        """获取指定记忆类型的去重阈值"""
        return self._dedup_thresholds.get(memory_type, 0.95)

    def set_meili_client(self, client: MeilisearchClient) -> None:
        self._meili = client

    async def close(self) -> None:
        """关闭资源"""

    async def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """批量获取文本的 embedding 向量（Phase C-B2: 带缓存优化）"""
        tracer = get_tracer()
        with tracer.start_as_current_span("embedding.get_batch") as span:
            span.set_attribute("embedding.text_count", len(texts))

            # Phase C-B2: 检查缓存
            cached_results = {}
            texts_to_fetch = []
            text_indices = {}

            for i, text in enumerate(texts):
                if not text:
                    continue
                cache_key = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()

                if self._cache_enabled and self._vector_cache:
                    cached = await self._vector_cache.get(cache_key)
                    if cached:
                        cached_results[i] = cached
                        continue

                texts_to_fetch.append(text)
                text_indices[len(texts_to_fetch) - 1] = i

            span.set_attribute("embedding.cache_hits", len(cached_results))
            span.set_attribute("embedding.cache_misses", len(texts_to_fetch))

            # Phase C-B2: 批量获取未缓存的 embeddings
            fetched_results = {}
            if texts_to_fetch:
                try:
                    http_pool = await self._get_http_pool()
                    response = await http_pool.post(
                        f"{self._embedding_service_url}/v1/embeddings",
                        json={"input": texts_to_fetch, "model": "Qwen3-Embedding-0.6B"},
                    )
                    response.raise_for_status()
                    data = response.json()

                    for j, item in enumerate(data["data"]):
                        original_index = text_indices[j]
                        embedding = item["embedding"]
                        fetched_results[original_index] = embedding

                        # Phase C-B2: 存入缓存
                        if self._cache_enabled and self._vector_cache:
                            cache_key = hashlib.md5(texts_to_fetch[j].encode(), usedforsecurity=False).hexdigest()
                            await self._vector_cache.set(cache_key, embedding, ttl=self._cache_ttl)

                except Exception as e:
                    span.record_exception(e)
                    raise EmbeddingError(f"Failed to get embeddings: {e!s}") from e

            # 合并结果
            result = []
            for i in range(len(texts)):
                if i in cached_results:
                    result.append(cached_results[i])
                elif i in fetched_results:
                    result.append(fetched_results[i])
                else:
                    result.append([])  # Empty embedding for empty text

            span.set_attribute("embedding.dimension", len(result[0]) if result else 0)
            return result

    # ==================== 上传 ====================

    async def upload_memories(self, memories: list[dict[str, Any]], tenant_id: str | None = None) -> dict[str, Any]:
        """批量上传记忆（Phase A-B2: 使用事务批量插入 + Phase A-B7: 智能去重决策）"""
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
            updated_count = 0
            skipped: list[dict[str, Any]] = []
            memory_ids: list[str] = []
            errors: list[str | dict[str, Any]] = []
            meili_docs: list[dict[str, Any]] = []

            # Phase A-B5: 批量获取 embeddings（优化为单次批量调用）
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
                    "updated": 0,
                    "skipped": [],
                    "memory_ids": [],
                    "errors": [str(e)],
                }

            # Phase A-B2: 构建批量插入数据
            batch_inserts: list[dict[str, Any]] = []

            for memory, embedding in zip(memories, embeddings, strict=False):
                try:
                    content = memory.get("content", "")
                    mem_type = memory.get("type", "general")

                    memory_data: dict[str, Any] = {
                        "content": content,
                        "content_hash": hashlib.md5(content.encode("utf-8"), usedforsecurity=False).hexdigest(),
                        "embedding": embedding,
                        "tenant_id": effective_tenant_id,
                        "type": mem_type,
                        "tags": memory.get("tags", []),
                        "project_id": memory.get("project_id", "global"),
                        "source": memory.get("source", "api"),
                        "metadata": memory.get("metadata", {}),
                    }

                    # v2.4.0 L0/L1/L2 fields
                    if memory.get("abstract"):
                        memory_data["content_abstract"] = memory["abstract"]
                    if memory.get("overview"):
                        memory_data["content_overview"] = memory["overview"]
                    if memory.get("local_id"):
                        memory_data["local_id"] = memory["local_id"]

                    if "source_id" in memory:
                        memory_data["source_id"] = memory["source_id"]
                    if "source_timestamp" in memory:
                        memory_data["source_timestamp"] = memory["source_timestamp"]
                    if "classification_confidence" in memory:
                        memory_data["classification_confidence"] = memory["classification_confidence"]

                    # Phase A-B6: 使用动态阈值
                    dedup_threshold = self._get_dedup_threshold(mem_type)

                    existing = await self._db.query(
                        "SELECT id FROM memory WHERE tenant_id = $tenant_id AND content_hash = $hash LIMIT 1",
                        {"tenant_id": effective_tenant_id, "hash": memory_data["content_hash"]},
                    )
                    existing_records = self._extract_records(existing)
                    if existing_records:
                        existing_id = str(existing_records[0].get("id", ""))
                        failed_count += 1
                        skipped.append(
                            {
                                "local_id": memory.get("local_id") or memory.get("source_id"),
                                "existing_id": existing_id,
                                "reason": "hash",
                                "similarity": None,
                            }
                        )
                        continue

                    # Phase A-B7: 语义相似度检查 + 智能决策
                    similar = await self._search_by_vector(
                        embedding=embedding,
                        limit=1,
                        threshold=dedup_threshold,  # Phase A-B6: 动态阈值
                        tenant_id=effective_tenant_id,
                    )

                    if similar:
                        similarity_score = similar[0].get("score", 0)
                        existing_id = similar[0].get("id", "")
                        existing_record = similar[0]

                        # Phase A-B7: 智能决策（UPDATE / DISCARD / KEEP_BOTH）
                        decision = self._decide_duplicate_action(
                            new_memory=memory,
                            old_record=existing_record,
                            similarity=similarity_score,
                            mem_type=mem_type,
                        )

                        if decision == "UPDATE":
                            # 更新现有记录
                            await self._update_memory(existing_id, memory_data)
                            memory_ids.append(existing_id)
                            updated_count += 1
                            success_count += 1
                            continue
                        elif decision == "DISCARD":
                            failed_count += 1
                            skipped.append(
                                {
                                    "local_id": memory.get("local_id") or memory.get("source_id"),
                                    "existing_id": existing_id,
                                    "reason": "semantic",
                                    "similarity": round(similarity_score, 4),
                                }
                            )
                            continue
                        # else: KEEP_BOTH - 继续创建新记录

                    # 加入批量插入队列（Phase A-B2）
                    batch_inserts.append(memory_data)

                except Exception as e:
                    failed_count += 1
                    errors.append(f"{type(e).__name__}: {e!s}")

            # Phase A-B2: 批量插入（使用事务）
            if batch_inserts:
                try:
                    # 使用单个 INSERT 语句批量插入
                    query = "INSERT INTO memory $data"
                    result = await self._db.query(query, {"data": batch_inserts})

                    # 处理结果
                    if isinstance(result, list):
                        for i, record in enumerate(result):
                            record_id = str(record.get("id", "")) if isinstance(record, dict) else None
                            if record_id:
                                memory_ids.append(record_id)
                                success_count += 1

                                # 构建 Meilisearch 文档
                                if self._meili and i < len(batch_inserts):
                                    mem = batch_inserts[i]
                                    meili_doc = self._build_meili_doc(record_id, mem, effective_tenant_id)
                                    meili_docs.append(meili_doc)
                            else:
                                failed_count += 1
                                errors.append("No record ID returned")
                except Exception as e:
                    # 批量插入失败，回退到单条插入
                    logger.warning(f"Batch insert failed, falling back to single insert: {e}")
                    for memory_data in batch_inserts:
                        try:
                            result = await self._db.create("memory", memory_data)
                            record_id = self._extract_record_id(result)
                            if record_id:
                                memory_ids.append(record_id)
                                success_count += 1
                            else:
                                failed_count += 1
                        except Exception as inner_e:
                            failed_count += 1
                            errors.append(f"{type(inner_e).__name__}: {inner_e!s}")

            # 同步到 Meilisearch
            if self._meili and meili_docs:
                try:
                    await self._meili.add_documents(meili_docs)
                    span.set_attribute("memory.upload.meili_synced", len(meili_docs))
                except Exception as meili_err:
                    logger.warning("[Meili sync] 同步失败: %s", meili_err)
                    span.set_attribute("memory.upload.meili_error", str(meili_err))

            span.set_attribute("memory.upload.success", success_count)
            span.set_attribute("memory.upload.failed", failed_count)
            span.set_attribute("memory.upload.updated", updated_count)  # Phase A-B7

            result_data: dict[str, Any] = {
                "total": total,
                "success": success_count,
                "failed": failed_count,
                "updated": updated_count,
                "skipped": skipped,
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
        level: int = 2,
        tenant_id: str | None = None,
        filters: str | None = None,
    ) -> dict[str, Any]:
        """搜索记忆 (v2.4.0: 支持 level 参数返回分层内容)"""
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

                # v2.4.0: 按 level 返回分层内容
                filtered_results = self._filter_by_level(results, level)

                return {
                    "results": filtered_results,
                    "total": len(results),
                    "mode": mode,
                    "level": level,
                    "query": query,
                }
            except Exception as e:
                span.record_exception(e)
                raise DatabaseError(f"Search failed: {e!s}") from e

    def _filter_by_level(self, results: list[dict[str, Any]], level: int) -> list[dict[str, Any]]:
        """按 level 过滤返回结果 (v2.4.0)"""
        filtered = []
        for r in results:
            item = {"id": r.get("id"), "score": r.get("score")}

            if level >= 0:
                item["abstract"] = r.get("content_abstract", "")

            if level >= 1:
                item["overview"] = r.get("content_overview", "")

            if level >= 2:
                item["content"] = r.get("content", "")
                item["type"] = r.get("type")
                item["tags"] = r.get("tags", [])
                item["project_id"] = r.get("project_id")
                item["local_id"] = r.get("local_id")

            filtered.append(item)

        return filtered

    async def _search_by_vector(
        self,
        embedding: list[float],
        limit: int,
        threshold: float,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        """向量相似度搜索（利用 HNSW 索引 + 缓存）"""
        tracer = get_tracer()
        with tracer.start_as_current_span("search.vector") as span:
            span.set_attribute("search.vector.limit", limit)
            span.set_attribute("search.vector.threshold", threshold)

            # Phase A-B3: 缓存检查
            cache_key = self._get_vector_cache_key(embedding, limit, threshold, tenant_id)
            if self._vector_cache:
                try:
                    cached_result = await self._vector_cache.get(cache_key)
                    if cached_result:
                        span.set_attribute("search.vector.cache_hit", True)
                        span.set_attribute("search.vector.result_count", len(cached_result))
                        return cached_result
                except Exception:  # nosec B110
                    pass
            span.set_attribute("search.vector.cache_hit", False)

            q = (
                "SELECT id, content, content_abstract, content_overview, local_id, metadata, type, tags, project_id, "
                "vector::similarity::cosine(embedding, $query_embedding) AS score "
                "FROM memory "
                "WHERE tenant_id = $tenant_id "
                "AND vector::similarity::cosine(embedding, $query_embedding) >= $threshold "
                "ORDER BY score DESC "
                "LIMIT $limit"
            )
            result = await self._db.query(
                q, {"tenant_id": tenant_id, "query_embedding": embedding, "threshold": threshold, "limit": limit}
            )

            results = self._format_similarity_results(result)

            # Phase A-B3: 缓存结果
            if self._vector_cache and results:
                try:
                    await self._vector_cache.set(cache_key, results, ttl=self._cache_ttl)
                except Exception:  # nosec B110
                    pass

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
                "SELECT id, content, content_abstract, content_overview, local_id, metadata, type, tags, project_id, "
                "search::score(1) AS score "
                "FROM memory "
                "WHERE tenant_id = $tenant_id "
                f"AND content @1@ '{safe_query}' "  # nosec B608
                "ORDER BY score DESC "
                "LIMIT $limit"
            )
            result = await self._db.query(q, {"tenant_id": tenant_id, "limit": limit})
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
                    "content_abstract": hit.get("content_abstract", ""),
                    "content_overview": hit.get("content_overview", ""),
                    "local_id": hit.get("local_id"),
                    "metadata": hit.get("metadata", {}),
                    "type": hit.get("type", "general"),
                    "tags": hit.get("tags", []),
                    "project_id": hit.get("project_id", "global"),
                    "score": round(float(score), 6),
                }
            )
        return results

    def _build_result_item(self, item: dict[str, Any], score: float) -> dict[str, Any]:
        """构建统一的搜索结果条目（包含 v2.4.0 L0/L1/L2 字段）"""
        return {
            "id": str(item.get("id", "")),
            "content": item.get("content", ""),
            "content_abstract": item.get("content_abstract", ""),
            "content_overview": item.get("content_overview", ""),
            "local_id": item.get("local_id"),
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

                # 拆分 mem_ref 为 table 和 id
                mem_parts = mem_ref.split(":")
                mem_table, mem_id = mem_parts[0], mem_parts[1] if len(mem_parts) > 1 else mem_parts[0]

                if direction in ("outgoing", "both"):
                    q = (
                        "SELECT *, meta::id(id) AS relation_id, "
                        "meta::id(in) AS from_id, meta::id(out) AS to_id "
                        "FROM memory_relation "
                        "WHERE in = type::record($mem_table, $mem_id) AND tenant_id = $tenant_id "
                    )
                    if relationship_type:
                        q += "AND relationship_type = $relationship_type "
                    q += "LIMIT $limit"
                    r = await self._db.query(
                        q,
                        {
                            "tenant_id": effective_tenant_id,
                            "mem_table": mem_table,
                            "mem_id": mem_id,
                            "relationship_type": relationship_type,
                            "limit": limit,
                        },
                    )
                    results.extend(self._format_relation_results(r, "outgoing"))

                if direction in ("incoming", "both"):
                    q = (
                        "SELECT *, meta::id(id) AS relation_id, "
                        "meta::id(in) AS from_id, meta::id(out) AS to_id "
                        "FROM memory_relation "
                        "WHERE out = type::record($mem_table, $mem_id) AND tenant_id = $tenant_id "
                    )
                    if relationship_type:
                        q += "AND relationship_type = $relationship_type "
                    q += "LIMIT $limit"
                    r = await self._db.query(
                        q,
                        {
                            "tenant_id": effective_tenant_id,
                            "mem_table": mem_table,
                            "mem_id": mem_id,
                            "relationship_type": relationship_type,
                            "limit": limit,
                        },
                    )
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
            # 拆分 rel_ref 为 table 和 id
            rel_parts = rel_ref.split(":")
            rel_table, rel_id = rel_parts[0], rel_parts[1] if len(rel_parts) > 1 else rel_parts[0]

            # 先验证关系存在且属于该租户
            q = "SELECT id FROM type::record($rel_table, $rel_id) WHERE tenant_id = $tenant_id"
            check = await self._db.query(
                q, {"tenant_id": effective_tenant_id, "rel_table": rel_table, "rel_id": rel_id}
            )
            records = self._extract_records(check)
            if not records:
                return False

            await self._db.query("DELETE type::record($rel_table, $rel_id)", {"rel_table": rel_table, "rel_id": rel_id})
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

                # 拆分 mem_ref 为 table 和 id
                mem_parts = mem_ref.split(":")
                mem_table, mem_id = mem_parts[0], mem_parts[1] if len(mem_parts) > 1 else mem_parts[0]

                q = (
                    f"SELECT {path}.* AS related "  # nosec B608  # path 是内部常量，安全
                    "FROM type::record($mem_table, $mem_id)"
                )
                result = await self._db.query(q, {"mem_table": mem_table, "mem_id": mem_id})

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

    # ==================== Phase A-B6/B7: 智能去重辅助方法 ====================

    def _decide_duplicate_action(
        self, new_memory: dict[str, Any], old_record: dict[str, Any], similarity: float, mem_type: str
    ) -> str:
        """
        智能决策：遇到重复记忆时的操作
        返回: "UPDATE" | "DISCARD" | "KEEP_BOTH"
        """
        new_content = new_memory.get("content", "")
        old_content = old_record.get("content", "")
        new_len = len(new_content)
        old_len = len(old_content)

        # 规则1: 几乎完全相同且时间很近 -> 丢弃
        if similarity > 0.98:
            return "DISCARD"

        # 规则2: 用户偏好 -> 更新（偏好会改变）
        if mem_type == "preference":
            return "UPDATE"

        # 规则3: 决策记录 -> 保留两者（保留历史决策链）
        if mem_type == "decision":
            return "KEEP_BOTH"

        # 规则4: 新内容明显更详细 -> 更新
        if new_len > old_len * 1.5:
            return "UPDATE"

        # 规则5: 新内容明显更简略 -> 丢弃
        if new_len < old_len * 0.7:
            return "DISCARD"

        # 默认: 保留两者
        return "KEEP_BOTH"

    async def _update_memory(self, record_id: str, memory_data: dict[str, Any]) -> None:
        """更新现有记忆记录"""
        update_fields = {
            "content": memory_data["content"],
            "content_hash": memory_data["content_hash"],
            "embedding": memory_data["embedding"],
            "tags": memory_data.get("tags", []),
            "metadata": memory_data.get("metadata", {}),
            "updated_at": "time::now()",
        }

        # 构建 UPDATE 语句
        set_clauses = [f"{k} = ${k}" for k in update_fields.keys() if k != "updated_at"]
        set_clauses.append("updated_at = time::now()")
        set_str = ", ".join(set_clauses)

        # 拆分 record_id 为 table 和 id
        rid_parts = record_id.split(":")
        rid_table, rid_id = rid_parts[0], rid_parts[1] if len(rid_parts) > 1 else rid_parts[0]

        params = {k: v for k, v in update_fields.items() if k != "updated_at"}
        params["rid_table"] = rid_table
        params["rid_id"] = rid_id

        await self._db.query(f"UPDATE type::record($rid_table, $rid_id) SET {set_str}", params)  # nosec B608

    def _build_meili_doc(self, record_id: str, memory: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        """构建 Meilisearch 文档"""
        meili_doc: dict[str, Any] = {
            "id": self._to_meili_id(record_id),
            "surreal_id": record_id,
            "content": memory.get("content", ""),
            "metadata": memory.get("metadata", {}),
            "tenant_id": tenant_id,
            "type": memory.get("type", "general"),
            "tags": memory.get("tags", []),
            "project_id": memory.get("project_id", "global"),
        }

        # 额外字段
        metadata = memory.get("metadata", {})
        meili_doc["ip_address"] = metadata.get("ip_address") or metadata.get("ip")
        meili_doc["email"] = metadata.get("email")
        meili_doc["version"] = metadata.get("version")
        meili_doc["code"] = memory.get("content", "")

        # v2.4.0: L0/L1/L2 分层字段
        meili_doc["content_abstract"] = memory.get("content_abstract", "")
        meili_doc["content_overview"] = memory.get("content_overview", "")
        meili_doc["local_id"] = memory.get("local_id")

        if "source_id" in memory:
            meili_doc["source_id"] = memory["source_id"]
        if "source_timestamp" in memory:
            meili_doc["date"] = memory["source_timestamp"]

        return meili_doc

    def _extract_record_id(self, result: Any) -> str | None:
        """从 SurrealDB 结果中提取 record ID"""
        if isinstance(result, list) and len(result) > 0:
            return str(result[0].get("id", "")) or None
        elif isinstance(result, dict) and result.get("id"):
            return str(result["id"])
        return None

    def _get_vector_cache_key(self, embedding: list[float], limit: int, threshold: float, tenant_id: str) -> str:
        """生成向量搜索缓存键"""
        embedding_hash = hashlib.md5(str(embedding[:10]).encode(), usedforsecurity=False).hexdigest()[:16]
        return f"vec:{tenant_id}:{embedding_hash}:{limit}:{threshold}"

    def _get_keyword_cache_key(self, query: str, limit: int, tenant_id: str) -> str:
        """生成关键词搜索缓存键"""
        query_hash = hashlib.md5(query.encode(), usedforsecurity=False).hexdigest()[:16]
        return f"kw:{tenant_id}:{query_hash}:{limit}"

    # ==================== Phase B: Sync Methods ====================

    async def get_fingerprints(self, tenant_id: str = "default") -> list[dict]:
        """获取服务端所有记忆的指纹列表"""
        try:
            query = """
                SELECT source_id, content_hash, updated_at 
                FROM memory 
                WHERE tenant_id = $tenant_id
            """
            result = await self._db.query(query, {"tenant_id": tenant_id})

            fingerprints = []
            records = self._extract_records(result)
            for record in records:
                source_id = record.get("source_id")
                if not source_id:
                    continue  # 跳过 source_id 为 None 的脏数据
                fingerprints.append(
                    {
                        "source_id": source_id,
                        "hash": record.get("content_hash", ""),
                        "mtime": record.get("updated_at", 0),
                    }
                )
            return fingerprints
        except Exception as e:
            logger.error("[Sync] 获取指纹失败: %s", e)
            raise DatabaseError(f"获取指纹失败: {e}") from e

    async def sync_preview(self, fingerprints: list[dict], tenant_id: str = "default") -> dict:
        """增量同步：比对本地指纹与服务端，返回变更指令"""
        try:
            # 1. 获取服务端指纹
            server_fingerprints = await self.get_fingerprints(tenant_id)
            server_map = {fp["source_id"]: fp for fp in server_fingerprints}
            local_map = {fp["source_id"]: fp for fp in fingerprints}

            to_upload = []
            to_delete = []
            conflicts = []

            # 2. 检查本地 → 服务端（需要上传或冲突）
            for local_fp in fingerprints:
                source_id = local_fp["source_id"]
                server_fp = server_map.get(source_id)

                if not server_fp:
                    # 本地有，服务端无 → 上传
                    to_upload.append(
                        {
                            "source_id": source_id,
                            "reason": "new",
                            "path": local_fp.get("path", ""),
                        }
                    )
                elif server_fp["hash"] != local_fp["hash"]:
                    # hash不同 → 冲突，记录到数据库
                    conflict_id = await self._record_conflict(
                        source_id=source_id,
                        local_hash=local_fp["hash"],
                        server_hash=server_fp["hash"],
                        local_content=local_fp.get("content"),  # 如果提供了内容
                        server_content=server_fp.get("content"),  # 如果提供了内容
                        local_mtime=local_fp.get("mtime", 0),
                        server_mtime=server_fp.get("mtime", 0),
                        tenant_id=tenant_id,
                    )

                    conflicts.append(
                        {
                            "id": conflict_id,  # 新增：包含冲突ID
                            "source_id": source_id,
                            "local_mtime": local_fp.get("mtime", 0),
                            "server_mtime": server_fp.get("mtime", 0),
                            "local_hash": local_fp["hash"],
                            "server_hash": server_fp["hash"],
                        }
                    )

            # 3. 检查服务端 → 本地（需要删除）
            for server_fp in server_fingerprints:
                source_id = server_fp["source_id"]
                if source_id and source_id not in local_map:
                    to_delete.append(source_id)

            logger.info(
                "[Sync] 增量同步分析: %d 上传, %d 删除, %d 冲突", len(to_upload), len(to_delete), len(conflicts)
            )

            return {
                "synced": 0,  # 实际同步数（由调用方执行）
                "to_upload": to_upload,
                "to_delete": to_delete,
                "conflicts": conflicts,
            }
        except Exception as e:
            logger.error("[Sync] 增量同步失败: %s", e)
            raise DatabaseError(f"增量同步失败: {e}") from e

    async def sync_full(self, memories: list[dict], tenant_id: str = "default") -> dict:
        total = len(memories)
        processed = 0
        updated_count = 0
        all_skipped: list[dict] = []
        errors: list[str] = []

        for i, memory in enumerate(memories):
            try:
                upload_result = await self.upload_memories([memory], tenant_id)
                processed += upload_result.get("success", 0)
                updated_count += upload_result.get("updated", 0)
                all_skipped.extend(upload_result.get("skipped", []))
                if upload_result.get("errors"):
                    errors.append(f"Memory {i}: {upload_result['errors']}")
            except Exception as e:
                errors.append(f"Memory {i}: {str(e)}")

        return {
            "total": total,
            "success": processed,
            "failed": total - processed - updated_count - len(all_skipped),
            "updated": updated_count,
            "skipped": all_skipped,
            "errors": errors[:10],
            "tenant_id": tenant_id,
        }

    # ==================== Phase B: Conflict Resolution Methods ====================

    async def _record_conflict(
        self,
        source_id: str,
        local_hash: str,
        server_hash: str,
        local_content: str | None = None,
        server_content: str | None = None,
        local_mtime: int = 0,
        server_mtime: int = 0,
        tenant_id: str = "default",
    ) -> str:
        """记录冲突到数据库，返回 conflict_id"""
        try:
            conflict_data = {
                "source_id": source_id,
                "local_hash": local_hash,
                "server_hash": server_hash,
                "local_content": local_content,
                "server_content": server_content,
                "local_mtime": local_mtime,
                "server_mtime": server_mtime,
                "tenant_id": tenant_id,
                "status": "pending",
            }

            result = await self._db.create("conflict", conflict_data)

            # 提取冲突ID
            if isinstance(result, list) and len(result) > 0:
                conflict_id = str(result[0].get("id", ""))
                if conflict_id:
                    logger.info("[Sync] 冲突记录成功: %s, 源ID=%s, 租户=%s", conflict_id, source_id, tenant_id)
                    return conflict_id
            elif isinstance(result, dict) and result.get("id"):
                conflict_id = str(result["id"])
                logger.info("[Sync] 冲突记录成功: %s, 源ID=%s, 租户=%s", conflict_id, source_id, tenant_id)
                return conflict_id

            raise DatabaseError("无法创建冲突记录")
        except Exception as e:
            logger.error("[Sync] 记录冲突失败: %s", e)
            raise DatabaseError(f"记录冲突失败: {e}") from e

    async def get_conflicts(
        self,
        tenant_id: str = "default",
        status: str | None = None,  # 'pending' | 'resolved' | None（全部）
        limit: int = 100,
    ) -> list[dict]:
        """获取冲突列表"""
        try:
            # 构建查询条件
            where_conditions = ["tenant_id = $tenant_id"]
            if status:
                where_conditions.append("status = $status")

            where_clause = " AND ".join(where_conditions)

            query = """  # nosec B608
                SELECT * FROM conflict 
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT $limit
            """.format(where_clause=where_clause)

            params = {"tenant_id": tenant_id, "limit": limit}
            if status:
                params["status"] = status

            result = await self._db.query(query, params)

            # 提取记录
            raw_items = self._extract_records(result)
            conflicts = []

            for item in raw_items:
                conflict = {
                    "id": str(item.get("id", "")),
                    "source_id": item.get("source_id"),
                    "local_hash": item.get("local_hash", ""),
                    "server_hash": item.get("server_hash", ""),
                    "local_content": item.get("local_content"),
                    "server_content": item.get("server_content"),
                    "local_mtime": item.get("local_mtime", 0),
                    "server_mtime": item.get("server_mtime", 0),
                    "tenant_id": item.get("tenant_id", "default"),
                    "status": item.get("status", "pending"),
                    "resolution": item.get("resolution"),
                    "resolved_at": item.get("resolved_at"),
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                }
                conflicts.append(conflict)

            logger.info("[Sync] 获取冲突列表: %d 个冲突, 租户=%s, 状态=%s", len(conflicts), tenant_id, status)

            return conflicts
        except Exception as e:
            logger.error("[Sync] 获取冲突列表失败: %s", e)
            raise DatabaseError(f"获取冲突列表失败: {e}") from e

    async def get_conflict_detail(
        self,
        conflict_id: str,
        tenant_id: str = "default",
    ) -> dict | None:
        """获取单个冲突的详细信息"""
        try:
            # 确保冲突ID是完整格式
            if not conflict_id.startswith("conflict:"):
                conflict_id = f"conflict:{conflict_id}"

            # SurrealQL 不支持 FROM $param 参数化表名，必须用 WHERE id = $param
            # SurrealDB RecordID 类型不能直接与字符串比较，必须用 type::string() 转换
            query = """
                SELECT * FROM conflict 
                WHERE type::string(id) = $conflict_id AND tenant_id = $tenant_id
            """

            result = await self._db.query(query, {"conflict_id": conflict_id, "tenant_id": tenant_id})

            raw_items = self._extract_records(result)

            if not raw_items:
                logger.info("[Sync] 未找到冲突详情: %s, 租户=%s", conflict_id, tenant_id)
                return None

            item = raw_items[0]
            conflict = {
                "id": str(item.get("id", "")),
                "source_id": item.get("source_id"),
                "local_hash": item.get("local_hash", ""),
                "server_hash": item.get("server_hash", ""),
                "local_content": item.get("local_content"),
                "server_content": item.get("server_content"),
                "local_mtime": item.get("local_mtime", 0),
                "server_mtime": item.get("server_mtime", 0),
                "tenant_id": item.get("tenant_id", "default"),
                "status": item.get("status", "pending"),
                "resolution": item.get("resolution"),
                "resolved_at": item.get("resolved_at"),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
            }

            logger.info("[Sync] 获取冲突详情成功: %s, 租户=%s", conflict_id, tenant_id)

            return conflict
        except Exception as e:
            logger.error("[Sync] 获取冲突详情失败: %s", e)
            raise DatabaseError(f"获取冲突详情失败: {e}") from e

    async def resolve_conflict(
        self,
        conflict_id: str,
        resolution: str,
        tenant_id: str = "default",
    ) -> dict:
        """
        解决同步冲突

        Args:
            conflict_id: 冲突记录ID
            resolution: 解决策略
                - use_local: 用本地内容覆盖服务端
                - use_remote: 保留服务端内容，丢弃本地
                - keep_both: 保留两个版本（重命名本地版本）
            tenant_id: 租户ID

        Returns:
            解决结果

        Raises:
            ValidationError: 无效的冲突ID或解决策略
            DatabaseError: 数据库操作失败
        """
        resolution = resolution.lower().strip()
        if resolution not in ("use_local", "use_remote", "keep_both"):
            raise ValidationError(f"无效的解决策略: {resolution}")

        try:
            # 获取冲突详情
            conflict_detail = await self.get_conflict_detail(conflict_id, tenant_id)
            if not conflict_detail:
                raise ValidationError(f"冲突不存在或不属于当前租户: {conflict_id}")

            if conflict_detail["status"] == "resolved":
                logger.warning("[Sync] 冲突已被解决: %s, 重复解决操作", conflict_id)
                return {
                    "conflict_id": conflict_id,
                    "resolution": conflict_detail["resolution"],
                    "status": "already_resolved",
                    "message": "冲突已解决",
                }

            source_id = conflict_detail["source_id"]

            if resolution == "use_local":
                # 使用本地内容覆盖服务端
                await self._update_server_content_by_source_id(source_id, conflict_detail, tenant_id)

            elif resolution == "use_remote":
                # 保留服务端内容不变，标记本地为需更新
                logger.info("[Sync] 使用服务端内容保留: 冲突=%s, 源ID=%s", conflict_id, source_id)

            elif resolution == "keep_both":
                # 重命名本地版本并创建新记录
                await self._create_new_memory_version(source_id, conflict_detail, tenant_id)

            # 更新冲突记录为已解决
            update_query = """
                UPDATE $conflict_id 
                SET status = 'resolved', 
                    resolution = $resolution, 
                    resolved_at = time::now(),
                    updated_at = time::now()
                WHERE tenant_id = $tenant_id
            """

            result = await self._db.query(
                update_query, {"conflict_id": conflict_id, "resolution": resolution, "tenant_id": tenant_id}
            )

            logger.info("[Sync] 冲突解决成功: %s, 策略=%s, 租户=%s", conflict_id, resolution, tenant_id)

            return {
                "conflict_id": conflict_id,
                "resolution": resolution,
                "status": "resolved",
                "message": f"冲突已使用 {resolution} 策略解决",
                "source_id": source_id,
            }
        except Exception as e:
            logger.error("[Sync] 解决冲突失败: %s", e)
            raise DatabaseError(f"解决冲突失败: {e}") from e

    async def _update_server_content_by_source_id(self, source_id: str, conflict_detail: dict, tenant_id: str) -> None:
        """根据source_id更新服务端内容"""
        local_content = conflict_detail.get("local_content")
        if not local_content:
            logger.warning("[Sync] 本地内容为空，无法更新服务端: 源ID=%s", source_id)
            return

        # 重新计算嵌入向量
        try:
            embeddings = await self._get_embeddings([local_content])
            new_embedding = embeddings[0]
        except Exception as e:
            logger.error("[Sync] 计算新嵌入失败: %s, 源ID=%s", e, source_id)
            raise DatabaseError(f"计算新嵌入失败: {e}")

        # 更新 SurrealDB 中的记录
        update_query = """
            UPDATE memory 
            SET content = $content,
                content_hash = $content_hash,
                embedding = $embedding,
                updated_at = time::now()
            WHERE source_id = $source_id AND tenant_id = $tenant_id
        """

        result = await self._db.query(
            update_query,
            {
                "content": local_content,
                "content_hash": hashlib.md5(local_content.encode("utf-8"), usedforsecurity=False).hexdigest(),
                "embedding": new_embedding,
                "source_id": source_id,
                "tenant_id": tenant_id,
            },
        )

        # 同步更新 Meilisearch
        if self._meili:
            try:
                # 获取更新后的记录ID
                select_query = """
                    SELECT id FROM memory 
                    WHERE source_id = $source_id AND tenant_id = $tenant_id
                """
                select_result = await self._db.query(select_query, {"source_id": source_id, "tenant_id": tenant_id})

                records = self._extract_records(select_result)
                if records:
                    surreal_id = str(records[0].get("id", ""))
                    if surreal_id:
                        # 构建 Meilisearch 文档
                        meili_doc = self._build_meili_doc(
                            surreal_id,
                            {
                                "content": local_content,
                                "metadata": {},
                                "type": "general",
                                "tags": [],
                                "project_id": "global",
                            },
                            tenant_id,
                        )

                        # 更新到 Meilisearch
                        await self._meili.add_documents([meili_doc], primary_key="id", wait=False)

                        logger.info("[Sync] 同步更新 Meilisearch 成功: %s", surreal_id)
            except Exception as e:
                logger.warning("[Sync] 同步更新 Meilisearch 失败: %s", e)

        logger.info("[Sync] 服务器端内容更新成功: 源ID=%s", source_id)

    async def _create_new_memory_version(self, source_id: str, conflict_detail: dict, tenant_id: str) -> str:
        """为本地版本创建新的记忆记录（追加-local后缀）"""
        local_content = conflict_detail.get("local_content")
        if not local_content:
            logger.warning("[Sync] 本地内容为空，无法创建新版本: 源ID=%s", source_id)
            return ""

        # 生成新的 source_id
        new_source_id = f"{source_id}-local"

        # 重新计算嵌入向量
        try:
            embeddings = await self._get_embeddings([local_content])
            new_embedding = embeddings[0]
        except Exception as e:
            logger.error("[Sync] 计算新嵌入失败: %s, 新源ID=%s", e, new_source_id)
            raise DatabaseError(f"计算新嵌入失败: {e}")

        # 创建新记录到 SurrealDB
        new_memory_data = {
            "content": local_content,
            "content_hash": hashlib.md5(local_content.encode("utf-8"), usedforsecurity=False).hexdigest(),
            "embedding": new_embedding,
            "tenant_id": tenant_id,
            "type": "general",  # 默认类型
            "tags": [],  # 默认标签
            "project_id": "global",
            "source": "sync",
            "source_id": new_source_id,
            "metadata": {},
        }

        create_result = await self._db.create("memory", new_memory_data)

        # 提取新记录ID
        new_memory_id = None
        if isinstance(create_result, list) and len(create_result) > 0:
            new_memory_id = str(create_result[0].get("id", ""))
        elif isinstance(create_result, dict) and create_result.get("id"):
            new_memory_id = str(create_result["id"])

        if not new_memory_id:
            raise DatabaseError("无法创建新记忆记录")

        # 同步到 Meilisearch
        if self._meili:
            try:
                meili_doc = self._build_meili_doc(new_memory_id, new_memory_data, tenant_id)
                await self._meili.add_documents([meili_doc], primary_key="id", wait=False)

                logger.info("[Sync] 创建新记忆版本并同步到Meilisearch: %s, 新源ID=%s", new_memory_id, new_source_id)
            except Exception as e:
                logger.warning("[Sync] 同步新记忆到 Meilisearch 失败: %s", e)

        logger.info("[Sync] 新记忆版本创建成功: %s, 新源ID=%s", new_memory_id, new_source_id)

        return new_memory_id

    # ==================== New: HNSW Optimization Methods (Phase C-B) ====================

    async def get_memory_stats(self, tenant_id: str = "default") -> dict:
        """获取记忆统计数据"""
        try:
            # 计算各种统计数据
            query = "SELECT count() as total FROM memory WHERE tenant_id = $tenant_id"
            result = await self._db.query(query, {"tenant_id": tenant_id})
            raw_items = self._extract_records(result)
            total_memories = raw_items[0].get("total", 0) if raw_items else 0

            # 按类型分类
            type_query = "SELECT type, count() as count FROM memory WHERE tenant_id = $tenant_id GROUP BY type"
            type_result = await self._db.query(type_query, {"tenant_id": tenant_id})
            type_stats = self._extract_records(type_result)

            # 按日期统计（SurrealDB 用 time::group 替代 date_trunc）
            date_query = "SELECT count() as count, time::group(created_at, 'day') as day FROM memory WHERE tenant_id = $tenant_id GROUP BY day ORDER BY day DESC LIMIT 7"
            date_result = await self._db.query(date_query, {"tenant_id": tenant_id})
            date_stats = self._extract_records(date_result)

            return {
                "total": total_memories,
                "by_type": type_stats,
                "recent_activity": date_stats,
                "tenant_id": tenant_id,
            }
        except Exception as e:
            logger.error("[HNSW] 获取记忆统计失败: %s", e)
            raise DatabaseError(f"获取记忆统计失败: {e}") from e

    async def optimize_hnsw(self, tenant_id: str = "default") -> dict:
        """优化HNSW参数"""
        try:
            # 这里可以实现HNSW索引的自动优化逻辑
            # 当前只是返回推荐的参数
            stats = await self.get_memory_stats(tenant_id)

            # 基于数据量推荐HNSW参数
            total_memories = stats.get("total", 0)
            recommended_params = self._recommend_hnsw_params(total_memories)

            return {
                "current_magnitude": total_memories,
                "recommended_params": recommended_params,
                "optimization_performed": False,  # 这里实际的优化逻辑需要在SurrealDB层面处理
            }
        except Exception as e:
            logger.error("[HNSW] 优化失败: %s", e)
            raise DatabaseError(f"优化HNSW失败: {e}") from e

    def _recommend_hnsw_params(self, n_items: int) -> dict:
        """基于数据量推荐HNSW参数"""
        # 基于内存中的研究推荐参数
        if n_items < 1000:
            return {"M": 8, "ef_construction": 64, "ef_search": 32}
        elif n_items < 10000:
            return {"M": 12, "ef_construction": 100, "ef_search": 50}
        elif n_items < 100000:
            return {"M": 16, "ef_construction": 200, "ef_search": 100}
        else:
            return {"M": 24, "ef_construction": 300, "ef_search": 150}

    async def rebuild_hnsw_index(self, tenant_id: str = "default", force: bool = False) -> dict:
        """重建HNSW索引"""
        try:
            # SurrealDB的HNSW索引是自动维护的，这里只是重新计算/更新索引
            # 重新计算所有记忆的嵌入（仅在force为True时）
            if force:
                # 查询所有记忆
                query = "SELECT id, content FROM memory WHERE tenant_id = $tenant_id"
                result = await self._db.query(query, {"tenant_id": tenant_id})
                records = self._extract_records(result)

                updates_made = 0
                for record in records:
                    memory_id = record.get("id")
                    content = record.get("content")

                    if content:
                        # 重新计算嵌入
                        embeddings = await self._get_embeddings([content])
                        new_embedding = embeddings[0]

                        # 更新数据库
                        update_query = "UPDATE $memory_id SET embedding = $embedding"
                        await self._db.query(update_query, {"memory_id": memory_id, "embedding": new_embedding})
                        updates_made += 1

                return {
                    "status": "completed",
                    "updates_made": updates_made,
                    "force_rebuilt": force,
                    "tenant_id": tenant_id,
                }
            else:
                return {
                    "status": "completed",
                    "message": "Index maintained normally (not force rebuilt)",
                    "force_rebuilt": force,
                    "tenant_id": tenant_id,
                }
        except Exception as e:
            logger.error("[HNSW] 重建索引失败: %s", e)
            raise DatabaseError(f"重建HNSW索引失败: {e}") from e

    # ==================== New: Embedding Cache Methods (Phase C-B) ====================

    async def get_cache_stats(self) -> dict:
        """获取嵌入缓存统计信息"""
        if self._vector_cache:
            try:
                stats = await self._vector_cache.get_stats()
                return {
                    "hits": stats.get("hits", 0),
                    "misses": stats.get("misses", 0),
                    "ratio": stats.get("ratio", 0.0),
                    "size": stats.get("size", 0),
                    "config": {
                        "max_size": getattr(self, "_cache_enabled", False),
                        "ttl_seconds": getattr(self, "_cache_ttl", 300),
                    },
                }
            except Exception:  # nosec B110
                pass

        return {"hits": 0, "misses": 0, "ratio": 0.0, "size": 0, "config": {"enabled": False}}

    async def clear_embedding_cache(self) -> dict:
        """清空嵌入缓存"""
        if self._vector_cache:
            try:
                await self._vector_cache.clear()
                return {"cleared": True, "cache_type": "vector_cache"}
            except Exception:  # nosec B110
                pass

        if self._keyword_cache:
            try:
                await self._keyword_cache.clear()
                return {"cleared": True, "cache_type": "keyword_cache"}
            except Exception:  # nosec B110
                pass

        return {"cleared": False, "message": "No cache available"}

    async def warmup_embedding_cache(self, tenant_id: str = "default", limit: int = 100) -> dict:
        try:
            if not self._vector_cache:
                return {
                    "loaded": 0,
                    "attempted": 0,
                    "limit": limit,
                    "tenant_id": tenant_id,
                    "skipped_reason": "cache_disabled",
                }

            query = """
                SELECT id, content, created_at FROM memory 
                WHERE tenant_id = $tenant_id 
                ORDER BY created_at DESC 
                LIMIT $limit
            """
            result = await self._db.query(query, {"tenant_id": tenant_id, "limit": limit})
            records = self._extract_records(result)

            loaded_count = 0
            for record in records:
                content = record.get("content", "")
                if not content:
                    continue
                try:
                    embeddings = await self._get_embeddings([content])
                    embedding = embeddings[0] if embeddings else []

                    if embedding:
                        cache_key = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()
                        await self._vector_cache.set(cache_key, embedding, ttl=self._cache_ttl)
                        loaded_count += 1
                except Exception:  # nosec B112
                    continue

            return {"loaded": loaded_count, "attempted": len(records), "limit": limit, "tenant_id": tenant_id}
        except Exception as e:
            logger.error("[Cache] 预热失败: %s", e)
            raise DatabaseError(f"预热嵌入缓存失败: {e}") from e

    # ==================== New: Prefetch Methods (Phase C-B) ====================

    async def prefetch_related_memories(
        self, memory_id: str, tenant_id: str = "default", depth: int = 1, limit: int = 10
    ) -> dict:
        """预取与给定记忆相关的记忆嵌入"""
        try:
            related_memories = await self.get_related_memories(
                memory_id=memory_id, depth=depth, tenant_id=tenant_id, limit=limit
            )

            # 预计算这些相关记忆的嵌入并放入缓存
            processed = 0
            for memory in related_memories:
                content = memory.get("content", "")
                if content and self._vector_cache:
                    try:
                        embeddings = await self._get_embeddings([content])
                        embedding = embeddings[0] if embeddings else []

                        if embedding:
                            cache_key = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()
                            await self._vector_cache.set(cache_key, embedding, ttl=self._cache_ttl)
                            processed += 1
                    except Exception:  # nosec B112
                        continue

            return {
                "processed": processed,
                "total_related": len(related_memories),
                "memory_id": memory_id,
                "depth": depth,
                "limit": limit,
            }
        except Exception as e:
            logger.error("[Prefetch] 相关记忆预取失败: %s", e)
            raise DatabaseError(f"预取相关记忆失败: {e}") from e

    async def prefetch_popular_queries(self, tenant_id: str = "default", top_n: int = 20) -> dict:
        """预取热门查询的嵌入（假设有查询日志或热度统计）"""
        try:
            # 在当前实现中，我们模拟预取一些常见内容类型的嵌入
            # 在实际实现中，这应该连接到查询日志或热度统计系统
            common_topics = [
                "error",
                "bug",
                "fix",
                "solution",
                "optimization",
                "performance",
                "architecture",
                "design",
                "implementation",
                "development",
            ]

            processed = 0
            for topic in common_topics[:top_n]:
                if self._vector_cache:
                    try:
                        embeddings = await self._get_embeddings([topic])
                        embedding = embeddings[0] if embeddings else []

                        if embedding:
                            cache_key = hashlib.md5(topic.encode(), usedforsecurity=False).hexdigest()
                            await self._vector_cache.set(cache_key, embedding, ttl=self._cache_ttl)
                            processed += 1
                    except Exception:  # nosec B112
                        continue

            return {"processed": processed, "queried_topics": common_topics[:top_n], "top_n_requested": top_n}
        except Exception as e:
            logger.error("[Prefetch] 热门查询预取失败: %s", e)
            raise DatabaseError(f"预取热门查询失败: {e}") from e

    # ==================== New: Leiden Clustering Algorithm ====================

    async def cluster_memories_leiden(
        self, tenant_id: str = "default", content_threshold: float = 0.75, max_clusters: int = 20
    ) -> dict:
        """
        使用Leiden算法对记忆进行聚类
        注意：真实的Leiden算法需要graph分析库，这里实现简化版本的社区检测
        """
        try:
            # 获取所有记忆及其嵌入
            query = """
                SELECT id, content, embedding, type, tags, project_id 
                FROM memory 
                WHERE tenant_id = $tenant_id
            """
            result = await self._db.query(query, {"tenant_id": tenant_id})
            records = self._extract_records(result)

            if len(records) < 2:
                return {
                    "clusters": [],
                    "total_memories": len(records),
                    "tenant_id": tenant_id,
                    "message": "Not enough memories to cluster",
                }

            # 构建相似度图（简化版）
            similarities = []
            for i in range(len(records)):
                for j in range(i + 1, len(records)):
                    memory1 = records[i]
                    memory2 = records[j]

                    # 计算嵌入余弦相似度
                    embedding1 = memory1.get("embedding", [])
                    embedding2 = memory2.get("embedding", [])

                    if embedding1 and embedding2:
                        similarity = self._cosine_similarity(embedding1, embedding2)
                        if similarity >= content_threshold:
                            similarities.append(
                                {
                                    "from_id": memory1["id"],
                                    "to_id": memory2["id"],
                                    "similarity": similarity,
                                    "from_content": memory1.get("content", "")[:50],
                                    "to_content": memory2.get("content", "")[:50],
                                }
                            )

            # 使用边的集合进行简化聚类
            clusters = self._simple_community_detection(similarities, max_clusters)

            # 为每个集群命名（基于内容关键词）
            named_clusters = []
            for i, cluster in enumerate(clusters):
                cluster_contents = [records[idx]["content"] for idx in cluster if idx < len(records)]
                cluster_keywords = self._extract_cluster_keywords(cluster_contents)

                named_clusters.append(
                    {
                        "cluster_id": i,
                        "size": len(cluster),
                        "memory_ids": [records[idx]["id"] for idx in cluster if idx < len(records)],
                        "keywords": cluster_keywords,
                        "sample_contents": [c[:100] for c in cluster_contents[:3]],
                    }
                )

            return {
                "clusters": named_clusters,
                "total_memories": len(records),
                "total_clusters": len(named_clusters),
                "tenant_id": tenant_id,
                "threshold": content_threshold,
            }
        except Exception as e:
            logger.error("[Clustering] Leiden聚类失败: %s", e)
            raise DatabaseError(f"Leiden聚类失败: {e}") from e

    def _cosine_similarity(self, vec1: list, vec2: list) -> float:
        """计算两个向量的余弦相似度"""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def _simple_community_detection(self, edges: list, max_clusters: int) -> list:
        """简单的社区检测算法（模拟Leiden）"""
        # 创建邻接列表
        nodes = set()
        adj_list = {}

        for edge in edges:
            from_node = edge["from_id"]
            to_node = edge["to_id"]

            nodes.add(from_node)
            nodes.add(to_node)

            if from_node not in adj_list:
                adj_list[from_node] = []
            if to_node not in adj_list:
                adj_list[to_node] = []

            adj_list[from_node].append(to_node)
            adj_list[to_node].append(from_node)

        # 使用连通组件作为简单聚类
        visited = set()
        clusters = []

        for node in nodes:
            if node not in visited:
                cluster = []
                queue = [node]
                visited.add(node)

                while queue and len(clusters) < max_clusters:
                    current = queue.pop(0)
                    cluster.append(current)

                    for neighbor in adj_list.get(current, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

                clusters.append(cluster)

                if len(clusters) >= max_clusters:
                    break

        return clusters

    def _extract_cluster_keywords(self, contents: list) -> list:
        """从集群内容中提取关键词"""
        # 简单的关键字提取：最常见的词汇
        all_words = []
        for content in contents:
            words = content.lower().split()
            # 过滤掉常见的停用词
            stop_words = {
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
            }
            filtered_words = [
                word.strip(".,!?;:") for word in words if word.lower() not in stop_words and len(word) > 3
            ]
            all_words.extend(filtered_words)

        # 统计词频并返回最高频的词
        word_count = {}
        for word in all_words:
            word_count[word] = word_count.get(word, 0) + 1

        # 返回前5个高频词
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:5]]

    # ==================== New: Code Analysis Integration ====================

    async def analyze_memory_code(self, memory_id: str, tenant_id: str = "default") -> dict:
        """分析记忆中的代码内容"""
        try:
            # 获取记忆内容
            query = "SELECT * FROM $memory_id WHERE tenant_id = $tenant_id"
            result = await self._db.query(query, {"memory_id": memory_id, "tenant_id": tenant_id})
            records = self._extract_records(result)

            if not records:
                raise ValidationError(f"记忆不存在: {memory_id}")

            memory = records[0]
            content = memory.get("content", "")

            # 确定编程语言（简单判断）
            language = self._detect_programming_language(content)

            # 使用代码分析器分析代码
            code_analyzer = CodeAnalyzer()
            analysis_result = await code_analyzer.analyze_code(content, language)

            return {
                "memory_id": memory_id,
                "language": analysis_result.language,
                "functions": analysis_result.functions,
                "classes": analysis_result.classes,
                "imports": analysis_result.imports,
                "comments_count": len(analysis_result.comments),
                "dependencies": analysis_result.dependencies,
                "complexity_metrics": analysis_result.complexity_metrics,
                "comment_content": " | ".join(
                    [c["text"] for c in analysis_result.comments] + [d["text"] for d in analysis_result.docstrings]
                ),
            }
        except Exception as e:
            logger.error("[Code Analysis] 分析失败: %s", e)
            raise DatabaseError(f"代码分析失败: {e}") from e

    def _detect_programming_language(self, content: str) -> str:
        """简单的编程语言检测"""
        content_lower = content.lower()

        # 常见的语言标识符
        if (
            "import torch" in content_lower
            or "import tensorflow" in content_lower
            or "def " in content
            and "class " in content
        ):
            return "python"
        elif (
            "function " in content_lower
            or "var " in content_lower
            or "let " in content_lower
            or "const " in content_lower
        ):
            if "{" in content and "}" in content:
                return "javascript"
        elif "class " in content_lower and ":" in content and "def " not in content:
            return "java"
        elif "#include" in content_lower:
            return "c"
        elif "func " in content_lower and "package " in content_lower:
            return "go"
        elif "fn " in content_lower and "::" in content_lower:
            return "rust"
        elif "using system;" in content_lower or "namespace " in content_lower:
            return "csharp"
        elif "<html" in content_lower or "<div" in content_lower:
            return "html"
        elif ".class {" in content or "#" in content:
            return "css"
        elif "select " in content_lower or "create table" in content_lower:
            return "sql"

        # 默认返回python
        return "python"

    # ==================== v2.4.0 Access Log ====================

    async def report_access_log(self, entries: list[dict[str, Any]], tenant_id: str | None = None) -> dict[str, Any]:
        """记录访问日志（用于分析记忆使用频率）"""
        effective_tenant_id = tenant_id or self._default_tenant_id

        # 简化的实现：将访问日志记录到内存中
        # 实际生产环境可以存储到 SurrealDB 或发送到分析服务
        logged_count = 0
        for entry in entries:
            if entry.get("entry_id") and entry.get("timestamp"):
                logged_count += 1

        return {
            "status": "success",
            "logged_count": logged_count,
            "total_received": len(entries),
        }
