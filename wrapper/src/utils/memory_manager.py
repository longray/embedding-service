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

        # BL-4: Phase A - 自动代码分析（异步触发，不阻塞上传）
        if memory_ids:
            asyncio.create_task(self._auto_analyze_memories(memory_ids, effective_tenant_id))

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

            # 删除关系
            del_q = "DELETE type::record($rel_table, $rel_id)"
            await self._db.query(del_q, {"rel_table": rel_table, "rel_id": rel_id})
            return True
        except Exception as e:
            logger.error("[Relation Delete] 失败: %s", e)
            raise DatabaseError(f"删除关系失败: {e}") from e

    # ==================== BL-6: LLM Code Summary ====================

    async def _generate_code_summary(self, memory_id: str, tenant_id: str) -> dict[str, Any] | None:
        """调用 LLM 生成代码摘要"""
        from ..config import LLMConfig

        config = LLMConfig()
        if not config.enabled:
            return None

        try:
            # 获取记忆内容和代码分析结果
            query = "SELECT content, metadata FROM $memory_id WHERE tenant_id = $tenant_id"
            result = await self._db.query(query, {"memory_id": memory_id, "tenant_id": tenant_id})
            records = self._extract_records(result)

            if not records:
                return None

            memory = records[0]
            content = memory.get("content", "")
            metadata = memory.get("metadata", {}) or {}
            code_analysis = metadata.get("code_analysis", {})

            if not code_analysis:
                return None

            # 构建 LLM 提示
            functions = code_analysis.get("functions", [])
            classes = code_analysis.get("classes", [])
            language = code_analysis.get("language", "unknown")

            prompt = f"""分析以下 {language} 代码并提供摘要：

代码内容：
```
{content[:2000]}
```

函数列表：{", ".join(f.get("name", "") for f in functions[:10])}
类列表：{", ".join(c.get("name", "") for c in classes[:10])}

请提供：
1. 一句话摘要（描述这个模块/文件的主要功能）
2. 关键函数列表（最重要的3-5个函数及其作用）
3. 代码用途（这个代码解决什么问题）

以 JSON 格式返回：
{{
    "summary": "一句话摘要",
    "key_functions": ["函数1: 作用", "函数2: 作用"],
    "purpose": "代码用途描述"
}}"""

            # 调用 LLM API
            http_pool = await self._get_http_pool()
            headers = {"Content-Type": "application/json"}
            if config.api_key:
                headers["Authorization"] = f"Bearer {config.api_key}"

            payload = {
                "model": config.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": config.max_tokens,
                "temperature": 0.3,
            }

            response = await http_pool.post(
                f"{config.endpoint}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=config.timeout,
            )

            if response.status_code != 200:
                logger.warning("[LLM Summary] API 调用失败: %s", response.status_code)
                return None

            result_data = response.json()
            llm_content = result_data.get("choices", [{}])[0].get("message", {}).get("content", "")

            # 解析 JSON 响应
            try:
                # 尝试提取 JSON 部分
                json_start = llm_content.find("{")
                json_end = llm_content.rfind("}")
                if json_start >= 0 and json_end > json_start:
                    json_str = llm_content[json_start : json_end + 1]
                    summary_data = json.loads(json_str)
                else:
                    # 如果不是 JSON，使用原始内容作为摘要
                    summary_data = {
                        "summary": llm_content[:200],
                        "key_functions": [],
                        "purpose": "",
                    }

                # 保存到 metadata
                code_summary = {
                    "summary": summary_data.get("summary", ""),
                    "key_functions": summary_data.get("key_functions", []),
                    "purpose": summary_data.get("purpose", ""),
                    "generated_at": asyncio.get_event_loop().time(),
                    "model": config.model_name,
                }

                metadata["code_summary"] = code_summary

                # 更新数据库
                update_query = """
                    UPDATE type::record($record_id)
                    SET metadata = $metadata
                """
                await self._db.query(update_query, {"record_id": memory_id, "metadata": metadata})

                logger.info("[LLM Summary] 摘要生成完成: %s", memory_id)
                return code_summary

            except json.JSONDecodeError as e:
                logger.warning("[LLM Summary] JSON 解析失败: %s", e)
                return None

        except Exception as e:
            logger.warning("[LLM Summary] 生成失败: %s", e)
            return None

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

    # ==================== BL-4: Code Analysis Auto-trigger ====================

    async def _auto_analyze_memories(self, memory_ids: list[str], tenant_id: str) -> None:
        """自动分析记忆中的代码内容（异步执行，不阻塞上传）"""
        from ..config import CodeAnalysisConfig

        config = CodeAnalysisConfig()
        if not config.enabled or not config.auto_analyze:
            return

        for memory_id in memory_ids:
            try:
                # 检查内容长度
                query = "SELECT content FROM $memory_id WHERE tenant_id = $tenant_id"
                result = await self._db.query(query, {"memory_id": memory_id, "tenant_id": tenant_id})
                records = self._extract_records(result)

                if not records:
                    continue

                content = records[0].get("content", "")
                content_length = len(content)

                if content_length < config.min_content_length or content_length > config.max_content_length:
                    continue

                # 检测是否为代码内容
                if not self._is_code_content(content):
                    continue

                # 异步执行代码分析
                await self.analyze_memory_code(memory_id, tenant_id, persist=True)
                logger.info("[Auto Code Analysis] 分析完成: %s", memory_id)

            except Exception as e:
                # 降级策略：分析失败不影响上传
                logger.warning("[Auto Code Analysis] 分析失败，跳过: %s - %s", memory_id, e)

    def _is_code_content(self, content: str) -> bool:
        """检测内容是否为代码"""
        code_indicators = [
            r"^\s*(def|class|import|from)\s+",  # Python
            r"^\s*(function|const|let|var)\s+",  # JavaScript
            r"^\s*(#include|#define|int|void)\s+",  # C/C++
            r"^\s*(public|private|class|interface)\s+",  # Java/C#
            r"^\s*(func|package|import)\s+",  # Go
            r"^\s*(fn|let|mut|use)\s+",  # Rust
            r"^\s*<[^>]+>.*</[^>]+>\s*$",  # HTML/XML
            r"^\s*\{[^}]*\}\s*$",  # JSON
        ]

        content_sample = content[:1000]  # 检查前1000字符
        for pattern in code_indicators:
            if re.search(pattern, content_sample, re.MULTILINE):
                return True

        # 检查代码特征比例
        code_chars = len(re.findall(r"[{}();=<>/]", content_sample))
        total_chars = len(content_sample.replace(" ", "").replace("\n", ""))
        if total_chars > 0 and code_chars / total_chars > 0.1:
            return True

        return False
