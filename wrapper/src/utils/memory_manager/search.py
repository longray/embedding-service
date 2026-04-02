"""搜索路由、RRF 融合、结果格式化"""

import asyncio
import logging
from typing import Any

from ..exceptions import DatabaseError, ValidationError, EmbeddingError
from ..tracing import get_tracer

logger = logging.getLogger(__name__)


class SearchMixin:
    """搜索相关方法"""

    def _filter_by_level(self, results: list[dict[str, Any]], level: int) -> list[dict[str, Any]]:
        """按 level 过滤返回结果 (v2.4.0)"""
        filtered = []
        for r in results:
            item = {"id": r.get("id"), "score": r.get("score")}

            if level >= 0:
                item["abstract"] = r.get("abstract", "")

            if level >= 1:
                item["overview"] = r.get("overview", "")

            if level >= 2:
                item["content"] = r.get("content", "")
                item["type"] = r.get("type")
                item["tags"] = r.get("tags", [])
                item["project_id"] = r.get("project_id")
                item["local_id"] = r.get("local_id")
                item["metadata"] = r.get("metadata", {})

            filtered.append(item)

        return filtered

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
                "SELECT id, content, abstract, overview, local_id, metadata, type, tags, project_id, "
                "vector::similarity::cosine(embedding, $query_embedding) AS score "
                "FROM memory "
                "WHERE tenant_id = $tenant_id "
                "AND vector::similarity::cosine(embedding, $query_embedding) >= $threshold "
                "ORDER BY score DESC "
                "LIMIT $limit"
            )
            result = await self._db_query(
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
                "SELECT id, content, abstract, overview, local_id, metadata, type, tags, project_id, "
                "search::score(1) AS score "
                "FROM memory "
                "WHERE tenant_id = $tenant_id "
                f"AND content @1@ '{safe_query}' "  # nosec B608
                "ORDER BY score DESC "
                "LIMIT $limit"
            )
            result = await self._db_query(q, {"tenant_id": tenant_id, "limit": limit})
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
        results: list[dict[str, Any]] = []
        for hit in meili_result.get("hits", []):
            doc_id = hit.get("id", "")
            if doc_id and not doc_id.startswith("memory:"):
                doc_id = f"memory:{doc_id}"
            score = hit.get("_rankingScore", 0.0)
            results.append(
                {
                    "id": str(doc_id),
                    "content": hit.get("content", ""),
                    "abstract": hit.get("abstract", ""),
                    "overview": hit.get("overview", ""),
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
            "abstract": item.get("abstract", ""),
            "overview": item.get("overview", ""),
            "local_id": item.get("local_id"),
            "metadata": item.get("metadata", {}),
            "type": item.get("type", "general"),
            "tags": item.get("tags", []),
            "project_id": item.get("project_id", "global"),
            "score": score,
        }
