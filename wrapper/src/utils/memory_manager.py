"""记忆管理器模块

封装记忆的批量上传、搜索等业务逻辑。
支持 KNN 向量搜索（HNSW 索引）、BM25 关键词搜索和 RRF 混合搜索。
支持 SurrealDB RELATE 图关系操作（创建/查询/删除记忆间关联）。
多租户隔离通过 tenant_id 字段实现。
"""

import asyncio
import json
import re
from typing import Any

from .exceptions import DatabaseError, EmbeddingError, ValidationError
from .http_pool import get_http_pool


class MemoryManager:
    """记忆管理器，协调 embedding 服务和数据库操作

    支持多租户隔离和 SurrealDB 3.0 新查询语法：
    - 向量搜索: KNN <|K,EF|> 算子 + HNSW 索引
    - 关键词搜索: @1@ 全文搜索算子 + BM25 评分 + ngram(2,8) 中文分词
    - 混合搜索: RRF (Reciprocal Rank Fusion) 融合算法
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

    async def close(self) -> None:
        """关闭资源"""

    async def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """批量获取文本的 embedding 向量"""
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
            raise EmbeddingError(f"Failed to get embeddings: {e!s}") from e

    # ==================== 上传 ====================

    async def upload_memories(self, memories: list[dict[str, Any]], tenant_id: str | None = None) -> dict[str, Any]:
        """批量上传记忆

        字段映射（API dict → SurrealDB 记录）：
        - content → content (必需)
        - embedding → embedding (服务端计算)
        - tenant_id → tenant_id (默认 "default")
        - type → type (默认 "general")
        - tags → tags (默认 [])
        - project_id → project_id (默认 "global")
        - source_id → source_id (可选，UNIQUE 索引去重)
        - source → source (默认 "api")
        - source_timestamp → source_timestamp (可选)
        - classification_confidence → classification_confidence (可选)
        - metadata → metadata (兜底扩展字段)
        """
        if not memories:
            raise ValidationError("Memories list cannot be empty")

        effective_tenant_id = tenant_id or self._default_tenant_id

        total = len(memories)
        success_count = 0
        failed_count = 0
        memory_ids: list[str] = []
        errors: list[str] = []

        # 批量获取 embeddings
        texts = [m.get("content", "") for m in memories]
        try:
            embeddings = await self._get_embeddings(texts)
        except EmbeddingError as e:
            return {
                "total": total,
                "success": 0,
                "failed": total,
                "memory_ids": [],
                "errors": [str(e)],
            }

        for memory, embedding in zip(memories, embeddings, strict=False):
            try:
                # 构建 SurrealDB 记录数据（顶层字段映射）
                memory_data: dict[str, Any] = {
                    "content": memory.get("content", ""),
                    "embedding": embedding,
                    "tenant_id": effective_tenant_id,
                    "type": memory.get("type", "general"),
                    "tags": memory.get("tags", []),
                    "project_id": memory.get("project_id", "global"),
                    "source": memory.get("source", "api"),
                    "metadata": memory.get("metadata", {}),
                }

                # 可选字段（仅在提供时设置，否则由 Schema DEFAULT 处理）
                if "source_id" in memory:
                    memory_data["source_id"] = memory["source_id"]
                if "source_timestamp" in memory:
                    memory_data["source_timestamp"] = memory["source_timestamp"]
                if "classification_confidence" in memory:
                    memory_data["classification_confidence"] = memory["classification_confidence"]

                result = await self._db.create("memory", memory_data)
                # 提取成功创建的记录 ID
                record_id: str | None = None
                if isinstance(result, list) and len(result) > 0:
                    record_id = str(result[0].get("id", "")) or None
                elif isinstance(result, dict) and result.get("id"):
                    record_id = str(result["id"])

                if record_id:
                    memory_ids.append(record_id)
                    success_count += 1
                else:
                    # SDK 未抛异常但未返回有效 ID（如 UNIQUE 约束冲突）
                    failed_count += 1
                    errors.append("No record ID returned (possible UNIQUE constraint violation)")
            except Exception as e:
                failed_count += 1
                errors.append(f"{type(e).__name__}: {e!s}")

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
    ) -> dict[str, Any]:
        """搜索记忆

        Args:
            query: 搜索查询文本
            mode: 搜索模式 (vector/keyword/hybrid)
            limit: 返回结果数量限制
            threshold: 相似度阈值 (0.0-1.0，向量搜索使用，转换为 distance 过滤)
            tenant_id: 租户 ID（不传则使用默认值）
        """
        if mode not in ("vector", "keyword", "hybrid"):
            raise ValidationError(f"Invalid search mode: {mode}")

        effective_tenant_id = tenant_id or self._default_tenant_id

        try:
            if mode == "vector":
                embeddings = await self._get_embeddings([query])
                results = await self._search_by_vector(embeddings[0], limit, threshold, effective_tenant_id)
            elif mode == "keyword":
                results = await self._search_by_keyword(query, limit, effective_tenant_id)
            else:
                embeddings = await self._get_embeddings([query])
                results = await self._hybrid_search(query, embeddings[0], limit, threshold, effective_tenant_id)

            return {"results": results, "total": len(results), "mode": mode, "query": query}
        except Exception as e:
            raise DatabaseError(f"Search failed: {e!s}") from e

    async def _search_by_vector(
        self,
        embedding: list[float],
        limit: int,
        threshold: float,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        """KNN 向量搜索（利用 HNSW 索引）

        使用 <|K,EF|> 算子触发 HNSW 索引：
        - K = 返回最近邻数量
        - EF = 搜索候选集大小（越大越精确但越慢）
        - vector::distance::knn() 复用索引已计算的距离（0=完全相同）
        """
        # EF 参数：max(配置值, 4*limit)，Oracle 建议动态调整
        ef_search = max(self._hnsw_ef_search, 4 * limit)

        # KNN 操作符 <|K,EF|> 要求字面整数；embedding 数组必须作为字面量嵌入
        # （SDK WebSocket 传递 $embedding 参数时 KNN 不生效 — 已验证）
        knn_op = f"<|{int(limit)},{int(ef_search)}|>"
        emb_literal = json.dumps(embedding)
        q = (  # nosec B608
            "SELECT id, content, metadata, type, tags, project_id, "  # nosec B608
            "vector::distance::knn() AS distance "
            "FROM memory "
            "WHERE tenant_id = $tenant_id "
            f"AND embedding {knn_op} {emb_literal} "  # nosec B608
            "ORDER BY distance ASC"
        )
        result = await self._db.query(q, {"tenant_id": tenant_id})

        # cosine_distance = 1 - cosine_similarity
        # 用户传入 threshold 为相似度 (0.7)，转换为最大距离 (0.3)
        max_distance = 1.0 - threshold
        return self._format_vector_results(result, max_distance)

    async def _search_by_keyword(self, query_text: str, limit: int, tenant_id: str) -> list[dict[str, Any]]:
        """BM25 全文搜索

        使用 @1@ 全文搜索操作符 + search::score(1) 获取 BM25 评分。
        注意: SDK WebSocket 参数化传递 $query 对 @1@ 算子无效（同 KNN 发现 #4），
        必须使用字面量嵌入查询文本。查询文本经 _sanitize_query() 清洗防止注入。
        """
        safe_query = self._sanitize_query(query_text)
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
        return self._format_keyword_results(result)

    async def _hybrid_search(
        self,
        query_text: str,
        embedding: list[float],
        limit: int,
        threshold: float,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        """RRF 混合搜索：并行执行向量+关键词搜索，然后用 RRF 算法融合"""
        # 并行执行两种搜索，各取 2*limit 候选以保证融合后有足够结果
        tasks = [
            self._search_by_vector(embedding, limit * 2, threshold, tenant_id),
            self._search_by_keyword(query_text, limit * 2, tenant_id),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        vector_results: list[dict[str, Any]] = [] if isinstance(results[0], BaseException) else results[0]
        keyword_results: list[dict[str, Any]] = [] if isinstance(results[1], BaseException) else results[1]

        # RRF 融合
        merged = self._rrf_merge(vector_results, keyword_results)
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

    def _format_keyword_results(self, db_result: Any) -> list[dict[str, Any]]:
        """格式化关键词搜索结果"""
        raw_items = self._extract_records(db_result)
        results: list[dict[str, Any]] = []
        for item in raw_items:
            score = item.get("score", 0.0)
            results.append(self._build_result_item(item, score=float(score)))
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
        """创建两条记忆之间的图关系

        使用 SurrealDB RELATE 语句创建有向边。
        edge 表为 memory_relation，类型限制为 IN memory OUT memory。

        Args:
            from_id: 源记忆 ID（如 "memory:abc123"）
            to_id: 目标记忆 ID（如 "memory:def456"）
            relationship_type: 关系类型 (related/follow_up/elaboration/contradiction/reference/derived_from)
            weight: 关系权重 0.0-1.0
            tenant_id: 租户 ID
            description: 关系描述（可选）
            metadata: 扩展元数据（可选）
        """
        effective_tenant_id = tenant_id or self._default_tenant_id

        # 验证参数
        valid_types = {"related", "follow_up", "elaboration", "contradiction", "reference", "derived_from"}
        if relationship_type not in valid_types:
            raise ValidationError(f"Invalid relationship_type: {relationship_type}. Must be one of {valid_types}")
        if not 0.0 <= weight <= 1.0:
            raise ValidationError(f"weight must be between 0.0 and 1.0, got {weight}")

        # 规范化 ID 格式
        from_ref = self._normalize_memory_id(from_id)
        to_ref = self._normalize_memory_id(to_id)

        try:
            # RELATE 使用字面量（同 KNN/BM25 的 SDK workaround）
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

            # 提取创建的关系 ID
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
            raise DatabaseError(f"Failed to create relation: {e!s}") from e

    async def get_relations(
        self,
        memory_id: str,
        direction: str = "both",
        relationship_type: str | None = None,
        tenant_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """查询记忆的关联关系

        Args:
            memory_id: 记忆 ID
            direction: 查询方向 (outgoing/incoming/both)
            relationship_type: 按关系类型过滤（可选）
            tenant_id: 租户 ID
            limit: 返回数量限制
        """
        effective_tenant_id = tenant_id or self._default_tenant_id
        mem_ref = self._normalize_memory_id(memory_id)

        if direction not in ("outgoing", "incoming", "both"):
            raise ValidationError(f"Invalid direction: {direction}. Must be outgoing/incoming/both")

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

            return results
        except Exception as e:
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
        """图遍历：获取关联的记忆内容（支持多层深度）

        Args:
            memory_id: 起始记忆 ID
            depth: 遍历深度 (1-3，默认 1)
            relationship_type: 按关系类型过滤
            tenant_id: 租户 ID
            limit: 返回数量限制
        """
        effective_tenant_id = tenant_id or self._default_tenant_id
        mem_ref = self._normalize_memory_id(memory_id)
        depth = max(1, min(depth, 3))  # 限制 1-3 层防止性能问题

        try:
            # 构建图遍历路径表达式
            # depth=1: ->memory_relation->memory
            # depth=2: ->memory_relation->memory->memory_relation->memory
            path = "->memory_relation->memory" * depth

            q = (
                f"SELECT {path}.* AS related "  # nosec B608
                f"FROM {mem_ref}"  # nosec B608
            )
            result = await self._db.query(q)

            # 提取并去重关联记忆
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
                    # 租户隔离检查
                    if mem.get("tenant_id") != effective_tenant_id:
                        continue
                    seen.add(mem_id)
                    memories.append(self._build_result_item(mem, score=0.0))
                    if len(memories) >= limit:
                        break
                if len(memories) >= limit:
                    break

            return memories
        except Exception as e:
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
