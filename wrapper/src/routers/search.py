"""搜索端点 - 记忆搜索 + v3.3 统一搜索"""

import logging
import re
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .. import state
from ..models import MemorySearchRequest
from ..utils.exceptions import ValidationError

logger = logging.getLogger(__name__)


def _sanitize_query_for_bm25(query: str) -> str:
    """清洗查询字符串用于 BM25 搜索，移除 SQL 特殊字符"""
    cleaned = re.sub(
        r"[^\w\s\u4e00-\u9fff\u3400-\u4dbf\uff00-\uffef-]",
        "",
        query
    ).strip()
    return cleaned[:500]


def _get_hybrid_weights(query: str) -> tuple[float, float]:
    """根据查询语言动态调整 hybrid 权重
    
    中文: 向量 50%，关键词 50%（语义鸿沟大，关键词更重要）
    英文: 向量 60%，关键词 40%（向量语义匹配效果好）
    """
    # Extended CJK: Basic + Extension A + Compatibility
    has_chinese = bool(re.search(r"[\u4e00-\u9fff\u3400-\u4dbf\uff00-\uffef]", query))
    if has_chinese:
        return 0.5, 0.5
    return 0.6, 0.4


router = APIRouter(prefix="/api/v1", tags=["search"])


# ============================================================
# v3.3 统一搜索端点 - 跨 Entity + Atom 搜索
# ============================================================

VALID_SEARCH_MODES = frozenset(["vector", "keyword", "hybrid"])
VALID_SEARCH_SCOPES = frozenset(["all", "memory", "code", "backlog", "atom", "entity"])


class UnifiedSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="搜索查询")
    mode: str = Field(default="hybrid", description="搜索模式: vector | keyword | hybrid")
    scope: str = Field(default="all", description="搜索范围: all | memory | code | backlog")
    types: list[str] | None = Field(default=None, description="过滤 Entity 类型")
    atom_types: list[str] | None = Field(default=None, description="过滤 Atom 类型")
    max_level: int | None = Field(default=None, ge=1, le=6, description="最大标题层级过滤（仅返回 heading_level <= max_level 的 Atom）")
    limit: int = Field(default=20, ge=1, le=100, description="返回数量限制")
    level: int = Field(default=1, ge=0, le=2, description="返回层级")
    tenant_id: str = Field(default="default", description="租户ID")


class UnifiedSearchResponse(BaseModel):
    results: list[dict[str, Any]] = Field(default_factory=list, description="搜索结果")
    total: int = Field(default=0, description="总结果数")
    mode: str = Field(..., description="使用的搜索模式")
    query: str = Field(..., description="原始查询")


@router.post("/search", response_model=UnifiedSearchResponse)
async def unified_search(request: UnifiedSearchRequest):
    """
    统一搜索端点 - 跨 Entity（Meilisearch）和 Atom（SurrealDB）搜索

    支持模式：vector（向量）、keyword（关键词）、hybrid（混合）
    支持范围：all、memory、code、backlog
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    if request.mode not in VALID_SEARCH_MODES:
        raise HTTPException(status_code=400, detail=f"无效的搜索模式: {request.mode}")

    if request.scope not in VALID_SEARCH_SCOPES:
        raise HTTPException(status_code=400, detail=f"无效的搜索范围: {request.scope}")

    try:
        results: list[dict[str, Any]] = []
        db = state.memory_manager.db

        # --- Entity 搜索（通过现有 search_memories） ---
        entity_scope = _should_search_entities(request.scope, request.types)
        if entity_scope:
            entity_results = await _search_entities(request)
            results.extend(entity_results)

        # --- Atom 搜索（SurrealDB） ---
        atom_scope = _should_search_atoms(request.scope, request.types)
        if atom_scope:
            atom_results = await _search_atoms(db, request)
            results.extend(atom_results)

        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        results = results[:request.limit]

        return UnifiedSearchResponse(
            results=results,
            total=len(results),
            mode=request.mode,
            query=request.query,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("[UnifiedSearch] 搜索失败: %s", e)
        raise HTTPException(status_code=500, detail=f"搜索失败: {e!s}") from e


async def _search_entities(request: UnifiedSearchRequest) -> list[dict[str, Any]]:
    """通过现有 search_memories 搜索 Entity"""
    assert state.memory_manager is not None
    try:
        # v3.3-opt: Atom scope uses lower threshold (0.1) because atom content is shorter
        threshold = 0.1 if request.scope == "atom" else 0.5
        
        entity_result = await state.memory_manager.search_memories(
            query=request.query,
            mode=request.mode,
            limit=request.limit,
            threshold=threshold,
            level=request.level,
            tenant_id=request.tenant_id,
        )

        results = []
        if isinstance(entity_result, list):
            items = entity_result
        elif isinstance(entity_result, dict):
            items = entity_result.get("results", [])
        else:
            items = []

        for item in items:
            if isinstance(item, dict):
                results.append({
                    "type": "entity",
                    "id": item.get("id", ""),
                    "entity_type": item.get("type", "memory"),
                    "abstract": item.get("abstract", ""),
                    "score": item.get("score", 0.0),
                })
        return results
    except Exception as e:
        logger.warning("[UnifiedSearch] Entity 搜索失败，跳过: %s", e)
        return []


async def _search_atoms(db: Any, request: UnifiedSearchRequest) -> list[dict[str, Any]]:
    """通过 SurrealDB 搜索 Atom - v3.3-opt: 支持向量+关键词混合搜索"""
    try:
        # v3.3-opt: 根据模式选择搜索策略
        if request.mode == "vector":
            return await _search_atoms_by_vector(db, request)
        elif request.mode == "hybrid":
            return await _search_atoms_hybrid(db, request)
        else:
            return await _search_atoms_by_keyword(db, request)
    except Exception as e:
        logger.warning("[UnifiedSearch] Atom 搜索失败，跳过: %s", e)
        return []


async def _search_atoms_by_keyword(db: Any, request: UnifiedSearchRequest) -> list[dict[str, Any]]:
    """Atom 关键词搜索 - Phase 3: 优先使用 Meilisearch，降级到 SurrealDB"""
    # Phase 3: 优先使用 Meilisearch
    if state.memory_manager and state.memory_manager._meili:
        try:
            logger.debug("[AtomSearch] 使用 Meilisearch 进行关键词搜索")
            return await _search_atoms_by_keyword_meili(request)
        except Exception as e:
            logger.warning("[AtomSearch] Meilisearch 搜索失败，降级到 SurrealDB: %s", e)
    else:
        logger.debug("[AtomSearch] Meilisearch 不可用，使用 SurrealDB")
    
    # 降级到 SurrealDB BM25 搜索
    conditions = ["tenant_id = $tenant_id"]
    params: dict[str, Any] = {"tenant_id": request.tenant_id}

    # 使用 BM25 @@ 操作符（索引序号 1=content, 2=name）
    # NOTE: @1@ = idx_atom_content_ft, @2@ = idx_atom_name_ft
    # 这些引用基于 FULLTEXT 索引定义顺序，见 init_surrealdb.surql
    safe_query = _sanitize_query_for_bm25(request.query)
    if not safe_query:
        logger.warning("[AtomSearch] Query sanitized to empty, returning empty results")
        return []
    conditions.append(f"(content @1@ '{safe_query}' OR name @2@ '{safe_query}')")
    params["query"] = request.query

    if request.atom_types:
        conditions.append("type IN $atom_types")
        params["atom_types"] = request.atom_types

    if request.max_level is not None:
        conditions.append("(heading_level IS NONE OR heading_level <= $max_level)")
        params["max_level"] = request.max_level

    where_clause = " AND ".join(conditions)
    # 使用 search::score 获取 BM25 相关性评分
    # 使用 math::sum 处理 NONE 值（当记录只匹配一个字段时）
    query = (  # nosec B608 - where_clause built from parameterized conditions
        f"SELECT *, math::sum(search::score(1), search::score(2)) AS score "
        f"FROM atom WHERE {where_clause} "
        f"ORDER BY score DESC LIMIT $limit"
    )
    params["limit"] = request.limit

    result = await db.query(query, params)
    raw_data = result or []

    seen_local_ids = set()
    results = []
    for record in raw_data:
        raw_id = record.get("id", "")
        if hasattr(raw_id, "table_name"):
            raw_id = f"{raw_id.table_name}:{raw_id.id}"

        local_id = record.get("local_id", record.get("source_id", ""))
        if local_id and local_id in seen_local_ids:
            continue
        if local_id:
            seen_local_ids.add(local_id)

        results.append({
            "type": "atom",
            "local_id": local_id,
            "atom_id": raw_id,
            "atom_type": record.get("type", ""),
            "name": record.get("name", ""),
            "content": record.get("content", ""),
            "heading_level": record.get("heading_level"),
            "parent_id": record.get("parent_id"),
            "order": record.get("order"),
            "tags": record.get("tags", []),
            "entity_id": record.get("entity_id", ""),
            "score": record.get("score") or 0.0,
        })
    return results


def _preprocess_chinese_query(query: str) -> str:
    """预处理中文查询，保留词组完整性并添加单字备选
    
    示例:
    - "Promise 错误" → "Promise 错误 错 误"
    - "错误处理" → "错误处理 错 误 处 理"
    - "异步 并发 控制" → "异步 异 步 并发 并 发 控制 控 制"
    """
    # 按空格分割（保留用户意图的词边界）
    parts = query.split()
    processed_parts = []
    
    for part in parts:
        # 检查是否包含中文
        if re.search(r'[\u4e00-\u9fff]', part):
            # 中文部分：保留原词，同时添加单字作为备选
            chinese_chars = re.findall(r'[\u4e00-\u9fff]', part)
            if len(chinese_chars) > 1:
                # 多字词：保留原词 + 单字
                processed_parts.append(part)
                processed_parts.append(' '.join(chinese_chars))
            else:
                # 单字：直接保留
                processed_parts.append(part)
        else:
            # 英文/数字部分，原样保留
            processed_parts.append(part)
    
    return ' '.join(processed_parts)


async def _search_atoms_by_keyword_meili(request: UnifiedSearchRequest) -> list[dict[str, Any]]:
    """Atom 关键词搜索 - 使用 Meilisearch（支持 CJK 分词）"""
    meili = state.memory_manager._meili
    
    # 构建 filter 表达式
    filter_parts = [f"tenant_id = '{request.tenant_id}'", "doc_type = 'atom'"]
    
    if request.atom_types:
        # Meilisearch IN 语法: atom_type IN ['note', 'section']
        # 转义单引号防止 filter 表达式被破坏
        type_list = ", ".join([f"'{t.replace(chr(39), chr(39)+chr(39))}'" for t in request.atom_types])
        filter_parts.append(f"atom_type IN [{type_list}]")
    
    if request.max_level is not None:
        # 包含 heading_level 为 null 的文档
        filter_parts.append(f"(heading_level <= {request.max_level} OR heading_level IS NULL)")
    
    filter_expr = " AND ".join(filter_parts)
    
    # 预处理中文查询（将中文字符用空格连接，便于 ngram 匹配）
    processed_query = _preprocess_chinese_query(request.query)
    logger.debug("[AtomSearch] 原始查询: %r, 预处理后: %r", request.query, processed_query)
    
    # 执行 Meilisearch 搜索
    result = await meili.search(
        query=processed_query,
        filter_expr=filter_expr,
        limit=request.limit,
        show_ranking_score=True,
    )
    
    # 格式化结果
    return _format_atom_meili_results(result)


def _format_atom_meili_results(meili_result: dict[str, Any]) -> list[dict[str, Any]]:
    """格式化 Meilisearch Atom 搜索结果"""
    results = []
    for hit in meili_result.get("hits", []):
        # ID 已由 meili_client.search() 转换 (atom_xxx -> atom:xxx)
        surreal_id = hit.get("id", "")
        
        results.append({
            "type": "atom",
            "local_id": hit.get("local_id"),
            "atom_id": surreal_id,
            "atom_type": hit.get("atom_type", "note"),
            "name": hit.get("name", ""),
            "content": hit.get("content", ""),
            "heading_level": hit.get("heading_level"),
            "parent_id": hit.get("parent_id"),
            "order": hit.get("order"),
            "tags": hit.get("tags", []),
            "entity_id": hit.get("entity_id", ""),
            "score": round(float(hit.get("_rankingScore", 0.0)), 6),
        })
    return results


async def _search_atoms_by_vector(db: Any, request: UnifiedSearchRequest) -> list[dict[str, Any]]:
    """Atom 向量搜索 - 使用 SurrealDB HNSW 索引"""
    assert state.memory_manager is not None
    
    # Get query embedding
    embeddings = await state.memory_manager._get_embeddings([request.query])
    query_embedding = embeddings[0]
    
    # Build filter conditions
    filters = ["tenant_id = $tenant_id"]
    params: dict[str, Any] = {"tenant_id": request.tenant_id}
    
    if request.atom_types:
        filters.append("type IN $atom_types")
        params["atom_types"] = request.atom_types
    
    if request.max_level is not None:
        filters.append("(heading_level IS NONE OR heading_level <= $max_level)")
        params["max_level"] = request.max_level
    
    where_clause = " AND ".join(filters) if filters else "TRUE"
    
    # nosec B608: where_clause 由参数化条件构建，非用户直接输入
    sql = f"""
        SELECT *, vector::similarity::cosine(embedding, $embedding) AS score
        FROM atom
        WHERE {where_clause}
        AND embedding IS NOT NONE
        ORDER BY score DESC
        LIMIT $limit
    """  # nosec B608
    params["embedding"] = query_embedding
    params["limit"] = request.limit
    
    result = await db.query(sql, params)
    raw_data = result or []
    
    raw_data_sorted = sorted(raw_data, key=lambda x: x.get("score", 0), reverse=True)
    
    seen_local_ids = set()
    results = []
    for record in raw_data_sorted:
        raw_id = record.get("id", "")
        if hasattr(raw_id, "table_name"):
            raw_id = f"{raw_id.table_name}:{raw_id.id}"
        
        score = record.get("score", 0.0)
        if score >= 0.1:
            local_id = record.get("local_id", record.get("source_id", ""))
            if local_id and local_id in seen_local_ids:
                continue
            if local_id:
                seen_local_ids.add(local_id)
            
            results.append({
                "type": "atom",
                "local_id": local_id,
                "atom_id": raw_id,
                "atom_type": record.get("type", ""),
                "name": record.get("name", ""),
                "content": record.get("content", ""),
                "heading_level": record.get("heading_level"),
                "parent_id": record.get("parent_id"),
                "order": record.get("order"),
                "tags": record.get("tags", []),
                "entity_id": record.get("entity_id", ""),
                "score": score,
            })
    return results


async def _search_atoms_hybrid(db: Any, request: UnifiedSearchRequest) -> list[dict[str, Any]]:
    """Atom 混合搜索 - 向量 + 关键词 RRF 融合"""
    # Get both vector and keyword results
    vector_results = await _search_atoms_by_vector(db, request)
    keyword_results = await _search_atoms_by_keyword(db, request)
    
    vector_weight, keyword_weight = _get_hybrid_weights(request.query)
    k = 60  # RRF smoothing constant
    
    # Merge using RRF
    scores: dict[str, float] = {}
    doc_data: dict[str, dict] = {}
    
    # Add vector contributions
    for rank, item in enumerate(vector_results):
        doc_id = item["atom_id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + vector_weight / (k + rank + 1)
        doc_data[doc_id] = item
    
    # Add keyword contributions
    for rank, item in enumerate(keyword_results):
        doc_id = item["atom_id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + keyword_weight / (k + rank + 1)
        if doc_id not in doc_data:
            doc_data[doc_id] = item
    
    # Sort by RRF score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    results = []
    for doc_id in sorted_ids[:request.limit]:
        item = doc_data[doc_id].copy()
        item["score"] = scores[doc_id]
        results.append(item)
    
    return results


def _should_search_entities(scope: str, types: list[str] | None) -> bool:
    """判断是否需要搜索 Entity"""
    if types and "atom" in types and "memory" not in types:
        return False
    return scope in ("all", "memory", "code", "backlog", "entity")


def _should_search_atoms(scope: str, types: list[str] | None) -> bool:
    """判断是否需要搜索 Atom"""
    if types and "memory" in types and "atom" not in types:
        return False
    return scope in ("all", "memory", "code", "backlog", "atom")


# ============================================================
# 现有记忆搜索端点（保持不变）
# ============================================================


@router.post("/memories/search")
async def search_memories(request: MemorySearchRequest):
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        filter_expr = None
        if request.code_filter:
            filter_parts = []
            if "language" in request.code_filter:
                filter_parts.append(f'code_language = "{request.code_filter["language"]}"')
            if "min_complexity" in request.code_filter:
                filter_parts.append(f"code_complexity >= {request.code_filter['min_complexity']}")
            if "max_complexity" in request.code_filter:
                filter_parts.append(f"code_complexity <= {request.code_filter['max_complexity']}")
            if "min_function_count" in request.code_filter:
                filter_parts.append(f"code_function_count >= {request.code_filter['min_function_count']}")
            if "max_function_count" in request.code_filter:
                filter_parts.append(f"code_function_count <= {request.code_filter['max_function_count']}")
            if "min_class_count" in request.code_filter:
                filter_parts.append(f"code_class_count >= {request.code_filter['min_class_count']}")
            if "max_class_count" in request.code_filter:
                filter_parts.append(f"code_class_count <= {request.code_filter['max_class_count']}")
            if "has_exports" in request.code_filter:
                filter_parts.append(f"code_has_exports = {request.code_filter['has_exports']}")
            if "analyzer" in request.code_filter:
                filter_parts.append(f'code_analyzer = "{request.code_filter["analyzer"]}"')
            if filter_parts:
                filter_expr = " AND ".join(filter_parts)

        result = await state.memory_manager.search_memories(
            query=request.query,
            mode=request.mode,
            limit=request.limit,
            threshold=request.threshold,
            level=request.level,
            tenant_id=request.tenant_id,
            filters=filter_expr,
        )
        return result
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {e!s}") from e
