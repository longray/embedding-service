"""搜索端点 - 记忆搜索 + v3.3 统一搜索"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .. import state
from ..models import MemorySearchRequest
from ..utils.exceptions import ValidationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["search"])


# ============================================================
# v3.3 统一搜索端点 - 跨 Entity + Atom 搜索
# ============================================================

VALID_SEARCH_MODES = frozenset(["vector", "keyword", "hybrid"])
VALID_SEARCH_SCOPES = frozenset(["all", "memory", "code", "backlog"])


class UnifiedSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="搜索查询")
    mode: str = Field(default="hybrid", description="搜索模式: vector | keyword | hybrid")
    scope: str = Field(default="all", description="搜索范围: all | memory | code | backlog")
    types: list[str] | None = Field(default=None, description="过滤 Entity 类型")
    atom_types: list[str] | None = Field(default=None, description="过滤 Atom 类型")
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
        entity_result = await state.memory_manager.search_memories(
            query=request.query,
            mode=request.mode,
            limit=request.limit,
            threshold=0.5,
            level=request.level,
            tenant_id=request.tenant_id,
        )

        results = []
        items = entity_result if isinstance(entity_result, list) else entity_result.get("results", [])

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
    """通过 SurrealDB 搜索 Atom"""
    try:
        conditions = ["tenant_id = $tenant_id"]
        params: dict[str, Any] = {"tenant_id": request.tenant_id}

        conditions.append("(content LIKE $query OR name LIKE $query)")
        params["query"] = f"%{request.query}%"

        if request.atom_types:
            conditions.append("type IN $atom_types")
            params["atom_types"] = request.atom_types

        where_clause = " AND ".join(conditions)
        # nosec B608: where_clause 由参数化条件构建，非用户直接输入
        query = f"SELECT * FROM atom WHERE {where_clause} LIMIT $limit"  # nosec B608
        params["limit"] = request.limit

        result = await db.query(query, params)
        raw_data = result or []

        results = []
        for record in raw_data:
            raw_id = record.get("id", "")
            if hasattr(raw_id, "table_name"):
                raw_id = f"{raw_id.table_name}:{raw_id.id}"

            results.append({
                "type": "atom",
                "local_id": record.get("local_id", record.get("source_id", "")),
                "atom_id": raw_id,
                "atom_type": record.get("type", ""),
                "name": record.get("name", ""),
                "entity_id": record.get("entity_id", ""),
                "score": 0.5,
            })
        return results
    except Exception as e:
        logger.warning("[UnifiedSearch] Atom 搜索失败，跳过: %s", e)
        return []


def _should_search_entities(scope: str, types: list[str] | None) -> bool:
    """判断是否需要搜索 Entity"""
    if types and "atom" in types and "memory" not in types:
        return False
    return scope in ("all", "memory", "code", "backlog")


def _should_search_atoms(scope: str, types: list[str] | None) -> bool:
    """判断是否需要搜索 Atom"""
    if types and "memory" in types and "atom" not in types:
        return False
    return scope in ("all", "memory", "code", "backlog")


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
