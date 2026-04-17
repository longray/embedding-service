"""符号查询路由 (BL-B-83)

GET /api/v1/symbols/search — 按符号名查找定义位置、类型过滤、前缀匹配。
"""

from fastapi import APIRouter, HTTPException, Query

from .. import state
from ..services.symbol_service import VALID_SYMBOL_TYPES

router = APIRouter(prefix="/api/v1", tags=["symbols"])


@router.get("/symbols/search", summary="符号查询")
async def search_symbols(
    query: str = Query(..., min_length=1, description="符号名称"),
    type: str | None = Query(None, description=f"符号类型: {', '.join(sorted(VALID_SYMBOL_TYPES))}"),
    project_id: str | None = Query(None, description="项目ID过滤"),
    tenant_id: str = Query("default", description="租户ID"),
    fuzzy: bool = Query(False, description="前缀模糊匹配"),
    limit: int = Query(20, ge=1, le=100, description="返回数量限制"),
):
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    from ..services.symbol_service import SymbolService

    service = SymbolService(db=state.memory_manager.db)

    try:
        result = await service.search(
            query=query,
            tenant_id=tenant_id,
            symbol_type=type,
            project_id=project_id,
            fuzzy=fuzzy,
            limit=limit,
        )
        return result.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"符号查询失败: {e!s}") from e
