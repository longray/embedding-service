"""预取功能路由 - 相关记忆和热门记忆预取端点"""

from fastapi import APIRouter, HTTPException

from .. import state

router = APIRouter(prefix="/api/v1/prefetch", tags=["prefetch"])


@router.post("/related")
async def prefetch_related(
    memory_id: str,
    tenant_id: str = "default",
    depth: int = 1,
    limit: int = 10,
):
    """预取相关记忆

    基于关系图遍历，预取与给定记忆相关的其他记忆。

    Args:
        memory_id: 起始记忆 ID
        tenant_id: 租户 ID
        depth: 遍历深度（1-3），默认 1
        limit: 返回数量限制，默认 10

    Returns:
        相关记忆列表
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await state.memory_manager.prefetch_related_memories(
            memory_id=memory_id,
            tenant_id=tenant_id,
            depth=depth,
            limit=limit,
        )

        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message", "预取失败"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预取失败: {e}") from e


@router.post("/popular")
async def prefetch_popular(
    tenant_id: str = "default",
    top_n: int = 20,
):
    """预取热门记忆

    基于访问统计和最近活跃度，预取热门记忆。

    Args:
        tenant_id: 租户 ID
        top_n: 返回数量，默认 20

    Returns:
        热门记忆列表
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await state.memory_manager.prefetch_popular_queries(
            tenant_id=tenant_id,
            top_n=top_n,
        )

        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message", "预取失败"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预取失败: {e}") from e
