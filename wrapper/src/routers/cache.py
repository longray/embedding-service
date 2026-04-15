"""缓存管理端点

提供缓存管理功能：
- 缓存统计
- 缓存清理
- 缓存预热
"""

from fastapi import APIRouter, HTTPException

from .. import state

router = APIRouter(prefix="/api/v1/cache", tags=["cache"])


@router.get("/stats")
async def get_cache_stats():
    """获取缓存统计信息

    Returns:
        缓存统计信息，包括命中率、大小、键数量等
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await state.memory_manager.get_cache_stats()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取缓存统计失败: {e}") from e


@router.post("/clear")
async def clear_cache():
    """清理缓存

    清除所有缓存数据。

    Returns:
        清理结果
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await state.memory_manager.clear_embedding_cache()
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清除缓存失败: {e}") from e


@router.post("/warmup")
async def warmup_cache(tenant_id: str = "default", limit: int = 100):
    """缓存预热

    预加载最近记忆的嵌入向量到缓存中。

    Args:
        tenant_id: 租户 ID
        limit: 预加载数量限制

    Returns:
        预热结果
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await state.memory_manager.warmup_embedding_cache(tenant_id, limit)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预热缓存失败: {e}") from e
