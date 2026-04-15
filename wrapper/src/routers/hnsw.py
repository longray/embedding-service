"""HNSW 索引管理端点

提供 HNSW（Hierarchical Navigable Small World）向量索引的管理功能：
- 索引统计
- 参数优化
- 索引重建
"""

from fastapi import APIRouter, HTTPException

from .. import state

router = APIRouter(prefix="/api/v1/hnsw", tags=["hnsw"])


@router.get("/stats")
async def get_hnsw_stats(tenant_id: str = "default"):
    """获取 HNSW 索引统计信息

    Args:
        tenant_id: 租户 ID

    Returns:
        索引统计信息，包括向量数量、维度、索引大小等
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await state.memory_manager.get_memory_stats(tenant_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计失败: {e}") from e


@router.post("/optimize")
async def optimize_hnsw(tenant_id: str = "default"):
    """自动优化 HNSW 参数

    根据当前数据特征自动调整 HNSW 参数（efConstruction、M等），
    无需重建索引即可优化搜索性能。

    Args:
        tenant_id: 租户 ID

    Returns:
        优化结果，包含优化前后的参数对比
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await state.memory_manager.optimize_hnsw(tenant_id)
        return {
            "status": "success",
            "result": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"优化失败: {e}") from e


@router.post("/rebuild")
async def rebuild_hnsw(tenant_id: str = "default", force: bool = False):
    """重建 HNSW 索引

    使用最优参数重建 HNSW 索引。重建过程会：
    1. 分析当前数据特征
    2. 计算最优参数
    3. 创建新索引
    4. 原子切换索引

    Args:
        tenant_id: 租户 ID
        force: 是否强制重建（即使当前索引状态良好）

    Returns:
        重建结果，包含新索引的统计信息
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await state.memory_manager.rebuild_hnsw_index(tenant_id, force)
        return {
            "status": "success",
            "result": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重建失败: {e}") from e
