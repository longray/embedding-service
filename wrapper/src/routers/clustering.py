"""记忆聚类路由 - Leiden 算法聚类端点"""

from fastapi import APIRouter, HTTPException

from .. import state

router = APIRouter(prefix="/api/v1/memories", tags=["clustering"])


@router.post("/cluster/leiden")
async def cluster_memories_leiden(
    tenant_id: str = "default",
    content_threshold: float = 0.75,
    max_clusters: int = 20,
):
    """使用 Leiden 算法对记忆进行聚类分析

    基于向量相似度对记忆进行聚类，发现语义相关的记忆组。

    Args:
        tenant_id: 租户 ID
        content_threshold: 内容相似度阈值（0-1），默认 0.75
        max_clusters: 最大聚类数量，默认 20

    Returns:
        聚类结果，包含簇列表、成员和中心点
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await state.memory_manager.cluster_memories_leiden(
            tenant_id=tenant_id,
            content_threshold=content_threshold,
            max_clusters=max_clusters,
        )

        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message", "聚类分析失败"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"聚类分析失败: {e}") from e
