"""项目代码分析端点 (BL-CA-23/25)

提供项目级别的代码地图和统计信息查询。
"""

from fastapi import APIRouter, HTTPException

from .. import state

router = APIRouter(prefix="/api/v1", tags=["projects"])


@router.get("/projects/{project_id}/map")
async def get_project_map(project_id: str, tenant_id: str = "default"):
    """获取项目代码地图 (BL-CA-23)

    返回项目文件树、模块依赖、热点文件和统计信息。
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await state.memory_manager.get_project_map(project_id, tenant_id)

        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message", "获取项目地图失败"))

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取项目地图失败: {e}") from e


@router.get("/projects/{project_id}/stats")
async def get_project_stats(project_id: str, tenant_id: str = "default"):
    """获取项目代码统计信息 (BL-CA-25)

    按 project_id 聚合统计代码文件信息，包括：
    - 总文件数
    - 总函数数
    - 总类数
    - 平均复杂度
    - 最大复杂度
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await state.memory_manager.get_project_stats(project_id, tenant_id)

        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message", "获取项目统计失败"))

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取项目统计失败: {e}") from e
