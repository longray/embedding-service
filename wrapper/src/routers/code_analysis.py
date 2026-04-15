"""代码分析端点

提供代码分析功能：
- 分析记忆内容中的代码
- 返回代码复杂度、依赖关系、质量评分
"""

from fastapi import APIRouter, HTTPException

from .. import state

router = APIRouter(prefix="/api/v1/memories", tags=["code-analysis"])


@router.post("/{memory_id}/analyze/code")
async def analyze_memory_code(memory_id: str, tenant_id: str = "default"):
    """分析记忆内容中的代码

    对记忆内容进行代码分析，提取：
    - 代码复杂度
    - 函数和类定义
    - 依赖关系
    - 质量评分

    Args:
        memory_id: 记忆 ID
        tenant_id: 租户 ID

    Returns:
        代码分析结果
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await state.memory_manager.analyze_memory_code(memory_id, tenant_id)

        if not result:
            return {
                "status": "skipped",
                "message": "记忆不存在、内容为空或非代码内容",
                "memory_id": memory_id,
            }

        return {
            "status": "success",
            "memory_id": memory_id,
            "result": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"代码分析失败: {e}") from e
