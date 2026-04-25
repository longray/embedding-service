"""图关系端点"""

from fastapi import APIRouter, HTTPException

from .. import state
from ..models import (
    CallRelationBatchRequest,
    GraphTraversalRequest,
    RelationCreateRequest,
    RelationQueryRequest,
)
from ..utils.exceptions import ValidationError

router = APIRouter(prefix="/api/v1", tags=["relations"])


@router.post("/memories/relations")
async def create_relation(request: RelationCreateRequest):
    """创建记忆间的图关系"""
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await state.memory_manager.create_relation(
            from_id=request.from_id,
            to_id=request.to_id,
            type=request.type,
            weight=request.weight,
            tenant_id=request.tenant_id,
            description=request.description,
        )
        return result
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建关系失败: {e!s}") from e


@router.post("/memories/{memory_id}/relations")
async def get_relations(memory_id: str, request: RelationQueryRequest):
    """查询记忆的关联关系"""
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await state.memory_manager.get_relations(
            memory_id=memory_id,
            direction=request.direction,
            type=request.type,
            tenant_id=request.tenant_id,
            limit=request.limit,
        )
        return {"relations": result, "total": len(result), "memory_id": memory_id}
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询关系失败: {e!s}") from e


@router.delete("/memories/relations/{relation_id}")
async def delete_relation(relation_id: str, tenant_id: str = "default"):
    """删除指定的关系"""
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        deleted = await state.memory_manager.delete_relation(
            relation_id=relation_id,
            tenant_id=tenant_id,
        )
        if not deleted:
            raise HTTPException(status_code=404, detail="关系不存在或无权删除")
        return {"deleted": True, "relation_id": relation_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除关系失败: {e!s}") from e


@router.post("/memories/{memory_id}/graph")
async def graph_traversal(memory_id: str, request: GraphTraversalRequest):
    """图遍历：获取关联的记忆内容"""
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await state.memory_manager.get_related_memories(
            memory_id=memory_id,
            depth=request.depth,
            type=request.type,
            tenant_id=request.tenant_id,
            limit=request.limit,
        )
        return {"memories": result, "total": len(result), "source": memory_id, "depth": request.depth}
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图遍历失败: {e!s}") from e


@router.post("/calls/batch")
async def create_call_relations_batch(request: CallRelationBatchRequest):
    """批量创建调用关系 (BL-CA-20)

    批量创建函数调用关系，用于代码分析 v1.4 调用关系追踪。

    **约束条件**:
    - 最大批量: 100 条/批次
    - callee_memory_id 不存在时返回错误列表，跳过不存在的调用
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        # 转换请求数据
        calls_data = []
        for call in request.calls:
            calls_data.append(
                {
                    "caller_memory_id": call.caller_memory_id,
                    "callee_memory_id": call.callee_memory_id,
                    "line": call.line,
                    "column": call.column,
                    "file_path": call.file_path,
                }
            )

        result = await state.memory_manager.create_call_relations_batch(
            calls=calls_data,
            tenant_id=request.tenant_id,
        )

        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail="批量创建调用关系失败")

        return result
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量创建调用关系失败: {e!s}") from e


@router.get("/memories/{memory_id}/references")
async def get_call_references(memory_id: str, tenant_id: str = "default", limit: int = 50):
    """查询谁调用了该符号 (BL-CA-21)

    查询所有调用该函数的代码位置。
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await state.memory_manager.get_call_references(
            memory_id=memory_id,
            tenant_id=tenant_id,
            limit=limit,
        )
        return result
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询引用失败: {e!s}") from e


@router.get("/memories/{memory_id}/dependencies")
async def get_call_dependencies(memory_id: str, tenant_id: str = "default", limit: int = 50):
    """查询该符号依赖了谁 (BL-CA-22)

    查询该函数调用了哪些其他函数。
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await state.memory_manager.get_call_dependencies(
            memory_id=memory_id,
            tenant_id=tenant_id,
            limit=limit,
        )
        return result
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询依赖失败: {e!s}") from e
