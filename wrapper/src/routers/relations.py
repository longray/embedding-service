"""图关系端点"""

from fastapi import APIRouter, HTTPException

from .. import state
from ..models import GraphTraversalRequest, RelationCreateRequest, RelationQueryRequest
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
            relationship_type=request.relationship_type,
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
            relationship_type=request.relationship_type,
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
            relationship_type=request.relationship_type,
            tenant_id=request.tenant_id,
            limit=request.limit,
        )
        return {"memories": result, "total": len(result), "source": memory_id, "depth": request.depth}
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图遍历失败: {e!s}") from e
