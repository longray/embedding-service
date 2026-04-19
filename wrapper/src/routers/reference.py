"""Reference (Graph Relation) 端点 - 图关系管理"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .. import state
from ..utils.exceptions import ValidationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["references"])


class ReferenceCreateRequest(BaseModel):
    """创建 Reference 请求"""

    from_id: str = Field(..., description="源 ID (atom:xxx 或 entity:xxx)")
    to_id: str = Field(..., description="目标 ID (atom:xxx 或 entity:xxx)")
    type: str = Field(
        ...,
        description="关系类型",
        examples=["calls", "imports", "depends_on", "implements", "wiki_link", "part_of"],
    )
    tenant_id: str = Field(default="default", description="租户ID")
    weight: float = Field(default=0.5, ge=0.0, le=1.0, description="关系权重")

    
    file_path: str | None = Field(default=None, description="文件路径")
    line: int | None = Field(default=None, description="行号")
    column: int | None = Field(default=None, description="列号")

    
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")


class ReferenceResponse(BaseModel):
    """Reference 响应"""

    id: str = Field(..., description="Reference ID")
    from_id: str = Field(..., description="源 ID")
    to_id: str = Field(..., description="目标 ID")
    type: str = Field(..., description="关系类型")
    tenant_id: str = Field(..., description="租户ID")
    weight: float = Field(default=0.5, description="权重")
    file_path: str | None = Field(default=None, description="文件路径")
    line: int | None = Field(default=None, description="行号")
    column: int | None = Field(default=None, description="列号")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")
    created_at: str | None = Field(default=None, description="创建时间")


@router.post("/references", response_model=ReferenceResponse)
async def create_reference(request: ReferenceCreateRequest):
    """
    创建关系 (Atom-Atom, Atom-Entity, Entity-Entity)

    使用 SurrealDB RELATE 语法创建原生图关系：
    RELATE atom:xxx->reference->atom:yyy
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        db = state.memory_manager.db

        
        from_check = await db.query(
            "SELECT id FROM atom WHERE id = $from_id",
            {"from_id": request.from_id}
        )
        if not from_check:
            from_check = await db.query(
                "SELECT id FROM entity WHERE id = $from_id",
                {"from_id": request.from_id}
            )
        if not from_check or len(from_check) == 0:
            raise ValidationError(f"from_id 不存在: {request.from_id}")

        
        to_check = await db.query(
            "SELECT id FROM atom WHERE id = $to_id",
            {"to_id": request.to_id}
        )
        if not to_check:
            to_check = await db.query(
                "SELECT id FROM entity WHERE id = $to_id",
                {"to_id": request.to_id}
            )
        if not to_check or len(to_check) == 0:
            raise ValidationError(f"to_id 不存在: {request.to_id}")

        
        query = """
        RELATE $from_id->reference->$to_id CONTENT {
            type: $type,
            tenant_id: $tenant_id,
            weight: $weight,
            file_path: $file_path,
            line: $line,
            column: $column,
            metadata: $metadata
        }
        """
        params = {
            "from_id": request.from_id,
            "to_id": request.to_id,
            "type": request.type,
            "tenant_id": request.tenant_id,
            "weight": request.weight,
            "file_path": request.file_path,
            "line": request.line,
            "column": request.column,
            "metadata": request.metadata,
        }

        # BL-B-100: 使用事务执行创建操作
        try:
            await db.query("BEGIN TRANSACTION")
            result = await db.query(query, params)

            if not result:
                await db.query("CANCEL TRANSACTION")
                raise HTTPException(status_code=500, detail="创建关系失败")

            if isinstance(result, dict):
                record = result
            elif isinstance(result, list) and result:
                record = result[0]
                if isinstance(record, list) and record:
                    record = record[0]
            else:
                await db.query("CANCEL TRANSACTION")
                raise HTTPException(status_code=500, detail="创建关系失败: 无效的响应格式")

            raw_id = record.get("id") if isinstance(record, dict) else record
            if raw_id and not isinstance(raw_id, list) and hasattr(raw_id, "table_name"):
                record_id = f"{raw_id.table_name}:{raw_id.id}"
            else:
                record_id = str(raw_id)

            await db.query("COMMIT TRANSACTION")
            return ReferenceResponse(
                id=record_id,
                from_id=request.from_id,
                to_id=request.to_id,
                type=request.type,
                tenant_id=request.tenant_id,
                weight=request.weight,
                file_path=request.file_path,
                line=request.line,
                column=request.column,
                metadata=request.metadata,
            )

        except Exception:
            try:
                await db.query("CANCEL TRANSACTION")
            except Exception as cancel_error:
                logger.error("[Reference] 事务回滚失败: %s", cancel_error)
            raise

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except Exception as e:
        logger.error("[Reference] 创建失败: %s", e)
        raise HTTPException(status_code=500, detail=f"创建关系失败: {e!s}") from e


@router.get("/references")
async def query_references(
    from_id: str | None = Query(default=None, description="源 ID"),
    to_id: str | None = Query(default=None, description="目标 ID"),
    type: str | None = Query(default=None, description="关系类型"),
    tenant_id: str = Query(default="default"),
    limit: int = Query(default=50, ge=1, le=200),
):
    """
    查询关系

    支持图遍历查询：
    - from_id: 查询从该节点出发的关系
    - to_id: 查询指向该节点的关系
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        db = state.memory_manager.db

        if from_id:
            
            query = "SELECT * FROM $from_id->reference WHERE tenant_id = $tenant_id"
            params = {"from_id": from_id, "tenant_id": tenant_id}
            if type:
                query += " AND type = $type"
                params["type"] = type
            query += f" LIMIT {limit}"
            result = await db.query(query, params)
        elif to_id:
            
            query = "SELECT * FROM <-reference-$to_id WHERE tenant_id = $tenant_id"
            params = {"to_id": to_id, "tenant_id": tenant_id}
            if type:
                query += " AND type = $type"
                params["type"] = type
            query += f" LIMIT {limit}"
            result = await db.query(query, params)
        else:
            
            query = "SELECT * FROM reference WHERE tenant_id = $tenant_id"
            params = {"tenant_id": tenant_id}
            if type:
                query += " AND type = $type"
                params["type"] = type
            query += f" LIMIT {limit}"
            result = await db.query(query, params)

        return result or []

    except Exception as e:
        logger.error("[Reference] 查询失败: %s", e)
        raise HTTPException(status_code=500, detail=f"查询失败: {e!s}") from e


@router.delete("/references/{reference_id}")
async def delete_reference(reference_id: str, tenant_id: str = Query(default="default")):
    """删除关系"""
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        db = state.memory_manager.db

        
        check = await db.query(
            "SELECT id FROM reference WHERE id = $reference_id AND tenant_id = $tenant_id",
            {"reference_id": reference_id, "tenant_id": tenant_id}
        )
        if not check or len(check) == 0:
            raise HTTPException(status_code=404, detail="关系不存在")

        # BL-B-100: 使用事务执行删除操作
        try:
            await db.query("BEGIN TRANSACTION")
            await db.delete(reference_id)
            await db.query("COMMIT TRANSACTION")
            return {"success": True, "message": "关系已删除"}

        except Exception:
            try:
                await db.query("CANCEL TRANSACTION")
            except Exception as cancel_error:
                logger.error("[Reference] 事务回滚失败: %s", cancel_error)
            raise

    except HTTPException:
        raise
    except Exception as e:
        logger.error("[Reference] 删除失败: %s", e)
        raise HTTPException(status_code=500, detail=f"删除失败: {e!s}") from e
