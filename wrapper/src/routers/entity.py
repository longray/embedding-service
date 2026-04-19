"""Entity CRUD 端点 - 知识实体管理"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .. import state
from ..utils.exceptions import ValidationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["entities"])


class EntityCreateRequest(BaseModel):
    """创建 Entity 请求"""

    type: str = Field(
        ...,
        description="Entity 类型",
        examples=["memory", "backlog", "wiki", "code"],
    )
    tenant_id: str = Field(default="default", description="租户ID")

    
    abstract: str = Field(..., description="摘要 (L0, ≤100字符)")
    overview: dict[str, Any] = Field(default_factory=dict, description="概览 (L1)")
    atoms: list[str] = Field(default_factory=list, description="Atom ID 列表 (L2)")

    
    title: str | None = Field(default=None, description="标题")
    aliases: list[str] = Field(default_factory=list, description="别名")

    
    priority: str | None = Field(default=None, description="优先级", examples=["P0", "P1", "P2", "P3"])
    status: str | None = Field(default=None, description="状态", examples=["backlog", "todo", "in_progress", "done"])
    scene: str | None = Field(default=None, description="场景")
    estimated_hours: float | None = Field(default=None, description="预估工时")
    actual_hours: float | None = Field(default=None, description="实际工时")

    
    file_path: str | None = Field(default=None, description="文件路径")
    language: str | None = Field(default=None, description="编程语言")
    quality_score: dict[str, Any] | None = Field(default=None, description="质量评分")
    complexity_metrics: dict[str, Any] | None = Field(default=None, description="复杂度指标")

    
    tags: list[str] = Field(default_factory=list, description="标签")
    project: str | None = Field(default=None, description="项目ID")
    created_by: str = Field(default="system", description="创建者")


class EntityUpdateRequest(BaseModel):
    """更新 Entity 请求"""

    abstract: str | None = Field(default=None, description="摘要")
    overview: dict[str, Any] | None = Field(default=None, description="概览")
    atoms: list[str] | None = Field(default=None, description="Atom ID 列表")
    tags: list[str] | None = Field(default=None, description="标签")
    status: str | None = Field(default=None, description="状态")
    priority: str | None = Field(default=None, description="优先级")


class EntityResponse(BaseModel):
    """Entity 响应"""

    id: str = Field(..., description="Entity ID")
    type: str = Field(..., description="Entity 类型")
    tenant_id: str = Field(..., description="租户ID")

    
    abstract: str = Field(..., description="摘要")
    overview: dict[str, Any] = Field(default_factory=dict, description="概览")
    atoms: list[str] = Field(default_factory=list, description="Atom ID 列表")

    
    title: str | None = Field(default=None, description="标题")
    aliases: list[str] = Field(default_factory=list, description="别名")

    
    priority: str | None = Field(default=None, description="优先级")
    status: str | None = Field(default=None, description="状态")
    scene: str | None = Field(default=None, description="场景")
    estimated_hours: float | None = Field(default=None, description="预估工时")
    actual_hours: float | None = Field(default=None, description="实际工时")

    
    file_path: str | None = Field(default=None, description="文件路径")
    language: str | None = Field(default=None, description="编程语言")
    quality_score: dict[str, Any] | None = Field(default=None, description="质量评分")
    complexity_metrics: dict[str, Any] | None = Field(default=None, description="复杂度指标")

    
    tags: list[str] = Field(default_factory=list, description="标签")
    project: str | None = Field(default=None, description="项目ID")
    created_by: str = Field(..., description="创建者")
    created_at: str | None = Field(default=None, description="创建时间")
    updated_at: str | None = Field(default=None, description="更新时间")


@router.post("/entities", response_model=EntityResponse)
async def create_entity(request: EntityCreateRequest):
    """
    创建 Entity

    Entity 是知识聚合单元，包含 L0/L1/L2 分层：
    - L0 (abstract): ≤100字符，用于列表展示
    - L1 (overview): 结构化数据，用于预览
    - L2 (atoms): Atom ID 列表，完整详情
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        db = state.memory_manager.db

        
        valid_types = ["memory", "backlog", "wiki", "code"]
        if request.type not in valid_types:
            raise ValidationError(f"无效的 Entity 类型: {request.type}. 必须是 {valid_types}")

        
        if request.atoms:
            from surrealdb.data.types.record_id import RecordID
            record_ids = []
            for atom_id in request.atoms:
                if ":" in atom_id:
                    parts = atom_id.split(":", 1)
                    record_ids.append(RecordID(parts[0], parts[1]))
                else:
                    record_ids.append(atom_id)
            
            logger.info("[Entity] Looking for atoms: %s, record_ids: %s", request.atoms, record_ids)
            
            atoms_check = await db.query(
                "SELECT id FROM atom WHERE id IN $atom_ids",
                {"atom_ids": record_ids}
            )
            
            logger.info("[Entity] atoms_check result: %s, type: %s", atoms_check, type(atoms_check))
            
            found_ids = set()
            if atoms_check:
                for record in atoms_check:
                    record_id = record["id"]
                    if hasattr(record_id, "table_name"):
                        found_ids.add(f"{record_id.table_name}:{record_id.id}")
                    else:
                        found_ids.add(str(record_id))
            
            logger.info("[Entity] found_ids: %s, request.atoms: %s", found_ids, set(request.atoms))
            
            missing = set(request.atoms) - found_ids
            if missing:
                raise ValidationError(f"Atoms 不存在: {missing}")

        
        entity_data = {
            "type": request.type,
            "tenant_id": request.tenant_id,
            "abstract": request.abstract,
            "overview": request.overview,
            "atoms": record_ids if request.atoms else [],
            "tags": request.tags,
            "project": request.project,
            "created_by": request.created_by,
        }

        
        if request.type == "wiki":
            entity_data.update({
                "title": request.title,
                "aliases": request.aliases,
            })
        elif request.type == "backlog":
            entity_data.update({
                "priority": request.priority,
                "status": request.status,
                "scene": request.scene,
                "estimated_hours": request.estimated_hours,
                "actual_hours": request.actual_hours,
            })
        elif request.type == "code":
            entity_data.update({
                "file_path": request.file_path,
                "language": request.language,
                "quality_score": request.quality_score,
                "complexity_metrics": request.complexity_metrics,
            })

        
        entity_data = {k: v for k, v in entity_data.items() if v is not None}

        
        # BL-B-100: 使用事务执行创建操作
        try:
            await db.query("BEGIN TRANSACTION")
            result = await db.create("entity", entity_data)

            if not result:
                await db.query("CANCEL TRANSACTION")
                raise HTTPException(status_code=500, detail="创建 Entity 失败")

            if isinstance(result, dict):
                record = result
            elif isinstance(result, list) and result:
                record = result[0]
                if isinstance(record, list) and record:
                    record = record[0]
            else:
                await db.query("CANCEL TRANSACTION")
                raise HTTPException(status_code=500, detail=f"创建 Entity 失败: 无效的响应格式 {type(result)}")

            raw_id = record.get("id") if isinstance(record, dict) else record
            if raw_id and not isinstance(raw_id, list) and hasattr(raw_id, "table_name"):
                record_id = f"{raw_id.table_name}:{raw_id.id}"
            else:
                record_id = str(raw_id)

            # Convert atoms back to string IDs for response
            response_data = entity_data.copy()
            if response_data.get("atoms"):
                response_data["atoms"] = request.atoms

            await db.query("COMMIT TRANSACTION")
            return EntityResponse(id=record_id, **response_data)

        except Exception:
            try:
                await db.query("CANCEL TRANSACTION")
            except Exception as cancel_error:
                logger.error("[Entity] 事务回滚失败: %s", cancel_error)
            raise

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except Exception as e:
        logger.error("[Entity] 创建失败: %s", e)
        raise HTTPException(status_code=500, detail=f"创建失败: {e!s}") from e


@router.get("/entities/{entity_id}")
async def get_entity(
    entity_id: str,
    level: int = Query(default=2, ge=0, le=2, description="返回层级: 0=abstract, 1=abstract+overview, 2=full"),
    tenant_id: str = Query(default="default"),
):
    """获取 Entity"""
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        db = state.memory_manager.db

        
        if level == 0:
            
            result = await db.query(
                "SELECT id, type, abstract, tenant_id, created_at FROM entity WHERE id = $entity_id AND tenant_id = $tenant_id",
                {"entity_id": entity_id, "tenant_id": tenant_id}
            )
        elif level == 1:
            
            result = await db.query(
                "SELECT id, type, abstract, overview, tags, project, created_at FROM entity WHERE id = $entity_id AND tenant_id = $tenant_id",
                {"entity_id": entity_id, "tenant_id": tenant_id}
            )
        else:
            
            result = await db.query(
                "SELECT * FROM entity WHERE id = $entity_id AND tenant_id = $tenant_id",
                {"entity_id": entity_id, "tenant_id": tenant_id}
            )

        if not result or len(result) == 0:
            raise HTTPException(status_code=404, detail="Entity 不存在")

        return result[0]

    except HTTPException:
        raise
    except Exception as e:
        logger.error("[Entity] 查询失败: %s", e)
        raise HTTPException(status_code=500, detail=f"查询失败: {e!s}") from e


@router.get("/entities")
async def list_entities(
    type: str | None = Query(default=None, description="Entity 类型过滤"),
    project: str | None = Query(default=None, description="项目过滤"),
    status: str | None = Query(default=None, description="状态过滤"),
    tenant_id: str = Query(default="default"),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """列出 Entities"""
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        db = state.memory_manager.db

        
        query = "SELECT id, type, abstract, tags, status, project, created_at FROM entity WHERE tenant_id = $tenant_id"
        params = {"tenant_id": tenant_id}

        if type:
            query += " AND type = $type"
            params["type"] = type
        if project:
            query += " AND project = $project"
            params["project"] = project
        if status:
            query += " AND status = $status"
            params["status"] = status

        query += f" LIMIT {limit} START {offset}"

        result = await db.query(query, params)
        return result or []

    except Exception as e:
        logger.error("[Entity] 列表查询失败: %s", e)
        raise HTTPException(status_code=500, detail=f"查询失败: {e!s}") from e


@router.put("/entities/{entity_id}")
async def update_entity(entity_id: str, request: EntityUpdateRequest):
    """更新 Entity"""
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        db = state.memory_manager.db

        
        check = await db.query(
            "SELECT id FROM entity WHERE id = $entity_id",
            {"entity_id": entity_id}
        )
        if not check or len(check) == 0:
            raise HTTPException(status_code=404, detail="Entity 不存在")

        
        if request.atoms:
            for atom_id in request.atoms:
                atom_check = await db.query(
                    "SELECT id FROM atom WHERE id = $atom_id",
                    {"atom_id": atom_id}
                )
                if not atom_check or len(atom_check) == 0:
                    raise ValidationError(f"Atom 不存在: {atom_id}")

        
        update_data = {}
        for field, value in request.model_dump(exclude_unset=True).items():
            if value is not None:
                update_data[field] = value

        if not update_data:
            raise HTTPException(status_code=400, detail="没有要更新的字段")

        update_data["updated_at"] = "time::now()"

        # BL-B-100: 使用事务执行更新操作
        try:
            await db.query("BEGIN TRANSACTION")
            result = await db.update(entity_id, update_data)

            if not result or len(result) == 0:
                await db.query("CANCEL TRANSACTION")
                raise HTTPException(status_code=500, detail="更新失败")

            await db.query("COMMIT TRANSACTION")
            return result[0]

        except Exception:
            try:
                await db.query("CANCEL TRANSACTION")
            except Exception as cancel_error:
                logger.error("[Entity] 事务回滚失败: %s", cancel_error)
            raise

    except HTTPException:
        raise
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except Exception as e:
        logger.error("[Entity] 更新失败: %s", e)
        raise HTTPException(status_code=500, detail=f"更新失败: {e!s}") from e


@router.delete("/entities/{entity_id}")
async def delete_entity(entity_id: str, tenant_id: str = Query(default="default")):
    """删除 Entity"""
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        db = state.memory_manager.db

        
        check = await db.query(
            "SELECT id FROM entity WHERE id = $entity_id AND tenant_id = $tenant_id",
            {"entity_id": entity_id, "tenant_id": tenant_id}
        )
        if not check or len(check) == 0:
            raise HTTPException(status_code=404, detail="Entity 不存在")

        # BL-B-100: 使用事务执行删除操作
        try:
            await db.query("BEGIN TRANSACTION")
            await db.delete(entity_id)
            await db.query("COMMIT TRANSACTION")
            return {"success": True, "message": "Entity 已删除"}

        except Exception:
            try:
                await db.query("CANCEL TRANSACTION")
            except Exception as cancel_error:
                logger.error("[Entity] 事务回滚失败: %s", cancel_error)
            raise

    except HTTPException:
        raise
    except Exception as e:
        logger.error("[Entity] 删除失败: %s", e)
        raise HTTPException(status_code=500, detail=f"删除失败: {e!s}") from e
