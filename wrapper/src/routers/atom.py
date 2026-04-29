"""Atom CRUD 端点 - 原子级知识单元管理"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from surrealdb.data.types.record_id import RecordID

from .. import state
from ..utils.db_helpers import extract_record_id, parse_surrealdb_result
from ..utils.exceptions import ValidationError
from ..utils.transaction import transaction

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["atoms"])

# 模块级常量：Atom 有效类型
# v3.3: 新增 chapter, section 支持知识文档场景
ATOM_VALID_TYPES = frozenset([
    "function", "class", "interface", "import",
    "goal", "scope", "task", "note",
    "chapter", "section",
])





class AtomCreateRequest(BaseModel):
    """创建 Atom 请求"""

    type: str = Field(
        ...,
        description="Atom 类型",
        examples=["function", "class", "interface", "import", "goal", "scope", "task", "note", "chapter", "section"],
    )
    content: str = Field(..., description="内容（函数源码、任务描述等）")
    tenant_id: str = Field(default="default", description="租户ID")

    
    name: str | None = Field(default=None, description="函数/类名")
    signature: str | None = Field(default=None, description="函数签名")
    params: list[dict[str, Any]] = Field(default_factory=list, description="参数列表")
    return_type: str | None = Field(default=None, description="返回类型")
    is_exported: bool | None = Field(default=None, description="是否导出")
    is_async: bool | None = Field(default=None, description="是否异步")
    complexity: int | None = Field(default=None, description="复杂度")
    max_nesting_depth: int | None = Field(default=None, description="最大嵌套深度")
    docstring: dict[str, Any] | None = Field(default=None, description="文档字符串")
    start_line: int | None = Field(default=None, description="起始行号")
    end_line: int | None = Field(default=None, description="结束行号")

    
    status: str | None = Field(
        default=None, description="状态", examples=["pending", "done", "blocked"]
    )

    
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")
    project: str | None = Field(default=None, description="项目ID")

    # v3.3 层级化知识图谱字段
    tags: list[str] = Field(default_factory=list, description="标签列表")
    heading_level: int | None = Field(default=None, ge=1, le=6, description="标题层级 1-6")
    parent_id: str | None = Field(default=None, description="父 Atom ID（树结构）")
    order: str | None = Field(default=None, description="分数索引（如 a0, aV）")
    aliases: list[str] = Field(default_factory=list, description="别名列表")
    entity_id: str | None = Field(default=None, description="所属 Entity ID")


class AtomUpdateRequest(BaseModel):
    """更新 Atom 请求"""

    content: str | None = Field(default=None, description="内容")
    name: str | None = Field(default=None, description="函数/类名")
    signature: str | None = Field(default=None, description="函数签名")
    params: list[dict[str, Any]] | None = Field(default=None, description="参数列表")
    return_type: str | None = Field(default=None, description="返回类型")
    is_exported: bool | None = Field(default=None, description="是否导出")
    is_async: bool | None = Field(default=None, description="是否异步")
    complexity: int | None = Field(default=None, description="复杂度")
    status: str | None = Field(default=None, description="状态")
    metadata: dict[str, Any] | None = Field(default=None, description="元数据")

    # v3.3 层级化知识图谱字段
    tags: list[str] | None = Field(default=None, description="标签列表")
    heading_level: int | None = Field(default=None, ge=1, le=6, description="标题层级 1-6")
    parent_id: str | None = Field(default=None, description="父 Atom ID")
    order: str | None = Field(default=None, description="分数索引")
    aliases: list[str] | None = Field(default=None, description="别名列表")
    entity_id: str | None = Field(default=None, description="所属 Entity ID")


class AtomResponse(BaseModel):
    """Atom 响应"""

    id: str = Field(..., description="Atom ID")
    type: str = Field(..., description="Atom 类型")
    content: str = Field(..., description="内容")
    tenant_id: str = Field(..., description="租户ID")

    
    name: str | None = Field(default=None, description="函数/类名")
    signature: str | None = Field(default=None, description="函数签名")
    params: list[dict[str, Any]] = Field(default_factory=list, description="参数列表")
    return_type: str | None = Field(default=None, description="返回类型")
    is_exported: bool | None = Field(default=None, description="是否导出")
    is_async: bool | None = Field(default=None, description="是否异步")
    complexity: int | None = Field(default=None, description="复杂度")
    max_nesting_depth: int | None = Field(default=None, description="最大嵌套深度")
    docstring: dict[str, Any] | None = Field(default=None, description="文档字符串")
    start_line: int | None = Field(default=None, description="起始行号")
    end_line: int | None = Field(default=None, description="结束行号")

    
    status: str | None = Field(default=None, description="状态")

    
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")
    project: str | None = Field(default=None, description="项目ID")
    version: int = Field(default=1, description="版本号")
    created_at: str | None = Field(default=None, description="创建时间")
    updated_at: str | None = Field(default=None, description="更新时间")

    # v3.3 层级化知识图谱字段
    tags: list[str] = Field(default_factory=list, description="标签列表")
    heading_level: int | None = Field(default=None, description="标题层级 1-6")
    parent_id: str | None = Field(default=None, description="父 Atom ID")
    order: str | None = Field(default=None, description="分数索引")
    aliases: list[str] = Field(default_factory=list, description="别名列表")
    entity_id: str | None = Field(default=None, description="所属 Entity ID")


class PaginatedAtomResponse(BaseModel):
    """分页 Atom 响应"""

    data: list[AtomResponse] = Field(..., description="Atom 列表")
    total: int = Field(..., description="总记录数")
    page: int = Field(..., description="当前页码")
    page_size: int = Field(..., description="每页大小")
    has_more: bool = Field(..., description="是否还有更多")


class BatchAtomRequest(BaseModel):
    """批量创建 Atom 请求"""

    atoms: list[AtomCreateRequest] = Field(..., description="Atom 创建请求列表")
    tenant_id: str = Field(default="default", description="租户ID")


class BatchAtomResponse(BaseModel):
    """批量创建 Atom 响应"""

    success: list[AtomResponse] = Field(default_factory=list, description="成功创建的 Atoms")
    failed: list[dict] = Field(default_factory=list, description="失败的条目 {index: int, error: str}")
    total: int = Field(..., description="总请求数")
    success_count: int = Field(..., description="成功数")
    failed_count: int = Field(..., description="失败数")


@router.post("/atoms", response_model=AtomResponse)
async def create_atom(request: AtomCreateRequest):
    """
    创建 Atom

    Atom 是最小知识单元，可以是：
    - function: 函数定义
    - class: 类定义
    - interface: 接口定义
    - import: 导入语句
    - goal: 目标
    - scope: 范围
    - task: 任务
    - note: 笔记
    - chapter: 章节（v3.3）
    - section: 小节（v3.3）
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        db = state.memory_manager.db

        
        if request.type not in ATOM_VALID_TYPES:
            raise ValidationError(f"无效的 Atom 类型: {request.type}. 必须是 {list(ATOM_VALID_TYPES)}")

        
        atom_data = {
            "type": request.type,
            "content": request.content,
            "tenant_id": request.tenant_id,
            "name": request.name,
            "signature": request.signature,
            "params": request.params,
            "return_type": request.return_type,
            "is_exported": request.is_exported,
            "is_async": request.is_async,
            "complexity": request.complexity,
            "max_nesting_depth": request.max_nesting_depth,
            "docstring": request.docstring,
            "start_line": request.start_line,
            "end_line": request.end_line,
            "status": request.status,
            "metadata": request.metadata,
            "project": request.project,
            "version": 1,
            "tags": request.tags,
            "heading_level": request.heading_level,
            "parent_id": request.parent_id,
            "order": request.order,
            "aliases": request.aliases,
            "entity_id": request.entity_id,
        }

        
        atom_data = {k: v for k, v in atom_data.items() if v is not None}

        # BL-B-100: 使用事务执行创建操作
        async with transaction(db, "Atom"):
            result = await db.create("atom", atom_data)

            if not result:
                raise HTTPException(status_code=500, detail="创建 Atom 失败")

            record = parse_surrealdb_result(result)
            if not record:
                raise HTTPException(status_code=500, detail="创建 Atom 失败: 无效的响应格式")

            record_id = extract_record_id(record)
            return AtomResponse(id=record_id, **atom_data)

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except Exception as e:
        logger.error("[Atom] 创建失败: %s", e)
        raise HTTPException(status_code=500, detail=f"创建失败: {e!s}") from e


@router.get("/atoms", response_model=PaginatedAtomResponse)
async def list_atoms(
    query: str | None = Query(default=None, description="按名称过滤（子串匹配）"),
    type: str | None = Query(default=None, description="Atom 类型过滤"),
    project: str | None = Query(default=None, description="项目过滤"),
    tenant_id: str = Query(default="default"),
    page: int = Query(default=1, ge=1, description="页码"),
    page_size: int = Query(default=50, ge=1, le=100, description="每页大小"),
    limit: int | None = Query(default=None, ge=1, le=100, description="返回数量限制（向后兼容）"),
    offset: int | None = Query(default=None, ge=0, description="偏移量（向后兼容）"),
):
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        db = state.memory_manager.db

        if limit is not None and offset is not None:
            skip = offset
            take = limit
        else:
            skip = (page - 1) * page_size
            take = page_size

        conditions = ["tenant_id = $tenant_id"]
        params: dict[str, Any] = {"tenant_id": tenant_id}

        if query:
            conditions.append("$name_query IN name")
            params["name_query"] = query
        if type:
            conditions.append("type = $atom_type")
            params["atom_type"] = type
        if project:
            conditions.append("project = $project")
            params["project"] = project

        where_clause = " AND ".join(conditions)

        count_result = await db.query(
            f"SELECT count() AS total FROM atom WHERE {where_clause} GROUP ALL",
            params,
        )
        records = state.memory_manager._extract_records(count_result)
        total = records[0].get("total", 0) if records else 0

        result = await db.query(
            f"SELECT * FROM atom WHERE {where_clause} ORDER BY created_at DESC LIMIT {take} START {skip}",
            params,
        )
        raw_data = result or []
        if total == 0 and raw_data:
            total = len(raw_data)

        data = []
        for record in raw_data:
            raw_id = record.get("id")
            if raw_id and hasattr(raw_id, "table_name"):
                record["id"] = f"{raw_id.table_name}:{raw_id.id}"
            for field in ["created_at", "updated_at"]:
                if field in record and record[field] is not None:
                    if hasattr(record[field], "isoformat"):
                        record[field] = record[field].isoformat()
            data.append(record)

        current_page = page if limit is None else (skip // take) + 1
        has_more = (skip + len(data)) < total

        return PaginatedAtomResponse(
            data=data,
            total=total,
            page=current_page,
            page_size=take,
            has_more=has_more,
        )

    except Exception as e:
        logger.error("[Atom] 列表查询失败: %s", e)
        raise HTTPException(status_code=500, detail=f"查询失败: {e!s}") from e


@router.get("/atoms/{atom_id}")
async def get_atom(atom_id: str, tenant_id: str = Query(default="default")):
    """获取 Atom 详情"""
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        db = state.memory_manager.db

        # 将字符串 ID 转换为 RecordID
        atom_parts = atom_id.split(":", 1)
        atom_record_id = RecordID(atom_parts[0], atom_parts[1])
        
        result = await db.query(
            "SELECT * FROM atom WHERE id = $atom_id AND tenant_id = $tenant_id",
            {"atom_id": atom_record_id, "tenant_id": tenant_id}
        )

        if not result or len(result) == 0:
            raise HTTPException(status_code=404, detail="Atom 不存在")

        return result[0]

    except HTTPException:
        raise
    except Exception as e:
        logger.error("[Atom] 查询失败: %s", e)
        raise HTTPException(status_code=500, detail=f"查询失败: {e!s}") from e


@router.post("/atoms/batch", response_model=BatchAtomResponse)
async def batch_create_atoms(request: BatchAtomRequest):
    """批量创建 Atoms"""
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    # 限制批量大小
    MAX_BATCH_SIZE = 100
    if len(request.atoms) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"批量创建数量超过限制: {len(request.atoms)} > {MAX_BATCH_SIZE}"
        )

    try:
        db = state.memory_manager.db

        results = {"success": [], "failed": []}

        # 使用事务执行批量创建
        async with transaction(db, "Atom"):
            for i, atom_req in enumerate(request.atoms):
                try:
                    # 验证类型
                    if atom_req.type not in ATOM_VALID_TYPES:
                        raise ValidationError(f"无效的 Atom 类型: {atom_req.type}")

                    # 准备数据
                    atom_data = {
                        "type": atom_req.type,
                        "content": atom_req.content,
                        "tenant_id": request.tenant_id,
                        "name": atom_req.name,
                        "signature": atom_req.signature,
                        "params": atom_req.params,
                        "return_type": atom_req.return_type,
                        "is_exported": atom_req.is_exported,
                        "is_async": atom_req.is_async,
                        "complexity": atom_req.complexity,
                        "max_nesting_depth": atom_req.max_nesting_depth,
                        "docstring": atom_req.docstring,
                        "start_line": atom_req.start_line,
                        "end_line": atom_req.end_line,
                        "status": atom_req.status,
                        "metadata": atom_req.metadata,
                        "project": atom_req.project,
                        "version": 1,
                        "tags": atom_req.tags,
                        "heading_level": atom_req.heading_level,
                        "parent_id": atom_req.parent_id,
                        "order": atom_req.order,
                        "aliases": atom_req.aliases,
                        "entity_id": atom_req.entity_id,
                    }
                    atom_data = {k: v for k, v in atom_data.items() if v is not None}

                    # 创建 atom
                    result = await db.create("atom", atom_data)

                    if not result:
                        raise HTTPException(status_code=500, detail="创建 Atom 失败")

                    # 处理返回结果
                    record = parse_surrealdb_result(result)
                    if not record:
                        raise HTTPException(status_code=500, detail="创建 Atom 失败: 无效的响应格式")

                    record_id = extract_record_id(record)
                    results["success"].append(AtomResponse(id=record_id, **atom_data))

                except Exception as e:
                    results["failed"].append({"index": i, "error": str(e)})

        return BatchAtomResponse(
            success=results["success"],
            failed=results["failed"],
            total=len(request.atoms),
            success_count=len(results["success"]),
            failed_count=len(results["failed"])
        )

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except Exception as e:
        logger.error("[Atom] 批量创建失败: %s", e)
        raise HTTPException(status_code=500, detail=f"批量创建失败: {e!s}") from e


@router.put("/atoms/{atom_id}")
async def update_atom(atom_id: str, request: AtomUpdateRequest, tenant_id: str = Query(default="default")):
    """更新 Atom"""
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        db = state.memory_manager.db

        # 将字符串 ID 转换为 RecordID
        atom_parts = atom_id.split(":", 1)
        atom_record_id = RecordID(atom_parts[0], atom_parts[1])

        # BL-B-105: 添加 tenant_id 验证
        check = await db.query(
            "SELECT id FROM atom WHERE id = $atom_id AND tenant_id = $tenant_id",
            {"atom_id": atom_record_id, "tenant_id": tenant_id}
        )
        if not check or len(check) == 0:
            raise HTTPException(status_code=404, detail="Atom 不存在")

        
        update_data = {}
        for field, value in request.model_dump(exclude_unset=True).items():
            if value is not None:
                update_data[field] = value

        if not update_data:
            raise HTTPException(status_code=400, detail="没有要更新的字段")

        # BL-B-104: 修复版本号更新逻辑 - 先查询当前版本
        current_result = await db.query(
            "SELECT version FROM atom WHERE id = $atom_id AND tenant_id = $tenant_id",
            {"atom_id": atom_record_id, "tenant_id": tenant_id}
        )
        current_version = current_result[0]["version"] if current_result and len(current_result) > 0 else 0
        update_data["version"] = current_version + 1
        update_data["updated_at"] = "time::now()"

        # BL-B-100: 使用事务执行更新操作
        async with transaction(db, "Atom"):
            # 使用 SurrealQL UPDATE 语句，将字符串 ID 转换为 RecordID
            atom_parts = atom_id.split(":", 1)
            atom_record_id = RecordID(atom_parts[0], atom_parts[1])
            
            set_clauses = []
            params: dict[str, Any] = {"atom_id": atom_record_id}
            for key, value in update_data.items():
                if key != "updated_at":
                    set_clauses.append(f"{key} = ${key}")
                    params[key] = value
            
            set_clause = ", ".join(set_clauses)
            # nosec B608: atom_record_id 来自已验证的 RecordID 对象，非用户输入
            query = f"UPDATE $atom_id SET {set_clause}, updated_at = time::now()"  # nosec B608
            result = await db.query(query, params)

            if not result or len(result) == 0:
                raise HTTPException(status_code=500, detail="更新失败")

            return result[0] if isinstance(result, list) else result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("[Atom] 更新失败: %s", e)
        raise HTTPException(status_code=500, detail=f"更新失败: {e!s}") from e


@router.delete("/atoms/{atom_id}")
async def delete_atom(atom_id: str, tenant_id: str = Query(default="default")):
    """删除 Atom"""
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        db = state.memory_manager.db

        # 将字符串 ID 转换为 RecordID
        atom_parts = atom_id.split(":", 1)
        atom_record_id = RecordID(atom_parts[0], atom_parts[1])
        
        check = await db.query(
            "SELECT id FROM atom WHERE id = $atom_id AND tenant_id = $tenant_id",
            {"atom_id": atom_record_id, "tenant_id": tenant_id}
        )
        if not check or len(check) == 0:
            raise HTTPException(status_code=404, detail="Atom 不存在")

        # BL-B-100: 使用事务执行删除操作
        async with transaction(db, "Atom"):
            await db.delete(atom_record_id)
            return {"success": True, "message": "Atom 已删除"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("[Atom] 删除失败: %s", e)
        raise HTTPException(status_code=500, detail=f"删除失败: {e!s}") from e
