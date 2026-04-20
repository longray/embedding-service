"""Atom CRUD 端点 - 原子级知识单元管理"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .. import state
from ..utils.exceptions import ValidationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["atoms"])





class AtomCreateRequest(BaseModel):
    """创建 Atom 请求"""

    type: str = Field(
        ...,
        description="Atom 类型",
        examples=["function", "class", "interface", "import", "goal", "scope", "task", "note"],
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
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        db = state.memory_manager.db

        
        valid_types = ["function", "class", "interface", "import", "goal", "scope", "task", "note"]
        if request.type not in valid_types:
            raise ValidationError(f"无效的 Atom 类型: {request.type}. 必须是 {valid_types}")

        
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
        }

        
        atom_data = {k: v for k, v in atom_data.items() if v is not None}

        # BL-B-100: 使用事务执行创建操作
        try:
            await db.query("BEGIN TRANSACTION")
            result = await db.create("atom", atom_data)

            if not result:
                await db.query("CANCEL TRANSACTION")
                raise HTTPException(status_code=500, detail="创建 Atom 失败")

            if isinstance(result, dict):
                record = result
            elif isinstance(result, list) and result:
                record = result[0]
                if isinstance(record, list) and record:
                    record = record[0]
            else:
                await db.query("CANCEL TRANSACTION")
                raise HTTPException(status_code=500, detail="创建 Atom 失败: 无效的响应格式")

            raw_id = record.get("id") if isinstance(record, dict) else record
            if raw_id and not isinstance(raw_id, list) and hasattr(raw_id, "table_name"):
                record_id = f"{raw_id.table_name}:{raw_id.id}"
            else:
                record_id = str(raw_id)

            await db.query("COMMIT TRANSACTION")
            return AtomResponse(id=record_id, **atom_data)

        except Exception:
            try:
                await db.query("CANCEL TRANSACTION")
            except Exception as cancel_error:
                logger.error("[Atom] 事务回滚失败: %s", cancel_error)
            raise

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except Exception as e:
        logger.error("[Atom] 创建失败: %s", e)
        raise HTTPException(status_code=500, detail=f"创建失败: {e!s}") from e


@router.get("/atoms/{atom_id}")
async def get_atom(atom_id: str, tenant_id: str = Query(default="default")):
    """获取 Atom 详情"""
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        db = state.memory_manager.db

        
        result = await db.query(
            "SELECT * FROM atom WHERE id = $atom_id AND tenant_id = $tenant_id",
            {"atom_id": atom_id, "tenant_id": tenant_id}
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
        await db.query("BEGIN TRANSACTION")
        try:
            for i, atom_req in enumerate(request.atoms):
                try:
                    # 验证类型
                    valid_types = ["function", "class", "interface", "import", "goal", "scope", "task", "note"]
                    if atom_req.type not in valid_types:
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
                    }
                    atom_data = {k: v for k, v in atom_data.items() if v is not None}

                    # 创建 atom
                    result = await db.create("atom", atom_data)

                    if not result:
                        raise HTTPException(status_code=500, detail="创建 Atom 失败")

                    # 处理返回结果
                    if isinstance(result, dict):
                        record = result
                    elif isinstance(result, list) and result:
                        record = result[0]
                        if isinstance(record, list) and record:
                            record = record[0]
                    else:
                        raise HTTPException(status_code=500, detail="创建 Atom 失败: 无效的响应格式")

                    raw_id = record.get("id") if isinstance(record, dict) else record
                    if raw_id and not isinstance(raw_id, list) and hasattr(raw_id, "table_name"):
                        record_id = f"{raw_id.table_name}:{raw_id.id}"
                    else:
                        record_id = str(raw_id)

                    results["success"].append(AtomResponse(id=record_id, **atom_data))

                except Exception as e:
                    results["failed"].append({"index": i, "error": str(e)})

            await db.query("COMMIT TRANSACTION")

        except Exception:
            await db.query("CANCEL TRANSACTION")
            raise

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


@router.get("/atoms", response_model=PaginatedAtomResponse)
async def list_atoms(
    type: str | None = Query(default=None, description="Atom 类型过滤"),
    project: str | None = Query(default=None, description="项目过滤"),
    tenant_id: str = Query(default="default", description="租户ID"),
    page: int = Query(default=1, ge=1, description="页码"),
    page_size: int = Query(default=50, ge=1, le=100, description="每页大小"),
    limit: int | None = Query(default=None, ge=1, le=100, description="返回数量限制（向后兼容）"),
    offset: int | None = Query(default=None, ge=0, description="偏移量（向后兼容）"),
):
    """列出 Atoms（支持分页）"""
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        db = state.memory_manager.db

        # 向后兼容：如果提供了 limit/offset，使用它们
        if limit is not None and offset is not None:
            skip = offset
            take = limit
        else:
            skip = (page - 1) * page_size
            take = page_size

        # 查询总数
        count_query = "SELECT count() FROM atom WHERE tenant_id = $tenant_id"
        count_params = {"tenant_id": tenant_id}

        if type:
            count_query += " AND type = $type"
            count_params["type"] = type
        if project:
            count_query += " AND project = $project"
            count_params["project"] = project

        count_result = await db.query(count_query, count_params)
        total = count_result[0]["count"] if count_result and len(count_result) > 0 else 0

        # 查询数据
        query = "SELECT * FROM atom WHERE tenant_id = $tenant_id"
        params = {"tenant_id": tenant_id}

        if type:
            query += " AND type = $type"
            params["type"] = type
        if project:
            query += " AND project = $project"
            params["project"] = project

        query += f" LIMIT {take} START {skip}"

        result = await db.query(query, params)
        raw_data = result or []

        # 转换数据格式以匹配 Pydantic 模型
        data = []
        for record in raw_data:
            # 处理 RecordID
            raw_id = record.get("id")
            if raw_id and hasattr(raw_id, "table_name"):
                record["id"] = f"{raw_id.table_name}:{raw_id.id}"
            # 处理 datetime
            for field in ["created_at", "updated_at"]:
                if field in record and record[field] is not None:
                    if hasattr(record[field], "isoformat"):
                        record[field] = record[field].isoformat()
            data.append(record)

        # 计算当前页码和 has_more
        current_page = page if limit is None else (skip // take) + 1
        current_page_size = take
        has_more = (skip + len(data)) < total

        return PaginatedAtomResponse(
            data=data,
            total=total,
            page=current_page,
            page_size=current_page_size,
            has_more=has_more
        )

    except Exception as e:
        logger.error("[Atom] 列表查询失败: %s", e)
        raise HTTPException(status_code=500, detail=f"查询失败: {e!s}") from e


@router.put("/atoms/{atom_id}")
async def update_atom(atom_id: str, request: AtomUpdateRequest):
    """更新 Atom"""
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        db = state.memory_manager.db

        
        check = await db.query("SELECT id FROM atom WHERE id = $atom_id", {"atom_id": atom_id})
        if not check or len(check) == 0:
            raise HTTPException(status_code=404, detail="Atom 不存在")

        
        update_data = {}
        for field, value in request.model_dump(exclude_unset=True).items():
            if value is not None:
                update_data[field] = value

        if not update_data:
            raise HTTPException(status_code=400, detail="没有要更新的字段")

        update_data["version"] = "version + 1"
        update_data["updated_at"] = "time::now()"

        # BL-B-100: 使用事务执行更新操作
        try:
            await db.query("BEGIN TRANSACTION")
            result = await db.update(atom_id, update_data)

            if not result or len(result) == 0:
                await db.query("CANCEL TRANSACTION")
                raise HTTPException(status_code=500, detail="更新失败")

            await db.query("COMMIT TRANSACTION")
            return result[0]

        except Exception:
            try:
                await db.query("CANCEL TRANSACTION")
            except Exception as cancel_error:
                logger.error("[Atom] 事务回滚失败: %s", cancel_error)
            raise

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

        
        check = await db.query(
            "SELECT id FROM atom WHERE id = $atom_id AND tenant_id = $tenant_id",
            {"atom_id": atom_id, "tenant_id": tenant_id}
        )
        if not check or len(check) == 0:
            raise HTTPException(status_code=404, detail="Atom 不存在")

        # BL-B-100: 使用事务执行删除操作
        try:
            await db.query("BEGIN TRANSACTION")
            await db.delete(atom_id)
            await db.query("COMMIT TRANSACTION")
            return {"success": True, "message": "Atom 已删除"}

        except Exception:
            try:
                await db.query("CANCEL TRANSACTION")
            except Exception as cancel_error:
                logger.error("[Atom] 事务回滚失败: %s", cancel_error)
            raise

    except HTTPException:
        raise
    except Exception as e:
        logger.error("[Atom] 删除失败: %s", e)
        raise HTTPException(status_code=500, detail=f"删除失败: {e!s}") from e
