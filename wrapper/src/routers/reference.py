"""Reference (Graph Relation) 端点 - 图关系管理"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from surrealdb.data.types.record_id import RecordID

from .. import state
from ..models import ReferenceType
from ..utils.db_helpers import extract_record_id
from ..utils.db_utils import extract_records
from ..utils.exceptions import ValidationError
from ..utils.transaction import transaction

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["references"])


class ReferenceCreateRequest(BaseModel):
    """创建 Reference 请求"""

    from_id: str = Field(..., description="源 ID (atom:xxx 或 entity:xxx)")
    to_id: str = Field(..., description="目标 ID (atom:xxx 或 entity:xxx)")
    type: str = Field(
        ...,
        description="关系类型",
        examples=ReferenceType.all_values(),
    )
    tenant_id: str = Field(default="default", description="租户ID")
    weight: float = Field(default=0.5, ge=0.0, le=1.0, description="关系权重")

    # v3.4 graphify 新增：置信度字段
    confidence: str | None = Field(
        default=None,
        description="关系发现方式: EXTRACTED/INFERRED/AMBIGUOUS"
    )
    confidence_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="置信度数值 0.0-1.0"
    )

    file_path: str | None = Field(default=None, description="文件路径")
    line: int | None = Field(default=None, description="行号")
    column: int | None = Field(default=None, description="列号")
    description: str | None = Field(default=None, description="关系描述")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")


class ReferenceResponse(BaseModel):
    """Reference 响应"""

    id: str = Field(..., description="Reference ID")
    from_id: str = Field(..., description="源 ID")
    to_id: str = Field(..., description="目标 ID")
    type: str = Field(..., description="关系类型")
    tenant_id: str = Field(..., description="租户ID")
    weight: float = Field(default=0.5, description="权重")

    # v3.4 graphify 新增：置信度字段
    confidence: str | None = Field(default=None, description="关系发现方式")
    confidence_score: float | None = Field(default=None, description="置信度数值")

    file_path: str | None = Field(default=None, description="文件路径")
    line: int | None = Field(default=None, description="行号")
    column: int | None = Field(default=None, description="列号")
    description: str | None = Field(default=None, description="关系描述")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")
    created_at: str | None = Field(default=None, description="创建时间")


class PaginatedReferenceResponse(BaseModel):
    """分页 Reference 响应"""

    data: list[ReferenceResponse] = Field(..., description="Reference 列表")
    total: int = Field(..., description="总记录数")
    page: int = Field(..., description="当前页码")
    page_size: int = Field(..., description="每页大小")
    has_more: bool = Field(..., description="是否还有更多")


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

        # 验证 type 是否为有效的关系类型
        valid_types = ReferenceType.all_values()
        if request.type not in valid_types:
            raise ValidationError(
                f"无效的关系类型: {request.type}，必须是 {', '.join(valid_types)} 之一"
            )

        # BL-B-33: 验证 from_id 和 to_id 格式
        if ":" not in request.from_id:
            raise ValidationError(f"from_id 格式无效: {request.from_id}，应为 table:id 格式")
        if ":" not in request.to_id:
            raise ValidationError(f"to_id 格式无效: {request.to_id}，应为 table:id 格式")

        from_table, from_id_part = request.from_id.split(":", 1)
        to_table, to_id_part = request.to_id.split(":", 1)

        # 验证 from_id 存在性 - 使用 type::record 函数
        try:
            from_check = await db.query(
                f"SELECT id FROM {from_table} WHERE id = type::record($from_id)",  # nosec B608
                {"from_id": request.from_id}
            )
        except Exception as e:
            logger.error(f"from_check query error: {e}")
            raise ValidationError(f"from_id 查询失败: {e}")

        from_records = extract_records(from_check)
        if not from_records:
            raise ValidationError(f"from_id 不存在: {request.from_id}")

        # 验证 to_id 存在性 - 使用 type::record 函数
        try:
            to_check = await db.query(
                f"SELECT id FROM {to_table} WHERE id = type::record($to_id)",  # nosec B608
                {"to_id": request.to_id}
            )
        except Exception as e:
            logger.error(f"to_check query error: {e}")
            raise ValidationError(f"to_id 查询失败: {e}")

        to_records = extract_records(to_check)
        if not to_records:
            raise ValidationError(f"to_id 不存在: {request.to_id}")

        # Build CONTENT object for RELATE
        # SurrealDB Python SDK works better with CONTENT than SET with variables
        content_obj = {
            "type": request.type,
            "tenant_id": request.tenant_id,
            "weight": request.weight,
        }
        if request.file_path is not None:
            content_obj["file_path"] = request.file_path
        if request.line is not None:
            content_obj["line"] = request.line
        if request.column is not None:
            content_obj["column"] = request.column
        if request.description is not None:
            content_obj["description"] = request.description
        if request.metadata:
            content_obj["metadata"] = request.metadata

        import json
        content_json = json.dumps(content_obj)
        query = f"""
        RELATE {request.from_id}->reference->{request.to_id} CONTENT {content_json}
        """

        # BL-B-100: 使用事务执行创建操作
        async with transaction(db, "Reference"):
            result = await db.query(query)

            if not result:
                raise HTTPException(status_code=500, detail="创建关系失败")

            records = extract_records(result)
            if not records:
                raise HTTPException(status_code=500, detail="创建关系失败: 无效的响应格式")

            record = records[0]
            record_id = extract_record_id(record)
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
                description=request.description,
                metadata=request.metadata,
            )

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except Exception as e:
        logger.error("[Reference] 创建失败: %s", e)
        raise HTTPException(status_code=500, detail=f"创建关系失败: {e!s}") from e


class BatchReferenceCreateRequest(BaseModel):
    """批量创建 Reference 请求"""

    references: list[ReferenceCreateRequest] = Field(
        ..., description="Reference 列表", max_length=100
    )


class BatchReferenceItemResponse(BaseModel):
    """批量创建 Reference 单项响应"""

    id: str | None = Field(default=None, description="Reference ID")
    from_id: str = Field(..., description="源 ID")
    to_id: str = Field(..., description="目标 ID")
    type: str = Field(..., description="关系类型")
    status: str = Field(..., description="状态: created/skipped/error")
    error: str | None = Field(default=None, description="错误信息")


class BatchReferenceResponse(BaseModel):
    """批量创建 Reference 响应"""

    references: list[BatchReferenceItemResponse] = Field(..., description="Reference 结果列表")
    total: int = Field(..., description="总请求数")
    created: int = Field(..., description="成功创建数")
    skipped: int = Field(..., description="跳过数（已存在）")
    errors: int = Field(..., description="失败数")


@router.post("/references/batch", response_model=BatchReferenceResponse)
async def create_references_batch(request: BatchReferenceCreateRequest):
    """
    批量创建关系

    一次请求创建多条 reference，支持部分成功：
    - 某条失败不影响其他条目
    - from_id + to_id + type 组合已存在则跳过
    - 单次请求上限 100 条
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    if len(request.references) > 100:
        raise HTTPException(status_code=400, detail="单次请求最多 100 条")

    db = state.memory_manager.db
    valid_types = ReferenceType.all_values()

    results: list[BatchReferenceItemResponse] = []
    created_count = 0
    skipped_count = 0
    error_count = 0

    # 检查重复（from_id + to_id + type）
    seen: set[tuple[str, str, str]] = set()

    for ref in request.references:
        key = (ref.from_id, ref.to_id, ref.type)

        # 检查请求内重复
        if key in seen:
            results.append(
                BatchReferenceItemResponse(
                    from_id=ref.from_id,
                    to_id=ref.to_id,
                    type=ref.type,
                    status="skipped",
                    error="请求内重复",
                )
            )
            skipped_count += 1
            continue

        seen.add(key)

        try:
            # 验证 type
            if ref.type not in valid_types:
                raise ValidationError(
                    f"无效的关系类型: {ref.type}，必须是 {', '.join(valid_types)} 之一"
                )

            # 验证 from_id 和 to_id 格式
            if ":" not in ref.from_id:
                raise ValidationError(f"from_id 格式无效: {ref.from_id}")
            if ":" not in ref.to_id:
                raise ValidationError(f"to_id 格式无效: {ref.to_id}")

            from_table, _ = ref.from_id.split(":", 1)
            to_table, _ = ref.to_id.split(":", 1)

            # 检查 from_id 存在性
            from_check = await db.query(
                f"SELECT id FROM {from_table} WHERE id = type::record($from_id)",  # nosec B608
                {"from_id": ref.from_id}
            )
            if not extract_records(from_check):
                raise ValidationError(f"from_id 不存在: {ref.from_id}")

            # 检查 to_id 存在性
            to_check = await db.query(
                f"SELECT id FROM {to_table} WHERE id = type::record($to_id)",  # nosec B608
                {"to_id": ref.to_id}
            )
            if not extract_records(to_check):
                raise ValidationError(f"to_id 不存在: {ref.to_id}")

            # 检查是否已存在
            existing = await db.query(
                "SELECT id FROM reference WHERE in = type::record($from_id) AND out = type::record($to_id) AND type = $type",
                {"from_id": ref.from_id, "to_id": ref.to_id, "type": ref.type}
            )
            if extract_records(existing):
                results.append(
                    BatchReferenceItemResponse(
                        from_id=ref.from_id,
                        to_id=ref.to_id,
                        type=ref.type,
                        status="skipped",
                        error="关系已存在",
                    )
                )
                skipped_count += 1
                continue

            # 构建 CONTENT
            content_obj = {
                "type": ref.type,
                "tenant_id": ref.tenant_id,
                "weight": ref.weight,
            }
            if ref.file_path is not None:
                content_obj["file_path"] = ref.file_path
            if ref.line is not None:
                content_obj["line"] = ref.line
            if ref.column is not None:
                content_obj["column"] = ref.column
            if ref.description is not None:
                content_obj["description"] = ref.description
            if ref.metadata:
                content_obj["metadata"] = ref.metadata

            import json

            content_json = json.dumps(content_obj)
            query = f"RELATE {ref.from_id}->reference->{ref.to_id} CONTENT {content_json}"

            result = await db.query(query)
            records = extract_records(result)

            if records:
                record_id = extract_record_id(records[0])
                results.append(
                    BatchReferenceItemResponse(
                        id=record_id,
                        from_id=ref.from_id,
                        to_id=ref.to_id,
                        type=ref.type,
                        status="created",
                    )
                )
                created_count += 1
            else:
                raise ValidationError("创建失败: 无效的响应格式")

        except ValidationError as e:
            results.append(
                BatchReferenceItemResponse(
                    from_id=ref.from_id,
                    to_id=ref.to_id,
                    type=ref.type,
                    status="error",
                    error=e.message,
                )
            )
            error_count += 1
        except Exception as e:
            logger.error("[Reference] 批量创建单项失败: %s", e)
            results.append(
                BatchReferenceItemResponse(
                    from_id=ref.from_id,
                    to_id=ref.to_id,
                    type=ref.type,
                    status="error",
                    error=str(e),
                )
            )
            error_count += 1

    return BatchReferenceResponse(
        references=results,
        total=len(request.references),
        created=created_count,
        skipped=skipped_count,
        errors=error_count,
    )


@router.get("/references", response_model=PaginatedReferenceResponse)
async def query_references(
    from_id: str | None = Query(default=None, description="源 ID"),
    to_id: str | None = Query(default=None, description="目标 ID"),
    type: str | None = Query(default=None, description="关系类型"),
    tenant_id: str = Query(default="default"),
    page: int = Query(default=1, ge=1, description="页码"),
    page_size: int = Query(default=50, ge=1, le=100, description="每页大小"),
    limit: int | None = Query(default=None, ge=1, le=100, description="返回数量限制（向后兼容）"),
    offset: int | None = Query(default=None, ge=0, description="偏移量（向后兼容）"),
):
    """
    查询关系（支持分页）

    支持图遍历查询：
    - from_id: 查询从该节点出发的关系
    - to_id: 查询指向该节点的关系
    """
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

        # 构建查询
        if from_id:
            query = "SELECT * FROM $from_id->reference WHERE tenant_id = $tenant_id"
            params = {"from_id": from_id, "tenant_id": tenant_id}
            if type:
                query += " AND type = $type"
                params["type"] = type
            query += f" LIMIT {take} START {skip}"
            result = await db.query(query, params)
            # 图查询不支持 count，使用 len(result) 作为 total 的近似值
            total = len(result) if result else 0
        elif to_id:
            query = "SELECT * FROM <-reference-$to_id WHERE tenant_id = $tenant_id"
            params = {"to_id": to_id, "tenant_id": tenant_id}
            if type:
                query += " AND type = $type"
                params["type"] = type
            query += f" LIMIT {take} START {skip}"
            result = await db.query(query, params)
            total = len(result) if result else 0
        else:
            # 查询总数
            count_query = "SELECT count() FROM reference WHERE tenant_id = $tenant_id"
            count_params = {"tenant_id": tenant_id}
            if type:
                count_query += " AND type = $type"
                count_params["type"] = type
            count_result = await db.query(f"{count_query} GROUP ALL", count_params)
            count_records = extract_records(count_result)
            total = count_records[0].get("count", 0) if count_records else 0

            # 查询数据
            query = "SELECT * FROM reference WHERE tenant_id = $tenant_id"
            params = {"tenant_id": tenant_id}
            if type:
                query += " AND type = $type"
                params["type"] = type
            query += f" LIMIT {take} START {skip}"
            result = await db.query(query, params)

        raw_data = result or []

        data = []
        for record in raw_data:
            raw_id = record.get("id")
            if raw_id and hasattr(raw_id, "table_name"):
                record["id"] = f"{raw_id.table_name}:{raw_id.id}"
            for edge_field, api_field in [("in", "from_id"), ("out", "to_id")]:
                field_id = record.get(edge_field)
                if field_id is not None and hasattr(field_id, "table_name"):
                    record[api_field] = f"{field_id.table_name}:{field_id.id}"
                else:
                    record[api_field] = str(field_id) if field_id is not None else None
            for field in ["created_at"]:
                if field in record and record[field] is not None:
                    if hasattr(record[field], "isoformat"):
                        record[field] = record[field].isoformat()
            data.append(record)

        # 计算当前页码和 has_more
        current_page = page if limit is None else (skip // take) + 1
        current_page_size = take
        has_more = (skip + len(data)) < total

        return PaginatedReferenceResponse(
            data=data,
            total=total,
            page=current_page,
            page_size=current_page_size,
            has_more=has_more
        )

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

        # 将字符串 ID 转换为 RecordID 对象
        if ":" in reference_id:
            parts = reference_id.split(":", 1)
            record_id = RecordID(parts[0], parts[1])
        else:
            record_id = reference_id

        check = await db.query(
            "SELECT id FROM reference WHERE id = $reference_id AND tenant_id = $tenant_id",
            {"reference_id": record_id, "tenant_id": tenant_id}
        )
        if not check or len(check) == 0:
            raise HTTPException(status_code=404, detail="关系不存在")

        # BL-B-100: 使用事务执行删除操作
        async with transaction(db, "Reference"):
            await db.delete(record_id)
            return {"success": True, "message": "关系已删除"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("[Reference] 删除失败: %s", e)
        raise HTTPException(status_code=500, detail=f"删除失败: {e!s}") from e
