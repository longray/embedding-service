"""Entity CRUD 端点 - 知识实体管理"""

import logging
from collections.abc import Sequence
from typing import Any, Union

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from surrealdb.data.types.record_id import RecordID

from .. import state
from ..utils.db_helpers import extract_record_id, parse_surrealdb_result
from ..utils.exceptions import ValidationError
from ..utils.transaction import transaction

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["entities"])

# 模块级常量：Entity 有效类型
ENTITY_VALID_TYPES = frozenset(["memory", "backlog", "wiki", "code"])


class AtomInlineCreate(BaseModel):
    """内联创建 Atom 请求 — 可嵌入 EntityCreateRequest / EntityUpdateRequest 的 atoms 列表"""

    type: str = Field(..., description="Atom 类型 (function, class, chapter, section, etc.)")
    content: str = Field(..., description="Atom 内容")
    name: str | None = Field(default=None, description="Atom 名称")
    local_id: str | None = Field(default=None, description="客户端侧 ID (用于树结构)")
    heading_level: int | None = Field(default=None, ge=1, le=6, description="标题层级 1-6")
    parent_id: str | None = Field(default=None, description="父 Atom 的 local_id")
    order: str | None = Field(default=None, description="排序键")
    aliases: list[str] | None = Field(default=None, description="别名")
    tags: list[str] | None = Field(default=None, description="标签")
    signature: str | None = Field(default=None, description="函数签名")
    params: list[dict[str, Any]] | None = Field(default=None, description="参数列表")
    return_type: str | None = Field(default=None, description="返回类型")
    is_exported: bool | None = Field(default=None, description="是否导出")
    is_async: bool | None = Field(default=None, description="是否异步")
    complexity: int | None = Field(default=None, description="复杂度")
    start_line: int | None = Field(default=None, description="起始行号")
    end_line: int | None = Field(default=None, description="结束行号")
    docstring: dict[str, Any] | None = Field(default=None, description="文档字符串")
    metadata: dict[str, Any] | None = Field(default=None, description="元数据")
    project: str | None = Field(default=None, description="项目 ID")
    fingerprint: str | None = Field(default=None, description="内容指纹")
    children: list["AtomInlineCreate"] | None = Field(default=None, description="子 Atom 列表")


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
    atoms: list[Union[str, AtomInlineCreate]] = Field(
        default_factory=list, description="Atom ID 或内联创建对象列表 (L2)"
    )

    
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
    atoms: list[Union[str, AtomInlineCreate]] | None = Field(
        default=None, description="Atom ID 或内联创建对象列表"
    )
    tags: list[str] | None = Field(default=None, description="标签")
    status: str | None = Field(default=None, description="状态")
    priority: str | None = Field(default=None, description="优先级")


class AtomRef(BaseModel):
    """Atom 引用 - 包含 record id 和 local_id"""

    id: str = Field(..., description="Atom Record ID (如 atom:xxx)")
    local_id: str | None = Field(default=None, description="Atom Local ID (插件端生成)")


class EntityResponse(BaseModel):
    """Entity 响应"""

    model_config = {"extra": "ignore"}

    id: str = Field(..., description="Entity ID")
    type: str = Field(..., description="Entity 类型")
    tenant_id: str = Field(..., description="租户ID")


    abstract: str = Field(..., description="摘要")
    overview: dict[str, Any] = Field(default_factory=dict, description="概览")
    atoms: list[AtomRef] = Field(default_factory=list, description="Atom 引用列表（含 id 和 local_id）")

    
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
    created_by: str | None = Field(default=None, description="创建者")
    created_at: str | None = Field(default=None, description="创建时间")
    updated_at: str | None = Field(default=None, description="更新时间")


class PaginatedEntityResponse(BaseModel):
    """分页 Entity 响应"""

    data: list[EntityResponse] = Field(..., description="Entity 列表")
    total: int = Field(..., description="总记录数")
    page: int = Field(..., description="当前页码")
    page_size: int = Field(..., description="每页大小")
    has_more: bool = Field(..., description="是否还有更多")


class BatchEntityRequest(BaseModel):
    """批量创建 Entity 请求"""

    entities: list[EntityCreateRequest] = Field(..., description="Entity 创建请求列表")
    tenant_id: str = Field(default="default", description="租户ID")


class BatchEntityResponse(BaseModel):
    """批量创建 Entity 响应"""

    success: list[EntityResponse] = Field(default_factory=list, description="成功创建的 Entities")
    failed: list[dict] = Field(default_factory=list, description="失败的条目 {index: int, error: str}")
    total: int = Field(..., description="总请求数")
    success_count: int = Field(..., description="成功数")
    failed_count: int = Field(..., description="失败数")


def _parse_record_id(atom_id: str) -> RecordID | str:
    if ":" in atom_id:
        table, id_part = atom_id.split(":", 1)
        return RecordID(table, id_part)
    return atom_id


async def _process_atoms_recursive(
    db: Any,
    atoms: Sequence[Union[str, AtomInlineCreate, dict[str, Any]]],
    entity_id: str | None = None,
    tenant_id: str = "default",
    entity_abstract: str = "",
    parent_local_id: str | None = None,
) -> list[AtomRef]:
    """递归处理 atoms 列表，包括嵌套的 children。

    v3.3-opt: 为 Atom 生成 embedding，拼接 entity abstract + atom content。
    递归处理 children，确保所有层级的 atoms 都有 entity_id。

    Args:
        db: SurrealDB 连接
        atoms: Atom ID (str)、内联创建对象 (AtomInlineCreate) 或 dict (来自 model_dump)
        entity_id: 关联的 Entity ID，用于反查
        tenant_id: 租户 ID，注入到内联创建的 Atom 中
        entity_abstract: Entity 摘要，用于 embedding context 拼接
        parent_local_id: 父 Atom 的 local_id，用于设置 parent_id

    Returns:
        AtomRef 列表，包含所有创建的 atoms（包括 children）的 record id 和 local_id。
    """
    if not atoms:
        return []

    str_ids: list[str] = []
    inline_atoms: list[AtomInlineCreate] = []

    for item in atoms:
        if isinstance(item, str):
            str_ids.append(item)
        elif isinstance(item, AtomInlineCreate):
            inline_atoms.append(item)
        elif isinstance(item, dict):
            inline_atoms.append(AtomInlineCreate(**item))
        else:
            raise ValidationError(f"atoms 元素类型无效: {type(item).__name__}")

    result_refs: list[AtomRef] = []
    created_atom_ids: list[str] = []

    # 收集所有需要生成 embedding 的 atoms（包括 children）
    embedding_inputs: list[tuple[AtomInlineCreate, str]] = []  # (atom, input_text)
    atom_children_map: list[tuple[AtomInlineCreate, list]] = []  # (atom, children_list)

    for atom_req in inline_atoms:
        parts = []
        if entity_abstract:
            parts.append(entity_abstract)
        if atom_req.name:
            parts.append(atom_req.name)
        if atom_req.content:
            parts.append(atom_req.content)
        embedding_input = "\n".join(parts) if parts else (atom_req.content or "")
        embedding_inputs.append((atom_req, embedding_input))

        # 收集 children
        if atom_req.children:
            atom_children_map.append((atom_req, atom_req.children))

    # 批量生成 embeddings
    embeddings: list[list[float]] = []
    if embedding_inputs:
        inputs = [inp for _, inp in embedding_inputs]
        if state.memory_manager and inputs:
            try:
                logger.warning("[Entity] Generating embeddings for %d atoms", len(inputs))
                embeddings = await state.memory_manager._get_embeddings(inputs)
                logger.warning("[Entity] Generated %d embeddings with dim=%d", len(embeddings), len(embeddings[0]) if embeddings else 0)
            except Exception as e:
                logger.error("[Entity] Failed to generate embeddings for atoms: %s", e)
                embeddings = [[] for _ in inputs]
        else:
            embeddings = [[] for _ in inputs]

    # 创建顶层 atoms
    for (atom_req, _), embedding in zip(embedding_inputs, embeddings):
        # BL-FIX-001: 修复 local_id 被 exclude_none=True 排除的问题
        local_id = atom_req.local_id
        atom_data = atom_req.model_dump(exclude_none=True, exclude={'children'})

        # 显式添加 local_id（如果提供了）
        if local_id:
            atom_data["local_id"] = local_id

        # 设置 parent_id（如果提供了父 local_id）
        if parent_local_id:
            atom_data["parent_id"] = parent_local_id

        if entity_id:
            atom_data["entity_id"] = entity_id
        atom_data["tenant_id"] = tenant_id

        # 添加 embedding
        if embedding:
            atom_data["embedding"] = embedding

        created = await db.create("atom", atom_data)
        record = parse_surrealdb_result(created)
        if not record:
            raise ValidationError(f"创建内联 Atom 失败: type={atom_req.type}, name={atom_req.name}")

        rid = extract_record_id(record)
        created_atom_ids.append(rid)
        result_refs.append(AtomRef(id=rid, local_id=local_id))
        logger.info("[Entity] 内联创建 Atom: %s (local_id=%s)", rid, atom_req.local_id)

        # BL-FIX-004: 同步 Atom 到 Meilisearch
        if state.memory_manager and state.memory_manager._meili:
            try:
                meili_doc = state.memory_manager._build_meili_doc(
                    rid, atom_data, tenant_id, doc_type="atom"
                )
                await state.memory_manager._meili.add_documents([meili_doc])
                logger.info("[Entity] Atom 同步到 Meilisearch: %s", rid)
            except Exception as meili_err:
                logger.warning("[Entity] Atom Meilisearch 同步失败 %s: %s", rid, meili_err)

        # 递归处理 children
        if atom_req.children:
            children_refs = await _process_atoms_recursive(
                db,
                atom_req.children,
                entity_id=entity_id,
                tenant_id=tenant_id,
                entity_abstract=entity_abstract,
                parent_local_id=atom_req.local_id
            )
            result_refs.extend(children_refs)
            logger.info("[Entity] 递归创建 %d 个子 atoms for %s", len(children_refs), atom_req.local_id)

    # --- 验证已有 Atom IDs 并获取 local_id ---
    if str_ids:
        record_ids = [_parse_record_id(aid) for aid in str_ids]

        # 合并查询：验证存在性并获取 local_id
        atoms_result = await db.query(
            "SELECT id, local_id FROM atom WHERE id IN $atom_ids",
            {"atom_ids": record_ids},
        )

        found_ids: set[str] = set()
        if atoms_result:
            for record in atoms_result:
                record_id = record.get("id")
                local_id = record.get("local_id")
                if hasattr(record_id, "table_name"):
                    rid = f"{record_id.table_name}:{record_id.id}"
                else:
                    rid = str(record_id)
                found_ids.add(rid)
                result_refs.append(AtomRef(id=rid, local_id=local_id))

        missing = set(str_ids) - found_ids
        if missing:
            raise ValidationError(f"Atoms 不存在: {missing}")

    return result_refs


async def _process_atoms(
    db: Any,
    atoms: Sequence[Union[str, AtomInlineCreate, dict[str, Any]]],
    entity_id: str | None = None,
    tenant_id: str = "default",
    entity_abstract: str = "",
) -> list[AtomRef]:
    """处理 atoms 列表（向后兼容的包装函数）。

    调用递归版本处理嵌套 children。
    """
    return await _process_atoms_recursive(db, atoms, entity_id, tenant_id, entity_abstract)


def _record_id_str(rid: RecordID | str) -> str:
    if isinstance(rid, RecordID):
        return f"{rid.table_name}:{rid.id}"
    return str(rid)


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

        
        if request.type not in ENTITY_VALID_TYPES:
            raise ValidationError(f"无效的 Entity 类型: {request.type}. 必须是 {list(ENTITY_VALID_TYPES)}")

        entity_data = {
            "type": request.type,
            "tenant_id": request.tenant_id,
            "abstract": request.abstract,
            "overview": request.overview,
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
        async with transaction(db, "Entity"):
            atom_refs = await _process_atoms(
                db,
                request.atoms,
                tenant_id=request.tenant_id,
                entity_abstract=request.abstract
            )
            # 存储到数据库时转换为 RecordID 对象列表
            if atom_refs:
                entity_data["atoms"] = [_parse_record_id(ref.id) for ref in atom_refs]
            else:
                entity_data["atoms"] = []

            result = await db.create("entity", entity_data)

            if not result:
                raise HTTPException(status_code=500, detail="创建 Entity 失败")

            # db.create 返回的是 RecordID 对象或字符串，不是字典
            if hasattr(result, 'table_name'):
                record_id = f"{result.table_name}:{result.id}"
            elif isinstance(result, str) and ':' in result:
                record_id = result
            elif isinstance(result, (dict, list)):
                record = parse_surrealdb_result(result)
                if not record:
                    raise HTTPException(status_code=500, detail=f"创建 Entity 失败: 无效的响应格式 {type(result)}")
                record_id = extract_record_id(record)
            else:
                raise HTTPException(status_code=500, detail=f"创建 Entity 失败: 未知的响应格式 {type(result)}")
            logger.warning("[Entity] Created entity with ID: %s, atom_refs count: %d", record_id, len(atom_refs))
            logger.warning("[Entity] atom_refs sample: %s", atom_refs[:3] if atom_refs else "[]")

            if atom_refs:
                # 将字符串 ID 转换为 RecordID 对象（用于 WHERE id IN $atom_ids）
                atom_record_ids = []
                for ref in atom_refs:
                    if ":" in ref.id:
                        table, id_part = ref.id.split(":", 1)
                        atom_record_ids.append(RecordID(table, id_part))
                    else:
                        atom_record_ids.append(ref.id)
                
                # BL-FIX-002: entity_id 保持为字符串（SurrealDB schema 定义为 option<string>）
                logger.warning("[Entity] Executing UPDATE for entity_id=%s, atom_record_ids=%s", record_id, atom_record_ids)
                update_result = await db.query(
                    "UPDATE atom SET entity_id = $entity_id WHERE id IN $atom_ids",
                    {"entity_id": record_id, "atom_ids": atom_record_ids}
                )
                logger.warning("[Entity] UPDATE result: %s", update_result)
                logger.info("[Entity] Updated entity_id for %d atoms", len(atom_refs))

            # BL-FIX-003: 同步 Entity 到 Meilisearch
            if state.memory_manager and state.memory_manager._meili:
                try:
                    meili_doc = state.memory_manager._build_meili_doc(
                        record_id, entity_data, request.tenant_id, doc_type="entity"
                    )
                    await state.memory_manager._meili.add_documents([meili_doc])
                    logger.info("[Entity] 同步到 Meilisearch: %s", record_id)
                except Exception as meili_err:
                    logger.warning("[Entity] Meilisearch 同步失败 %s: %s", record_id, meili_err)

            response_data = entity_data.copy()
            # 将 atoms 从 record id 列表转换为 AtomRef 列表
            response_data["atoms"] = atom_refs if atom_refs else []

            return EntityResponse(id=record_id, **response_data)

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except Exception as e:
        logger.error("[Entity] 创建失败: %s", e)
        raise HTTPException(status_code=500, detail=f"创建失败: {e!s}") from e


@router.post("/entities/batch", response_model=BatchEntityResponse)
async def batch_create_entities(request: BatchEntityRequest):
    """批量创建 Entities"""
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    # 限制批量大小
    MAX_BATCH_SIZE = 100
    if len(request.entities) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"批量创建数量超过限制: {len(request.entities)} > {MAX_BATCH_SIZE}"
        )

    db = state.memory_manager.db

    results = {"success": [], "failed": []}

    # 每个 entity 独立事务，避免一个失败影响其他
    for i, entity_req in enumerate(request.entities):
        try:
            # 验证类型
            if entity_req.type not in ENTITY_VALID_TYPES:
                raise ValidationError(f"无效的 Entity 类型: {entity_req.type}")

            async with transaction(db, "Entity"):
                atom_refs = await _process_atoms(
                    db,
                    entity_req.atoms,
                    tenant_id=request.tenant_id,
                    entity_abstract=entity_req.abstract
                )

                # 准备数据 - 存储时转换为 RecordID 对象列表
                entity_data = {
                    "type": entity_req.type,
                    "tenant_id": request.tenant_id,
                    "abstract": entity_req.abstract,
                    "overview": entity_req.overview,
                    "atoms": [_parse_record_id(ref.id) for ref in atom_refs] if atom_refs else [],
                    "tags": entity_req.tags,
                    "project": entity_req.project,
                    "created_by": entity_req.created_by,
                }

                # 根据类型添加特定字段
                if entity_req.type == "wiki":
                    entity_data.update({
                        "title": entity_req.title,
                        "aliases": entity_req.aliases,
                    })
                elif entity_req.type == "backlog":
                    entity_data.update({
                        "priority": entity_req.priority,
                        "status": entity_req.status,
                        "scene": entity_req.scene,
                        "estimated_hours": entity_req.estimated_hours,
                        "actual_hours": entity_req.actual_hours,
                    })
                elif entity_req.type == "code":
                    entity_data.update({
                        "file_path": entity_req.file_path,
                        "language": entity_req.language,
                        "quality_score": entity_req.quality_score,
                        "complexity_metrics": entity_req.complexity_metrics,
                    })

                entity_data = {k: v for k, v in entity_data.items() if v is not None}

                # 创建 entity
                result = await db.create("entity", entity_data)

                if not result:
                    raise HTTPException(status_code=500, detail="创建 Entity 失败")

                # db.create 返回的是 RecordID 对象或字符串，不是字典
                if hasattr(result, 'table_name'):
                    record_id = f"{result.table_name}:{result.id}"
                elif isinstance(result, str) and ':' in result:
                    record_id = result
                elif isinstance(result, (dict, list)):
                    record = parse_surrealdb_result(result)
                    if not record:
                        raise HTTPException(status_code=500, detail=f"创建 Entity 失败: 无效的响应格式 {type(result)}")
                    record_id = extract_record_id(record)
                else:
                    raise HTTPException(status_code=500, detail=f"创建 Entity 失败: 未知的响应格式 {type(result)}")

                if atom_refs:
                    # 将字符串 ID 转换为 RecordID 对象（用于 WHERE id IN $atom_ids）
                    atom_record_ids = []
                    for ref in atom_refs:
                        if ":" in ref.id:
                            table, id_part = ref.id.split(":", 1)
                            atom_record_ids.append(RecordID(table, id_part))
                        else:
                            atom_record_ids.append(ref.id)
                    
                    # BL-FIX-002: entity_id 保持为字符串（SurrealDB schema 定义为 option<string>）
                    await db.query(
                        "UPDATE atom SET entity_id = $entity_id WHERE id IN $atom_ids",
                        {"entity_id": record_id, "atom_ids": atom_record_ids}
                    )

                response_data = entity_data.copy()
                # 将 atoms 从 record id 列表转换为 AtomRef 列表
                response_data["atoms"] = atom_refs if atom_refs else []

                results["success"].append(EntityResponse(id=record_id, **response_data))

        except Exception as e:
            results["failed"].append({"index": i, "error": str(e)})

    return BatchEntityResponse(
        success=results["success"],
        failed=results["failed"],
        total=len(request.entities),
        success_count=len(results["success"]),
        failed_count=len(results["failed"])
    )


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

        # BL-B-32: 将字符串 ID 转换为 RecordID 对象
        if ":" in entity_id:
            parts = entity_id.split(":", 1)
            entity_record_id = RecordID(parts[0], parts[1])
        else:
            entity_record_id = entity_id

        
        if level == 0:
            
            result = await db.query(
                "SELECT id, type, abstract, tenant_id, created_at FROM entity WHERE id = $entity_id AND tenant_id = $tenant_id",
                {"entity_id": entity_record_id, "tenant_id": tenant_id}
            )
        elif level == 1:
            
            result = await db.query(
                "SELECT id, type, abstract, overview, tags, project, created_at FROM entity WHERE id = $entity_id AND tenant_id = $tenant_id",
                {"entity_id": entity_record_id, "tenant_id": tenant_id}
            )
        else:
            
            result = await db.query(
                "SELECT * FROM entity WHERE id = $entity_id AND tenant_id = $tenant_id",
                {"entity_id": entity_record_id, "tenant_id": tenant_id}
            )

        if not result or len(result) == 0:
            raise HTTPException(status_code=404, detail="Entity 不存在")

        record = result[0]

        # 处理 RecordID
        raw_id = record.get("id")
        if raw_id and hasattr(raw_id, "table_name"):
            record["id"] = f"{raw_id.table_name}:{raw_id.id}"

        # 处理 atoms 的 local_id
        if record.get("atoms"):
            # BL-FIX-003: 将字符串 ID 转换为 RecordID 对象用于查询
            atom_record_ids = []
            for a in record["atoms"]:
                if hasattr(a, "table_name"):
                    atom_record_ids.append(a)
                elif ":" in str(a):
                    table, id_part = str(a).split(":", 1)
                    atom_record_ids.append(RecordID(table, id_part))
                else:
                    atom_record_ids.append(a)
            
            # 查询这些 atoms 的 local_id
            atoms_result = await db.query(
                "SELECT id, local_id FROM atom WHERE id IN $atom_ids",
                {"atom_ids": atom_record_ids}
            )
            atom_refs = []
            if atoms_result:
                for atom_record in atoms_result:
                    atom_raw_id = atom_record.get("id")
                    if atom_raw_id and hasattr(atom_raw_id, "table_name"):
                        atom_id = f"{atom_raw_id.table_name}:{atom_raw_id.id}"
                    else:
                        atom_id = str(atom_raw_id)
                    atom_refs.append(AtomRef(id=atom_id, local_id=atom_record.get("local_id")))
            record["atoms"] = atom_refs

        # 处理 datetime
        for field in ["created_at", "updated_at"]:
            if field in record and record[field] is not None:
                if hasattr(record[field], "isoformat"):
                    record[field] = record[field].isoformat()

        return record

    except HTTPException:
        raise
    except Exception as e:
        logger.error("[Entity] 查询失败: %s", e)
        raise HTTPException(status_code=500, detail=f"查询失败: {e!s}") from e


@router.get("/entities", response_model=PaginatedEntityResponse)
async def list_entities(
    type: str | None = Query(default=None, description="Entity 类型过滤"),
    project: str | None = Query(default=None, description="项目过滤"),
    status: str | None = Query(default=None, description="状态过滤"),
    tenant_id: str = Query(default="default"),
    page: int = Query(default=1, ge=1, description="页码"),
    page_size: int = Query(default=50, ge=1, le=100, description="每页大小"),
    limit: int | None = Query(default=None, ge=1, le=100, description="返回数量限制（向后兼容）"),
    offset: int | None = Query(default=None, ge=0, description="偏移量（向后兼容）"),
):
    """列出 Entities（支持分页）"""
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
        count_query = "SELECT count() AS total FROM entity WHERE tenant_id = $tenant_id"
        count_params = {"tenant_id": tenant_id}

        if type:
            count_query += " AND type = $type"
            count_params["type"] = type
        if project:
            count_query += " AND project = $project"
            count_params["project"] = project
        if status:
            count_query += " AND status = $status"
            count_params["status"] = status

        count_result = await db.query(f"{count_query} GROUP ALL", count_params)
        count_records = state.memory_manager._extract_records(count_result)
        total = count_records[0].get("total", 0) if count_records else 0

        query = "SELECT * FROM entity WHERE tenant_id = $tenant_id"
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

        query += f" LIMIT {take} START {skip}"

        result = await db.query(query, params)
        raw_data = result or []

        # 批量收集所有 atom_ids，避免 N+1 查询
        all_atom_ids = []
        entity_atom_map = {}  # 记录每个 entity 的 atom 索引

        for idx, record in enumerate(raw_data):
            raw_id = record.get("id")
            if raw_id and hasattr(raw_id, "table_name"):
                record["id"] = f"{raw_id.table_name}:{raw_id.id}"
            if record.get("atoms"):
                atom_ids = [
                    f"{a.table_name}:{a.id}" if hasattr(a, "table_name") else str(a)
                    for a in record["atoms"]
                ]
                all_atom_ids.extend(atom_ids)
                entity_atom_map[idx] = atom_ids

        # 一次性查询所有 atoms 的 local_id
        atom_local_id_map = {}
        if all_atom_ids:
            atoms_result = await db.query(
                "SELECT id, local_id FROM atom WHERE id IN $atom_ids",
                {"atom_ids": all_atom_ids}
            )
            if atoms_result:
                for atom_record in atoms_result:
                    atom_raw_id = atom_record.get("id")
                    if atom_raw_id and hasattr(atom_raw_id, "table_name"):
                        atom_id = f"{atom_raw_id.table_name}:{atom_raw_id.id}"
                    else:
                        atom_id = str(atom_raw_id)
                    atom_local_id_map[atom_id] = atom_record.get("local_id")

        # 为每个 entity 设置 atoms
        data = []
        for idx, record in enumerate(raw_data):
            if idx in entity_atom_map:
                atom_refs = []
                for atom_id in entity_atom_map[idx]:
                    local_id = atom_local_id_map.get(atom_id)
                    atom_refs.append(AtomRef(id=atom_id, local_id=local_id))
                record["atoms"] = atom_refs
            for field in ["created_at", "updated_at"]:
                if field in record and record[field] is not None:
                    if hasattr(record[field], "isoformat"):
                        record[field] = record[field].isoformat()
            data.append(record)

        # 计算当前页码和 has_more
        current_page = page if limit is None else (skip // take) + 1
        current_page_size = take
        has_more = (skip + len(data)) < total

        return PaginatedEntityResponse(
            data=data,
            total=total,
            page=current_page,
            page_size=current_page_size,
            has_more=has_more
        )

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

        # BL-B-105: 将字符串 ID 转换为 RecordID 对象
        if ":" in entity_id:
            parts = entity_id.split(":", 1)
            entity_record_id = RecordID(parts[0], parts[1])
        else:
            entity_record_id = entity_id

        check = await db.query(
            "SELECT id FROM entity WHERE id = $entity_id",
            {"entity_id": entity_record_id}
        )
        if not check or len(check) == 0:
            raise HTTPException(status_code=404, detail="Entity 不存在")

        
        update_data: dict[str, Any] = {}
        raw_update = request.model_dump(exclude_unset=True)
        for field, value in raw_update.items():
            if value is not None:
                if field == "atoms":
                    pass
                else:
                    update_data[field] = value

        if not update_data and "atoms" not in raw_update:
            raise HTTPException(status_code=400, detail="没有要更新的字段")

        update_data["updated_at"] = "time::now()"

        # BL-B-100: 使用事务执行更新操作
        async with transaction(db, "Entity"):
            # atoms 在事务内处理，避免孤立 Atom
            if "atoms" in raw_update and raw_update["atoms"] is not None:
                atom_refs = await _process_atoms(db, raw_update["atoms"], entity_id=entity_id)
                # 存储时转换为 RecordID 对象列表
                update_data["atoms"] = [_parse_record_id(ref.id) for ref in atom_refs]

            # 使用 SurrealQL UPDATE 语句来更新字段
            set_clauses = []
            params: dict[str, Any] = {"entity_id": entity_record_id}
            for field, value in update_data.items():
                if field == "atoms":
                    # atoms 是 RecordID 对象列表
                    set_clauses.append("atoms = $atoms")
                    params["atoms"] = value
                elif field == "updated_at":
                    set_clauses.append(f"{field} = time::now()")
                elif isinstance(value, str):
                    set_clauses.append(f"{field} = ${field}")
                    params[field] = value
                elif isinstance(value, (list, dict)):
                    set_clauses.append(f"{field} = ${field}")
                    params[field] = value
                else:
                    set_clauses.append(f"{field} = ${field}")
                    params[field] = value

            # 获取 RecordID 的 ID 部分
            if isinstance(entity_record_id, RecordID):
                record_id_str = str(entity_record_id.id)
            else:
                record_id_str = str(entity_record_id)
            # nosec B608: record_id_str 来自已验证的 RecordID 对象，非用户输入
            query = f"UPDATE entity:{record_id_str} SET {', '.join(set_clauses)}"  # nosec B608
            await db.query(query, params)

            # 重新查询获取完整 Entity 数据
            updated = await db.query(
                "SELECT * FROM entity WHERE id = $entity_id",
                {"entity_id": entity_record_id}
            )
            if updated and len(updated) > 0:
                record = updated[0]
                # 处理 RecordID
                raw_id = record.get("id")
                if raw_id and hasattr(raw_id, "table_name"):
                    record["id"] = f"{raw_id.table_name}:{raw_id.id}"
                # 处理 atoms 中的 RecordID - 需要查询 local_id
                if record.get("atoms"):
                    atom_ids = [
                        f"{a.table_name}:{a.id}" if hasattr(a, "table_name") else str(a)
                        for a in record["atoms"]
                    ]
                    # 查询这些 atoms 的 local_id
                    atoms_result = await db.query(
                        "SELECT id, local_id FROM atom WHERE id IN $atom_ids",
                        {"atom_ids": atom_ids}
                    )
                    atom_refs = []
                    if atoms_result:
                        for atom_record in atoms_result:
                            atom_raw_id = atom_record.get("id")
                            if atom_raw_id and hasattr(atom_raw_id, "table_name"):
                                atom_id = f"{atom_raw_id.table_name}:{atom_raw_id.id}"
                            else:
                                atom_id = str(atom_raw_id)
                            atom_refs.append(AtomRef(id=atom_id, local_id=atom_record.get("local_id")))
                    record["atoms"] = atom_refs
                # 处理 datetime
                for field in ["created_at", "updated_at"]:
                    if field in record and record[field] is not None:
                        if hasattr(record[field], "isoformat"):
                            record[field] = record[field].isoformat()
                return record
            else:
                raise HTTPException(status_code=500, detail="更新后查询失败")

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

        # 将字符串 ID 转换为 RecordID 对象
        if ":" in entity_id:
            parts = entity_id.split(":", 1)
            entity_record_id = RecordID(parts[0], parts[1])
        else:
            entity_record_id = entity_id

        check = await db.query(
            "SELECT id FROM entity WHERE id = $entity_id AND tenant_id = $tenant_id",
            {"entity_id": entity_record_id, "tenant_id": tenant_id}
        )
        if not check or len(check) == 0:
            raise HTTPException(status_code=404, detail="Entity 不存在")

        # BL-B-100: 使用事务执行删除操作
        async with transaction(db, "Entity"):
            await db.delete(entity_record_id)
            return {"success": True, "message": "Entity 已删除"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("[Entity] 删除失败: %s", e)
        raise HTTPException(status_code=500, detail=f"删除失败: {e!s}") from e
