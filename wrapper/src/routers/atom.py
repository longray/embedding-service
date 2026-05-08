"""Atom CRUD 端点 - 原子级知识单元管理"""

import logging
import re
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
    local_id: str | None = Field(default=None, description="客户端侧 ID (用于树结构)")
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


class AtomBudgetRequest(BaseModel):
    """Atom 上下文预算请求"""

    entity_id: str = Field(..., description="Entity ID")
    query: str | None = Field(default=None, description="搜索关键词（relevance 策略用）")
    max_tokens: int = Field(default=4000, ge=100, le=100000, description="Token 预算上限")
    strategy: str = Field(default="relevance", description="选择策略: relevance | hierarchy")
    max_level: int | None = Field(default=None, ge=1, le=6, description="最大标题层级过滤")
    tenant_id: str = Field(default="default", description="租户ID")


class AtomBudgetResponse(BaseModel):
    """Atom 上下文预算响应"""

    atoms: list[dict[str, Any]] = Field(..., description="选中的 Atom 列表（按原始树结构顺序）")
    total_atoms: int = Field(..., description="Entity 总 Atom 数")
    selected_count: int = Field(..., description="选中数量")
    used_tokens: int = Field(..., description="估算使用的 token 数")
    max_tokens: int = Field(..., description="预算上限")
    strategy: str = Field(..., description="使用的策略")
    budget_exhausted: bool = Field(..., description="预算是否耗尽")


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
            
            # Phase 2: 同步到 Meilisearch
            await _sync_atom_to_meili(record_id, atom_data, request.tenant_id)
            
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
    max_level: int | None = Query(default=None, ge=1, le=6, description="最大标题层级过滤（仅返回 heading_level <= max_level 的 Atom）"),
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
        if max_level is not None:
            conditions.append("(heading_level IS NONE OR heading_level <= $max_level)")
            params["max_level"] = max_level

        where_clause = " AND ".join(conditions)

        count_result = await db.query(
            f"SELECT count() AS total FROM atom WHERE {where_clause} GROUP ALL",  # nosec B608
            params,
        )
        records = state.memory_manager._extract_records(count_result)
        total = records[0].get("total", 0) if records else 0

        result = await db.query(
            f"SELECT * FROM atom WHERE {where_clause} ORDER BY created_at DESC LIMIT {take} START {skip}",  # nosec B608
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


@router.get("/entities/{entity_id}/atoms/{atom_id}")
async def get_atom_by_entity(
    entity_id: str,
    atom_id: str,
    tenant_id: str = Query(default="default"),
):
    """
    获取属于指定 Entity 的 Atom 详情

    验证 atom 的 entity_id 字段与路径参数匹配，返回完整 atom 数据。
    用于跨 Entity 的 Atom 链接解析。
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        db = state.memory_manager.db

        # 将字符串 ID 转换为 RecordID
        atom_parts = atom_id.split(":", 1)
        atom_record_id = RecordID(atom_parts[0], atom_parts[1])

        result = await db.query(
            "SELECT * FROM atom WHERE id = $atom_id AND entity_id = $entity_id AND tenant_id = $tenant_id",
            {"atom_id": atom_record_id, "entity_id": entity_id, "tenant_id": tenant_id},
        )

        if not result or len(result) == 0:
            raise HTTPException(status_code=404, detail="Atom 不存在或不属于该 Entity")

        return result[0]

    except HTTPException:
        raise
    except Exception as e:
        logger.error("[Atom] 按Entity查询失败: entity_id=%s atom_id=%s error=%s", entity_id, atom_id, e)
        raise HTTPException(status_code=500, detail=f"查询失败: {e!s}") from e


@router.post("/atoms/budget", response_model=AtomBudgetResponse)
async def atoms_budget(request: AtomBudgetRequest):
    """在 token 预算内选择最相关的 Atoms"""
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        db = state.memory_manager.db

        conditions = ["entity_id = $entity_id", "tenant_id = $tenant_id"]
        params: dict[str, Any] = {
            "entity_id": request.entity_id,
            "tenant_id": request.tenant_id,
        }

        if request.max_level is not None:
            conditions.append("(heading_level IS NONE OR heading_level <= $max_level)")
            params["max_level"] = request.max_level

        where_clause = " AND ".join(conditions)
        result = await db.query(
            f"SELECT * FROM atom WHERE {where_clause}",  # nosec B608
            params,
        )

        raw_atoms: list[dict[str, Any]] = []
        if result:
            for record in result:
                atom = _normalize_atom(record)
                if atom:
                    raw_atoms.append(atom)

        total_atoms = len(raw_atoms)

        if total_atoms == 0:
            return AtomBudgetResponse(
                atoms=[],
                total_atoms=0,
                selected_count=0,
                used_tokens=0,
                max_tokens=request.max_tokens,
                strategy=request.strategy,
                budget_exhausted=False,
            )

        strategy = request.strategy
        if strategy == "relevance" and not request.query:
            logger.warning("[Atom Budget] relevance 策略缺少 query，回退到 hierarchy")
            strategy = "hierarchy"

        if strategy == "relevance":
            scored = _score_atoms_relevance(raw_atoms, request.query)
        else:
            scored = _sort_atoms_hierarchy(raw_atoms)

        selected = _greedy_select(scored, raw_atoms, request.max_tokens)

        # 按树结构重排: order → heading_level → id
        selected.sort(key=lambda a: (
            a.get("order") or "",
            a.get("heading_level") or 999,
            a.get("id") or "",
        ))

        used_tokens = sum(_estimate_atom_tokens(a) for a in selected)

        return AtomBudgetResponse(
            atoms=selected,
            total_atoms=total_atoms,
            selected_count=len(selected),
            used_tokens=used_tokens,
            max_tokens=request.max_tokens,
            strategy=strategy,
            budget_exhausted=len(selected) < total_atoms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("[Atom Budget] 预算选择失败: %s", e)
        raise HTTPException(status_code=500, detail=f"预算选择失败: {e!s}") from e


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 2)


def _estimate_atom_tokens(atom: dict[str, Any]) -> int:
    parts = [atom.get("name", "") or "", atom.get("content", "") or ""]
    sig = atom.get("signature")
    if sig:
        parts.append(sig)
    return _estimate_tokens(" ".join(parts))


def _simple_bm25(query: str, text: str, avg_doc_len: float = 100.0) -> float:
    if not query or not text:
        return 0.0

    k1, b = 1.5, 0.75
    query_terms = set(re.findall(r"\w+", query.lower()))
    doc_terms = re.findall(r"\w+", text.lower())
    doc_len = len(doc_terms)

    tf: dict[str, int] = {}
    for term in doc_terms:
        tf[term] = tf.get(term, 0) + 1

    score = 0.0
    for term in query_terms:
        if term in tf:
            freq = tf[term]
            numerator = freq * (k1 + 1)
            denominator = freq + k1 * (1 - b + b * doc_len / avg_doc_len)
            score += numerator / denominator

    return score


def _normalize_atom(record: dict[str, Any]) -> dict[str, Any] | None:
    raw_id = record.get("id")
    if raw_id and hasattr(raw_id, "table_name"):
        record["id"] = f"{raw_id.table_name}:{raw_id.id}"
    for field in ("created_at", "updated_at"):
        if field in record and record[field] is not None:
            if hasattr(record[field], "isoformat"):
                record[field] = record[field].isoformat()
    return record


def _score_atoms_relevance(
    atoms: list[dict[str, Any]], query: str | None
) -> list[dict[str, Any]]:
    if not query:
        return atoms
    for atom in atoms:
        text = (atom.get("name") or "") + " " + (atom.get("content") or "")
        atom["_score"] = _simple_bm25(query, text)
        hl = atom.get("heading_level")
        if hl is not None:
            atom["_score"] += (7 - hl) * 0.1
    return sorted(atoms, key=lambda a: a.get("_score", 0.0), reverse=True)


def _sort_atoms_hierarchy(atoms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        atoms,
        key=lambda a: (
            a.get("heading_level") if a.get("heading_level") is not None else 999,
            a.get("order") or "",
            a.get("name") or "",
        ),
    )


def _greedy_select(
    ranked: list[dict[str, Any]], all_atoms: list[dict[str, Any]], max_tokens: int
) -> list[dict[str, Any]]:
    atom_map: dict[str, dict[str, Any]] = {}
    for a in all_atoms:
        aid = a.get("id")
        if aid:
            atom_map[aid] = a

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    used_tokens = 0

    for candidate in ranked:
        cid = candidate.get("id")
        if cid and cid in selected_ids:
            continue

        cand_tokens = _estimate_atom_tokens(candidate)

        # Resolve full ancestor chain (parent → grandparent → ...)
        ancestors: list[dict[str, Any]] = []
        ancestor_ids: set[str] = set()
        current_id = candidate.get("parent_id")
        while current_id and current_id not in selected_ids and current_id not in ancestor_ids and current_id in atom_map:
            ancestor = atom_map[current_id]
            ancestors.append(ancestor)
            ancestor_ids.add(current_id)
            current_id = ancestor.get("parent_id")

        if ancestors:
            ancestor_tokens = sum(_estimate_atom_tokens(a) for a in ancestors)
            if used_tokens + ancestor_tokens + cand_tokens > max_tokens and selected:
                continue
            # Add ancestors (deepest first, so parent comes before grandchild)
            for ancestor in reversed(ancestors):
                aid = ancestor.get("id")
                if aid and aid not in selected_ids:
                    selected.append(ancestor)
                    selected_ids.add(aid)
                    used_tokens += _estimate_atom_tokens(ancestor)

        if used_tokens + cand_tokens > max_tokens and selected:
            continue

        selected.append(candidate)
        if cid:
            selected_ids.add(cid)
        used_tokens += cand_tokens

    # 至少返回 1 个 atom
    if not selected and ranked:
        first = ranked[0].copy()
        first.pop("_score", None)
        selected.append(first)

    # 移除内部评分字段
    for a in selected:
        a.pop("_score", None)

    return selected


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
                    
                    # Phase 2: 同步到 Meilisearch
                    await _sync_atom_to_meili(record_id, atom_data, request.tenant_id)

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

            # Phase 2: 同步更新到 Meilisearch（查询完整记录）
            full_result = await db.query(
                "SELECT * FROM atom WHERE id = $atom_id AND tenant_id = $tenant_id",
                {"atom_id": atom_record_id, "tenant_id": tenant_id}
            )
            if full_result and len(full_result) > 0:
                full_record = parse_surrealdb_result(full_result)
                if full_record:
                    await _sync_atom_to_meili(atom_id, full_record, tenant_id)

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
            
            # Phase 2: 从 Meilisearch 删除
            await _delete_atom_from_meili(atom_id)
            
            return {"success": True, "message": "Atom 已删除"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("[Atom] 删除失败: %s", e)
        raise HTTPException(status_code=500, detail=f"删除失败: {e!s}") from e


async def _sync_atom_to_meili(atom_id: str, atom_data: dict[str, Any], tenant_id: str) -> None:
    """同步 Atom 到 Meilisearch（Phase 2: 双写同步）
    
    Args:
        atom_id: Atom ID (e.g., "atom:xxx")
        atom_data: Atom 数据字典
        tenant_id: 租户 ID
    """
    if not state.memory_manager:
        return
    
    meili = state.memory_manager._meili
    if not meili:
        return
    
    try:
        # 使用 MeiliSyncMixin._build_meili_doc 构建文档
        meili_doc = state.memory_manager._build_meili_doc(
            atom_id, atom_data, tenant_id, doc_type="atom"
        )
        await meili.add_documents([meili_doc])
        logger.debug("[Atom] 同步到 Meilisearch: %s", atom_id)
    except Exception as e:
        # 错误处理：记录日志但不阻塞主流程
        logger.warning("[Atom] Meilisearch 同步失败 %s: %s", atom_id, e)


async def _delete_atom_from_meili(atom_id: str) -> None:
    """从 Meilisearch 删除 Atom（Phase 2: 双写同步）
    
    Args:
        atom_id: Atom ID (e.g., "atom:xxx")
    """
    if not state.memory_manager:
        return
    
    meili = state.memory_manager._meili
    if not meili:
        return
    
    try:
        # meili_client.delete_document 内部会自动转换 ID 格式
        await meili.delete_document(atom_id)
        logger.debug("[Atom] 从 Meilisearch 删除: %s", atom_id)
    except Exception as e:
        # 错误处理：记录日志但不阻塞主流程
        logger.warning("[Atom] Meilisearch 删除失败 %s: %s", atom_id, e)
