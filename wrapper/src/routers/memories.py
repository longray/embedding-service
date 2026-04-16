"""记忆 CRUD 端点"""

import logging

from fastapi import APIRouter, HTTPException, Request
from surrealdb import AsyncSurreal  # type: ignore[import-untyped]

from .. import state
from ..config import config
from ..models import AccessLogRequest, MemoryUploadRequest
from ..utils.exceptions import ValidationError
from ..utils.meili_client import MeilisearchClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["memories"])


@router.post("/memories")
async def upload_memories(request: MemoryUploadRequest):
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        memories = [m.model_dump() for m in request.memories]
        result = await state.memory_manager.upload_memories(
            memories,
            tenant_id=request.tenant_id,
        )

        if request.auto_analyze_code and config.code_analysis.enabled:
            for memory in memories:
                content = memory.get("content", "")
                content_length = len(content)
                if (
                    content_length >= config.code_analysis.min_content_length
                    and content_length <= config.code_analysis.max_content_length
                ):
                    try:
                        await state.memory_manager.analyze_memory_code(
                            memory.get("id", memory.get("source_id", "")),
                            tenant_id=request.tenant_id,
                            persist=True,
                        )
                    except Exception as analyze_error:
                        logger.warning("[Auto Analyze] 分析失败: %s", analyze_error)

        return result
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {e!s}") from e


@router.delete("/memories/clear")
async def clear_memories(request: Request):
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    api_key = request.headers.get("WRAPPER_MEILI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing WRAPPER_MEILI_API_KEY header")

    if api_key != config.meilisearch.api_key:
        raise HTTPException(status_code=403, detail="Invalid WRAPPER_MEILI_API_KEY")

    client = state.meili_client
    if not client:
        client = MeilisearchClient(
            url=config.meilisearch.url,
            api_key=config.meilisearch.api_key,
            index_name=config.meilisearch.index_name,
            timeout=config.meilisearch.timeout,
        )
        await client.connect()

    try:
        logger.warning("[Clear] 清空 Meilisearch...")

        await client.delete_all_documents()
        logger.info("[Clear] Meilisearch 已清空")

        logger.warning("[Clear] 清空 SurrealDB...")
        await state.memory_manager._db.query("DELETE memory;")
        await state.memory_manager._db.query("DELETE memory_relation;")
        await state.memory_manager._db.query("DELETE conflict;")

        logger.info("[Clear] SurrealDB 已清空")

        return {"success": True, "message": "所有记忆数据已清空"}
    except Exception as e:
        logger.error(f"[Clear] 清空失败: {e}")
        raise HTTPException(status_code=500, detail=f"清空失败: {e!s}") from e


@router.get("/memories/{memory_id}")
async def get_memory(
    memory_id: str,
    tenant_id: str = "default",
    include_embedding: bool = False,
):
    """获取单个记忆详情

    Args:
        memory_id: 记忆ID
        tenant_id: 租户ID
        include_embedding: 是否包含embedding向量（默认false，减少响应体积）
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        # 根据参数选择查询字段
        if include_embedding:
            query = "SELECT * FROM type::record($memory_id) WHERE tenant_id = $tenant_id"
        else:
            query = "SELECT id, content, abstract, overview, type, metadata, project_id, tags, source, content_hash, local_id, created_at, mtime FROM type::record($memory_id) WHERE tenant_id = $tenant_id"

        result = await state.memory_manager._db_query(
            query,
            {"memory_id": memory_id, "tenant_id": tenant_id},
        )
        records = state.memory_manager._extract_records(result)

        if not records:
            raise HTTPException(status_code=404, detail="记忆不存在")

        return {"status": "success", "memory": records[0]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {e!s}") from e


@router.get("/memories/{memory_id}/summary")
async def get_memory_summary(memory_id: str, tenant_id: str = "default"):
    """获取记忆的代码摘要"""
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        # 查询记忆的 metadata
        db = AsyncSurreal(config.surrealdb.url)
        await db.signin({"user": config.surrealdb.username, "pass": config.surrealdb.password})
        await db.use(config.surrealdb.namespace, config.surrealdb.database)

        query = "SELECT metadata FROM type::record($memory_id) WHERE tenant_id = $tenant_id"
        result = await db.query(query, {"memory_id": memory_id, "tenant_id": tenant_id})

        if not result or not result[0]["result"]:  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]
            raise HTTPException(status_code=404, detail="记忆不存在")

        raw_result = result[0]["result"][0]  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]
        metadata = raw_result.get("metadata", {})  # type: ignore[union-attr]
        code_summary = metadata.get("code_summary")  # type: ignore[union-attr]

        await db.close()

        if code_summary:
            return {"status": "success", "summary": code_summary}
        else:
            return {"status": "not_found", "message": "该记忆没有代码摘要"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取摘要失败: {e!s}") from e


@router.post("/memories/{memory_id}/enrich/llm")
async def enrich_memory_llm(memory_id: str, tenant_id: str = "default"):
    """手动触发 LLM 代码摘要生成"""
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await state.memory_manager._generate_code_summary(memory_id, tenant_id)
        if result:
            return {"status": "success", "summary": result}
        else:
            return {"status": "skipped", "message": "无法生成摘要（可能不是代码或LLM未启用）"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM摘要生成失败: {e!s}") from e


@router.post("/access-log")
async def report_access_log(request: AccessLogRequest):
    """接收访问日志（用于分析记忆使用频率）"""
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await state.memory_manager.report_access_log(
            [entry.model_dump() for entry in request.entries],
            tenant_id=request.tenant_id,
        )
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"记录访问日志失败: {e!s}") from e


@router.get("/projects/{project_id}/stats")
async def get_project_stats(project_id: str, tenant_id: str = "default"):
    """获取项目代码统计信息 (BL-CA-25)

    返回项目中的代码文件统计：
    - total_files: 代码文件总数
    - total_functions: 函数总数
    - total_classes: 类总数
    - avg_complexity: 平均圈复杂度
    - max_complexity: 最大圈复杂度
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await state.memory_manager.get_project_stats(
            project_id=project_id,
            tenant_id=tenant_id,
        )
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message", "获取项目统计失败"))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取项目统计失败: {e!s}") from e


@router.get("/projects/{project_id}/map")
async def get_project_map(project_id: str, tenant_id: str = "default"):
    """获取项目代码地图 (BL-CA-23)

    返回项目的完整代码地图：
    - file_tree: 文件树结构
    - module_dependencies: 模块依赖关系
    - hot_files: 热点文件（复杂度最高）
    - statistics: 统计信息
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await state.memory_manager.get_project_map(
            project_id=project_id,
            tenant_id=tenant_id,
        )
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message", "获取项目地图失败"))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取项目地图失败: {e!s}") from e
