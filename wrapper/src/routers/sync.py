"""同步端点 (Phase B)"""

import logging

from fastapi import APIRouter, HTTPException

from .. import state
from ..models import (
    CodeFingerprintRequest,
    CodeFingerprintResponse,
    ConflictResolutionRequest,
    SyncFullRequest,
    SyncFullResponse,
    SyncIncrementalRequest,
    SyncIncrementalResponse,
    SyncPreviewRequest,
    SyncPreviewResponse,
)
from ..services.code_fingerprint_service import CodeFingerprintService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["sync"])


@router.post("/sync/preview", response_model=SyncPreviewResponse)
async def sync_preview(request: SyncPreviewRequest):
    """同步预览：比对指纹，返回变更指令（不执行上传）"""
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await state.memory_manager.sync_preview(
            fingerprints=[f.model_dump() for f in request.fingerprints],
            tenant_id=request.tenant_id,
        )
        return SyncPreviewResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"同步预览失败: {e!s}") from e


@router.post("/sync/incremental", response_model=SyncIncrementalResponse)
async def sync_incremental_legacy(request: SyncIncrementalRequest):
    """增量同步（已弃用，请使用 /api/v1/sync/preview）"""
    return await sync_preview(SyncPreviewRequest(**request.model_dump()))


@router.post("/sync/full", response_model=SyncFullResponse)
async def sync_full(request: SyncFullRequest):
    """全量同步：上传所有记忆"""
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await state.memory_manager.sync_full(
            memories=[m.model_dump() for m in request.memories],
            tenant_id=request.tenant_id,
        )
        return SyncFullResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"全量同步失败: {e!s}") from e


@router.get("/sync/fingerprints")
async def get_server_fingerprints(tenant_id: str = "default"):
    """获取服务端所有记忆的指纹"""
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        fingerprints = await state.memory_manager.get_fingerprints(tenant_id=tenant_id)
        return {"fingerprints": fingerprints, "count": len(fingerprints)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取指纹失败: {e!s}") from e


@router.post("/sync/conflicts/{conflict_id}/resolve")
async def resolve_conflict_endpoint(conflict_id: str, request: ConflictResolutionRequest):
    """解决同步冲突"""
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await state.memory_manager.resolve_conflict(
            conflict_id=conflict_id,
            resolution=request.resolution,
            tenant_id=request.tenant_id,
        )
        return {"resolved": True, "action": request.resolution, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解决冲突失败: {e!s}") from e


# ==================== Code Fingerprint Sync (BL-B-80) ====================


@router.post("/sync/code-fingerprints", response_model=CodeFingerprintResponse)
async def sync_code_fingerprints(request: CodeFingerprintRequest):
    """代码指纹增量同步：比对文件指纹，返回变更文件列表

    使用 SurrealDB 事务确保数据一致性：
    - 所有更新和删除操作在事务中执行
    - 失败时自动回滚
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    db = state.memory_manager.db
    service = CodeFingerprintService(db)

    try:
        # 比对指纹（在事务外执行，因为是只读操作）
        result = await service.compare_fingerprints(
            fingerprints=[f.model_dump() for f in request.fingerprints],
            tenant_id=request.tenant_id,
            project_id=request.project_id,
        )

        # 准备更新和删除的数据
        files_to_update = result["changed_files"] + result["new_files"]
        fingerprints_to_update = [f.model_dump() for f in request.fingerprints if f.file in files_to_update]
        files_to_delete = result["deleted_files"]

        # 如果没有需要修改的数据，直接返回
        if not fingerprints_to_update and not files_to_delete:
            return CodeFingerprintResponse(**result)

        # 使用事务执行写操作
        try:
            # 开始事务
            await db.query("BEGIN TRANSACTION")

            # 更新指纹
            if fingerprints_to_update:
                await service.update_fingerprints(
                    fingerprints=fingerprints_to_update,
                    tenant_id=request.tenant_id,
                    project_id=request.project_id,
                )

            # 删除指纹
            if files_to_delete:
                await service.delete_fingerprints(
                    file_paths=files_to_delete,
                    tenant_id=request.tenant_id,
                    project_id=request.project_id,
                )

            # 提交事务
            await db.query("COMMIT TRANSACTION")

        except Exception as tx_error:
            # 回滚事务
            try:
                await db.query("CANCEL TRANSACTION")
            except Exception as cancel_error:
                logger.error("[Sync] 事务回滚失败: %s", cancel_error)
            raise tx_error

        return CodeFingerprintResponse(**result)

    except Exception as e:
        logger.error("[Sync] 指纹同步失败: %s", e)
        raise HTTPException(status_code=500, detail=f"指纹同步失败: {e!s}") from e
