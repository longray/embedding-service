"""同步端点 (Phase B)"""

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
    """代码指纹增量同步：比对文件指纹，返回变更文件列表"""
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        # 通过 memory_manager 获取 SurrealDB 连接
        db = state.memory_manager._db
        service = CodeFingerprintService(db)

        # 比对指纹
        result = await service.compare_fingerprints(
            fingerprints=[f.model_dump() for f in request.fingerprints],
            tenant_id=request.tenant_id,
            project_id=request.project_id,
        )

        # 更新数据库中的指纹（只更新变更和新增的文件）
        files_to_update = result["changed_files"] + result["new_files"]
        fingerprints_to_update = [f.model_dump() for f in request.fingerprints if f.file in files_to_update]
        if fingerprints_to_update:
            await service.update_fingerprints(
                fingerprints=fingerprints_to_update,
                tenant_id=request.tenant_id,
                project_id=request.project_id,
            )

        # 删除已删除文件的指纹
        if result["deleted_files"]:
            await service.delete_fingerprints(
                file_paths=result["deleted_files"],
                tenant_id=request.tenant_id,
                project_id=request.project_id,
            )

        return CodeFingerprintResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"指纹同步失败: {e!s}") from e
