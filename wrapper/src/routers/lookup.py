"""Lookup router"""

from fastapi import APIRouter, HTTPException, Query

from .. import state

router = APIRouter(prefix="/api/v1", tags=["lookup"])


@router.get("/memories/lookup")
async def lookup_memory(
    source_id: str | None = Query(None, description="本地生成的 ULID"),
    hash: str | None = Query(None, description="内容哈希（32位十六进制）"),
    hash_algorithm: str = Query("md5", description="哈希算法"),
    file_path: str | None = Query(None, description="文件相对路径"),
    project_id: str | None = Query(None, description="项目ID"),
    type: str | None = Query(None, description="记忆类型过滤"),
    tenant_id: str = Query("default", description="租户ID"),
    limit: int = Query(1, ge=1, le=100, description="返回数量限制"),
    all: bool = Query(False, description="返回全部历史版本"),
):
    """查询记忆

    支持通过 source_id、file_path、hash 查询记忆，用于缓存重建和多设备同步。

    查询优先级：source_id > hash > file_path+project_id
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        # 确定查询策略
        records = []

        if source_id:
            # 优先级 1: source_id 查询
            records = await state.memory_manager.lookup_by_source_id(
                source_id=source_id,
                tenant_id=tenant_id,
                type_filter=type,
                limit=100 if all else limit,
            )
        elif hash:
            # 优先级 2: hash 查询
            records = await state.memory_manager.lookup_by_hash(
                content_hash=hash,
                tenant_id=tenant_id,
                type_filter=type,
                limit=100 if all else limit,
            )
        elif file_path and project_id:
            # 优先级 3: file_path + project_id 查询
            records = await state.memory_manager.lookup_by_file_path(
                file_path=file_path,
                project_id=project_id,
                tenant_id=tenant_id,
                type_filter=type,
                limit=100 if all else limit,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="参数不足。请提供 source_id、hash 或 (file_path + project_id) 之一",
            )

        if not records:
            return {
                "found": False,
                "message": "未找到匹配的记忆",
            }

        # 构建响应
        if limit == 1 and not all:
            # 单条响应
            record = records[0]
            return {
                "found": True,
                "memory_id": str(record.get("id", "")),
                "source_id": record.get("source_id"),
                "file_path": record.get("metadata", {}).get("file_path"),
                "project_id": record.get("project_id"),
                "type": record.get("type"),
                "content_hash": record.get("content_hash"),
                "created_at": record.get("created_at"),
                "updated_at": record.get("updated_at"),
            }
        else:
            # 多条响应
            return {
                "found": True,
                "count": len(records),
                "memories": [
                    {
                        "memory_id": str(r.get("id", "")),
                        "source_id": r.get("source_id"),
                        "file_path": r.get("metadata", {}).get("file_path"),
                        "created_at": r.get("created_at"),
                    }
                    for r in records
                ],
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {e}") from e
