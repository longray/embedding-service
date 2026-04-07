"""搜索端点"""

from fastapi import APIRouter, HTTPException

from .. import state
from ..models import MemorySearchRequest
from ..utils.exceptions import ValidationError

router = APIRouter(prefix="/api/v1", tags=["search"])


@router.post("/memories/search")
async def search_memories(request: MemorySearchRequest):
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        filter_expr = None
        if request.code_filter:
            filter_parts = []
            if "language" in request.code_filter:
                filter_parts.append(f'code_language = "{request.code_filter["language"]}"')
            if "min_complexity" in request.code_filter:
                filter_parts.append(f"code_complexity >= {request.code_filter['min_complexity']}")
            if "max_complexity" in request.code_filter:
                filter_parts.append(f"code_complexity <= {request.code_filter['max_complexity']}")
            # BL-CA-24: 新增代码过滤字段
            if "min_function_count" in request.code_filter:
                filter_parts.append(f"code_function_count >= {request.code_filter['min_function_count']}")
            if "max_function_count" in request.code_filter:
                filter_parts.append(f"code_function_count <= {request.code_filter['max_function_count']}")
            if "min_class_count" in request.code_filter:
                filter_parts.append(f"code_class_count >= {request.code_filter['min_class_count']}")
            if "max_class_count" in request.code_filter:
                filter_parts.append(f"code_class_count <= {request.code_filter['max_class_count']}")
            if "has_exports" in request.code_filter:
                filter_parts.append(f"code_has_exports = {request.code_filter['has_exports']}")
            if "analyzer" in request.code_filter:
                filter_parts.append(f'code_analyzer = "{request.code_filter["analyzer"]}"')
            if filter_parts:
                filter_expr = " AND ".join(filter_parts)

        result = await state.memory_manager.search_memories(
            query=request.query,
            mode=request.mode,
            limit=request.limit,
            threshold=request.threshold,
            level=request.level,
            tenant_id=request.tenant_id,
            filters=filter_expr,
        )
        return result
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {e!s}") from e
