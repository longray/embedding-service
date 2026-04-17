"""预计算端点 (BL-B-81)

提供代码预计算 API：
- 分析代码文件并上传结果
- 创建符号和调用关系
"""

from fastapi import APIRouter, HTTPException

from .. import state
from ..models import PrecomputeAnalysisRequest, PrecomputeAnalysisResponse
from ..services.precompute import PrecomputeService

router = APIRouter(prefix="/api/v1", tags=["precompute"])


@router.post("/precompute/analysis", response_model=PrecomputeAnalysisResponse)
async def precompute_analysis(request: PrecomputeAnalysisRequest):
    """代码分析预计算

    接收代码分析结果（文件、符号、调用关系），创建记忆条目。

    Args:
        request: 预计算分析请求，包含项目ID、文件、符号、调用关系

    Returns:
        创建的 memory_id 映射表
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        # 获取数据库连接
        db = state.memory_manager._db

        # 创建 PrecomputeService 实例
        service = PrecomputeService(
            db=db,
            tenant_id=request.tenant_id,
            max_concurrent=5,
            timeout_seconds=30.0,
        )

        # 启动服务
        await service.start()

        try:
            # 准备批次数据
            batch = []
            for file_info in request.files:
                batch.append(
                    {
                        "file_path": file_info.path,
                        "content": file_info.content,
                        "project_id": request.project_id,
                    }
                )

            # 处理批次
            result = await service.process_batch(batch)

            # 构建 memory_ids 映射
            memory_ids = {}
            for item in result.get("results", []):
                if "file_path" in item and "memory_id" in item:
                    memory_ids[item["file_path"]] = item["memory_id"]

            # 确定状态
            status = "success"
            if result.get("error_count", 0) > 0:
                if result.get("processed_count", 0) > 0:
                    status = "partial"
                else:
                    status = "failed"

            return PrecomputeAnalysisResponse(
                memory_ids=memory_ids,
                status=status,
                processed_count=result.get("processed_count", 0),
                failed_count=result.get("error_count", 0),
                errors=result.get("errors", []),
            )

        finally:
            # 停止服务
            await service.stop()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预计算分析失败: {e!s}") from e
