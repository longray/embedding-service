"""PrecomputeService 单例池管理

提供按 tenant 缓存的 PrecomputeService 单例管理。
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 按 tenant 缓存的 PrecomputeService 实例
_precompute_services: dict[str, Any] = {}


async def get_precompute_service(tenant_id: str, db: Any) -> Any:
    """获取或创建 PrecomputeService 单例（按 tenant 缓存）"""
    from ..services.precompute import PrecomputeService

    if tenant_id not in _precompute_services:
        service = PrecomputeService(
            db=db,
            tenant_id=tenant_id,
            max_concurrent=5,
            timeout_seconds=30.0,
        )
        await service.start()
        _precompute_services[tenant_id] = service
        logger.info("[PrecomputeService] 创建单例: tenant_id=%s", tenant_id)

    return _precompute_services[tenant_id]


async def close_precompute_services():
    """关闭所有 PrecomputeService 单例"""
    for tenant_id, service in list(_precompute_services.items()):
        try:
            await service.stop()
            logger.info("[PrecomputeService] 停止: tenant_id=%s", tenant_id)
        except Exception as e:
            logger.error("[PrecomputeService] 停止失败: tenant_id=%s, error=%s", tenant_id, e)
    _precompute_services.clear()
