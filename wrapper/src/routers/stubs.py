"""Stub 端点 — 返回 NotImplementedError

保留 API 路由兼容性，功能待实现。
"""

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["stubs"])

# 所有 stub 端点已迁移到独立 router
# - /prefetch/related -> routers/prefetch.py
# - /prefetch/popular -> routers/prefetch.py
# - /memories/cluster/leiden -> routers/clustering.py
