"""WebSocket 实时推送端点 (Phase 3D)"""

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..utils.auth import verify_websocket_token

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/memories/live")
async def websocket_live_memories(websocket: WebSocket, tenant_id: str = "default", token: str | None = None):
    """WebSocket 端点：实时推送记忆变更通知

    连接后自动订阅指定租户的 memory 表变更，推送 CREATE/UPDATE/DELETE 通知。
    认证：通过 token 查询参数传递（可选，取决于 WRAPPER_WEBSOCKET_TOKEN 配置）。
    """
    # 认证检查
    if not verify_websocket_token(token):
        await websocket.close(code=1008, reason="Unauthorized")
        logger.warning("[WebSocket] 认证失败，拒绝连接")
        return

    await websocket.accept()
    query_uuid = None

    try:
        from ..main import SurrealDBManager

        db_manager = await SurrealDBManager.get_instance()

        # 启动 LIVE SELECT 查询（过滤租户）
        query_result = await db_manager.db.query(
            "LIVE SELECT * FROM memory WHERE tenant_id = $tenant_id",
            {"tenant_id": tenant_id},
        )
        query_uuid = query_result[0]["result"]

        logger.info("[WebSocket] 客户端已连接，订阅租户: %s, query_uuid: %s", tenant_id, query_uuid)

        # 订阅并转发通知
        async for notification in db_manager.db.subscribe_live(query_uuid):
            await websocket.send_json(notification)

    except WebSocketDisconnect:
        logger.info("[WebSocket] 客户端断开连接")
    except Exception as e:
        logger.error("[WebSocket] 错误: %s", e)
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:  # nosec B110 - 发送错误失败时静默
            pass
    finally:
        # 清理 LIVE 查询
        if query_uuid:
            try:
                await db_manager.db.kill(query_uuid)
                logger.info("[WebSocket] 已停止 LIVE 查询: %s", query_uuid)
            except Exception:  # nosec B110 - kill 失败不影响断开
                pass
