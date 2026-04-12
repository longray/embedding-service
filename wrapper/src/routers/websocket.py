"""WebSocket 实时推送端点 (Phase 3D + v3.2 心跳机制)"""

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..utils.auth import verify_websocket_token
from ..websocket.reliable_server import ReliableWebSocketServer

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/memories/live")
async def websocket_live_memories(
    websocket: WebSocket,
    tenant_id: str = "default",
    token: str | None = None,
):
    """WebSocket 端点：实时推送记忆变更通知（带心跳保活）

    连接后自动订阅指定租户的 memory 表变更，推送 CREATE/UPDATE/DELETE 通知。
    内置心跳机制：每 30s 发送 ping，5s 内等待 pong，连续 2 次未响应断开连接。

    认证：通过 token 查询参数传递（可选，取决于 WRAPPER_WEBSOCKET_TOKEN 配置）。
    """
    # 认证检查
    if not verify_websocket_token(token):
        await websocket.close(code=1008, reason="Unauthorized")
        logger.warning("[WebSocket] 认证失败，拒绝连接")
        return

    # 包装为可靠 WebSocket 服务器（带心跳）
    reliable_ws = ReliableWebSocketServer(
        websocket=websocket,
        heartbeat_interval=30.0,
        heartbeat_timeout=5.0,
        max_missing_pongs=2,
    )

    query_uuid = None

    try:
        from ..main import SurrealDBManager

        db_manager = await SurrealDBManager.get_instance()

        # 接受连接并启动心跳
        await reliable_ws.accept()

        # 启动 LIVE SELECT 查询（过滤租户）
        query_result = await db_manager.db.query(
            "LIVE SELECT * FROM memory WHERE tenant_id = $tenant_id",
            {"tenant_id": tenant_id},
        )
        query_uuid = query_result[0]["result"]

        logger.info(
            "[WebSocket] 客户端已连接，订阅租户: %s, query_uuid: %s",
            tenant_id,
            query_uuid,
        )

        # 创建任务：转发 LIVE 通知到客户端
        forward_task = asyncio.create_task(
            _forward_notifications(db_manager, query_uuid, reliable_ws),
            name="websocket_forward",
        )

        # 等待连接断开（心跳超时或客户端断开）
        await reliable_ws.wait_for_disconnect()

        # 取消转发任务
        forward_task.cancel()
        try:
            await forward_task
        except asyncio.CancelledError:
            pass

    except WebSocketDisconnect:
        logger.info("[WebSocket] 客户端断开连接")
    except Exception as e:
        logger.error("[WebSocket] 错误: %s", e)
        try:
            await reliable_ws.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        # 清理
        if query_uuid:
            try:
                await db_manager.db.kill(query_uuid)
                logger.info("[WebSocket] 已停止 LIVE 查询: %s", query_uuid)
            except Exception:
                pass

        # 关闭可靠 WebSocket（如果还未关闭）
        if reliable_ws.is_connected:
            await reliable_ws.close()


async def _forward_notifications(db_manager, query_uuid, reliable_ws):
    """转发 LIVE SELECT 通知到 WebSocket 客户端"""
    try:
        async for notification in db_manager.db.subscribe_live(query_uuid):
            if not reliable_ws.is_connected:
                break
            await reliable_ws.send_json(notification)
    except asyncio.CancelledError:
        logger.debug("[WebSocket] 转发任务已取消")
        raise
    except Exception as e:
        logger.error("[WebSocket] 转发通知错误: %s", e)
