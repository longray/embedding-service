"""WebSocket 实时推送端点 (Phase 3D + v3.2 心跳机制 + DIFF 模式)"""

import asyncio
import logging
from typing import Literal

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..utils.auth import verify_websocket_token
from ..websocket.live_diff_handler import LiveDiffHandler
from ..websocket.reliable_server import ReliableWebSocketServer

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/memories/live")
async def websocket_live_memories(
    websocket: WebSocket,
    tenant_id: str = "default",
    token: str | None = None,
    mode: Literal["diff", "full"] = "full",
):
    """WebSocket 端点：实时推送记忆变更通知（带心跳保活 + DIFF 模式）

    连接后自动订阅指定租户的 memory 表变更，推送 CREATE/UPDATE/DELETE 通知。
    内置心跳机制：每 30s 发送 ping，5s 内等待 pong，连续 2 次未响应断开连接。

    支持 DIFF 模式：通过 mode 参数选择 diff（增量）或 full（完整）模式。

    参数:
        tenant_id: 租户 ID，默认 default
        token: 认证 token（可选）
        mode: 同步模式，diff（增量）或 full（完整），默认 full

    认证：通过 token 查询参数传递（可选，取决于 WRAPPER_WEBSOCKET_TOKEN 配置）。
    """
    # 认证检查
    if not verify_websocket_token(token):
        await websocket.close(code=1008, reason="Unauthorized")
        logger.warning("[WebSocket] 认证失败，拒绝连接")
        return

    # 包装为可靠 WebSocket 服务器（带心跳和 DIFF 支持）
    reliable_ws = ReliableWebSocketServer(
        websocket=websocket,
        heartbeat_interval=30.0,
        heartbeat_timeout=5.0,
        max_missing_pongs=2,
        diff_mode=mode,
        diff_threshold=50.0,
        diff_min_size=100,
    )

    query_uuid = None
    live_diff_handler = None

    try:
        from ..main import SurrealDBManager

        db_manager = await SurrealDBManager.get_instance()

        # 确保数据库连接已建立
        try:
            _ = db_manager.db  # 测试连接是否存在
        except RuntimeError:
            await db_manager.reconnect()

        # 定期清理过期 session（每次新连接时执行）
        if reliable_ws._state_recovery:
            cleaned = reliable_ws._state_recovery.cleanup_expired()
            if cleaned > 0:
                logger.debug("[WebSocket] 清理过期 session: %d", cleaned)

        # 创建 Session（必须在 accept 之前，这样 connected 消息才有 session_id）
        session_id = reliable_ws.create_session()

        # 接受连接并启动心跳
        await reliable_ws.accept()

        # 如果有恢复的状态，尝试恢复
        restore_session_id = websocket.query_params.get("session_id")
        if restore_session_id:
            if await reliable_ws.restore_session(restore_session_id):
                session_id = restore_session_id
                logger.info("[WebSocket] 恢复 Session: %s", session_id)

        logger.info(
            "[WebSocket] 客户端已连接，租户: %s, 模式: %s, Session: %s",
            tenant_id,
            mode,
            session_id,
        )

        # 根据模式选择处理方式
        if mode == "diff":
            # 使用 LiveDiffHandler 处理增量同步
            live_diff_handler = LiveDiffHandler(
                surrealdb_client=db_manager.db,
                websocket_server=reliable_ws,
                diff_manager=reliable_ws._diff_manager,
                table_name="memory",
            )

            # 启动 LIVE SELECT 监听
            success = await live_diff_handler.start()
            if not success:
                await reliable_ws.send_json(
                    {
                        "type": "error",
                        "message": "启动 LIVE SELECT 失败",
                    }
                )
                await reliable_ws.close(code=1011, reason="Live select failed")
                return

            # 等待连接断开
            await reliable_ws.wait_for_disconnect()

            # 停止 LiveDiffHandler
            await live_diff_handler.stop()
        else:
            # 使用传统方式处理完整同步
            query_result = await db_manager.db.query(
                "LIVE SELECT * FROM memory WHERE tenant_id = $tenant_id",
                {"tenant_id": tenant_id},
            )
            # 处理 SurrealDB 3.0 返回格式（可能是 UUID 对象或列表）
            if isinstance(query_result, list) and len(query_result) > 0:
                if isinstance(query_result[0], dict):
                    query_uuid = query_result[0].get("result") or query_result[0].get("id")
                else:
                    query_uuid = str(query_result[0])
            else:
                # 直接返回 UUID 对象
                query_uuid = str(query_result)

            logger.info(
                "[WebSocket] 启动 LIVE 查询: %s",
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
        import traceback

        error_msg = f"{type(e).__name__}: {e}"
        logger.error("[WebSocket] 错误: %s", error_msg)
        traceback.print_exc()
        try:
            await reliable_ws.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        # 清理 LiveDiffHandler
        if live_diff_handler:
            try:
                await live_diff_handler.stop()
            except Exception:
                pass

        # 清理传统 LIVE 查询
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
        # subscribe_live 返回 coroutine，需要 await
        subscription = await db_manager.db.subscribe_live(query_uuid)
        async for notification in subscription:
            if not reliable_ws.is_connected:
                break
            await reliable_ws.send_json(notification)
    except asyncio.CancelledError:
        logger.debug("[WebSocket] 转发任务已取消")
        raise
    except Exception as e:
        logger.error("[WebSocket] 转发通知错误: %s", e)
