"""WebSocket 认证工具（最小化实现）"""

import os


def get_websocket_token() -> str | None:
    """获取配置的 WebSocket token（可选）"""
    return os.getenv("WRAPPER_WEBSOCKET_TOKEN")


def verify_websocket_token(token: str | None) -> bool:
    """验证 WebSocket token

    如果未配置 token（WRAPPER_WEBSOCKET_TOKEN 为空），则允许所有连接（向后兼容）。
    如果配置了 token，则必须匹配。
    """
    configured_token = get_websocket_token()

    # 未配置 token = 不启用认证
    if not configured_token:
        return True

    # 配置了 token = 必须匹配
    return token == configured_token
