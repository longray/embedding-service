"""
测试配置和共享fixtures
"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中（单元测试需要导入 wrapper 模块）
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import socket


def _check_port(host: str, port: int) -> bool:
    """检查端口是否可连接"""
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except (ConnectionRefusedError, socket.timeout, OSError):
        return False


LLM_AVAILABLE = _check_port("localhost", 18001)


def pytest_collection_modifyitems(items):
    """自动跳过依赖 LLM 服务的测试（当服务未启动时）。

    基于测试的 fixture 依赖关系判断：如果测试使用了 llm_client fixture
    且 LLM 服务未启动，则自动标记为 skip。
    """
    if LLM_AVAILABLE:
        return
    for item in items:
        if "llm_client" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.skip(reason="LLM 服务 (localhost:18001) 未启动"))


import pytest
import pytest_asyncio
import httpx
from typing import AsyncGenerator

# 服务配置
EMBEDDING_SERVICE_URL = "http://localhost:18000"
LLM_SERVICE_URL = "http://localhost:18001"
WRAPPER_SERVICE_URL = "http://localhost:18008"
WRAPPER_MINIMAL_URL = "http://localhost:18008"

# 超时配置
DEFAULT_TIMEOUT = 30.0
HEALTH_CHECK_TIMEOUT = 5.0


@pytest_asyncio.fixture(scope="session")
async def http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """HTTP客户端fixture"""
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        yield client


@pytest_asyncio.fixture
async def embedding_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Embedding服务客户端"""
    async with httpx.AsyncClient(base_url=EMBEDDING_SERVICE_URL, timeout=DEFAULT_TIMEOUT) as client:
        yield client


@pytest_asyncio.fixture(scope="session")
async def llm_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """LLM服务客户端"""
    async with httpx.AsyncClient(base_url=LLM_SERVICE_URL, timeout=DEFAULT_TIMEOUT) as client:
        yield client


@pytest_asyncio.fixture
async def wrapper_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """包装层服务客户端"""
    async with httpx.AsyncClient(base_url=WRAPPER_SERVICE_URL, timeout=DEFAULT_TIMEOUT) as client:
        yield client


@pytest_asyncio.fixture(scope="session")
async def wrapper_minimal_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """最小化包装服务客户端（端口18008）"""
    async with httpx.AsyncClient(base_url=WRAPPER_MINIMAL_URL, timeout=DEFAULT_TIMEOUT) as client:
        yield client


@pytest.fixture
def sample_text() -> str:
    """测试文本"""
    return "Hello, world! This is a test."


@pytest.fixture
def sample_texts() -> list[str]:
    """批量测试文本"""
    return [
        "First test text",
        "Second test text",
        "Third test text",
    ]


@pytest.fixture
def sample_messages() -> list[dict]:
    """测试对话消息"""
    return [
        {"role": "user", "content": "你好，请介绍一下自己"},
    ]
