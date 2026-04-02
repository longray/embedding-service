"""全局共享状态单例

避免 main.py ↔ routers 循环导入，将运行时单例集中在此模块。
main.py (lifespan) 负责初始化，routers 按需读取。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .utils.cache import ThreadSafeLRUCache
    from .utils.meili_client import MeilisearchClient
    from .utils.memory_manager import MemoryManager

embedding_cache: ThreadSafeLRUCache | None = None
memory_manager: MemoryManager | None = None
meili_client: MeilisearchClient | None = None
