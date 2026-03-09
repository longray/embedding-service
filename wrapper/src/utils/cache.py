"""
线程安全的LRU缓存实现
"""

from collections import OrderedDict
from threading import RLock
from typing import Optional, Any, Tuple
import time
import hashlib


def hash_text(text: str) -> str:
    """生成文本的哈希键"""
    return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()  # nosec B324


class ThreadSafeLRUCache:
    """线程安全的 LRU 缓存，支持 TTL"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds  # ✅ 修复：正确的变量名
        self._cache: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
        self._lock = RLock()  # ✅ 修复：使用可重入锁实现线程安全
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值（线程安全）"""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            timestamp, value = self._cache[key]

            # 检查是否过期
            if time.time() - timestamp > self._ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None

            # 移动到末尾（标记为最近使用）
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        """设置缓存值（线程安全）"""
        with self._lock:
            current_time = time.time()

            if key in self._cache:
                # 更新现有键
                self._cache[key] = (current_time, value)
                self._cache.move_to_end(key)
            else:
                # 添加新键
                if len(self._cache) >= self._max_size:
                    # 删除最久未使用的项（第一个）
                    self._cache.popitem(last=False)

                self._cache[key] = (current_time, value)

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def get_stats(self) -> dict:
        """获取缓存统计"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0

            return {
                "max_size": self._max_size,
                "current_size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 2),
                "ttl_seconds": self._ttl_seconds,
            }
