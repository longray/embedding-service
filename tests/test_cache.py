"""Unit tests for wrapper.src.utils.cache"""

import time
import pytest
from wrapper.src.utils.cache import ThreadSafeLRUCache, hash_text

pytestmark = pytest.mark.unit


class TestHashText:
    def test_same_input_same_hash(self):
        assert hash_text("hello") == hash_text("hello")

    def test_different_input_different_hash(self):
        assert hash_text("hello") != hash_text("world")

    def test_empty_string(self):
        result = hash_text("")
        assert isinstance(result, str)
        assert len(result) == 32  # MD5 hex digest


class TestThreadSafeLRUCache:
    def test_set_and_get(self):
        cache = ThreadSafeLRUCache(max_size=10, ttl_seconds=3600)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_miss_returns_none(self):
        cache = ThreadSafeLRUCache()
        assert cache.get("nonexistent") is None

    def test_ttl_expiration(self):
        cache = ThreadSafeLRUCache(max_size=10, ttl_seconds=1)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_max_size_eviction(self):
        cache = ThreadSafeLRUCache(max_size=3, ttl_seconds=3600)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.set("d", 4)  # Should evict "a" (oldest)
        assert cache.get("a") is None
        assert cache.get("d") == 4

    def test_lru_eviction_order(self):
        cache = ThreadSafeLRUCache(max_size=3, ttl_seconds=3600)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.get("a")  # Access "a" to make it recently used
        cache.set("d", 4)  # Should evict "b" (least recently used)
        assert cache.get("b") is None
        assert cache.get("a") == 1

    def test_clear(self):
        cache = ThreadSafeLRUCache()
        cache.set("key1", "value1")
        cache.clear()
        assert cache.get("key1") is None

    def test_get_stats(self):
        cache = ThreadSafeLRUCache(max_size=100, ttl_seconds=3600)
        cache.set("key1", "value1")
        cache.get("key1")  # hit
        cache.get("miss")  # miss
        stats = cache.get_stats()
        assert stats["max_size"] == 100
        assert stats["current_size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 50.0

    def test_overwrite_existing_key(self):
        cache = ThreadSafeLRUCache()
        cache.set("key1", "old_value")
        cache.set("key1", "new_value")
        assert cache.get("key1") == "new_value"
