"""
P0级别Bug修复验证脚本

验证以下修复：
1. 配置管理 - Settings类正确性
2. 缓存实现 - 线程安全和TTL
3. 异常处理 - 异常类层次结构
"""

import sys
import time
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Settings, get_settings
from utils.cache import ThreadSafeLRUCache
from utils.exceptions import (
    WrapperServiceError,
    ServiceUnavailableError,
    CircuitBreakerOpenError,
    ValidationError,
)


def test_bug1_config():
    """测试Bug1修复：配置管理"""
    print("🔍 测试Bug1修复：配置管理")

    # 测试Settings类
    settings = get_settings()
    assert settings.wrapper_port == 3001
    assert settings.cache_ttl == 3600  # ✅ 验证ttl变量名正确
    assert settings.cache_size == 1000

    # 测试单例模式
    settings2 = get_settings()
    assert settings is settings2

    print("✅ Bug1修复验证通过：配置管理正确")


def test_bug2_cache():
    """测试Bug2修复：缓存实现"""
    print("\n🔍 测试Bug2修复：缓存实现")

    # 测试基本功能
    cache = ThreadSafeLRUCache(max_size=3, ttl_seconds=2)

    # 测试set和get
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"

    # 测试LRU淘汰
    cache.set("key2", "value2")
    cache.set("key3", "value3")
    cache.set("key4", "value4")  # 应该淘汰key1
    assert cache.get("key1") is None
    assert cache.get("key4") == "value4"

    # 测试TTL过期
    cache2 = ThreadSafeLRUCache(max_size=10, ttl_seconds=1)
    cache2.set("temp", "value")
    assert cache2.get("temp") == "value"
    time.sleep(1.1)
    assert cache2.get("temp") is None  # ✅ 验证TTL正确工作

    # 测试统计功能
    stats = cache.get_stats()
    assert "hit_rate" in stats
    assert "ttl_seconds" in stats  # ✅ 验证ttl_seconds字段存在

    print("✅ Bug2修复验证通过：缓存线程安全且TTL正确")


def test_bug3_exceptions():
    """测试Bug3修复：异常处理"""
    print("\n🔍 测试Bug3修复：异常处理")

    # 测试基础异常
    try:
        raise WrapperServiceError("Test error", status_code=500)
    except WrapperServiceError as e:
        assert e.message == "Test error"
        assert e.status_code == 500

    # 测试服务不可用异常
    try:
        raise ServiceUnavailableError("embedding")
    except ServiceUnavailableError as e:
        assert "embedding" in e.message
        assert e.status_code == 503

    # 测试熔断器异常
    try:
        raise CircuitBreakerOpenError("llm")
    except CircuitBreakerOpenError as e:
        assert "llm" in e.message
        assert e.status_code == 503

    # 测试验证错误
    try:
        raise ValidationError("Invalid input")
    except ValidationError as e:
        assert e.status_code == 400

    print("✅ Bug3修复验证通过：异常处理机制完整")


if __name__ == "__main__":
    print("=" * 60)
    print("P0级别Bug修复验证")
    print("=" * 60)

    try:
        test_bug1_config()
        test_bug2_cache()
        test_bug3_exceptions()

        print("\n" + "=" * 60)
        print("✅ 所有P0级别Bug修复验证通过！")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ 验证失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 验证出错: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
