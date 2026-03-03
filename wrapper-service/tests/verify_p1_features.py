"""
验证P1功能代码
"""

import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports():
    """测试所有模块导入"""
    print("测试模块导入...")

    try:
        from config import get_settings

        print("✅ config.py 导入成功")
    except Exception as e:
        print(f"❌ config.py 导入失败: {e}")
        return False

    try:
        from utils.cache import ThreadSafeLRUCache

        print("✅ cache.py 导入成功")
    except Exception as e:
        print(f"❌ cache.py 导入失败: {e}")
        return False

    try:
        from utils.exceptions import WrapperServiceError

        print("✅ exceptions.py 导入成功")
    except Exception as e:
        print(f"❌ exceptions.py 导入失败: {e}")
        return False

    try:
        from utils.circuit_breaker import CircuitBreaker

        print("✅ circuit_breaker.py 导入成功")
    except Exception as e:
        print(f"❌ circuit_breaker.py 导入失败: {e}")
        return False

    try:
        from utils.http_pool import HTTPClientPool

        print("✅ http_pool.py 导入成功")
    except Exception as e:
        print(f"❌ http_pool.py 导入失败: {e}")
        return False

    try:
        from utils.logging import setup_logging

        print("✅ logging.py 导入成功")
    except Exception as e:
        print(f"❌ logging.py 导入失败: {e}")
        return False

    try:
        from utils import metrics

        print("✅ metrics.py 导入成功")
    except Exception as e:
        print(f"❌ metrics.py 导入失败: {e}")
        return False

    return True


def test_circuit_breaker():
    """测试熔断器基本功能"""
    print("\n测试熔断器功能...")

    from utils.circuit_breaker import CircuitBreaker, CircuitState

    cb = CircuitBreaker(failure_threshold=2, timeout=1.0)

    # 测试初始状态
    assert cb.state == CircuitState.CLOSED, "初始状态应为CLOSED"
    print("✅ 初始状态正确")

    # 测试成功调用
    result = cb.call(lambda: "success")
    assert result == "success", "成功调用应返回结果"
    print("✅ 成功调用正常")

    # 测试失败调用
    try:
        cb.call(lambda: 1 / 0)
    except ZeroDivisionError:
        pass

    try:
        cb.call(lambda: 1 / 0)
    except ZeroDivisionError:
        pass

    # 应该转为OPEN状态
    assert cb.state == CircuitState.OPEN, "失败次数达到阈值应转为OPEN"
    print("✅ 熔断器打开正常")

    return True


def test_cache():
    """测试缓存基本功能"""
    print("\n测试缓存功能...")

    from utils.cache import ThreadSafeLRUCache

    cache = ThreadSafeLRUCache(max_size=2, ttl_seconds=60)

    # 测试put和get
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1", "应该能获取到缓存值"
    print("✅ 缓存存取正常")

    # 测试LRU
    cache.set("key2", "value2")
    cache.set("key3", "value3")  # 应该淘汰key1

    assert cache.get("key1") is None, "key1应该被淘汰"
    assert cache.get("key2") == "value2", "key2应该存在"
    print("✅ LRU淘汰正常")

    # 测试统计
    stats = cache.get_stats()
    assert "current_size" in stats, "统计应包含current_size"
    print("✅ 缓存统计正常")

    return True


def test_exceptions():
    """测试异常类"""
    print("\n测试异常类...")

    from utils.exceptions import (
        WrapperServiceError,
        ServiceUnavailableError,
        CircuitBreakerError,
    )

    # 测试基类
    try:
        raise WrapperServiceError("test error")
    except WrapperServiceError as e:
        assert e.message == "test error"
        assert e.status_code == 500
        print("✅ 基础异常类正常")

    # 测试子类
    try:
        raise ServiceUnavailableError("service down")
    except ServiceUnavailableError as e:
        assert e.status_code == 503
        print("✅ ServiceUnavailableError正常")

    try:
        raise CircuitBreakerError("circuit open")
    except CircuitBreakerError as e:
        assert e.status_code == 503
        print("✅ CircuitBreakerError正常")

    return True


def main():
    """运行所有验证"""
    print("=" * 60)
    print("P1功能代码验证")
    print("=" * 60)

    all_passed = True

    # 测试导入
    if not test_imports():
        all_passed = False

    # 测试熔断器
    try:
        if not test_circuit_breaker():
            all_passed = False
    except Exception as e:
        print(f"❌ 熔断器测试失败: {e}")
        all_passed = False

    # 测试缓存
    try:
        if not test_cache():
            all_passed = False
    except Exception as e:
        print(f"❌ 缓存测试失败: {e}")
        all_passed = False

    # 测试异常
    try:
        if not test_exceptions():
            all_passed = False
    except Exception as e:
        print(f"❌ 异常测试失败: {e}")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有验证通过")
        return 0
    else:
        print("❌ 部分验证失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
