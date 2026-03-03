"""
P0级别Bug修复验证脚本（简化版）

直接检查代码文件的正确性，无需运行
"""

import re
from pathlib import Path


def verify_bug1_config():
    """验证Bug1修复：配置管理"""
    print("🔍 验证Bug1修复：配置管理")

    config_file = Path(__file__).parent.parent / "src" / "config.py"
    content = config_file.read_text(encoding="utf-8")

    # 检查关键修复点
    checks = [
        (
            "使用pydantic_settings",
            "from pydantic_settings import BaseSettings" in content,
        ),
        ("使用lru_cache单例", "@lru_cache()" in content),
        ("正确的ttl变量名", "cache_ttl: int = 3600" in content),
        ("环境变量前缀", 'env_prefix = "WRAPPER_"' in content),
    ]

    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if not result:
            return False

    print("✅ Bug1修复验证通过")
    return True


def verify_bug2_cache():
    """验证Bug2修复：缓存实现"""
    print("\n🔍 验证Bug2修复：缓存实现")

    cache_file = Path(__file__).parent.parent / "src" / "utils" / "cache.py"
    content = cache_file.read_text(encoding="utf-8")

    # 检查关键修复点
    checks = [
        ("使用RLock线程安全", "from threading import RLock" in content),
        ("使用OrderedDict实现LRU", "from collections import OrderedDict" in content),
        ("正确的ttl变量名", "self._ttl_seconds" in content),
        ("没有错误的tl变量", "self.tl" not in content and "self._tl" not in content),
        ("实现move_to_end", "move_to_end" in content),
        ("实现get_stats", "def get_stats" in content),
    ]

    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if not result:
            return False

    print("✅ Bug2修复验证通过")
    return True


def verify_bug3_exceptions():
    """验证Bug3修复：异常处理"""
    print("\n🔍 验证Bug3修复：异常处理")

    exc_file = Path(__file__).parent.parent / "src" / "utils" / "exceptions.py"
    content = exc_file.read_text(encoding="utf-8")

    # 检查关键修复点
    checks = [
        ("基础异常类", "class WrapperServiceError(Exception)" in content),
        ("服务不可用异常", "class ServiceUnavailableError" in content),
        ("熔断器异常", "class CircuitBreakerOpenError" in content),
        ("验证错误异常", "class ValidationError" in content),
        ("状态码支持", "status_code" in content),
        ("详情支持", "details" in content),
    ]

    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if not result:
            return False

    print("✅ Bug3修复验证通过")
    return True


def verify_file_structure():
    """验证文件结构"""
    print("\n🔍 验证文件结构")

    base_dir = Path(__file__).parent.parent

    required_files = [
        "src/config.py",
        "src/utils/cache.py",
        "src/utils/exceptions.py",
    ]

    all_exist = True
    for file_path in required_files:
        full_path = base_dir / file_path
        exists = full_path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {file_path}")
        if not exists:
            all_exist = False

    if all_exist:
        print("✅ 文件结构验证通过")
    return all_exist


if __name__ == "__main__":
    print("=" * 60)
    print("P0级别Bug修复验证（代码检查）")
    print("=" * 60)

    results = []

    try:
        results.append(("文件结构", verify_file_structure()))
        results.append(("Bug1-配置管理", verify_bug1_config()))
        results.append(("Bug2-缓存实现", verify_bug2_cache()))
        results.append(("Bug3-异常处理", verify_bug3_exceptions()))

        print("\n" + "=" * 60)
        print("验证结果汇总：")
        print("=" * 60)

        all_passed = True
        for name, passed in results:
            status = "✅ 通过" if passed else "❌ 失败"
            print(f"{name}: {status}")
            if not passed:
                all_passed = False

        print("=" * 60)
        if all_passed:
            print("✅ 所有P0级别Bug修复验证通过！")
        else:
            print("❌ 部分验证失败，请检查")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 验证出错: {e}")
        import traceback

        traceback.print_exc()
