#!/usr/bin/env python3
"""验证服务是否正常运行的测试脚本"""

import requests
import sys


def check_service(name, url):
    """检查服务健康状态"""
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            print(f"✅ {name}: 正常运行")
            return True
        else:
            print(f"❌ {name}: HTTP {resp.status_code}")
            return False
    except Exception as e:
        print(f"❌ {name}: {e}")
        return False


def main():
    print("验证 Embedding Service 服务状态\n")
    print("-" * 40)

    services = [
        ("Embedding服务", "http://localhost:18000/health"),
        ("包装层服务", "http://localhost:18008/health"),
    ]

    results = []
    for name, url in services:
        results.append(check_service(name, url))

    print("-" * 40)
    if all(results):
        print("\n✅ 所有服务运行正常！")
        return 0
    else:
        print("\n❌ 部分服务未正常运行")
        return 1


if __name__ == "__main__":
    sys.exit(main())
