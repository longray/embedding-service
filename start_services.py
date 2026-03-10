#!/usr/bin/env python3
"""
统一服务启动脚本

功能：
- 必须启动：Embedding服务（端口18000）
- 可选启动：LLM服务（端口18001）
- 自动启动：包装层服务（端口3001）

使用方法：
    python start_services.py              # 只启动Embedding + 包装层
    python start_services.py --with-llm   # 启动所有服务
    python start_services.py --help       # 查看帮助
"""

import argparse
import subprocess  # nosec B404
import sys
import time
from pathlib import Path
from typing import Optional

import requests

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 服务配置
SERVICES = {
    "embedding": {
        "name": "Embedding服务",
        "port": 18000,
        "script": "src/qwen3_embedding_service/embedding_service.py",
        "health_url": "http://localhost:18000/health",
        "required": True,
        "startup_time": 30,  # 预计启动时间（秒）
    },
    "llm": {
        "name": "LLM服务",
        "port": 18001,
        "script": "src/qwen3_embedding_service/llm_service.py",
        "health_url": "http://localhost:18001/health",
        "required": False,
        "startup_time": 30,
    },
    "wrapper": {
        "name": "包装层服务",
        "port": 3001,
        "script": "wrapper/src/main.py",
        "health_url": "http://localhost:3001/health",
        "required": True,
        "startup_time": 5,
    },
}


def print_banner():
    """打印启动横幅"""
    print("=" * 60)
    print("  Embedding Service 统一启动脚本")
    print("=" * 60)
    print()


def check_service_health(url: str, timeout: int = 2) -> bool:
    """检查服务健康状态"""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def wait_for_service(service_name: str, health_url: str, max_wait: int = 60, check_interval: int = 2) -> bool:
    """等待服务就绪"""
    print(f"⏳ 等待 {service_name} 就绪...", end="", flush=True)

    start_time = time.time()
    while time.time() - start_time < max_wait:
        if check_service_health(health_url):
            elapsed = time.time() - start_time
            print(f" ✅ 就绪 ({elapsed:.1f}秒)")
            return True

        print(".", end="", flush=True)
        time.sleep(check_interval)

    print(f" ❌ 超时 ({max_wait}秒)")
    return False


def start_service(service_key: str) -> Optional[subprocess.Popen]:
    """启动服务"""
    service = SERVICES[service_key]
    script_path = PROJECT_ROOT / service["script"]

    if not script_path.exists():
        print(f"❌ 错误: 脚本不存在 - {script_path}")
        return None

    print(f"🚀 启动 {service['name']} (端口 {service['port']})...")

    try:
        # 启动服务进程
        process = subprocess.Popen(  # nosec B607, B603
            ["uv", "run", "python", str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=PROJECT_ROOT,
        )

        # 给服务一点时间启动
        time.sleep(2)

        # 检查进程是否立即失败
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            print(f"❌ 启动失败:")
            print(stderr.decode("utf-8", errors="ignore"))
            return None

        return process

    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return None


def stop_services(processes: dict):
    """停止所有服务"""
    print("\n🛑 停止所有服务...")
    for service_key, process in processes.items():
        if process and process.poll() is None:
            service = SERVICES[service_key]
            print(f"  停止 {service['name']}...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    print("✅ 所有服务已停止")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="统一启动Embedding服务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python start_services.py              # 只启动Embedding + 包装层
  python start_services.py --with-llm   # 启动所有服务
        """,
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="同时启动LLM服务（可选）",
    )
    parser.add_argument(
        "--no-wrapper",
        action="store_true",
        help="不启动包装层服务（仅用于测试）",
    )
    args = parser.parse_args()

    print_banner()

    # 确定要启动的服务
    services_to_start = ["embedding"]
    if args.with_llm:
        services_to_start.append("llm")
    if not args.no_wrapper:
        services_to_start.append("wrapper")

    print("📋 启动计划:")
    for service_key in services_to_start:
        service = SERVICES[service_key]
        status = "必需" if service["required"] else "可选"
        print(f"  - {service['name']} (端口 {service['port']}) [{status}]")
    print()
    # 存储启动的进程
    processes = {}
    
    try:
        # 阶段1：启动后端服务
        print("📦 阶段1：启动后端服务")
        print("-" * 60)
        
        # 必须启动Embedding服务
        process = start_service("embedding")
        if not process:
            print("❌ Embedding服务启动失败，退出")
            return 1
        processes["embedding"] = process
        
        # 等待Embedding服务就绪
        if not wait_for_service(
            SERVICES["embedding"]["name"],
            SERVICES["embedding"]["health_url"],
            max_wait=SERVICES["embedding"]["startup_time"] + 30,
        ):
            print("❌ Embedding服务未能就绪，退出")
            stop_services(processes)
            return 1
        
        # 可选启动LLM服务
        if "llm" in services_to_start:
            process = start_service("llm")
            if not process:
                print("⚠️  LLM服务启动失败，继续（可选服务）")
            else:
                processes["llm"] = process
                
                # 等待LLM服务就绪
                if not wait_for_service(
                    SERVICES["llm"]["name"],
                    SERVICES["llm"]["health_url"],
                    max_wait=SERVICES["llm"]["startup_time"] + 30,
                ):
                    print("⚠️  LLM服务未能就绪，继续（可选服务）")
        
        print()
        # 阶段2：启动包装层服务
        if "wrapper" in services_to_start:
            print("🎁 阶段2：启动包装层服务")
            print("-" * 60)
            
            process = start_service("wrapper")
            if not process:
                print("❌ 包装层服务启动失败，退出")
                stop_services(processes)
                return 1
            processes["wrapper"] = process
            
            # 等待包装层服务就绪
            if not wait_for_service(
                SERVICES["wrapper"]["name"],
                SERVICES["wrapper"]["health_url"],
                max_wait=SERVICES["wrapper"]["startup_time"] + 10,
            ):
                print("❌ 包装层服务未能就绪，退出")
                stop_services(processes)
                return 1
            
            print()
        
        # 启动成功
        print("=" * 60)
        print("✅ 所有服务启动成功！")
        print("=" * 60)
        print()
        print("📡 服务访问地址：")
        for service_key in services_to_start:
            service = SERVICES[service_key]
            print(f"  - {service['name']}: http://localhost:{service['port']}")
        print()
        print("💡 提示：")
        print("  - 按 Ctrl+C 停止所有服务")
        if "wrapper" in services_to_start:
            print("  - 推荐使用包装层服务（端口3001）访问")
        print()
        
        # 保持运行
        print("⏸️  服务运行中，按 Ctrl+C 停止...")
        print()
        while True:
            time.sleep(1)
            # 检查进程是否意外退出
            for service_key, process in list(processes.items()):
                if process.poll() is not None:
                    service = SERVICES[service_key]
                    print(f"\n❌ {service['name']} 意外退出")
                    stop_services(processes)
                    return 1
    
    except KeyboardInterrupt:
        print("\n\n⚠️  收到停止信号...")
        stop_services(processes)
        return 0
    
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        stop_services(processes)
        return 1
if __name__ == "__main__":
    sys.exit(main())
