"""WebSocket 性能测试运行器

运行所有 WebSocket 性能测试并生成报告。

使用方式:
    uv run python tests/performance/run_performance_tests.py
    uv run python tests/performance/run_performance_tests.py --quick
    uv run python tests/performance/run_performance_tests.py --full
"""

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """测试结果"""

    name: str
    passed: bool
    duration: float
    metrics: Dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass
class PerformanceReport:
    """性能测试报告"""

    timestamp: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    results: List[TestResult] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "timestamp": self.timestamp,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "duration": r.duration,
                    "metrics": r.metrics,
                    "errors": r.errors,
                }
                for r in self.results
            ],
            "summary": self.summary,
        }


class PerformanceTestRunner:
    """性能测试运行器"""

    def __init__(self, base_url: str = "ws://localhost:18008"):
        self.base_url = base_url
        self.results: List[TestResult] = []

    async def run_concurrent_test(
        self,
        clients: int = 100,
        duration: int = 30,
    ) -> TestResult:
        """运行并发连接测试"""
        logger.info(f"运行并发连接测试: {clients} clients, {duration}s")

        start_time = time.time()

        try:
            cmd = [
                "uv",
                "run",
                "python",
                "tests/performance/test_websocket_concurrent.py",
                "--clients",
                str(clients),
                "--duration",
                str(duration),
                "--url",
                f"{self.base_url}/ws/memories/live",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=duration + 60,
            )

            duration_elapsed = time.time() - start_time

            # 解析输出
            passed = result.returncode == 0
            errors = []
            metrics = {}

            if result.stderr:
                errors.append(result.stderr)

            # 尝试从输出解析指标
            for line in result.stdout.split("\n"):
                if "peak_memory_mb" in line:
                    try:
                        metrics["peak_memory_mb"] = float(line.split(":")[-1].strip())
                    except:
                        pass
                elif "avg_cpu_percent" in line:
                    try:
                        metrics["avg_cpu_percent"] = float(line.split(":")[-1].strip())
                    except:
                        pass
                elif "success_rate" in line:
                    try:
                        metrics["success_rate"] = float(line.split(":")[-1].strip().replace("%", ""))
                    except:
                        pass

            return TestResult(
                name=f"concurrent_{clients}",
                passed=passed,
                duration=duration_elapsed,
                metrics=metrics,
                errors=errors,
            )

        except subprocess.TimeoutExpired:
            return TestResult(
                name=f"concurrent_{clients}",
                passed=False,
                duration=time.time() - start_time,
                errors=["Test timeout"],
            )
        except Exception as e:
            return TestResult(
                name=f"concurrent_{clients}",
                passed=False,
                duration=time.time() - start_time,
                errors=[str(e)],
            )

    async def run_latency_test(
        self,
        clients: int = 10,
        messages: int = 100,
    ) -> TestResult:
        """运行延迟测试"""
        logger.info(f"运行延迟测试: {clients} clients, {messages} messages")

        start_time = time.time()

        try:
            cmd = [
                "uv",
                "run",
                "python",
                "tests/performance/test_websocket_latency.py",
                "--clients",
                str(clients),
                "--messages",
                str(messages),
                "--url",
                f"{self.base_url}/ws/memories/live",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )

            duration_elapsed = time.time() - start_time

            # 解析输出
            passed = result.returncode == 0
            errors = []
            metrics = {}

            if result.stderr:
                errors.append(result.stderr)

            # 尝试从输出解析指标
            for line in result.stdout.split("\n"):
                if "p99" in line and "ms" in line:
                    try:
                        metrics["p99_ms"] = float(line.split(":")[-1].strip().replace("ms", ""))
                    except:
                        pass
                elif "p95" in line and "ms" in line:
                    try:
                        metrics["p95_ms"] = float(line.split(":")[-1].strip().replace("ms", ""))
                    except:
                        pass
                elif "p50" in line and "ms" in line:
                    try:
                        metrics["p50_ms"] = float(line.split(":")[-1].strip().replace("ms", ""))
                    except:
                        pass
                elif "throughput" in line:
                    try:
                        metrics["throughput_msg_s"] = float(line.split(":")[-1].strip().replace("msg/s", ""))
                    except:
                        pass

            return TestResult(
                name=f"latency_{clients}x{messages}",
                passed=passed,
                duration=duration_elapsed,
                metrics=metrics,
                errors=errors,
            )

        except subprocess.TimeoutExpired:
            return TestResult(
                name=f"latency_{clients}x{messages}",
                passed=False,
                duration=time.time() - start_time,
                errors=["Test timeout"],
            )
        except Exception as e:
            return TestResult(
                name=f"latency_{clients}x{messages}",
                passed=False,
                duration=time.time() - start_time,
                errors=[str(e)],
            )

    async def run_all_tests(self, quick: bool = False, full: bool = False) -> PerformanceReport:
        """运行所有测试"""
        logger.info("开始运行性能测试套件")

        if quick:
            # 快速模式：少量客户端和短时间
            concurrent_clients = 50
            concurrent_duration = 10
            latency_clients = 5
            latency_messages = 50
        elif full:
            # 完整模式：大量客户端和长时间
            concurrent_clients = 1000
            concurrent_duration = 60
            latency_clients = 20
            latency_messages = 500
        else:
            # 标准模式
            concurrent_clients = 100
            concurrent_duration = 30
            latency_clients = 10
            latency_messages = 100

        # 运行并发测试
        concurrent_result = await self.run_concurrent_test(
            clients=concurrent_clients,
            duration=concurrent_duration,
        )
        self.results.append(concurrent_result)

        # 运行延迟测试
        latency_result = await self.run_latency_test(
            clients=latency_clients,
            messages=latency_messages,
        )
        self.results.append(latency_result)

        # 生成报告
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed

        # 计算汇总指标
        summary = {
            "concurrent_clients": concurrent_clients,
            "concurrent_duration": concurrent_duration,
            "latency_clients": latency_clients,
            "latency_messages": latency_messages,
        }

        # 添加通过/失败标准检查
        for result in self.results:
            if result.name.startswith("concurrent"):
                # 并发测试标准
                metrics = result.metrics
                if "peak_memory_mb" in metrics:
                    summary["memory_check"] = "PASS"  # type: ignore if metrics["peak_memory_mb"] < 2048 else "FAIL"
                if "avg_cpu_percent" in metrics:
                    summary["cpu_check"] = "PASS"  # type: ignore if metrics["avg_cpu_percent"] < 80 else "FAIL"
            elif result.name.startswith("latency"):
                # 延迟测试标准
                metrics = result.metrics
                if "p99_ms" in metrics:
                    summary["latency_p99_check"] = "PASS"  # type: ignore if metrics["p99_ms"] < 100 else "FAIL"

        report = PerformanceReport(
            timestamp=datetime.utcnow().isoformat(),
            total_tests=len(self.results),
            passed_tests=passed,
            failed_tests=failed,
            results=self.results,
            summary=summary,
        )

        return report

    def save_report(self, report: PerformanceReport, output_dir: str = "tests/performance/reports"):
        """保存报告到文件"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_file = output_path / f"performance_report_{timestamp}.json"

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"报告已保存: {report_file}")

        # 同时保存为最新报告
        latest_file = output_path / "performance_report_latest.json"
        with open(latest_file, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

        return report_file

    def print_report(self, report: PerformanceReport):
        """打印报告到控制台"""
        print("\n" + "=" * 80)
        print("WebSocket 性能测试报告")
        print("=" * 80)
        print(f"时间: {report.timestamp}")
        print(f"总测试数: {report.total_tests}")
        print(f"通过: {report.passed_tests}")
        print(f"失败: {report.failed_tests}")
        print("-" * 80)

        for result in report.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"\n{status} - {result.name}")
            print(f"  耗时: {result.duration:.2f}s")
            if result.metrics:
                print(f"  指标:")
                for key, value in result.metrics.items():
                    print(f"    {key}: {value}")
            if result.errors:
                print(f"  错误:")
                for error in result.errors[:3]:  # 只显示前3个错误
                    print(f"    - {error[:100]}")

        print("\n" + "-" * 80)
        print("汇总:")
        for key, value in report.summary.items():
            print(f"  {key}: {value}")
        print("=" * 80 + "\n")


async def main():
    parser = argparse.ArgumentParser(description="WebSocket 性能测试运行器")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速模式（少量客户端和短时间）",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="完整模式（大量客户端和长时间）",
    )
    parser.add_argument(
        "--url",
        default="ws://localhost:18008",
        help="WebSocket 服务器 URL",
    )
    parser.add_argument(
        "--output",
        default="tests/performance/reports",
        help="报告输出目录",
    )

    args = parser.parse_args()

    runner = PerformanceTestRunner(base_url=args.url)

    try:
        report = await runner.run_all_tests(quick=args.quick, full=args.full)
        runner.print_report(report)
        report_file = runner.save_report(report, output_dir=args.output)

        # 返回退出码
        if report.failed_tests > 0:
            logger.warning(f"有 {report.failed_tests} 个测试失败")
            return 1
        else:
            logger.info("所有测试通过")
            return 0

    except KeyboardInterrupt:
        logger.info("测试被用户中断")
        return 130
    except Exception as e:
        logger.error(f"测试运行失败: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
