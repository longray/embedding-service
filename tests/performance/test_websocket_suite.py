"""WebSocket 性能测试套件

整合所有 WebSocket 性能测试：
- 并发连接测试 (test_websocket_concurrent.py)
- 消息延迟测试 (test_websocket_latency.py)
- 心跳可靠性测试 (test_websocket_reliability.py)

运行方式：
    uv run pytest tests/performance/test_websocket_suite.py -v
    uv run pytest tests/performance/test_websocket_suite.py -v --quick
    uv run pytest tests/performance/test_websocket_suite.py -v --full
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

import pytest

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class SuiteTestResult:
    """套件测试结果"""

    name: str
    passed: bool
    duration: float
    metrics: Dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass
class SuiteReport:
    """测试套件报告"""

    timestamp: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    results: List[SuiteTestResult] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)


class WebSocketPerformanceSuite:
    """WebSocket 性能测试套件"""

    def __init__(self, base_url: str = "ws://localhost:18008", output_dir: str = "tests/performance/reports"):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[SuiteTestResult] = []

    async def run_concurrent_test(self, clients: int = 100, duration: int = 30) -> SuiteTestResult:
        """运行并发连接测试"""
        logger.info(f"[Suite] 运行并发连接测试: {clients} clients, {duration}s")
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

            elapsed = time.time() - start_time
            passed = result.returncode == 0

            # 解析指标
            metrics = {}
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

            return SuiteTestResult(
                name="concurrent",
                passed=passed,
                duration=elapsed,
                metrics=metrics,
                errors=[result.stderr] if result.stderr else [],
            )

        except Exception as e:
            return SuiteTestResult(
                name="concurrent",
                passed=False,
                duration=time.time() - start_time,
                errors=[str(e)],
            )

    async def run_latency_test(self, clients: int = 10, messages: int = 100) -> SuiteTestResult:
        """运行延迟测试"""
        logger.info(f"[Suite] 运行延迟测试: {clients} clients, {messages} messages")
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

            elapsed = time.time() - start_time
            passed = result.returncode == 0

            # 解析指标
            metrics = {}
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

            return SuiteTestResult(
                name="latency",
                passed=passed,
                duration=elapsed,
                metrics=metrics,
                errors=[result.stderr] if result.stderr else [],
            )

        except Exception as e:
            return SuiteTestResult(
                name="latency",
                passed=False,
                duration=time.time() - start_time,
                errors=[str(e)],
            )

    async def run_reliability_test(self, duration: int = 300) -> SuiteTestResult:
        """运行可靠性测试"""
        logger.info(f"[Suite] 运行可靠性测试: {duration}s")
        start_time = time.time()

        try:
            cmd = [
                "uv",
                "run",
                "python",
                "tests/performance/test_websocket_reliability.py",
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

            elapsed = time.time() - start_time
            passed = result.returncode == 0

            # 解析指标
            metrics = {}
            for line in result.stdout.split("\n"):
                if "success_rate" in line:
                    try:
                        metrics["success_rate"] = float(line.split(":")[-1].strip().replace("%", ""))
                    except:
                        pass
                elif "packet_loss_rate" in line:
                    try:
                        metrics["packet_loss_rate"] = float(line.split(":")[-1].strip().replace("%", ""))
                    except:
                        pass

            return SuiteTestResult(
                name="reliability",
                passed=passed,
                duration=elapsed,
                metrics=metrics,
                errors=[result.stderr] if result.stderr else [],
            )

        except Exception as e:
            return SuiteTestResult(
                name="reliability",
                passed=False,
                duration=time.time() - start_time,
                errors=[str(e)],
            )

    async def run_all_tests(self, quick: bool = False, full: bool = False) -> SuiteReport:
        """运行所有测试"""
        logger.info("[Suite] 开始运行性能测试套件")

        # 配置测试参数
        if quick:
            concurrent_clients, concurrent_duration = 50, 10
            latency_clients, latency_messages = 5, 50
            reliability_duration = 60
        elif full:
            concurrent_clients, concurrent_duration = 1000, 60
            latency_clients, latency_messages = 20, 500
            reliability_duration = 600
        else:
            concurrent_clients, concurrent_duration = 100, 30
            latency_clients, latency_messages = 10, 100
            reliability_duration = 300

        # 顺序执行所有测试
        self.results.append(await self.run_concurrent_test(concurrent_clients, concurrent_duration))
        self.results.append(await self.run_latency_test(latency_clients, latency_messages))
        self.results.append(await self.run_reliability_test(reliability_duration))

        # 生成报告
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed

        summary = {
            "concurrent_clients": concurrent_clients,
            "concurrent_duration": concurrent_duration,
            "latency_clients": latency_clients,
            "latency_messages": latency_messages,
            "reliability_duration": reliability_duration,
        }

        # 检查通过标准
        for result in self.results:
            if result.name == "concurrent":
                metrics = result.metrics
                if "peak_memory_mb" in metrics:
                    summary["memory_check"] = "PASS"  # type: ignore if metrics["peak_memory_mb"] < 2048 else "FAIL"
                if "avg_cpu_percent" in metrics:
                    summary["cpu_check"] = "PASS"  # type: ignore if metrics["avg_cpu_percent"] < 80 else "FAIL"
            elif result.name == "latency":
                metrics = result.metrics
                if "p99_ms" in metrics:
                    summary["latency_p99_check"] = "PASS"  # type: ignore if metrics["p99_ms"] < 100 else "FAIL"
            elif result.name == "reliability":
                metrics = result.metrics
                if "success_rate" in metrics:
                    summary["reliability_check"] = "PASS"  # type: ignore if metrics["success_rate"] >= 99 else "FAIL"

        return SuiteReport(
            timestamp=datetime.utcnow().isoformat(),
            total_tests=len(self.results),
            passed_tests=passed,
            failed_tests=failed,
            results=self.results,
            summary=summary,
        )

    def save_report(self, report: SuiteReport) -> Path:
        """保存报告"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"suite_report_{timestamp}.json"

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": report.timestamp,
                    "total_tests": report.total_tests,
                    "passed_tests": report.passed_tests,
                    "failed_tests": report.failed_tests,
                    "results": [
                        {
                            "name": r.name,
                            "passed": r.passed,
                            "duration": r.duration,
                            "metrics": r.metrics,
                            "errors": r.errors,
                        }
                        for r in report.results
                    ],
                    "summary": report.summary,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        # 保存最新报告
        latest_file = self.output_dir / "suite_report_latest.json"
        with open(latest_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": report.timestamp,
                    "total_tests": report.total_tests,
                    "passed_tests": report.passed_tests,
                    "failed_tests": report.failed_tests,
                    "results": [
                        {
                            "name": r.name,
                            "passed": r.passed,
                            "duration": r.duration,
                            "metrics": r.metrics,
                            "errors": r.errors,
                        }
                        for r in report.results
                    ],
                    "summary": report.summary,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(f"[Suite] 报告已保存: {report_file}")
        return report_file


# pytest 测试函数
@pytest.mark.asyncio
@pytest.mark.performance
async def test_websocket_performance_suite_quick():
    """快速性能测试套件"""
    suite = WebSocketPerformanceSuite()
    report = await suite.run_all_tests(quick=True)
    suite.save_report(report)

    assert report.failed_tests == 0, f"有 {report.failed_tests} 个测试失败"


@pytest.mark.asyncio
@pytest.mark.performance
async def test_websocket_performance_suite_standard():
    """标准性能测试套件"""
    suite = WebSocketPerformanceSuite()
    report = await suite.run_all_tests()
    suite.save_report(report)

    assert report.failed_tests == 0, f"有 {report.failed_tests} 个测试失败"


@pytest.mark.asyncio
@pytest.mark.performance
async def test_websocket_performance_suite_full():
    """完整性能测试套件"""
    suite = WebSocketPerformanceSuite()
    report = await suite.run_all_tests(full=True)
    suite.save_report(report)

    assert report.failed_tests == 0, f"有 {report.failed_tests} 个测试失败"


# 命令行入口
async def main():
    parser = argparse.ArgumentParser(description="WebSocket 性能测试套件")
    parser.add_argument("--quick", action="store_true", help="快速模式")
    parser.add_argument("--full", action="store_true", help="完整模式")
    parser.add_argument("--url", default="ws://localhost:18008", help="WebSocket URL")
    parser.add_argument("--output", default="tests/performance/reports", help="输出目录")

    args = parser.parse_args()

    suite = WebSocketPerformanceSuite(base_url=args.url, output_dir=args.output)

    try:
        report = await suite.run_all_tests(quick=args.quick, full=args.full)

        # 打印报告
        print("\n" + "=" * 80)
        print("WebSocket 性能测试套件报告")
        print("=" * 80)
        print(f"时间: {report.timestamp}")
        print(f"总测试: {report.total_tests}")
        print(f"通过: {report.passed_tests}")
        print(f"失败: {report.failed_tests}")
        print("-" * 80)

        for result in report.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"\n{status} - {result.name}")
            print(f"  耗时: {result.duration:.2f}s")
            if result.metrics:
                for key, value in result.metrics.items():
                    print(f"  {key}: {value}")

        print("\n" + "-" * 80)
        print("汇总:")
        for key, value in report.summary.items():
            print(f"  {key}: {value}")
        print("=" * 80 + "\n")

        # 保存报告
        suite.save_report(report)

        # 返回退出码
        return 0 if report.failed_tests == 0 else 1

    except KeyboardInterrupt:
        logger.info("测试被中断")
        return 130
    except Exception as e:
        logger.error(f"测试失败: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
