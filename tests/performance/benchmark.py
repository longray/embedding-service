"""WebSocket 性能基准测试

建立 WebSocket 性能基准，记录关键指标，支持性能回归检测。

使用方式:
    uv run python tests/performance/benchmark.py
    uv run python tests/performance/benchmark.py --report
    uv run python tests/performance/benchmark.py --compare baseline.json
    uv run python tests/performance/benchmark.py --quick
    uv run python tests/performance/benchmark.py --full

输出:
    - tests/performance/reports/benchmark_YYYYMMDD_HHMMSS.json
    - tests/performance/reports/benchmark_YYYYMMDD_HHMMSS.md
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetric:
    """基准测试指标"""

    name: str
    values: list[float] = field(default_factory=list)
    unit: str = "ms"
    description: str = ""

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def avg(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0

    @property
    def min(self) -> float:
        return min(self.values) if self.values else 0

    @property
    def max(self) -> float:
        return max(self.values) if self.values else 0

    @property
    def p50(self) -> float:
        return self._percentile(50)

    @property
    def p95(self) -> float:
        return self._percentile(95)

    @property
    def p99(self) -> float:
        return self._percentile(99)

    def _percentile(self, p: int) -> float:
        if not self.values:
            return 0
        sorted_vals = sorted(self.values)
        idx = int(len(sorted_vals) * p / 100)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "unit": self.unit,
            "count": self.count,
            "avg": round(self.avg, 2),
            "min": round(self.min, 2),
            "max": round(self.max, 2),
            "p50": round(self.p50, 2),
            "p95": round(self.p95, 2),
            "p99": round(self.p99, 2),
        }


@dataclass
class BenchmarkResult:
    """基准测试结果"""

    timestamp: str
    version: str
    environment: dict
    metrics: list[BenchmarkMetric]
    duration: float
    passed: bool
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "version": self.version,
            "environment": self.environment,
            "metrics": [m.to_dict() for m in self.metrics],
            "duration": round(self.duration, 2),
            "passed": self.passed,
            "errors": self.errors,
        }


class PerformanceBenchmark:
    """性能基准测试运行器"""

    VERSION = "3.2.0"

    def __init__(
        self,
        base_url: str = "ws://localhost:18008",
        output_dir: str = "tests/performance/reports",
        quick: bool = False,
        full: bool = False,
    ):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quick = quick
        self.full = full
        self.metrics: list[BenchmarkMetric] = []
        self.errors: list[str] = []

        # 根据模式设置参数
        if quick:
            self.concurrent_clients = 50
            self.concurrent_duration = 10
            self.latency_clients = 10
            self.latency_messages = 100
            self.reliability_duration = 60
        elif full:
            self.concurrent_clients = 1000
            self.concurrent_duration = 300
            self.latency_clients = 100
            self.latency_messages = 1000
            self.reliability_duration = 3600
        else:
            self.concurrent_clients = 100
            self.concurrent_duration = 30
            self.latency_clients = 20
            self.latency_messages = 500
            self.reliability_duration = 300

    def _get_environment(self) -> dict:
        """获取环境信息"""
        import platform
        import sys

        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "base_url": self.base_url,
            "test_mode": "quick" if self.quick else "full" if self.full else "standard",
        }

    async def _run_concurrent_test(self) -> BenchmarkMetric:
        """运行并发连接测试"""
        logger.info(f"运行并发连接测试: {self.concurrent_clients} clients, {self.concurrent_duration}s")

        metric = BenchmarkMetric(
            name="concurrent_connections",
            description=f"并发连接测试 ({self.concurrent_clients} clients, {self.concurrent_duration}s)",
            unit="count",
        )

        start_time = time.time()

        try:
            cmd = [
                "uv",
                "run",
                "python",
                "tests/performance/test_websocket_concurrent.py",
                "--clients",
                str(self.concurrent_clients),
                "--duration",
                str(self.concurrent_duration),
                "--url",
                f"{self.base_url}/ws/memories/live",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.concurrent_duration + 120,
            )

            elapsed = time.time() - start_time

            if result.returncode == 0:
                # 解析指标
                for line in result.stdout.split("\n"):
                    if "success_rate" in line and "%" in line:
                        try:
                            rate = float(line.split(":")[-1].strip().replace("%", ""))
                            metric.values.append(rate)
                        except:
                            pass
                    elif "peak_memory_mb" in line:
                        try:
                            memory = float(line.split(":")[-1].strip())
                            metric.values.append(memory)
                        except:
                            pass

                logger.info(f"✅ 并发测试完成: {elapsed:.1f}s")
            else:
                self.errors.append(f"并发测试失败: {result.stderr}")
                logger.error("❌ 并发测试失败")

        except subprocess.TimeoutExpired:
            self.errors.append("并发测试超时")
            logger.error("❌ 并发测试超时")
        except Exception as e:
            self.errors.append(f"并发测试异常: {e}")
            logger.error(f"❌ 并发测试异常: {e}")

        return metric

    async def _run_latency_test(self) -> BenchmarkMetric:
        """运行消息延迟测试"""
        logger.info(f"运行消息延迟测试: {self.latency_clients} clients, {self.latency_messages} messages")

        metric = BenchmarkMetric(
            name="message_latency",
            description=f"消息延迟测试 ({self.latency_clients} clients, {self.latency_messages} messages)",
            unit="ms",
        )

        start_time = time.time()

        try:
            cmd = [
                "uv",
                "run",
                "python",
                "tests/performance/test_websocket_latency.py",
                "--clients",
                str(self.latency_clients),
                "--messages",
                str(self.latency_messages),
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

            if result.returncode == 0:
                # 解析指标
                for line in result.stdout.split("\n"):
                    if "p99" in line and "ms" in line:
                        try:
                            # 提取 p99 值
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if "p99" in part and i + 1 < len(parts):
                                    val = parts[i + 1].replace("ms", "").replace(",", "")
                                    metric.values.append(float(val))
                                    break
                        except:
                            pass

                logger.info(f"✅ 延迟测试完成: {elapsed:.1f}s")
            else:
                self.errors.append(f"延迟测试失败: {result.stderr}")
                logger.error("❌ 延迟测试失败")

        except subprocess.TimeoutExpired:
            self.errors.append("延迟测试超时")
            logger.error("❌ 延迟测试超时")
        except Exception as e:
            self.errors.append(f"延迟测试异常: {e}")
            logger.error(f"❌ 延迟测试异常: {e}")

        return metric

    async def _run_reliability_test(self) -> BenchmarkMetric:
        """运行可靠性测试"""
        duration_str = f"{self.reliability_duration}s"
        if self.reliability_duration >= 60:
            duration_str = f"{self.reliability_duration // 60}m"

        logger.info(f"运行可靠性测试: {duration_str}")

        metric = BenchmarkMetric(
            name="heartbeat_reliability",
            description=f"心跳可靠性测试 ({duration_str})",
            unit="%",
        )

        start_time = time.time()

        try:
            cmd = [
                "uv",
                "run",
                "python",
                "tests/performance/test_websocket_reliability.py",
                "--duration",
                str(self.reliability_duration),
                "--url",
                f"{self.base_url}/ws/memories/live",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.reliability_duration + 120,
            )

            elapsed = time.time() - start_time

            if result.returncode == 0:
                # 解析指标
                for line in result.stdout.split("\n"):
                    if "success_rate" in line and "%" in line:
                        try:
                            rate = float(line.split(":")[-1].strip().replace("%", ""))
                            metric.values.append(rate)
                        except:
                            pass

                logger.info(f"✅ 可靠性测试完成: {elapsed:.1f}s")
            else:
                self.errors.append(f"可靠性测试失败: {result.stderr}")
                logger.error("❌ 可靠性测试失败")

        except subprocess.TimeoutExpired:
            self.errors.append("可靠性测试超时")
            logger.error("❌ 可靠性测试超时")
        except Exception as e:
            self.errors.append(f"可靠性测试异常: {e}")
            logger.error(f"❌ 可靠性测试异常: {e}")

        return metric

    async def run(self) -> BenchmarkResult:
        """运行完整基准测试"""
        logger.info("=" * 60)
        logger.info("WebSocket 性能基准测试")
        logger.info("=" * 60)
        logger.info(f"模式: {'快速' if self.quick else '完整' if self.full else '标准'}")
        logger.info(f"目标: {self.base_url}")
        logger.info("")

        start_time = time.time()

        # 运行各项测试
        concurrent_metric = await self._run_concurrent_test()
        if concurrent_metric.values:
            self.metrics.append(concurrent_metric)

        latency_metric = await self._run_latency_test()
        if latency_metric.values:
            self.metrics.append(latency_metric)

        reliability_metric = await self._run_reliability_test()
        if reliability_metric.values:
            self.metrics.append(reliability_metric)

        duration = time.time() - start_time

        # 判断是否通过
        passed = len(self.errors) == 0 and len(self.metrics) >= 2

        result = BenchmarkResult(
            timestamp=datetime.utcnow().isoformat(),
            version=self.VERSION,
            environment=self._get_environment(),
            metrics=self.metrics,
            duration=duration,
            passed=passed,
            errors=self.errors,
        )

        return result

    def _generate_json_report(self, result: BenchmarkResult) -> Path:
        """生成 JSON 报告"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"benchmark_{timestamp}.json"

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"📄 JSON 报告: {report_file}")
        return report_file

    def _generate_markdown_report(self, result: BenchmarkResult) -> Path:
        """生成 Markdown 报告"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"benchmark_{timestamp}.md"

        lines = [
            "# WebSocket 性能基准测试报告",
            "",
            f"**版本**: {result.version}",
            f"**时间**: {result.timestamp}",
            f"**耗时**: {result.duration:.1f}s",
            f"**状态**: {'✅ 通过' if result.passed else '❌ 失败'}",
            "",
            "## 环境信息",
            "",
            f"- Python: {result.environment.get('python_version', 'N/A').split()[0]}",
            f"- 平台: {result.environment.get('platform', 'N/A')}",
            f"- 架构: {result.environment.get('machine', 'N/A')}",
            f"- 测试模式: {result.environment.get('test_mode', 'N/A')}",
            "",
            "## 性能指标",
            "",
        ]

        for metric in result.metrics:
            lines.extend(
                [
                    f"### {metric.name}",
                    "",
                    f"**描述**: {metric.description}",
                    "",
                    "| 指标 | 值 |",
                    "|------|-----|",
                    f"| 样本数 | {metric.count} |",
                    f"| 平均值 | {metric.avg:.2f} {metric.unit} |",
                    f"| 最小值 | {metric.min:.2f} {metric.unit} |",
                    f"| 最大值 | {metric.max:.2f} {metric.unit} |",
                    f"| P50 | {metric.p50:.2f} {metric.unit} |",
                    f"| P95 | {metric.p95:.2f} {metric.unit} |",
                    f"| P99 | {metric.p99:.2f} {metric.unit} |",
                    "",
                ]
            )

        if result.errors:
            lines.extend(
                [
                    "## 错误",
                    "",
                ]
            )
            for error in result.errors:
                lines.append(f"- ❌ {error}")
            lines.append("")

        lines.extend(
            [
                "## 基准标准",
                "",
                "| 指标 | 目标 | 状态 |",
                "|------|------|------|",
                "| 并发连接 | ≥1000 | 待验证 |",
                "| 消息延迟 P99 | <100ms | 待验证 |",
                "| 心跳成功率 | ≥99% | 待验证 |",
                "",
                "---",
                "",
                "*由 performance benchmark 自动生成*",
            ]
        )

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"📄 Markdown 报告: {report_file}")
        return report_file

    def _compare_with_baseline(self, result: BenchmarkResult, baseline_file: str) -> None:
        """与基线对比"""
        baseline_path = Path(baseline_file)
        if not baseline_path.exists():
            logger.warning(f"基线文件不存在: {baseline_file}")
            return

        try:
            with open(baseline_path, encoding="utf-8") as f:
                baseline = json.load(f)

            logger.info("")
            logger.info("=" * 60)
            logger.info("性能回归检测")
            logger.info("=" * 60)

            for metric in result.metrics:
                baseline_metric = next(
                    (m for m in baseline.get("metrics", []) if m["name"] == metric.name),
                    None,
                )

                if baseline_metric:
                    baseline_avg = baseline_metric.get("avg", 0)
                    current_avg = metric.avg

                    if baseline_avg > 0:
                        change_pct = ((current_avg - baseline_avg) / baseline_avg) * 100

                        if change_pct > 10:
                            status = "⚠️  退化"
                        elif change_pct < -10:
                            status = "✅ 提升"
                        else:
                            status = "➡️  持平"

                        logger.info(
                            f"{metric.name}: {baseline_avg:.2f} -> {current_avg:.2f} ({change_pct:+.1f}%) {status}"
                        )
                    else:
                        logger.info(f"{metric.name}: 基线值为 0，无法比较")
                else:
                    logger.info(f"{metric.name}: 无基线数据")

        except Exception as e:
            logger.error(f"对比基线失败: {e}")


async def main():
    parser = argparse.ArgumentParser(description="WebSocket 性能基准测试")
    parser.add_argument("--url", default="ws://localhost:18008", help="WebSocket URL")
    parser.add_argument("--output-dir", default="tests/performance/reports", help="报告输出目录")
    parser.add_argument("--report", action="store_true", help="生成报告")
    parser.add_argument("--compare", help="与基线文件对比")
    parser.add_argument("--quick", action="store_true", help="快速模式（减少测试规模）")
    parser.add_argument("--full", action="store_true", help="完整模式（增加测试规模）")

    args = parser.parse_args()

    benchmark = PerformanceBenchmark(
        base_url=args.url,
        output_dir=args.output_dir,
        quick=args.quick,
        full=args.full,
    )

    result = await benchmark.run()

    # 生成报告
    if args.report or args.compare:
        json_file = benchmark._generate_json_report(result)
        benchmark._generate_markdown_report(result)

        # 对比基线
        if args.compare:
            benchmark._compare_with_baseline(result, args.compare)

    # 输出摘要
    logger.info("")
    logger.info("=" * 60)
    logger.info("测试摘要")
    logger.info("=" * 60)
    logger.info(f"总耗时: {result.duration:.1f}s")
    logger.info(f"测试项: {len(result.metrics)}")
    logger.info(f"错误数: {len(result.errors)}")
    logger.info(f"状态: {'✅ 通过' if result.passed else '❌ 失败'}")

    if result.errors:
        logger.info("")
        logger.info("错误详情:")
        for error in result.errors:
            logger.info(f"  ❌ {error}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
