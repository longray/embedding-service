"""性能基准测试单元测试

验证 benchmark.py 的核心功能。
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add tests/performance to path
sys.path.insert(0, str(Path(__file__).parent / "performance"))

from benchmark import BenchmarkMetric, BenchmarkResult, PerformanceBenchmark


class TestBenchmarkMetric:
    """测试 BenchmarkMetric 类"""

    def test_basic_properties(self):
        """测试基本属性"""
        metric = BenchmarkMetric(name="test", description="Test metric", unit="ms")
        metric.values = [10, 20, 30, 40, 50]

        assert metric.count == 5
        assert metric.avg == 30.0
        assert metric.min == 10
        assert metric.max == 50
        assert metric.p50 == 30
        assert metric.p95 == 50
        assert metric.p99 == 50

    def test_empty_values(self):
        """测试空值处理"""
        metric = BenchmarkMetric(name="empty", description="Empty metric", unit="ms")

        assert metric.count == 0
        assert metric.avg == 0
        assert metric.min == 0
        assert metric.max == 0
        assert metric.p50 == 0

    def test_to_dict(self):
        """测试字典转换"""
        metric = BenchmarkMetric(name="test", description="Test", unit="ms")
        metric.values = [10, 20, 30]

        d = metric.to_dict()
        assert d["name"] == "test"
        assert d["description"] == "Test"
        assert d["unit"] == "ms"
        assert d["count"] == 3
        assert d["avg"] == 20.0
        assert d["min"] == 10
        assert d["max"] == 30


class TestBenchmarkResult:
    """测试 BenchmarkResult 类"""

    def test_to_dict(self):
        """测试结果字典转换"""
        metric = BenchmarkMetric(name="test", description="Test", unit="ms")
        metric.values = [10, 20, 30]

        result = BenchmarkResult(
            timestamp="2024-01-01T00:00:00",
            version="3.2.0",
            environment={"test": "env"},
            metrics=[metric],
            duration=10.5,
            passed=True,
            errors=[],
        )

        d = result.to_dict()
        assert d["timestamp"] == "2024-01-01T00:00:00"
        assert d["version"] == "3.2.0"
        assert d["duration"] == 10.5
        assert d["passed"] is True
        assert len(d["metrics"]) == 1
        assert d["environment"]["test"] == "env"


class TestPerformanceBenchmark:
    """测试 PerformanceBenchmark 类"""

    def test_init_standard(self):
        """测试标准模式初始化"""
        benchmark = PerformanceBenchmark()

        assert benchmark.quick is False
        assert benchmark.full is False
        assert benchmark.concurrent_clients == 100
        assert benchmark.concurrent_duration == 30

    def test_init_quick(self):
        """测试快速模式初始化"""
        benchmark = PerformanceBenchmark(quick=True)

        assert benchmark.quick is True
        assert benchmark.concurrent_clients == 50
        assert benchmark.concurrent_duration == 10

    def test_init_full(self):
        """测试完整模式初始化"""
        benchmark = PerformanceBenchmark(full=True)

        assert benchmark.full is True
        assert benchmark.concurrent_clients == 1000
        assert benchmark.concurrent_duration == 300

    def test_get_environment(self):
        """测试环境信息获取"""
        benchmark = PerformanceBenchmark()
        env = benchmark._get_environment()

        assert "python_version" in env
        assert "platform" in env
        assert "base_url" in env
        assert "test_mode" in env

    def test_generate_markdown_report(self, tmp_path):
        """测试 Markdown 报告生成"""
        benchmark = PerformanceBenchmark(output_dir=str(tmp_path))

        metric = BenchmarkMetric(name="test", description="Test", unit="ms")
        metric.values = [10, 20, 30]

        result = BenchmarkResult(
            timestamp="2024-01-01T00:00:00",
            version="3.2.0",
            environment={},
            metrics=[metric],
            duration=10.0,
            passed=True,
            errors=[],
        )

        report_file = benchmark._generate_markdown_report(result)

        assert report_file.exists()
        content = report_file.read_text(encoding="utf-8")
        assert "# WebSocket 性能基准测试报告" in content
        assert "test" in content
        assert "20.0" in content

    def test_generate_json_report(self, tmp_path):
        """测试 JSON 报告生成"""
        benchmark = PerformanceBenchmark(output_dir=str(tmp_path))

        metric = BenchmarkMetric(name="test", description="Test", unit="ms")
        metric.values = [10, 20, 30]

        result = BenchmarkResult(
            timestamp="2024-01-01T00:00:00",
            version="3.2.0",
            environment={"platform": "test"},
            metrics=[metric],
            duration=10.0,
            passed=True,
            errors=[],
        )

        report_file = benchmark._generate_json_report(result)

        assert report_file.exists()
        data = json.loads(report_file.read_text(encoding="utf-8"))
        assert data["version"] == "3.2.0"
        assert data["passed"] is True
        assert len(data["metrics"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
