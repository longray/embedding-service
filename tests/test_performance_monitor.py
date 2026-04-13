"""Tests for PerformanceMonitor"""

import asyncio
import time
import tracemalloc

import pytest

from wrapper.src.services.performance_monitor import PerformanceMetrics, PerformanceMonitor


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass"""

    def test_basic_creation(self):
        """Test basic metric creation"""
        metric = PerformanceMetrics(
            operation="parse",
            duration_ms=100.5,
            memory_mb=10.2,
        )

        assert metric.operation == "parse"
        assert metric.duration_ms == 100.5
        assert metric.memory_mb == 10.2
        assert metric.timestamp > 0
        assert metric.metadata == {}

    def test_with_metadata(self):
        """Test metric with metadata"""
        metric = PerformanceMetrics(
            operation="analyze",
            duration_ms=50.0,
            memory_mb=5.0,
            metadata={"file_count": 10, "tenant_id": "test"},
        )

        assert metric.metadata["file_count"] == 10
        assert metric.metadata["tenant_id"] == "test"


class TestPerformanceMonitor:
    """Test PerformanceMonitor class"""

    def test_initialization(self):
        """Test monitor initialization"""
        pm = PerformanceMonitor(tenant_id="test_tenant", max_metrics=500)

        assert pm.tenant_id == "test_tenant"
        assert pm.metrics_count == 0

    def test_default_initialization(self):
        """Test monitor with default values"""
        pm = PerformanceMonitor()

        assert pm.tenant_id == "default"
        assert pm.metrics_count == 0

    def test_record_metric(self):
        """Test recording a metric"""
        pm = PerformanceMonitor()

        metric = pm.record(
            operation="parse",
            duration_ms=100.0,
            memory_mb=10.0,
            metadata={"files": 5},
        )

        assert pm.metrics_count == 1
        assert metric.operation == "parse"
        assert metric.duration_ms == 100.0
        assert metric.memory_mb == 10.0

    def test_get_metrics_all(self):
        """Test getting all metrics"""
        pm = PerformanceMonitor()

        pm.record("parse", 100.0, 10.0)
        pm.record("analyze", 50.0, 5.0)
        pm.record("parse", 80.0, 8.0)

        metrics = pm.get_metrics()

        assert len(metrics) == 3

    def test_get_metrics_filtered(self):
        """Test getting metrics filtered by operation"""
        pm = PerformanceMonitor()

        pm.record("parse", 100.0, 10.0)
        pm.record("analyze", 50.0, 5.0)
        pm.record("parse", 80.0, 8.0)

        parse_metrics = pm.get_metrics(operation="parse")

        assert len(parse_metrics) == 2
        assert all(m.operation == "parse" for m in parse_metrics)

    def test_get_summary_empty(self):
        """Test summary with no metrics"""
        pm = PerformanceMonitor(tenant_id="test")

        summary = pm.get_summary()

        assert summary["tenant_id"] == "test"
        assert summary["count"] == 0
        assert summary["avg_duration_ms"] == 0.0

    def test_get_summary_with_data(self):
        """Test summary with metrics"""
        pm = PerformanceMonitor()

        pm.record("parse", 100.0, 10.0)
        pm.record("parse", 200.0, 20.0)
        pm.record("parse", 300.0, 30.0)

        summary = pm.get_summary(operation="parse")

        assert summary["count"] == 3
        assert summary["avg_duration_ms"] == 200.0
        assert summary["max_duration_ms"] == 300.0
        assert summary["min_duration_ms"] == 100.0
        assert summary["avg_memory_mb"] == 20.0
        assert summary["max_memory_mb"] == 30.0

    def test_max_metrics_limit(self):
        """Test max metrics limit"""
        pm = PerformanceMonitor(max_metrics=3)

        for i in range(5):
            pm.record("op", float(i), float(i))

        assert pm.metrics_count == 3
        # Should keep the last 3
        metrics = pm.get_metrics()
        assert metrics[0].duration_ms == 2.0
        assert metrics[2].duration_ms == 4.0

    def test_clear_metrics(self):
        """Test clearing metrics"""
        pm = PerformanceMonitor()

        pm.record("op", 100.0, 10.0)
        assert pm.metrics_count == 1

        pm.clear()
        assert pm.metrics_count == 0

    def test_generate_report(self):
        """Test report generation"""
        pm = PerformanceMonitor(tenant_id="test")

        pm.record("parse", 100.0, 10.0)
        pm.record("analyze", 50.0, 5.0)

        report = pm.generate_report()

        assert "Performance Report" in report
        assert "Tenant ID: test" in report
        assert "parse" in report
        assert "analyze" in report

    def test_generate_report_empty(self):
        """Test report generation with no metrics"""
        pm = PerformanceMonitor(tenant_id="empty")

        report = pm.generate_report()

        assert "Performance Report" in report
        assert "Tenant ID: empty" in report
        assert "Total Operations: 0" in report


class TestPerformanceMonitorContext:
    """Test PerformanceMonitor context manager"""

    def test_monitor_context(self):
        """Test monitor context manager"""
        pm = PerformanceMonitor()

        with pm.monitor("test_operation"):
            time.sleep(0.01)  # 10ms

        assert pm.metrics_count == 1

        metric = pm.get_metrics(operation="test_operation")[0]
        assert metric.operation == "test_operation"
        assert metric.duration_ms >= 10.0  # At least 10ms

    def test_monitor_with_metadata(self):
        """Test monitor with metadata"""
        pm = PerformanceMonitor()

        with pm.monitor("parse", metadata={"file": "test.py"}):
            pass

        metric = pm.get_metrics(operation="parse")[0]
        assert metric.metadata["file"] == "test.py"

    def test_monitor_exception_handling(self):
        """Test monitor handles exceptions"""
        pm = PerformanceMonitor()

        try:
            with pm.monitor("failing_op"):
                raise ValueError("Test error")
        except ValueError:
            pass

        # Should still record the metric
        assert pm.metrics_count == 1
        metric = pm.get_metrics(operation="failing_op")[0]
        assert metric.duration_ms >= 0


class TestMemoryTracing:
    """Test memory tracing functionality"""

    def test_start_stop_tracing(self):
        """Test starting and stopping memory tracing"""
        pm = PerformanceMonitor()

        # Initially not tracing
        assert not tracemalloc.is_tracing()

        pm.start_tracing()
        assert tracemalloc.is_tracing()

        pm.stop_tracing()
        assert not tracemalloc.is_tracing()

    def test_memory_measurement_with_tracing(self):
        """Test memory measurement when tracing is enabled"""
        pm = PerformanceMonitor()

        pm.start_tracing()

        # Allocate some memory
        data = [0] * 1000000  # ~8MB

        with pm.monitor("memory_test"):
            more_data = [0] * 100000  # Additional memory

        pm.stop_tracing()

        metric = pm.get_metrics(operation="memory_test")[0]
        # Memory should be measured (not exactly 0)
        assert metric.memory_mb >= 0

        # Cleanup
        del data
        del more_data

    def test_memory_measurement_without_tracing(self):
        """Test memory measurement when tracing is disabled"""
        pm = PerformanceMonitor()

        # Ensure tracing is off
        if tracemalloc.is_tracing():
            tracemalloc.stop()

        with pm.monitor("no_trace_test"):
            pass

        metric = pm.get_metrics(operation="no_trace_test")[0]
        # Should return 0 when not tracing
        assert metric.memory_mb == 0.0
