"""PerformanceMonitor - 性能监控模块

提供性能指标收集和监控功能，支持：
- 耗时测量（parse_time, analysis_time）
- 内存使用监控
- 性能报告生成
- 上下文管理器支持
"""

import logging
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标数据类

    Attributes:
        operation: 操作名称
        duration_ms: 耗时（毫秒）
        memory_mb: 内存使用（MB）
        timestamp: 时间戳
        metadata: 额外元数据
    """

    operation: str
    duration_ms: float
    memory_mb: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """性能监控器

    提供性能指标收集和监控功能。

    Attributes:
        tenant_id: 租户 ID
        metrics: 收集的指标列表
        max_metrics: 最大保留指标数
    """

    def __init__(self, tenant_id: str = "default", max_metrics: int = 1000):
        """初始化性能监控器

        Args:
            tenant_id: 租户 ID，默认 "default"
            max_metrics: 最大保留指标数，默认 1000
        """
        self._tenant_id = tenant_id
        self._max_metrics = max_metrics
        self._metrics: List[PerformanceMetrics] = []
        self._logger = logging.getLogger(f"{__name__}.{tenant_id}")

        self._logger.debug(
            "[PerformanceMonitor] 初始化: tenant_id=%s, max_metrics=%d",
            tenant_id,
            max_metrics,
        )

    def record(
        self,
        operation: str,
        duration_ms: float,
        memory_mb: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PerformanceMetrics:
        """记录性能指标

        Args:
            operation: 操作名称
            duration_ms: 耗时（毫秒）
            memory_mb: 内存使用（MB）
            metadata: 额外元数据

        Returns:
            记录的指标对象
        """
        metric = PerformanceMetrics(
            operation=operation,
            duration_ms=duration_ms,
            memory_mb=memory_mb,
            metadata=metadata or {},
        )

        self._metrics.append(metric)

        # 限制指标数量
        if len(self._metrics) > self._max_metrics:
            self._metrics = self._metrics[-self._max_metrics :]

        self._logger.debug(
            "[PerformanceMonitor] 记录指标: operation=%s, duration=%.2fms, memory=%.2fMB",
            operation,
            duration_ms,
            memory_mb,
        )

        return metric

    @contextmanager
    def monitor(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """监控上下文管理器

        使用示例:
            with pm.monitor("parse"):
                await parse_code(content)

        Args:
            operation: 操作名称
            metadata: 额外元数据

        Yields:
            None
        """
        start_time = time.perf_counter()
        start_memory = self._get_current_memory_mb()

        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = self._get_current_memory_mb()

            duration_ms = (end_time - start_time) * 1000
            memory_mb = end_memory - start_memory

            self.record(
                operation=operation,
                duration_ms=duration_ms,
                memory_mb=memory_mb,
                metadata=metadata,
            )

    def _get_current_memory_mb(self) -> float:
        """获取当前内存使用（MB）"""
        try:
            # 使用 tracemalloc 获取内存使用
            if tracemalloc.is_tracing():
                current, _ = tracemalloc.get_traced_memory()
                return current / (1024 * 1024)
            else:
                # 如果未启用 tracemalloc，返回 0
                return 0.0
        except Exception:
            return 0.0

    def get_metrics(self, operation: Optional[str] = None) -> List[PerformanceMetrics]:
        """获取性能指标

        Args:
            operation: 操作名称过滤，None 返回所有

        Returns:
            指标列表
        """
        if operation is None:
            return self._metrics.copy()

        return [m for m in self._metrics if m.operation == operation]

    def get_summary(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """获取性能摘要

        Args:
            operation: 操作名称过滤，None 返回所有

        Returns:
            性能摘要字典
        """
        metrics = self.get_metrics(operation)

        if not metrics:
            return {
                "tenant_id": self._tenant_id,
                "operation": operation or "all",
                "count": 0,
                "avg_duration_ms": 0.0,
                "max_duration_ms": 0.0,
                "min_duration_ms": 0.0,
                "avg_memory_mb": 0.0,
                "max_memory_mb": 0.0,
            }

        durations = [m.duration_ms for m in metrics]
        memories = [m.memory_mb for m in metrics]

        return {
            "tenant_id": self._tenant_id,
            "operation": operation or "all",
            "count": len(metrics),
            "avg_duration_ms": sum(durations) / len(durations),
            "max_duration_ms": max(durations),
            "min_duration_ms": min(durations),
            "avg_memory_mb": sum(memories) / len(memories) if memories else 0.0,
            "max_memory_mb": max(memories) if memories else 0.0,
        }

    def generate_report(self) -> str:
        """生成性能报告

        Returns:
            格式化的性能报告字符串
        """
        summary = self.get_summary()

        lines = [
            "=" * 60,
            "Performance Report",
            "=" * 60,
            f"Tenant ID: {self._tenant_id}",
            f"Total Operations: {summary['count']}",
            "",
            "Duration Statistics:",
            f"  Average: {summary['avg_duration_ms']:.2f} ms",
            f"  Maximum: {summary['max_duration_ms']:.2f} ms",
            f"  Minimum: {summary['min_duration_ms']:.2f} ms",
            "",
            "Memory Statistics:",
            f"  Average: {summary['avg_memory_mb']:.2f} MB",
            f"  Maximum: {summary['max_memory_mb']:.2f} MB",
            "",
            "Operation Breakdown:",
        ]

        # 按操作分组统计
        operations = set(m.operation for m in self._metrics)
        for op in sorted(operations):
            op_summary = self.get_summary(op)
            lines.extend(
                [
                    f"  {op}:",
                    f"    Count: {op_summary['count']}",
                    f"    Avg Duration: {op_summary['avg_duration_ms']:.2f} ms",
                    f"    Avg Memory: {op_summary['avg_memory_mb']:.2f} MB",
                ]
            )

        lines.append("=" * 60)

        return "\n".join(lines)

    def clear(self) -> None:
        """清除所有指标"""
        self._metrics.clear()
        self._logger.debug("[PerformanceMonitor] 指标已清除")

    def start_tracing(self) -> None:
        """启动内存追踪"""
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._logger.debug("[PerformanceMonitor] 内存追踪已启动")

    def stop_tracing(self) -> None:
        """停止内存追踪"""
        if tracemalloc.is_tracing():
            tracemalloc.stop()
            self._logger.debug("[PerformanceMonitor] 内存追踪已停止")

    @property
    def tenant_id(self) -> str:
        """租户 ID"""
        return self._tenant_id

    @property
    def metrics_count(self) -> int:
        """当前指标数量"""
        return len(self._metrics)
