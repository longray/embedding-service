"""OpenTelemetry 分布式追踪集成

按需初始化 OpenTelemetry，支持：
- FastAPI 自动 instrumentation（请求级别 span）
- httpx 客户端追踪（下游服务调用 span）
- 自定义 span（SurrealDB 查询、业务操作）
- OTLP gRPC 导出到 Jaeger/OTEL Collector

环境变量控制：
    WRAPPER_OTEL_ENABLED=true       # 开启追踪
    WRAPPER_OTEL_ENDPOINT=...       # OTLP gRPC 端点
    WRAPPER_OTEL_SERVICE_NAME=...   # 服务名
    WRAPPER_OTEL_SAMPLE_RATE=1.0    # 采样率
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

    from ..config import TelemetryConfig

logger = logging.getLogger(__name__)

# 追踪器单例（惰性初始化）
_tracer = None
_initialized = False


def init_tracing(app: FastAPI, telemetry_config: TelemetryConfig) -> bool:
    """初始化 OpenTelemetry 追踪

    Args:
        app: FastAPI 应用实例
        telemetry_config: 追踪配置

    Returns:
        是否成功初始化
    """
    global _tracer, _initialized

    if not telemetry_config.enabled:
        logger.info("[Tracing] OpenTelemetry 已禁用 (WRAPPER_OTEL_ENABLED=false)")
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

        # 资源标识
        resource = Resource.create(
            {
                "service.name": telemetry_config.service_name,
                "service.version": "2.1.0",
                "deployment.environment": "development",
            }
        )

        # 采样器
        sampler = TraceIdRatioBased(telemetry_config.sample_rate)

        # TracerProvider
        provider = TracerProvider(resource=resource, sampler=sampler)

        # OTLP gRPC 导出器
        exporter = OTLPSpanExporter(
            endpoint=telemetry_config.otlp_endpoint,
            insecure=True,  # 本地开发不需要 TLS
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))

        # 设置全局 TracerProvider
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(telemetry_config.service_name, "2.1.0")

        # 自动 instrument FastAPI（请求级别 span）
        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls="health",  # 健康检查不追踪
        )

        # 自动 instrument httpx（下游服务调用 span）
        HTTPXClientInstrumentor().instrument()

        _initialized = True
        logger.info(
            "[Tracing] OpenTelemetry 已初始化 | "
            f"endpoint={telemetry_config.otlp_endpoint} | "
            f"sample_rate={telemetry_config.sample_rate}"
        )
        return True

    except ImportError as e:
        logger.warning(f"[Tracing] OpenTelemetry 依赖未安装，跳过: {e}")
        return False
    except Exception as e:
        logger.warning(f"[Tracing] OpenTelemetry 初始化失败，服务继续运行: {e}")
        return False


def shutdown_tracing() -> None:
    """关闭 OpenTelemetry，确保所有 span 被导出"""
    global _initialized

    if not _initialized:
        return

    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        if hasattr(provider, "shutdown"):
            provider.shutdown()  # type: ignore[reportAttributeAccessIssue]
        _initialized = False
        logger.info("[Tracing] OpenTelemetry 已关闭")
    except Exception as e:
        logger.warning(f"[Tracing] 关闭异常: {e}")


def get_tracer():
    """获取追踪器实例（未初始化时返回 NoOp tracer）"""
    global _tracer

    if _tracer is not None:
        return _tracer

    # 返回 NoOp tracer（追踪未启用时不影响业务）
    try:
        from opentelemetry import trace

        return trace.get_tracer("embedding-wrapper", "2.1.0")
    except ImportError:
        return _NoOpTracer()


class _NoOpSpan:
    """无操作 Span — 当 OpenTelemetry 未安装时使用"""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_attribute(self, key, value):
        pass

    def set_status(self, status):
        pass

    def record_exception(self, exception):
        pass

    def add_event(self, name, attributes=None):
        pass


class _NoOpTracer:
    """无操作 Tracer — 当 OpenTelemetry 未安装时使用"""

    def start_as_current_span(self, name, **kwargs):
        return _NoOpSpan()
