"""
Prometheus监控指标
"""

from prometheus_client import Counter, Histogram, Gauge, Info
from functools import wraps
import time


# 请求指标
request_count = Counter(
    "wrapper_requests_total", "Total request count", ["method", "endpoint", "status"]
)

request_duration = Histogram(
    "wrapper_request_duration_seconds",
    "Request duration in seconds",
    ["method", "endpoint"],
)

# 缓存指标
cache_hits = Counter("wrapper_cache_hits_total", "Cache hit count")
cache_misses = Counter("wrapper_cache_misses_total", "Cache miss count")

# 熔断器指标
circuit_breaker_state = Gauge(
    "wrapper_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=half_open, 2=open)",
    ["service"],
)

circuit_breaker_failures = Counter(
    "wrapper_circuit_breaker_failures_total",
    "Circuit breaker failure count",
    ["service"],
)

# 后端服务指标
backend_request_duration = Histogram(
    "wrapper_backend_request_duration_seconds", "Backend request duration", ["service"]
)

backend_errors = Counter(
    "wrapper_backend_errors_total", "Backend error count", ["service", "error_type"]
)

# 系统信息
service_info = Info("wrapper_service", "Service information")


def track_request(method: str, endpoint: str):
    """请求追踪装饰器"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                request_count.labels(
                    method=method, endpoint=endpoint, status=status
                ).inc()
                request_duration.labels(method=method, endpoint=endpoint).observe(
                    duration
                )

        return wrapper

    return decorator
