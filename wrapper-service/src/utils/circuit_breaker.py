"""
熔断器实现 - 防止级联故障
"""

from enum import Enum
from typing import Callable, Any
import time
import threading
from functools import wraps


class CircuitState(Enum):
    """熔断器状态"""

    CLOSED = "closed"  # 正常状态
    OPEN = "open"  # 熔断状态
    HALF_OPEN = "half_open"  # 半开状态


class CircuitBreaker:
    """
    简单但有效的熔断器实现

    特性：
    - 三状态机制（CLOSED/OPEN/HALF_OPEN）
    - 失败计数和阈值判断
    - 超时自动恢复
    - 线程安全
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        half_open_max_calls: int = 3,
    ):
        self._failure_threshold = failure_threshold
        self._timeout = timeout
        self._half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0
        self._lock = threading.RLock()

    @property
    def state(self) -> CircuitState:
        """获取当前状态"""
        with self._lock:
            return self._state

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        执行函数调用，带熔断保护

        Args:
            func: 要执行的函数
            *args, **kwargs: 函数参数

        Returns:
            函数执行结果

        Raises:
            CircuitBreakerError: 熔断器打开时
        """
        with self._lock:
            # 检查是否需要从OPEN转到HALF_OPEN
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                else:
                    from .exceptions import CircuitBreakerError

                    raise CircuitBreakerError("Circuit breaker is OPEN")

            # HALF_OPEN状态下限制调用次数
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self._half_open_max_calls:
                    from .exceptions import CircuitBreakerError

                    raise CircuitBreakerError("Circuit breaker HALF_OPEN limit reached")
                self._half_open_calls += 1

        # 执行函数
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """判断是否应该尝试重置"""
        if self._last_failure_time is None:
            return True
        return time.time() - self._last_failure_time >= self._timeout

    def _on_success(self):
        """成功回调"""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                # HALF_OPEN状态下成功，转为CLOSED
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._half_open_calls = 0
            elif self._state == CircuitState.CLOSED:
                # CLOSED状态下成功，重置失败计数
                self._failure_count = 0

    def _on_failure(self):
        """失败回调"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # HALF_OPEN状态下失败，立即转为OPEN
                self._state = CircuitState.OPEN
            elif self._failure_count >= self._failure_threshold:
                # CLOSED状态下失败次数达到阈值，转为OPEN
                self._state = CircuitState.OPEN

    def reset(self):
        """手动重置熔断器"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_calls = 0
            self._last_failure_time = None


def circuit_breaker(
    failure_threshold: int = 5, timeout: float = 60.0, half_open_max_calls: int = 3
):
    """
    熔断器装饰器

    Usage:
        @circuit_breaker(failure_threshold=5, timeout=60.0)
        async def call_external_service():
            ...
    """
    cb = CircuitBreaker(failure_threshold, timeout, half_open_max_calls)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return cb.call(func, *args, **kwargs)

        # 暴露熔断器实例
        wrapper.circuit_breaker = cb
        return wrapper

    return decorator
