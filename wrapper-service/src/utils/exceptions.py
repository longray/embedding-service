"""
统一异常处理机制

修复了原设计中的问题：
1. 创建自定义异常类层次结构
2. 实现全局异常处理器
3. 统一错误响应格式
"""

from datetime import datetime
from typing import Optional


class WrapperServiceError(Exception):
    """包装服务基础异常"""

    def __init__(
        self, message: str, status_code: int = 500, details: Optional[dict] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ServiceUnavailableError(WrapperServiceError):
    """服务不可用异常"""

    def __init__(self, service_name: str, details: Optional[dict] = None):
        super().__init__(
            message=f"Service {service_name} is unavailable",
            status_code=503,
            details=details,
        )


class CircuitBreakerOpenError(WrapperServiceError):
    """熔断器打开异常"""

    def __init__(self, service_name: str):
        super().__init__(
            message=f"Circuit breaker is open for {service_name}", status_code=503
        )


class ValidationError(WrapperServiceError):
    """验证错误异常"""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message=message, status_code=400, details=details)


class RateLimitExceededError(WrapperServiceError):
    """限流异常"""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message=message, status_code=429)
