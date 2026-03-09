"""
统一异常处理机制
"""

from typing import Optional


class WrapperServiceError(Exception):
    """包装服务基础异常"""

    def __init__(self, message: str, status_code: int = 500, details: Optional[dict] = None):
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


class ValidationError(WrapperServiceError):
    """验证错误异常"""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message=message, status_code=400, details=details)


class EmbeddingError(WrapperServiceError):
    """Embedding服务错误"""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message=message, status_code=502, details=details)


class DatabaseError(WrapperServiceError):
    """数据库操作错误"""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message=message, status_code=500, details=details)
