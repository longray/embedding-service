"""
包装服务配置管理模块

修复了原设计中的问题：
1. 使用 pydantic_settings 正确管理配置
2. 使用依赖注入模式
3. 支持环境变量覆盖
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """应用配置（从环境变量加载）"""

    # 服务配置
    wrapper_port: int = 3001
    wrapper_host: str = "0.0.0.0"

    # 后端服务配置
    embedding_service_url: str = "http://localhost:18000"
    llm_service_url: str = "http://localhost:18001"

    # 超时配置
    http_timeout: float = 30.0
    http_connect_timeout: float = 5.0

    # 缓存配置
    cache_enabled: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600  # ✅ 修复：使用正确的变量名 ttl

    # 限流配置
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

    # 熔断配置
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60

    # 日志配置
    log_level: str = "INFO"
    log_format: str = "json"

    class Config:
        env_prefix = "WRAPPER_"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()
