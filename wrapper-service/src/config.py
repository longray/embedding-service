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
    port: int = 3001
    wrapper_host: str = "0.0.0.0"  # nosec B104

    # 后端服务配置
    embedding_service_url: str = "http://localhost:18000"
    llm_service_url: str = "http://localhost:18001"

    # SurrealDB配置
    surrealdb_url: str = "ws://localhost:8000/rpc"
    surrealdb_namespace: str = "memory_ns"
    surrealdb_database: str = "memory_db"
    surrealdb_username: str = "root"
    surrealdb_password: str = "root"
    surrealdb_pool_size: int = 10
    surrealdb_max_overflow: int = 5

    # 超时配置
    http_timeout: float = 30.0
    http_connect_timeout: float = 5.0

    # 缓存配置
    cache_enabled: bool = True
    cache_max_size: int = 1000
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
    json_logs: bool = False

    # 认证配置（P2新增）
    auth_enabled: bool = False  # 默认关闭，向后兼容
    api_key_header: str = "X-API-Key"  # API Key请求头名称
    # API Keys格式: "key1:read,key2:read;write,key3:admin"
    api_keys: str = ""  # 从环境变量读取，格式见上

    @property
    def parsed_api_keys(self) -> dict[str, list[str]]:
        """解析API Keys配置为字典"""
        if not self.api_keys:
            return {}
        result: dict[str, list[str]] = {}
        # 支持逗号分割的多条配置，例如 "k1:read;write,k2:admin"
        for key_config in self.api_keys.split(","):
            if ":" not in key_config:
                continue
            key, perms = key_config.split(":", 1)
            result[key.strip()] = [p.strip() for p in perms.split(";") if p.strip()]
        return result

    class Config:
        env_prefix = "WRAPPER_"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()
