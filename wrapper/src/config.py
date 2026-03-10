"""配置管理模块"""

import os
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class CacheConfig:
    enabled: bool = True
    max_size: int = 1000
    ttl_seconds: int = 3600


@dataclass
class HTTPConfig:
    max_connections: int = 100
    max_keepalive_connections: int = 20
    timeout: float = 30.0
    connect_timeout: float = 5.0
    max_retries: int = 3


@dataclass
class SurrealDBConfig:
    url: str = "ws://localhost:8000/rpc"
    namespace: str = "memory_ns"
    database: str = "memory_db"
    username: str = "root"
    password: str = "root"


@dataclass
class ServiceConfig:
    embedding_service_url: str = "http://localhost:18000"
    llm_service_url: str = "http://localhost:18001"


@dataclass
class SearchConfig:
    keyword_threshold: float = 0.0
    vector_threshold: float = 0.75
    hybrid_threshold: float = 0.75


@dataclass
class AppConfig:
    host: str = "0.0.0.0"  # nosec B104 - 容器环境需要绑定所有接口
    port: int = 17999
    debug: bool = False
    cache: CacheConfig = field(default_factory=CacheConfig)
    http: HTTPConfig = field(default_factory=HTTPConfig)
    surrealdb: SurrealDBConfig = field(default_factory=SurrealDBConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)
    search: SearchConfig = field(default_factory=SearchConfig)


def _load_env_file() -> None:
    project_root = Path(__file__).resolve().parents[2]
    candidates = [project_root / ".env", project_root / "wrapper" / ".env"]

    for env_path in candidates:
        if not env_path.exists() or not env_path.is_file():
            continue

        content = env_path.read_text(encoding="utf-8")
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)


def load_config():
    _load_env_file()
    cfg = AppConfig()
    cfg.host = os.getenv("WRAPPER_HOST", cfg.host)
    cfg.port = int(os.getenv("WRAPPER_PORT", str(cfg.port)))
    cfg.cache.enabled = os.getenv("WRAPPER_CACHE_ENABLED", "true").lower() == "true"
    cfg.service.embedding_service_url = os.getenv("WRAPPER_EMBEDDING_SERVICE_URL", cfg.service.embedding_service_url)
    cfg.surrealdb.url = os.getenv("WRAPPER_SURREALDB_URL", cfg.surrealdb.url)
    cfg.search.keyword_threshold = float(
        os.getenv("WRAPPER_SEARCH_KEYWORD_THRESHOLD", str(cfg.search.keyword_threshold))
    )
    cfg.search.vector_threshold = float(os.getenv("WRAPPER_SEARCH_VECTOR_THRESHOLD", str(cfg.search.vector_threshold)))
    cfg.search.hybrid_threshold = float(os.getenv("WRAPPER_SEARCH_HYBRID_THRESHOLD", str(cfg.search.hybrid_threshold)))
    return cfg


config = load_config()
