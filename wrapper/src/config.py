"""配置管理模块"""

import os
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
    """搜索配置（新增 - SurrealDB 3.0 升级）"""

    # 阈值
    keyword_threshold: float = 0.0
    vector_threshold: float = 0.75
    hybrid_threshold: float = 0.75

    # RRF 参数
    rrf_k: int = 60  # RRF 平滑常数（Cormack et al. 2009 推荐值）
    rrf_vector_weight: float = 0.7  # 向量搜索权重
    rrf_keyword_weight: float = 0.3  # 关键词搜索权重

    # HNSW 查询参数
    hnsw_ef_search: int = 50  # HNSW 查询候选集大小

    # 多租户
    default_tenant_id: str = "default"  # API 未传 tenant_id 时使用

@dataclass
class TelemetryConfig:
    """OpenTelemetry 追踪配置"""

    enabled: bool = False  # 默认关闭，按需开启
    service_name: str = "embedding-wrapper"
    otlp_endpoint: str = "http://localhost:4317"  # Jaeger OTLP gRPC
    sample_rate: float = 1.0  # 1.0 = 全采样，生产环境可降低



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
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)


def load_config():
    cfg = AppConfig()
    cfg.host = os.getenv("WRAPPER_HOST", cfg.host)
    cfg.port = int(os.getenv("WRAPPER_PORT", str(cfg.port)))
    cfg.cache.enabled = os.getenv("WRAPPER_CACHE_ENABLED", "true").lower() == "true"
    cfg.service.embedding_service_url = os.getenv("WRAPPER_EMBEDDING_SERVICE_URL", cfg.service.embedding_service_url)
    cfg.surrealdb.url = os.getenv("WRAPPER_SURREALDB_URL", cfg.surrealdb.url)

    # 搜索配置
    cfg.search.vector_threshold = float(os.getenv("WRAPPER_SEARCH_VECTOR_THRESHOLD", str(cfg.search.vector_threshold)))
    cfg.search.hybrid_threshold = float(os.getenv("WRAPPER_SEARCH_HYBRID_THRESHOLD", str(cfg.search.hybrid_threshold)))
    cfg.search.keyword_threshold = float(
        os.getenv("WRAPPER_SEARCH_KEYWORD_THRESHOLD", str(cfg.search.keyword_threshold))
    )
    cfg.search.rrf_k = int(os.getenv("WRAPPER_SEARCH_RRF_K", str(cfg.search.rrf_k)))
    cfg.search.rrf_vector_weight = float(
        os.getenv("WRAPPER_SEARCH_RRF_VECTOR_WEIGHT", str(cfg.search.rrf_vector_weight))
    )
    cfg.search.rrf_keyword_weight = float(
        os.getenv("WRAPPER_SEARCH_RRF_KEYWORD_WEIGHT", str(cfg.search.rrf_keyword_weight))
    )
    cfg.search.hnsw_ef_search = int(os.getenv("WRAPPER_SEARCH_HNSW_EF_SEARCH", str(cfg.search.hnsw_ef_search)))
    cfg.search.default_tenant_id = os.getenv("WRAPPER_DEFAULT_TENANT_ID", cfg.search.default_tenant_id)

    # OpenTelemetry 配置
    cfg.telemetry.enabled = os.getenv("WRAPPER_OTEL_ENABLED", "false").lower() == "true"
    cfg.telemetry.service_name = os.getenv("WRAPPER_OTEL_SERVICE_NAME", cfg.telemetry.service_name)
    cfg.telemetry.otlp_endpoint = os.getenv("WRAPPER_OTEL_ENDPOINT", cfg.telemetry.otlp_endpoint)
    cfg.telemetry.sample_rate = float(os.getenv("WRAPPER_OTEL_SAMPLE_RATE", str(cfg.telemetry.sample_rate)))

    return cfg


config = load_config()
