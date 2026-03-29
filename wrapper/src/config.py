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
class TLSConfig:
    """TLS/HTTPS configuration for the wrapper service."""

    enabled: bool = False
    cert_path: str = ""
    key_path: str = ""
    # For development: generate self-signed certs
    # For production: use Let's Encrypt or purchased certificate
    min_version: str = "1.2"
    # Redirect HTTP to HTTPS
    redirect_http: bool = True


@dataclass
class SurrealDBConfig:
    url: str = "ws://localhost:18002/rpc"
    namespace: str = "memory_ns"
    database: str = "memory_db"
    username: str = "root"  # 迁移用户（OWNER 权限）
    password: str = "root"
    runtime_username: str = "runtime_user"  # 运行时用户（EDITOR 权限）
    runtime_password: str = "change_me_in_production"
    use_runtime_credentials: bool = True  # 生产环境启用权限分离


@dataclass
class ServiceConfig:
    embedding_service_url: str = "http://localhost:18000"
    # 注：LLM 服务独立运行（端口 18001），wrapper 不调用它


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

    # 动态去重阈值（Phase A-B6）
    dedup_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "preference": 0.88,  # 用户偏好（宽松，偏好会改变）
            "decision": 0.90,  # 决策记录（保留历史）
            "long-term": 0.93,  # 长期记忆（中等）
            "general": 0.95,  # 一般记忆（严格）
            "daily": 1.0,  # 日志（不去重）
        }
    )
    default_dedup_threshold: float = 0.95  # 默认严格阈值

    # 查询缓存配置（Phase A-B3）
    cache_enabled: bool = True
    cache_ttl: int = 300  # 5分钟


@dataclass
class TelemetryConfig:
    """OpenTelemetry 追踪配置"""

    enabled: bool = False  # 默认关闭，按需开启
    service_name: str = "embedding-wrapper"
    otlp_endpoint: str = "http://localhost:4317"  # Jaeger OTLP gRPC
    sample_rate: float = 1.0  # 1.0 = 全采样，生产环境可降低


@dataclass
class MeilisearchConfig:
    url: str = "http://localhost:18003"
    api_key: str = "RDo25RtbmF8BSyLyOjBgpBOH8XZo1unrbu83Gz_rX4M"
    index_name: str = "memories"
    timeout: float = 30.0
    enabled: bool = True


@dataclass
class CodeAnalysisConfig:
    enabled: bool = True
    auto_analyze: bool = False
    min_content_length: int = 50
    max_content_length: int = 50000


@dataclass
class AppConfig:
    host: str = "0.0.0.0"  # nosec B104 - 容器环境需要绑定所有接口
    port: int = 17999
    debug: bool = False
    cache: CacheConfig = field(default_factory=CacheConfig)
    http: HTTPConfig = field(default_factory=HTTPConfig)
    tls: TLSConfig = field(default_factory=TLSConfig)
    surrealdb: SurrealDBConfig = field(default_factory=SurrealDBConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    meilisearch: MeilisearchConfig = field(default_factory=MeilisearchConfig)
    code_analysis: CodeAnalysisConfig = field(default_factory=CodeAnalysisConfig)


def load_config():
    cfg = AppConfig()
    cfg.host = os.getenv("WRAPPER_HOST", cfg.host)
    cfg.port = int(os.getenv("WRAPPER_PORT", str(cfg.port)))
    cfg.cache.enabled = os.getenv("WRAPPER_CACHE_ENABLED", "true").lower() == "true"
    cfg.service.embedding_service_url = os.getenv("WRAPPER_EMBEDDING_SERVICE_URL", cfg.service.embedding_service_url)
    cfg.surrealdb.url = os.getenv("WRAPPER_SURREALDB_URL", cfg.surrealdb.url)
    cfg.surrealdb.namespace = os.getenv("WRAPPER_SURREALDB_NAMESPACE", cfg.surrealdb.namespace)
    cfg.surrealdb.database = os.getenv("WRAPPER_SURREALDB_DATABASE", cfg.surrealdb.database)
    cfg.surrealdb.username = os.getenv("WRAPPER_SURREALDB_USERNAME", cfg.surrealdb.username)
    cfg.surrealdb.password = os.getenv("WRAPPER_SURREALDB_PASSWORD", cfg.surrealdb.password)
    cfg.surrealdb.runtime_username = os.getenv("WRAPPER_SURREALDB_RUNTIME_USERNAME", cfg.surrealdb.runtime_username)
    cfg.surrealdb.runtime_password = os.getenv("WRAPPER_SURREALDB_RUNTIME_PASSWORD", cfg.surrealdb.runtime_password)
    cfg.surrealdb.use_runtime_credentials = (
        os.getenv("WRAPPER_SURREALDB_USE_RUNTIME_CREDENTIALS", "true").lower() == "true"
    )

    # Meilisearch 配置
    cfg.meilisearch.enabled = os.getenv("WRAPPER_MEILI_ENABLED", "true").lower() == "true"
    cfg.meilisearch.url = os.getenv("WRAPPER_MEILI_URL", cfg.meilisearch.url)
    cfg.meilisearch.api_key = os.getenv("WRAPPER_MEILI_API_KEY", cfg.meilisearch.api_key)
    cfg.meilisearch.index_name = os.getenv("WRAPPER_MEILI_INDEX_NAME", cfg.meilisearch.index_name)
    cfg.meilisearch.timeout = float(os.getenv("WRAPPER_MEILI_TIMEOUT", str(cfg.meilisearch.timeout)))

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

    # TLS/HTTPS 配置
    cfg.tls.enabled = os.getenv("WRAPPER_TLS_ENABLED", "false").lower() == "true"
    cfg.tls.cert_path = os.getenv("WRAPPER_TLS_CERT_PATH", cfg.tls.cert_path)
    cfg.tls.key_path = os.getenv("WRAPPER_TLS_KEY_PATH", cfg.tls.key_path)
    cfg.tls.min_version = os.getenv("WRAPPER_TLS_MIN_VERSION", cfg.tls.min_version)
    cfg.tls.redirect_http = os.getenv("WRAPPER_TLS_REDIRECT_HTTP", "true").lower() == "true"

    cfg.code_analysis.enabled = os.getenv("WRAPPER_CODE_ANALYSIS_ENABLED", "true").lower() == "true"
    cfg.code_analysis.auto_analyze = os.getenv("WRAPPER_AUTO_ANALYZE_CODE", "false").lower() == "true"
    cfg.code_analysis.min_content_length = int(
        os.getenv("WRAPPER_CODE_ANALYSIS_MIN_LENGTH", str(cfg.code_analysis.min_content_length))
    )
    cfg.code_analysis.max_content_length = int(
        os.getenv("WRAPPER_CODE_ANALYSIS_MAX_LENGTH", str(cfg.code_analysis.max_content_length))
    )

    return cfg


config = load_config()
