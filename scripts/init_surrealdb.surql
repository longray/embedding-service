-- ============================================================
-- scripts/init_surrealdb.surql
-- Embedding Service 数据库初始化脚本
-- 版本: 2.1.0
-- 兼容: SurrealDB >= 2.0（推荐 3.0+）
-- 注意: SurrealDB 3.0 要求 TYPE NORMAL SCHEMAFULL（非旧的 SCHEMAFULL）
-- 幂等: 所有 DEFINE 使用 IF NOT EXISTS，支持重复执行
-- ============================================================

-- ==================== 命名空间初始化 ====================
-- 注意: 通过 Python SDK 执行时，namespace/database 已通过 db.use() 设置
-- 通过 HTTP API 直接执行时，需要先设置 surreal-ns/surreal-db header
-- 此处不写 DEFINE NAMESPACE/DATABASE，避免权限问题

-- ==================== memory 表 ====================
DEFINE TABLE IF NOT EXISTS memory TYPE NORMAL SCHEMAFULL;

-- 核心字段（必需）
DEFINE FIELD IF NOT EXISTS content     ON memory TYPE string;
    -- 记忆原文，全文索引目标

DEFINE FIELD IF NOT EXISTS embedding   ON memory TYPE array<float>;
    -- 1024 维，匹配 Qwen3-Embedding-0.6B

-- 多租户字段（必需，带默认值以保持向后兼容）
DEFINE FIELD IF NOT EXISTS tenant_id   ON memory TYPE string
    DEFAULT 'default';
    -- 行级多租户隔离。默认值 "default" 保证旧数据和未传 tenant_id 的请求兼容

-- 分类与组织字段（从 metadata 提升为顶层，可索引）
DEFINE FIELD IF NOT EXISTS type        ON memory TYPE option<string>
    DEFAULT 'general';
    -- 类型: general | preference | decision | note | analysis | test | long-term

DEFINE FIELD IF NOT EXISTS tags        ON memory TYPE option<array<string>>
    DEFAULT [];
    -- 自由标签，支��多标签过滤

DEFINE FIELD IF NOT EXISTS project_id  ON memory TYPE option<string>
    DEFAULT 'global';
    -- 项目标识。"global" = 跨项目通用记忆

-- 来源追踪字段（用于去重和审计）
DEFINE FIELD IF NOT EXISTS source_id   ON memory TYPE option<string>;
    -- 客户端原始 ID（如 "mem_1709123456789_abc12345"），UNIQUE 索引防止重复上传

DEFINE FIELD IF NOT EXISTS source      ON memory TYPE option<string>
    DEFAULT 'api';
    -- 来源标识: api | plugin | import | migration

DEFINE FIELD IF NOT EXISTS source_timestamp ON memory TYPE option<datetime>;
    -- 客户端原始时间戳（ISO 8601），与 created_at（服务端时间）区分

-- 质量信号（可选）
DEFINE FIELD IF NOT EXISTS classification_confidence ON memory TYPE option<float>;
    -- 自动分类的置信度 [0.0, 1.0]

-- 服务端时间戳（不可变）
DEFINE FIELD IF NOT EXISTS created_at  ON memory TYPE datetime
    DEFAULT time::now();

-- 灵活扩展字段（兜底，向后兼容旧 API 的 metadata 传参）
-- 注意: SurrealDB 3.0 语法为 TYPE object FLEXIBLE（FLEXIBLE 在 TYPE 后面）
DEFINE FIELD IF NOT EXISTS metadata    ON memory TYPE object FLEXIBLE
    DEFAULT {};

-- ==================== memory 索引 ====================

-- 向量搜索索引（HNSW）
-- M=12: SurrealDB 默认值，1K-100K 数据量下平衡精度与内存
-- EFC=200: Oracle 建议 200-300，索引构建质量更高
-- DIST=COSINE: 语义搜索推荐（契约验证确认支持）
DEFINE INDEX IF NOT EXISTS memory_embedding_hnsw ON memory
    FIELDS embedding HNSW DIMENSION 1024 DIST COSINE
    EFC 200 M 12;

-- 全文搜索索引（BM25）
-- 注意: ngram(2,8) 对中英文混合内容都能正确分词
-- - blank: 按空格切分
-- - class: 按 Unicode 类别（Latin/CJK/数字）切分
-- - ngram(2,8): 从每个 token 生成 2-8 字符的 n-gram 子串
-- 效果: 中文词组（推理/服务/知识图谱）和英文词（Python/FastAPI）均可搜索
-- ⚠️ 使用 REMOVE + DEFINE（非 IF NOT EXISTS）确保 analyzer 变更生效
REMOVE ANALYZER IF EXISTS memory_analyzer;
DEFINE ANALYZER memory_analyzer
    TOKENIZERS blank, class
    FILTERS lowercase, ngram(2,8);

-- 重建全文索引以使用更新后的 analyzer
REMOVE INDEX IF EXISTS memory_content_ft ON memory;
DEFINE INDEX memory_content_ft ON memory
    FIELDS content FULLTEXT ANALYZER memory_analyzer BM25;

-- 时间戳索引（用于时间范围查询）
DEFINE INDEX IF NOT EXISTS memory_created_at ON memory
    FIELDS created_at;

-- 多租户索引
DEFINE INDEX IF NOT EXISTS memory_tenant ON memory FIELDS tenant_id;

-- 分类索引
DEFINE INDEX IF NOT EXISTS memory_type ON memory FIELDS type;

-- 项目索引
DEFINE INDEX IF NOT EXISTS memory_project ON memory FIELDS project_id;

-- 去重索引（防止重复上传，NONE 值允许多个共存）
DEFINE INDEX IF NOT EXISTS memory_source_id ON memory FIELDS source_id UNIQUE;

-- 复合索引（高频查询路径）
DEFINE INDEX IF NOT EXISTS memory_tenant_type ON memory FIELDS tenant_id, type;
DEFINE INDEX IF NOT EXISTS memory_tenant_project ON memory FIELDS tenant_id, project_id;

-- ==================== memory_relation 边表 (图关系) ====================

-- 记忆间的关联关系表（SurrealDB RELATE 图边表）
-- 使用 TYPE RELATION 定义边，IN/OUT 限制端点为 memory 表
DEFINE TABLE IF NOT EXISTS memory_relation TYPE RELATION SCHEMAFULL
    IN memory OUT memory;

-- 关系类型: related | follow_up | elaboration | contradiction | reference | derived_from
DEFINE FIELD IF NOT EXISTS relationship_type ON memory_relation TYPE string
    DEFAULT 'related';

-- 关系权重: 0.0-1.0，表示关联强度
DEFINE FIELD IF NOT EXISTS weight ON memory_relation TYPE float
    DEFAULT 0.5;

-- 多租户字段（与 memory 表保持一致）
DEFINE FIELD IF NOT EXISTS tenant_id ON memory_relation TYPE string
    DEFAULT 'default';

-- 关系描述（可选，如 "基于此决策的后续分析"）
DEFINE FIELD IF NOT EXISTS description ON memory_relation TYPE option<string>;

-- 扩展元数据
DEFINE FIELD IF NOT EXISTS metadata ON memory_relation TYPE object FLEXIBLE
    DEFAULT {};

-- 创建时间
DEFINE FIELD IF NOT EXISTS created_at ON memory_relation TYPE datetime
    DEFAULT time::now();

-- 边表索引
DEFINE INDEX IF NOT EXISTS relation_type ON memory_relation FIELDS relationship_type;
DEFINE INDEX IF NOT EXISTS relation_tenant ON memory_relation FIELDS tenant_id;
DEFINE INDEX IF NOT EXISTS relation_tenant_type ON memory_relation FIELDS tenant_id, relationship_type;

-- ==================== project 表 ====================

DEFINE TABLE IF NOT EXISTS project TYPE NORMAL SCHEMAFULL;

DEFINE FIELD IF NOT EXISTS name        ON project TYPE string;
    -- 项目显示名，如 "embedding_service"

DEFINE FIELD IF NOT EXISTS tenant_id   ON project TYPE string
    DEFAULT 'default';
    -- 项目所属租户

DEFINE FIELD IF NOT EXISTS project_tag ON project TYPE option<string>;
    -- 项目标签（如 "global"/"unclassified"/自定义）

DEFINE FIELD IF NOT EXISTS first_seen  ON project TYPE datetime
    DEFAULT time::now();

DEFINE FIELD IF NOT EXISTS last_seen   ON project TYPE datetime
    DEFAULT time::now();

DEFINE FIELD IF NOT EXISTS entry_count ON project TYPE int
    DEFAULT 0;

DEFINE INDEX IF NOT EXISTS project_tenant ON project FIELDS tenant_id;
DEFINE INDEX IF NOT EXISTS project_name   ON project FIELDS tenant_id, name UNIQUE;
    -- 同一租户下项目名唯一

-- ==================== schema_version 表 ====================

DEFINE TABLE IF NOT EXISTS schema_version TYPE NORMAL SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS version     ON schema_version TYPE string;
DEFINE FIELD IF NOT EXISTS applied_at  ON schema_version TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS description ON schema_version TYPE string;

-- 记录当前版本（UPSERT 保证幂等）
-- 注意: CONTENT 会替换所有字段，必须显式设置 applied_at
UPSERT schema_version:current CONTENT {
    version: '2.1.0',
    applied_at: time::now(),
    description: 'Phase 2: ngram(2,8) 中文分词 + memory_relation 图关系边表'
};
