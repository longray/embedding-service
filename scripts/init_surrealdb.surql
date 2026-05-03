-- ============================================================
-- scripts/init_surrealdb.surql
-- Embedding Service 数据库初始化脚本
-- 版本: 2.2.0
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
DEFINE FIELD IF NOT EXISTS local_id    ON memory TYPE option<string>;
    -- 客户端本地 ID（用于多设备同步和冲突检测）

DEFINE FIELD IF NOT EXISTS source_id   ON memory TYPE option<string>;
    -- 客户端原始 ID（如 "mem_1709123456789_abc12345"），UNIQUE 索引防止重复上传

DEFINE FIELD IF NOT EXISTS source      ON memory TYPE option<string>
    DEFAULT 'api';
    -- 来源标识: api | plugin | import | migration

DEFINE FIELD IF NOT EXISTS source_timestamp ON memory TYPE option<datetime>;
    -- 客户端原始时间戳（ISO 8601），与 created_at（服务端时间）区分

DEFINE FIELD IF NOT EXISTS content_hash ON memory TYPE option<string>;
    -- 内容哈希（32位十六进制），用于去重和指纹比对

-- 质量信号（可选）
DEFINE FIELD IF NOT EXISTS classification_confidence ON memory TYPE option<float>;
    -- 自动分类的置信度 [0.0, 1.0]

-- L0/L1/L2 分层内容字段 (v2.4.0)
DEFINE FIELD IF NOT EXISTS abstract ON memory TYPE option<string>
    ASSERT $value == NONE OR string::len($value) <= 200;
    -- L0 摘要，≤200字符

DEFINE FIELD IF NOT EXISTS overview ON memory TYPE option<string>
    ASSERT $value == NONE OR string::len($value) <= 1000;
    -- L1 概览，≤1000字符

-- 服务端时间戳（不可变）
DEFINE FIELD IF NOT EXISTS created_at  ON memory TYPE datetime
    DEFAULT time::now();

-- 灵活扩展字段（兜底，向后兼容旧 API 的 metadata 传参）
-- 注意: SurrealDB 3.0 语法为 TYPE object FLEXIBLE（FLEXIBLE 在 TYPE 后面）
DEFINE FIELD IF NOT EXISTS metadata    ON memory TYPE object FLEXIBLE
    DEFAULT {};

-- 文件路径字段（从 metadata 提升，用于 lookup 查询）
DEFINE FIELD IF NOT EXISTS file_path ON memory TYPE option<string>;
    -- 文件相对路径，用于 file_path + project_id 组合查询

-- ==================== memory 索引 ====================

-- 向量搜索索引（HNSW）
-- M=16: 1K-100K数据量下平衡精度与内存（推荐16-24）
-- EFC=200: Oracle建议200-300，索引构建质量更高
-- DIST=COSINE: 语义搜索推荐
DEFINE INDEX IF NOT EXISTS memory_embedding_hnsw ON memory
    FIELDS embedding HNSW DIMENSION 1024 DIST COSINE
    EFC 200 M 16;

-- 全文搜索索引（BM25）— 降级路径
-- ⚠️ 主搜索引擎已迁移至 Meilisearch（Polyglot 架构 v2.3.0）
-- 此 BM25 索引仅在 Meilisearch 不可用时作为降级路径使用
-- 已知限制: 日期格式（如 2026-03-11）会被 blank tokenizer 切分
-- 如需精确日期搜索，请确保 Meilisearch 已启用
--
-- Tokenizer: blank（按空格切分）+ class（按 Unicode 类别切分）
-- Filter: ngram(2,8) 对中英文混合内容分词
-- ⚠️ 使用 REMOVE + DEFINE（非 IF NOT EXISTS）确保 analyzer 变更生效
REMOVE ANALYZER IF EXISTS memory_analyzer;
DEFINE ANALYZER memory_analyzer
    TOKENIZERS blank, class
    FILTERS lowercase, ngram(2,8);

-- 重建全文索引以使用更新后的 analyzer（降级路径）
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
DEFINE INDEX IF NOT EXISTS memory_tenant_type_project ON memory FIELDS tenant_id, type, project_id;

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
    description: 'Phase 2: ngram(2,8) 中文分词 + reference 图关系边表'
};

-- ==================== 运行时用户（安全加固 Phase 3C）====================
-- 运行时应用用户 - 仅数据操作权限（EDITOR 角色）
-- 不能执行 DEFINE TABLE/INDEX 等 schema 操作
DEFINE USER IF NOT EXISTS runtime_user ON DATABASE
    PASSWORD 'change_me_in_production'
    ROLES EDITOR
    DURATION FOR SESSION 12h, FOR TOKEN 1h
    COMMENT 'Runtime application user with data-only permissions';

-- ==================== conflict 表 ====================

DEFINE TABLE IF NOT EXISTS conflict TYPE NORMAL SCHEMAFULL;

DEFINE FIELD IF NOT EXISTS source_id ON conflict TYPE string;
    -- 原始记忆 source_id，用于关联冲突的记忆

DEFINE FIELD IF NOT EXISTS local_hash ON conflict TYPE string;
    -- 本地内容哈希值

DEFINE FIELD IF NOT EXISTS server_hash ON conflict TYPE string;
    -- 服务端内容哈希值

DEFINE FIELD IF NOT EXISTS local_content ON conflict TYPE option<string>;
    -- 本地内容（可选存储）

DEFINE FIELD IF NOT EXISTS server_content ON conflict TYPE option<string>;
    -- 服务端内容（可选存储）

DEFINE FIELD IF NOT EXISTS local_mtime ON conflict TYPE int DEFAULT 0;
    -- 本地修改时间戳

DEFINE FIELD IF NOT EXISTS server_mtime ON conflict TYPE int DEFAULT 0;
    -- 服务端修改时间戳

DEFINE FIELD IF NOT EXISTS tenant_id ON conflict TYPE string DEFAULT 'default';
    -- 多租户隔离字段

DEFINE FIELD IF NOT EXISTS status ON conflict TYPE string DEFAULT 'pending';
    -- 状态: pending | resolved

DEFINE FIELD IF NOT EXISTS resolution ON conflict TYPE option<string>;
    -- 解决方案: use_local | use_remote | keep_both

DEFINE FIELD IF NOT EXISTS resolved_at ON conflict TYPE option<datetime>;
    -- 解决时间戳

DEFINE FIELD IF NOT EXISTS created_at ON conflict TYPE datetime DEFAULT time::now();
    -- 创建时间戳

DEFINE FIELD IF NOT EXISTS updated_at ON conflict TYPE datetime DEFAULT time::now();
    -- 更新时间戳

-- 冲突表索引
DEFINE INDEX IF NOT EXISTS conflict_source_tenant ON conflict FIELDS source_id, tenant_id;
    -- 按 source_id 和 tenant 查询冲突

DEFINE INDEX IF NOT EXISTS conflict_status_tenant ON conflict FIELDS status, tenant_id;
    -- 按状态和租户过滤冲突

-- 更新版本号
UPSERT schema_version:current CONTENT {
    version: '2.4.1',
    applied_at: time::now(),
    description: 'Phase 8: 代码指纹同步 API + PrecomputeService 代码分析 API - file_fingerprint 表'
};

-- ==================== audit_log 表 (P3-5) ====================

DEFINE TABLE IF NOT EXISTS audit_log TYPE NORMAL SCHEMAFULL;

-- 核心字段
DEFINE FIELD IF NOT EXISTS timestamp ON audit_log TYPE datetime DEFAULT time::now();
    -- 审计时间戳

DEFINE FIELD IF NOT EXISTS user_id ON audit_log TYPE option<string>;
    -- 用户标识（可选，未认证请求为空）

DEFINE FIELD IF NOT EXISTS action ON audit_log TYPE string;
    -- 操作类型: memory_create | memory_read | memory_update | memory_delete |
    --           relation_create | relation_delete | sync_preview | sync_full |
    --           login | logout | admin_action | system_cleanup

DEFINE FIELD IF NOT EXISTS resource_type ON audit_log TYPE option<string>;
    -- 资源类型: memory | relation | project | user | system

DEFINE FIELD IF NOT EXISTS resource_id ON audit_log TYPE option<string>;
    -- 资源ID（如 memory:xxx）

DEFINE FIELD OVERWRITE details ON audit_log TYPE option<string>;
    -- 详细信息（JSON字符串，应用层负责序列化/反序列化）

DEFINE FIELD IF NOT EXISTS ip_address ON audit_log TYPE option<string>;
    -- 客户端IP地址

DEFINE FIELD IF NOT EXISTS user_agent ON audit_log TYPE option<string>;
    -- 客户端User-Agent

DEFINE FIELD IF NOT EXISTS tenant_id ON audit_log TYPE string DEFAULT 'default';
    -- 租户ID

DEFINE FIELD IF NOT EXISTS created_at ON audit_log TYPE datetime DEFAULT time::now();
    -- 创建时间

-- 审计日志索引
DEFINE INDEX IF NOT EXISTS audit_timestamp ON audit_log FIELDS timestamp;
    -- 按时间查询

DEFINE INDEX IF NOT EXISTS audit_user_action ON audit_log FIELDS user_id, action;
    -- 按用户和操作查询

DEFINE INDEX IF NOT EXISTS audit_resource ON audit_log FIELDS resource_type, resource_id;
    -- 按资源查询

DEFINE INDEX IF NOT EXISTS audit_tenant_timestamp ON audit_log FIELDS tenant_id, timestamp;
    -- 按租户和时间范围查询（最常用的查询模式）

-- ==================== Memory Lookup 索引 (BL-CA-34) ====================

-- source_id 索引（用于精确查询）
DEFINE INDEX IF NOT EXISTS idx_memory_source_id ON memory FIELDS source_id;
    -- 按 source_id 查询（缓存重建、多设备同步）

-- content_hash 索引（用于内容去重）
DEFINE INDEX IF NOT EXISTS idx_memory_content_hash ON memory FIELDS content_hash;
    -- 按 content_hash 查询（内容去重）

-- file_path + project_id 复合索引（用于路径查询）
DEFINE INDEX IF NOT EXISTS idx_memory_file_path_project ON memory FIELDS file_path, project_id;

-- ==================== Code Fingerprint 表 (BL-B-80) ====================

DEFINE TABLE IF NOT EXISTS file_fingerprint TYPE NORMAL SCHEMAFULL;
    -- 代码文件指纹表，用于增量同步

-- 核心字段
DEFINE FIELD IF NOT EXISTS file_path ON file_fingerprint TYPE string;
    -- 文件路径（唯一标识）

DEFINE FIELD IF NOT EXISTS content_hash ON file_fingerprint TYPE string;
    -- 文件内容哈希（SHA256）

DEFINE FIELD IF NOT EXISTS symbols_hash ON file_fingerprint TYPE string;
    -- 符号哈希（函数名+参数哈希）

DEFINE FIELD IF NOT EXISTS tenant_id ON file_fingerprint TYPE string DEFAULT 'default';
    -- 租户ID

DEFINE FIELD IF NOT EXISTS project_id ON file_fingerprint TYPE string DEFAULT 'global';
    -- 项目ID

DEFINE FIELD IF NOT EXISTS updated_at ON file_fingerprint TYPE datetime DEFAULT time::now();
    -- 更新时间

-- 复合唯一索引（文件路径 + 租户 + 项目）
DEFINE INDEX IF NOT EXISTS idx_fingerprint_unique ON file_fingerprint FIELDS file_path, tenant_id, project_id UNIQUE;
    -- 唯一约束：同一项目内文件路径唯一

-- 查询索引
DEFINE INDEX IF NOT EXISTS idx_fingerprint_tenant_project ON file_fingerprint FIELDS tenant_id, project_id;
    -- 按租户和项目查询（最常用的查询模式）

DEFINE INDEX IF NOT EXISTS idx_fingerprint_content_hash ON file_fingerprint FIELDS content_hash;
    -- 按内容哈希查询（用于去重）
    -- 按 file_path + project_id 组合查询（lookup API）

-- ==================== v3.2: Atom/Entity/Reference 表 ====================
-- 新增: 支持代码分析的四层架构

-- 1. atom 表 - 原子级数据单元
DEFINE TABLE IF NOT EXISTS atom TYPE NORMAL;

DEFINE FIELD IF NOT EXISTS id ON atom TYPE record<atom>;
DEFINE FIELD IF NOT EXISTS tenant_id ON atom TYPE string
    ASSERT $value != NONE AND string::len($value) > 0;
DEFINE FIELD OVERWRITE type ON atom TYPE string
    ASSERT $value IN ['function', 'class', 'interface', 'import', 'goal', 'scope', 'task', 'note', 'symbol', 'token', 'fragment', 'chapter', 'section'];
DEFINE FIELD IF NOT EXISTS content ON atom TYPE string;
DEFINE FIELD IF NOT EXISTS name ON atom TYPE option<string>;
DEFINE FIELD IF NOT EXISTS signature ON atom TYPE option<string>;
DEFINE FIELD IF NOT EXISTS params ON atom TYPE option<array<object>>;
DEFINE FIELD IF NOT EXISTS return_type ON atom TYPE option<string>;
DEFINE FIELD IF NOT EXISTS is_exported ON atom TYPE option<bool>;
DEFINE FIELD IF NOT EXISTS is_async ON atom TYPE option<bool>;
DEFINE FIELD IF NOT EXISTS complexity ON atom TYPE option<int>;
DEFINE FIELD IF NOT EXISTS max_nesting_depth ON atom TYPE option<int>;
DEFINE FIELD IF NOT EXISTS docstring ON atom TYPE option<object>;
DEFINE FIELD IF NOT EXISTS start_line ON atom TYPE option<int>;
DEFINE FIELD IF NOT EXISTS end_line ON atom TYPE option<int>;
DEFINE FIELD IF NOT EXISTS status ON atom TYPE option<string>;
DEFINE FIELD IF NOT EXISTS metadata ON atom TYPE option<object>;
DEFINE FIELD IF NOT EXISTS project ON atom TYPE option<string>;
DEFINE FIELD IF NOT EXISTS fingerprint ON atom TYPE option<string>;
DEFINE FIELD IF NOT EXISTS version ON atom TYPE option<int>;
DEFINE FIELD IF NOT EXISTS created_at ON atom TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS updated_at ON atom TYPE datetime DEFAULT time::now();

-- v3.3 层级化知识图谱字段
DEFINE FIELD IF NOT EXISTS tags ON atom TYPE option<array<string>>;
DEFINE FIELD IF NOT EXISTS heading_level ON atom TYPE option<int>
    ASSERT $value == NONE OR ($value >= 1 AND $value <= 6);
DEFINE FIELD IF NOT EXISTS parent_id ON atom TYPE option<string>;
DEFINE FIELD IF NOT EXISTS order ON atom TYPE option<string>;
DEFINE FIELD IF NOT EXISTS aliases ON atom TYPE option<array<string>>;
DEFINE FIELD IF NOT EXISTS entity_id ON atom TYPE option<string>;
DEFINE FIELD IF NOT EXISTS embedding ON atom TYPE option<array<float>>;

-- atom 表索引
DEFINE INDEX IF NOT EXISTS idx_atom_embedding_hnsw ON atom
FIELDS embedding HNSW DIMENSION 1024 DIST COSINE;
DEFINE INDEX IF NOT EXISTS idx_atom_tenant ON atom FIELDS tenant_id;
DEFINE INDEX IF NOT EXISTS idx_atom_type ON atom FIELDS type;
DEFINE INDEX IF NOT EXISTS idx_atom_project ON atom FIELDS project;
DEFINE INDEX IF NOT EXISTS idx_atom_fingerprint ON atom FIELDS fingerprint;
DEFINE INDEX IF NOT EXISTS idx_atom_created_at ON atom FIELDS created_at;

-- v3.3 层级索引
DEFINE INDEX IF NOT EXISTS idx_atom_parent ON atom FIELDS parent_id;
DEFINE INDEX IF NOT EXISTS idx_atom_order ON atom FIELDS order;
DEFINE INDEX IF NOT EXISTS idx_atom_entity ON atom FIELDS entity_id;

-- 2. entity 表 - 实体（由 atom 组成）
DEFINE TABLE IF NOT EXISTS entity TYPE NORMAL;

DEFINE FIELD IF NOT EXISTS id ON entity TYPE record<entity>;
DEFINE FIELD IF NOT EXISTS tenant_id ON entity TYPE string
    ASSERT $value != NONE AND string::len($value) > 0;
DEFINE FIELD IF NOT EXISTS type ON entity TYPE string
    ASSERT $value IN ['memory', 'backlog', 'wiki', 'code'];
DEFINE FIELD IF NOT EXISTS abstract ON entity TYPE string;
DEFINE FIELD IF NOT EXISTS overview ON entity TYPE option<object>;
DEFINE FIELD IF NOT EXISTS atoms ON entity TYPE option<array<record<atom>>>;
DEFINE FIELD IF NOT EXISTS tags ON entity TYPE option<array<string>>;
DEFINE FIELD IF NOT EXISTS project ON entity TYPE option<string>;
DEFINE FIELD IF NOT EXISTS created_by ON entity TYPE option<string>;
DEFINE FIELD IF NOT EXISTS title ON entity TYPE option<string>;
DEFINE FIELD IF NOT EXISTS aliases ON entity TYPE option<array<string>>;
DEFINE FIELD IF NOT EXISTS priority ON entity TYPE option<string>;
DEFINE FIELD IF NOT EXISTS status ON entity TYPE option<string>;
DEFINE FIELD IF NOT EXISTS scene ON entity TYPE option<string>;
DEFINE FIELD IF NOT EXISTS estimated_hours ON entity TYPE option<float>;
DEFINE FIELD IF NOT EXISTS actual_hours ON entity TYPE option<float>;
DEFINE FIELD IF NOT EXISTS file_path ON entity TYPE option<string>;
DEFINE FIELD IF NOT EXISTS language ON entity TYPE option<string>;
DEFINE FIELD IF NOT EXISTS quality_score ON entity TYPE option<object>;
DEFINE FIELD IF NOT EXISTS complexity_metrics ON entity TYPE option<object>;
DEFINE FIELD IF NOT EXISTS created_at ON entity TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS updated_at ON entity TYPE datetime DEFAULT time::now();

-- entity 表索引
DEFINE INDEX IF NOT EXISTS idx_entity_tenant ON entity FIELDS tenant_id;
DEFINE INDEX IF NOT EXISTS idx_entity_type ON entity FIELDS type;
DEFINE INDEX IF NOT EXISTS idx_entity_project ON entity FIELDS project;
DEFINE INDEX IF NOT EXISTS idx_entity_created_at ON entity FIELDS created_at;

-- 3. reference 表 - 引用关系（图关系）
-- 使用 OVERWRITE 强制更新 Schema
DEFINE TABLE OVERWRITE reference TYPE RELATION;

DEFINE FIELD in ON reference TYPE record<atom|entity|memory>;
DEFINE FIELD out ON reference TYPE record<atom|entity|memory>;
DEFINE FIELD IF NOT EXISTS tenant_id ON reference TYPE string
    ASSERT $value != NONE AND string::len($value) > 0;
DEFINE FIELD OVERWRITE type ON reference TYPE string
    ASSERT $value IN ['calls', 'imports', 'extends', 'implements', 'depends_on', 'wiki_link', 'part_of', 'related', 'follow_up', 'elaboration', 'contradiction', 'reference', 'derived_from', 'similar_to', 'summarizes', 'contains'];
DEFINE FIELD IF NOT EXISTS weight ON reference TYPE float DEFAULT 0.5 ASSERT $value >= 0.0 AND $value <= 1.0;
DEFINE FIELD IF NOT EXISTS file_path ON reference TYPE option<string>;
DEFINE FIELD IF NOT EXISTS line ON reference TYPE option<int>;
DEFINE FIELD IF NOT EXISTS column ON reference TYPE option<int>;
DEFINE FIELD IF NOT EXISTS metadata ON reference TYPE option<object>;
DEFINE FIELD IF NOT EXISTS created_at ON reference TYPE datetime DEFAULT time::now();
