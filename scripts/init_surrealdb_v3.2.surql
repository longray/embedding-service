-- ============================================================
-- scripts/init_surrealdb_v3.2.surql
-- v3.2 Schema - Core Tables: atom, entity, reference
-- 版本: 3.2.0
-- 兼容: SurrealDB >= 2.0
-- 变更: 新增 atom, entity, reference 核心表
-- ============================================================

-- ==================== v3.2: 核心表定义 ====================

-- 1. atom 表 - 原子级数据单元
DEFINE TABLE IF NOT EXISTS atom TYPE NORMAL;

DEFINE FIELD IF NOT EXISTS id ON atom TYPE record<atom>;
DEFINE FIELD IF NOT EXISTS tenant_id ON atom TYPE string
    ASSERT $value != NONE AND string::len($value) > 0;
DEFINE FIELD IF NOT EXISTS type ON atom TYPE string
    ASSERT $value IN ['symbol', 'token', 'fragment'];
DEFINE FIELD IF NOT EXISTS content ON atom TYPE string;
DEFINE FIELD IF NOT EXISTS metadata ON atom TYPE option<object>;
DEFINE FIELD IF NOT EXISTS fingerprint ON atom TYPE option<string>;
DEFINE FIELD IF NOT EXISTS created_at ON atom TYPE datetime
    DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS updated_at ON atom TYPE datetime
    DEFAULT time::now();

-- atom 表索引
DEFINE INDEX IF NOT EXISTS atom_tenant ON atom FIELDS tenant_id;
DEFINE INDEX IF NOT EXISTS atom_type ON atom FIELDS type;
DEFINE INDEX IF NOT EXISTS atom_fingerprint ON atom FIELDS fingerprint;
DEFINE INDEX IF NOT EXISTS atom_created_at ON atom FIELDS created_at;

-- 2. entity 表 - 实体（由 atom 组成）
DEFINE TABLE IF NOT EXISTS entity TYPE NORMAL;

DEFINE FIELD IF NOT EXISTS id ON entity TYPE record<entity>;
DEFINE FIELD IF NOT EXISTS tenant_id ON entity TYPE string
    ASSERT $value != NONE AND string::len($value) > 0;
DEFINE FIELD IF NOT EXISTS type ON entity TYPE string
    ASSERT $value IN ['function', 'class', 'module', 'variable', 'constant'];
DEFINE FIELD IF NOT EXISTS name ON entity TYPE string;
DEFINE FIELD IF NOT EXISTS qualified_name ON entity TYPE option<string>;
DEFINE FIELD IF NOT EXISTS file_path ON entity TYPE option<string>;
DEFINE FIELD IF NOT EXISTS line_start ON entity TYPE option<int>;
DEFINE FIELD IF NOT EXISTS line_end ON entity TYPE option<int>;
DEFINE FIELD IF NOT EXISTS metadata ON entity TYPE option<object>;
DEFINE FIELD IF NOT EXISTS fingerprint ON entity TYPE option<string>;
DEFINE FIELD IF NOT EXISTS created_at ON entity TYPE datetime
    DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS updated_at ON entity TYPE datetime
    DEFAULT time::now();

-- entity 表索引
DEFINE INDEX IF NOT EXISTS entity_tenant ON entity FIELDS tenant_id;
DEFINE INDEX IF NOT EXISTS entity_type ON entity FIELDS type;
DEFINE INDEX IF NOT EXISTS entity_name ON entity FIELDS name;
DEFINE INDEX IF NOT EXISTS entity_file_path ON entity FIELDS file_path;
DEFINE INDEX IF NOT EXISTS entity_fingerprint ON entity FIELDS fingerprint;
DEFINE INDEX IF NOT EXISTS entity_created_at ON entity FIELDS created_at;

-- 3. reference 表 - 引用关系（entity 之间的引用）
DEFINE TABLE IF NOT EXISTS reference TYPE RELATION;

DEFINE FIELD IF NOT EXISTS in ON reference TYPE record<entity>;
DEFINE FIELD IF NOT EXISTS out ON reference TYPE record<entity>;
DEFINE FIELD IF NOT EXISTS tenant_id ON reference TYPE string
    ASSERT $value != NONE AND string::len($value) > 0;
DEFINE FIELD IF NOT EXISTS type ON reference TYPE string
    ASSERT $value IN ['call', 'import', 'extend', 'implement', 'use'];
DEFINE FIELD IF NOT EXISTS weight ON reference TYPE float
    DEFAULT 0.5
    ASSERT $value >= 0.0 AND $value <= 1.0;
DEFINE FIELD IF NOT EXISTS metadata ON reference TYPE option<object>;
DEFINE FIELD IF NOT EXISTS created_at ON reference TYPE datetime
    DEFAULT time::now();

-- reference 表索引
DEFINE INDEX IF NOT EXISTS reference_tenant ON reference FIELDS tenant_id;
DEFINE INDEX IF NOT EXISTS reference_type ON reference FIELDS type;
DEFINE INDEX IF NOT EXISTS reference_weight ON reference FIELDS weight;

-- ==================== v3.2: ChangeFeed 配置 ====================

-- 启用 ChangeFeed 用于实时变更通知
DEFINE CHANGE FEED IF NOT EXISTS atom_feed ON TABLE atom DURATION 7d;
DEFINE CHANGE FEED IF NOT EXISTS entity_feed ON TABLE entity DURATION 7d;
DEFINE CHANGE FEED IF NOT EXISTS reference_feed ON TABLE reference DURATION 7d;

-- ==================== v3.2: 辅助表 ====================

-- 4. performance_log 表 - 性能监控
DEFINE TABLE IF NOT EXISTS performance_log TYPE NORMAL;

DEFINE FIELD IF NOT EXISTS id ON performance_log TYPE record<performance_log>;
DEFINE FIELD IF NOT EXISTS tenant_id ON performance_log TYPE string
    ASSERT $value != NONE AND string::len($value) > 0;
DEFINE FIELD IF NOT EXISTS operation ON performance_log TYPE string;
DEFINE FIELD IF NOT EXISTS duration_ms ON performance_log TYPE float;
DEFINE FIELD IF NOT EXISTS memory_mb ON performance_log TYPE float;
DEFINE FIELD IF NOT EXISTS metadata ON performance_log TYPE option<object>;
DEFINE FIELD IF NOT EXISTS created_at ON performance_log TYPE datetime
    DEFAULT time::now();

-- performance_log 表索引
DEFINE INDEX IF NOT EXISTS perf_tenant ON performance_log FIELDS tenant_id;
DEFINE INDEX IF NOT EXISTS perf_operation ON performance_log FIELDS operation;
DEFINE INDEX IF NOT EXISTS perf_created_at ON performance_log FIELDS created_at;

-- ==================== v3.2: Schema 版本记录 ====================

-- 记录 schema 版本
DEFINE TABLE IF NOT EXISTS schema_version TYPE NORMAL;

DEFINE FIELD IF NOT EXISTS id ON schema_version TYPE record<schema_version>;
DEFINE FIELD IF NOT EXISTS version ON schema_version TYPE string;
DEFINE FIELD IF NOT EXISTS applied_at ON schema_version TYPE datetime
    DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS description ON schema_version TYPE option<string>;

-- 插入当前版本记录
CREATE schema_version CONTENT {
    version: '3.2.0',
    description: 'v3.2 Schema - Core Tables: atom, entity, reference',
    applied_at: time::now()
};

-- ==================== 验证查询 ====================

-- 查询所有表
-- INFO FOR DB;

-- 查询 atom 表结构
-- INFO FOR TABLE atom;

-- 查询 entity 表结构
-- INFO FOR TABLE entity;

-- 查询 reference 表结构
-- INFO FOR TABLE reference;
