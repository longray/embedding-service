-- ============================================================
-- scripts/init_surrealdb_v2.4.0.surql
-- v2.4.0 L0/L1/L2 Layered Storage Schema Update
-- 版本: 2.4.0
-- 兼容: SurrealDB >= 2.0
-- 变更: 添加 content_abstract, content_overview, local_id 字段
-- ============================================================

-- ==================== v2.4.0: L0/L1/L2 分层字段 ====================

-- 1. 添加 L0 摘要字段（Abstract）
DEFINE FIELD IF NOT EXISTS content_abstract ON memory TYPE option<string>
    ASSERT string::len($value) <= 200;

-- 2. 添加 L1 概览字段（Overview）
DEFINE FIELD IF NOT EXISTS content_overview ON memory TYPE option<string>
    ASSERT string::len($value) <= 1000;

-- 3. 添加 local_id 字段（插件端 ULID，用于双向关联）
DEFINE FIELD IF NOT EXISTS local_id ON memory TYPE option<string>
    ASSERT $value == NONE OR string::len($value) == 18;

-- 4. 为 local_id 创建唯一索引（可选，防止重复上传）
DEFINE INDEX IF NOT EXISTS memory_local_id ON memory
    FIELDS local_id;

-- 5. 为 content_abstract 创建全文索引（L0 快速查询）
-- 注意：需要先用 memory_analyzer 确保已存在
DEFINE INDEX IF NOT EXISTS memory_abstract_ft ON memory
    FIELDS content_abstract FULLTEXT ANALYZER memory_analyzer BM25;

-- ==================== 验证字段 ====================

-- 查询当前表的字段定义（验证用）
-- INFO FOR TABLE memory;

-- ==================== 测试数据 ====================

-- 插入测试数据（可选）
-- CREATE memory CONTENT {
--     content: 'User prefers TypeScript for all new projects because of type safety.',
--     content_abstract: 'User prefers TypeScript',
--     content_overview: '- TypeScript preference established\n- Type safety benefit',
--     local_id: '01HV8J3K2M4N5P6Q7R8',
--     type: 'preference',
--     tenant_id: 'test'
-- };

-- 验证查询:
-- SELECT id, content, content_abstract, content_overview, local_id FROM memory LIMIT 1;
