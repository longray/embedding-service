-- ============================================================
-- scripts/add_deduplication.surql
-- 添加内容去重机制 - 防止重复记忆上传
-- 版本: 2.3.1
-- 兼容: SurrealDB >= 3.0
-- 方案: 内容哈希 + 复合 UNIQUE 索引（租户级去重）
-- ============================================================

-- ==================== 第一步：添加 content_hash 字段 ====================

-- 添加内容哈希字段（用于快速去重）
DEFINE FIELD IF NOT EXISTS content_hash ON memory TYPE string;

-- ==================== 第二步：创建复合 UNIQUE 索引 ====================

-- 租户级内容去重索引（同一租户内不允许相同内容）
-- 注意: 如果已有重复数据，需要先清理才能创建此索引
DEFINE INDEX IF NOT EXISTS memory_tenant_content_unique ON memory 
    FIELDS tenant_id, content_hash UNIQUE;

-- ==================== 第三步：自动哈希触发器 ====================

-- 自动计算内容哈希的触发器
-- 在 CREATE 或 UPDATE 时自动计算 content_hash
DEFINE EVENT IF NOT EXISTS auto_content_hash ON memory
    WHEN $event = "CREATE" OR $event = "UPDATE"
    THEN {
        -- 使用 MD5 计算内容哈希（快速且足够用于去重）
        UPDATE $after.id SET content_hash = crypto::md5($after.content);
    };

-- ==================== 第四步：数据迁移（为现有记录生成哈希）====================

-- 注意: 此步骤会为所有现有记录生成 content_hash
-- 如果数据量大，建议分批执行

-- 方式1: 一次性更新所有记录（适用于数据量 < 10000）
-- UPDATE memory SET content_hash = crypto::md5(content) WHERE content_hash = NONE;

-- 方式2: 分批更新（适用于大数据量，需要在应用层循环执行）
-- UPDATE memory SET content_hash = crypto::md5(content) WHERE content_hash = NONE LIMIT 1000;

-- ==================== 验证 ====================

-- 1. 检查 content_hash 字段是否已添加
-- INFO FOR TABLE memory;

-- 2. 检查索引是否已创建
-- SHOW INDEX ON memory;

-- 3. 测试去重（应该失败）
-- CREATE memory CONTENT {
--     content: 'Test duplicate content',
--     tenant_id: 'test',
--     type: 'test'
-- };
-- CREATE memory CONTENT {
--     content: 'Test duplicate content',
--     tenant_id: 'test',
--     type: 'test'
-- };
-- 第二次插入应该失败: "Database index `memory_tenant_content_unique` already contains..."

-- ==================== 回滚脚本（如需撤销）====================

-- REMOVE EVENT IF EXISTS auto_content_hash ON memory;
-- REMOVE INDEX IF EXISTS memory_tenant_content_unique ON memory;
-- REMOVE FIELD IF EXISTS content_hash ON memory;
