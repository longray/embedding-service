-- SurrealDB 数据库初始化脚本
-- 用于记忆管理API的schema定义

-- 使用数据库和命名空间
USE NS memory_ns;
USE DB memory_db;

-- ============================================================================
-- 表定义
-- ============================================================================

-- Memory表：存储记忆内容和向量
DEFINE TABLE memory SCHEMAFULL;
DEFINE FIELD content ON memory TYPE string;
DEFINE FIELD embedding ON memory TYPE array<float>;
DEFINE FIELD metadata ON memory TYPE option<object>;
DEFINE FIELD entities ON memory TYPE option<array<record<entity>>>;
DEFINE FIELD created_at ON memory TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON memory TYPE datetime DEFAULT time::now();

-- Entity表：存储实体
DEFINE TABLE entity SCHEMAFULL;
DEFINE FIELD name ON entity TYPE string;
DEFINE FIELD type ON entity TYPE string;
DEFINE FIELD properties ON entity TYPE option<object>;
DEFINE FIELD created_at ON entity TYPE datetime DEFAULT time::now();

-- Relation表：存储关系（支持自动去重）
DEFINE TABLE relation SCHEMAFULL;
DEFINE FIELD in ON relation TYPE record<entity>;
DEFINE FIELD out ON relation TYPE record<entity>;
DEFINE FIELD type ON relation TYPE string;
DEFINE FIELD properties ON relation TYPE option<object>;
DEFINE FIELD key ON relation TYPE string VALUE array::sort([in, out]);
DEFINE FIELD created_at ON relation TYPE datetime DEFAULT time::now();

-- ============================================================================
-- 索引定义
-- ============================================================================

-- Memory表：HNSW向量索引（优化版：F32类型，减少50%内存）
DEFINE INDEX memory_embedding_hnsw ON memory FIELDS embedding 
    HNSW DIMENSION 1024 TYPE F32 DIST COSINE EFC 200 M 16 CONCURRENTLY;

-- Memory表：时间索引（用于排序和过滤）
DEFINE INDEX memory_created_at_idx ON memory FIELDS created_at;

-- Entity表：名称+类型唯一索引（防止重复实体）
DEFINE INDEX entity_name_type_idx ON entity FIELDS name, type UNIQUE;

-- Relation表：key+type唯一索引（自动去重对称关系）
DEFINE INDEX relation_unique_key_idx ON relation FIELDS key, type UNIQUE;

-- Relation表：in/out索引（用于图遍历）
DEFINE INDEX relation_in_idx ON relation FIELDS in;
DEFINE INDEX relation_out_idx ON relation FIELDS out;
