-- init_hnsw_index.surql
-- HNSW 向量索引初始化脚本
-- 用于优化记忆搜索性能

-- 删除旧索引（如果存在）
REMOVE INDEX IF EXISTS memory_embedding_hnsw ON memory;

-- 创建 HNSW 索引
-- HNSW (Hierarchical Navigable Small World) 是一种近似最近邻搜索算法
-- 相比暴力搜索 O(n)，HNSW 的时间复杂度为 O(log n)
DEFINE INDEX memory_embedding_hnsw ON memory 
  FIELDS embedding 
  TYPE HNSW 
  DIMENSION 1024 
  DISTANCE COSINE;

-- 验证索引创建
INFO FOR INDEX memory_embedding_hnsw ON memory;
