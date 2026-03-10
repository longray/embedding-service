# SurrealDB 3.0 升级设计文档

> **版本**: v1.3 (数据模型融合 + 多租户版)
> **日期**: 2026-03-11
> **作者**: OpenCode Agent
> **状态**: 待用户最终审批

---

## 目录

0. [评审历史](#0-评审历史)
1. [执行摘要](#1-执行摘要)
2. [现状审计](#2-现状审计)
3. [SurrealDB 3.0 能力评估](#3-surrealdb-30-能力评估)
4. [升级方案设计](#4-升级方案设计)
5. [数据模型设计](#5-数据模型设计)
6. [搜索引擎升级](#6-搜索引擎升级)
7. [实施计划](#7-实施计划)
8. [迁移策略](#8-迁移策略)
9. [风险评估](#9-风险评估)
10. [附录](#10-附录)（含版本矩阵、可扩展性、安全路线图）

---

## 0. 评审历史

### 0.1 Oracle 评审 (v1.0 → v1.1)

**评审日期**: 2026-03-10
**评审人**: Oracle Agent
**总体结论**: ✅ 方案可行，但需补充 6 个高风险点

| 风险 | 严重程度 | 改进措施 |
|------|---------|---------|
| HNSW DDL 参数名可能不兼容 | 🔴 高 | 验证 `EFC` 语法 |
| `latest` + SDK 未锁版本 | 🔴 高 | 固定 SurrealDB 镜像版本 + SDK 版本范围 |
| 初始化脚本并发执行风险 | 🟡 中 | 添加 migration lock 机制 |
| FTS 对中文效果可能很差 | 🟡 中 | 评估 `class` tokenizer 效果，预留自定义分词方案 |
| 评分体系混乱风险 | 🟡 中 | 统一 distance/similarity 语义 |
| 安全基线不足 | 🟡 中 | 生产环境使用专用账号 + TLS/WSS |

**Oracle 建议要点**：
- HNSW: M=16 ✅，EFC 建议 200-300，EF 建议动态 `max(40, 4*K)`
- KNN + `vector::distance::knn()` 配合使用
- Schema migration 必须版本化 + migration lock
- 推荐 RRF ���合算法（不吃分数尺度）
- 开发环境建议默认 rocksdb（memory 有偶发慢查询）
- SCHEMAFULL + metadata FLEXIBLE ✅ 正确

### 0.2 多专家审查 (v1.1 → v1.2)

**审查日期**: 2026-03-10
**审查团队**: Librarian ×2 + Explore + Oracle #2
**综合评分**: 72/100 — 架构方向正确，7 处关键错误已修复

| 专家 | 职责 | 核心发现 |
|------|------|----------|
| Librarian #1 | DDL 语法验证 | 🔴 `EF_CONSTRUCTION` → `EFC`，`<\|{ef_search}\|>` → `<\|K,EF\|>` |
| Librarian #2 | Python SDK 验证 | 🔴 SDK 稳定版为 1.0.8，`query()` 多语句只返回最后结果 |
| Explore | 代码一致性 | ✅ 85/100 吻合度，ROADMAP "HNSW 已完成" 与代码不符 |
| Oracle #2 | 实现深度审查 | 🔴 迁移锁需 UPSERT+TTL，Schema 初始化需 fail-fast |

**v1.2 修复清单**：
- [x] HNSW DDL: `EF_CONSTRUCTION` → `EFC`，移除 DDL 中不存在的 `EF_SEARCH` (§5.1)
- [x] KNN 算子: `<|{ef_search}|>` → `<|$limit,$ef_search|>` (§6.1)
- [x] M 默认值: 16 → 12 (§5.2)
- [x] SDK 版本: `>=1.0.0,<3.0.0` → `>=1.0.0,<2.0.0` (§10.5)
- [x] 多语句执行: `query()` → `query_raw()` 或拆分 (§8.2)
- [x] 迁移锁: `CREATE` → `UPSERT` + TTL 防死锁 (§8.2)
- [x] Schema 初始化: 失败时 fail-fast (§8.2)
- [x] 向量查询: 移除 cosine() 双重调用，改用 `vector::distance::knn()` (§6.1)
- [x] RRF: `item["id"]` → `item.get("id")` 防 KeyError (§6.3)
- [x] 实施阶段: Docker 修复提前到 Phase 1 (§7)
- [x] FTS 索引语法: `SEARCH ANALYZER` → `FULLTEXT ANALYZER` (§5.1, 3.0 语法)

**验证来源**：SurrealDB 官方文档、Rust 源码 (`core/src/sql/index.rs`)、官方测试文件、Agno/Camel-AI 实际项目、PyPI SDK 版本。

**下一步**: 待老曹审核通过后，进入 Phase 1 实施。

### 0.3 数据模型融合 + 多租户设计 (v1.2 → v1.3)

**日期**: 2026-03-11
**变更内容**:
- 融合 OpenCode Memory Plugin MemoryEntry 结构，提升 `type`/`tags`/`project_id` 等高频字段为顶层字段
- 加入 `tenant_id` 字段支持行级多租户隔离
- 新增 `project` 表做项目归一化
- 预留组织结构升级路径（Organization → Team → Tenant）
- Schema 版本从 1.0.0 升级到 2.0.0


## 1. 执行摘要

### 1.1 问题陈述

当前 Embedding Service 的记忆管理系统存在以下关键瓶颈：

| 问题 | 严重程度 | 影响 |
|------|---------|------|
| 向量搜索无 HNSW 索引，全表扫描 | 🔴 严重 | 数据量增长后搜索延迟线性增加 |
| 无全文搜索索引，使用 `CONTAINS` 字符串匹配 | 🔴 严重 | 关键词搜索无法利用 BM25 排序 |
| 混合搜索使用硬编码分数合并 | 🟡 中等 | 搜索结果质量不稳定 |
| 数据库无 Schema 定义 | 🟡 中等 | 数据一致性无保障 |
| 无数据库初始化脚本 | 🟡 中等 | 部署不可重复 |
| Docker 数据使用内存模式，重启丢失 | 🔴 严重 | 生产环境数据不持久 |
| SurrealDB SDK 未纳入依赖管理 | 🟡 中等 | 版本不可控 |
| docker-compose.yml 的 wrapper 配置过时 | 🟡 中等 | Docker 部署无法正常工作 |

### 1.2 目标

利用 SurrealDB 3.0（2026-02-17 GA）的新特性，系统性解决上述问题：

- **P0**: 搜索性能提升 10-100x（HNSW 索引 + KNN 操作符）
- **P1**: 搜索质量提升（全文索引 + RRF 混合搜索）
- **P2**: 数据可靠性（Schema 定义 + 持久化存储 + 初始化脚本）
- **P3**: ���构升级（SDK 升级 + Docker 修复 + 图关系基础）

### 1.3 非目标（本次不实施）

- Surrealism WASM 扩展集成
- LIVE SELECT 实时推送
- `DEFINE API` 自定义端点
- 嵌入式部署模式
- Kubernetes Helm Chart

---

## 2. 现状审计

### 2.1 架构概览

```
客户端 → 包装服务 (17999) → Embedding 服务 (18000)
              ↕
          SurrealDB (8000)
              ↕
          memory 存储模式（内存，重启丢失）
```

### 2.2 代码资产清单

| 文件 | 行数 | 职责 | SurrealDB 相关 |
|------|------|------|---------------|
| `wrapper/src/main.py` | 294 | FastAPI 主程序，SurrealDB 连接管理 | `SurrealDBManager` 单例，lifespan 管理 |
| `wrapper/src/config.py` | 94 | 配置管理 | `SurrealDBConfig` dataclass |
| `wrapper/src/utils/memory_manager.py` | 246 | 搜索/上传业务逻辑 | 3 种搜索查询，向量/关键词/混合 |
| `tests/test_db_connection.py` | 49 | 数据库连接测试脚本 | 直接 SDK 调用 |
| `docker-compose.yml` | 101 | Docker 编排 | SurrealDB 容器配置 |
| `docker-compose.dev.yml` | 107 | 开发环境 Docker 编排 | 同上 |

### 2.3 当前 SurrealDB 使用情况

#### 连接方式

```python
# wrapper/src/main.py:71-87
db = Surreal(config.surrealdb.url)  # ws://localhost:8000/rpc
db.connect(config.surrealdb.url)
db.signin({"username": "root", "password": "root"})
db.use(config.surrealdb.namespace, config.surrealdb.database)
```

**注意**：使用 `inspect.isawaitable()` 兼容同步/异步 SDK 版本，说明代码是在 SDK API 不稳定时期编写的。

#### 向量搜索查询

```sql
-- memory_manager.py:170-175 — 全表扫描，未利用索引
SELECT id, content, metadata,
       vector::similarity::cosine(embedding, $embedding) AS score
FROM memory
WHERE vector::similarity::cosine(embedding, $embedding) > $threshold
ORDER BY score DESC LIMIT $limit
```

**问题**：`vector::similarity::cosine()` 函数在 WHERE 子句中每行都要计算一次相似度，时间复杂度 O(n)。

#### 关键词搜索查询

```sql
-- memory_manager.py:183-186 — 无全文索引
SELECT id, content, metadata
FROM memory WHERE content CONTAINS $query LIMIT $limit
```

**问题**：`CONTAINS` 是简单子字符串匹配，无分词、无 BM25 排序、无相关性评分。

#### 混合搜索合并

```python
# memory_manager.py:192-215 — 硬编码分数
for item in keyword_results:
    item["score"] = 0.5  # ← 所有关键词结果都是 0.5 分
merged.sort(key=lambda x: x.get("score", 0), reverse=True)
```

**问题**：关键词结果分数固定为 0.5，无法真实反映相关性。向量结果和关键词结果的分数不在同一尺度上。

### 2.4 缺失项清单

| 项目 | 状态 | 说明 |
|------|------|------|
| HNSW 向量索引 | ❌ 不存在 | 虽然 ROADMAP 标注 "P3-2 HNSW 已完成"，但代码中无任何索引定义 |
| 全文搜索索引 | ❌ 不存在 | 无 `DEFINE ANALYZER`、无 `SEARCH` 索引 |
| 数据库 Schema | ❌ 不存在 | SCHEMALESS 模式，无类型约束 |
| 初始化脚本 | ❌ 不存在 | 无 `.surql` 文件，无 migration 机制 |
| SDK 依赖声明 | ❌ 不存在 | `surrealdb` 不在 `pyproject.toml` 中 |
| 数据持久化 | ❌ 使用 `memory` | Docker 容器重启后数据全部丢失 |
| 图关系 | ❌ 不存在 | 未使用 RELATE、图遍历 |
| Record References | ❌ 不存在 | 3.0 新特性，未采用 |

### 2.5 Docker Compose 问题

**`docker-compose.yml` 和 `docker-compose.dev.yml` 的 wrapper 服务配置已过时**：

```yaml
# 当前（错误）
wrapper:
  build:
    context: ./wrapper-service     # ← 旧目录，实际已改为 ./wrapper
    dockerfile: Dockerfile
  ports:
    - "3001:3001"                  # ← 旧端口，实际已改为 17999
  environment:
    - WRAPPER_PORT=3001            # ← 同上
```

这意味着 `docker-compose up` 无法正确启动 wrapper 服务。

---

## 3. SurrealDB 3.0 能力评估

### 3.1 版本信息

| 项目 | 值 |
|------|-----|
| 版本 | SurrealDB 3.0 GA |
| 发布日期 | 2026-02-17 |
| Python SDK | 1.0.8 稳定版（2.0.0a1 为 alpha，勿用于生产） |
| 序列化 | CBOR 二进制（替代 JSON） |
| 存储引擎 | memory, rocksdb, SurrealKV, TiKV |

### 3.2 与本项目相关的新特性

#### 🔴 立即可用的特性

| 特性 | 收益 | 依赖 |
|------|------|------|
| **HNSW 向量索引优化** | 搜索速度 ~8x 提升 | 创建索引即可 |
| **新执行引擎** | 全局查询加速 3-22x | 升级 SurrealDB 版本即可 |
| **Computed Fields** | 预计算派生字段 | 升级后可用 |
| **Record References** | 双向关系链接 | Schema 定义时添加 |
| **Client-side Transactions** | 多步原子操作 | SDK 2.x 支持 |

#### 🟡 中期可用的特性

| 特性 | 收益 | 依赖 |
|------|------|------|
| **`DEFINE API` 自定义端点** | 减少中间层代码 | 需评估是否替代 FastAPI |
| **GraphQL 稳定版** | 新查询接口 | 需评估是否需要 |
| **Surrealism WASM** | 数据库内执行逻辑 | 需编写 Rust/WASM 扩展 |
| **LIVE SELECT** | 实时推送 | 需 WebSocket 长连接 |

#### ⚪ 暂不适用

| 特性 | 原因 |
|------|------|
| SurrealMCP | 用于连接 AI IDE（Claude/Cursor），非应用层功能 |
| File Storage | 实验性功能，记忆系统不需要文件存储 |
| Surreal Sync | 数据迁移工具，当前无迁移需求 |

### 3.3 性能基准（官方数据，已验证）

| 操作 | SurrealDB 2.x | SurrealDB 3.0 | 提升 |
|------|--------------|--------------|------|
| HNSW 向量搜索 | 38,581 ms | 4,847 ms | **8x** |
| 图遍历 depth3 | 18.02 ms | 3.02 ms | **6x** |
| ORDER BY | 17,270 ms | 5,274 ms | **3.3x** |
| WHERE id= | 3,935 ms | 0.68 ms | **5,787x** |
| Create (SurrealKV) | 221 ms | 8.25 ms | **26.8x** |
| 全文索引构建 | 51,134 ms | 3,511 ms | **14.6x** |

> **数据来源**：[SurrealDB 3.0 Benchmarks](https://surrealdb.com/blog/surrealdb-3-0-benchmarks-a-new-foundation-for-performance)，基于 AMD Ryzen Threadripper 9970X 测试。

---

## 4. 升级方案设计

### 4.1 设计原则

1. **渐进式升级**：不做大爆炸式变更，分阶段实施
2. **向后兼容**：升级过程中现有 API 接口不变
3. **幂等初始化**：数据库初始化脚本可重复执行
4. **配置化**：所有参数通过环境变量或配置管理
5. **可回滚**：每阶段有明确回滚方案

### 4.2 目标架构

```
客户端 → 包装服务 (17999) → Embedding 服务 (18000)
              ↕
          SurrealDB 3.0 (8000)
              ↕
          RocksDB 持久存储 + HNSW 索引
              ↕
          Schema: SCHEMAFULL + 全文索引 + 向量索引
```

### 4.3 变更范围

| 文件 | 变更类型 | 变更内容 |
|------|---------|---------|
| `scripts/init_surrealdb.surql` | **新建** | 数据库 Schema v2.0（memory + project + schema_version + 索引） |
| `wrapper/src/utils/memory_manager.py` | **修改** | 搜索查询增加 tenant_id 过滤 + 上传支持新字段映射 |
| `wrapper/src/main.py` | **修改** | MemoryUploadRequest/MemorySearchRequest 增加 tenant_id + Schema 初始化 |
| `wrapper/src/config.py` | **修改** | 添加 HNSW、搜索、多租户默认配置项 |
| `docker-compose.yml` | **修改** | 修复 wrapper 配置 + 持久化存储 |
| `docker-compose.dev.yml` | **修改** | 同上 |
| `pyproject.toml` | **修改** | 添加 surrealdb 依赖 |
| `tests/test_wrapper_api.py` | **修改** | 新增多租户搜索测试、字段验证测试 |

---

## 5. 数据模型设计

### 5.1 Schema 定义

```sql
-- ============================================================
-- scripts/init_surrealdb.surql
-- Embedding Service 数据库初始化脚本
-- 版本: 2.0.0
-- 兼容: SurrealDB >= 2.0（推荐 3.0+）
-- ============================================================

-- ==================== memory 表 ====================

DEFINE TABLE memory SCHEMAFULL;

-- 核心字段（必需）
DEFINE FIELD content     ON memory TYPE string
    ASSERT $value != NONE AND $value != "";
    -- 记忆原文，全文索引目标

DEFINE FIELD embedding   ON memory TYPE array<float>
    ASSERT array::len($value) = 1024;
    -- 1024 维，匹配 Qwen3-Embedding-0.6B

-- 多租户字段（必需，带默认值以保持向后兼容）
DEFINE FIELD tenant_id   ON memory TYPE string
    DEFAULT "default"
    ASSERT $value != NONE AND $value != "";
    -- 行级多租户隔离。默认值 "default" 保证旧数据和未传 tenant_id 的请求兼容

-- 分类与组织字段（从 metadata 提���为顶层，可索引）
DEFINE FIELD type        ON memory TYPE option<string>
    DEFAULT "general";
    -- 类型: general | preference | decision | note | analysis | test | long-term

DEFINE FIELD tags        ON memory TYPE option<array<string>>
    DEFAULT [];
    -- 自由标签，支持多标签过滤

DEFINE FIELD project_id  ON memory TYPE option<string>
    DEFAULT "global";
    -- 项目标识。"global" = 跨项目通用记忆

-- 来源追踪字段（用于去重和审计）
DEFINE FIELD source_id   ON memory TYPE option<string>;
    -- 客户端原始 ID（如 "mem_1709123456789_abc12345"），UNIQUE 索引防止重复上传

DEFINE FIELD source      ON memory TYPE option<string>
    DEFAULT "api";
    -- 来源标识: api | plugin | import | migration

DEFINE FIELD source_timestamp ON memory TYPE option<datetime>;
    -- 客户端原始时间戳（ISO 8601），与 created_at（服务端时间）区分

-- 质量信号（可选）
DEFINE FIELD classification_confidence ON memory TYPE option<float>;
    -- 自动分类的置信度 [0.0, 1.0]

-- 服务端时间戳（不可变）
DEFINE FIELD created_at  ON memory TYPE datetime
    DEFAULT time::now();

-- 灵活扩展字段（兜底，向后兼容旧 API 的 metadata 传参）
DEFINE FIELD metadata    ON memory FLEXIBLE TYPE object
    DEFAULT {};

-- ==================== memory 索引 ====================

-- 向量搜索索引（HNSW）
-- M=12: SurrealDB 默认值，1K-100K 数据量下平衡精度与内存
-- EFC=200: Oracle 建议 200-300，索引构建质量更高
-- DIST=COSINE: 语义搜索推荐
DEFINE INDEX memory_embedding_hnsw ON memory
    FIELDS embedding HNSW DIMENSION 1024 DIST COSINE
    EFC 200 M 12;

-- 全文搜索索引（BM25）
DEFINE ANALYZER memory_analyzer
    TOKENIZERS blank, class
    FILTERS lowercase, ascii;

DEFINE INDEX memory_content_ft ON memory
    FIELDS content FULLTEXT ANALYZER memory_analyzer BM25;

-- 时间戳索引（用于时间范围查询）
DEFINE INDEX memory_created_at ON memory
    FIELDS created_at;

-- 多租户索引
DEFINE INDEX memory_tenant ON memory FIELDS tenant_id;

-- 分类索引
DEFINE INDEX memory_type ON memory FIELDS type;

-- 项目索引
DEFINE INDEX memory_project ON memory FIELDS project_id;

-- 去重索引（防止重复上传）
DEFINE INDEX memory_source_id ON memory FIELDS source_id UNIQUE;

-- 复合索引（高频查询路径）
DEFINE INDEX memory_tenant_type ON memory FIELDS tenant_id, type;
DEFINE INDEX memory_tenant_project ON memory FIELDS tenant_id, project_id;

-- ==================== project 表 ====================

DEFINE TABLE project SCHEMAFULL;

DEFINE FIELD name        ON project TYPE string;
    -- 项目显示名，如 "embedding_service"

DEFINE FIELD tenant_id   ON project TYPE string
    DEFAULT "default";
    -- 项目所属租户

DEFINE FIELD project_tag ON project TYPE option<string>;
    -- 项目标签（如 "global"/"unclassified"/自定义）

DEFINE FIELD first_seen  ON project TYPE datetime
    DEFAULT time::now();

DEFINE FIELD last_seen   ON project TYPE datetime
    DEFAULT time::now();

DEFINE FIELD entry_count ON project TYPE int
    DEFAULT 0;

DEFINE INDEX project_tenant ON project FIELDS tenant_id;
DEFINE INDEX project_name   ON project FIELDS tenant_id, name UNIQUE;
    -- 同一租户下项目名唯一

-- ==================== schema_version 表 ====================

DEFINE TABLE schema_version SCHEMAFULL;
DEFINE FIELD version     ON schema_version TYPE string;
DEFINE FIELD applied_at  ON schema_version TYPE datetime DEFAULT time::now();
DEFINE FIELD description ON schema_version TYPE string;

-- 记录当前版本（UPSERT 保证幂等）
UPSERT schema_version:current CONTENT {
    version: "2.0.0",
    description: "融合 MemoryEntry 数据模型 + 多租户 tenant_id + project 表"
};
```

### 5.2 HNSW 参数选择依据

| 参数 | 选定值 | 依据 |
|------|--------|------|
| DIMENSION | 1024 | Qwen3-Embedding-0.6B 输出维度 |
| DIST | cosine | 语义相似度标准度量，与现有代码一致 |
| M | 12 | SurrealDB 默认值（12）。1K-100K 数据量下平衡精度与内存 |
|| EFC | 200 | **Oracle 建议 200-300**，高于默认值（150），索引构建质量更高，一次性成本可接受 |
|| EF (查询时) | 50 | 通过 `<|K,EF|>` 算子指定。**Oracle 建议动态 `max(40, 4*K)`**。运行时可配置，不在 DDL 中定义 |

### 5.2.1 字段映射说明（MemoryEntry 融合）

本 Schema 融合了 OpenCode Memory Plugin 的 `MemoryEntry` 接口与原有 SurrealDB 结构。

#### 字段来源映射

| SurrealDB 字段 | 类型 | 来源 | 说明 |
|---|---|---|---|
| `content` | string (必需) | 双方共有 | 记忆原文 |
| `embedding` | array\<float\> (必需) | 服务端计算 | 1024 维向量，上传时自动生成 |
| `tenant_id` | string (必需) | 新增 | 多租户隔离标识，默认 "default" |
| `type` | option\<string\> | 插件 `type` 字段 | 从 metadata 提升为顶层，可索引过滤 |
| `tags` | option\<array\<string\>\> | 插件 `tags` 字段 | 从 metadata 提升为顶层 |
| `project_id` | option\<string\> | 插件 `project_id` | 项目隔离标识 |
| `source_id` | option\<string\> | 插件 `id` (mem_xxx) | 客户端 ID，UNIQUE 索引去重 |
| `source` | option\<string\> | 新增 | 来源标识 (api/plugin/import) |
| `source_timestamp` | option\<datetime\> | 插件 `timestamp` | 客户端原始时间 |
| `classification_confidence` | option\<float\> | 插件同名字段 | 分类置信度信号 |
| `created_at` | datetime | 服务端 | 不可变服务端时间戳 |
| `metadata` | FLEXIBLE object | 双方共有 | 兜底扩展字段，向后兼容 |

#### 明确不入库的插件字段

| 插件字段 | 不入库原因 |
|---|---|
| `uploaded` / `upload_timestamp` / `upload_error` | 客户端同步状态。记录存在于 DB 即代表已上传成功 |
| `project_tag` | 与 `project_id` 冗余，归一化到 `project` 表的 `project_tag` 字段 |
| `project_name` | 归一化到 `project` 表 |
| `classified_at` | 低频使用，可放入 `metadata` 兜底字段 |

#### 向后兼容保证

旧格式 API 调用仍然有效：
```json
{
  "content": "Hello world",
  "metadata": {"key": "value"}
}
```
所有新字段均为 `option<T>` 或带 `DEFAULT`，旧客户端无需任何修改。


### 5.3 全文搜索 Analyzer 设计

```
输入: "Hello World 你好世界"
  ↓ TOKENIZER blank (空格分词)
  → ["Hello", "World", "你好世界"]
  ↓ TOKENIZER class (Unicode 类别分词)
  → ["Hello", "World", "你好", "世界"]
  ↓ FILTER lowercase
  → ["hello", "world", "你好", "世界"]
  ↓ FILTER ascii (去除变音符号)
  → ["hello", "world", "你好", "世界"]
```

**注意**：当前不添加 `snowball` 词干提取器，因为记忆内容以中文为主，snowball 仅对英文有效。后续如需中文分词增强，可考虑自定义 Analyzer。

> ⚠️ **Oracle 警告**：`blank/class` tokenizer 对中文分词效果有限，仅按 Unicode 类别切分（如 CJK 统一表意文字每字一 token）。如果关键词搜索质量不足，需考虑以下替代方案：
> - **预分词字段**：在应用层使用 jieba 等分词器预处理，存储分词结果到独立字段
> - **外部搜索引擎**：接入 Meilisearch/Typesense 做中文全文搜索
> - **纯向量搜索降级**：中文场景放弃 keyword，依赖 hybrid = vector + 元数据过滤

### 5.4 未来扩展：图关系模型

> 本期不实施，但预留设计空间。

```sql
-- 记忆间的关联关系（未来 Phase 3）
DEFINE TABLE relates_to SCHEMAFULL TYPE RELATION
    IN memory OUT memory;
DEFINE FIELD strength   ON relates_to TYPE float DEFAULT 0.5;
DEFINE FIELD created_at ON relates_to TYPE datetime DEFAULT time::now();

-- 使用示例
RELATE memory:abc->relates_to->memory:xyz SET strength = 0.8;

-- 查询某条记忆的关联记忆
SELECT ->relates_to->memory.* FROM memory:abc;
```

### 5.5 多租户架构设计

#### 5.5.1 当前实现：行级隔离（Phase 1）

采用共享表 + `tenant_id` 字段的行级多租户模式：

```
┌──────────────────────────────────────────┐
│          SurrealDB memory_db             │
│                                          │
│  ┌─────────────────────────────────────┐ │
│  │          memory 表                   │ │
│  │  ┌───────────┬───────────────────┐  │ │
│  │  │ tenant_id │ content, ...      │  │ │
│  │  ├───────────┼───────────────────┤  │ │
│  │  │ "default" │ 默认租户的记忆     │  │ │
│  │  │ "user_A"  │ 用户A的记忆        │  │ │
│  │  │ "user_B"  │ 用户B的记忆        │  │ │
│  │  └───────────┴───────────────────┘  │ │
│  └─────────────────────────────────────┘ │
│                                          │
│  ┌─────────────────────────────────────┐ │
│  │          project 表                  │ │
│  │  tenant_id + name = UNIQUE          │ │
│  └─────────────────────────────────────┘ │
└──────────────────────────────────────────┘
```

**优势**：
- 实现简单，无需修改连接管理
- 复合索引 `(tenant_id, type)` 和 `(tenant_id, project_id)` 保证查询性能
- 默认值 "default" 保证向后兼容

**查询示例**：
```sql
-- 租户隔离的向量搜索
SELECT id, content, metadata, type, tags, project_id,
       vector::distance::knn() AS distance
FROM memory
WHERE tenant_id = $tenant_id
  AND embedding <|$limit,$ef_search|> $embedding
ORDER BY distance ASC

-- 租户隔离的关键词搜索
SELECT id, content, metadata, type, tags, project_id,
       search::score(1) AS score
FROM memory
WHERE tenant_id = $tenant_id
  AND content @1@ $query
ORDER BY score DESC LIMIT $limit
```

**API 变更**：
```python
class MemoryUploadRequest(BaseModel):
    memories: list[dict] = Field(..., description="记忆列表")
    tenant_id: str = Field(default="default", description="租户ID")

class MemorySearchRequest(BaseModel):
    query: str = Field(..., description="搜索查询")
    mode: str = Field(default="hybrid", description="搜索模式")
    limit: int = Field(default=10, ge=1, le=100)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    tenant_id: str = Field(default="default", description="租户ID")
```

#### 5.5.2 未来演进：数据库级隔离（Phase 2，暂不实施）

当租户数量增长或需要更强隔离时，可利用 SurrealDB 的原生层级结构：

```
SurrealDB 层级:
  Namespace (组织) → Database (租户) → Table (数据)

示例:
  namespace: "org_acme"
    database: "tenant_alice"
      table: memory, project, schema_version
    database: "tenant_bob"
      table: memory, project, schema_version
```

**迁移策略**：
1. 为每个租户创建独立 database
2. 按 `tenant_id` 分批迁移数据
3. 更新连接管理：按 tenant_id 路由到对应 database
4. 此时可移除 `tenant_id` 字段（隔离由 database 保证）

#### 5.5.3 远期预留：组织结构（Phase 3，仅设计不实施）

```
Organization (组织)
  └── Team (团队)
       └── Tenant (租户/用户)
            └── Project (项目)
                 └── Memory (记忆条目)
```

**预留 Schema（远期实施时创建）**：

```sql
-- 组织表
DEFINE TABLE organization SCHEMAFULL;
DEFINE FIELD name       ON organization TYPE string;
DEFINE FIELD settings   ON organization FLEXIBLE TYPE object DEFAULT {};
DEFINE FIELD created_at ON organization TYPE datetime DEFAULT time::now();

-- 团队表
DEFINE TABLE team SCHEMAFULL;
DEFINE FIELD name       ON team TYPE string;
DEFINE FIELD org_id     ON team TYPE record<organization>;
DEFINE FIELD created_at ON team TYPE datetime DEFAULT time::now();
DEFINE INDEX team_org   ON team FIELDS org_id;

-- 租户详情表（扩展 tenant_id 的元数据）
DEFINE TABLE tenant SCHEMAFULL;
DEFINE FIELD name       ON tenant TYPE string;
DEFINE FIELD team_id    ON tenant TYPE option<record<team>>;
DEFINE FIELD org_id     ON tenant TYPE option<record<organization>>;
DEFINE FIELD settings   ON tenant FLEXIBLE TYPE object DEFAULT {};
DEFINE FIELD created_at ON tenant TYPE datetime DEFAULT time::now();
DEFINE INDEX tenant_org ON tenant FIELDS org_id;

-- 关系图示例
-- RELATE organization:acme->has_team->team:backend
-- RELATE team:backend->has_member->tenant:alice
-- RELATE tenant:alice->owns_project->project:embedding_service
```

**注意**：远期实施时，`memory.tenant_id` 可改为 `record<tenant>` 类型以支持图查询：
```sql
-- 远期：查询某组织下所有记忆
SELECT * FROM memory
WHERE tenant_id->team_id->org_id = organization:acme
```


---

## 6. 搜索引擎升级

### 6.1 向量搜索：KNN 操作符 + 索引利用

**升级前**（全表扫描，O(n)）：
```sql
SELECT id, content, metadata,
       vector::similarity::cosine(embedding, $embedding) AS score
FROM memory
WHERE vector::similarity::cosine(embedding, $embedding) > $threshold
ORDER BY score DESC LIMIT $limit
```

**升级后**（HNSW 索引，O(log n)）：
```sql
-- <|K,EF|> 算子：K=返回最近邻数量，EF=搜索候选集大小（越大越精确但越慢）
-- vector::distance::knn() 复用索引已计算的距离，无需重复调用 cosine()
SELECT id, content, metadata, type, tags, project_id,
       vector::distance::knn() AS distance
FROM memory
WHERE tenant_id = $tenant_id
  AND embedding <|$limit,$ef_search|> $embedding
ORDER BY distance ASC
```

> ⚠️ **v1.2 修复说明**：
> - `<|{ef_search}|>` → `<|$limit,$ef_search|>`：KNN 算子需要两个参数（K=结果数, EF=候选集），不是一个
> - `vector::similarity::cosine()` → `vector::distance::knn()`：KNN 算子已通过索引计算距离，用 `knn()` 复用结果避免重复计算
> - `ORDER BY score DESC` → `ORDER BY distance ASC`：距离越小越相似（与相似度方向相反）
> - 如需阈值过滤，可在应用层对 distance 结果做后过滤（`distance < threshold`）

**关键变化**：
- `<|K,EF|>` KNN 操作符触发 HNSW 索引查询，K 为返回数量，EF 为候选集大小
- `vector::distance::knn()` 复用索引已计算的距离分数，无重复计算开销
- EF 参数运行时可配置，默认 50，Oracle 建议 `max(40, 4*K)`

> **验证来源**：SurrealDB 算子文档 `<|K,EF|>` + 官方测试 `WHERE point <|10,40|> [2,3,4,5]`（K=10, EF=40）

### 6.2 关键词搜索：全文索引 + BM25 评分

**升级前**（子字符串匹配，无排序）：
```sql
SELECT id, content, metadata
FROM memory WHERE content CONTAINS $query LIMIT $limit
```

**升级后**（全文搜索 + BM25 评分）：
```sql
SELECT id, content, metadata, type, tags, project_id,
       search::score(1) AS score
FROM memory
WHERE tenant_id = $tenant_id
  AND content @1@ $query
ORDER BY score DESC LIMIT $limit
```

**关键变化**：
- `@1@` 是 SurrealDB 全文搜索操作符，数字 `1` 是索引引用 ID
- `search::score(1)` 返回 BM25 相关性分数
- 自动分词、小写化、相关性排序

### 6.3 混合搜索：RRF 融合算法

**升级前**（硬编码分数合并）：
```python
for item in keyword_results:
    item["score"] = 0.5  # 所有关键词结果固定 0.5 分
merged.sort(key=lambda x: x.get("score", 0), reverse=True)
```

**升级后**（Reciprocal Rank Fusion）：

```python
def _rrf_merge(
    self,
    vector_results: list[dict],
    keyword_results: list[dict],
    k: int = 60,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
) -> list[dict]:
    """
    RRF (Reciprocal Rank Fusion) 混合搜索合并算法

    公式: RRF_score(d) = Σ (weight_i / (k + rank_i(d)))

    参数:
        k: 排名平滑常数（默认 60，来自原始论文）
        vector_weight: 向量搜索权重（默认 0.7）
        keyword_weight: 关键词搜索权重（默认 0.3）
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    # 向量搜索贡献
    for rank, item in enumerate(vector_results):
        doc_id = item.get("id")  # 防御性检查，避免 KeyError
        if not doc_id:
            continue
        scores[doc_id] = vector_weight / (k + rank + 1)
        items[doc_id] = item

    # 关键词搜索贡献
    for rank, item in enumerate(keyword_results):
        doc_id = item.get("id")  # 防御性检查，避免 KeyError
        if not doc_id:
            continue
        scores.setdefault(doc_id, 0.0)
        scores[doc_id] += keyword_weight / (k + rank + 1)
        if doc_id not in items:
            items[doc_id] = item

    # 按 RRF 分数排序
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    results = []
    for doc_id in sorted_ids:
        item = items[doc_id].copy()
        item["score"] = round(scores[doc_id], 6)
        results.append(item)

    return results
```

**RRF 算法优势**：
- 不依赖原始分数的绝对值（向量分数和 BM25 分数天然不可比）
- 只依赖排名位置，天然归一化
- k=60 是原始论文（Cormack et al. 2009）推荐值
- 权重参数可通过配置调整

### 6.4 搜索配置项扩展

```python
# config.py 新增配置
@dataclass
class SearchConfig:
    # 阈值（已有）
    keyword_threshold: float = 0.0
    vector_threshold: float = 0.75
    hybrid_threshold: float = 0.75

    # RRF 参数（新增）
    rrf_k: int = 60                    # RRF 平滑常数
    rrf_vector_weight: float = 0.7     # 向量搜索权重
    rrf_keyword_weight: float = 0.3    # 关键词搜索权重

    # HNSW 查询参数（新增）
    hnsw_ef_search: int = 50           # HNSW 查询候选集大小
```

对应环境变量：
```
WRAPPER_SEARCH_RRF_K=60
WRAPPER_SEARCH_RRF_VECTOR_WEIGHT=0.7
WRAPPER_SEARCH_RRF_KEYWORD_WEIGHT=0.3
WRAPPER_SEARCH_HNSW_EF_SEARCH=50
```

---

## 7. 实施计划

### 7.1 阶段划分

```
Phase 1 (P0): 基础设施修复 + 搜索引擎升级  ← 最大 ROI，3-4 天
Phase 2 (P1): 增强功能                      ← 未来迭代，按需
```

> ⚠️ **v1.2 调整**：原 Phase 2（Docker 修复）合并到 Phase 1。
> 原因：开发者需要先修好 docker-compose 才能测试搜索查询变更（Oracle #2 建议）。

### 7.2 Phase 1: 基础设施修复 + 搜索引擎升级（P0，3-4 天）

**Phase 1A：基础设施（前置，1 天）**

| 序号 | 任务 | 文件 | 预计工时 |
|------|------|------|---------|
| 1.1 | 修复 docker-compose.yml wrapper 配置 | `docker-compose.yml` | 0.25 天 |
| 1.2 | 修复 docker-compose.dev.yml | `docker-compose.dev.yml` | 0.25 天 |
| 1.3 | SurrealDB 切换到 RocksDB 持久化 | `docker-compose*.yml` | 0.25 天 |
| 1.4 | 固定 SurrealDB 镜像版本 `v3.0.0` | `docker-compose*.yml` | 0.1 天 |
| 1.5 | 添加 `surrealdb>=1.0.0,<2.0.0` 到 pyproject.toml | `pyproject.toml` | 0.1 天 |

**Phase 1B：搜索引擎升级（核心，2-3 天）**

| 序号 | 任务 | 文件 | 预计工时 |
|------|------|------|---------|
| 1.6 | 创建数据库初始化脚本（Schema v2.0） | `scripts/init_surrealdb.surql` | 0.5 天 |
| 1.7 | 实现 Schema 自动初始化（fail-fast + migration lock） | `wrapper/src/main.py` | 0.5 天 |
| 1.8 | upload_memories 字段映射（tenant_id/type/tags/project_id/source_id 提取） | `wrapper/src/utils/memory_manager.py` | 0.5 天 |
| 1.9 | 向量搜索查询重写（KNN `<\|K,EF\|>` + tenant_id 过滤） | `wrapper/src/utils/memory_manager.py` | 0.5 天 |
| 1.10 | 关键词搜索查询重写（全文索引 BM25 + tenant_id 过滤） | `wrapper/src/utils/memory_manager.py` | 0.5 天 |
| 1.11 | 实现 RRF 混合搜索算法 | `wrapper/src/utils/memory_manager.py` | 0.5 天 |
| 1.12 | 新增搜索配置项 | `wrapper/src/config.py` | 0.25 天 |
| 1.13 | 更新/新增测试 | `tests/` | 0.5 天 |

**Phase 1 验收标准**：
- [ ] `docker-compose up` 可以正确启动所有服务
- [ ] SurrealDB 容器重启后数据不丢失
- [ ] `uv pip install -e .` 包含 surrealdb SDK
- [ ] HNSW 索引成功创建（通过 `INFO FOR TABLE memory` 验证）
- [ ] 向量搜索使用 KNN 操作符（通过查询 EXPLAIN 验证）
- [ ] 关键词搜索返回 BM25 分数（score > 0 且有排序）
- [ ] 混合搜索使用 RRF 算法（向量优先结果排在前面）
- [ ] Schema 初始化失败时服务拒绝启动（fail-fast）
- [ ] 不同 tenant_id 的数据互相不可见（多租户隔离验证）
- [ ] 重复 source_id 上传被 UNIQUE 索引拒绝
- [ ] 所有现有测试通过（无回归）
- [ ] 新增测试覆盖 HNSW/全文搜索/RRF 场景

### 7.3 Phase 2: 增强功能（P1，按需）

| 序号 | 任务 | 预计工时 | 触发条件 |
|------|------|---------|---------|
| 2.1 | 清理 `inspect.isawaitable()` 兼容层 | 0.5 天 | SDK 版本确认后 |
| 2.2 | 更新 .env.example | 0.1 天 | Phase 1 完成后 |
| 2.3 | 图关系模型（RELATE） | 2-3 天 | 需要记忆间关联时 |
| 2.4 | Record References | 1 天 | 需要双向查询时 |
| 2.5 | Computed Fields | 0.5 天 | 需要预计算派生值时 |
| 2.6 | LIVE SELECT 实时推送 | 2 天 | 需要实时通知时 |
| 2.7 | Client-side Transactions | 1 天 | 需要多步原子操作时 |
---

## 8. 迁移策略

### 8.1 数据迁移方案

**当前状态**：Docker 使用 `memory` 模式，重启丢失。

**迁移路径**：

```
场景 A: 开发环境（无需迁移现有数据）
  1. 更新 docker-compose.yml（memory → rocksdb）
  2. 重新启动容器
  3. 运行 init_surrealdb.surql 初始化 Schema
  4. 重新上传测试数据

场景 B: 生产环境（如有持久化数据需迁移）
  1. 导出现有数据: surreal export --conn ... > backup.surql
  2. 停止旧版 SurrealDB
  3. 启动 SurrealDB 3.0（rocksdb 模式）
  4. 运行 init_surrealdb.surql 创建 Schema
  5. 导入数据: surreal import --conn ... < backup.surql
  6. 验证数据完整性
  7. 验证索引状态: INFO FOR TABLE memory
```

### 8.2 Schema 初始化流程

> ⚠️ **v1.2 重大修复**：
> 1. Schema 初始化失败时 **fail-fast**（`raise SystemExit(1)`），不允许服务在残缺状态运行
> 2. 迁移锁改用 **UPSERT + TTL** 防死锁，取代 `CREATE`（非幂等）
> 3. 多语句执行改用 **`query_raw()`** 或逐条拆分执行（`query()` 只返回最后结果）

```python
# wrapper/src/main.py 新增方法

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SurrealDBManager:
    async def ensure_schema(self):
        """确保数据库 Schema 已初始化（幂等操作 + migration lock + fail-fast）"""
        # [v1.2] Schema 初始化失败必须阻止服务启动
        lock_acquired = False
        try:
            lock_acquired = await self._acquire_migration_lock()
            if not lock_acquired:
                logger.info("[Schema] 其他实例正在执行 migration，跳过")
                return

            # 检查是否已初始化
            result = await self._db_query("SELECT * FROM schema_version ORDER BY applied_at DESC LIMIT 1")
            if result and len(result) > 0:
                version = result[0].get("version", "unknown")
                logger.info(f"[Schema] 当前版本: {version}")
                return

            # 首次初始化
            logger.info("[Schema] 首次初始化，执行 init_surrealdb.surql...")
            init_script = (
                Path(__file__).parent.parent.parent / "scripts" / "init_surrealdb.surql"
            )
            if not init_script.exists():
                raise FileNotFoundError(f"初始化脚本不存在: {init_script}")

            sql = init_script.read_text(encoding="utf-8")

            # [v1.2] 使用 query_raw() 获取所有语句的结果（query() 只返回最后一条）
            # 或者拆分为单条执行以逐条验证
            statements = [s.strip() for s in sql.split(";") if s.strip()]
            for stmt in statements:
                await self._db_query(stmt)

            logger.info("[Schema] 初始化完成")

        except Exception as e:
            # [v1.2] fail-fast：Schema 初始化失败必须终止服务
            logger.critical(f"[Schema] 初始化失败，服务无法启动: {e}")
            raise SystemExit(1) from e
        finally:
            if lock_acquired:
                await self._release_migration_lock()

    async def _acquire_migration_lock(self) -> bool:
        """
        获取 migration 锁（基于 SurrealDB 记录）

        [v1.2 修复]:
        - CREATE → UPSERT（幂等，不抛 AlreadyExistsError）
        - 添加 TTL（locked_until）防止进程崩溃后永久死锁
        - 区分锁竞争 vs 连接失败
        """
        try:
            result = await self._db_query(
                "UPSERT migration_lock:global SET "
                "  locked = true, "
                "  locked_by = $instance_id, "
                "  locked_at = time::now(), "
                "  locked_until = time::now() + 5m "
                "WHERE locked = false OR locked_until < time::now()",
                {"instance_id": str(id(self))}
            )
            # UPSERT 返回空列表表示 WHERE 不满足（锁被其他实例持有且未过期）
            return bool(result and len(result) > 0)
        except ConnectionError:
            logger.error("[Schema] 无法连接 SurrealDB，获取锁失败")
            raise  # 连接失败应上抛，触发 fail-fast
        except Exception as e:
            logger.warning(f"[Schema] 获取锁异常: {e}")
            return False

    async def _release_migration_lock(self):
        """释放 migration 锁"""
        try:
            await self._db_query(
                "UPDATE migration_lock:global SET locked = false"
            )
        except Exception:
            pass  # 锁释放失败不影响正常流程（TTL 会自动过期）
```

**调用时机**：在 `lifespan` 启动阶段，`connect()` 之后、`MemoryManager` 初始化之前。
Schema 初始化失败将导致服务直接退出（`SystemExit(1)`），确保不会在残缺状态下接受请求。

### 8.3 回滚方案

| 阶段 | 回滚方式 | 操作 |
|------|---------|------|
| Phase 1 | Git revert | 回退搜索查询和 Schema 初始化代码 |
| Phase 2 | Docker 配置回退 | 恢复 docker-compose.yml 到 `memory` 模式 |
| Phase 3 | 增量回退 | 只移除新增的图关系代码，不影响核心搜索 |

---

## 9. 风险评估

### 9.1 风险矩阵

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| HNSW DDL 参数名不兼容 | ~~中~~ | ~~高~~ | **[v1.2 已解决]** 确认正确语法为 `EFC`（非 `EF_CONSTRUCTION`），已通过源码+官方文档+实际项目三重验证 |
| HNSW 索引参数不优 | 中 | 低 | 参数可运行时通过 `REBUILD INDEX` 调整 |
| SurrealDB 3.0 Python SDK 兼容性 | 低 | 高 | **[v1.2 更新]** 稳定版为 1.0.8，2.0.0a1 为 alpha。约束 `>=1.0.0,<2.0.0` |
| 全文搜索对中文分词支持不足 | 中 | 中 | **[Oracle 警告]** `class` tokenizer 对 CJK 每字一 token，预留替代方案 |
| 评分体系混乱 | 中 | 中 | **[Oracle 新增]** 统一 distance/similarity 语义，文档写清 |
| RocksDB 模式下磁盘空间增长 | 低 | 低 | 监控磁盘使用，设置数据保留策略 |
| 初始化脚本并发执行 | 中 | 高 | **[Oracle 新增]** migration lock 机制（已添加到 8.2 设计） |
| 安全基线不足 | 中 | 中 | **[Oracle 新增]** 生产用专用 DB 账号，migration/runtime 权限拆分 |
| docker-compose 修改影响现有部署 | 低 | 中 | 保留 docker-compose.dev.yml 作为开发备选 |
| KNN 操作符语法不兼容旧版 SurrealDB | 低 | 高 | 配置化开关：当检测到旧版时回退到函数调用方式 |

### 9.2 已解决的设计问题（v1.2 验证完成）

| 问题 | v1.1 状态 | v1.2 结论 |
|------|---------|--------|
| `@1@` 全文搜索操作符中的索引引用 ID | 假设使用 `1` | ✅ **已验证**：`1` 是 DEFINE INDEX 时的引用 ID，`search::score(1)` 返回对应索引的 BM25 分数 |
| KNN `<\|K\|>` 中参数的含义 | 假设为 ef_search | ✅ **已修正**：`<\|K,EF\|>` 需要两个参数，K=返回数量，EF=候选集大小 |
| HNSW 索引在 `memory` 存储模式下是否持久 | 假设不持久 | ✅ **确认**：内存模式重启丢失，已改用 RocksDB |
| Python SDK `query()` 是否支持多语句 | 假设支持 | ⚠️ **已修正**：`query()` 多语句只返回最后结果，改为逐条执行或 `query_raw()` |

### 9.3 仍需实施验证的问题

| 问题 | 验证方式 | 阶段 |
|------|---------|------|
| EFC=200 是否显著优于默认 EFC=150 | 构建索引后对比 Recall@K | Phase 1 |
| `class` tokenizer 对中文分词效果 | 测试实际中文记忆搜索质量 | Phase 1 |
| `vector::distance::knn()` 返回值语义（距离 vs 相似度） | 实际查询验证 | Phase 1 |
| UPSERT + WHERE 条件锁在 SurrealDB 3.0 的行为 | 多实例并发测试 | Phase 1 |
| `source_id` UNIQUE 索引对 NONE 值的处理 | 多条无 source_id 记录并存测试（SurrealDB 是否允许多个 NONE） | Phase 1 |

---

## 10. 附录

### 10.1 Kimi vs Qwen 分析对比摘要

对两份参考文档的事实核查摘要：

| 声明 | 核查结果 |
|------|---------|
| SurrealDB 3.0 GA 发布 | ✅ 2026-02-17 确认 |
| 向量搜索 8x 提升 | ✅ 基准数据验证 |
| 图遍历 8-22x 提升 | ✅ 基准数据验证 |
| Kimi 的 `embedding<->query_vector` 语法 | ❌ `<->` 是图遍历操作符，非向量相似度 |
| Qwen 的 `system:metrics` 查询 | ⚠️ 无法在官方文档中验证 |
| Kimi 的 MTREE 索引建议 | ⚠️ MTREE 是旧方案，3.0 推荐 HNSW |
| 两者的 Python SDK 示例 | ⚠️ 部分 API 与最新 SDK 2.x 不完全一致 |

### 10.2 当前 SurrealDB 查询与升级后查询对照表

| 搜索模式 | 当前查询 | 升级后查询 |
|---------|---------|-----------|
| 向量 | `vector::similarity::cosine(embedding, $e) > $t` | `embedding <\|$limit,$ef_search\|> $e` + `vector::distance::knn() AS distance` |
| 关键词 | `content CONTAINS $q` | `content @1@ $q` + `search::score(1) AS score` |
| 混合 | 并行执行两种搜索 + 硬编码分数合并 | 并行执行两种搜索 + RRF 融合 |

### 10.3 参考资料

1. [SurrealDB 3.0 发布博客](https://surrealdb.com/blog/introducing-surrealdb-3-0--the-future-of-ai-agent-memory)
2. [SurrealDB 3.0 性能基准](https://surrealdb.com/blog/surrealdb-3-0-benchmarks-a-new-foundation-for-performance)
3. [SurrealDB Python SDK 文档](https://surrealdb.com/docs/sdk/python)
4. [SurrealDB HNSW 索引文档](https://surrealdb.com/docs/surrealql/statements/define/indexes)
5. [RRF 论文 - Cormack et al. 2009](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
6. [LangChain SurrealDB Vector Store](https://github.com/langchain-ai/langchain-community/blob/main/libs/community/langchain_community/vectorstores/surrealdb.py)
7. [Agno + SurrealDB 集成](https://github.com/agno-agi/agno/blob/main/cookbook/07_knowledge/09_archive/filters/filtering_surrealdb.py)

### 10.4 环境变量完整清单（升级后）

| 变量名 | 默认值 | 说明 | 状态 |
|--------|--------|------|------|
| `WRAPPER_PORT` | `17999` | 包装服务端口 | 已有 |
| `WRAPPER_HOST` | `0.0.0.0` | 监听地址 | 已有 |
| `WRAPPER_CACHE_ENABLED` | `true` | 启用 LRU 缓存 | 已有 |
| `WRAPPER_EMBEDDING_SERVICE_URL` | `http://localhost:18000` | Embedding 服务地址 | 已有 |
| `WRAPPER_SURREALDB_URL` | `ws://localhost:8000/rpc` | SurrealDB 地址 | 已有 |
| `WRAPPER_SEARCH_VECTOR_THRESHOLD` | `0.75` | 向量搜索阈值 | 已有 |
| `WRAPPER_SEARCH_HYBRID_THRESHOLD` | `0.75` | 混合搜索阈值 | 已有 |
| `WRAPPER_SEARCH_KEYWORD_THRESHOLD` | `0.0` | 关键词搜索阈值 | 已有 |
| `WRAPPER_SEARCH_RRF_K` | `60` | RRF 平滑常数 | **新增** |
| `WRAPPER_SEARCH_RRF_VECTOR_WEIGHT` | `0.7` | RRF 向量搜索权重 | **新增** |
| `WRAPPER_SEARCH_RRF_KEYWORD_WEIGHT` | `0.3` | RRF 关键词搜索权重 | **新增** |
| `WRAPPER_SEARCH_HNSW_EF_SEARCH` | `50` | HNSW 查询候选集 | **新增** |
| `WRAPPER_DEFAULT_TENANT_ID` | `default` | 默认租户 ID（API 未传时使用） | **新增** |
| `WRAPPER_SURREALDB_NAMESPACE` | `memory_ns` | 命名空间 | 已有（docker-compose） |
| `WRAPPER_SURREALDB_DATABASE` | `memory_db` | 数据库名 | 已有（docker-compose） |

---

> **下一步**：待老曹审核通过后，进入 Phase 1 实施。
### 10.5 版本兼容矩阵（Oracle 建议新增）

> 确保 Server 与 SDK 版本配对正确。

| SurrealDB Server | Python SDK | 兼容性 | 备注 |
|-----------------|-----------|--------|------|
| 3.0.x | 1.0.x | ⚠️ 未验证 | 旧 SDK 可能缺少 3.0 新特性支持 |
| 3.0.x | 2.x | ✅ 推荐 | 官方文档确认支持 |
| 2.x | 2.x | ✅ 兼容 | 无 3.0 新特性 |
| 2.x | 1.0.x | ✅ 兼容 | 当前使用中 |

**建议**：
- `pyproject.toml` 中声明 `surrealdb>=1.0.0,<2.0.0`（排除 2.0.0 alpha）
- 或锁定已验证版本 `surrealdb==1.0.8`
- `docker-compose.yml` 中固定 `surrealdb/surrealdb:v3.0.0`
- 定期验证版本兼容性

> ⚠️ **v1.2 修复**：原 `surrealdb>=1.0.0,<3.0.0` 允许安装 2.0.0a1 alpha 版本，已收窄为 `<2.0.0`。

### 10.6 可扩展性改进（Oracle 建议新增）

当数据量从 1K 增长到 100K+ 时，需关注：

1. **metadata 热点字段已提升** ✅：`tenant_id`、`type`、`project_id`、`tags` 已在 v1.3 中提升为顶层字段并添加索引（见 §5.1）。后续如有新热点字段（如 `session_id`），按同样模式处理

2. **避免全表重复计算**：`vector::similarity::cosine()` 仅对候选集计算，不在全表 WHERE 中重复使用

3. **观测性指标**（Oracle 建议跟踪）：
   - P95 搜索延迟（按搜索模式分类）
   - Recall@K（需要离线评测集）
   - 索引命中率（EXPLAIN 分析）
   - FTS 命中率

### 10.7 安全改进路线图（Oracle 建议新增）

| 阶段 | 改进 | 优先级 |
|------|------|--------|
| Phase 1 | 确保 init 脚本使用 root 权限执行 | P0 |
| Phase 2 | 创建专用 DB 用户（runtime 最小权限） | P1 |
| Phase 2 | migration 权限与 runtime ��限拆分 | P1 |
| Phase 2 | 内网隔离，外网走 TLS/WSS | P1 |
| Phase 3 | 组织结构权限模型（Organization/Team 级隔离，见 §5.5.3） | P2 |

**当前 root/root + ws:// 的风险**：
- 开发环境：可接受（本地访问）
- 生产环境：必须替换为专用账号 + WSS 加密连接

