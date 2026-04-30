# BACKLOG v3.3

> **版本**: v3.3.0  
> **创建日期**: 2026-04-15  
> **目标**: 完成 v3.2 遗留工作，实现 PrecomputeService 核心功能和 Stub 端点  
> **预估总工时**: 8 天  
> **协议**: AGENT-COLLABORATION-PROTOCOL-v1.0

---

## 任务总览

| 编号 | 目标 | 优先级 | 工时 | 状态 | 详情 |
|------|------|--------|------|------|------|
| **PrecomputeService 完善** |
| BL-C-1 | PrecomputeService — 初始化资源实现 | P1 | 1.5 天 | ✅ | [详情](#bl-c-1-p1-precomputeservice--初始化资源实现) |
| BL-C-2 | PrecomputeService — 资源清理实现 | P1 | 0.5 天 | ✅ | [详情](#bl-c-2-p1-precomputeservice--资源清理实现) |
| BL-C-3 | PrecomputeService — 文件处理逻辑实现 | P1 | 2 天 | ✅ | [详情](#bl-c-3-p1-precomputeservice--文件处理逻辑实现) |
| **Stub 端点实现** |
| BL-C-4 | HNSW 索引管理端点 | P2 | 1 天 | ✅ | [详情](#bl-c-4-p2-hnsw-索引管理端点) |
| BL-C-5 | 缓存管理端点 | P2 | 1 天 | ✅ | [详情](#bl-c-5-p2-缓存管理端点) |
| BL-C-6 | 代码分析端点 | P2 | 1 天 | ✅ | [详情](#bl-c-6-p2-代码分析端点) |
| BL-C-7 | 记忆聚类端点 | P3 | 0.5 天 | ✅ | [详情](#bl-c-7-p3-记忆聚类端点) |
| BL-C-8 | 预取功能端点 | P3 | 0.5 天 | ✅ | [详情](#bl-c-8-p3-预取功能端点) |
| **Atom Architecture 后端实施** |
| BL-C-9 | Entity 内联 Atom 创建 | P1 | 1 天 | ✅ | [详情](#bl-c-9-p1-entity-内联-atom-创建) |
| BL-C-10 | 统一搜索 + Atom 层级过滤 | P1 | 1 天 | ✅ | [详情](#bl-c-10-p1-统一搜索--atom-层级过滤) |
| BL-C-11 | 跨 Entity Atom 链接 | P2 | 0.5 天 | ✅ | [详情](#bl-c-11-p2-跨-entity-atom-链接) |
| BL-C-12 | 上下文预算管理 | P1 | 1 天 | ✅ | [详情](#bl-c-12-p1-上下文预算管理) |

---

## PrecomputeService 完善

### BL-C-1 [P1] PrecomputeService — 初始化资源实现

**目标**  
实现 PrecomputeService 的初始化资源逻辑，包括 tree-sitter 初始化、配置加载和数据库连接建立。

**涉及范围**  

- 文件: `wrapper/src/services/precompute.py`（修改）
- 文件: `wrapper/src/services/code_parser.py`（依赖）
- 新增: `tests/test_precompute_init.py`（测试）

**前置依赖**  

- v3.2 PrecomputeService 基础架构已完成
- tree-sitter 语言包已安装
- SurrealDB 连接配置可用

**完成标准**  

- [x] tree-sitter 解析器初始化（支持 Python、JavaScript、TypeScript、Go）
- [x] 配置加载（从 config.py 读取预计算配置）
- [x] 数据库连接建立（SurrealDB 和 Meilisearch）
- [x] 性能监控器启动
- [x] 并发控制器初始化
- [x] 删除 `start()` 方法中的 TODO 注释

**验证方式**  

```bash
# 单元测试
uv run pytest tests/test_precompute_init.py -v
```

**实际完成**  

- ✅ 修改 `wrapper/src/services/precompute.py`:
  - 导入 `CodeParser`
  - 添加 `_code_parser` 属性
  - 实现 `start()` 方法中的初始化逻辑
  - 删除 TODO 注释
- ✅ 创建 `tests/test_precompute_init.py` - 6 个测试全部通过
- ✅ 验证 tree-sitter 解析器正确初始化
- ✅ 验证数据库连接检查
- ✅ 验证性能监控器启动

---

### BL-C-2 [P1] PrecomputeService — 资源清理实现

**目标**  
实现 PrecomputeService 的资源清理逻辑，确保服务停止时正确释放资源。

**涉及范围**  

- 文件: `wrapper/src/services/precompute.py`（修改）
- 新增: `tests/test_precompute_cleanup.py`（测试）

**前置依赖**  

- BL-C-1 初始化资源实现完成

**完成标准**  

- [x] 关闭 tree-sitter 解析器
- [x] 关闭数据库连接（由调用方管理）
- [x] 停止性能监控器
- [x] 释放并发控制资源
- [x] 清理队列和处理中任务
- [x] 删除 `stop()` 方法中的 TODO 注释

**验证方式**  

```bash
# 单元测试
uv run pytest tests/test_precompute_cleanup.py -v
```

**实际完成**  

- ✅ 修改 `wrapper/src/services/precompute.py`:
  - 实现 `_cleanup_concurrency_resources()` 方法
  - 清理处理中集合和队列
  - 清理 tree-sitter 解析器
  - 删除 TODO 注释
- ✅ 创建 `tests/test_precompute_cleanup.py` - 6 个测试全部通过
- ✅ 验证并发控制资源正确清理
- ✅ 验证性能监控器停止

---

### BL-C-3 [P1] PrecomputeService — 文件处理逻辑实现

**目标**  
实现完整的文件处理逻辑，包括 AST 解析、符号提取和批量创建 atoms。

**涉及范围**  

- 文件: `wrapper/src/services/precompute.py`（修改）
- 文件: `wrapper/src/services/code_parser.py`（修改）
- 文件: `wrapper/src/services/fingerprint.py`（依赖）
- 新增: `tests/test_precompute_file_processing.py`（测试）

**前置依赖**  

- BL-C-1 初始化资源实现完成
- tree-sitter Query API 可用
- FingerprintManager 已实现

**完成标准**  

- [x] 文件读取和编码检测
- [x] AST 解析（使用 tree-sitter）
- [x] 符号提取（函数、类、接口等）
- [x] 指纹计算和变更检测
- [x] Atoms 批量创建
- [ ] 关系提取和创建（后续任务）
- [x] 性能指标收集
- [x] 删除 `process_file()` 方法中的 TODO 注释

**验证方式**  

```bash
# 单元测试
uv run pytest tests/test_precompute_file_processing.py -v
```

**实际完成**  

- ✅ 修改 `wrapper/src/services/precompute.py`:
  - 导入 `FingerprintManager`
  - 添加 `_fingerprint_manager` 属性
  - 实现 `_process_file()` 方法
  - 实现 `_read_file()` 方法
  - 删除 TODO 注释
- ✅ 修改 `wrapper/src/services/code_parser.py`:
  - 修复 tree-sitter 0.25.x API 兼容性
  - 支持多种语言包命名格式
- ✅ 创建 `tests/test_precompute_file_processing.py` - 6 个测试全部通过
- ✅ 验证文件读取（UTF-8 和 GBK 编码）
- ✅ 验证 AST 解析和符号提取
- ✅ 验证指纹计算和变更检测
- ✅ 验证批量处理

**技术债务**  

- 关系提取和创建需要 SurrealDB 集成，将在后续任务中实现

---

## Stub 端点实现

### BL-C-4 [P2] HNSW 索引管理端点

**目标**  
实现 HNSW（Hierarchical Navigable Small World）向量索引的管理端点，包括统计、优化和重建功能。

**涉及范围**  

- 文件: `wrapper/src/routers/stubs.py`（修改）
- 文件: `wrapper/src/routers/hnsw.py`（新建）
- 文件: `wrapper/src/utils/memory_manager/stubs.py`（修改）
- 新增: `tests/test_hnsw_manager.py`（测试）

**前置依赖**  

- SurrealDB HNSW 索引已创建
- 向量搜索功能可用

**完成标准**  

- [x] `GET /api/v1/hnsw/stats` - 返回索引统计信息
- [x] `POST /api/v1/hnsw/optimize` - 自动优化 HNSW 参数
- [x] `POST /api/v1/hnsw/rebuild` - 重建 HNSW 索引
- [x] 所有端点从 stubs.py 移到独立 router
- [x] 添加完整的错误处理
- [x] 添加单元测试

**验证方式**  

```bash
# 单元测试
uv run pytest tests/test_hnsw_manager.py -v
```

**实际完成**  

- ✅ 创建 `wrapper/src/routers/hnsw.py` - HNSW 独立 router
  - `GET /api/v1/hnsw/stats` - 获取索引统计
  - `POST /api/v1/hnsw/optimize` - 优化参数
  - `POST /api/v1/hnsw/rebuild` - 重建索引
- ✅ 修改 `wrapper/src/routers/stubs.py` - 移除 HNSW 端点
- ✅ 修改 `wrapper/src/utils/memory_manager/stubs.py`:
  - 实现 `optimize_hnsw()` 方法
  - 实现 `rebuild_hnsw_index()` 方法
- ✅ 创建 `tests/test_hnsw_manager.py` - 5 个测试全部通过
- ✅ 添加完整的错误处理（503、500）

---

### BL-C-5 [P2] 缓存管理端点

**目标**  
实现缓存管理端点，支持统计、清理和预热功能。

**涉及范围**  

- 文件: `wrapper/src/routers/stubs.py`（修改）
- 文件: `wrapper/src/routers/cache.py`（新建）
- 文件: `wrapper/src/utils/memory_manager/stubs.py`（修改）
- 新增: `tests/test_cache_manager.py`（测试）

**前置依赖**  

- 缓存系统已集成（如 Redis 或内存缓存）

**完成标准**  

- [x] `GET /api/v1/cache/stats` - 返回缓存统计
- [x] `POST /api/v1/cache/clear` - 清理缓存
- [x] `POST /api/v1/cache/warmup` - 缓存预热
- [x] 所有端点从 stubs.py 移到独立 router
- [x] 添加完整的错误处理
- [x] 添加单元测试

**验证方式**  

```bash
# 单元测试
uv run pytest tests/test_cache_manager.py -v
```

**实际完成**  

- ✅ 创建 `wrapper/src/routers/cache.py` - 缓存管理独立 router
  - `GET /api/v1/cache/stats` - 获取缓存统计
  - `POST /api/v1/cache/clear` - 清除缓存
  - `POST /api/v1/cache/warmup` - 缓存预热
- ✅ 修改 `wrapper/src/routers/stubs.py` - 移除缓存端点
- ✅ 修改 `wrapper/src/utils/memory_manager/stubs.py`:
  - 实现 `clear_embedding_cache()` 方法
  - 实现 `warmup_embedding_cache()` 方法
- ✅ 创建 `tests/test_cache_manager.py` - 5 个测试全部通过
- ✅ 添加完整的错误处理（503、500）

---

### BL-C-6 [P2] 代码分析端点

**目标**  
实现代码分析端点，支持对记忆内容进行代码分析。

**涉及范围**  

- 文件: `wrapper/src/routers/stubs.py`（修改）
- 文件: `wrapper/src/routers/code_analysis.py`（新建）
- 文件: `wrapper/src/utils/memory_manager/code_analysis.py`（依赖）
- 新增: `tests/test_code_analysis_endpoint.py`（测试）

**前置依赖**  

- CodeAnalyzer 服务可用
- tree-sitter 解析器已配置

**完成标准**  

- [x] `POST /api/v1/memories/{id}/analyze/code` - 分析记忆内容中的代码
- [x] 返回代码复杂度、依赖关系、质量评分
- [x] 支持多种语言（Python、JavaScript、TypeScript、Go）
- [x] 所有端点从 stubs.py 移到独立 router
- [x] 添加完整的错误处理
- [x] 添加单元测试

**验证方式**  

```bash
# 单元测试
uv run pytest tests/test_code_analysis_endpoint.py -v
```

**实际完成**  

- ✅ 创建 `wrapper/src/routers/code_analysis.py` - 代码分析独立 router
  - `POST /api/v1/memories/{id}/analyze/code` - 分析记忆代码
- ✅ 修改 `wrapper/src/routers/stubs.py` - 移除代码分析端点
- ✅ 依赖 `wrapper/src/utils/memory_manager/code_analysis.py`:
  - `analyze_memory_code()` 方法已存在
  - 支持多种语言检测
  - 返回代码复杂度、函数、类等信息
- ✅ 创建 `tests/test_code_analysis_endpoint.py` - 4 个测试全部通过
- ✅ 添加完整的错误处理（503、500）

---

### BL-C-7 [P3] 记忆聚类端点

**目标**  
实现记忆聚类端点，使用 Leiden 算法对记忆进行聚类分析。

**涉及范围**  

- 文件: `wrapper/src/routers/stubs.py`（修改）
- 文件: `wrapper/src/services/clustering.py`（新建）

**前置依赖**  

- 向量数据可用
- Leiden 算法库已安装（如 `leidenalg` 或 `igraph`）

**完成标准**  

- [x] `POST /api/v1/memories/cluster/leiden` - 执行 Leiden 聚类
- [x] 支持自定义分辨率参数
- [x] 返回聚类结果（簇 ID、成员、中心点）
- [x] 所有端点从 stubs.py 移到独立 router
- [x] 添加完整的错误处理
- [x] 添加单元测试

**验证方式**  

```bash
# 单元测试
uv run pytest tests/test_clustering.py -v

# API 测试
curl -X POST "http://localhost:18008/api/v1/memories/cluster/leiden?resolution=1.0"
```

**实际完成**  

- ✅ 创建 `wrapper/src/services/clustering.py` - 聚类服务
  - 使用连通分量算法实现类似 Leiden 的聚类效果
  - 支持余弦相似度计算
  - 支持自定义相似度阈值和最大簇数量
- ✅ 创建 `wrapper/src/routers/clustering.py` - 聚类路由
  - `POST /api/v1/memories/cluster/leiden` - 执行聚类
- ✅ 修改 `wrapper/src/routers/stubs.py` - 移除聚类端点
- ✅ 修改 `wrapper/src/utils/memory_manager/stubs.py`:
  - 实现 `cluster_memories_leiden()` 方法
- ✅ 创建 `tests/test_clustering.py` - 6 个测试全部通过
- ✅ 添加完整的错误处理（503、500）

---

### BL-C-8 [P3] 预取功能端点

**目标**  
实现预取功能端点，支持预取相关记忆和热门记忆。

**涉及范围**  

- 文件: `wrapper/src/routers/stubs.py`（修改）
- 文件: `wrapper/src/services/prefetch_service.py`（新建）

**前置依赖**  

- 记忆关系数据可用
- 访问统计数据可用

**完成标准**  

- [x] `POST /api/v1/prefetch/related` - 预取相关记忆（基于关系图）
- [x] `POST /api/v1/prefetch/popular` - 预取热门记忆（基于访问统计）
- [x] 支持限制预取数量
- [x] 所有端点从 stubs.py 移到独立 router
- [x] 添加完整的错误处理
- [x] 添加单元测试

**验证方式**  

```bash
# 单元测试
uv run pytest tests/test_prefetch.py -v

# API 测试
curl -X POST "http://localhost:18008/api/v1/prefetch/related?memory_id=xxx&limit=10"

curl -X POST "http://localhost:18008/api/v1/prefetch/popular?limit=10"
```

**实际完成**  

- ✅ 创建 `wrapper/src/services/prefetch_service.py` - 预取服务
  - 基于关系图遍历实现相关记忆预取
  - 基于最近活跃度实现热门记忆预取
  - 支持深度遍历（1-3层）
- ✅ 创建 `wrapper/src/routers/prefetch.py` - 预取路由
  - `POST /api/v1/prefetch/related` - 预取相关记忆
  - `POST /api/v1/prefetch/popular` - 预取热门记忆
- ✅ 修改 `wrapper/src/routers/stubs.py` - 移除预取端点
- ✅ 修改 `wrapper/src/utils/memory_manager/stubs.py`:
  - 实现 `prefetch_related_memories()` 方法
  - 实现 `prefetch_popular_queries()` 方法
- ✅ 创建 `tests/test_prefetch.py` - 10 个测试全部通过
- ✅ 添加完整的错误处理（503、500）

---

## Atom Architecture 后端实施

### BL-C-9 [P1] Entity 内联 Atom 创建

**目标**

实现 Entity 创建时内联创建 Atom 的能力，支持 `str | AtomInlineCreate` 双格式。

**涉及范围**

- 文件: `wrapper/src/routers/entity.py`（修改）
- 文件: `wrapper/src/models.py`（修改）
- 新增: `AtomInlineCreate` 模型

**完成标准**

- [x] `AtomInlineCreate` 模型定义
- [x] `EntityCreateRequest.atoms` 支持 `Sequence[Union[str, AtomInlineCreate, dict]]`
- [x] `_process_atoms` 自动注入 `tenant_id`
- [x] 事务安全：Atom 创建在事务块内
- [x] `model_dump` dict 类型兼容处理
- [x] `batch_create_entities` atoms 硬编码空数组修复

**验证方式**

```bash
uv run pytest tests/ -v -k "atom"
```

---

### BL-C-10 [P1] 统一搜索 + Atom 层级过滤

**目标**

扩展统一搜索端点，支持 Atom/Entity scope 和层级过滤。

**涉及范围**

- 文件: `wrapper/src/routers/search.py`（修改）
- 文件: `wrapper/src/routers/atom.py`（修改）
- 常量: `VALID_SEARCH_SCOPES` 新增 `"atom"` / `"entity"`

**完成标准**

- [x] `POST /api/v1/search` 支持 `scope=atom/entity`
- [x] Atom 搜索结果含 `content`、`heading_level`、`parent_id`、`order`、`tags`
- [x] `GET /api/v1/atoms` 支持 `max_level` 过滤
- [x] `POST /api/v1/search` 支持 `max_level` 过滤
- [x] `_should_search_atoms/entities` scope 路由修复

**验证方式**

```bash
uv run pytest tests/ -v -k "search"
```

---

### BL-C-11 [P2] 跨 Entity Atom 链接

**目标**

实现跨 Entity 的 Atom 引用链接。

**涉及范围**

- 文件: `wrapper/src/routers/entity.py`（修改）
- 文件: `wrapper/src/routers/atom.py`（修改）

**完成标准**

- [x] `GET /api/v1/entities/{entity_id}/atoms/{atom_id}` 端点
- [x] 支持跨 Entity 查找 Atom
- [x] 错误处理（Atom 不存在、Entity 不存在）

**验证方式**

```bash
uv run pytest tests/ -v -k "entity"
```

---

### BL-C-12 [P1] 上下文预算管理

**目标**

实现基于 token 预算的 Atom 选择策略，支持 BM25 relevance + hierarchy 双策略。

**涉及范围**

- 文件: `wrapper/src/routers/atom.py`（修改）
- 新增: `POST /api/v1/atoms/budget` 端点

**完成标准**

- [x] `POST /api/v1/atoms/budget` 端点实现
- [x] BM25 relevance 策略
- [x] hierarchy 策略（祖先链完整性）
- [x] token 预算控制
- [x] `max_level` 参数支持
- [x] 循环引用防护（`ancestor_ids` 检查）
- [x] 多级 parent 补全（grandparent → parent → child）

**验证方式**

```bash
uv run pytest tests/ -v -k "budget"
```

---

## 统计汇总

| 分类 | 总数 | P1 | P2 | P3 | 工时 |
|------|------|----|----|----|------|
| PrecomputeService 完善 | 3 | 3 | 0 | 0 | 4 天 |
| Stub 端点实现 | 5 | 0 | 3 | 2 | 4 天 |
| Atom Architecture 后端实施 | 4 | 3 | 1 | 0 | 3.5 天 |
| **总计** | **12** | **6** | **4** | **2** | **11.5 天** |

---

## 执行协议

**协议**: AGENT-COLLABORATION-PROTOCOL-v1.0  
**优先级**: P1 > P2 > P3  
**提交规范**: `feat(BL-C-xxx): description`  
**测试要求**: 每个任务必须包含单元测试

---

## 附录

### 技术债务与优化

#### 2026-04-26: SurrealDB RELATE 实现优化

**问题**: SurrealDB Python SDK 的 `RELATE ... SET` 参数绑定存在限制

**解决方案**: 使用 `db.insert_relation()` 替代 `RELATE ... CONTENT`

**实施状态**: ✅ 已完成

**变更文件**:

- `wrapper/src/utils/memory_manager/relations.py`

**优化效果**:

- 代码更简洁（10 行 vs 12 行）
- 更安全（无字符串插值）
- 使用 SDK 原生方法
- 所有 65 个测试通过

**参考文档**:

- `docs/dev/SURREALDB_RELATE_LIMITATION.md`
- `docs/dev/INSERT_RELATION_EVALUATION.md`
- `docs/dev/INSERT_RELATION_COMPARISON.md`

---

_文档版本: v3.3.1_
_最后更新: 2026-04-30_
