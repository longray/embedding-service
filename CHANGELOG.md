# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.9.0] - 2026-05-08

### Added

- **Atom Meilisearch 统一搜索**
  - Phase 1: Meilisearch 索引配置扩展，支持 Atom 字段（`atom_type`, `entity_id`, `heading_level`, `local_id`）
  - Phase 2: Atom CRUD 双写同步，创建/更新/删除时自动同步到 Meilisearch
  - Phase 3: Atom 搜索路由重构，关键词搜索优先使用 Meilisearch（支持 CJK 分词），降级到 SurrealDB
  - 新增 `_search_atoms_by_keyword_meili()` 和 `_format_atom_meili_results()` 方法
  - 新增 `delete_all_documents()` 备选方案（删除并重建索引）解决 Meilisearch 1.13.3 清空问题
  - 新增 `reset_all.py` 脚本，一键清空并重新初始化所有数据

### Changed

- `clear_memories` API 现在同时清空 `atom` 和 `entity` 表
- `_search_atoms_by_keyword()` 优先使用 Meilisearch，失败后降级到 SurrealDB
- RRF 权重从硬编码改为从 MemoryManager 配置读取

### Fixed

- Meilisearch `delete_all_documents` 不工作问题，添加删除并重建索引的备选方案
- `heading_level` 过滤添加 `IS NULL` 处理
- `atom_types` 单引号转义防止 filter 表达式注入
- ID 转换冗余问题（依赖 meili_client 自动转换）

## [2.8.4] - 2026-04-29

### Added

- **v3.3 Atom Architecture 后端完整实施**
  - Entity 内联 Atom 创建：`AtomInlineCreate` 模型，`atoms` 字段支持 `str | AtomInlineCreate` 双格式
  - 统一搜索端点扩展：`POST /api/v1/search` 支持 `scope=atom/entity`，Atom 搜索结果含 `content`、`heading_level`、`parent_id`、`order`、`tags`
  - 跨 Entity Atom 链接：`GET /api/v1/entities/{entity_id}/atoms/{atom_id}`
  - 上下文预算管理：`POST /api/v1/atoms/budget`（BM25 relevance + hierarchy 双策略，token 预算控制，祖先链完整性保证）
  - Atom 层级过滤：`max_level` 参数支持 `GET /api/v1/atoms`、`POST /api/v1/search`、`POST /api/v1/atoms/budget`

### Fixed

- **batch_create_entities atoms 硬编码空数组**：`atoms` 字段从 `[]` 改为正确处理请求中的 atoms
- **scope 路由缺失**：`_should_search_atoms/entities` 新增 `"atom"` / `"entity"` scope 支持
- **model_dump 类型兼容**：`_process_atoms` 新增 `dict` 类型处理（`model_dump` 后 `AtomInlineCreate` 变为 dict）
- **内联 Atom 缺少 tenant_id**：`_process_atoms` 自动注入 `tenant_id` 到内联创建的 Atom
- **事务安全**：`create_entity` / `update_entity` 的 Atom 创建移入事务块，避免孤儿数据
- **祖先链完整性**：`_greedy_select` 支持多级 parent 补全（grandparent → parent → child）
- **循环引用防护**：`_greedy_select` while 循环添加 `ancestor_ids` 检查，防止死循环
- Bandit B608 标记：`atom.py` 新增 3 处 `# nosec B608` 行内标记

### Changed

- `EntityCreateRequest.atoms`: `list[str]` → `Sequence[Union[str, AtomInlineCreate, dict]]`
- `EntityUpdateRequest.atoms`: `list[str] | None` → `Sequence[...] | None`
- `_process_atoms` 新增 `tenant_id` 参数，签名改为异步函数接受 `Sequence` 类型
- `VALID_SEARCH_SCOPES` 新增 `"atom"` 和 `"entity"`

## [2.8.3] - 2026-04-22

### Added

- **WebSocket Session 恢复** (TC-WS-004)
  - 断线重连时保持 session_id 一致性
  - 支持通过 `session_id` 查询参数恢复现有 session
  - 自动清理过期 session（7 天 TTL）

- **WebSocket 变更推送** (TC-WS-003)
  - 使用 `db.live()` + `subscribe_live()` 监听 memory 表变更
  - 自动推送 CREATE/UPDATE/DELETE 通知到所有连接的客户端
  - 消息格式：`{"type": "memory_change", "action": "UPDATE", "data": {...}, "timestamp": ...}`
  - 处理 RecordID 和 datetime 对象序列化

### Fixed

- **WebSocket 变更推送空数据问题**
  - 修复 UUID 不匹配导致通知未路由到队列的问题
  - 改用 `db.live("memory")` 确保正确注册到 `live_queues`
  - 添加客户端 `tenant_id` 过滤

- **source_id 去重更新缺失** (TC-LOOKUP-001)
  - 去重时不再更新 `source_id` 和 `local_id`，保持引用完整性
  - 新增 `dedup_info` 返回字段，包含已有记忆的 `memory_id`、`source_id`、`local_id`
  - 插件端可根据 `dedup_info` 自行决策：直接使用已有记录或删除重建
  - 旧 `source_id` 保持有效，所有引用不会断裂

- **图谱关系悬空问题** (TC-GRAPH-001)
  - 创建关系前验证源节点和目标节点是否存在
  - 使用 `type::record()` 确保 SurrealDB 3.0 RecordID 正确比较
  - 返回清晰的 400 错误和同步指引
  - 防止"悬空关系"导致图谱遍历失败

## [2.8.2] - 2026-04-22

### Fixed

- **Layer 1 数据层测试修复** (插件端反馈)
  - 修复 Entity 列表查询缺失 `tenant_id` 字段导致的 Pydantic 验证错误
  - 修复 Reference 创建失败的 `type` 字段验证问题
  - 更新 SurrealDB Schema 支持 atom-atom 关系（`record<atom|entity>`）

### Added

- **ReferenceType 枚举**：统一管理 14 种关系类型
  - 代码调用：`calls`, `imports`, `extends`, `implements`, `depends_on`
  - 知识关联：`wiki_link`, `part_of`, `related`, `follow_up`, `elaboration`, `contradiction`, `reference`, `derived_from`, `similar_to`
  - 避免字符串拼写错误导致的查询失败

### Changed

- `reference.py` 添加类型验证，使用 `ReferenceType.all_values()` 检查
- SurrealDB `reference.type` 字段更新为 14 种枚举值

## [2.8.1] - 2026-04-18

### Fixed

- **WebSocket 实时推送修复** (Phase 10)
  - 修复 datetime 时区问题：`datetime.utcnow()` 与 `datetime.now(timezone.utc)` 混用导致的 `can't subtract offset-naive and offset-aware datetimes` 错误
  - 修复数据库连接问题：`SurrealDBManager.get_instance()` 只创建实例不自动连接，添加 `reconnect()` 调用
  - 修复 SurrealDB 3.0 UUID 返回格式：`LIVE SELECT` 直接返回 UUID 对象而非列表，使用 `isinstance()` 灵活处理
  - 修复 `subscribe_live` coroutine 未 await：先 `await` 获取 subscription，再 `async for` 遍历
  - 修复 `connected` 消息缺失：在 `accept()` 后发送 `connected` 消息，包含有效 `session_id`
  - 修复 session 初始化顺序：`create_session()` 在 `accept()` 之前调用，确保 `session_id` 可用
  - 修复 SurrealDB 参数化表名问题：`$table` 参数不能用作表标识符，改用 f-string

### Changed

- **datetime 统一处理**：所有模块统一使用 naive datetime，解析时兼容 `Z`/`+00:00` 后缀
  - 涉及文件：`state_recovery.py`, `message_queue.py`, `code_analyzer.py`, `audit.py`, `meili_sync.py`

### Technical

- **WebSocket 连接验证**：15/15 单元测试通过，连接测试验证通过
- **插件集成就绪**：WebSocket 端点 `ws://localhost:18008/ws/memories/live` 已就绪，等待插件端集成测试

## [2.8.0] - 2026-04-15

### Added

- **BACKLOG v3.3: PrecomputeService 完善 + Stub 端点实现**
  - **PrecomputeService 核心功能**
    - 初始化资源实现（tree-sitter、配置加载、数据库连接）
    - 资源清理实现（并发控制、队列清理、性能监控停止）
    - 文件处理逻辑（AST 解析、指纹计算、符号提取）
  - **HNSW 索引管理端点**
    - `GET /api/v1/hnsw/stats` - HNSW 索引统计
    - `POST /api/v1/hnsw/optimize` - 自动优化 HNSW 参数
    - `POST /api/v1/hnsw/rebuild` - 重建 HNSW 索引
  - **缓存管理端点**
    - `GET /api/v1/cache/stats` - 缓存统计
    - `POST /api/v1/cache/clear` - 清空缓存
    - `POST /api/v1/cache/warmup` - 预热缓存
  - **代码分析端点**
    - `POST /api/v1/memories/{id}/analyze/code` - 分析记忆内容中的代码
    - 支持 Python、JavaScript、TypeScript、Go
    - 返回代码复杂度、函数、类等信息
  - **记忆聚类端点**
    - `POST /api/v1/memories/cluster/leiden` - Leiden 算法聚类
    - 基于向量相似度的连通分量聚类
    - 支持自定义相似度阈值和最大簇数量
  - **预取功能端点**
    - `POST /api/v1/prefetch/related` - 预取相关记忆（基于关系图）
    - `POST /api/v1/prefetch/popular` - 预取热门记忆（基于访问统计）
    - 支持深度遍历（1-3层）

### Changed

- **测试套件增强**: 新增 42 个单元测试，总计 1000+ 测试用例
- **路由模块化**: 新增独立 routers（hnsw.py, cache.py, code_analysis.py, clustering.py, prefetch.py）
- **tree-sitter 兼容性**: 修复 0.25.x API 兼容性（Parser 初始化方式）

### Technical

- **ClusteringService**: 基于 NumPy 的连通分量聚类算法（无需 leidenalg 依赖）
- **PrefetchService**: 关系图遍历 + 最近活跃度排序
- **FingerprintManager**: SHA256 指纹计算和变更检测
- **ConcurrencyControl**: Semaphore-based 并发控制

## [2.7.2] - 2026-04-09

### Added

- **BL-CA-34: Memory Lookup API**
  - 新增 `GET /api/v1/memories/lookup` 端点
  - 支持 source_id、content_hash、file_path 三种查询方式
  - 用于缓存重建和多设备同步场景
  - 添加 SurrealDB 索引优化查询性能

## [2.7.1] - 2026-04-08

### Added

- **BL-CA-OPT-01~06: SQL 查询优化和安全性修复**
  - RELATE 语句 SQL 注入防护（参数化 SET 子句）
  - RecordID 格式统一（使用 `type::record()` 函数）
  - 嵌套字段查询优化（添加复合索引 `memory_tenant_type_project`）
  - 批量插入分批处理（50条/批，避免超时）
  - Meilisearch 同步分批（50条/批，避免超时）
  - SQL 查询规范文档（`docs/dev/SURREALDB_SQL_BEST_PRACTICES.md`）

- **BL-CA-OPT-08: embedding 字段优化**
  - `GET /api/v1/memories/{id}` 添加 `include_embedding` 查询参数
  - 默认不返回 embedding 向量，减少响应体积 97.9%
  - 向后兼容，需要时显式指定 `include_embedding=true`

### Changed

- **批次大小统一**: SurrealDB 插入和 Meilisearch 同步统一为 50条/批
- **查询性能优化**: 项目地图查询使用复合索引，响应时间 < 500ms

### Fixed

- **代码分析数据上传**: 修复上传成功但数据未写入的问题
- **hash 去重问题**: 代码数据（`type: "code"`）跳过 hash 去重检查
- **项目地图边数据**: 修复 `module_dependencies` 为空的问题
- **字段名不一致**: 统一使用 `abstract` 和 `overview`（非 `content_abstract`）
- **项目地图查询返回空**: 移除 `metadata.code_analysis IS NOT NONE` 查询条件，支持无 code_analysis 的数据

---

## [2.7.0] - 2026-04-04

### Added

- **BL-29: get_fingerprints 实现**
  - 查询 SurrealDB `SELECT source_id, content_hash, updated_at FROM memory`
  - 字段映射: `content_hash` → `hash`, `updated_at` → `mtime`
  - 按 `tenant_id` 隔离，过滤无 `source_id` 记录

- **BL-30: sync_preview 实现**
  - 比对本地与服务端指纹，三分类输出: `to_upload` / `to_delete` / `conflicts`
  - 检测到冲突时自动写入 `conflict` 表（status=pending）

- **BL-31: sync_full 实现**
  - 透传 `upload_memories()`，复用已有的 embedding + 去重 + Meilisearch 双写逻辑

- **BL-32: resolve_conflict 实现**
  - 三种冲突解决策略: `use_local` / `use_remote` / `keep_both`
  - `use_local`: UPDATE memory 内容 + Meilisearch 同步
  - `use_remote`: 仅标记 conflict 为 resolved
  - `keep_both`: CREATE 新 memory（source_id 加 `-local` 后缀）+ Meilisearch 同步
  - 新增辅助方法: `_record_conflict`, `get_conflicts`, `get_conflict_detail`

- **测试架构优化 (BL-T1~T11)**
  - pytest 分层标记: unit (50P) / integration (128P) / e2e
  - pre-commit 仅跑 unit 测试 (9.52s)
  - LLM 服务条件跳过: 未启动时自动 skip
  - 文件合并: 27 → 21 文件
  - wrapper 接口适配: 删除 8 个已移除功能测试

- **代码分析完善 (BL-CA-05/06/09/10)**
  - code_filter 新增 `max_complexity` 支持
  - 代码分析集成测试补充
  - API 文档更新

### Changed

- **sync.py**: 从 206 行扩展到 347 行（+141 行），4 个 stub → 实际实现
- **test_phase_b_sync.py**: 移除 14 个 skip 装饰器，32/32 测试通过

### Fixed

- **场景 7 修复** (2026-04-06)
  - 修复 HNSW stats 返回结构双重嵌套问题
  - 修复 cache stats 返回结构双重嵌套问题
  - 修复 SurrealDB 健康检查逻辑（使用 `state.memory_manager` 替代单例模式）
  - 修复 HNSW 查询语法（添加 `ON memory`）
  - diagnose.py 现在显示：✅ SurrealDB 连接正常，✅ 缓存系统已启用

---

## [2.6.0] - 2026-04-02

### Changed

- **BL-35: memory_manager.py 重构**
  - 将 1715 行的上帝文件拆分为 Mixin 模式 10 子模块
  - `manager.py` (268行): 主类 + 生命周期 + 基础设施
  - `crud.py` (415行): 上传/更新/embedding
  - `search.py` (~310行): 向量/关键词/混合搜索 + RRF
  - `sync.py` (~110行): 指纹/预览/全量同步/冲突解决
  - `relations.py` (~265行): 图关系 + 遍历
  - `dedup.py`: 去重决策逻辑
  - `meili_sync.py`: Meilisearch 双写 + ID 转换
  - `code_analysis.py`: analyze_memory_code
  - `stubs.py`: 9 个 NotImplementedError 占位方法
  - `__init__.py`: 重导出 `MemoryManager`，保持向后兼容
  - 导入路径 `from wrapper.src.utils.memory_manager import MemoryManager` 不变

- **BL-36: main.py 路由拆分**
  - 将 1063 行拆分为 12 个模块
  - `models.py` (150行): 17 个 Pydantic 模型集中管理
  - `state.py` (18行): 共享单例，避免循环导入
  - `routers/`: 8 个路由模块（health/embeddings/memories/search/relations/sync/websocket/stubs）
  - `main.py` (303行): app + lifespan + SurrealDBManager + 路由注册
  - 导入路径 `from wrapper.src.main import app` 不变

- **FastAPI 版本号**: `version="2.4.1"` → `version="2.6.0"`

### Fixed

- **BL-33: pyproject.toml 过时配置**
  - 移除重复的 `RUF001/RUF002/RUF003` 忽略规则
  - `testpaths` 从 `wrapper-service/tests` 修正为 `tests`
  - coverage source 从 `wrapper-service/src` 修正为 `wrapper/src`

- **BL-34: meilisearch_code/ Pyright 类型错误**
  - `IndexStats.get("numberOfDocuments")` → `stats.model_dump().get("number_of_documents")`（IndexStats 是 Pydantic model）
  - 空的 `except:` 添加 `pass`

- **BL-38: 移除硬编码 API Key**
  - `meili_client.py` docstring 中 `masterKey` → `your_api_key`

- **BL-39: scripts/ 裸 except 清理**
  - 5 处 `except:` → `except Exception:`

- **测试修复**: 4 个 pre-existing 测试失败
  - `test_memory_with_empty_content`: content `min_length=1` → 期望 422
  - `test_memory_with_special_characters`: 添加 UUID 避免去重跳过
  - `test_source_id_deduplication`: 当前不强制 UNIQUE → 断言 `total==1`
  - `test_memory_without_content`: BL-36 重构时已移除（语义合并到其他测试）
  - 测试结果: 61/61 passed（0 failed）

### Added

- **BL-28: analyze_memory_code 实现**
  - 上传 `type="code"` 记忆时自动调用 `CodeAnalyzer.analyze()`
  - 分析结果写入 `metadata.code_analysis`
  - 分析失败不影响上传（记录 warning）

- **BL-37: utils 单元测试**
  - `test_cache.py` (11 tests): ThreadSafeLRUCache — 命中/未命中/TTL/淘汰/LRU 顺序/统计
  - `test_http_pool.py` (5 tests): HTTPClientPool — 创建/复用/关闭/请求
  - `test_auth.py` (6 tests): WebSocket 认证 — 无 token/正确/错误/None
  - `test_exceptions.py` (13 tests): 异常层级 — 基类/子类/状态码/消息/继承关系

- **BL-D1: 文档归档**
  - 归档 14 个过时设计文档到 `archive/docs/`
  - 归档 13 个 JSON 评估报告到 `archive/reports/`
  - 归档 10 个 benchmark JSON 到 `archive/reports/benchmarks/`
  - 清理根目录临时文件（test_*.json, *.log, *.heapsnapshot）

---

## [2.5.0] - 2026-03-30

### Changed

- **性能基线建立**：新增 `scripts/benchmark.py` 性能基准测试工具

### Added

- **Docker 部署支持**：新增 `Dockerfile.embedding`, `Dockerfile.llm`, `wrapper/Dockerfile`, `docker-compose.dev.yml`
- **启动脚本**：`docker-start.bat`, `docker-stop.bat`
- **Meilisearch 代码搜索优化**：104 词代码术语字典、nonSeparatorTokens 配置、双字段策略

### Technical Details

- **环境**: NVIDIA GTX 1060 6GB, Qwen3-Embedding-0.6B, SurrealDB 3.0 + Meilisearch 1.4

---

## [2.4.1] - 2026-03-28

### Fixed

- **B-005: sync_preview conflict 检测修复**

  修复 `get_fingerprints` 返回空导致 `sync_preview` 无法检测冲突的问题。

  - **B-005-B: SurrealDB 3.0 SDK 结果解析逻辑错误**
    - 问题：`get_fingerprints` 期望 `result[0]` 是 `{"result": [...]}` 格式，但 SurrealDB 3.0 SDK 直接返回 `[record1, record2, ...]`
    - 修复：复用已有的 `_extract_records()` 方法
    - 文件：`wrapper/src/utils/memory_manager.py` 第 1140-1150 行

  - **B-005-C: `get_conflict_detail` 参数化表名语法错误**
    - 问题：SurrealQL 不支持 `FROM $param` 参数化表名
    - 问题：SurrealDB RecordID 类型不能直接与字符串比较
    - 修复：用 `WHERE type::string(id) = $conflict_id` 替代 `FROM $conflict_id`
    - 文件：`wrapper/src/utils/memory_manager.py` 第 1367-1372 行

- **代码质量修复**
  - 修复 `SCHEMA_TARGET_VERSION`: `2.3.0` → `2.4.1`
  - 修复 `app = FastAPI()` 缩进错误（在 lifespan 函数内导致路由注册失败）
  - 删除重复的 API 端点定义（`analyze_memory_code`, `cluster_memories_leiden` 各两份）
  - 添加 `tree_sitter` 导入的类型忽略标记 `# type: ignore`

### Verification

- E2E 测试通过：上传 memory → 获取 fingerprints → 检测 conflict → 解决 conflict
- Pyright 类型检查：34 errors → 0 errors
- 同步测试：32/32 passed

---

## [2.4.0] - 2026-03-28

### Changed
- **sync_incremental → sync_preview 重命名**：API 路径 `/api/v1/sync/incremental` 改为 `/api/v1/sync/preview`，更准确反映"预览差异、不执行上传"的语义
  - `/api/v1/sync/incremental` 保留为向后兼容别名
  - Pydantic schema: `SyncIncrementalRequest/Response` → `SyncPreviewRequest/Response`
  - 方法: `sync_incremental()` → `sync_preview()`
- **conflict resolution 大小写兼容**：`USE_LOCAL`、`Use_Local`、`use_local` 均可正常工作

### Added
- **full_sync 返回 skipped 列表**：全量同步时，被去重跳过的条目返回详细信息
  - `SyncFullResponse` 新增 `skipped` 数组和 `updated` 字段
  - `skipped` 每项含 `local_id`、`existing_id`、`reason`（hash/semantic）、`similarity`
  - `errors` 仅保留真正的异常，去重信息不再混入 errors

### Fixed
- **test_sync_preview_conflicts**：修复测试 mock 缺少 `create` 方法的已有 bug

### Technical Details
- **修改文件**：`wrapper/src/main.py`、`wrapper/src/utils/memory_manager.py`、`tests/test_phase_b_sync.py`
- **测试覆盖**：TestSyncPreview 4/4、TestSyncFull 3/3、路由/模型/兼容性测试全部通过

---

## [2.3.1] - 2026-03-16

### Fixed
- **语义去重功能修复**：修复向量相似度搜索无法找到相似记忆的问题
  - 问题：使用 `vector::distance::knn()` 的 KNN 查询无法返回相似记忆
  - 解决：改用 `vector::similarity::cosine()` 直接计算余弦相似度
  - 影响：语义去重现在能正确识别和拒绝相似度 >= 0.95 的重复记忆
  - 测试：新增 5 个 pytest 测试验证去重功能（高/中/低相似度、哈希去重、批量去重）

### Added
- **语义去重测试套件**：`tests/test_semantic_deduplication.py`
  - 测试高相似度去重（>= 0.95）
  - 测试中等相似度接受（< 0.95）
  - 测试低相似度接受（完全不同主题）
  - 测试内容哈希去重（完全相同内容）
  - 测试批量上传去重

### Technical Details
- **修改文件**：`wrapper/src/utils/memory_manager.py`
- **查询优化**：使用 `vector::similarity::cosine(embedding, $query_embedding) >= $threshold` 在数据库层面过滤
- **性能**：直接返回相似度分数，无需距离到相似度的转换
- **测试覆盖**：5/5 测试通过（8.51秒）

## [2.3.1] - 2026-03-25

### Added
- **调试清空 API**：新增 `DELETE /api/v1/memories/clear` 端点
  - 安全机制：先清空 Meilisearch（验证 `WRAPPER_MEILI_API_KEY`），再清空 SurrealDB
  - 如果 API key 错误 → Meilisearch 清空失败，SurrealDB 不被清空（数据保护）
  - 使用方法：`curl -X DELETE http://localhost:17999/api/v1/memories/clear -H "WRAPPER_MEILI_API_KEY: your_api_key"`
  - 响应：成功返回 `{"success": true, "message": "所有记忆数据已清空"}`
  - 错误响应：401（缺少 key）、403（key 错误）、500（清空失败）
- **清空脚本**：`scripts/clear_all_data.py` 用于清空后端所有数据（SurrealDB + Meilisearch）

### Changed
- **架构优化**：Polyglot Persistence 模式
  - SurrealDB 专注：向量搜索(HNSW) + 图关系(RELATE) + 数据存储 + LIVE SELECT
  - Meilisearch 专注：全文搜索 + 中文分词 + 日期精确匹配
  - 消除 SurrealDB FTS 的所有 workaround（提取引擎、三重降级、双表双写、安全转义层）
  - **测试覆盖**：Meilisearch 集成测试 23 个，全部通过
  - **性能优化**：HNSW 向量索引（10x 加速），批量 Embedding（10x 加速）

### Technical Details
- **修改文件**：`wrapper/src/main.py`
- **API 端点**：`DELETE /api/v1/memories/clear`
- **认证方式**：`WRAPPER_MEILI_API_KEY` header（从 `config.meilisearch.api_key` 获取）
- **执行顺序**：
  1. 验证 API key（必须提供且匹配配置）
  2. 清空 Meilisearch（使用 `client.delete_document("*")` 或 `delete_documents_by_filter("")`）
  3. 清空 SurrealDB（删除 `memories`、`memory_relation`、`conflict` 表）

---

## [2.3.0] - 2026-03-12

### Added
- **WebSocket 实时推送**：LIVE SELECT 记忆变更通知
- **安全加固**：SurrealDB 运行时用户权限分离
- **OpenTelemetry 分布式追踪**：全链路 span 覆盖

### Changed
- SurrealDB 端口 8000 → 18002（避免与 LLM 服务冲突）

---

## [2.1.0] - 2026-03-10

### Added
- **批量 Embedding 性能优化**：10x 加速
- **Prometheus 监控指标**
- **健康检查级联验证**

---

## [2.0.0] - 2026-03-09

### Added
- **完整包装服务**（端口 3001）：熔断器、缓存、连接池
- **API 认证授权**：API Key 认证和权限控制
- **完整测试套件**：150+ 测试用例
- **CI/CD**：GitHub Actions 自动测试

---

## [1.0.0] - 2026-03-08

### Initial Release
- Embedding 服务（端口 18000）：Qwen3-Embedding-0.6B
- LLM 服务（端口 18001）：MiniCPM4-0.5B
- 最小化包装服务（端口 17999）：基础 API 代理
- SurrealDB 向量存储
- SurrealDB 全文搜索（BM25）
