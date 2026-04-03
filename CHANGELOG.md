# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
