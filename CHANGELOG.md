# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.3.0] - 2026-03-12

### Added
- **Polyglot 搜索架构**：Meilisearch 全文搜索 + SurrealDB 向量/图
  - Meilisearch 客户端 (`meili_client.py`) 支持 CJK 中文分词、日期精确匹配
  - 双写同步机制：上传记忆时同步到 SurrealDB 和 Meilisearch
  - 搜索路由：keyword → Meilisearch，vector → SurrealDB，hybrid → RRF 融合
  - 降级回退：Meilisearch 不可用时自动降级到 SurrealDB BM25
- **搜索质量评估器**：重构评估脚本支持 Meilisearch
  - 自动生成测试数据（文档、日期、代码标识符）
  - 评估指标：命中率、误报率、假阳性率、假阴性率
  - 支持混合搜索（RRF）与纯向量搜索对比
  - 详细的评估报告（JSON + Markdown）
- **数据迁移脚本**：`migrate_to_meilisearch.py`
  - 批量同步 SurrealDB 现有记忆到 Meilisearch
  - 幂等设计，可重复运行
  - 支持断点续传和错误重试
- **23 个 Meilisearch 集成测试**：覆盖客户端、双写、搜索路由

### Fixed
- **Meilisearch metadata 丢失**：修复双写时 `meili_doc` 未包含 `metadata` 字段
- **SurrealDB 返回格式不一致**：统一 `format_surreal_results` 返回结构
- **Meilisearch 评分归一化**：修复混合搜索时 Min-Max 归一化问题
- **搜索质量评估器 bug**：修复评估器无法正确计算命中率的问题

### Changed
- **架构优化**：Polyglot Persistence 模式
  - SurrealDB 专注：向量搜索(HNSW) + 图关系(RELATE) + 数据存储 + LIVE SELECT
  - Meilisearch 专注：全文搜索 + 中文分词 + 日期精确匹配
  - 消除 SurrealDB FTS 的所有 workaround（提取引擎、三重降级、双表双写、安全转义层）
- **测试覆盖**：Meilisearch 集成测试 23 个，全部通过
- **性能优化**：HNSW 向量索引（10x 加速），批量 Embedding（10x 加速）

### Removed
- `tests/test_wrapper_minimal.py`：移除有语法错误的临时测试文件
- SurrealDB 全文索引（Meilisearch 接管）
- SurrealDB `memory_analyzer`（不再需要）

### Technical Details
- **Meilisearch 版本**：v1.38.2（charabia v0.9.x）
- **端口配置**：Meilisearch 7701（避免与代理冲突）
- **HNSW 索引**：`DEFINE INDEX memory_embedding ON memory FIELDS embedding HNSW`
- **RRF 融合算法**：`score = 1/(k+rank)`，k=60

### Migration Notes
```bash
# 1. 启动 Meilisearch
nohup meilisearch --http-addr 127.0.0.1:7701 --master-key "YOUR_KEY" --env development &

# 2. 运行数据迁移（幂等，可重复）
export SURREAL_URL=ws://localhost:18002/rpc
export SURREAL_NS=memory_ns
export SURREAL_DB=memory_db
uv run python scripts/migrate_to_meilisearch.py --batch-size 200

# 3. 启动服务
uv run python start_services.py --with-llm

# 4. 验证
uv run pytest tests/test_meili_integration.py -v
```

---

## [2.2.0] - 2026-03-11

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
