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
| BL-C-2 | PrecomputeService — 资源清理实现 | P1 | 0.5 天 | ⏳ | [详情](#bl-c-2-p1-precomputeservice--资源清理实现) |
| BL-C-3 | PrecomputeService — 文件处理逻辑实现 | P1 | 2 天 | ⏳ | [详情](#bl-c-3-p1-precomputeservice--文件处理逻辑实现) |
| **Stub 端点实现** |
| BL-C-4 | HNSW 索引管理端点 | P2 | 1 天 | ⏳ | [详情](#bl-c-4-p2-hnsw-索引管理端点) |
| BL-C-5 | 缓存管理端点 | P2 | 1 天 | ⏳ | [详情](#bl-c-5-p2-缓存管理端点) |
| BL-C-6 | 代码分析端点 | P2 | 1 天 | ⏳ | [详情](#bl-c-6-p2-代码分析端点) |
| BL-C-7 | 记忆聚类端点 | P3 | 0.5 天 | ⏳ | [详情](#bl-c-7-p3-记忆聚类端点) |
| BL-C-8 | 预取功能端点 | P3 | 0.5 天 | ⏳ | [详情](#bl-c-8-p3-预取功能端点) |

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

**前置依赖**  
- BL-C-1 初始化资源实现完成

**完成标准**  
- [ ] 关闭 tree-sitter 解析器
- [ ] 关闭数据库连接
- [ ] 停止性能监控器
- [ ] 释放并发控制资源
- [ ] 清理临时文件和缓存
- [ ] 删除 `stop()` 方法中的 TODO 注释

**验证方式**  
```bash
# 单元测试
uv run pytest tests/test_precompute_service.py::TestPrecomputeService::test_cleanup -v

# 资源泄漏测试
uv run pytest tests/test_precompute_service.py::TestPrecomputeService::test_no_resource_leak -v
```

---

### BL-C-3 [P1] PrecomputeService — 文件处理逻辑实现

**目标**  
实现完整的文件处理逻辑，包括 AST 解析、符号提取和批量创建 atoms。

**涉及范围**  
- 文件: `wrapper/src/services/precompute.py`（修改）
- 文件: `wrapper/src/services/code_parser.py`（依赖）
- 文件: `wrapper/src/services/fingerprint.py`（依赖）

**前置依赖**  
- BL-C-1 初始化资源实现完成
- tree-sitter Query API 可用
- FingerprintManager 已实现

**完成标准**  
- [ ] 文件读取和编码检测
- [ ] AST 解析（使用 tree-sitter）
- [ ] 符号提取（函数、类、接口等）
- [ ] 指纹计算和变更检测
- [ ] Atoms 批量创建
- [ ] 关系提取和创建
- [ ] 性能指标收集
- [ ] 删除 `process_file()` 方法中的 TODO 注释

**验证方式**  
```bash
# 单元测试
uv run pytest tests/test_precompute_service.py -v

# 集成测试
uv run pytest tests/integration/test_precompute_e2e.py -v

# 性能测试
uv run python tests/performance/test_precompute_performance.py
```

---

## Stub 端点实现

### BL-C-4 [P2] HNSW 索引管理端点

**目标**  
实现 HNSW（Hierarchical Navigable Small World）向量索引的管理端点，包括统计、优化和重建功能。

**涉及范围**  
- 文件: `wrapper/src/routers/stubs.py`（修改）
- 文件: `wrapper/src/services/hnsw_manager.py`（新建）
- 文件: `wrapper/src/utils/memory_manager.py`（修改）

**前置依赖**  
- SurrealDB HNSW 索引已创建
- 向量搜索功能可用

**完成标准**  
- [ ] `GET /api/v1/hnsw/stats` - 返回索引统计信息（向量数量、维度、索引大小）
- [ ] `POST /api/v1/hnsw/optimize` - 自动优化 HNSW 参数（efConstruction、M）
- [ ] `POST /api/v1/hnsw/rebuild` - 重建 HNSW 索引（支持强制重建）
- [ ] 所有端点从 stubs.py 移到独立 router
- [ ] 添加完整的错误处理
- [ ] 添加单元测试

**验证方式**  
```bash
# 单元测试
uv run pytest tests/test_hnsw_manager.py -v

# API 测试
curl http://localhost:18008/api/v1/hnsw/stats?tenant_id=default

curl -X POST http://localhost:18008/api/v1/hnsw/optimize?tenant_id=default

curl -X POST "http://localhost:18008/api/v1/hnsw/rebuild?tenant_id=default&force=true"
```

---

### BL-C-5 [P2] 缓存管理端点

**目标**  
实现缓存管理端点，支持统计、清理和预热功能。

**涉及范围**  
- 文件: `wrapper/src/routers/stubs.py`（修改）
- 文件: `wrapper/src/utils/cache_manager.py`（新建或完善）

**前置依赖**  
- 缓存系统已集成（如 Redis 或内存缓存）

**完成标准**  
- [ ] `GET /api/v1/cache/stats` - 返回缓存统计（命中率、大小、键数量）
- [ ] `POST /api/v1/cache/clear` - 清理缓存（支持按 pattern 清理）
- [ ] `POST /api/v1/cache/warmup` - 缓存预热（加载热门数据）
- [ ] 所有端点从 stubs.py 移到独立 router
- [ ] 添加完整的错误处理
- [ ] 添加单元测试

**验证方式**  
```bash
# 单元测试
uv run pytest tests/test_cache_manager.py -v

# API 测试
curl http://localhost:18008/api/v1/cache/stats

curl -X POST http://localhost:18008/api/v1/cache/clear

curl -X POST http://localhost:18008/api/v1/cache/warmup
```

---

### BL-C-6 [P2] 代码分析端点

**目标**  
实现代码分析端点，支持对记忆内容进行代码分析。

**涉及范围**  
- 文件: `wrapper/src/routers/stubs.py`（修改）
- 文件: `wrapper/src/services/code_analyzer.py`（依赖）

**前置依赖**  
- CodeAnalyzer 服务可用
- tree-sitter 解析器已配置

**完成标准**  
- [ ] `POST /api/v1/memories/{id}/analyze/code` - 分析记忆内容中的代码
- [ ] 返回代码复杂度、依赖关系、质量评分
- [ ] 支持多种语言（Python、JavaScript、TypeScript、Go）
- [ ] 所有端点从 stubs.py 移到独立 router
- [ ] 添加完整的错误处理
- [ ] 添加单元测试

**验证方式**  
```bash
# 单元测试
uv run pytest tests/test_code_analysis_endpoint.py -v

# API 测试
curl -X POST http://localhost:18008/api/v1/memories/memory:xxx/analyze/code
```

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
- [ ] `POST /api/v1/memories/cluster/leiden` - 执行 Leiden 聚类
- [ ] 支持自定义分辨率参数
- [ ] 返回聚类结果（簇 ID、成员、中心点）
- [ ] 所有端点从 stubs.py 移到独立 router
- [ ] 添加完整的错误处理
- [ ] 添加单元测试

**验证方式**  
```bash
# 单元测试
uv run pytest tests/test_clustering.py -v

# API 测试
curl -X POST "http://localhost:18008/api/v1/memories/cluster/leiden?resolution=1.0"
```

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
- [ ] `GET /api/v1/prefetch/related` - 预取相关记忆（基于关系图）
- [ ] `GET /api/v1/prefetch/popular` - 预取热门记忆（基于访问统计）
- [ ] 支持限制预取数量
- [ ] 所有端点从 stubs.py 移到独立 router
- [ ] 添加完整的错误处理
- [ ] 添加单元测试

**验证方式**  
```bash
# 单元测试
uv run pytest tests/test_prefetch_service.py -v

# API 测试
curl "http://localhost:18008/api/v1/prefetch/related?memory_id=xxx&limit=10"

curl "http://localhost:18008/api/v1/prefetch/popular?limit=10"
```

---

## 统计汇总

| 分类 | 总数 | P1 | P2 | P3 | 工时 |
|------|------|----|----|----|------|
| PrecomputeService 完善 | 3 | 3 | 0 | 0 | 4 天 |
| Stub 端点实现 | 5 | 0 | 3 | 2 | 4 天 |
| **总计** | **8** | **3** | **3** | **2** | **8 天** |

---

## 执行协议

**协议**: AGENT-COLLABORATION-PROTOCOL-v1.0  
**优先级**: P1 > P2 > P3  
**提交规范**: `feat(BL-C-xxx): description`  
**测试要求**: 每个任务必须包含单元测试

---

_文档版本: v3.3.0_  
_最后更新: 2026-04-15_
