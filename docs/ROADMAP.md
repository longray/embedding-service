# Embedding Service 项目路线图

> **本文档定位**: 产品路线图，描述版本规划和功能演进方向  
> **详细任务**: 见 `BACKLOG.md`  
> **执行状态**: 见 `BACKLOG.md` 中的场景和任务状态

---

## 文档分工说明

| 文档 | 用途 | 更新频率 |
|------|------|----------|
| **ROADMAP.md**（本文档） | 产品路线图，版本规划，功能演进方向 | 每版本发布时 |
| **BACKLOG.md** | 详细任务清单，执行状态，优先级 | 每周更新 |
| **CHANGELOG.md** | 版本变更记录，已完成功能 | 每版本发布时 |

---

## 版本历史

### v2.8.0 (当前) - PrecomputeService + Stub 端点 ✅

**发布日期**: 2026-04-15

**核心功能**:

- Embedding 服务（Qwen3-Embedding-0.6B）
- LLM 服务（MiniCPM4-0.5B）
- 最小化包装服务（端口 18008）
  - LRU 缓存
  - HTTP 连接池
  - SurrealDB 长期连接
  - **PrecomputeService** - 代码预计算服务（AST 解析、指纹、符号提取）
  - **HNSW 管理端点** - 索引统计、优化、重建
  - **缓存管理端点** - 统计、清空、预热
  - **代码分析端点** - 记忆内容代码分析
  - **记忆聚类端点** - Leiden 算法聚类
  - **预取端点** - 相关记忆、热门记忆预取
- 包装层服务（端口 3001，完整功能）
- API 认证授权（API Key + 权限控制）
- 记忆管理系统（SurrealDB + 向量搜索）
- **Polyglot 搜索架构**（Meilisearch 全文搜索 + SurrealDB 向量/图）
- **同步冲突解决**（多设备/多用户/离线编辑场景）
- **代码分析**（调用关系、引用查询、代码地图、代码统计）
- CI/CD（GitHub Actions）
- **完整测试套件（190+ 测试用例）**

---

### v2.7.2 - Memory Lookup API ✅

**发布日期**: 2026-04-09

**核心功能**:

- Embedding 服务（Qwen3-Embedding-0.6B）
- LLM 服务（MiniCPM4-0.5B）
- 最小化包装服务（端口 17999）
  - LRU 缓存
  - HTTP 连接池
  - SurrealDB 长期连接
  - **Memory Lookup API** - 支持 source_id/hash/file_path 查询
- 包装层服务（端口 3001，完整功能）
- API 认证授权（API Key + 权限控制）
- 记忆管理系统（SurrealDB + 向量搜索）
- **Polyglot 搜索架构**（Meilisearch 全文搜索 + SurrealDB 向量/图）
- **同步冲突解决**（多设备/多用户/离线编辑场景）
- **代码分析**（调用关系、引用查询、代码地图、代码统计）
- CI/CD（GitHub Actions）
- **完整测试套件（150+ 测试用例）**

---

### v2.7.1 - SQL 优化与安全性修复 ✅

**发布日期**: 2026-04-08

**核心功能**:

- SQL 查询优化（RecordID 统一、复合索引、分批处理）
- embedding 字段优化（默认不返回，减少 97.9% 响应体积）
- 代码分析数据上传修复
- API 健壮性增强（字段自动填充）

---

### v2.7.0 - 多设备同步 ✅

**发布日期**: 2026-04-04

**核心功能**:

- 指纹查询（`GET /api/v1/sync/fingerprints`）
- 同步预览（`POST /api/v1/sync/preview`）
- 全量同步（`POST /api/v1/sync/full`）
- 冲突解决（`POST /api/v1/sync/conflicts/{id}/resolve`）
- 测试架构优化（分层标记、条件跳过）

---

### v2.3.1 - 同步冲突解决 ✅

**发布日期**: 2026-03-23

**核心功能**:

- Embedding 服务（Qwen3-Embedding-0.6B）
- LLM 服务（MiniCPM4-0.5B）
- 最小化包装服务（端口 17999，无熔断器）
  - LRU 缓存
  - HTTP 连接池
  - SurrealDB 长期连接
  - 三个核心 API：`/v1/embeddings`、`/api/v1/memories`、`/api/v1/memories/search`
- 包装层服务（端口 3001，完整功能）
- API 认证授权（API Key + 权限控制）
- 记忆管理系统（SurrealDB + 向量搜索）
- **Polyglot 搜索架构**（Meilisearch 全文搜索 + SurrealDB 向量/图）
- **同步冲突解决**（多设备/多用户/离线编辑场景）
- CI/CD（GitHub Actions）
- **完整测试套件（150+ 测试用例）**

---

### v2.3.0 - Polyglot 搜索架构 ✅

**发布日期**: 2026-03-12

**核心功能**:

- Embedding 服务（Qwen3-Embedding-0.6B）
- LLM 服务（MiniCPM4-0.5B）
- 最小化包装服务（端口 17999，无熔断器）
  - LRU 缓存
  - HTTP 连接池
  - SurrealDB 长期连接
  - 三个核心 API：`/v1/embeddings`、`/api/v1/memories`、`/api/v1/memories/search`
- 包装层服务（端口 3001，完整功能）
- API 认证授权（API Key + 权限控制）
- 记忆管理系统（SurrealDB + 向量搜索）
- **Polyglot 搜索架构**（Meilisearch 全文搜索 + SurrealDB 向量/图）
- CI/CD（GitHub Actions）
- **完整测试套件（150+ 测试用例）**

---

## P3 优化阶段

### P3-1: Docker Compose 配置 ✅ 已完成

**目标**: 一键启动所有服务

**任务清单**:

- [x] 创建 docker-compose.yml
- [x] 为 embedding-service 创建 Dockerfile
- [x] 为 llm-service 创建 Dockerfile
- [x] 为 wrapper-service 创建 Dockerfile
- [x] 配置 SurrealDB 服务
- [x] 配置网络和卷
- [x] 编写启动脚本

**状态**: ✅ 已完成 (2026-03-09)

**预期收益**:

- 开发环境标准化
- 一键启动所有服务
- 便于测试和演示

**工作量**: 2-3天
**依赖**: 无

---

### P3-2: HNSW向量索引 ✅ 已完成

**目标**: 将记忆搜索延迟降低 90%

**现状**: 暴力搜索 O(n)，延迟 200-500ms
**目标**: HNSW 索引 O(log n)，延迟 <50ms

**任务清单**:

- [x] 在 SurrealDB 创建 HNSW 索引
- [x] 修改 search_memories 使用索引
- [ ] 添加性能基准测试
- [ ] 对比测试（暴力 vs HNSW）

**状态**: ✅ 核心功能已完成 (2026-03-09)

**技术方案**:

```sql
DEFINE INDEX memory_embedding_hnsw ON memory 
  FIELDS embedding 
  TYPE HNSW 
  DIMENSION 1024 
  DISTANCE COSINE;
```text

**预期收益**:

- 搜索延迟: 500ms → 50ms
- 支持百万级记忆
- CPU使用率降低

**工作量**: 2-3天
**依赖**: 无

---

### Phase 3E: Polyglot 搜索架构 ✅ 已完成

**目标**: 解决 SurrealDB 3.0.1 BM25 全文搜索的日期分词 bug 和中文分词不足

**架构决策**: Polyglot Persistence（多模搜索引擎协作）

- SurrealDB → 向量搜索 (HNSW) + 图关系 (RELATE) + 数据存储
- Meilisearch → 全文搜索 + CJK 中文分词 + 日期精确匹配

**任务清单**:

- [x] MeilisearchConfig 配置类 + 环境变量
- [x] MeilisearchClient 异步客户端 (meili_client.py)
- [x] 上传双写同步（SurrealDB + Meilisearch）
- [x] 关键词搜索路由到 Meilisearch
- [x] 混合搜索 RRF 融合（向量走 SurrealDB + 关键词走 Meilisearch）
- [x] 数据迁移脚本 (migrate_to_meilisearch.py)
- [x] 23 个单元测试（全部通过）
- [x] 端到��验证（日期/中文/混合搜索全部通过）
- [x] docker-compose.yml 新增 Meilisearch 服务
- [x] README.md 更新

**状态**: ✅ 已完成 (2026-03-12)

**预期收益**:

- 日期搜索精确匹配（"2026-03-12" 不再被拆分为三个 token）
- CJK 中文分词开箱即用
- 优雅降级（Meilisearch 不可用时回退到 SurrealDB BM25）
- 搜索质量显著提升

**工作量**: 2-3天
**依赖**: P3-2（HNSW 向量索引）

---

### P3-3: 监控告警完善

**目标**: 生产环境自动告警

**任务清单**:

- [ ] 添加 Alertmanager 配置
- [ ] 配置告警规则（错误率、延迟、容量）
- [ ] 创建 Grafana 仪表板
- [ ] 集成企业通知（钉钉/企业微信/Slack）

**告警规则示例**:

```yaml
- alert: HighErrorRate
  expr: rate(wrapper_requests_total{status=~"5.."}[5m]) > 0.05
  for: 5m
  labels:
    severity: critical
```

**预期收益**:

- 问题及时发现
- 减少故障恢复时间
- 提高SLA

**工作量**: 3-4天
**依赖**: P3-1（Docker Compose）

---

### P3-4: Kubernetes 部署

**目标**: 云原生部署

**任务清单**:

- [ ] 创建 Helm Charts
- [ ] 配置 Deployment + Service
- [ ] 配置 HPA 自动扩缩容
- [ ] 配置 Ingress
- [ ] 配置 ConfigMap + Secret
- [ ] 编写部署文档

**预期收益**:

- 自动扩缩容
- 高可用部署
- 云厂商无关

**工作量**: 5-7天
**依赖**: P3-1

---

### P3-5: 审计日志

**目标**: 合规审计

**任务清单**:

- [ ] 创建审计日志中间件
- [ ] 记录 API 调用（用户、时间、结果）
- [ ] 审计日志存储（文件/Elasticsearch）
- [ ] 审计日志查询 API

**预期收益**:

- 安全合规
- 故障溯源
- 用户行为分析

**工作量**: 2-3天
**依赖**: 无

---

### Phase 3F: 同步冲突解决 ✅ 已完成

**目标**: 支持多设备、多用户、离线编辑等场景的数据同步

**任务清单**:

- [x] 冲持久化存储（conflict 表）
- [x] 冲突检测逻辑（sync_incremental 集成）
- [x] 三种解决策略（use_local/use_remote/keep_both）
- [x] 冲突状态跟踪
- [x] Meilisearch 同步更新
- [x] 完整测试覆盖
- [x] 最佳实践文档

**状态**: ✅ 已完成 (2026-03-23)

**核心功能**:

- 冲突记录持久化到 SurrealDB
- 支持多租户隔离
- 事务安全的冲突解决
- 自动同步 Meilisearch 索引
- 幂等操作支持

**预期收益**:

- 多设备无缝同步
- 离线编辑支持
- 多用户协作能力
- 数据备份与恢复

**文档**: [同步冲突解决最佳实践](SYNC_CONFLICT_RESOLUTION.md)

**工作量**: 1-2天
**依赖**: Phase 3E（Polyglot 搜索架构）

---

## 未来展望 (P4+)

### P4: 高级功能

- [ ] 模型热更新（不停机切换模型）
- [ ] 多模型路由（负载均衡）
- [ ] A/B 测试支持
- [ ] 模型量化（边缘设备部署）

### P5: 企业级

- [ ] 多租户支持
- [ ] 数据加密（传输+存储）
- [ ] 密钥管理系统（Vault 集成）
- [ ] GDPR 合规

---

## 贡献指南

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何参与。

## 许可证

[MIT](LICENSE)

```text

## 文件位置

D:\embedding_service\ROADMAP.md

## 验证

检查 Markdown 语法正确。
<!-- OMO_INTERNAL_INITIATOR -->
