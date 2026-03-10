# Embedding Service 项目路线图

## 版本历史

### v1.1.0 (当前) - 最小化包装服务 ✅
**发布日期**: 2026-03-10

**核心功能**:
- Embedding 服务（Qwen3-Embedding-0.6B）
- LLM 服务（MiniCPM4-0.5B）
- 最小化包装服务（端口 17999）
  - LRU 缓存
  - HTTP 连接池
  - SurrealDB 长期连接
  - 四个 API：`/health`、`/v1/embeddings`、`/api/v1/memories`、`/api/v1/memories/search`
- 记忆管理系统（SurrealDB + 向量搜索 + HNSW 索引）
- Docker Compose 一键部署
- CI/CD（GitHub Actions）

---

## P3 优化阶段 (进行中)

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
```

**预期收益**:
- 搜索延迟: 500ms → 50ms
- 支持百万级记忆
- CPU使用率降低

**工作量**: 2-3天
**依赖**: 无

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
