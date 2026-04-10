# v3.2 验收标准

> **版本**: v3.2.0  
> **日期**: 2026-04-10  
> **状态**: 实施版  
> **基线**: [v3.2-design-baseline](https://github.com/longray/opencode-memory-plugin/releases/tag/v3.2-design-baseline)

---

## 概述

本文档定义 v3.2 版本开发完成的验收标准。所有标准必须满足，方可发布 v3.2.0。

---

## 1. 功能验收标准

### 1.1 API 端点完整性 ✅

**标准**: 所有 19 个 API 端点必须实现并通过测试

| 类别 | 端点 | 方法 | 状态要求 |
|------|------|------|----------|
| Memory | `/api/v1/memories` | POST | ✅ 实现 |
| Memory | `/api/v1/memories/{id}` | GET | ✅ 实现 |
| Memory | `/api/v1/memories/{id}` | PATCH | ✅ 实现 |
| Memory | `/api/v1/memories/{id}` | DELETE | ✅ 实现 |
| Memory | `/api/v1/memories/search` | GET | ✅ 实现 |
| Code | `/api/v1/code/analyze` | POST | ✅ 实现 |
| Code | `/api/v1/code/navigate` | GET | ✅ 实现 |
| Code | `/api/v1/code/impact` | GET | ✅ 实现 |
| Code | `/api/v1/code/search` | GET | ✅ 实现 |
| WebSocket | `/api/v1/ws` | WS | ✅ 实现 |
| WebSocket | `/api/v1/ws/subscribe` | WS | ✅ 实现 |
| WebSocket | `/api/v1/ws/ack` | WS | ✅ 实现 |
| Sync | `/api/v1/sync/incremental` | POST | ✅ 实现 |
| Sync | `/api/v1/sync/full` | POST | ✅ 实现 |
| Sync | `/api/v1/sync/status` | GET | ✅ 实现 |
| Health | `/health` | GET | ✅ 实现 |
| Health | `/health/db` | GET | ✅ 实现 |
| Health | `/health/ws` | GET | ✅ 实现 |
| Metrics | `/metrics` | GET | ✅ 实现 |

**验证方式**:
```bash
# 检查所有端点是否实现
curl http://localhost:18008/api/v1/memories -X POST
curl http://localhost:18008/health
# ... 测试所有 19 个端点
```

---

### 1.2 WebSocket 功能完整性 ✅

**标准**: 所有 WebSocket 功能必须实现

| 功能 | 要求 | 验证方式 |
|------|------|----------|
| 心跳机制 | 30s 间隔，自动 ping/pong | 抓包验证 |
| 重连机制 | 2次未响应触发重连 | 模拟断网测试 |
| 指数退避 | 1s→2s→4s... 最大300s | 日志验证 |
| 最大重连 | 10次后停止 | 配置验证 |
| 消息确认 | 5s 超时，3次重试 | 测试验证 |
| DIFF 模式 | 减少 90% 数据传输 | 流量对比 |

---

### 1.3 PrecomputeService 功能完整性 ✅

**标准**: 所有预计算功能必须实现

| 功能 | 要求 | 验证方式 |
|------|------|----------|
| 批处理 | 批大小 100 | 代码审查 |
| 增量分析 | 指纹比对，只分析变更文件 | 测试验证 |
| 调用关系 | 创建函数调用图 | 数据库验证 |
| 循环检测 | 检测并标记循环调用 | 测试验证 |
| 权重计算 | 基于复杂度计算关系权重 | 代码审查 |
| 性能监控 | 记录处理时间和内存 | 日志验证 |
| 并发控制 | 最大并发 5 个文件 | 配置验证 |

---

## 2. 性能验收标准

### 2.1 WebSocket 性能 ✅

| 指标 | 基准值 | 测试条件 | 验证方式 |
|------|--------|----------|----------|
| 并发连接 | ≥ 1000 | 单服务器实例 | 压力测试 |
| 消息吞吐量 | ≥ 10,000 msg/s | 1000 并发连接 | 性能测试 |
| 消息延迟 | p99 < 100ms | 局域网环境 | 延迟测试 |
| 心跳成功率 | ≥ 99% | 30s 间隔，1小时 | 统计测试 |
| 重连时间 | < 5s | 首次重连 | 计时测试 |
| 内存使用 | < 500MB | 1000 并发连接 | 内存监控 |

**验证命令**:
```bash
# 运行性能测试
npm run test:performance:websocket
```

---

### 2.2 PrecomputeService 性能 ✅

| 指标 | 基准值 | 测试条件 | 验证方式 |
|------|--------|----------|----------|
| 处理速度 | > 1000 行/秒 | 标准代码文件 | 性能测试 |
| 内存占用 | < 100MB | 处理 1000 行代码 | 内存监控 |
| 批处理时间 | < 10s | 批大小 100 | 计时测试 |

---

## 3. 配置项验收标准

### 3.1 配置项完整性 ✅

**标准**: 所有 28 个配置项必须可配置并验证

| 类别 | 配置项 | 默认值 | 验证方式 |
|------|--------|--------|----------|
| WebSocket | `WS_HEARTBEAT_INTERVAL` | 30 | 配置读取 |
| WebSocket | `WS_RECONNECT_MAX_ATTEMPTS` | 10 | 配置读取 |
| WebSocket | `WS_RECONNECT_BASE_DELAY` | 1 | 配置读取 |
| WebSocket | `WS_MESSAGE_TIMEOUT` | 5 | 配置读取 |
| Precompute | `PRECOMPUTE_BATCH_SIZE` | 100 | 配置读取 |
| Precompute | `PRECOMPUTE_INTERVAL` | 300 | 配置读取 |
| Precompute | `PRECOMPUTE_MAX_CONCURRENT` | 5 | 配置读取 |
| Port | `PORT` | 18008 | 配置读取 |
| Port | `LEGACY_PORT` | 17999 | 配置读取 |
| ... | ... | ... | ... |

**验证方式**:
```bash
# 检查所有配置项
cat .env.example | grep -E "^[A-Z]" | wc -l  # 应输出 28
```

---

## 4. 代码质量验收标准

### 4.1 代码覆盖率 ✅

**标准**: 代码覆盖率 ≥ 85%

| 模块 | 要求 | 验证方式 |
|------|------|----------|
| 总体覆盖率 | ≥ 85% | `npm run test:coverage` |
| WebSocket 模块 | ≥ 90% | 覆盖率报告 |
| PrecomputeService | ≥ 85% | 覆盖率报告 |
| API 层 | ≥ 80% | 覆盖率报告 |

---

### 4.2 代码规范 ✅

**标准**: 所有代码通过 lint 检查

```bash
# 验证命令
npm run lint        # JavaScript
npm run lint:md     # Markdown
```

---

## 5. 文档验收标准

### 5.1 文档完整性 ✅

**标准**: 所有设计文档与实现一致

| 文档 | 要求 | 验证方式 |
|------|------|----------|
| UNIFIED-ARCHITECTURE-v3.2.md | 与实现一致 | 代码审查 |
| BACKEND-v3.2-IMPLEMENTATION.md | 与实现一致 | 代码审查 |
| BACKEND-v3.2-WEBSOCKET.md | 与实现一致 | 代码审查 |
| BACKEND-v3.2-PRECOMPUTE.md | 与实现一致 | 代码审查 |
| PLUGIN-v3.2-API.md | 与实现一致 | 代码审查 |
| DATABASE-v3.2-SCHEMA.md | 与实现一致 | 数据库验证 |
| DEPLOYMENT-v3.2.md | 与实现一致 | 部署验证 |

---

### 5.2 RTM 更新 ✅

**标准**: RTM 中所有 34 个追踪项状态已更新

| 状态 | 数量要求 |
|------|----------|
| ✅ 已完成 | ≥ 34 |
| ❌ 已取消 | 0 |
| ⏳ 待实现 | 0 |

---

## 6. 部署验收标准

### 6.1 Docker 部署 ✅

**标准**: Docker 镜像可正常构建和运行

```bash
# 验证命令
docker-compose build
docker-compose up -d
curl http://localhost:18008/health  # 应返回 200
```

---

### 6.2 Kubernetes 部署 ✅

**标准**: K8s 配置可正常部署

```bash
# 验证命令
kubectl apply -f k8s/
kubectl get pods  # 应显示 Running
```

---

## 7. 验收测试清单

### 7.1 功能测试 ✅

- [ ] 所有 19 个 API 端点测试通过
- [ ] WebSocket 功能测试通过
- [ ] PrecomputeService 功能测试通过
- [ ] 数据库迁移测试通过

### 7.2 性能测试 ✅

- [ ] WebSocket 性能测试通过（1000 并发）
- [ ] PrecomputeService 性能测试通过
- [ ] 内存泄漏测试通过

### 7.3 集成测试 ✅

- [ ] 端到端流程测试通过
- [ ] 端口迁移测试通过
- [ ] 向后兼容测试通过

### 7.4 部署测试 ✅

- [ ] Docker 部署测试通过
- [ ] Kubernetes 部署测试通过
- [ ] SSL 配置测试通过

---

## 8. 验收流程

### 8.1 自验收（开发团队）

1. 开发团队完成所有功能开发
2. 运行所有测试，确保通过
3. 更新 RTM 状态
4. 提交验收申请

### 8.2 正式验收（QA/架构师）

1. QA 运行验收测试清单
2. 架构师审查代码与设计一致性
3. 性能测试验证
4. 签署验收报告

### 8.3 发布准备

1. 更新 CHANGELOG.md
2. 创建发布标签 `v3.2.0`
3. 发布 Release Notes

---

## 附录

### A. 验收工具

| 工具 | 用途 | 命令 |
|------|------|------|
| Jest | 单元测试 | `npm test` |
| Artillery | 性能测试 | `npm run test:performance` |
| curl | API 测试 | `curl http://localhost:18008/health` |
| Docker | 部署测试 | `docker-compose up` |

### B. 参考文档

- [RTM.md](./RTM.md) - 实施追踪矩阵
- [BACKLOG.md](../../BACKLOG.md) - 任务清单
- [CHANGELOG.md](../../CHANGELOG.md) - 版本历史

---

_基线标签: v3.2-design-baseline_  
_文档版本: v3.2.0_  
_最后更新: 2026-04-10_
