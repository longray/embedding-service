# v3.2 实施追踪矩阵（RTM）

> **版本**: v3.2.0  
> **基线标签**: [v3.2-design-baseline](https://github.com/longray/opencode-memory-plugin/releases/tag/v3.2-design-baseline)  
> **最后更新**: 2026-04-10  
> **状态**: 开发中

---

## 说明

本矩阵追踪 v3.2 架构设计到代码实现的映射关系，确保每个设计点都有对应的实现和测试。

**状态图例**:
- ⏳ 待实现
- 🔄 进行中
- ⚠️ 有风险
- ✅ 已完成
- ❌ 已取消

---

## 1. WebSocket 模块

| 设计ID | 设计文档 | 功能点 | 代码位置 | 测试文件 | 状态 | 风险 |
|--------|----------|--------|----------|----------|------|------|
| WS-001 | BACKEND-v3.2-WEBSOCKET.md | 心跳机制 30s | `lib/ws-client.js` | `test/ws-client.test.js` | ⏳ | 🔴 高 |
| WS-002 | BACKEND-v3.2-WEBSOCKET.md | 2次未响应触发重连 | `lib/ws-client.js` | `test/ws-client.test.js` | ⏳ | 🔴 高 |
| WS-003 | BACKEND-v3.2-WEBSOCKET.md | 指数退避重连 1→2→4s | `lib/ws-client.js` | `test/ws-client.test.js` | ⏳ | 🔴 高 |
| WS-004 | BACKEND-v3.2-WEBSOCKET.md | 最大重连 10次 | `lib/ws-client.js` | `test/ws-client.test.js` | ⏳ | 🔴 高 |
| WS-005 | BACKEND-v3.2-WEBSOCKET.md | 消息确认 ACK | `lib/ws-client.js` | `test/ws-client.test.js` | ⏳ | 🔴 高 |
| WS-006 | BACKEND-v3.2-WEBSOCKET.md | DIFF 模式 | `lib/ws-client.js` | `test/ws-client.test.js` | ⏳ | 🟡 中 |
| WS-007 | PLUGIN-v3.2-API.md | 并发连接 ≥1000 | `test/performance/ws-load-test.js` | 性能测试 | ⏳ | 🔴 高 |
| WS-008 | PLUGIN-v3.2-API.md | 消息延迟 p99<100ms | `test/performance/ws-load-test.js` | 性能测试 | ⏳ | 🔴 高 |
| WS-009 | PLUGIN-v3.2-API.md | 心跳成功率 ≥99% | `test/performance/ws-load-test.js` | 性能测试 | ⏳ | 🟡 中 |

---

## 2. PrecomputeService 模块

| 设计ID | 设计文档 | 功能点 | 代码位置 | 测试文件 | 状态 | 风险 |
|--------|----------|--------|----------|----------|------|------|
| PC-001 | BACKEND-v3.2-PRECOMPUTE.md | 批处理大小 100 | `services/precompute.py` | `test/precompute.test.js` | ⏳ | 🟡 中 |
| PC-002 | BACKEND-v3.2-PRECOMPUTE.md | 增量分析（指纹） | `lib/code-fingerprint.js` | `test/code-fingerprint.test.js` | ⏳ | 🟡 中 |
| PC-003 | BACKEND-v3.2-PRECOMPUTE.md | 调用关系创建 | `services/precompute.py` | `test/precompute.test.js` | ⏳ | 🟡 中 |
| PC-004 | BACKEND-v3.2-PRECOMPUTE.md | 循环检测 | `services/precompute.py` | `test/precompute.test.js` | ⏳ | 🟢 低 |
| PC-005 | BACKEND-v3.2-PRECOMPUTE.md | 权重计算 | `services/precompute.py` | `test/precompute.test.js` | ⏳ | 🟢 低 |
| PC-006 | BACKEND-v3.2-PRECOMPUTE.md | 性能监控 | `services/performance_monitor.py` | `test/performance_monitor.test.js` | ⏳ | 🟡 中 |
| PC-007 | BACKEND-v3.2-PRECOMPUTE.md | 并发控制 | `services/concurrency_control.py` | `test/concurrency.test.js` | ⏳ | 🟡 中 |

---

## 3. 数据库 Schema 模块

| 设计ID | 设计文档 | 功能点 | 代码位置 | 测试文件 | 状态 | 风险 |
|--------|----------|--------|----------|----------|------|------|
| DB-001 | DATABASE-v3.2-SCHEMA.md | atom 表 | `db/migrations/v3.2_schema.sql` | `test/db.test.js` | ⏳ | 🟢 低 |
| DB-002 | DATABASE-v3.2-SCHEMA.md | entity 表 | `db/migrations/v3.2_schema.sql` | `test/db.test.js` | ⏳ | 🟢 低 |
| DB-003 | DATABASE-v3.2-SCHEMA.md | reference 表 | `db/migrations/v3.2_schema.sql` | `test/db.test.js` | ⏳ | 🟢 低 |
| DB-004 | DATABASE-v3.2-SCHEMA.md | tenant_id 预留字段 | `db/migrations/v3.2_schema.sql` | `test/db.test.js` | ⏳ | 🟢 低 |
| DB-005 | DATABASE-v3.2-SCHEMA.md | ChangeFeed 7d | `db/migrations/v3.2_schema.sql` | `test/db.test.js` | ⏳ | 🟡 中 |

---

## 4. API 模块

| 设计ID | 设计文档 | 功能点 | 代码位置 | 测试文件 | 状态 | 风险 |
|--------|----------|--------|----------|----------|------|------|
| API-001 | PLUGIN-v3.2-API.md | Memory CRUD | `tools/core.js` | `test/tools/core.test.js` | ⏳ | 🟢 低 |
| API-002 | PLUGIN-v3.2-API.md | Code Analysis | `tools/code.js` | `test/tools/code.test.js` | ⏳ | 🟡 中 |
| API-003 | PLUGIN-v3.2-API.md | WebSocket 连接 | `lib/ws-client.js` | `test/ws-client.test.js` | ⏳ | 🔴 高 |
| API-004 | BACKEND-v3.2-MEILISEARCH.md | Meilisearch SDK 0.40 | `utils/meili_client.py` | `test/meili_client.test.js` | ⏳ | 🟢 低 |

---

## 5. 部署配置模块

| 设计ID | 设计文档 | 功能点 | 代码位置 | 测试文件 | 状态 | 风险 |
|--------|----------|--------|----------|----------|------|------|
| DEP-001 | DEPLOYMENT-v3.2.md | Docker 多阶段构建 | `Dockerfile` | CI 构建 | ⏳ | 🟢 低 |
| DEP-002 | DEPLOYMENT-v3.2.md | docker-compose 配置 | `docker-compose.yml` | 手动测试 | ⏳ | 🟢 低 |
| DEP-003 | DEPLOYMENT-v3.2.md | Kubernetes 部署 | `k8s/` | 手动测试 | ⏳ | 🟡 中 |
| DEP-004 | DEPLOYMENT-v3.2.md | SSL 自动续期 | `scripts/ssl-renew.sh` | 手动测试 | ⏳ | 🟡 中 |
| DEP-005 | BACKEND-v3.2-MIGRATION.md | 端口迁移 17999→18008 | `config.js` | 集成测试 | ⏳ | 🟡 中 |

---

## 6. 依赖版本模块

| 设计ID | 设计文档 | 功能点 | 代码位置 | 测试文件 | 状态 | 风险 |
|--------|----------|--------|----------|----------|------|------|
| VER-001 | DEPENDENCY-VERSIONS.md | tree-sitter 0.25.x | `pyproject.toml` | 依赖安装测试 | ⏳ | 🔴 高 |
| VER-002 | DEPENDENCY-VERSIONS.md | surrealdb 1.0.8 | `pyproject.toml` | 依赖安装测试 | ⏳ | 🟢 低 |
| VER-003 | DEPENDENCY-VERSIONS.md | meilisearch 0.40.0 | `pyproject.toml` | 依赖安装测试 | ⏳ | 🟢 低 |
| VER-004 | DEPENDENCY-VERSIONS.md | ws 8.20.0 | `package.json` | 依赖安装测试 | ⏳ | 🟢 低 |

---

## 统计摘要

| 模块 | 总数 | 已完成 | 进行中 | 待实现 | 高风险 |
|------|------|--------|--------|--------|--------|
| WebSocket | 9 | 0 | 0 | 9 | 6 |
| PrecomputeService | 7 | 0 | 0 | 7 | 0 |
| Database Schema | 5 | 0 | 0 | 5 | 0 |
| API | 4 | 0 | 0 | 4 | 1 |
| Deployment | 5 | 0 | 0 | 5 | 0 |
| Dependencies | 4 | 0 | 0 | 4 | 1 |
| **总计** | **34** | **0** | **0** | **34** | **8** |

---

## 更新记录

| 日期 | 更新内容 | 更新人 |
|------|----------|--------|
| 2026-04-10 | 初始版本，创建 34 个追踪项 | OpenCode |

---

_基线标签: v3.2-design-baseline_  
_文档版本: v3.2.0_
