# v3.2 实施追踪矩阵（RTM）- 后端专用

> **版本**: v3.2.0  
> **项目**: Embedding Service (Python/FastAPI)  
> **基线标签**: [v3.2-design-baseline](https://github.com/longray/opencode-memory-plugin/releases/tag/v3.2-design-baseline)  
> **最后更新**: 2026-04-10  
> **状态**: 开发中

---

## 说明

本矩阵追踪 v3.2 架构设计到后端代码实现的映射关系，确保每个设计点都有对应的实现和测试。

**状态图例**:
- ⏳ 待实现
- 🔄 进行中
- ⚠️ 有风险
- ✅ 已完成
- ❌ 已取消

**风险图例**:
- 🔴 高风险
- 🟡 中风险
- 🟢 低风险

---

## 1. WebSocket 服务端模块

| 设计ID | 设计文档 | 功能点 | 代码位置 | 测试文件 | 状态 | 风险 | Backlog |
|--------|----------|--------|----------|----------|------|------|---------|
| WS-SRV-001 | BACKEND-v3.2-WEBSOCKET.md | 心跳处理 30s | `wrapper/src/routers/websocket.py` | `tests/test_websocket.py` | ⏳ | 🔴 高 | BL-B-36 |
| WS-SRV-002 | BACKEND-v3.2-WEBSOCKET.md | 连接池管理 | `wrapper/src/utils/websocket/pool.py` | `tests/test_websocket.py` | ⏳ | 🔴 高 | BL-B-36 |
| WS-SRV-003 | BACKEND-v3.2-WEBSOCKET.md | ACK 确认机制 | `wrapper/src/utils/websocket/ack.py` | `tests/test_websocket.py` | ⏳ | 🔴 高 | BL-B-36 |
| WS-SRV-004 | BACKEND-v3.2-WEBSOCKET.md | 消息队列持久化 | `wrapper/src/utils/websocket/persistent_queue.py` | `tests/test_websocket.py` | ⏳ | 🔴 高 | BL-B-36 |
| WS-SRV-005 | BACKEND-v3.2-WEBSOCKET.md | 状态恢复 | `wrapper/src/utils/websocket/state_recovery.py` | `tests/test_websocket.py` | ⏳ | 🟡 中 | BL-B-36 |
| WS-SRV-006 | BACKEND-v3.2-WEBSOCKET.md | DIFF 模式处理 | `wrapper/src/utils/websocket/diff_handler.py` | `tests/test_websocket.py` | ⏳ | 🟡 中 | BL-B-36 |
| WS-SRV-007 | BACKEND-v3.2-WEBSOCKET.md | 并发连接 ≥1000 | `wrapper/src/routers/websocket.py` | `tests/performance/test_ws_load.py` | ⏳ | 🔴 高 | BL-B-43 |
| WS-SRV-008 | BACKEND-v3.2-WEBSOCKET.md | 消息延迟 p99<100ms | `wrapper/src/routers/websocket.py` | `tests/performance/test_ws_load.py` | ⏳ | 🔴 高 | BL-B-43 |
| WS-SRV-009 | BACKEND-v3.2-WEBSOCKET.md | 心跳成功率 ≥99% | `wrapper/src/routers/websocket.py` | `tests/performance/test_ws_load.py` | ⏳ | 🟡 中 | BL-B-43 |

---

## 2. PrecomputeService 模块

| 设计ID | 设计文档 | 功能点 | 代码位置 | 测试文件 | 状态 | 风险 | Backlog |
|--------|----------|--------|----------|----------|------|------|---------|
| PC-SRV-001 | BACKEND-v3.2-PRECOMPUTE.md | 批处理大小 100 | `wrapper/src/services/precompute.py` | `tests/test_precompute.py` | ⏳ | 🟡 中 | BL-B-37 |
| PC-SRV-002 | BACKEND-v3.2-PRECOMPUTE.md | 增量分析（指纹比对） | `wrapper/src/services/precompute.py` | `tests/test_precompute.py` | ⏳ | 🟡 中 | BL-B-37 |
| PC-SRV-003 | BACKEND-v3.2-PRECOMPUTE.md | 调用关系创建 | `wrapper/src/services/precompute.py` | `tests/test_precompute.py` | ⏳ | 🟡 中 | BL-B-44 |
| PC-SRV-004 | BACKEND-v3.2-PRECOMPUTE.md | 循环依赖检测 | `wrapper/src/services/precompute.py` | `tests/test_precompute.py` | ⏳ | 🟢 低 | BL-B-37 |
| PC-SRV-005 | BACKEND-v3.2-PRECOMPUTE.md | 权重计算 | `wrapper/src/services/precompute.py` | `tests/test_precompute.py` | ⏳ | 🟢 低 | BL-B-37 |
| PC-SRV-006 | BACKEND-v3.2-PRECOMPUTE.md | 性能监控 | `wrapper/src/utils/performance_monitor.py` | `tests/test_performance_monitor.py` | ⏳ | 🟡 中 | BL-B-37 |
| PC-SRV-007 | BACKEND-v3.2-PRECOMPUTE.md | 并发控制 | `wrapper/src/utils/concurrency_control.py` | `tests/test_concurrency.py` | ⏳ | 🟡 中 | BL-B-37 |

---

## 3. 数据库 Schema 模块

| 设计ID | 设计文档 | 功能点 | 代码位置 | 测试文件 | 状态 | 风险 | Backlog |
|--------|----------|--------|----------|----------|------|------|---------|
| DB-SRV-001 | DATABASE-v3.2-SCHEMA.md | atom 表创建 | `wrapper/src/db/migrations/v3.2_schema.surql` | `tests/test_db_schema.py` | ⏳ | 🟢 低 | BL-B-41 |
| DB-SRV-002 | DATABASE-v3.2-SCHEMA.md | entity 表创建 | `wrapper/src/db/migrations/v3.2_schema.surql` | `tests/test_db_schema.py` | ⏳ | 🟢 低 | BL-B-41 |
| DB-SRV-003 | DATABASE-v3.2-SCHEMA.md | reference 表创建 | `wrapper/src/db/migrations/v3.2_schema.surql` | `tests/test_db_schema.py` | ⏳ | 🟢 低 | BL-B-41 |
| DB-SRV-004 | DATABASE-v3.2-SCHEMA.md | tenant_id 预留字段 | `wrapper/src/db/migrations/v3.2_schema.surql` | `tests/test_db_schema.py` | ⏳ | 🟢 低 | BL-B-41 |
| DB-SRV-005 | DATABASE-v3.2-SCHEMA.md | ChangeFeed 7d TTL | `wrapper/src/db/migrations/v3.2_schema.surql` | `tests/test_db_schema.py` | ⏳ | 🟡 中 | BL-B-41 |

---

## 4. API 模块

| 设计ID | 设计文档 | 功能点 | 代码位置 | 测试文件 | 状态 | 风险 | Backlog |
|--------|----------|--------|----------|----------|------|------|---------|
| API-SRV-001 | PLUGIN-v3.2-API.md | Memory CRUD 端点 | `wrapper/src/routers/memories.py` | `tests/test_wrapper_api.py` | ⏳ | 🟢 低 | BL-B-35 |
| API-SRV-002 | PLUGIN-v3.2-API.md | Code Analysis 端点 | `wrapper/src/routers/code.py` | `tests/test_code_analysis.py` | ⏳ | 🟡 中 | BL-B-35 |
| API-SRV-003 | PLUGIN-v3.2-API.md | WebSocket 端点 | `wrapper/src/routers/websocket.py` | `tests/test_websocket.py` | ⏳ | 🔴 高 | BL-B-36 |
| API-SRV-004 | BACKEND-v3.2-MEILISEARCH.md | Meilisearch SDK 0.40 | `wrapper/src/utils/meili_client.py` | `tests/test_meili_client.py` | ⏳ | 🟢 低 | BL-B-38 |

---

## 5. 部署配置模块

| 设计ID | 设计文档 | 功能点 | 代码位置 | 测试文件 | 状态 | 风险 | Backlog |
|--------|----------|--------|----------|----------|------|------|---------|
| DEP-SRV-001 | DEPLOYMENT-v3.2.md | Docker 多阶段构建 | `wrapper/Dockerfile` | CI 构建 | ⏳ | 🟢 低 | BL-B-35 |
| DEP-SRV-002 | DEPLOYMENT-v3.2.md | docker-compose 配置 | `docker-compose.yml` | 手动测试 | ⏳ | 🟢 低 | BL-B-35 |
| DEP-SRV-003 | DEPLOYMENT-v3.2.md | Kubernetes 部署 | `k8s/` | 手动测试 | ⏳ | 🟡 中 | BL-B-48 |
| DEP-SRV-004 | DEPLOYMENT-v3.2.md | SSL 自动续期 | `scripts/ssl-renew.sh` | 手动测试 | ⏳ | 🟡 中 | BL-B-50 |
| DEP-SRV-005 | BACKEND-v3.2-MIGRATION.md | 端口迁移 17999→18008 | `wrapper/src/config.py` | 集成测试 | ⏳ | 🟡 中 | BL-B-39 |

---

## 6. 依赖版本模块

| 设计ID | 设计文档 | 功能点 | 代码位置 | 测试文件 | 状态 | 风险 | Backlog |
|--------|----------|--------|----------|----------|------|------|---------|
| VER-SRV-001 | DEPENDENCY-VERSIONS.md | tree-sitter 0.25.x | `pyproject.toml` | 依赖安装测试 | ⏳ | 🔴 高 | BL-B-35 |
| VER-SRV-002 | DEPENDENCY-VERSIONS.md | surrealdb 1.0.8 | `pyproject.toml` | 依赖安装测试 | ⏳ | 🟢 低 | BL-B-35 |
| VER-SRV-003 | DEPENDENCY-VERSIONS.md | meilisearch 0.40.0 | `pyproject.toml` | 依赖安装测试 | ⏳ | 🟢 低 | BL-B-38 |
| VER-SRV-004 | DEPENDENCY-VERSIONS.md | fastapi 0.115.x | `pyproject.toml` | 依赖安装测试 | ⏳ | 🟢 低 | BL-B-35 |

---

## 统计摘要

| 模块 | 总数 | 已完成 | 进行中 | 待实现 | 高风险 |
|------|------|--------|--------|--------|--------|
| WebSocket 服务端 | 9 | 0 | 0 | 9 | 6 |
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
| 2026-04-10 | 更新为后端专用版本，修正代码路径 | OpenCode |

---

_基线标签: v3.2-design-baseline_  
_文档版本: v3.2.0-backend_
