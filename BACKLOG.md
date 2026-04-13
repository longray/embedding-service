# BACKLOG v3.2

> **版本**: v3.2.0  
> **创建日期**: 2026-04-12  
> **最后更新**: 2026-04-12  
> **总任务数**: 40  
> **预估总工时**: 28 天  
> **协议**: AGENT-COLLABORATION-PROTOCOL-v1.0

**历史场景**: 场景 1-4 已完成并归档至 `backlog_archive.md`

---

## 快速导航

- [任务总览（表格）](#任务总览表格)
- [Phase 1: 依赖升级](#phase-1-依赖升级1-天)
- [Phase 2: WebSocket 重写](#phase-2-websocket-重写55-天)
- [Phase 3: PrecomputeService](#phase-3-precomputeservice7-天)
- [Phase 4: Meilisearch SDK](#phase-4-meilisearch-sdk-升级2-天)
- [Phase 5: Schema 升级](#phase-5-surrealdb-schema-升级25-天)
- [Phase 6: 端口迁移](#phase-6-端口迁移25-天)
- [Phase 7: 测试](#phase-7-测试45-天)
- [文档完善](#文档完善5-天)
- [统计汇总](#统计汇总)

---

## 任务总览（表格）

| 编号 | 目标 | 优先级 | 工时 | 状态 | 详情 |
|------|------|--------|------|------|------|
| **Phase 1** |
| BL-B-31 | 依赖升级 — pyproject.toml | P0 | 1 天 | ✅ | [详情](#bl-b-31-p0-依赖升级--pyprojecttoml) |
| **Phase 2** |
| BL-B-1 | WebSocket — 心跳机制 | P0 | 1 天 | ✅ | [详情](#bl-b-1-p0-websocket-可靠连接--心跳机制) |
| BL-B-2 | WebSocket — 指数退避重连 | P0 | 1 天 | ✅ | [详情](#bl-b-2-p0-websocket-可靠连接--指数退避重连) |
| BL-B-3 | WebSocket — ACK 确认系统 | P0 | 1 天 | ✅ | [详情](#bl-b-3-p0-websocket-可靠连接--ack-确认系统) |
| BL-B-4 | WebSocket — DIFF 模式 | P1 | 1 天 | ✅ | [详情](#bl-b-4-p1-websocket-可靠连接--diff-模式) |
| BL-B-5 | WebSocket — 状态恢复 | P0 | 1 天 | ✅ | [详情](#bl-b-5-p0-websocket-可靠连接--状态恢复) |
| BL-B-6 | WebSocket — 并发连接测试 | P1 | 0.5 天 | ✅ | [详情](#bl-b-6-p1-websocket-性能--并发连接测试) |
| BL-B-7 | WebSocket — 消息延迟测试 | P1 | 0.5 天 | ✅ | [详情](#bl-b-7-p1-websocket-性能--消息延迟测试) |
| BL-B-51 | WebSocket — 心跳成功率 ≥99% | P1 | 0.5 天 | ✅ | [详情](#bl-b-51-p1-websocket-可靠性--心跳成功率-99-验证) |
| **Phase 3** |
| BL-B-8 | PrecomputeService — 基础架构 | P0 | 1 天 | ✅ | [详情](#bl-b-8-p0-precomputeservice--基础架构) |
| BL-B-9 | PrecomputeService — tree-sitter + 指纹 | P0 | 1.5 天 | ✅ | [详情](#bl-b-9-p0-precomputeservice--tree-sitter-集成--指纹) |
| BL-B-10 | PrecomputeService — 调用关系创建 | P1 | 1 天 | ✅ | [详情](#bl-b-10-p1-precomputeservice--调用关系创建) |
| BL-B-11 | PrecomputeService — 循环检测 | P2 | 0.5 天 | ✅ | [详情](#bl-b-11-p2-precomputeservice--循环检测) |
| BL-B-12 | PrecomputeService — 权重计算 | P2 | 0.5 天 | ✅ | [详情](#bl-b-12-p2-precomputeservice--权重计算) |
| BL-B-13 | PrecomputeService — 性能监控 | P1 | 0.5 天 | ⏳ | [详情](#bl-b-13-p1-precomputeservice--性能监控) |
| BL-B-14 | PrecomputeService — 并发控制 | P1 | 0.5 天 | ⏳ | [详情](#bl-b-14-p1-precomputeservice--并发控制) |
| **Phase 4** |
| BL-B-15 | Meilisearch SDK — 客户端迁移 | P0 | 1 天 | ⏳ | [详情](#bl-b-15-p0-meilisearch-sdk-040--客户端迁移) |
| BL-B-16 | Meilisearch SDK — 索引设置迁移 | P1 | 0.5 天 | ⏳ | [详情](#bl-b-16-p1-meilisearch-sdk-040--索引设置迁移) |
| BL-B-17 | Meilisearch SDK — 批量操作 | P1 | 0.5 天 | ⏳ | [详情](#bl-b-17-p1-meilisearch-sdk-040--批量操作支持) |
| **Phase 5** |
| BL-B-18 | Schema — 核心表创建 | P0 | 1 天 | ⏳ | [详情](#bl-b-18-p0-schema-v32--核心表创建) |
| BL-B-19 | Schema — ChangeFeed 配置 | P1 | 0.5 天 | ⏳ | [详情](#bl-b-19-p1-schema-v32--changefeed-配置) |
| BL-B-20 | Schema — 辅助表创建 | P1 | 0.5 天 | ⏳ | [详情](#bl-b-20-p1-schema-v32--辅助表创建) |
| BL-B-21 | Schema — 迁移脚本 | P1 | 0.5 天 | ⏳ | [详情](#bl-b-21-p1-schema-v32--迁移脚本) |
| **Phase 6** |
| BL-B-22 | 端口迁移 17999 → 18008 | P0 | 1 天 | ⏳ | [详情](#bl-b-22-p0-端口迁移-17999--18008) |
| BL-B-23 | Docker 多阶段构建优化 | P1 | 0.5 天 | ⏳ | [详情](#bl-b-23-p1-docker-多阶段构建优化) |
| BL-B-24 | docker-compose 健康检查 | P1 | 0.5 天 | ⏳ | [详情](#bl-b-24-p1-docker-compose-健康检查) |
| BL-B-25 | SSL 自动续期 | P2 | 0.5 天 | ⏳ | [详情](#bl-b-25-p2-ssl-自动续期) |
| **Phase 7** |
| BL-B-26 | 单元测试 — WebSocket 模块 | P0 | 1 天 | ⏳ | [详情](#bl-b-26-p0-单元测试--websocket-模块) |
| BL-B-27 | 单元测试 — Precompute 模块 | P0 | 1 天 | ⏳ | [详情](#bl-b-27-p0-单元测试--precompute-模块) |
| BL-B-28 | 集成测试 — WebSocket 端到端 | P1 | 1 天 | ⏳ | [详情](#bl-b-28-p1-集成测试--websocket-端到端) |
| BL-B-29 | 集成测试 — API 端到端 | P1 | 0.5 天 | ⏳ | [详情](#bl-b-29-p1-集成测试--api-端到端) |
| BL-B-30 | 性能基准测试 | P2 | 0.5 天 | ⏳ | [详情](#bl-b-30-p2-性能基准测试) |
| **WebSocket 后续** |
| BL-B-52 | WebSocket — AckManager 集成 | P1 | 0.5 天 | ⏳ | [详情](#bl-b-52-p1-websocket-ackmanager-集成) |
| BL-B-53 | WebSocket — ACK 消息协议定义 | P1 | 0.5 天 | ⏳ | [详情](#bl-b-53-p1-websocket-ack-消息协议定义) |
| BL-B-54 | WebSocket — 消息持久化 | P2 | 1 天 | ⏳ | [详情](#bl-b-54-p2-websocket-消息持久化) |
| BL-B-55 | WebSocket — DiffManager 集成 | P1 | 0.5 天 | ⏳ | [详情](#bl-b-55-p1-websocket-diffmanager-集成) |
| BL-B-56 | WebSocket — LIVE SELECT DIFF 订阅 | P1 | 1 天 | ⏳ | [详情](#bl-b-56-p1-websocket-live-select-diff-订阅) |
| BL-B-57 | WebSocket — DIFF 客户端配置接口 | P1 | 0.5 天 | ⏳ | [详情](#bl-b-57-p1-websocket-diff-客户端配置接口) |
| BL-B-58 | WebSocket — StateRecoveryManager 集成 | P1 | 0.5 天 | ⏳ | [详情](#bl-b-58-p1-websocket-staterecoverymanager-集成) |
| BL-B-59 | WebSocket — 同步丢失消息 (from_offset) | P1 | 1 天 | ⏳ | [详情](#bl-b-59-p1-websocket-同步丢失消息-from_offset) |
| BL-B-60 | WebSocket — 断线重连自动恢复 | P1 | 0.5 天 | ⏳ | [详情](#bl-b-60-p1-websocket-断线重连自动恢复) |
| BL-B-61 | WebSocket — 性能测试实际运行 | P1 | 0.5 天 | ⏳ | [详情](#bl-b-61-p1-websocket-性能测试实际运行) |
| BL-B-62 | WebSocket — CI/CD 性能测试集成 | P2 | 0.5 天 | ⏳ | [详情](#bl-b-62-p2-websocket-cicd-性能测试集成) |
| BL-B-63 | WebSocket — 性能测试套件整合 | P1 | 0.5 天 | ⏳ | [详情](#bl-b-63-p1-websocket-性能测试套件整合) |
| **PrecomputeService 后续** |
| BL-B-64 | PrecomputeService — SurrealDB RELATE 集成 | P1 | 0.5 天 | ⏳ | [详情](#bl-b-64-p1-precomputeservice--surrealdb-relate-集成) |
| BL-B-65 | PrecomputeService — CycleDetector 集成 | P1 | 0.5 天 | ⏳ | [详情](#bl-b-65-p1-precomputeservice--cycledetector-集成) |
| BL-B-66 | PrecomputeService — 循环依赖解决策略 | P2 | 0.5 天 | ⏳ | [详情](#bl-b-66-p2-precomputeservice--循环依赖解决策略) |
| BL-B-67 | PrecomputeService — 权重持久化 | P1 | 0.5 天 | ⏳ | [详情](#bl-b-67-p1-precomputeservice--权重持久化) |
| BL-B-68 | PrecomputeService — WeightCalculator 集成 | P1 | 0.5 天 | ⏳ | [详情](#bl-b-68-p1-precomputeservice--weightcalculator-集成) |
| **文档** |
| BL-CA-43 | 补充 WebSocket 性能测试基准 | P1 | 0.5 天 | ⏳ | [详情](#bl-ca-43-p1-补充-websocket-性能测试基准) |
| BL-CA-44 | 完善 PrecomputeService 关系创建 | P1 | 1 天 | ⏳ | [详情](#bl-ca-44-p1-完善-precomputeservice-关系创建实现) |
| BL-CA-45 | 统一预计算批处理大小参数 | P2 | 0.5 天 | ⏳ | [详情](#bl-ca-45-p2-统一预计算批处理大小参数) |
| BL-CA-46 | 扩充后端实施指南 | P2 | 1 天 | ⏳ | [详情](#bl-ca-46-p2-扩充后端实施指南) |
| BL-CA-47 | 添加 WebSocket 错误处理示例 | P2 | 0.5 天 | ⏳ | [详情](#bl-ca-47-p2-添加-websocket-错误处理示例) |
| BL-CA-48 | 添加 Kubernetes 部署配置 | P2 | 1 天 | ⏳ | [详情](#bl-ca-48-p2-添加-kubernetes-部署配置) |
| BL-CA-49 | 添加数据库 ER 关系图 | P3 | 0.5 天 | ⏳ | [详情](#bl-ca-49-p3-添加数据库-er-关系图) |
| BL-CA-50 | 添加 SSL 自动续期配置 | P3 | 0.5 天 | ⏳ | [详情](#bl-ca-50-p3-添加-ssl-自动续期配置) |

---

## Phase 1: 依赖升级（1 天）

### BL-B-31 [P0] 依赖升级 — pyproject.toml

**目标**  
更新 pyproject.toml 依赖版本，为 v3.2 新功能提供基础支持。

**涉及范围**  
- 文件: `pyproject.toml`
- 依赖项:
  - surrealdb: `>=1.0.0,<2.0.0` → `>=1.0.8,<1.1.0`
  - 新增: meilisearch `>=0.40.0,<0.41.0`
  - 新增: websockets `>=12.0,<13.0`
  - 新增: tree-sitter `>=0.25.0,<0.26.0`
  - 新增: tree-sitter-python `>=0.25.0,<0.26.0`
  - 新增: tree-sitter-javascript `>=0.25.0,<0.26.0`
  - 新增: tree-sitter-typescript `>=0.23.0,<0.24.0`
  - 新增: fast-json-patch `>=1.32`
  - 新增: portalocker `>=2.7`
  - 新增: aiofiles `>=23.0`

**前置依赖**  
无

**完成标准**  
- [ ] pyproject.toml 更新完成
- [ ] `uv pip install` 无错误
- [ ] `uv run python -c "import all_new_deps"` 成功
- [ ] 依赖版本锁定文件更新

**验证方式**  
```bash
uv pip install -e .
uv run python -c "import surrealdb, meilisearch, websockets, tree_sitter"
uv run python -c "import surrealdb; print(surrealdb.__version__)"
```

---

## Phase 2: WebSocket 重写（5.5 天）

### BL-B-1 [P0] WebSocket 可靠连接 — 心跳机制

**目标**  
实现 WebSocket 心跳机制，确保连接存活检测，2 次未响应自动触发重连。

**涉及范围**  
- 文件: `wrapper/src/websocket/reliable_server.py`（新建）
- 文件: `wrapper/src/websocket/heartbeat.py`（新建）
- 类: `HeartbeatManager`
- 方法: `start_heartbeat()`, `stop_heartbeat()`, `check_pong_timeout()`

**前置依赖**  
BL-B-31 依赖升级完成

**完成标准**  
- [x] 每 30s 发送 ping 消息
- [x] 5s 内等待 pong 响应
- [x] 连续 2 次未响应触发 `on_connection_lost`
- [x] 心跳日志记录（DEBUG 级别）
- [x] 可配置参数（interval, timeout, max_missing）

**验证方式**  
```bash
# 运行 WebSocket 心跳测试
uv run pytest tests/test_websocket_heartbeat.py -v
```

**实现结果**  
- ✅ `wrapper/src/websocket/heartbeat.py` - HeartbeatManager 类 (151行)
- ✅ `wrapper/src/websocket/reliable_server.py` - ReliableWebSocketServer 类 (192行)
- ✅ `wrapper/src/websocket/__init__.py` - 模块导出
- ✅ `tests/test_websocket_heartbeat.py` - 15个测试用例，全部通过
- ✅ `wrapper/src/routers/websocket.py` - 迁移到 ReliableWebSocketServer

**测试覆盖**  
- HeartbeatManager 基础功能 (7个测试)
- ReliableWebSocketServer 集成 (6个测试)
- 心跳机制集成 (2个测试)
- 总计: 15/15 测试通过

---

### BL-B-2 [P0] WebSocket 可靠连接 — 指数退避重连

**目标**  
实现指数退避重连机制，避免惊群效应，最大重试 10 次。

**涉及范围**  
- 文件: `wrapper/src/websocket/reconnection.py`（新建）
- 类: `ReconnectionManager`
- 方法: `schedule_reconnect()`, `calculate_delay()`, `reset_counter()`

**前置依赖**  
BL-B-1 心跳机制完成

**完成标准**  
- [x] 指数退避序列: 1→2→4→8→16→32→64→128→256→300s
- [x] 随机抖动: +random.uniform(0, 1)s
- [x] 最大重试: 10 次
- [ ] 重连后恢复 session（留待后续集成）
- [ ] 重连失败进入降级模式（留待后续定义）

**验证方式**  
```bash
uv run pytest tests/test_websocket_reconnection.py -v
```

**实现结果**  
- ✅ `wrapper/src/websocket/reconnection.py` - ReconnectionManager 类 (160行)
- ✅ `tests/test_websocket_reconnection.py` - 16个测试用例，全部通过
- ✅ 指数退避序列: BACKOFF_SEQUENCE = [1, 2, 4, 8, 16, 32, 64, 128, 256, 300]
- ✅ 随机抖动: random.uniform(0, 1)
- ✅ 核心方法: calculate_delay(), schedule_reconnect(), reset_counter(), cancel_reconnect()

**测试覆盖**  
- ReconnectionManager 基础功能 (11个测试)
- 指数退避序列测试 (2个测试)
- 集成测试 (2个测试)
- 总计: 16/16 测试通过

**未结事项**  
- session 恢复需要调用方实现（集成到 ReliableWebSocketServer 时处理）
- 降级模式需要定义具体行为（如切换到轮询模式）

---

### BL-B-3 [P0] WebSocket 可靠连接 — ACK 确认系统

**目标**  
实现消息确认机制，确保消息可靠投递，5s 超时，最多 3 次重试。

**涉及范围**  
- 文件: `wrapper/src/websocket/ack_manager.py`（新建）
- 类: `AckManager`
- 方法: `send_with_ack()`, `handle_ack()`, `retry_message()`

**前置依赖**  
BL-B-1 心跳机制完成

**完成标准**  
- [x] 消息发送后启动 5s 超时计时器
- [x] 收到 ACK 后清除超时
- [x] 超时后自动重试（最多 3 次）
- [x] 达到最大重试次数后 reject
- [x] ACK 消息格式: `{"type": "ack", "_ackId": "..."}`

**验证方式**  
```bash
uv run pytest tests/test_websocket_ack.py -v
```

**实现结果**  
- ✅ `wrapper/src/websocket/ack_manager.py` - AckManager 类 (166行)
- ✅ `tests/test_websocket_ack.py` - 12个测试用例，全部通过
- ✅ 核心方法: send_with_ack(), handle_ack(), is_pending(), get_retry_count()
- ✅ 超时重试机制，最多3次重试
- ✅ UUID 生成 ackId，自动添加到消息

**测试覆盖**  
- AckManager 基础功能 (8个测试)
- 并发测试 (1个测试)
- 配置测试 (2个测试)
- 总计: 12/12 测试通过

**未结事项（已规划到后续任务）**  
- BL-B-52: 与 ReliableWebSocketServer 集成（将 AckManager 集成到 WebSocket 服务器）
- BL-B-53: ACK 消息协议定义（定义客户端如何发送 ACK 消息）
- BL-B-54: 消息持久化（消息队列持久化实现）

---

### BL-B-4 [P1] WebSocket 可靠连接 — DIFF 模式

**目标**  
实现 DIFF 增量同步模式，使用 JSON Patch（RFC 6902），减少 90% 数据传输。

**涉及范围**  
- 文件: `wrapper/src/websocket/diff_manager.py`（新建）
- 文件: `wrapper/src/websocket/patch_generator.py`（新建）
- 类: `DiffManager`, `PatchGenerator`

**前置依赖**  
BL-B-3 ACK 系统完成

**完成标准**  
- [x] 生成 RFC 6902 标准 JSON Patch
- [x] Patch 操作: replace/add/remove
- [x] 带宽节省计算
- [x] 客户端可配置 diff/full 模式
- [ ] 支持 `LIVE SELECT DIFF` 订阅（已规划到 BL-B-56）

**验证方式**  
```bash
uv run pytest tests/test_websocket_diff.py -v
```

**实现结果**  
- ✅ `wrapper/src/websocket/patch_generator.py` - PatchGenerator 类 (186行)
- ✅ `wrapper/src/websocket/diff_manager.py` - DiffManager 类 (212行)
- ✅ `tests/test_websocket_diff.py` - 23个测试用例，全部通过
- ✅ RFC 6902 JSON Patch 生成 (replace/add/remove)
- ✅ Patch 应用和验证
- ✅ 带宽节省计算
- ✅ diff/full 模式切换

**测试覆盖**  
- PatchGenerator 测试 (10个测试)
- DiffManager 测试 (11个测试)
- 带宽节省测试 (2个测试)
- 总计: 23/23 测试通过

**未结事项（已规划到后续任务）**  
- BL-B-55: DiffManager 集成到 ReliableWebSocketServer
- BL-B-56: LIVE SELECT DIFF 订阅实现
- BL-B-57: DIFF 客户端配置接口

---

### BL-B-5 [P0] WebSocket 可靠连接 — 状态恢复

**目标**  
实现连接断开后状态恢复，支持 session + offset 机制。

**涉及范围**  
- 文件: `wrapper/src/websocket/state_recovery.py`（新建）
- 类: `StateRecoveryManager`
- 存储: `.opencode/ws-state.json`

**前置依赖**  
BL-B-2 重连机制完成

**完成标准**  
- [x] Session ID 生成: `sess-{timestamp}-{uuid[:9]}`
- [x] Offset 持久化到文件
- [x] 断线重连后恢复 session（核心功能实现，集成待后续）
- [x] 状态文件 7 天 TTL 清理
- [ ] 同步丢失消息（from_offset）（已规划到 BL-B-59）

**验证方式**  
```bash
uv run pytest tests/test_websocket_state_recovery.py -v
```

**实现结果**  
- ✅ `wrapper/src/websocket/state_recovery.py` - StateRecoveryManager 类 (254行)
- ✅ `tests/test_websocket_state_recovery.py` - 15个测试用例，全部通过
- ✅ Session ID 生成: `sess-{timestamp}-{uuid[:9]}`
- ✅ Offset 持久化到 `.opencode/ws-state.json`
- ✅ 状态保存/恢复/删除
- ✅ TTL 清理（7天过期）

**测试覆盖**  
- StateRecoveryManager 基础功能 (11个测试)
- TTL 清理测试 (2个测试)
- 持久化测试 (2个测试)
- 总计: 15/15 测试通过

**未结事项（已规划到后续任务）**  
- BL-B-58: StateRecoveryManager 集成到 ReliableWebSocketServer
- BL-B-59: 同步丢失消息 (from_offset) 实现
- BL-B-60: 断线重连自动恢复（结合 ReconnectionManager）

---

### BL-B-6 [P1] WebSocket 性能 — 并发连接测试

**目标**  
验证 WebSocket 服务端支持 ≥1000 并发连接。

**涉及范围**  
- 文件: `tests/performance/test_websocket_concurrent.py`（新建）
- 工具: `locust` 或 `asyncio` 并发测试

**前置依赖**  
BL-B-1~B-5 WebSocket 核心功能完成

**完成标准**  
- [x] 支持 1000+ 并发连接（测试脚本实现）
- [x] 内存使用监控
- [x] CPU 使用监控
- [x] 连接成功率统计
- [ ] 实际性能测试（已规划到 BL-B-61）

**验证方式**  
```bash
uv run python tests/performance/test_websocket_concurrent.py --help
```

**实现结果**  
- ✅ `tests/performance/test_websocket_concurrent.py` - 并发测试脚本 (354行)
- ✅ `WebSocketLoadClient` 类 - 模拟 WebSocket 客户端
- ✅ `PerformanceMonitor` 类 - 监控内存和 CPU
- ✅ `WebSocketConcurrentTest` 类 - 并发测试主类
- ✅ 命令行参数支持：`--clients`, `--duration`, `--url`
- ✅ 详细测试报告生成

**测试功能**  
- 创建 N 个并发 WebSocket 连接
- 定期发送心跳消息
- 监控内存和 CPU 使用
- 统计连接成功率
- 生成详细测试报告

**未结事项（已规划到后续任务）**  
- BL-B-61: 实际性能测试（需要启动 WebSocket 服务器）
- BL-B-62: CI/CD 性能测试集成

---

### BL-B-7 [P1] WebSocket 性能 — 消息延迟测试

**目标**  
验证 WebSocket 消息延迟 p99 < 100ms。

**涉及范围**  
- 文件: `tests/performance/test_websocket_latency.py`（新建）
- 指标: p50/p95/p99 延迟、吞吐量

**前置依赖**  
BL-B-1~B-5 WebSocket 核心功能完成

**完成标准**  
- [x] p99 延迟测量实现
- [x] p95 延迟测量实现
- [x] p50 延迟测量实现
- [x] 吞吐量测量实现
- [x] 测试报告生成
- [ ] 实际性能测试（已规划到 BL-B-61）

**验证方式**  
```bash
uv run python tests/performance/test_websocket_latency.py --help
```

**实现结果**  
- ✅ `tests/performance/test_websocket_latency.py` - 延迟测试脚本 (298行)
- ✅ `LatencySample` 类 - 延迟样本
- ✅ `LatencyMetrics` 类 - 延迟指标（含 p50/p95/p99 计算）
- ✅ `LatencyTestClient` 类 - 延迟测试客户端
- ✅ `WebSocketLatencyTest` 类 - 延迟测试主类
- ✅ 命令行参数支持：`--clients`, `--messages`, `--delay`, `--url`

**测试功能**  
- 测量消息往返延迟（RTT）
- 计算 p50/p95/p99 百分位数
- 计算吞吐量（msg/s）
- 生成详细测试报告

**未结事项（已规划到后续任务）**  
- BL-B-61: 实际性能测试（与 BL-B-6 一起运行）

---

### BL-B-51 [P1] WebSocket 可靠性 — 心跳成功率 ≥99% 验证

**目标**  
验证 WebSocket 心跳成功率 ≥99%。

**涉及范围**  
- 文件: `tests/performance/test_websocket_reliability.py`（新建）
- 指标: 心跳成功率、丢包率

**前置依赖**  
BL-B-1 心跳机制完成

**完成标准**  
- [x] 心跳成功率测量实现
- [x] 丢包率测量实现
- [x] 长时间运行支持（可配置）
- [x] 定期统计输出
- [x] 测试报告生成
- [ ] 实际 24 小时运行（已规划到 BL-B-61）

**验证方式**  
```bash
uv run python tests/performance/test_websocket_reliability.py --help
```

**实现结果**  
- ✅ `tests/performance/test_websocket_reliability.py` - 可靠性测试脚本 (320行)
- ✅ `ReliabilityStats` 类 - 可靠性统计（成功率、丢包率）
- ✅ `ReliabilityTestClient` 类 - 可靠性测试客户端
- ✅ `WebSocketReliabilityTest` 类 - 可靠性测试主类
- ✅ 命令行参数支持：`--duration`, `--interval`, `--url`
- ✅ 信号中断支持（Ctrl+C）

**测试功能**  
- 长时间运行的心跳测试
- 统计心跳成功/失败次数
- 计算成功率（≥99%）
- 计算丢包率（<1%）
- 定期输出统计（每分钟）
- 生成详细测试报告

**未结事项（已规划到后续任务）**  
- BL-B-61: 实际性能测试（与 BL-B-6、BL-B-7 一起运行）
- BL-B-63: 性能测试套件整合

---

## Phase 3: PrecomputeService（7 天）

### BL-B-8 [P0] PrecomputeService — 基础架构

**目标**  
创建 PrecomputeService 服务骨架，实现服务化架构，支持 tenant 隔离。

**涉及范围**  
- 文件: `wrapper/src/services/precompute.py`（新建）
- 类: `PrecomputeService`
- 方法: `__init__()`, `start()`, `stop()`, `process_batch()`

**前置依赖**  
BL-B-31 依赖升级完成

**完成标准**  
- [x] PrecomputeService 类实现
- [x] 支持 tenant_id 隔离
- [x] 支持 DB 连接注入
- [x] 支持启动/停止生命周期
- [x] 基础日志记录

**验证方式**  
```bash
uv run pytest tests/test_precompute_service.py -v
```

**实现结果**  
- ✅ `wrapper/src/services/__init__.py` - 模块导出
- ✅ `wrapper/src/services/precompute.py` - PrecomputeService 类 (154行)
- ✅ `tests/test_precompute_service.py` - 11个测试用例，全部通过
- ✅ 支持 tenant_id 隔离
- ✅ 支持 DB 连接注入
- ✅ 生命周期管理（start/stop）
- ✅ 健康检查接口

**测试覆盖**  
- PrecomputeService 基础功能 (9个测试)
- Tenant 隔离测试 (1个测试)
- 生命周期测试 (1个测试)
- 总计: 11/11 测试通过

**未结事项（已规划到后续任务）**  
- BL-B-9: tree-sitter 集成 + 指纹
- BL-B-10: 调用关系创建
- BL-B-11: 循环检测
- BL-B-12: 权重计算
- BL-B-13: 性能监控
- BL-B-14: 并发控制

---

### BL-B-9 [P0] PrecomputeService — tree-sitter 集成 + 指纹

**目标**  
集成 tree-sitter 进行代码解析，实现 SHA256 指纹计算，支持增量分析。

**涉及范围**  
- 文件: `wrapper/src/services/code_parser.py`（新建）
- 文件: `wrapper/src/services/fingerprint.py`（新建）
- 类: `CodeParser`, `FingerprintManager`

**前置依赖**  
BL-B-8 基础架构完成

**完成标准**  
- [x] 支持 Python/JavaScript/TypeScript 解析
- [x] SHA256 指纹计算
- [x] 变更检测（指纹比对）
- [ ] 指纹持久化到 DB（后续任务）
- [ ] 未变更文件跳过分析（集成到 PrecomputeService 时实现）

**验证方式**  
```bash
uv run pytest tests/test_fingerprint.py -v
```

**实现结果**  
- ✅ `wrapper/src/services/fingerprint.py` - FingerprintManager 类 (135行)
- ✅ `wrapper/src/services/code_parser.py` - CodeParser 类 (236行)
- ✅ `tests/test_fingerprint.py` - 16个测试用例，全部通过
- ✅ SHA256 指纹计算
- ✅ 变更检测（has_changed）
- ✅ 支持 Python/JavaScript/TypeScript 解析
- ✅ 符号提取（函数、类）

**测试覆盖**  
- FingerprintManager 基础功能 (15个测试)
- 集成测试 (1个测试)
- 总计: 16/16 测试通过

**未结事项（已规划到后续任务）**  
- BL-B-10: 调用关系创建
- BL-B-11: 循环检测
- BL-B-12: 权重计算
- BL-B-13: 性能监控
- BL-B-14: 并发控制

---

### BL-B-10 [P1] PrecomputeService — 调用关系创建

**目标**  
从 AST 中提取函数调用关系，自动创建 RELATE 关系。

**涉及范围**  
- 文件: `wrapper/src/services/relation_builder.py`（新建）
- 类: `RelationBuilder`
- 方法: `extract_calls()`, `create_relations()`, `batch_relate()`

**前置依赖**  
BL-B-9 tree-sitter 集成完成

**完成标准**  
- [x] 提取函数调用关系
- [x] 批量创建关系（100 条/批）
- [x] 自调用过滤（caller != callee）
- [x] 关系权重计算（基础）
- [ ] 创建 atom → atom RELATE（已规划到 BL-B-64）

**验证方式**  
```bash
uv run pytest tests/test_relation_builder.py -v
```

**实现结果**  
- ✅ `wrapper/src/services/relation_builder.py` - RelationBuilder 类 (241行)
- ✅ `tests/test_relation_builder.py` - 7个测试用例，全部通过
- ✅ `CallRelation` 数据类 - 调用关系
- ✅ `extract_calls(ast, file_path)` - 从 AST 提取调用关系
- ✅ `create_relations(relations)` - 创建关系（过滤自调用）
- ✅ `batch_relate(relations, batch_size)` - 批量创建关系
- ✅ `_calculate_weight(caller, callee, file_path)` - 权重计算
- ✅ 支持 Mock 模式（无 DB）

**测试覆盖**  
- RelationBuilder 基础功能 (7个测试)
- 总计: 7/7 测试通过

**未结事项（已规划到后续任务）**  
- BL-B-64: SurrealDB RELATE 集成
- BL-B-11: 循环检测
- BL-B-12: 权重计算
- BL-B-13: 性能监控
- BL-B-14: 并发控制

---

### BL-B-11 [P2] PrecomputeService — 循环检测

**目标**  
检测代码中的循环依赖（circular dependencies）。

**涉及范围**  
- 文件: `wrapper/src/services/cycle_detector.py`（新建）
- 类: `CycleDetector`
- 算法: DFS（深度优先搜索）

**前置依赖**  
BL-B-10 调用关系创建完成

**完成标准**  
- [x] DFS 算法实现
- [x] 检测循环调用链
- [x] 记录循环路径
- [x] 日志输出警告
- [x] 时间复杂度 O(V+E)

**验证方式**  
```bash
uv run pytest tests/test_cycle_detector.py -v
```

**实现结果**  
- ✅ `wrapper/src/services/cycle_detector.py` - CycleDetector 类 (191行)
- ✅ `tests/test_cycle_detector.py` - 12个测试用例，全部通过
- ✅ `Cycle` 数据类 - 循环信息（path, length）
- ✅ `detect_cycles(relations)` - 检测循环
- ✅ `_build_graph(relations)` - 构建有向图
- ✅ `_dfs(node, graph, visited, rec_stack, path)` - 深度优先搜索
- ✅ `_extract_cycle(path, start_node)` - 提取循环路径
- ✅ 使用三色标记法（白/灰/黑）

**测试覆盖**  
- CycleDetector 基础功能 (10个测试)
- 性能测试 (2个测试)
- 总计: 12/12 测试通过

**未结事项（已规划到后续任务）**  
- BL-B-65: CycleDetector 集成到 RelationBuilder
- BL-B-66: 循环依赖解决策略

---

### BL-B-12 [P2] PrecomputeService — 权重计算

**目标**  
计算调用关系的权重，用于图遍历优先级。

**涉及范围**  
- 文件: `wrapper/src/services/weight_calculator.py`（新建）
- 类: `WeightCalculator`
- 因素: 调用频率、复杂度、参数数量、跨文件

**前置依赖**  
BL-B-10 调用关系创建完成

**完成标准**  
- [x] 权重因子定义
- [x] 权重计算公式
- [x] 归一化处理
- [ ] 权重持久化（BL-B-67 后续任务）

**验证方式**  
```python
def test_calculate_weight():
    wc = WeightCalculator()
    weight = wc.calculate_weight(call_frequency=10, complexity=5, param_count=3, is_cross_file=True)
    assert 0 <= weight <= 1
```

**实现结果**  
- ✅ `wrapper/src/services/weight_calculator.py` (195行)
- ✅ WeightFactors 数据类定义
- ✅ 权重计算公式: `base + frequency_factor + complexity_factor + param_factor + cross_file_factor`
- ✅ 归一化到 [0, 1] 范围
- ✅ 18个测试全部通过
- ⏳ 权重持久化到 DB（BL-B-67 后续任务）
- ⏳ 与 RelationBuilder 集成（BL-B-68 后续任务）

---

### BL-B-13 [P1] PrecomputeService — 性能监控

**目标**  
监控 PrecomputeService 性能，记录耗时、内存使用。

**涉及范围**  
- 文件: `wrapper/src/services/performance_monitor.py`（新建）
- 类: `PerformanceMonitor`
- 指标: parse_time, analysis_time, memory_usage

**前置依赖**  
BL-B-8 基础架构完成

**完成标准**  
- [ ] 性能指标收集
- [ ] 内存监控
- [ ] 日志记录
- [ ] 性能报告生成

**验证方式**  
```python
async def test_performance_monitor():
    pm = PerformanceMonitor()
    with pm.monitor("parse"):
        await parse_code(content)
    metrics = pm.get_metrics()
    assert "parse_time_ms" in metrics
```

---

### BL-B-14 [P1] PrecomputeService — 并发控制

**目标**  
实现并发控制，防止同文件重复处理，限制并发数。

**涉及范围**  
- 文件: `wrapper/src/services/concurrency_control.py`（新建）
- 类: `ConcurrencyControl`
- 机制: Semaphore(5) + processing Set

**前置依赖**  
BL-B-8 基础架构完成

**完成标准**  
- [ ] Semaphore(5) 并发限制
- [ ] processing Set 去重
- [ ] 队列机制
- [ ] 超时处理

**验证方式**  
```python
async def test_concurrency_limit():
    cc = ConcurrencyControl(max_concurrent=5)
    tasks = [cc.process(f"file_{i}") for i in range(10)]
    results = await asyncio.gather(*tasks)
    assert cc.max_concurrent_reached <= 5
```

---

## Phase 4: Meilisearch SDK 升级（2 天）

### BL-B-15 [P0] Meilisearch SDK 0.40 — 客户端迁移

**目标**  
将 Meilisearch 客户端从 httpx REST 调用迁移到官方 SDK 0.40。

**涉及范围**  
- 文件: `wrapper/src/utils/meili_client.py`（修改）
- 依赖: `meilisearch>=0.40.0,<0.41.0`

**前置依赖**  
BL-B-31 依赖升级完成

**完成标准**  
- [ ] 替换 httpx 为 meilisearch SDK
- [ ] 更新所有 API 调用
- [ ] 错误处理适配
- [ ] 配置迁移

**验证方式**  
```python
async def test_meilisearch_sdk():
    client = MeiliClient()
    await client.connect()
    result = await client.search("test")
    assert "hits" in result
```

---

### BL-B-16 [P1] Meilisearch SDK 0.40 — 索引设置迁移

**目标**  
迁移 Meilisearch 索引设置到新 SDK。

**涉及范围**  
- 文件: `wrapper/src/utils/meili_client.py`（修改）
- 索引: `memories`, `code_search_index`

**前置依赖**  
BL-B-15 客户端迁移完成

**完成标准**  
- [ ] 索引设置迁移
- [ ] 字段映射更新
- [ ] 搜索配置更新

**验证方式**  
```python
async def test_index_settings():
    client = MeiliClient()
    settings = await client.get_settings("memories")
    assert "filterableAttributes" in settings
```

---

### BL-B-17 [P1] Meilisearch SDK 0.40 — 批量操作支持

**目标**  
实现批量操作支持，提升导入性能。

**涉及范围**  
- 文件: `wrapper/src/utils/meili_client.py`（修改）
- 方法: `batch_add_documents()`, `batch_update_documents()`

**前置依赖**  
BL-B-15 客户端迁移完成

**完成标准**  
- [ ] 批量添加文档
- [ ] 批量更新文档
- [ ] 批量删除文档
- [ ] 批处理大小 100 条

**验证方式**  
```python
async def test_batch_operations():
    client = MeiliClient()
    documents = [{"id": i} for i in range(100)]
    result = await client.batch_add_documents("memories", documents)
    assert result["processed"] == 100
```

---

## Phase 5: SurrealDB Schema 升级（2.5 天）

### BL-B-18 [P0] Schema v3.2 — 核心表创建

**目标**  
创建 v3.2 核心表：atom, entity, reference。

**涉及范围**  
- 文件: `scripts/init_surrealdb_v3.2.surql`（新建）
- 表: `atom`, `entity`, `reference`

**前置依赖**  
SurrealDB 1.0.8 已安装

**完成标准**  
- [ ] atom 表创建
- [ ] entity 表创建
- [ ] reference 表创建
- [ ] tenant_id 预留字段
- [ ] 索引创建

**验证方式**  
```sql
INFO FOR DB;
-- 应显示 atom, entity, reference 表
```

---

### BL-B-19 [P1] Schema v3.2 — ChangeFeed 配置

**目标**  
配置 SurrealDB ChangeFeed，支持实时变更通知。

**涉及范围**  
- 文件: `scripts/init_surrealdb_v3.2.surql`（修改）
- 配置: `CHANGE FEED 7d ON TABLE ...`

**前置依赖**  
BL-B-18 核心表创建完成

**完成标准**  
- [ ] ChangeFeed 启用
- [ ] 7 天 TTL 配置
- [ ] 支持 atom/entity/reference 表

**验证方式**  
```sql
LIVE SELECT * FROM atom;
-- 应返回 query UUID
```

---

### BL-B-20 [P1] Schema v3.2 — 辅助表创建

**目标**  
创建辅助表：performance_log, session_state。

**涉及范围**  
- 文件: `scripts/init_surrealdb_v3.2.surql`（修改）
- 表: `performance_log`, `session_state`

**前置依赖**  
BL-B-18 核心表创建完成

**完成标准**  
- [ ] performance_log 表创建
- [ ] session_state 表创建
- [ ] 索引创建

**验证方式**  
```sql
INFO FOR DB;
-- 应显示所有表
```

---

### BL-B-21 [P1] Schema v3.2 — 迁移脚本

**目标**  
创建数据迁移脚本，从 v2.x 迁移到 v3.2。

**涉及范围**  
- 文件: `scripts/migrate_v2_to_v3.2.py`（新建）
- 迁移: memory → atom/entity/reference

**前置依赖**  
BL-B-18~B-20 Schema 创建完成

**完成标准**  
- [ ] 数据迁移脚本
- [ ] 数据验证
- [ ] 回滚机制
- [ ] 迁移日志

**验证方式**  
```bash
uv run python scripts/migrate_v2_to_v3.2.py --dry-run
uv run python scripts/migrate_v2_to_v3.2.py --execute
```

---

## Phase 6: 端口迁移（2.5 天）

### BL-B-22 [P0] 端口迁移 17999 → 18008

**目标**  
将服务端口从 17999 迁移到 18008，支持双端口并行期。

**涉及范围**  
- 文件: `wrapper/src/config.py`（修改）
- 配置: 端口配置更新

**前置依赖**  
无

**完成标准**  
- [ ] 默认端口改为 18008
- [ ] 双端口并行支持（1-2 周）
- [ ] 环境变量覆盖支持
- [ ] 文档更新

**验证方式**  
```bash
curl http://localhost:18008/health
curl http://localhost:17999/health  # 并行期
```

---

### BL-B-23 [P1] Docker 多阶段构建优化

**目标**  
优化 Docker 镜像构建，使用多阶段构建减少镜像体积。

**涉及范围**  
- 文件: `Dockerfile`（修改）
- 优化: 多阶段构建、缓存优化

**前置依赖**  
无

**完成标准**  
- [ ] 多阶段构建 Dockerfile
- [ ] 镜像体积减少 50%+
- [ ] 构建时间减少 30%+

**验证方式**  
```bash
docker build -t embedding-service:v3.2 .
docker images | grep embedding-service
```

---

### BL-B-24 [P1] docker-compose 健康检查

**目标**  
添加 docker-compose 健康检查配置。

**涉及范围**  
- 文件: `docker-compose.yml`（修改）
- 配置: healthcheck

**前置依赖**  
BL-B-22 端口迁移完成

**完成标准**  
- [ ] healthcheck 配置
- [ ] 依赖服务启动顺序
- [ ] 自动重启策略

**验证方式**  
```bash
docker-compose up -d
docker-compose ps
# 应显示 healthy
```

---

### BL-B-25 [P2] SSL 自动续期

**目标**  
配置 SSL 证书自动续期（Certbot）。

**涉及范围**  
- 文件: `docker-compose.yml`（修改）
- 配置: Certbot 容器

**前置依赖**  
域名已配置

**完成标准**  
- [ ] Certbot 配置
- [ ] 自动续期脚本
- [ ] 证书验证

**验证方式**  
```bash
openssl s_client -connect api.example.com:443
```

---

## Phase 7: 测试（4.5 天）

### BL-B-26 [P0] 单元测试 — WebSocket 模块

**目标**  
为 WebSocket 模块编写单元测试，覆盖率 ≥80%。

**涉及范围**  
- 文件: `tests/test_websocket_*.py`（新建）
- 覆盖: heartbeat, ack, reconnection, diff, state_recovery

**前置依赖**  
BL-B-1~B-5 WebSocket 实现完成

**完成标准**  
- [ ] 单元测试覆盖率 ≥80%
- [ ] 所有关键路径测试
- [ ] Mock 外部依赖

**验证方式**  
```bash
uv run pytest tests/test_websocket_*.py --cov=wrapper/src/websocket --cov-report=html
```

---

### BL-B-27 [P0] 单元测试 — Precompute 模块

**目标**  
为 Precompute 模块编写单元测试，覆盖率 ≥80%。

**涉及范围**  
- 文件: `tests/test_precompute_*.py`（新建）
- 覆盖: parser, fingerprint, relations, cycles, weights

**前置依赖**  
BL-B-8~B-14 Precompute 实现完成

**完成标准**  
- [ ] 单元测试覆盖率 ≥80%
- [ ] 所有关键路径测试
- [ ] Mock 外部依赖

**验证方式**  
```bash
uv run pytest tests/test_precompute_*.py --cov=wrapper/src/services --cov-report=html
```

---

### BL-B-28 [P1] 集成测试 — WebSocket 端到端

**目标**  
编写 WebSocket 端到端集成测试。

**涉及范围**  
- 文件: `tests/integration/test_websocket_e2e.py`（新建）
- 场景: 连接、心跳、ACK、重连、DIFF

**前置依赖**  
BL-B-26 单元测试完成

**完成标准**  
- [ ] 端到端测试通过
- [ ] 真实服务测试
- [ ] 性能基准测试

**验证方式**  
```bash
uv run pytest tests/integration/test_websocket_e2e.py -v
```

---

### BL-B-29 [P1] 集成测试 — API 端到端

**目标**  
编写 API 端到端集成测试。

**涉及范围**  
- 文件: `tests/integration/test_api_e2e.py`（新建）
- 场景: Precompute API, Memory API, Search API

**前置依赖**  
BL-B-27 单元测试完成

**完成标准**  
- [ ] 端到端测试通过
- [ ] 真实服务测试
- [ ] 数据一致性验证

**验证方式**  
```bash
uv run pytest tests/integration/test_api_e2e.py -v
```

---

### BL-B-30 [P2] 性能基准测试

**目标**  
建立性能基准，记录关键指标。

**涉及范围**  
- 文件: `tests/performance/benchmark.py`（新建）
- 指标: 延迟、吞吐量、并发、内存

**前置依赖**  
BL-B-28~B-29 集成测试完成

**完成标准**  
- [ ] 性能基准建立
- [ ] 基准报告生成
- [ ] 性能回归检测

**验证方式**  
```bash
uv run python tests/performance/benchmark.py --report
```

---

## 文档完善（5 天）

### BL-CA-43 [P1] 补充 WebSocket 性能测试基准

**目标**  
补充 WebSocket 性能测试基准文档。

**涉及范围**  
- 文件: `docs/v3.2/BACKEND-v3.2-WEBSOCKET.md`（补充）

**前置依赖**  
BL-B-6~B-7 性能测试完成

**完成标准**  
- [ ] 性能指标文档
- [ ] 测试方法说明
- [ ] 基准数据记录

**验证方式**  
文档评审通过

---

### BL-CA-44 [P1] 完善 PrecomputeService 关系创建实现

**目标**  
完善 PrecomputeService 关系创建实现文档。

**涉及范围**  
- 文件: `docs/v3.2/BACKEND-v3.2-PRECOMPUTE.md`（补充）

**前置依赖**  
BL-B-10~B-12 实现完成

**完成标准**  
- [ ] 关系创建算法文档
- [ ] 权重计算说明
- [ ] 循环检测算法

**验证方式**  
文档评审通过

---

### BL-CA-45 [P2] 统一预计算批处理大小参数

**目标**  
统一预计算批处理大小参数文档。

**涉及范围**  
- 文件: `docs/v3.2/BACKEND-v3.2-PRECOMPUTE.md`（补充）

**前置依赖**  
BL-B-8 基础架构完成

**完成标准**  
- [ ] 批处理参数统一
- [ ] 文档更新
- [ ] 配置说明

**验证方式**  
文档评审通过

---

### BL-CA-46 [P2] 扩充后端实施指南

**目标**  
扩充后端实施指南文档。

**涉及范围**  
- 文件: `docs/v3.2/BACKEND-v3.2-IMPLEMENTATION.md`（扩充）

**前置依赖**  
Phase 2-3 开发完成

**完成标准**  
- [ ] 详细实施步骤
- [ ] 最佳实践总结
- [ ] FAQ 整理

**验证方式**  
文档评审通过

---

### BL-CA-47 [P2] 添加 WebSocket 错误处理示例

**目标**  
添加 WebSocket 错误处理示例代码。

**涉及范围**  
- 文件: `docs/v3.2/BACKEND-v3.2-WEBSOCKET.md`（补充）

**前置依赖**  
BL-B-1~B-5 实现完成

**完成标准**  
- [ ] 错误码定义
- [ ] 处理示例代码
- [ ] 故障排查指南

**验证方式**  
文档评审通过

---

### BL-CA-48 [P2] 添加 Kubernetes 部署配置

**目标**  
添加 Kubernetes 部署配置。

**涉及范围**  
- 文件: `k8s/`（新建目录）

**前置依赖**  
BL-B-22~B-25 部署配置完成

**完成标准**  
- [ ] Kubernetes 配置
- [ ] Helm chart（可选）
- [ ] 部署文档

**验证方式**  
```bash
kubectl apply -f k8s/
kubectl get pods
```

---

### BL-CA-49 [P3] 添加数据库 ER 关系图

**目标**  
添加数据库 ER 关系图。

**涉及范围**  
- 文件: `docs/v3.2/DATABASE-v3.2-ER.md`（新建）

**前置依赖**  
BL-B-18~B-21 Schema 完成

**完成标准**  
- [ ] ER 图绘制
- [ ] 关系说明
- [ ] 文档集成

**验证方式**  
文档评审通过

---

### BL-CA-50 [P3] 添加 SSL 自动续期配置

**目标**  
添加 SSL 自动续期配置文档。

**涉及范围**  
- 文件: `docs/v3.2/DEPLOYMENT-v3.2.md`（补充）

**前置依赖**  
BL-B-25 SSL 配置完成

**完成标准**  
- [ ] Certbot 配置说明
- [ ] 自动续期脚本
- [ ] 验证方法

**验证方式**  
文档评审通过

---

### BL-B-52 [P1] WebSocket — AckManager 集成

**目标**  
将 AckManager 集成到 ReliableWebSocketServer，实现消息确认机制。

**涉及范围**  
- 文件: `wrapper/src/websocket/reliable_server.py`（修改）
- 集成: AckManager 到 ReliableWebSocketServer

**前置依赖**  
BL-B-3 ACK 系统完成

**完成标准**  
- [ ] ReliableWebSocketServer 初始化时创建 AckManager
- [ ] 发送消息时调用 ack_manager.send_with_ack()
- [ ] 收到客户端 ACK 消息时调用 ack_manager.handle_ack()
- [ ] 消息发送失败时自动重试

**验证方式**  
```bash
uv run pytest tests/test_websocket_integration.py -v
```

---

### BL-B-53 [P1] WebSocket — ACK 消息协议定义

**目标**  
定义客户端如何发送 ACK 消息的协议规范。

**涉及范围**  
- 文件: `docs/v3.2/WEBSOCKET-v3.2-PROTOCOL.md`（新建）

**前置依赖**  
BL-B-52 AckManager 集成完成

**完成标准**  
- [ ] ACK 消息格式定义
- [ ] 客户端 ACK 发送时机说明
- [ ] 服务端 ACK 处理流程
- [ ] 错误处理规范

**验证方式**  
文档评审通过

---

### BL-B-54 [P2] WebSocket — 消息持久化

**目标**  
实现消息队列持久化，确保消息不丢失。

**涉及范围**  
- 文件: `wrapper/src/websocket/persistent_queue.py`（新建）

**前置依赖**  
BL-B-53 ACK 消息协议定义完成

**完成标准**  
- [ ] 消息队列持久化存储
- [ ] 服务重启后恢复未确认消息
- [ ] 消息过期清理机制

**验证方式**  
```bash
uv run pytest tests/test_websocket_persistent.py -v
```

---

### BL-B-55 [P1] WebSocket — DiffManager 集成

**目标**  
将 DiffManager 集成到 ReliableWebSocketServer，实现增量同步。

**涉及范围**  
- 文件: `wrapper/src/websocket/reliable_server.py`（修改）
- 集成: DiffManager 到 ReliableWebSocketServer

**前置依赖**  
BL-B-4 DIFF 模式完成

**完成标准**  
- [ ] ReliableWebSocketServer 初始化时创建 DiffManager
- [ ] 发送消息时根据配置选择 diff/full 模式
- [ ] 缓存消息状态用于生成 diff
- [ ] 支持客户端切换 diff/full 模式

**验证方式**  
```bash
uv run pytest tests/test_websocket_diff_integration.py -v
```

---

### BL-B-56 [P1] WebSocket — LIVE SELECT DIFF 订阅

**目标**  
实现 `LIVE SELECT DIFF` 订阅，支持 SurrealDB 变更通知的增量同步。

**涉及范围**  
- 文件: `wrapper/src/websocket/live_diff_handler.py`（新建）

**前置依赖**  
BL-B-55 DiffManager 集成完成

**完成标准**  
- [ ] 监听 SurrealDB LIVE SELECT 变更
- [ ] 将变更转换为 JSON Patch
- [ ] 发送 diff 消息到客户端
- [ ] 支持变更合并（减少消息数量）

**验证方式**  
```bash
uv run pytest tests/test_websocket_live_diff.py -v
```

---

### BL-B-57 [P1] WebSocket — DIFF 客户端配置接口

**目标**  
提供客户端配置接口，允许客户端选择 diff/full 模式。

**涉及范围**  
- 文件: `wrapper/src/routers/websocket.py`（修改）
- 文件: `docs/v3.2/WEBSOCKET-v3.2-PROTOCOL.md`（更新）

**前置依赖**  
BL-B-56 LIVE SELECT DIFF 订阅完成

**完成标准**  
- [ ] WebSocket 连接参数支持 `mode=diff|full`
- [ ] 动态切换模式 API
- [ ] 客户端配置文档
- [ ] 向后兼容（默认 full 模式）

**验证方式**  
```bash
uv run pytest tests/test_websocket_client_config.py -v
```

---

### BL-B-58 [P1] WebSocket — StateRecoveryManager 集成

**目标**  
将 StateRecoveryManager 集成到 ReliableWebSocketServer，实现状态恢复。

**涉及范围**  
- 文件: `wrapper/src/websocket/reliable_server.py`（修改）

**前置依赖**  
BL-B-5 状态恢复完成

**完成标准**  
- [ ] ReliableWebSocketServer 初始化时创建 StateRecoveryManager
- [ ] 连接建立时恢复 session
- [ ] 消息发送时更新 offset
- [ ] 连接断开时保存状态

**验证方式**  
```bash
uv run pytest tests/test_websocket_state_integration.py -v
```

---

### BL-B-59 [P1] WebSocket — 同步丢失消息 (from_offset)

**目标**  
实现同步丢失消息功能，支持从指定 offset 恢复消息。

**涉及范围**  
- 文件: `wrapper/src/websocket/message_queue.py`（新建）
- 或: 集成 SurrealDB 消息历史

**前置依赖**  
BL-B-58 StateRecoveryManager 集成完成

**完成标准**  
- [ ] 消息队列持久化存储
- [ ] 支持 from_offset 查询
- [ ] 返回指定 offset 之后的所有消息
- [ ] 消息过期清理（7天）

**验证方式**  
```bash
uv run pytest tests/test_websocket_from_offset.py -v
```

---

### BL-B-60 [P1] WebSocket — 断线重连自动恢复

**目标**  
实现断线重连后自动恢复状态，结合 ReconnectionManager 和 StateRecoveryManager。

**涉及范围**  
- 文件: `wrapper/src/websocket/reliable_server.py`（修改）

**前置依赖**  
BL-B-59 同步丢失消息完成

**完成标准**  
- [ ] 重连后自动恢复 session
- [ ] 同步丢失消息（from_offset）
- [ ] 恢复后发送 ACK 确认
- [ ] 恢复失败进入降级模式

**验证方式**  
```bash
uv run pytest tests/test_websocket_auto_recovery.py -v
```

---

### BL-B-61 [P1] WebSocket — 性能测试实际运行

**目标**  
运行实际的 WebSocket 性能测试，验证服务端性能指标。

**涉及范围**  
- 执行 `tests/performance/test_websocket_concurrent.py`
- 执行 `tests/performance/test_websocket_latency.py`

**前置依赖**  
BL-B-6 并发连接测试脚本完成
BL-B-7 消息延迟测试脚本完成

**完成标准**  
- [ ] 启动 WebSocket 服务器
- [ ] 运行 1000+ 并发连接测试
- [ ] 验证内存使用 < 2GB
- [ ] 验证 CPU 使用 < 80%
- [ ] 验证消息延迟 p99 < 100ms
- [ ] 生成性能测试报告

**验证方式**  
```bash
# 启动服务器
uv run python start_services.py

# 运行测试
uv run python tests/performance/test_websocket_concurrent.py --clients 1000 --duration 60
uv run python tests/performance/test_websocket_latency.py --duration 60
```

---

### BL-B-62 [P2] WebSocket — CI/CD 性能测试集成

**目标**  
将 WebSocket 性能测试集成到 CI/CD 流程。

**涉及范围**  
- 文件: `.github/workflows/performance.yml`（新建）

**前置依赖**  
BL-B-61 性能测试实际运行完成

**完成标准**  
- [ ] GitHub Actions 工作流配置
- [ ] 定时运行性能测试（每日/每周）
- [ ] 性能指标趋势图
- [ ] 性能退化告警

**验证方式**  
GitHub Actions 运行成功

---

### BL-B-63 [P1] WebSocket — 性能测试套件整合

**目标**  
整合 BL-B-6、BL-B-7、BL-B-51 三个测试脚本，形成完整的性能测试套件。

**涉及范围**  
- 文件: `tests/performance/test_websocket_suite.py`（新建）
- 文件: `tests/performance/run_all_tests.py`（新建）

**前置依赖**  
BL-B-6 并发连接测试完成
BL-B-7 消息延迟测试完成
BL-B-51 心跳成功率验证完成

**完成标准**  
- [ ] 统一的测试套件入口
- [ ] 顺序执行所有性能测试
- [ ] 统一的测试报告格式
- [ ] 支持选择性运行特定测试
- [ ] 综合性能评分

**验证方式**  
```bash
# 运行完整测试套件
uv run python tests/performance/test_websocket_suite.py

# 运行特定测试
uv run python tests/performance/test_websocket_suite.py --test concurrent
uv run python tests/performance/test_websocket_suite.py --test latency
uv run python tests/performance/test_websocket_suite.py --test reliability
```

---

### BL-B-64 [P1] PrecomputeService — SurrealDB RELATE 集成

**目标**  
将 RelationBuilder 与 SurrealDB 集成，实际创建 atom → atom RELATE 关系。

**涉及范围**  
- 文件: `wrapper/src/services/relation_builder.py`（修改）

**前置依赖**  
BL-B-10 调用关系创建完成

**完成标准**  
- [ ] 集成 SurrealDB 客户端
- [ ] 实现 RELATE 语句生成
- [ ] 批量执行 RELATE 操作
- [ ] 错误处理和重试机制
- [ ] 关系查询接口

**验证方式**  
```bash
uv run pytest tests/test_relation_builder_integration.py -v
```

---

### BL-B-65 [P1] PrecomputeService — CycleDetector 集成

**目标**  
将 CycleDetector 集成到 RelationBuilder，在创建关系时检测循环。

**涉及范围**  
- 文件: `wrapper/src/services/relation_builder.py`（修改）

**前置依赖**  
BL-B-11 循环检测完成

**完成标准**  
- [ ] 在 RelationBuilder 中集成 CycleDetector
- [ ] 创建关系前检测循环
- [ ] 发现循环时记录警告
- [ ] 支持跳过循环关系创建

**验证方式**  
```bash
uv run pytest tests/test_relation_builder_cycle.py -v
```

---

### BL-B-66 [P2] PrecomputeService — 循环依赖解决策略

**目标**  
定义循环依赖的解决策略，如何处理检测到的循环。

**涉及范围**  
- 文件: `wrapper/src/services/cycle_resolver.py`（新建）

**前置依赖**  
BL-B-65 CycleDetector 集成完成

**完成标准**  
- [ ] 定义循环类型分类
- [ ] 实现循环打破策略
- [ ] 支持循环标记（跳过/警告/错误）
- [ ] 循环依赖报告生成

**验证方式**  
```bash
uv run pytest tests/test_cycle_resolver.py -v
```

---

## 统计汇总

| 分类 | 总数 | P0 | P1 | P2 | P3 | 工时 |
|------|------|----|----|----|----|------|
| 依赖升级 | 1 | 1 | 0 | 0 | 0 | 1 天 |
| WebSocket | 8 | 3 | 5 | 0 | 0 | 5.5 天 |
| WebSocket 后续 | 12 | 0 | 11 | 1 | 0 | 6.5 天 |
| Precompute | 7 | 2 | 3 | 2 | 0 | 7 天 |
| Precompute 后续 | 3 | 0 | 3 | 0 | 0 | 1.5 天 |
| Meilisearch | 3 | 1 | 2 | 0 | 0 | 2 天 |
| Schema | 4 | 1 | 3 | 0 | 0 | 2.5 天 |
| Deployment | 4 | 1 | 2 | 1 | 0 | 2.5 天 |
| Testing | 6 | 2 | 2 | 1 | 0 | 4.5 天 |
| 文档完善 | 8 | 0 | 2 | 4 | 2 | 5 天 |
| **总计** | **55** | **11** | **33** | **9** | **2** | **36 天** |

---

## 执行协议

**协议**: AGENT-COLLABORATION-PROTOCOL-v1.0  
**通信**: `D:\mailbox\` 目录交换邮件  
**响应**: 5 分钟内自动响应  
**评审**: 阶段完成自动评审  

---

**最后更新**: 2026-04-12  
**维护者**: Agent A (后端团队) + Agent B (插件端团队)
