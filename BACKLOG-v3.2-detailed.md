# BACKLOG v3.2 详细版

> **版本**: v3.2.0  
> **创建日期**: 2026-04-12  
> **最后更新**: 2026-04-12  
> **总任务数**: 40  
> **预估总工时**: 28 天

---

## 任务编号规则

- **BL-B-{N}**: 后端任务（Backend）
- **BL-P-{N}**: 插件端任务（Plugin）
- **BL-C-{N}**: 共同任务（Collaboration）

---

## Phase 1: 依赖升级（1 天）

### BL-B-31 [P0] 依赖升级 — pyproject.toml

**1. 目标**
更新 pyproject.toml 依赖版本，为 v3.2 新功能提供基础支持。

**2. 涉及范围**
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

**3. 前置依赖**
- 无

**4. 完成标准**
- [ ] pyproject.toml 更新完成
- [ ] `uv pip install` 无错误
- [ ] `uv run python -c "import all_new_deps"` 成功
- [ ] 依赖版本锁定文件更新

**5. 验证方式**
```bash
# 验证安装
uv pip install -e .

# 验证导入
uv run python -c "import surrealdb, meilisearch, websockets, tree_sitter"

# 验证版本
uv run python -c "import surrealdb; print(surrealdb.__version__)"
```

---

## Phase 2: WebSocket 重写（5.5 天）

### BL-B-1 [P0] WebSocket 可靠连接 — 心跳机制

**1. 目标**
实现 WebSocket 心跳机制，确保连接存活检测，2 次未响应自动触发重连。

**2. 涉及范围**
- 文件: `wrapper/src/websocket/reliable_server.py`（新建）
- 文件: `wrapper/src/websocket/heartbeat.py`（新建）
- 类: `HeartbeatManager`
- 方法: `start_heartbeat()`, `stop_heartbeat()`, `check_pong_timeout()`

**3. 前置依赖**
- BL-B-31 依赖升级完成

**4. 完成标准**
- [ ] 每 30s 发送 ping 消息
- [ ] 5s 内等待 pong 响应
- [ ] 连续 2 次未响应触发 `on_connection_lost`
- [ ] 心跳日志记录（DEBUG 级别）
- [ ] 可配置参数（interval, timeout, max_missing）

**5. 验证方式**
```python
# 单元测试
async def test_heartbeat():
    server = ReliableWebSocketServer()
    await server.start_heartbeat()
    await asyncio.sleep(35)  # 等待 1 次心跳
    assert server.heartbeat_count >= 1
    
# 故障测试
async def test_heartbeat_timeout():
    server = ReliableWebSocketServer()
    await server.simulate_pong_missing(2)
    assert server.connection_state == "RECONNECTING"
```

---

### BL-B-2 [P0] WebSocket 可靠连接 — 指数退避重连

**1. 目标**
实现指数退避重连机制，避免惊群效应，最大重试 10 次。

**2. 涉及范围**
- 文件: `wrapper/src/websocket/reconnection.py`（新建）
- 类: `ReconnectionManager`
- 方法: `schedule_reconnect()`, `calculate_delay()`, `reset_counter()`

**3. 前置依赖**
- BL-B-1 心跳机制完成

**4. 完成标准**
- [ ] 指数退避序列: 1→2→4→8→16→32→64→128→256→300s
- [ ] 随机抖动: +random.uniform(0, 1)s
- [ ] 最大重试: 10 次
- [ ] 重连后恢复 session
- [ ] 重连失败进入降级模式

**5. 验证方式**
```python
# 单元测试
def test_exponential_backoff():
    rm = ReconnectionManager()
    delays = [rm.calculate_delay(i) for i in range(10)]
    assert delays == [1, 2, 4, 8, 16, 32, 64, 128, 256, 300]
    
# 集成测试
async def test_reconnect_after_failure():
    server = ReliableWebSocketServer()
    await server.simulate_disconnect()
    await asyncio.sleep(1)  # 等待第 1 次重连
    assert server.reconnect_count >= 1
```

---

### BL-B-3 [P0] WebSocket 可靠连接 — ACK 确认系统

**1. 目标**
实现消息确认机制，确保消息可靠投递，5s 超时，最多 3 次重试。

**2. 涉及范围**
- 文件: `wrapper/src/websocket/ack_manager.py`（新建）
- 类: `AckManager`
- 方法: `send_with_ack()`, `handle_ack()`, `retry_message()`
- 数据结构: `Dict[str, PendingAck]`

**3. 前置依赖**
- BL-B-1 心跳机制完成

**4. 完成标准**
- [ ] 消息发送后启动 5s 超时计时器
- [ ] 收到 ACK 后清除超时
- [ ] 超时后自动重试（最多 3 次）
- [ ] 达到最大重试次数后 reject
- [ ] ACK 消息格式: `{"type": "ack", "_ackId": "..."}`

**5. 验证方式**
```python
# 单元测试
async def test_ack_success():
    ack_mgr = AckManager()
    result = await ack_mgr.send_with_ack(message, timeout=5000)
    assert result.status == "ACKED"
    
async def test_ack_timeout_retry():
    ack_mgr = AckManager()
    result = await ack_mgr.send_with_ack(message, timeout=5000, max_retries=3)
    assert result.retry_count == 3
    assert result.status == "FAILED"
```

---

### BL-B-4 [P1] WebSocket 可靠连接 — DIFF 模式

**1. 目标**
实现 DIFF 增量同步模式，使用 JSON Patch（RFC 6902），减少 90% 数据传输。

**2. 涉及范围**
- 文件: `wrapper/src/websocket/diff_manager.py`（新建）
- 文件: `wrapper/src/websocket/patch_generator.py`（新建）
- 类: `DiffManager`, `PatchGenerator`
- 方法: `generate_patch()`, `apply_patch()`, `subscribe_diff()`

**3. 前置依赖**
- BL-B-3 ACK 系统完成
- fast-json-patch 依赖已安装

**4. 完成标准**
- [ ] 支持 `LIVE SELECT DIFF` 订阅
- [ ] 生成 RFC 6902 标准 JSON Patch
- [ ] Patch 操作: replace/add/remove
- [ ] 带宽节省 ≥90%（对比全量传输）
- [ ] 客户端可配置 diff/full 模式

**5. 验证方式**
```python
# 单元测试
def test_generate_patch():
    old = {"content": "hello"}
    new = {"content": "world", "tags": ["new"]}
    patch = generate_patch(old, new)
    assert patch == [
        {"op": "replace", "path": "/content", "value": "world"},
        {"op": "add", "path": "/tags", "value": ["new"]}
    ]
    
# 性能测试
def test_bandwidth_savings():
    full_size = len(json.dumps(full_document))
    diff_size = len(json.dumps(diff_patch))
    savings = (full_size - diff_size) / full_size
    assert savings >= 0.9
```

---

### BL-B-5 [P0] WebSocket 可靠连接 — 状态恢复

**1. 目标**
实现连接断开后状态恢复，支持 session + offset 机制。

**2. 涉及范围**
- 文件: `wrapper/src/websocket/state_recovery.py`（新建）
- 类: `StateRecoveryManager`
- 方法: `save_state()`, `restore_state()`, `sync_missed_messages()`
- 存储: `.opencode/ws-state.json`

**3. 前置依赖**
- BL-B-2 重连机制完成

**4. 完成标准**
- [ ] Session ID 生成: `sess-{timestamp}-{uuid[:9]}`
- [ ] Offset 持久化到文件
- [ ] 断线重连后恢复 session
- [ ] 同步丢失消息（from_offset）
- [ ] 状态文件 7 天 TTL 清理

**5. 验证方式**
```python
# 单元测试
async def test_state_recovery():
    rm = StateRecoveryManager()
    await rm.save_state(session_id="sess-123", offset=100)
    state = await rm.restore_state(session_id="sess-123")
    assert state["offset"] == 100
    
# 集成测试
async def test_sync_missed_messages():
    server = ReliableWebSocketServer()
    await server.simulate_disconnect()
    await server.reconnect()
    missed = await server.sync_missed_messages(from_offset=100)
    assert len(missed) > 0
```

---

### BL-B-6 [P1] WebSocket 性能 — 并发连接测试

**1. 目标**
验证 WebSocket 服务端支持 ≥1000 并发连接。

**2. 涉及范围**
- 文件: `tests/performance/test_websocket_concurrent.py`（新建）
- 工具: `locust` 或 `asyncio` 并发测试
- 指标: 并发连接数、内存使用、CPU 使用

**3. 前置依赖**
- BL-B-1~B-5 WebSocket 核心功能完成

**4. 完成标准**
- [ ] 支持 1000+ 并发连接
- [ ] 内存使用 < 2GB
- [ ] CPU 使用 < 80%
- [ ] 无连接丢失

**5. 验证方式**
```bash
# 性能测试
uv run python tests/performance/test_websocket_concurrent.py --clients 1000 --duration 60

# 验证输出
# [PASS] Concurrent connections: 1000/1000
# [PASS] Memory usage: 1.5GB < 2GB
# [PASS] CPU usage: 65% < 80%
```

---

### BL-B-7 [P1] WebSocket 性能 — 消息延迟测试

**1. 目标**
验证 WebSocket 消息延迟 p99 < 100ms。

**2. 涉及范围**
- 文件: `tests/performance/test_websocket_latency.py`（新建）
- 指标: p50/p95/p99 延迟、吞吐量

**3. 前置依赖**
- BL-B-1~B-5 WebSocket 核心功能完成

**4. 完成标准**
- [ ] p99 延迟 < 100ms
- [ ] p95 延迟 < 50ms
- [ ] p50 延迟 < 20ms
- [ ] 吞吐量 ≥ 1000 msg/s

**5. 验证方式**
```bash
# 延迟测试
uv run python tests/performance/test_websocket_latency.py --duration 60

# 验证输出
# [PASS] p50 latency: 15ms < 20ms
# [PASS] p95 latency: 45ms < 50ms
# [PASS] p99 latency: 85ms < 100ms
# [PASS] Throughput: 1200 msg/s >= 1000 msg/s
```

---

### BL-B-51 [P1] WebSocket 可靠性 — 心跳成功率 ≥99% 验证

**1. 目标**
验证 WebSocket 心跳成功率 ≥99%。

**2. 涉及范围**
- 文件: `tests/performance/test_websocket_reliability.py`（新建）
- 指标: 心跳成功率、丢包率

**3. 前置依赖**
- BL-B-1 心跳机制完成
- BL-B-6 并发测试完成

**4. 完成标准**
- [ ] 心跳成功率 ≥99%
- [ ] 连续运行 24 小时无故障
- [ ] 丢包率 < 1%

**5. 验证方式**
```bash
# 可靠性测试
uv run python tests/performance/test_websocket_reliability.py --duration 86400

# 验证输出
# [PASS] Heartbeat success rate: 99.5% >= 99%
# [PASS] Packet loss: 0.3% < 1%
```

---

## Phase 3: PrecomputeService（7 天）

### BL-B-8 [P0] PrecomputeService — 基础架构

**1. 目标**
创建 PrecomputeService 服务骨架，实现服务化架构，支持 tenant 隔离。

**2. 涉及范围**
- 文件: `wrapper/src/services/precompute.py`（新建）
- 文件: `wrapper/src/services/__init__.py`（新建）
- 类: `PrecomputeService`
- 方法: `__init__()`, `start()`, `stop()`, `process_batch()`

**3. 前置依赖**
- BL-B-31 依赖升级完成
- BL-B-18 Schema 升级完成

**4. 完成标准**
- [ ] PrecomputeService 类实现
- [ ] 支持 tenant_id 隔离
- [ ] 支持 DB 连接注入
- [ ] 支持启动/停止生命周期
- [ ] 基础日志记录

**5. 验证方式**
```python
# 单元测试
async def test_precompute_service_init():
    service = PrecomputeService(db=mock_db, tenant_id="default")
    assert service.tenant_id == "default"
    await service.start()
    assert service.is_running == True
    await service.stop()
    assert service.is_running == False
```

---

### BL-B-9 [P0] PrecomputeService — tree-sitter 集成 + 指纹

**1. 目标**
集成 tree-sitter 进行代码解析，实现 SHA256 指纹计算，支持增量分析。

**2. 涉及范围**
- 文件: `wrapper/src/services/code_parser.py`（新建）
- 文件: `wrapper/src/services/fingerprint.py`（新建）
- 类: `CodeParser`, `FingerprintManager`
- 方法: `parse_code()`, `calculate_fingerprint()`, `check_changes()`

**3. 前置依赖**
- BL-B-8 基础架构完成
- tree-sitter 依赖已安装

**4. 完成标准**
- [ ] 支持 Python/JavaScript/TypeScript 解析
- [ ] SHA256 指纹计算
- [ ] 指纹持久化到 DB
- [ ] 变更检测（指纹比对）
- [ ] 未变更文件跳过分析

**5. 验证方式**
```python
# 单元测试
def test_fingerprint():
    fm = FingerprintManager()
    fp = fm.calculate_fingerprint("def hello(): pass")
    assert len(fp) == 64  # SHA256 hex
    
async def test_incremental_analysis():
    service = PrecomputeService()
    result = await service.process_file(
        file_path="test.py",
        content="def hello(): pass",
        fingerprint="abc123..."
    )
    if result["skipped"]:
        assert result["reason"] == "No changes"
```

---

### BL-B-10 [P1] PrecomputeService — 调用关系创建

**1. 目标**
从 AST 中提取函数调用关系，自动创建 RELATE 关系。

**2. 涉及范围**
- 文件: `wrapper/src/services/relation_builder.py`（新建）
- 类: `RelationBuilder`
- 方法: `extract_calls()`, `create_relations()`, `batch_relate()`

**3. 前置依赖**
- BL-B-9 tree-sitter 集成完成

**4. 完成标准**
- [ ] 提取函数调用关系
- [ ] 创建 atom → atom RELATE
- [ ] 批量创建关系（100 条/批）
- [ ] 自调用过滤（caller != callee）
- [ ] 关系权重计算（基础）

**5. 验证方式**
```python
# 单元测试
def test_extract_calls():
    code = """
def foo():
    bar()
    baz()
"""
    rb = RelationBuilder()
    calls = rb.extract_calls(code)
    assert calls == [{"caller": "foo", "callee": "bar"}, {"caller": "foo", "callee": "baz"}]
    
async def test_create_relations():
    rb = RelationBuilder()
    await rb.create_relations(calls)
    assert db.query("SELECT count(*) FROM relations") > 0
```

---

### BL-B-11 [P2] PrecomputeService — 循环检测

**1. 目标**
检测代码中的循环依赖（circular dependencies）。

**2. 涉及范围**
- 文件: `wrapper/src/services/cycle_detector.py`（新建）
- 类: `CycleDetector`
- 方法: `detect_cycles()`, `dfs()`, `report_cycles()`
- 算法: DFS（深度优先搜索）

**3. 前置依赖**
- BL-B-10 调用关系创建完成

**4. 完成标准**
- [ ] DFS 算法实现
- [ ] 检测循环调用链
- [ ] 记录循环路径
- [ ] 日志输出警告
- [ ] 时间复杂度 O(V+E)

**5. 验证方式**
```python
# 单元测试
def test_detect_cycles():
    # A → B → C → A (cycle)
    relations = [
        {"from": "A", "to": "B"},
        {"from": "B", "to": "C"},
        {"from": "C", "to": "A"}
    ]
    cd = CycleDetector()
    cycles = cd.detect_cycles(relations)
    assert len(cycles) == 1
    assert cycles[0] == ["A", "B", "C", "A"]
```

---

### BL-B-12 [P2] PrecomputeService — 权重计算

**1. 目标**
计算调用关系的权重，用于图遍历优先级。

**2. 涉及范围**
- 文件: `wrapper/src/services/weight_calculator.py`（新建）
- 类: `WeightCalculator`
- 方法: `calculate_weight()`, `normalize_weights()`
- 因素: 调用频率、复杂度、参数数量、跨文件

**3. 前置依赖**
- BL-B-10 调用关系创建完成

**4. 完成标准**
- [ ] 权重因子定义
- [ ] 权重计算公式
- [ ] 归一化处理
- [ ] 权重持久化

**5. 验证方式**
```python
# 单元测试
def test_calculate_weight():
    wc = WeightCalculator()
    weight = wc.calculate_weight(
        call_frequency=10,
        complexity=5,
        param_count=3,
        is_cross_file=True
    )
    assert 0 <= weight <= 1
```

---

### BL-B-13 [P1] PrecomputeService — 性能监控

**1. 目标**
监控 PrecomputeService 性能，记录耗时、内存使用。

**2. 涉及范围**
- 文件: `wrapper/src/services/performance_monitor.py`（新建）
- 类: `PerformanceMonitor`
- 方法: `monitor()`, `log_metrics()`, `report()`
- 指标: parse_time, analysis_time, memory_usage

**3. 前置依赖**
- BL-B-8 基础架构完成

**4. 完成标准**
- [ ] 性能指标收集
- [ ] 内存监控
- [ ] 日志记录
- [ ] 性能报告生成

**5. 验证方式**
```python
# 单元测试
async def test_performance_monitor():
    pm = PerformanceMonitor()
    with pm.monitor("parse"):
        await parse_code(content)
    metrics = pm.get_metrics()
    assert "parse_time_ms" in metrics
    assert metrics["parse_time_ms"] > 0
```

---

### BL-B-14 [P1] PrecomputeService — 并发控制

**1. 目标**
实现并发控制，防止同文件重复处理，限制并发数。

**2. 涉及范围**
- 文件: `wrapper/src/services/concurrency_control.py`（新建）
- 类: `ConcurrencyControl`
- 方法: `acquire()`, `release()`, `is_processing()`
- 机制: Semaphore(5) + processing Set

**3. 前置依赖**
- BL-B-8 基础架构完成

**4. 完成标准**
- [ ] Semaphore(5) 并发限制
- [ ] processing Set 去重
- [ ] 队列机制
- [ ] 超时处理

**5. 验证方式**
```python
# 单元测试
async def test_concurrency_limit():
    cc = ConcurrencyControl(max_concurrent=5)
    tasks = [cc.process(f"file_{i}") for i in range(10)]
    results = await asyncio.gather(*tasks)
    assert cc.max_concurrent_reached <= 5
```

---

## Phase 4: Meilisearch SDK 升级（2 天）

### BL-B-15 [P0] Meilisearch SDK 0.40 — 客户端迁移

**1. 目标**
将 Meilisearch 客户端从 httpx REST 调用迁移到官方 SDK 0.40。

**2. 涉及范围**
- 文件: `wrapper/src/utils/meili_client.py`（修改）
- 依赖: `meilisearch>=0.40.0,<0.41.0`
- API: 所有 Meilisearch 调用

**3. 前置依赖**
- BL-B-31 依赖升级完成

**4. 完成标准**
- [ ] 替换 httpx 为 meilisearch SDK
- [ ] 更新所有 API 调用
- [ ] 错误处理适配
- [ ] 配置迁移

**5. 验证方式**
```python
# 单元测试
async def test_meilisearch_sdk():
    client = MeiliClient()
    await client.connect()
    result = await client.search("test")
    assert "hits" in result
```

---

### BL-B-16 [P1] Meilisearch SDK 0.40 — 索引设置迁移

**1. 目标**
迁移 Meilisearch 索引设置到新 SDK。

**2. 涉及范围**
- 文件: `wrapper/src/utils/meili_client.py`（修改）
- 索引: `memories`, `code_search_index`
- 设置: filterableAttributes, searchableAttributes, rankingRules

**3. 前置依赖**
- BL-B-15 客户端迁移完成

**4. 完成标准**
- [ ] 索引设置迁移
- [ ] 字段映射更新
- [ ] 搜索配置更新

**5. 验证方式**
```python
# 单元测试
async def test_index_settings():
    client = MeiliClient()
    settings = await client.get_settings("memories")
    assert "filterableAttributes" in settings
```

---

### BL-B-17 [P1] Meilisearch SDK 0.40 — 批量操作支持

**1. 目标**
实现批量操作支持，提升导入性能。

**2. 涉及范围**
- 文件: `wrapper/src/utils/meili_client.py`（修改）
- 方法: `batch_add_documents()`, `batch_update_documents()`

**3. 前置依赖**
- BL-B-15 客户端迁移完成

**4. 完成标准**
- [ ] 批量添加文档
- [ ] 批量更新文档
- [ ] 批量删除文档
- [ ] 批处理大小 100 条

**5. 验证方式**
```python
# 单元测试
async def test_batch_operations():
    client = MeiliClient()
    documents = [{"id": i} for i in range(100)]
    result = await client.batch_add_documents("memories", documents)
    assert result["processed"] == 100
```

---

## Phase 5: SurrealDB Schema 升级（2.5 天）

### BL-B-18 [P0] Schema v3.2 — 核心表创建

**1. 目标**
创建 v3.2 核心表：atom, entity, reference。

**2. 涉及范围**
- 文件: `scripts/init_surrealdb_v3.2.surql`（新建）
- 表: `atom`, `entity`, `reference`
- 字段: 完整字段定义

**3. 前置依赖**
- SurrealDB 1.0.8 已安装

**4. 完成标准**
- [ ] atom 表创建
- [ ] entity 表创建
- [ ] reference 表创建
- [ ] tenant_id 预留字段
- [ ] 索引创建

**5. 验证方式**
```sql
-- 验证表创建
INFO FOR DB;
-- 应显示 atom, entity, reference 表
```

---

### BL-B-19 [P1] Schema v3.2 — ChangeFeed 配置

**1. 目标**
配置 SurrealDB ChangeFeed，支持实时变更通知。

**2. 涉及范围**
- 文件: `scripts/init_surrealdb_v3.2.surql`（修改）
- 配置: `CHANGE FEED 7d ON TABLE ...`

**3. 前置依赖**
- BL-B-18 核心表创建完成

**4. 完成标准**
- [ ] ChangeFeed 启用
- [ ] 7 天 TTL 配置
- [ ] 支持 atom/entity/reference 表

**5. 验证方式**
```sql
-- 验证 ChangeFeed
LIVE SELECT * FROM atom;
-- 应返回 query UUID
```

---

### BL-B-20 [P1] Schema v3.2 — 辅助表创建

**1. 目标**
创建辅助表：performance_log, session_state。

**2. 涉及范围**
- 文件: `scripts/init_surrealdb_v3.2.surql`（修改）
- 表: `performance_log`, `session_state`

**3. 前置依赖**
- BL-B-18 核心表创建完成

**4. 完成标准**
- [ ] performance_log 表创建
- [ ] session_state 表创建
- [ ] 索引创建

**5. 验证方式**
```sql
INFO FOR DB;
-- 应显示所有表
```

---

### BL-B-21 [P1] Schema v3.2 — 迁移脚本

**1. 目标**
创建数据迁移脚本，从 v2.x 迁移到 v3.2。

**2. 涉及范围**
- 文件: `scripts/migrate_v2_to_v3.2.py`（新建）
- 迁移: memory → atom/entity/reference

**3. 前置依赖**
- BL-B-18~B-20 Schema 创建完成

**4. 完成标准**
- [ ] 数据迁移脚本
- [ ] 数据验证
- [ ] 回滚机制
- [ ] 迁移日志

**5. 验证方式**
```bash
uv run python scripts/migrate_v2_to_v3.2.py --dry-run
uv run python scripts/migrate_v2_to_v3.2.py --execute
```

---

## Phase 6: 端口迁移（2.5 天）

### BL-B-22 [P0] 端口迁移 17999 → 18008

**1. 目标**
将服务端口从 17999 迁移到 18008，支持双端口并行期。

**2. 涉及范围**
- 文件: `wrapper/src/config.py`（修改）
- 配置: 端口配置更新
- 部署: 双端口并行支持

**3. 前置依赖**
- 无

**4. 完成标准**
- [ ] 默认端口改为 18008
- [ ] 双端口并行支持（1-2 周）
- [ ] 环境变量覆盖支持
- [ ] 文档更新

**5. 验证方式**
```bash
# 验证端口
curl http://localhost:18008/health
curl http://localhost:17999/health  # 并行期
```

---

### BL-B-23 [P1] Docker 多阶段构建优化

**1. 目标**
优化 Docker 镜像构建，使用多阶段构建减少镜像体积。

**2. 涉及范围**
- 文件: `Dockerfile`（修改）
- 优化: 多阶段构建、缓存优化

**3. 前置依赖**
- 无

**4. 完成标准**
- [ ] 多阶段构建 Dockerfile
- [ ] 镜像体积减少 50%+
- [ ] 构建时间减少 30%+

**5. 验证方式**
```bash
docker build -t embedding-service:v3.2 .
docker images | grep embedding-service
# 验证体积
```

---

### BL-B-24 [P1] docker-compose 健康检查

**1. 目标**
添加 docker-compose 健康检查配置。

**2. 涉及范围**
- 文件: `docker-compose.yml`（修改）
- 配置: healthcheck

**3. 前置依赖**
- BL-B-22 端口迁移完成

**4. 完成标准**
- [ ] healthcheck 配置
- [ ] 依赖服务启动顺序
- [ ] 自动重启策略

**5. 验证方式**
```bash
docker-compose up -d
docker-compose ps
# 应显示 healthy
```

---

### BL-B-25 [P2] SSL 自动续期

**1. 目标**
配置 SSL 证书自动续期（Certbot）。

**2. 涉及范围**
- 文件: `docker-compose.yml`（修改）
- 配置: Certbot 容器

**3. 前置依赖**
- 域名已配置

**4. 完成标准**
- [ ] Certbot 配置
- [ ] 自动续期脚本
- [ ] 证书验证

**5. 验证方式**
```bash
openssl s_client -connect api.example.com:443
# 验证证书有效期
```

---

## Phase 7: 测试（4.5 天）

### BL-B-26 [P0] 单元测试 — WebSocket 模块

**1. 目标**
为 WebSocket 模块编写单元测试，覆盖率 ≥80%。

**2. 涉及范围**
- 文件: `tests/test_websocket_*.py`（新建）
- 覆盖: heartbeat, ack, reconnection, diff, state_recovery

**3. 前置依赖**
- BL-B-1~B-5 WebSocket 实现完成

**4. 完成标准**
- [ ] 单元测试覆盖率 ≥80%
- [ ] 所有关键路径测试
- [ ] Mock 外部依赖

**5. 验证方式**
```bash
uv run pytest tests/test_websocket_*.py --cov=wrapper/src/websocket --cov-report=html
# 验证覆盖率 >= 80%
```

---

### BL-B-27 [P0] 单元测试 — Precompute 模块

**1. 目标**
为 Precompute 模块编写单元测试，覆盖率 ≥80%。

**2. 涉及范围**
- 文件: `tests/test_precompute_*.py`（新建）
- 覆盖: parser, fingerprint, relations, cycles, weights

**3. 前置依赖**
- BL-B-8~B-14 Precompute 实现完成

**4. 完成标准**
- [ ] 单元测试覆盖率 ≥80%
- [ ] 所有关键路径测试
- [ ] Mock 外部依赖

**5. 验证方式**
```bash
uv run pytest tests/test_precompute_*.py --cov=wrapper/src/services --cov-report=html
```

---

### BL-B-28 [P1] 集成测试 — WebSocket 端到端

**1. 目标**
编写 WebSocket 端到端集成测试。

**2. 涉及范围**
- 文件: `tests/integration/test_websocket_e2e.py`（新建）
- 场景: 连接、心跳、ACK、重连、DIFF

**3. 前置依赖**
- BL-B-26 单元测试完成

**4. 完成标准**
- [ ] 端到端测试通过
- [ ] 真实服务测试
- [ ] 性能基准测试

**5. 验证方式**
```bash
uv run pytest tests/integration/test_websocket_e2e.py -v
```

---

### BL-B-29 [P1] 集成测试 — API 端到端

**1. 目标**
编写 API 端到端集成测试。

**2. 涉及范围**
- 文件: `tests/integration/test_api_e2e.py`（新建）
- 场景: Precompute API, Memory API, Search API

**3. 前置依赖**
- BL-B-27 单元测试完成

**4. 完成标准**
- [ ] 端到端测试通过
- [ ] 真实服务测试
- [ ] 数据一致性验证

**5. 验证方式**
```bash
uv run pytest tests/integration/test_api_e2e.py -v
```

---

### BL-B-30 [P2] 性能基准测试

**1. 目标**
建立性能基准，记录关键指标。

**2. 涉及范围**
- 文件: `tests/performance/benchmark.py`（新建）
- 指标: 延迟、吞吐量、并发、内存

**3. 前置依赖**
- BL-B-28~B-29 集成测试完成

**4. 完成标准**
- [ ] 性能基准建立
- [ ] 基准报告生成
- [ ] 性能回归检测

**5. 验证方式**
```bash
uv run python tests/performance/benchmark.py --report
```

---

### BL-B-51 [P1] WebSocket 可靠性 — 心跳成功率 ≥99% 验证

**1. 目标**
验证 WebSocket 心跳成功率 ≥99%。

**2. 涉及范围**
- 文件: `tests/performance/test_websocket_reliability.py`（新建）
- 指标: 心跳成功率、丢包率

**3. 前置依赖**
- BL-B-1 心跳机制完成
- BL-B-6 并发测试完成

**4. 完成标准**
- [ ] 心跳成功率 ≥99%
- [ ] 连续运行 24 小时无故障
- [ ] 丢包率 < 1%

**5. 验证方式**
```bash
uv run python tests/performance/test_websocket_reliability.py --duration 86400
```

---

## 文档完善任务（5 天）

### BL-CA-43 [P1] 补充 WebSocket 性能测试基准

**1. 目标**
补充 WebSocket 性能测试基准文档。

**2. 涉及范围**
- 文件: `docs/v3.2/BACKEND-v3.2-WEBSOCKET.md`（补充）
- 内容: 性能指标、测试方法、基准数据

**3. 前置依赖**
- BL-B-6~B-7 性能测试完成

**4. 完成标准**
- [ ] 性能指标文档
- [ ] 测试方法说明
- [ ] 基准数据记录

**5. 验证方式**
- 文档评审通过

---

### BL-CA-44 [P1] 完善 PrecomputeService 关系创建实现

**1. 目标**
完善 PrecomputeService 关系创建实现文档。

**2. 涉及范围**
- 文件: `docs/v3.2/BACKEND-v3.2-PRECOMPUTE.md`（补充）
- 内容: 关系创建算法、权重计算、循环检测

**3. 前置依赖**
- BL-B-10~B-12 实现完成

**4. 完成标准**
- [ ] 关系创建算法文档
- [ ] 权重计算说明
- [ ] 循环检测算法

**5. 验证方式**
- 文档评审通过

---

### BL-CA-45 [P2] 统一预计算批处理大小参数

**1. 目标**
统一预计算批处理大小参数文档。

**2. 涉及范围**
- 文件: `docs/v3.2/BACKEND-v3.2-PRECOMPUTE.md`（补充）
- 参数: batch_size = 100

**3. 前置依赖**
- BL-B-8 基础架构完成

**4. 完成标准**
- [ ] 批处理参数统一
- [ ] 文档更新
- [ ] 配置说明

**5. 验证方式**
- 文档评审通过

---

### BL-CA-46 [P2] 扩充后端实施指南

**1. 目标**
扩充后端实施指南文档。

**2. 涉及范围**
- 文件: `docs/v3.2/BACKEND-v3.2-IMPLEMENTATION.md`（扩充）
- 内容: 详细实施步骤、最佳实践、常见问题

**3. 前置依赖**
- Phase 2-3 开发完成

**4. 完成标准**
- [ ] 详细实施步骤
- [ ] 最佳实践总结
- [ ] FAQ 整理

**5. 验证方式**
- 文档评审通过

---

### BL-CA-47 [P2] 添加 WebSocket 错误处理示例

**1. 目标**
添加 WebSocket 错误处理示例代码。

**2. 涉及范围**
- 文件: `docs/v3.2/BACKEND-v3.2-WEBSOCKET.md`（补充）
- 内容: 错误码、处理示例、故障排查

**3. 前置依赖**
- BL-B-1~B-5 实现完成

**4. 完成标准**
- [ ] 错误码定义
- [ ] 处理示例代码
- [ ] 故障排查指南

**5. 验证方式**
- 文档评审通过

---

### BL-CA-48 [P2] 添加 Kubernetes 部署配置

**1. 目标**
添加 Kubernetes 部署配置。

**2. 涉及范围**
- 文件: `k8s/`（新建目录）
- 配置: deployment.yaml, service.yaml, ingress.yaml

**3. 前置依赖**
- BL-B-22~B-25 部署配置完成

**4. 完成标准**
- [ ] Kubernetes 配置
- [ ] Helm chart（可选）
- [ ] 部署文档

**5. 验证方式**
```bash
kubectl apply -f k8s/
kubectl get pods
```

---

### BL-CA-49 [P3] 添加数据库 ER 关系图

**1. 目标**
添加数据库 ER 关系图。

**2. 涉及范围**
- 文件: `docs/v3.2/DATABASE-v3.2-ER.md`（新建）
- 图表: atom, entity, reference 关系图

**3. 前置依赖**
- BL-B-18~B-21 Schema 完成

**4. 完成标准**
- [ ] ER 图绘制
- [ ] 关系说明
- [ ] 文档集成

**5. 验证方式**
- 文档评审通过

---

### BL-CA-50 [P3] 添加 SSL 自动续期配置

**1. 目标**
添加 SSL 自动续期配置文档。

**2. 涉及范围**
- 文件: `docs/v3.2/DEPLOYMENT-v3.2.md`（补充）
- 内容: Certbot 配置、自动续期脚本

**3. 前置依赖**
- BL-B-25 SSL 配置完成

**4. 完成标准**
- [ ] Certbot 配置说明
- [ ] 自动续期脚本
- [ ] 验证方法

**5. 验证方式**
- 文档评审通过

---

## 统计

| 分类 | 总数 | P0 | P1 | P2 | P3 |
|------|------|----|----|----|----|
| 依赖升级 | 1 | 1 | 0 | 0 | 0 |
| WebSocket | 8 | 3 | 5 | 0 | 0 |
| Precompute | 7 | 2 | 3 | 2 | 0 |
| Meilisearch | 3 | 1 | 2 | 0 | 0 |
| Schema | 4 | 1 | 3 | 0 | 0 |
| Deployment | 4 | 1 | 2 | 1 | 0 |
| Testing | 6 | 2 | 2 | 1 | 0 |
| 文档完善 | 8 | 0 | 2 | 4 | 2 |
| **总计** | **40** | **11** | **19** | **8** | **2** |

---

**最后更新**: 2026-04-12  
**维护者**: Agent A (后端团队)
