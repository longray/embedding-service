# BACKLOG v3.4 - 补充测试与质量提升

> **版本**: v3.4.0  
> **创建日期**: 2026-04-15  
> **目标**: 完善测试覆盖，提升代码质量与系统稳定性  
> **预估总工时**: 5 天  
> **协议**: AGENT-COLLABORATION-PROTOCOL-v1.0

---

## 任务总览

| 编号 | 目标 | 优先级 | 工时 | 状态 | 详情 |
|------|------|--------|------|------|------|
| **测试补充** |
| BL-T-1 | Audit 日志端点测试 | P1 | 1 天 | ⏳ | [详情](#bl-t-1-p1-audit-日志端点测试) |
| BL-T-2 | Projects API 测试 | P1 | 1 天 | ⏳ | [详情](#bl-t-2-p1-projects-api-测试) |
| BL-T-3 | Lookup API 测试 | P1 | 0.5 天 | ⏳ | [详情](#bl-t-3-p1-lookup-api-测试) |
| BL-T-4 | 并发压力测试 | P2 | 1 天 | ⏳ | [详情](#bl-t-4-p2-并发压力测试) |
| BL-T-5 | 故障恢复测试 | P2 | 1 天 | ⏳ | [详情](#bl-t-5-p2-故障恢复测试) |
| BL-T-6 | E2E 测试套件整合 | P2 | 0.5 天 | ⏳ | [详情](#bl-t-6-p2-e2e-测试套件整合) |

---

## 测试补充

### BL-T-1 [P1] Audit 日志端点测试

**目标**  
为 Audit 日志端点补充完整的单元测试和集成测试，确保审计功能稳定可靠。

**涉及范围**  
- 文件: `wrapper/src/routers/audit.py`（被测对象）
- 文件: `tests/test_audit_endpoints.py`（新建）
- 文件: `tests/integration/test_audit_integration.py`（新建）

**前置依赖**  
- Audit router 已实现 (`wrapper/src/routers/audit.py`)
- MemoryManager 已集成 audit 方法
- SurrealDB 连接可用

**完成标准**  
- [ ] `POST /api/v1/audit/log` 单元测试（5+ 场景）
  - [ ] 正常记录审计日志
  - [ ] 缺少必填字段验证
  - [ ] MemoryManager 未初始化错误
  - [ ] 数据库写入失败处理
  - [ ] 大 details 字段处理
- [ ] `GET /api/v1/audit/logs` 单元测试（6+ 场景）
  - [ ] 无参数查询全部日志
  - [ ] 按时间范围过滤
  - [ ] 按 user_id 过滤
  - [ ] 按 action 过滤
  - [ ] 按 resource_type 过滤
  - [ ] 组合过滤条件
  - [ ] 分页查询
  - [ ] 无结果返回空列表
- [ ] `DELETE /api/v1/audit/logs` 单元测试（2+ 场景）
  - [ ] 正常清理过期日志
  - [ ] 自定义 retention_days
- [ ] 集成测试（2+ 场景）
  - [ ] 端到端日志记录与查询流程
  - [ ] 并发写入测试

**验证方式**  
```bash
# 单元测试
uv run pytest tests/test_audit_endpoints.py -v

# 集成测试
uv run pytest tests/integration/test_audit_integration.py -v

# 覆盖率检查
uv run pytest tests/test_audit_endpoints.py --cov=wrapper.src.routers.audit --cov-report=term-missing
```

**测试要点**  
- 验证请求体字段验证（action, resource_type, resource_id 必填）
- 验证时间范围查询的边界处理
- 验证分页参数（skip, limit）的默认值和边界
- 验证审计日志的 retention 机制

---

### BL-T-2 [P1] Projects API 测试

**目标**  
为 Projects 端点（代码地图、统计信息）补充测试，确保项目级代码分析功能正确。

**涉及范围**  
- 文件: `wrapper/src/routers/projects.py`（被测对象）
- 文件: `wrapper/src/utils/memory_manager/projects.py`（被测对象）
- 文件: `tests/test_projects_api.py`（新建）

**前置依赖**  
- Projects router 已实现
- MemoryManager.get_project_map() 已实现
- MemoryManager.get_project_stats() 已实现
- 测试数据：包含 project_id 的记忆数据

**完成标准**  
- [ ] `GET /api/v1/projects/{project_id}/map` 测试（5+ 场景）
  - [ ] 正常获取项目地图
  - [ ] 项目不存在返回空结构
  - [ ] 验证返回字段完整性（file_tree, dependencies, hot_files, stats）
  - [ ] 多租户隔离（不同 tenant_id 返回不同数据）
  - [ ] 大数据量项目性能测试
- [ ] `GET /api/v1/projects/{project_id}/stats` 测试（5+ 场景）
  - [ ] 正常获取项目统计
  - [ ] 验证统计字段（total_files, total_functions, total_classes, avg_complexity, max_complexity）
  - [ ] 空项目统计（全为 0）
  - [ ] 多租户隔离
  - [ ] 复杂项目统计准确性验证
- [ ] 错误处理测试（3+ 场景）
  - [ ] MemoryManager 未初始化
  - [ ] 数据库查询失败
  - [ ] 无效 project_id 格式

**验证方式**  
```bash
# 单元测试
uv run pytest tests/test_projects_api.py -v

# 带覆盖率
uv run pytest tests/test_projects_api.py --cov=wrapper.src.routers.projects --cov-report=term-missing
```

**测试数据准备**  
```python
# 需要创建包含以下字段的记忆数据
test_memories = [
    {
        "content": "def hello(): pass",
        "metadata": {
            "project_id": "test-project",
            "file_path": "src/main.py",
            "code_analysis": {
                "functions": [{"name": "hello"}],
                "classes": [],
                "complexity": 1
            }
        }
    }
]
```

---

### BL-T-3 [P1] Lookup API 测试

**目标**  
为 Lookup 端点（source_id/hash/file_path 查询）补充测试，确保多设备同步查询功能正确。

**涉及范围**  
- 文件: `wrapper/src/routers/lookup.py`（被测对象）
- 文件: `wrapper/src/utils/memory_manager/lookup.py`（被测对象）
- 文件: `tests/test_lookup_api.py`（新建）

**前置依赖**  
- Lookup router 已实现
- MemoryManager lookup 方法已实现
- 测试数据：包含 source_id、content_hash、file_path 的记忆

**完成标准**  
- [ ] source_id 查询测试（4+ 场景）
  - [ ] 正常查询返回单条记忆
  - [ ] 查询不存在的 source_id
  - [ ] 返回字段验证（memory_id, source_id, file_path, project_id, type）
  - [ ] all=true 返回历史版本
- [ ] hash 查询测试（4+ 场景）
  - [ ] 正常查询返回记忆
  - [ ] 不同 hash_algorithm（md5/sha256）
  - [ ] 查询不存在的 hash
  - [ ] limit 参数测试
- [ ] file_path + project_id 查询测试（3+ 场景）
  - [ ] 正常查询返回记忆
  - [ ] 不同 project_id 隔离
  - [ ] 查询不存在的路径
- [ ] 参数优先级测试（2+ 场景）
  - [ ] source_id 优先级高于 hash
  - [ ] hash 优先级高于 file_path
- [ ] 错误处理测试（3+ 场景）
  - [ ] 缺少所有查询参数
  - [ ] 只提供 file_path 不提供 project_id
  - [ ] 无效 limit 值（<1 或 >100）
- [ ] 多租户隔离测试（2+ 场景）
  - [ ] 不同 tenant_id 返回不同数据
  - [ ] 默认 tenant_id="default"

**验证方式**  
```bash
uv run pytest tests/test_lookup_api.py -v
```

**测试要点**  
- 验证查询优先级：source_id > hash > file_path+project_id
- 验证返回字段的完整性和类型
- 验证 limit 参数的边界值（1, 100）
- 验证 all 参数控制返回单条还是多条

---

### BL-T-4 [P2] 并发压力测试

**目标**  
建立并发压力测试套件，验证系统在高并发场景下的稳定性和性能。

**涉及范围**  
- 文件: `tests/performance/test_concurrency.py`（新建）
- 文件: `tests/performance/test_load.py`（新建）
- 依赖: `locust` 或 `pytest-asyncio` + `httpx`

**前置依赖**  
- 核心 API 端点已实现
- Docker Compose 开发环境可用
- 测试数据准备脚本

**完成标准**  
- [ ] 并发连接测试（3+ 场景）
  - [ ] 100 并发连接健康检查
  - [ ] 500 并发连接健康检查
  - [ ] 1000 并发连接健康检查（记录性能指标）
- [ ] 并发写入测试（3+ 场景）
  - [ ] 50 并发记忆上传
  - [ ] 100 并发记忆上传
  - [ ] 验证数据一致性（无丢失、无重复）
- [ ] 并发搜索测试（3+ 场景）
  - [ ] 50 并发搜索请求
  - [ ] 100 并发搜索请求
  - [ ] 混合读写并发（读 80% + 写 20%）
- [ ] 资源监控（2+ 指标）
  - [ ] 内存使用监控（无内存泄漏）
  - [ ] CPU 使用率监控
  - [ ] 响应时间 P99/P95/P50 统计
- [ ] 性能基准（2+ 指标）
  - [ ] 单请求平均响应时间 < 200ms
  - [ ] 并发场景下错误率 < 0.1%

**验证方式**  
```bash
# 并发测试
uv run pytest tests/performance/test_concurrency.py -v

# 压力测试（需要服务运行）
uv run pytest tests/performance/test_load.py -v --stress

# 生成报告
uv run pytest tests/performance/test_concurrency.py --benchmark-json=report.json
```

**技术方案**  
```python
# 使用 pytest-asyncio + httpx 实现并发
import asyncio
import httpx
import pytest

@pytest.mark.asyncio
async def test_concurrent_health_checks():
    async with httpx.AsyncClient() as client:
        tasks = [client.get("http://localhost:18008/health") for _ in range(100)]
        responses = await asyncio.gather(*tasks)
        assert all(r.status_code == 200 for r in responses)
```

---

### BL-T-5 [P2] 故障恢复测试

**目标**  
建立故障恢复测试套件，验证系统在依赖服务故障时的恢复能力。

**涉及范围**  
- 文件: `tests/resilience/test_service_recovery.py`（新建）
- 文件: `tests/resilience/test_db_reconnect.py`（新建）
- 依赖: Docker Compose 健康检查、网络模拟工具

**前置依赖**  
- Docker Compose 环境可用
- 服务健康检查端点已实现
- 自动重连逻辑已实现

**完成标准**  
- [ ] SurrealDB 故障恢复（4+ 场景）
  - [ ] 连接断开检测
  - [ ] 自动重连机制
  - [ ] 重连后数据一致性
  - [ ] 连接池恢复
- [ ] Meilisearch 故障恢复（3+ 场景）
  - [ ] 服务不可用降级（跳过搜索）
  - [ ] 服务恢复后自动重连
  - [ ] 索引同步恢复
- [ ] Embedding 服务故障（3+ 场景）
  - [ ] 服务不可用错误处理
  - [ ] 缓存命中时无需 Embedding 服务
  - [ ] 服务恢复后自动转发
- [ ] Wrapper 服务重启（3+ 场景）
  - [ ] 优雅关闭（处理完当前请求）
  - [ ] 重启后状态恢复
  - [ ] 重启后缓存预热
- [ ] 网络分区测试（2+ 场景）
  - [ ] 短暂网络中断恢复
  - [ ] 长时间网络中断处理

**验证方式**  
```bash
# 需要 Docker Compose 环境运行
uv run pytest tests/resilience/ -v

# 特定测试
uv run pytest tests/resilience/test_db_reconnect.py -v
```

**测试方法**  
```python
# 使用 Docker 控制服务启停
import subprocess
import time

def test_surrealdb_recovery():
    # 1. 停止 SurrealDB
    subprocess.run(["docker", "compose", "stop", "surrealdb"])
    
    # 2. 验证服务检测到故障
    response = client.get("/health")
    assert response.json()["surrealdb"]["status"] == "unhealthy"
    
    # 3. 启动 SurrealDB
    subprocess.run(["docker", "compose", "start", "surrealdb"])
    time.sleep(5)  # 等待恢复
    
    # 4. 验证自动恢复
    response = client.get("/health")
    assert response.json()["surrealdb"]["status"] == "connected"
```

---

### BL-T-6 [P2] E2E 测试套件整合

**目标**  
整合现有 E2E 测试，建立完整的端到端测试套件，覆盖核心用户场景。

**涉及范围**  
- 文件: `tests/e2e/test_complete_workflow.py`（新建）
- 文件: `tests/e2e/test_multi_device_sync.py`（新建）
- 整合: `tests/integration/test_api_e2e.py`
- 整合: `tests/test_websocket_e2e.py`

**前置依赖**  
- Docker Compose 完整环境可用
- 所有核心 API 端点已实现
- 测试数据准备脚本

**完成标准**  
- [ ] 完整工作流测试（3+ 场景）
  - [ ] 上传记忆 → 搜索 → 创建关系 → 图遍历
  - [ ] 代码分析 → 项目地图 → 项目统计
  - [ ] 同步预览 → 全量同步 → 冲突解决
- [ ] 多设备同步场景（3+ 场景）
  - [ ] 设备 A 上传，设备 B 查询
  - [ ] 离线编辑后同步
  - [ ] 并发编辑冲突检测与解决
- [ ] WebSocket 实时推送场景（2+ 场景）
  - [ ] 订阅记忆变更 → 上传记忆 → 接收推送
  - [ ] 断线重连后恢复订阅
- [ ] 性能基准场景（2+ 场景）
  - [ ] 1000 条记忆批量上传性能
  - [ ] 复杂搜索查询性能
- [ ] 测试报告生成
  - [ ] HTML 测试报告
  - [ ] 性能指标汇总
  - [ ] 覆盖率报告

**验证方式**  
```bash
# 完整 E2E 测试套件
uv run pytest tests/e2e/ -v --html=report.html

# 特定工作流
uv run pytest tests/e2e/test_complete_workflow.py -v

# 多设备同步
uv run pytest tests/e2e/test_multi_device_sync.py -v
```

**测试环境要求**  
```yaml
# docker-compose.e2e.yml
services:
  surrealdb: ...
  meilisearch: ...
  embedding: ...
  llm: ...
  wrapper:
    build: .
    environment:
      - WRAPPER_TEST_MODE=true
```

---

## 统计汇总

| 分类 | 总数 | P1 | P2 | 工时 |
|------|------|----|----|------|
| **测试补充** | 6 | 3 | 3 | 5 天 |

---

## 执行协议

**协议**: AGENT-COLLABORATION-PROTOCOL-v1.0  
**优先级**: P1 > P2  
**提交规范**: `test(BL-T-xxx): description`  
**测试要求**: 每个任务必须包含单元测试 + 集成测试

---

_文档版本: v3.4.0_  
_最后更新: 2026-04-15_
