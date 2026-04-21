# BACKLOG v3.2

> **版本**: v3.2.0  
> **创建日期**: 2026-04-12  
> **最后更新**: 2026-04-21  
> **总任务数**: 73  
> **已完成**: 66（已归档）  
> **未完成**: 7  
> **预估总工时**: 28 天  
> **协议**: AGENT-COLLABORATION-PROTOCOL-v1.0

**历史场景**: 场景 1-4 已完成并归档至 [backlog_archive.md](./backlog_archive.md)

**📊 完成状态**: ✅ **90%** (66/73 任务完成)

**📚 相关文档**:

- [产品文档](./PRODUCT.md) - 面向终端用户
- [开发文档](./DEVELOPMENT.md) - 面向开发人员
- [v3.2 设计文档](./v3.2/) - 详细设计规范
- [BACKLOG v3.3](./BACKLOG-v3.3.md) - PrecomputeService + Stub 端点（8 个任务，100% 完成）
- [归档任务](./backlog_archive.md) - 已完成的 66 个任务详情

---

## 快速导航

- [任务总览（表格）](#任务总览表格)
- [测试补充 (v3.4)](#测试补充-v34)
- [Phase 8: API 优化](#phase-8-api-优化)
- [Phase 9: 代码审查修复](#phase-9-代码审查修复-v321)
- [统计汇总](#统计汇总)

---

## 任务总览（表格）

### 已完成任务（已归档）

| 阶段 | 任务数 | 说明 | 归档位置 |
|------|--------|------|----------|
| Phase 1-7 | 40 | 核心功能开发 | [backlog_archive.md](./backlog_archive.md) |
| WebSocket 后续 | 12 | WebSocket 集成优化 | [backlog_archive.md](./backlog_archive.md) |
| PrecomputeService 后续 | 9 | Precompute 集成优化 | [backlog_archive.md](./backlog_archive.md) |
| 文档 | 8 | 文档完善 | [backlog_archive.md](./backlog_archive.md) |
| **小计** | **66** | **全部完成** | - |

### 未完成任务

| 编号 | 目标 | 优先级 | 工时 | 状态 | 详情 |
|------|------|--------|------|------|------|
| **测试补充 (v3.4)** |
| BL-T-1 | Audit 日志端点测试 | P1 | 1 天 | ⏳ | [详情](#bl-t-1-p1-audit-日志端点测试) |
| BL-T-2 | Projects API 测试 | P1 | 1 天 | ⏳ | [详情](#bl-t-2-p1-projects-api-测试) |
| BL-T-3 | Lookup API 测试 | P1 | 0.5 天 | ⏳ | [详情](#bl-t-3-p1-lookup-api-测试) |
| BL-T-4 | 并发压力测试 | P2 | 1 天 | ⏳ | [详情](#bl-t-4-p2-并发压力测试) |
| BL-T-5 | 故障恢复测试 | P2 | 1 天 | ⏳ | [详情](#bl-t-5-p2-故障恢复测试) |
| BL-T-6 | E2E 测试套件整合 | P2 | 0.5 天 | ⏳ | [详情](#bl-t-6-p2-e2e-测试套件整合) |
| **Phase 8** |
| BL-B-80 | 代码指纹增量同步 API | P0 | 2 天 | 🆕 | [详情](#bl-b-80-p0-代码指纹增量同步-api) |
| BL-B-81 | PrecomputeService 代码分析 API | P0 | 2 天 | 🆕 | [详情](#bl-b-81-p0-precomputeservice-代码分析-api) |
| BL-B-82 | 集成测试环境部署 | P1 | 1 天 | 🆕 | [详情](#bl-b-82-p1-集成测试环境部署) |
| BL-B-83 | 符号查询 API | P3 | 3-5 天 | ⏸️ | [详情](#bl-b-83-p3-符号查询-api) |
| **Phase 9** |
| BL-B-84 | 封装性修复：添加 db 公开属性 | P0 | 0.5 天 | 🆕 | [详情](#bl-b-84-p0-封装性修复添加-db-公开属性) |
| BL-B-85 | 统一 _extract_records 实现 | P0 | 0.5 天 | 🆕 | [详情](#bl-b-85-p0-统一-extractrecords-实现) |
| BL-B-86 | PrecomputeService 生命周期管理 | P0 | 1 天 | 🆕 | [详情](#bl-b-86-p0-precomputeservice-生命周期管理) |
| BL-B-87 | 代码指纹批量 SQL 优化 | P1 | 0.5 天 | 🆕 | [详情](#bl-b-87-p1-代码指纹批量-sql-优化) |
| BL-B-88 | 添加事务保护 | P1 | 0.5 天 | 🆕 | [详情](#bl-b-88-p1-添加事务保护) |
| BL-B-89 | 测试质量提升 | P2 | 1 天 | 🆕 | [详情](#bl-b-89-p2-测试质量提升) |
| **场景十四** |
| BL-B-32 | 修复 Entity GET 端点返回 404 | P0 | 2 小时 | 🆕 | [详情](#bl-b-32-p0-修复-entity-get-端点返回-404) |
| BL-B-33 | 修复 Reference 创建失败 | P0 | 2 小时 | 🆕 | [详情](#bl-b-33-p0-修复-reference-创建失败) |

---

## 场景十四：后端 Bug 修复（v3.2.3）

> **背景**: Layer 1 数据层测试发现 2 个严重后端 Bug
>
> **目标**: 修复 Entity GET 端点和 Reference 创建端点的 Bug
>
> **状态**: 🆕 新建

---

### BL-B-32 [P0] 修复 Entity GET 端点返回 404

**目标**: 修复 `GET /api/v1/entities/:id` 端点返回 404 的问题，即使 Entity 刚创建成功

**涉及范围**:

1. `wrapper/src/routers/entities.py` — 检查 GET 端点实现
2. `wrapper/src/services/entity_service.py` — 检查查询逻辑
3. SurrealDB 查询语句 — 验证 ID 格式匹配

**前置依赖**:

- 无

**完成标准**:

1. `GET /api/v1/entities/{id}` 返回正确的 Entity 数据
2. 支持 `level` 参数（0=abstract, 1=overview, 2=full）
3. 响应时间 < 50ms
4. 现有 Entity 创建测试不受影响

**验证方式**:

1. **API 测试**: 创建 Entity 后立即查询

   ```bash
   # 创建
   curl -X POST http://localhost:18008/api/v1/entities \
     -H "Content-Type: application/json" \
     -H "WRAPPER_MEILI_API_KEY: your-key" \
     -d '{"type":"code","abstract":"test"}'

   # 查询（应返回 200）
   curl http://localhost:18008/api/v1/entities/{id} \
     -H "WRAPPER_MEILI_API_KEY: your-key"
   ```

2. **Level 参数测试**: 验证 level=0/1/2 返回正确字段
3. **单元测试**: Mock SurrealDB 查询，验证逻辑正确
4. **集成测试**: 实际调用后端 API

**工时**: 2 小时

**状态**: 🆕 新建

---

### BL-B-33 [P0] 修复 Reference 创建失败（from_id 不存在）

**目标**: 修复 `POST /api/v1/references` 返回 "from_id 不存在" 的问题，即使 Atom 确实存在

**涉及范围**:

1. `wrapper/src/routers/references.py` — 检查 POST 端点实现
2. `wrapper/src/services/reference_service.py` — 检查 Atom 存在性验证逻辑
3. SurrealDB 查询 — 验证 Atom ID 解析逻辑

**前置依赖**:

- 无

**完成标准**:

1. `POST /api/v1/references` 成功创建 Reference
2. 正确验证 `from_id` 和 `to_id` 存在性
3. 支持所有 Reference 类型（calls, imports, extends, implements, related）
4. 响应时间 < 100ms

**验证方式**:

1. **API 测试**: 创建 Atom 后创建 Reference

   ```bash
   # 创建 Atom
   curl -X POST http://localhost:18008/api/v1/atoms \
     -H "Content-Type: application/json" \
     -H "WRAPPER_MEILI_API_KEY: your-key" \
     -d '{"type":"function","name":"testFunc"}'

   # 创建 Reference（应返回 201）
   curl -X POST http://localhost:18008/api/v1/references \
     -H "Content-Type: application/json" \
     -H "WRAPPER_MEILI_API_KEY: your-key" \
     -d '{"from_id":"atom:xxx","to_id":"atom:yyy","type":"calls"}'
   ```

2. **存在性验证测试**: 验证不存在的 ID 返回 400
3. **单元测试**: Mock SurrealDB 查询，验证逻辑正确
4. **集成测试**: 实际调用后端 API

**工时**: 2 小时

**状态**: 🆕 新建

---

## 测试补充 (v3.4)

### BL-T-1 [P1] Audit 日志端点测试

**目标**  
测试 `/api/v1/access-log` 端点的功能和性能。

**涉及范围**  

- 文件: `tests/test_audit_log.py`（新建）
- 端点: `POST /api/v1/access-log`
- 场景: 正常上报、批量上报、错误处理

**前置依赖**  
无

**完成标准**  

- [ ] 单条日志上报测试
- [ ] 批量日志上报测试
- [ ] 错误参数处理测试
- [ ] 性能测试（>1000条/秒）

**验证方式**  

```bash
uv run pytest tests/test_audit_log.py -v
```

---

### BL-T-2 [P1] Projects API 测试

**目标**  
测试 Projects 相关 API 的完整功能。

**涉及范围**  

- 文件: `tests/test_projects_api.py`（新建）
- 端点: `/api/v1/projects/*`
- 场景: CRUD、统计、代码地图

**前置依赖**  
无

**完成标准**  

- [ ] 项目 CRUD 测试
- [ ] 代码地图 API 测试
- [ ] 代码统计 API 测试
- [ ] 权限控制测试

**验证方式**  

```bash
uv run pytest tests/test_projects_api.py -v
```

---

### BL-T-3 [P1] Lookup API 测试

**目标**  
测试记忆查询 API（source_id/hash/file_path）。

**涉及范围**  

- 文件: `tests/test_lookup_api.py`（新建）
- 端点: `GET /api/v1/memories/lookup`
- 场景: 各种查询参数组合

**前置依赖**  
无

**完成标准**  

- [ ] source_id 查询测试
- [ ] hash 查询测试
- [ ] file_path 查询测试
- [ ] 组合条件查询测试

**验证方式**  

```bash
uv run pytest tests/test_lookup_api.py -v
```

---

### BL-T-4 [P2] 并发压力测试

**目标**  
进行高并发压力测试，验证系统稳定性。

**涉及范围**  

- 文件: `tests/stress/test_concurrent.py`（新建）
- 工具: locust 或自定义脚本
- 指标: 吞吐量、延迟、错误率

**前置依赖**  
BL-T-1~T-3 完成

**完成标准**  

- [ ] 1000+ 并发连接
- [ ] 持续运行 1 小时
- [ ] 错误率 < 0.1%
- [ ] 内存无泄漏

**验证方式**  

```bash
uv run python tests/stress/test_concurrent.py --clients 1000 --duration 3600
```

---

### BL-T-5 [P2] 故障恢复测试

**目标**  
测试系统在各种故障场景下的恢复能力。

**涉及范围**  

- 文件: `tests/stress/test_recovery.py`（新建）
- 场景: DB 断开、网络中断、服务重启

**前置依赖**  
BL-T-4 完成

**完成标准**  

- [ ] DB 断开恢复测试
- [ ] 网络中断恢复测试
- [ ] 服务重启恢复测试
- [ ] 数据一致性验证

**验证方式**  

```bash
uv run pytest tests/stress/test_recovery.py -v
```

---

### BL-T-6 [P2] E2E 测试套件整合

**目标**  
整合所有 E2E 测试，形成完整的测试套件。

**涉及范围**  

- 文件: `tests/e2e/`（目录）
- 包含: WebSocket、API、同步、搜索

**前置依赖**  
BL-T-1~T-5 完成

**完成标准**  

- [ ] E2E 测试分类整理
- [ ] 测试数据准备自动化
- [ ] CI/CD 集成
- [ ] 测试报告生成

**验证方式**  

```bash
uv run pytest tests/e2e/ -v --html=report.html
```

---

## Phase 8: API 优化

### BL-B-80 [P0] 代码指纹增量同步 API

**目标**  
实现代码指纹的增量同步 API，支持多设备同步。

**涉及范围**  

- 文件: `wrapper/src/routers/fingerprint_sync.py`（新建）
- 端点: `/api/v1/fingerprints/sync`
- 功能: 增量同步、冲突解决

**前置依赖**  
无

**完成标准**  

- [ ] 指纹列表获取 API
- [ ] 增量同步 API
- [ ] 冲突检测与解决
- [ ] 性能优化（<1s）

**验证方式**  

```bash
uv run pytest tests/test_fingerprint_sync.py -v
```

---

### BL-B-81 [P0] PrecomputeService 代码分析 API

**目标**  
提供代码分析的 REST API 接口。

**涉及范围**  

- 文件: `wrapper/src/routers/code_analysis.py`（新建）
- 端点: `/api/v1/code/analyze`
- 功能: 代码解析、关系提取

**前置依赖**  
无

**完成标准**  

- [ ] 代码上传接口
- [ ] 异步分析任务
- [ ] 分析结果查询
- [ ] 进度回调

**验证方式**  

```bash
uv run pytest tests/test_code_analysis_api.py -v
```

---

### BL-B-82 [P1] 集成测试环境部署

**目标**  
搭建自动化集成测试环境。

**涉及范围**  

- 文件: `docker-compose.test.yml`（新建）
- 配置: 测试数据库、Mock 服务

**前置依赖**  
BL-B-80~B-81 完成

**完成标准**  

- [ ] Docker Compose 测试环境
- [ ] 自动化测试脚本
- [ ] CI/CD 集成
- [ ] 测试数据准备

**验证方式**  

```bash
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

---

### BL-B-83 [P3] 符号查询 API

**目标**  
实现符号级别的代码查询 API。

**涉及范围**  

- 文件: `wrapper/src/routers/symbol_query.py`（新建）
- 端点: `/api/v1/symbols/*`
- 功能: 符号搜索、引用查询

**前置依赖**  
BL-B-81 完成

**完成标准**  

- [ ] 符号搜索 API
- [ ] 引用查询 API
- [ ] 定义跳转 API
- [ ] 模糊匹配支持

**验证方式**  

```bash
uv run pytest tests/test_symbol_query.py -v
```

---

## Phase 9: 代码审查修复 (v3.2.1)

### BL-B-84 [P0] 封装性修复：添加 db 公开属性

**目标**  
修复 SurrealDBManager 封装性问题，添加公开的 db 属性。

**涉及范围**  

- 文件: `wrapper/src/utils/surrealdb_manager.py`（修改）
- 添加: `db` 公开属性

**前置依赖**  
无

**完成标准**  

- [ ] 添加 `db` 属性
- [ ] 向后兼容
- [ ] 文档更新

**验证方式**  

```python
from wrapper.src.utils.surrealdb_manager import SurrealDBManager
manager = SurrealDBManager()
assert manager.db is not None
```

---

### BL-B-85 [P0] 统一 _extract_records 实现

**目标**  
统一所有模块的 `_extract_records` 方法实现。

**涉及范围**  

- 文件: 多个 router 文件
- 方法: `_extract_records()`

**前置依赖**  
无

**完成标准**  

- [ ] 提取公共函数
- [ ] 统一错误处理
- [ ] 统一日志记录

**验证方式**  

```bash
uv run pytest tests/test_extract_records.py -v
```

---

### BL-B-86 [P0] PrecomputeService 生命周期管理

**目标**  
完善 PrecomputeService 的生命周期管理。

**涉及范围**  

- 文件: `wrapper/src/services/precompute.py`（修改）
- 功能: 启动、停止、健康检查

**前置依赖**  
无

**完成标准**  

- [ ] 完善 start/stop 方法
- [ ] 添加健康检查
- [ ] 优雅关闭支持

**验证方式**  

```bash
uv run pytest tests/test_precompute_lifecycle.py -v
```

---

### BL-B-87 [P1] 代码指纹批量 SQL 优化

**目标**  
优化代码指纹批量查询的 SQL 性能。

**涉及范围**  

- 文件: `wrapper/src/services/fingerprint.py`（修改）
- 优化: 批量查询 SQL

**前置依赖**  
无

**完成标准**  

- [ ] 使用参数化查询
- [ ] 批量查询优化
- [ ] 索引优化

**验证方式**  

```bash
uv run python scripts/benchmark_fingerprint.py
```

---

### BL-B-88 [P1] 添加事务保护

**目标**  
为关键操作添加数据库事务保护。

**涉及范围**  

- 文件: 多个 service 文件
- 功能: BEGIN/COMMIT/ROLLBACK

**前置依赖**  
无

**完成标准**  

- [ ] 识别关键操作
- [ ] 添加事务装饰器
- [ ] 错误回滚处理

**验证方式**  

```bash
uv run pytest tests/test_transaction.py -v
```

---

### BL-B-89 [P2] 测试质量提升

**目标**  
提升测试覆盖率和质量。

**涉及范围**  

- 文件: 多个 test 文件
- 指标: 覆盖率 >90%

**前置依赖**  
BL-B-84~B-88 完成

**完成标准**  

- [ ] 覆盖率提升到 90%
- [ ] 边界条件测试
- [ ] 异常路径测试

**验证方式**  

```bash
uv run pytest tests/ --cov=wrapper/src --cov-report=html
```

---

## 统计汇总

### 任务统计

| 分类 | 总数 | 已完成 | 未完成 | 完成率 |
|------|------|--------|--------|--------|
| 核心功能 | 40 | 40 | 0 | 100% |
| WebSocket | 12 | 12 | 0 | 100% |
| Precompute | 9 | 9 | 0 | 100% |
| 文档 | 8 | 8 | 0 | 100% |
| 测试补充 | 6 | 0 | 6 | 0% |
| Phase 8 | 4 | 0 | 4 | 0% |
| Phase 9 | 6 | 0 | 6 | 0% |
| 场景十四 | 2 | 0 | 2 | 0% |
| **总计** | **73** | **66** | **7** | **90%** |

### 工时统计

| 分类 | 预估工时 | 已完成工时 |
|------|----------|------------|
| 核心功能 | 28 天 | 28 天 |
| 测试补充 | 5 天 | 0 天 |
| Phase 8 | 8 天 | 0 天 |
| Phase 9 | 4 天 | 0 天 |
| **总计** | **45 天** | **28 天** |

### 优先级分布

| 优先级 | 任务数 | 状态 |
|--------|--------|------|
| P0 | 15 | 12 完成，3 未完成 |
| P1 | 35 | 32 完成，3 未完成 |
| P2 | 18 | 17 完成，1 未完成 |
| P3 | 3 | 3 完成，0 未完成 |

---

_最后更新: 2026-04-21_
