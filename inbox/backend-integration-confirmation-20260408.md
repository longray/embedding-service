# 致插件端团队：代码分析 v1.4 联调确认函

**发件人**: Embedding Service (后端) 团队  
**日期**: 2026-04-08  
**主题**: Phase 2/3 API 完成确认与联调安排  

---

## 1. 后端 API 完成状态 ✅

所有 Phase 2 和 Phase 3 API 已完成实现：

### Phase 2: 调用关系 API（已完成 ✅）

| API | 端点 | 状态 | 说明 |
|-----|------|------|------|
| 批量创建调用关系 | `POST /api/v1/calls/batch` | ✅ 可用 | 最大 100 条/批次 |
| 引用查询 | `GET /api/v1/memories/{id}/references` | ✅ 可用 | 查询谁调用了该函数 |
| 依赖查询 | `GET /api/v1/memories/{id}/dependencies` | ✅ 可用 | 查询该函数调用了谁 |

### Phase 3: 代码地图与统计 API（已完成 ✅）

| API | 端点 | 状态 | 说明 |
|-----|------|------|------|
| 代码地图 | `GET /api/v1/projects/{id}/map` | ✅ 可用 | 返回文件树、模块依赖、热点文件 |
| 代码统计 | `GET /api/v1/projects/{id}/stats` | ✅ 可用 | 返回项目统计信息 |

---

## 2. 联调安排确认

### 2.1 时间确认

**原定时间**: 2026-04-11（周五）16:00-17:00  
**状态**: ✅ 后端团队确认参加

**请插件端确认**:
- [ ] 插件端可以参加 04-11 16:00 联调
- [ ] 需要调整时间（请提供备选时间）

### 2.2 联调议程建议

| 时间 | 内容 | 负责人 |
|------|------|--------|
| **16:00-16:10** | 环境检查 | 双方 |
| | - 后端服务启动状态 | 后端 |
| | - 插件端连接配置 | 插件端 |
| **16:10-16:40** | 端到端测试 | 双方 |
| | - 上传代码文件 → 获取 memory_id | 插件端 |
| | - 分析调用关系 → 批量上传 | 插件端 |
| | - 查询引用/依赖 → 验证结果 | 双方 |
| | - 获取代码地图 → 验证结构 | 双方 |
| **16:40-17:00** | 问题讨论 | 双方 |
| | - API 响应时间是否满足需求 | 双方 |
| | - 错误处理是否符合预期 | 双方 |
| | - 下一步优化计划 | 双方 |

---

## 3. 联调前准备清单

### 3.1 后端准备（已完成 ✅）

- [x] 所有 API 实现完成
- [x] 单元测试通过（21 个测试）
- [x] 测试环境就绪（localhost:17999）
- [x] API 文档更新

### 3.2 需要插件端准备

**请确认以下准备项**:

- [ ] **测试项目**: 是否已准备包含调用关系的测试项目？
- [ ] **memory_id 缓存**: memory_id → file_path 的映射是否已实现？
- [ ] **测试脚本**: 自动化测试脚本是否已准备？
- [ ] **后端地址**: 插件端配置的后端地址是否正确？
  - 后端地址: `http://localhost:17999`
  - 如需远程联调，请提前告知

---

## 4. 联调测试场景

建议按以下顺序测试：

### 场景 1: 基础调用关系（必测）

```
1. 上传 src/utils/crypto.ts（包含 hashPassword）
   → 获取 memory_id: mem_abc123

2. 上传 src/auth.ts（包含 validateUser，调用 hashPassword）
   → 获取 memory_id: mem_def456

3. 批量上传调用关系:
   POST /api/v1/calls/batch
   {
     "calls": [{
       "caller_memory_id": "mem_def456",
       "callee_memory_id": "mem_abc123",
       "line": 42,
       "column": 10
     }]
   }

4. 查询引用:
   GET /api/v1/memories/mem_abc123/references
   → 应返回 validateUser 调用信息

5. 查询依赖:
   GET /api/v1/memories/mem_def456/dependencies
   → 应返回 hashPassword 调用信息
```

### 场景 2: 代码地图（必测）

```
1. 上传多个代码文件
2. 获取项目地图:
   GET /api/v1/projects/github.com/user/repo/map
   → 验证 file_tree 结构
   → 验证 module_dependencies
   → 验证 hot_files

3. 获取项目统计:
   GET /api/v1/projects/github.com/user/repo/stats
   → 验证 total_files, total_functions 等
```

### 场景 3: 错误处理（必测）

```
1. 批量上传包含不存在 memory_id 的调用关系
   → 验证是否正确返回错误列表
   → 验证是否跳过错误继续处理其他

2. 查询不存在的 memory_id 的引用/依赖
   → 验证返回空列表而非报错
```

---

## 5. 联调环境信息

### 5.1 后端测试环境

```yaml
服务地址: http://localhost:17999
Meilisearch: http://localhost:7700
SurrealDB: http://localhost:8000

启动命令:
  cd D:/embedding_service
  uv run python -m wrapper.src.main

健康检查:
  curl http://localhost:17999/health
```

### 5.2 API 快速测试

```bash
# 测试代码地图
curl http://localhost:17999/api/v1/projects/github.com/test/repo/map

# 测试代码统计
curl http://localhost:17999/api/v1/projects/github.com/test/repo/stats

# 测试引用查询（需要先创建调用关系）
curl http://localhost:17999/api/v1/memories/memory:abc123/references

# 测试依赖查询
curl http://localhost:17999/api/v1/memories/memory:def456/dependencies
```

---

## 6. 问题与风险

### 6.1 已知限制

1. **依赖类型判断**: 目前使用简化逻辑（基于文件路径），可能不够精确
2. **函数名提取**: 从 code_analysis.functions 提取第一个函数名，可能不准确
3. **性能**: 引用/依赖查询涉及多次数据库查询，大数据量时可能较慢

### 6.2 需要讨论的问题

1. **API 响应时间**: 当前是否满足需求？是否需要优化？
2. **错误信息**: 错误提示是否清晰？是否需要更详细的错误码？
3. **分页**: 引用/依赖查询是否需要支持分页（cursor/offset）？

---

## 7. 联系方式

**联调期间联系方式**:
- **后端负责人**: Embedding Service Team
- **实时沟通**: 建议准备即时通讯工具（如微信/钉钉/Slack）
- **问题记录**: GitHub Issue 或共享文档

---

## 8. 请插件端回复

请确认以下事项：

1. **联调时间**: 04-11 16:00 是否可行？
2. **测试准备**: 测试项目和脚本是否已准备？
3. **远程联调**: 是否需要远程联调（而非 localhost）？
4. **其他需求**: 是否有其他需要后端配合的事项？

---

**期待与贵团队的联调，共同验证 v1.4 功能的完整性和稳定性！**

如有任何疑问或需要调整，请随时联系。

---

*文档版本: v1.0*  
*日期: 2026-04-08*  
*状态: 等待插件端确认*
