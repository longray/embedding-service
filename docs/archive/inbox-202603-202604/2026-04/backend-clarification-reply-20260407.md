# 致插件团队：代码分析 v1.4 技术细节回复函

**日期**: 2026-04-07  
**发件人**: OpenCode Memory 后端团队  
**主题**: 回复《代码分析 v1.4 技术细节澄清函》

---

## 概要

感谢插件团队的详细澄清函，以下逐一回复 5 个技术问题。所有回复均以双方顺利对接为前提，如有争议随时沟通。

---

## 问题 1: `calls` 数组中的 `target` 解析机制

**结论：采用 `(project_id, file_path, function_name)` 复合键方案。**

### 实现细节

后端将使用三元组 `(project_id, file_path, function_name)` 作为唯一标识来解析 `target` → `memory_id`：

```sql
-- 解析 calls[].target 为 memory_id
SELECT id FROM memories
WHERE project_id = $project_id
  AND metadata.code_analysis.file_path = $file_path
  AND metadata.code_analysis.function_name = $target;
```

### 对插件端的要求

- `calls` 数组中的每个条目**必须包含 `file_path` 字段**，用于支持跨文件调用解析：

  ```json
  {
    "target": "funcB",
    "file_path": "src/utils/helpers.ts",
    "line": 42,
    "column": 10
  }
  ```

- `file_path` 应为相对于项目根目录的路径（与上传时的 `file_path` 一致）

### 后端优化

- 将在 SurrealDB 上为 `(project_id, metadata.code_analysis.file_path, metadata.code_analysis.function_name)` 创建复合索引
- 索引将在 BL-CA-18 实施阶段一并完成
- 解析失败时（target 对应的函数尚未上传），该 `calls` 条目标记为 `unresolved`，不阻塞上传流程

---

## 问题 2: 调用关系双向查询的实现

**结论：Phase 2 仅支持单层查询；递归查询规划在 Phase 3+。**

### Phase 2（当前）行为

| 场景 | 行为 | 返回值 |
|------|------|--------|
| 目标函数已分析且关系已建立 | 正常查询 | 匹配的 memory_id 列表 |
| 目标函数未分析（关系不存在） | **静默处理** | **空数组 `[]`** |
| 目标函数已分析但无调用关系 | 正常查询 | 空数组 `[]` |

**关键约定**：关系不存在时一律返回空数组，**不返回错误**。原因：
- 项目分析是渐进式的，目标函数可能尚未上传
- 插件端不应因关系缺失而中断展示逻辑

### Phase 3+ 规划

- **递归调用链查询**：计划在 Phase 3+ 引入，支持可配置深度限制（默认 5 层，最大 10 层）
- **循环依赖检测**：递归查询内置环检测，避免无限递归
- **调用链缓存**：频繁访问的调用链结果将缓存，减少查询开销

### API 行为确认

```json
// GET /api/v1/memories/{id}/references（谁调用了我）
{
  "memory_id": "abc123",
  "references": []  // 目标未找到时返回空数组，非错误
}

// GET /api/v1/memories/{id}/dependencies（我调用了谁）
{
  "memory_id": "abc123",
  "dependencies": ["def456", "ghi789"]
}
```

---

## 问题 3: 代码地图 `file_tree` 数据来源

**结论：采用方案 B（从已上传记忆聚合）。**

### 数据来源

后端将通过查询已上传的记忆数据聚合 `file_tree`：

```sql
-- 从记忆数据聚合文件树
SELECT DISTINCT metadata.code_analysis.file_path
FROM memories
WHERE project_id = $project_id
  AND type = 'code_analysis';
```

### 对插件端的要求

- **确保项目全量上传**：代码地图的完整性依赖插件端上传了项目中所有代码文件的记忆
- 建议插件端在首次分析后提供**上传完整性校验**，确保无遗漏文件
- 如果项目中存在未上传的文件，`file_tree` 将仅反映已上传部分

### 后端处理

- 后端将从聚合的 `file_path` 列表构建树形结构（按目录层级）
- 支持文件类型过滤（如仅显示 `.ts`、`.py` 文件）
- 暂不需要插件端额外提供文件列表接口；如后续有性能问题再协商

### 补充说明

不采用方案 A（实时扫描文件系统）的原因：
- 后端是纯 API 服务，不直接访问项目文件系统
- 从记忆聚合可确保 `file_tree` 与分析数据完全一致
- 避免因代码修改与记忆数据不同步导致的信息不一致

---

## 问题 4: 复杂度字段命名统一

**结论：统一使用 `min_complexity` / `max_complexity`。**

### 确认命名

| 字段 | 类型 | 说明 |
|------|------|------|
| `min_complexity` | `integer \| null` | 函数最小圈复杂度 |
| `max_complexity` | `integer \| null` | 函数最大圈复杂度（含嵌套） |

### 统一原因

- 与设计文档 v1.4 §6.1 保持一致
- 与现有 `code_complexity` 字段的命名风格统一（简洁前缀）
- 更符合 Meilisearch filter 惯例（`min_complexity > 5` 比 `code_complexity_min > 5` 更自然）

### 后端回复函中的命名偏差

此前后端回复函中使用的 `code_complexity_min` / `code_complexity_max` 命名有误，以本回复为准。将在后端代码实现中统一使用 `min_complexity` / `max_complexity`。

---

## 问题 5: 每周同步的具体安排

**结论：同意插件团队建议，具体安排如下。**

### 时间

- **每周五 16:00 - 16:30**（北京时间，30 分钟）
- 首次同步时间：2026-04-10（本周五）

### 形式

- **以异步文档更新为主**：双方在共享文档中更新本周进展
- **即时沟通处理阻塞问题**：如遇阻塞，通过即时通讯工具快速对齐，无需等待周五同步

### 同步内容模板

```markdown
## [团队] 周报 - 2026-Wxx

### 本周完成
- [x] 任务描述（关联 BL-CA-xx）

### 下周计划
- [ ] 任务描述（关联 BL-CA-xx）

### 阻塞问题
- ⚠️ 问题描述 → 负责人 → 预期解决时间

### 技术决策
- 决策内容 → 原因 → 影响范围
```

### 补充说明

- 共享文档位置待双方确认（建议在项目仓库 `docs/` 下创建 `WEEKLY_SYNC.md`）
- 阻塞问题标注 `⚠️` 优先级，确保对方及时关注
- 非阻塞的技术讨论可异步在 GitHub Issue 中进行

---

## 下一步行动

### 后端（本周）:
- [x] 回复本澄清函（5 个问题已全部确认）
- [ ] 创建 BL-CA-18 实施分支
- [ ] 建立复合索引（问题 1 相关）
- [ ] 准备联调环境

### 插件端（本周）:
- [ ] 提供示例 CodeAnalysisResult JSON
- [ ] `calls` 条目增加 `file_path` 字段
- [ ] 确认项目全量上传策略
- [ ] 启动 BL-CA-12 (CallSymbol 提取) 实现

### 双方共同:
- [ ] 确认首次同步时间（2026-04-10 周五 16:00）
- [ ] 建立共享同步文档

---

*文档版本: v1.0*  
*日期: 2026-04-07*  
*状态: 已确认，待双方执行*
