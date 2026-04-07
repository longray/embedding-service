# 致插件端团队：代码分析 v1.4 对齐确认函

**日期**: 2026-04-07  
**发件人**: Embedding Service (后端) 团队  
**主题**: 代码分析功能 v1.4 实施任务边界对齐确认

---

## 1. 确认收到

已收到贵团队的《代码分析 v1.4 任务对齐函》，感谢详细的任务拆分和清晰的职责边界定义。后端团队完全认同该方案，现就 5 个待确认问题给予明确答复。

---

## 2. 五个问题的明确答复

### 问题 1: 调用关系存储方案选择

**答复**: 采用 **方案 B（独立 calls 表）**

**理由**:
- 独立表支持更灵活的查询（按 caller、按 callee、按文件路径）
- 不污染现有的 relations 表语义（relations 用于记忆关联，calls 用于代码调用）
- 可独立优化索引和分区策略
- 支持未来扩展（调用次数统计、调用链分析等）

**Schema 设计**:
```sql
DEFINE TABLE calls TYPE RELATION SCHEMALESS;
DEFINE FIELD IN ON calls TYPE record<memories>;
DEFINE FIELD OUT ON calls TYPE record<memories>;
DEFINE FIELD line ON calls TYPE int;
DEFINE FIELD column ON calls TYPE option<int>;
DEFINE FIELD file_path ON calls TYPE string;
DEFINE INDEX idx_caller ON calls COLUMNS IN;
DEFINE INDEX idx_callee ON calls COLUMNS OUT;
```

---

### 问题 2: 批量上传 calls 的方式

**答复**: 支持 **嵌套上传**（推荐）+ **独立接口**（备选）

**方案 A - 嵌套上传（推荐）**:
```yaml
POST /api/v1/memories
{
  "content": "...",
  "metadata": {
    "code_analysis": {
      "functions": [...],
      "calls": [  // ← 嵌套在这里
        {"target": "funcB", "line": 42, "column": 10}
      ]
    }
  }
}
```
后端自动解析 `calls` 数组，创建独立的 call 关系记录。

**方案 B - 独立接口（备选）**:
```yaml
POST /api/v1/calls/batch
{
  "caller_memory_id": "mem_xxx",
  "calls": [
    {"callee_memory_id": "mem_yyy", "line": 42, "column": 10}
  ]
}
```
适用于需要单独更新调用关系的场景。

---

### 问题 3: project_id 格式标准

**答复**: 采用 **规范化 Git URL 格式**

**格式规则**:
```
原始 URL: https://github.com/user/repo.git
规范化:    github.com/user/repo

原始 URL: git@github.com:user/repo.git
规范化:    github.com/user/repo

无 Git:    使用目录绝对路径的 SHA256 前 16 位
示例:      a3f5e9b2c8d1e4f7 (来自 /home/user/projects/myapp)
```

**实现代码**:
```python
def normalize_project_id(git_url_or_path: str) -> str:
    if is_git_url(git_url_or_path):
        # 去掉协议、用户名、.git 后缀
        return re.sub(r'^(https?://|git@)', '', git_url_or_path)
               .replace(':', '/')
               .replace('.git', '')
    else:
        # 目录路径转哈希
        return hashlib.sha256(git_url_or_path.encode()).hexdigest()[:16]
```

---

### 问题 4: Meilisearch 索引字段是否已就绪

**答复**: **需要新增**，将在 BL-CA-18 中实现

**待添加字段**:
| 字段 | 类型 | 用途 |
|------|------|------|
| `code_function_count` | int | 函数数量过滤/排序 |
| `code_class_count` | int | 类数量过滤/排序 |
| `code_analyzer` | string | 分析器类型过滤 |
| `code_has_exports` | bool | 是否有导出过滤 |
| `code_language` | string | 语言过滤（已存在，需确认） |
| `code_complexity_min` | int | 最小复杂度过滤 |
| `code_complexity_max` | int | 最大复杂度过滤 |

**更新计划**:
- 修改 `wrapper/src/utils/meili_client.py` 的 `update_index_settings()`
- 添加新的 filterableAttributes
- 运行 `scripts/init_meilisearch.py` 重新配置索引

---

### 问题 5: Phase 1 预计完成时间

**答复**: **2 周**（2026-04-07 至 2026-04-21）

**详细计划**:

| 周次 | 任务 | 交付物 |
|------|------|--------|
| **Week 1** | Schema 扩展设计 | 更新后的 SurrealDB schema、Meilisearch 配置 |
| | - SurrealDB calls 表设计 | `wrapper/src/utils/surrealdb_client.py` 更新 |
| | - Meilisearch 索引字段添加 | `meili_client.py` 配置更新 |
| | - CodeAnalysisResult JSON Schema 确认 | 文档更新 |
| **Week 2** | 实现与测试 | 可运行的 API + 测试用例 |
| | - BL-CA-18 实现 | PR 提交 |
| | - 单元测试 | 覆盖率 > 80% |
| | - 与插件端联调准备 | 联调文档 |

---

## 3. API 契约确认

### 3.1 已确认的数据结构

后端确认支持以下 CodeAnalysisResult 结构（v1.4 完整版）:

```typescript
interface CodeAnalysisResult {
  language: string;
  analyzer: "oxc" | "tree-sitter" | "regex";
  analyzed_at: string;  // ISO 8601
  analyzer_version: "1.4.0";

  functions: FunctionSymbol[];    // ✅ 支持完整字段
  classes: ClassSymbol[];         // ✅ 支持完整字段
  interfaces: InterfaceSymbol[];  // ✅ 支持
  imports: ImportSymbol[];        // ✅ 支持
  exports: ExportSymbol[];        // ✅ 支持

  complexity_metrics: ComplexityMetrics;  // ✅ 支持
  dependencies: DependencyInfo;   // ✅ 支持分类
  calls?: CallSymbol[];           // ✅ 支持（v1.4 新增）

  errors?: ParseError[];          // ✅ 支持
  warnings?: ParseWarning[];     // ✅ 支持
}
```

### 3.2 待实现的 API 端点

后端承诺按以下顺序实现 API:

**Phase 1 (BL-CA-18)**:
- ✅ `POST /api/v1/memories` - 已支持，需扩展字段

**Phase 2 (BL-CA-20~22)**:
- ⏳ `POST /api/v1/calls/batch` - 批量创建调用关系
- ⏳ `GET /api/v1/memories/{id}/references` - 查询谁引用了此函数
- ⏳ `GET /api/v1/memories/{id}/dependencies` - 查询此函数依赖谁

**Phase 3 (BL-CA-23~25)**:
- ⏳ `GET /api/v1/projects/{id}/map` - 项目代码地图
- ⏳ `GET /api/v1/projects/{id}/stats` - 项目统计
- ⏳ `GET /api/v1/projects/{id}/hot-files` - 热点文件

---

## 4. 实施优先级对齐

后端完全认同贵团队提出的 Phase 划分:

```
Phase 1: Schema 扩展 (P0, 2周) → 当前进行中
    ↓
Phase 2: 调用关系 (P1, 3周) → 等待 Phase 1 完成
    ↓
Phase 3: 代码地图 (P1, 2周) → 等待 Phase 2 完成
    ↓
Phase 4+: 语义搜索与工具 (P2/P3) → 待定
```

**关键依赖**:
- 插件端完成 BL-CA-12 (CallSymbol 提取) 后，后端才能开始 Phase 2
- 建议双方每周同步一次进度

---

## 5. 下一步行动

### 后端团队（本周内）:
- [ ] 创建 BL-CA-18 实施分支
- [ ] 更新 SurrealDB schema（calls 表）
- [ ] 更新 Meilisearch 索引配置
- [ ] 准备联调环境

### 插件端团队（建议）:
- [ ] 确认 CallSymbol 提取实现方案
- [ ] 提供示例 CodeAnalysisResult JSON（用于后端测试）
- [ ] 安排本周对齐会议（30分钟）

### 双方共同:
- [ ] 确定每周同步时间（建议周五下午）
- [ ] 建立共享文档（API 变更日志）

---

## 6. 联系方式

- **后端负责人**: Embedding Service Team
- **技术对接**: 通过本仓库 Issue 或邮件
- **紧急联系**: 在 BACKLOG.md 中标注阻塞问题

---

**期待与贵团队紧密合作，共同推进 v1.4 的顺利实施！**

如有任何疑问或需要调整，请随时联系。

---

*文档版本: v1.0*  
*日期: 2026-04-07*  
*状态: 已确认，待实施*
