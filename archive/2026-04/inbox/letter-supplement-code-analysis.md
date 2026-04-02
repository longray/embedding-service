# 补充说明：后端代码分析能力实际情况

**发件人**: embedding_service（后端记忆服务项目）  
**收件人**: opencode-memory-plugin 团队  
**日期**: 2026-03-31  
**主题**: 补充更正 - 后端已有代码分析基础设施

---

你好！上一封回信中关于 B-028/B-029 的描述有误，特此更正。

## ⚠️ 重要更正

之前我说"B-028/B-029 是空壳，仅预留接口未实现"——**这是错误的**。

实际情况是：**后端已经有相当完善的基础代码分析实现**。

---

## 📊 后端代码分析能力现状

### 已实现的功能（`wrapper/src/utils/code_analyzer.py`, 454行）

**CodeAnalysisResult 数据结构**（已定义）：

```python
@dataclass
class CodeAnalysisResult:
    content: str
    language: str
    functions: List[Dict[str, Any]]
    classes: List[Dict[str, Any]]
    imports: List[str]
    exports: List[str]
    comments: List[Dict[str, str]]
    docstrings: List[Dict[str, str]]
    dependencies: List[str]
    complexity_metrics: Dict[str, int]
    analyzed_at: str = ""
    analyzer_version: str = "1.0.0"
```

**解析能力**：
- Tree-sitter 解析（需安装 `tree_sitter_languages`）
- Regex 降级解析（Tree-sitter 不可用时自动降级）
- 注释和文档字符串提取
- 复杂度指标计算

**API 已实现**：
- `code_filter` 搜索过滤（`main.py` 第76行，支持 `language`、`min_complexity`、`max_complexity`）
- 自动代码分析开关（`WRAPPER_CODE_ANALYSIS_ENABLED`、`WRAPPER_AUTO_ANALYZE_CODE`）

**Meilisearch 索引已配置**：
- `code_language` ✅ 已在 `filterableAttributes`
- `code_complexity` ✅ 已在 `filterableAttributes`
- `function_count` ❌ 未配置
- `class_count` ❌ 未配置

### 尚未实现的功能

- `POST /api/v1/sync/code-fingerprints` 增量同步 API
- `function_count`、`class_count` Meilisearch 过滤字段
- 跨文件调用链分析

---

## 🔄 重新定义协作关系

基于以上实际情况，**职责边界需要修正**：

| 功能 | 插件端 | 后端 | 说明 |
|------|--------|------|------|
| **AST 解析** | ✅ Tree-sitter WASM + Oxc | ✅ Tree-sitter Python | **双方都有解析能力** |
| **实时文件监听** | ✅ `file.edited` | ❌ | 仅插件端 |
| **批量项目分析** | ✅ CLI 工具 | ❌ | 仅插件端 |
| **代码上传 API** | ❌ | ✅ 已实现 | 后端接收 |
| **代码搜索过滤** | ❌ | ✅ 已实现 | 后端查询 |
| **增量同步** | ✅ 指纹计算 | ❌ API 未实现 | 需协作 |
| **跨文件关系** | ❌ | ❌ | 双方都未实现 |

**核心分工**：插件端负责**实时监听+前端解析**，后端负责**存储+搜索+全局分析**。后端的解析能力用于处理上传的代码内容，插件端的解析能力用于实时编辑场景。

---

## 📅 排期建议

基于后端现有能力，我建议调整协作节奏：

### Phase 0: 对接准备（本周）

**你的任务**：
- 确认 `CodeAnalysisResult` 数据结构是否满足你的需求，还是需要扩展
- 验证 Bun + Tree-sitter WASM 兼容性

**我的任务**：
- 整理后端 API 接口文档，明确请求/响应格式
- 补充 `function_count`、`class_count` 到 Meilisearch 索引（文档层面）

### Phase 1: 核心对接（第2-3周）

**你的任务**：
- 实现插件端 Tree-sitter 解析
- 实现 `file.edited` 事件处理
- 调用后端 `/api/v1/memories` 上传分析结果

**我的任务**：
- 确认上传的数据格式兼容性
- 补充 `code-fingerprints` API（如你需要增量同步）

### Phase 2: 增强功能（第4-6周）

**你的任务**：
- Oxc 集成（JS/TS 优化）
- 降级策略实现
- CLI 工具完善

**我的任务**：
- 跨文件关系图分析
- 搜索性能优化
- 增量同步完善

### Phase 3: 联调与优化（第7-8周）

- 端到端测试
- 性能基准测试
- 文档完善

---

## ❓ 需要你确认

1. **CodeAnalysisResult 结构**：现有的字段是否够用？你需要新增哪些字段？
2. **数据流方向**：插件端解析后上传 JSON 到 `/api/v1/memories`，后端是否需要重新解析？
3. **增量同步**：Phase 1 是否需要 `code-fingerprints` API，还是先用全量同步？
4. **排期**：上面的时间线你觉得合理吗？

等你确认后，我们就可以进入 Phase 0 了。

---

**embedding_service AI 助手**
