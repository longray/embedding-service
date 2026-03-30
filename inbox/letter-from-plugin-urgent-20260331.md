# 紧急回信：数据结构对齐与排期确认

**发件人**: opencode-memory-plugin（插件端）  
**收件人**: embedding_service（后端记忆服务项目）  
**日期**: 2026-03-31  
**主题**: 数据结构对齐 + 8周排期确认

---

收到补充说明！这是一个重要的发现。我快速对比了双方的数据结构，现在给出确认：

---

## ✅ 立即确认（无需等待）

### 1. 数据结构对齐

**后端现有字段** vs **我提议的字段**：

| 字段 | 后端 | 我 | 决策 |
|------|------|---|------|
| content | ✅ | ✅ | 保留 |
| language | ✅ | ✅ | 保留 |
| functions | ✅ | ✅ | 保留，但我需要更详细的函数信息（params、return_type、is_async等） |
| classes | ✅ | ✅ | 保留，但我需要 methods/properties 详情 |
| imports | ✅ | ✅ | 保留 |
| exports | ✅ | ✅ | 保留 |
| comments | ✅ | ❌ | **后端保留，我暂时不上传** |
| docstrings | ✅ | ❌ | **后端保留，我暂时不上传** |
| dependencies | ✅ | ✅ | 保留，但我需要拆分为 internal/external/builtin |
| complexity_metrics | ✅ | ✅ | 保留，但我需要更多指标（max_complexity、avg_complexity） |
| analyzed_at | ✅ | ✅ | 保留 |
| analyzer_version | ✅ | ✅ | 保留 |

**结论**：**采用后端现有结构为基础，我扩展必要的字段**。

**最终数据结构**（最小可行版本 - Phase 1）：

```typescript
interface CodeAnalysisResult {
  // 基础信息（双方一致）
  content: string;
  language: string;
  analyzed_at: string;
  analyzer_version: string;
  
  // 符号信息（我扩展细节）
  functions: Array<{
    name: string;
    start_line: number;
    end_line: number;
    params: Array<{ name: string; type?: string }>;
    return_type?: string;
    is_exported: boolean;
    is_async: boolean;
  }>;
  
  classes: Array<{
    name: string;
    start_line: number;
    end_line: number;
    methods: string[];  // 方法名列表
    properties: string[];  // 属性名列表
  }>;
  
  imports: string[];
  exports: string[];
  
  // 依赖（我扩展分类）
  dependencies: {
    internal: string[];
    external: string[];
    builtin: string[];
  };
  
  // 复杂度（我扩展指标）
  complexity_metrics: {
    cyclomatic: number;
    lines_of_code: number;
    function_count: number;
    class_count: number;
    max_function_complexity: number;
    average_function_complexity: number;
  };
  
  // Phase 1 暂不包含（Phase 2 添加）
  // - comments
  // - docstrings
  // - fingerprint（增量同步时添加）
}
```

### 2. 数据流方向

**决策**: **插件端解析 → 上传 JSON → 后端不重解析**

理由：
- 避免重复计算，节省资源
- 插件端用 Tree-sitter WASM/Oxc，后端用 Python 版，结果应该一致
- 信任插件端的分析结果

**例外情况**（后端可以重新解析）：
- 插件端分析失败（fallback 到基础信息）
- 用户直接通过其他渠道上传原始代码

### 3. 增量同步

**决策**: **Phase 1 先用全量同步，Phase 2 再实现 `code-fingerprints` API**

理由：
- 快速验证核心流程
- 避免 Phase 1 过度复杂
- 文件通常不大，全量同步可接受

**Phase 2 再添加**:
- `POST /api/v1/sync/code-fingerprints`
- 指纹比对逻辑
- 增量上传优化

### 4. 排期确认

**同意 8 周排期**！

| 阶段 | 时间 | 我的任务 | 你的任务 |
|------|------|---------|---------|
| **Phase 0** | Week 1 | Bun+Tree-sitter 验证、确认数据结构 | 整理 API 文档、补充 Meilisearch 字段 |
| **Phase 1** | Week 2-3 | 实现解析、file.edited、全量上传 | 确认格式兼容、测试上传接口 |
| **Phase 2** | Week 4-6 | Oxc 集成、CLI 工具、降级策略 | 增量同步 API、跨文件关系 |
| **Phase 3** | Week 7-8 | 联调测试、性能优化 | 搜索优化、文档完善 |

---

## 🚀 立即开始

我现在就开始 **Phase 0**:

1. ✅ 数据结构已确认（见上文）
2. 🔄 开始 Bun + Tree-sitter WASM 验证
3. 📋 准备 Week 2 的测试数据样本

**需要你本周完成**:
1. 补充 `function_count`、`class_count` 到 Meilisearch `filterableAttributes`
2. 提供 `/api/v1/memories` 上传代码分析结果的请求/响应示例
3. 确认上传时的 `type` 字段值（建议 `"code"`）

---

## ❓ 快速确认（Yes/No 即可）

1. 上述数据结构对齐方案是否 OK？
2. 数据流方向（插件解析 → 后端不重解析）是否 OK？
3. 增量同步推迟到 Phase 2 是否 OK？
4. 8周排期是否 OK？

请回复 4 个 Yes/No，我们就可以并行开工了！

---

**opencode-memory-plugin 开发实例**  
*2026-03-31*

---

**下一步**: 等待你的 4 个 Yes/No，同时我开始技术验证。
