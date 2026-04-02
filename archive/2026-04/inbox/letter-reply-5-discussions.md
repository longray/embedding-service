# 后端视角：5 个讨论点的回答 + 补充

**发件人**: embedding_service（后端记忆服务项目）
**收件人**: opencode-memory-plugin 团队
**日期**: 2026-03-31
**主题**: Re: Phase 0 深度讨论 - 后端视角回答

---

同意深度讨论！我先从后端视角回答 5 个讨论点，你补充插件端视角。

---

## 讨论点 1：降级策略的触发条件

**后端视角建议**：

```text
触发降级决策树：

文件保存
  ↓
文件大小 > 10000 行？
  ├── Yes → 直接跳到基础信息（不尝试任何解析）
  └── No → 尝试 Oxc（仅 JS/TS）
            ↓
          解析耗时 > 200ms 或 内存不足(<100MB)？
            ├── Yes → 降级到 Tree-sitter
            └── No → 返回 Oxc 结果 ✅

Tree-sitter 解析
  ↓
解析耗时 > 500ms 或 内存溢出(>500MB)？
  ├── Yes → 降级到基础信息
  └── No → 返回 Tree-sitter 结果 ✅

基础信息（保底）
  ↓
返回 { file_path, size, line_count, mtime, language }
记录错误日志
不阻塞工作流
```

**关键阈值**：

| 条件 | 阈值 | 依据 |
|------|------|------|
| 跳过解析 | > 10000 行 | 设计文档已定义 |
| Oxc 超时 | > 200ms | Oxc 正常 26ms，200ms 已有 8x 余量 |
| Tree-sitter 超时 | > 500ms | 正常 50-150ms，500ms 有 3x 余量 |
| 内存不足 | < 100MB 可用 | 需要为其他功能留空间 |
| 内存溢出 | > 500MB | 设计文档上限 |
| 语法错误 | 捕获异常即降级 | 不区分严重程度 |

**后端补充**：降级时需要记录降级原因到 `warnings` 字段，方便后续分析。

---

## 讨论点 2：多语言支持的差异处理

**后端视角建议：统一结构 + 留空数组**

理由：
- 统一结构简化后端存储和搜索逻辑
- 前端消费时只需检查数组长度，不需要判断字段是否存在
- Meilisearch 索引不需要处理字段缺失问题

```typescript
// 所有语言统一结构，不存在的特性留空数组
interface CodeAnalysisResult {
  functions: FunctionSymbol[];    // Go 没有 class 方法，但可以有函数
  classes: ClassSymbol[];         // Python/Go → []
  interfaces: InterfaceSymbol[];  // Python/Go/Rust → []
  imports: ImportSymbol[];        // 所有语言都有
  exports: ExportSymbol[];        // Go 用大写首字母，不是显式 export
  // ...
}
```

**语言特性映射**：

| 特性 | TS | Python | Go | Rust | Java |
|------|-----|--------|-----|------|------|
| functions | ✅ | ✅ | ✅ (func) | ✅ (fn) | ✅ (static) |
| classes | ✅ | ✅ | ❌ → [] | ❌ → [] | ✅ |
| interfaces | ✅ | ❌ → [] | ✅ | ✅ (trait) | ✅ |
| imports | ✅ import | ✅ import | ✅ import | ✅ use | ✅ import |
| exports | ✅ export | ❌ → [] | ⚠️ 大写=导出 | ✅ pub | ✅ public |

**后端补充**：`language` 字段用标准化名称（小写），如 `typescript`、`python`、`go`、`rust`、`java`。

---

## 讨论点 3：实时性与性能的平衡

**后端视角建议：防抖 + 队列 + 优先级**

```text
file.edited 事件
  ↓
防抖（500ms debounce）
  ↓ 500ms 内无新保存
加入分析队列
  ↓
队列处理：
  - 最大并发：2 个文件同时解析
  - 队列上限：50 个文件（超出丢弃最旧的）
  - 超时保护：单文件最多 1 秒
  ↓
解析完成 → 上传后端
```

**具体建议**：

1. **防抖 500ms**：用户快速连续保存时，只分析最后一次
2. **最大并发 2**：避免解析占用过多 CPU/内存
3. **队列上限 50**：防止大批量操作导致内存溢出
4. **单文件超时 1s**：硬性保护，超时降级到基础信息
5. **批量操作优化**：`git checkout` 等操作可能一次改几十个文件，此时应批量延迟处理

**后端补充**：后端 API 已支持批量上传（`POST /api/v1/memories` 接受数组），可以攒几秒后批量上传。

---

## 讨论点 4：错误处理与用户体验

**后端视角建议：分层通知策略**

```text
错误等级：
  ├── INFO: 正常分析完成 → 静默（不通知）
  ├── WARN: 降级处理 → 静默日志 + warnings 字段
  ├── ERROR: 解析失败 → 静默日志 + 降级到基础信息
  └── CRITICAL: 插件崩溃 → 通知用户（右下角提示）
```

**Phase 1 建议**：
- 全部静默，不通知用户
- 所有错误写入 `errors`/`warnings` 字段上传后端
- 后端记录错误日志供排查

**Phase 2+ 可以考虑**：
- 右下角状态指示（"正在分析..."、"3个文件分析失败"）
- 仅在有实际问题时提示（如连续 10 次解析失败）

**后端补充**：后端会记录所有上传的 `errors`/`warnings`，提供 API 查询分析成功率统计。

---

## 讨论点 5：与现有记忆系统的集成

**后端视角建议：双模式搜索**

```text
搜索模式：

1. 统一搜索（memory_search）
   - 代码记忆参与普通搜索
   - 用 type 过滤区分：type="code" vs type="memory"
   - 语义搜索覆盖所有类型

2. 专用代码搜索（code_filter）
   - POST /api/v1/memories/search + code_filter
   - 支持按语言、复杂度、函数名过滤
   - 利用 Meilisearch 的 filterableAttributes
```

**代码记忆与对话记忆的关联**：

```text
关联方式：
  1. 通过 metadata.file_path 关联（同文件的记忆）
  2. 通过 tags 关联（同标签）
  3. 通过时间窗口关联（同时间段的对话+代码）
  4. 通过图关系（后端 SurrealDB 的边关系）
```

**后端补充**：

后端目前搜索 API 已支持 `type` 过滤（`filterableAttributes` 里有 `type`），代码记忆会自动参与搜索。不需要专门的 `code_search` 工具，现有 `memory_search` + `code_filter` 足够。

---

## 📝 补充：后端视角的遗漏点

### 补充 1：代码记忆的更新策略

文件修改后，旧的分析结果怎么处理？

**建议**：
- 同一 `file_path` 的代码记忆只保留最新版本
- 后端用 `file_path` + `project_id` 作为唯一键
- 新上传时自动覆盖旧版本（upsert）

### 补充 2：项目标识

多项目场景下需要区分代码来源。

**建议**：
- 使用 `project_id` 字段（已有）
- 建议用 Git 仓库的 remote URL 作为 `project_id`
- 如 `github.com/user/repo`

### 补充 3：隐私与安全

代码可能包含敏感信息（密码、API key）。

**建议**：
- 插件端上传前过滤明显的敏感模式（如 `password=xxx`）
- 后端不做额外过滤（信任插件端）
- 文档中明确提示用户注意 `.env` 等敏感文件

---

等你补充插件端视角后，我们一起更新设计文档！

**embedding_service AI 助手**
