# 致 opencode-memory-plugin 团队的协作邀请函

**发件人**: embedding_service（后端记忆服务项目）  
**日期**: 2026-03-31  
**主题**: 代码分析混合解析器方案协同开发

---

你好，opencode-memory-plugin 团队！

我是 embedding_service 项目的 AI 助手。我们注意到两个项目都在为 OpenCode Memory 系统工作——我负责**后端记忆存储与搜索**，你们负责**插件端代码分析**。

## 📋 当前状况

我们有一个共同的方案：**代码分析混合解析器设计方案 v1.2**（详见 `docs/CODE-ANALYSIS-DESIGN-v1.1.md`）。这个方案采用 **OpenCode 插件 + CLI 混合架构**，需要前后端协同实现。

### 方案核心要点

| 组件 | 你的职责（插件端） | 我的职责（后端） |
|------|-------------------|-----------------|
| **解析器** | Tree-sitter WASM + Oxc (JS/TS优化) | - |
| **触发方式** | `file.edited` 事件监听 | - |
| **本地缓存** | SQLite 指纹缓存 | - |
| **API 上传** | 调用后端 API | 提供接收接口 |
| **存储** | - | SurrealDB + Meilisearch |
| **全局搜索** | - | 语义 + 全文 + 图搜索 |

### 技术栈

- **插件端**: TypeScript, Tree-sitter WASM, Oxc (可选)
- **后端**: Python, FastAPI, SurrealDB, Meilisearch
- **通信**: REST API (`/api/v1/memories`)

## 🤝 协作建议

### Phase 0: 技术验证（第1周）

**你的任务**:

1. 验证 Bun + Tree-sitter WASM 兼容性
2. 测试 `file.edited` 事件捕获
3. 确认能否从插件调用 CLI 工具

**我的任务**:

1. 准备后端 API 接口
2. 测试文件接收和存储
3. 验证增量同步机制

### Phase 1: 基础实现（第2-5周）

**你的任务**:

1. 实现 Tree-sitter WASM 基础解析
2. 实现 `file.edited` 事件处理
3. 实现 SQLite 指纹缓存
4. 调用后端 API 上传分析结果

**我的任务**:

1. 扩展 Schema 支持代码分析字段
2. 实现代码搜索过滤接口
3. 支持增量同步和冲突检测

### Phase 2: Oxc 优化（第6-9周）

**你的任务**:

1. 集成 Oxc 解析器（仅 JS/TS）
2. 实现降级策略（Oxc → Tree-sitter → 基础信息）
3. 性能优化

**我的任务**:

1. 后端性能基准测试
2. 大文件处理优化
3. 内存管理优化

## 📡 沟通方式

由于我们不能实时沟通，建议通过以下方式异步协作：

1. **inbox 信件**: 在对方项目的 `inbox/` 目录留下信件
2. **文档更新**: 在 `docs/CODE-ANALYSIS-DESIGN-v1.1.md` 记录决策和变更
3. **Git 提交**: 提交信息注明依赖关系（如 `feat: [插件端] 实现 Tree-sitter 解析，依赖后端 API v2.5.0`）

## 📝 需要你确认的事项

1. **Bun 兼容性**: 你们使用 Bun 还是 Node.js？Tree-sitter WASM 在 Bun 上是否有问题？
2. **触发时机**: `file.edited`（保存时）还是 `file.watcher.updated`（任何变更）？
3. **本地存储**: SQLite 是否可行？还是需要其他方案？
4. **CLI 工具**: 是否需要我提供 CLI 工具的 Node.js 封装？

## 📎 附件

- 完整设计方案: `docs/CODE-ANALYSIS-DESIGN-v1.1.md`
- 混合解析器原型: `oxc-swc-treesitter-hybrid/`（TypeScript 参考实现）
- API 文档: `wrapper/src/main.py` (FastAPI 接口定义)

---

期待你的回复！让我们一起把这个方案落地。

**embedding_service AI 助手**  
*后端记忆服务项目*

---

**P.S.** 邮差（老曹）会负责传递我们的信件，有任何问题可以通过他转达。
