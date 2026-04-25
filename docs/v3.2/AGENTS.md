# v3.2 架构文档 - Agent 指南

> **Scope**: v3.2 统一架构实施文档  
> **Version**: v3.2.0  
> **Status**: 实施完成  
> **技术栈**: Python 3.10+ + SurrealDB 3.0+ + SDK 1.0.8

---

## Structure

```text
docs/v3.2/
├── UNIFIED-ARCHITECTURE-v3.2.md    # 统一架构总览 (724行)
├── BACKEND-v3.2-IMPLEMENTATION.md  # 后端实现指南
├── BACKEND-v3.2-MEILISEARCH.md     # Meilisearch集成
├── BACKEND-v3.2-MIGRATION.md       # 数据迁移
├── BACKEND-v3.2-PRECOMPUTE.md      # 预计算服务 (1075行)
├── BACKEND-v3.2-WEBSOCKET.md       # WebSocket设计
├── DATABASE-v3.2-ER.md             # 数据库ER图
├── DATABASE-v3.2-SCHEMA.md         # Schema定义 (800+行)
├── DEPENDENCY-VERSIONS.md          # 依赖版本管理
├── DEPLOYMENT-v3.2.md              # 部署方案
├── DEVELOPMENT-v3.2.md             # 开发指南
├── PLUGIN-v3.2-API.md              # 插件API规范
├── PLUGIN-v3.2-IMPLEMENTATION.md   # 插件实现
├── WEBSOCKET-v3.2-PROTOCOL.md      # WebSocket协议
├── ACCEPTANCE-CRITERIA.md          # 验收标准
├── EVALUATION-*.md                 # 评估报告
├── RTM.md                          # 需求追溯矩阵
└── tracking/                       # 实施跟踪
    ├── COMPONENTS.md
    └── VALIDATION.md
```

---

## Where to Look

| 任务 | 文档 | 说明 |
|------|------|------|
| 架构总览 | `UNIFIED-ARCHITECTURE-v3.2.md` | 四层架构、设计原则、数据模型 |
| 后端实现 | `BACKEND-v3.2-IMPLEMENTATION.md` | API路由、服务层、工具类 |
| 数据库设计 | `DATABASE-v3.2-SCHEMA.md` | SurrealDB 3.0+语法、表定义 |
| 预计算服务 | `BACKEND-v3.2-PRECOMPUTE.md` | AST解析、指纹、符号提取 |
| WebSocket | `BACKEND-v3.2-WEBSOCKET.md` | 实时同步、LIVE SELECT |
| 插件集成 | `PLUGIN-v3.2-*.md` | 插件API、实现指南 |
| 部署方案 | `DEPLOYMENT-v3.2.md` | Docker、K8s、SSL |
| 依赖版本 | `DEPENDENCY-VERSIONS.md` | Python包版本锁定 |

---

## Conventions

### 文档命名

```text
<领域>-v3.2-<主题>.md

领域: BACKEND | DATABASE | PLUGIN | DEPLOYMENT | DEVELOPMENT
主题: IMPLEMENTATION | SCHEMA | API | PROTOCOL | MIGRATION
```

### 版本标记

```markdown
> **版本**: v3.2.0  
> **日期**: 2026-04-10  
> **状态**: 实施版 | 规划版 | 已归档
```

### 关键决策标记

```markdown
- ✅ **已实施**: 功能已完成
- 🚧 **进行中**: 正在开发
- ⏳ **待开始**: 计划中的功能
- ❌ **已取消**: 不再实施
```

---

## Anti-Patterns (文档维护)

| 问题 | 说明 | 状态 |
|------|------|------|
| 端口不一致 | 部分文档写17999，实际18008 | 需更新 |
| 版本号漂移 | 产品v2.8.0 vs 架构v3.2.0 | 已说明 |
| 重复内容 | 预计算服务在多个文档描述 | 以BACKEND-v3.2-PRECOMPUTE.md为准 |

---

## Commands

```bash
# 查看架构总览
cat docs/v3.2/UNIFIED-ARCHITECTURE-v3.2.md | head -100

# 搜索特定主题
grep -r "tree-sitter" docs/v3.2/
grep -r "WebSocket" docs/v3.2/

# 检查实施状态
grep "^\- " docs/v3.2/RTM.md | grep -E "(✅|🚧|⏳)"
```

---

## Notes

- **v3.2已实施完成**: 所有标记✅的功能已上线
- **端口迁移**: 17999 → 18008 (v3.2核心变更)
- **SurrealDB 3.0+**: 使用`COMPUTED`、`FULLTEXT`、`type::record()`
- **单租户**: `tenant_id`预留字段，多租户物理隔离暂缓
- **tree-sitter**: 性能提升3.32x，已纳入代码分析
- **四层架构**: Atom/Entity/Relation/Backlog (保留v2.0设计)

---

## Related

- 实现代码: `wrapper/src/` (参见 `wrapper/src/AGENTS.md`)
- 测试代码: `tests/` (参见 `tests/AGENTS.md`)
- 根目录指南: `AGENTS.md`
