# 代码分析功能

> **版本**: v1.4  
> **状态**: Phase 3 已完成，Phase 2 开发中
> **适用**: OpenCode Memory 插件用户

---

## 功能概述

代码分析功能自动解析你的代码文件，提取关键信息并建立可搜索的知识库。让你在开发过程中快速找到相关代码、理解项目结构、追踪调用关系。

---

## 核心能力

### 1. 自动代码解析

保存代码文件时自动触发分析：

- **支持语言**: TypeScript, JavaScript, Python, Go, Rust, Java, C, C++
- **提取内容**: 函数、类、接口、导入/导出、注释、复杂度
- **存储位置**: 本地 SurrealDB + Meilisearch 索引
- **隐私保护**: 代码不上传云端，完全本地处理

**触发方式**:
```
文件保存 → chokidar 监听 → 300ms 防抖 → 自动分析 → 本地存储
```

---

### 2. 代码搜索

#### 基础搜索
搜索代码中的函数名、类名、关键词：

```bash
# 搜索所有认证相关代码
curl -X POST http://localhost:17999/api/v1/memories/search \
  -d '{"query": "authentication", "type": "code"}'
```

#### 过滤搜索
按语言、复杂度等条件过滤：

```bash
# 搜索 TypeScript 中高复杂度的函数
curl -X POST http://localhost:17999/api/v1/memories/search \
  -d '{
    "query": "validate",
    "type": "code",
    "code_filter": {
      "language": "typescript",
      "min_complexity": 5,
      "max_complexity": 15
    }
  }'
```

**支持的过滤条件**:
- `language`: 编程语言
- `min_complexity` / `max_complexity`: 圈复杂度范围
- `min_function_count` / `max_function_count`: 函数数量（v1.4）
- `min_class_count` / `max_class_count`: 类数量（v1.4）
- `has_exports`: 是否有导出（v1.4）
- `is_async`: 是否异步函数（v1.4）

---

### 3. 调用关系追踪（v1.4 Phase 2）

查找函数之间的调用关系：

```bash
# 查询某函数被哪些代码调用
curl http://localhost:17999/api/v1/memories/mem_xxx/references

# 返回:
# {
#   "references": [
#     { "file_path": "src/auth.ts", "line": 42, "caller": "login" },
#     { "file_path": "src/api.ts", "line": 15, "caller": "verifyToken" }
#   ]
# }
```

**使用场景**:
- 重构前了解影响范围
- 理解代码依赖关系
- 追踪 bug 传播路径

---

### 4. 项目代码地图（v1.4 Phase 3）

生成项目结构可视化：

```bash
# 获取项目代码地图
curl http://localhost:17999/api/v1/projects/github.com/user/repo/map

# 返回:
# {
#   "file_tree": { ... },
#   "module_dependencies": [ ... ],
#   "hot_files": ["src/core/auth.ts", "src/utils/api.ts"],
#   "statistics": {
#     "total_functions": 150,
#     "total_classes": 30,
#     "avg_complexity": 5.2
#   }
# }
```

**包含信息**:
- 目录结构树
- 模块间依赖关系
- 热点文件（高复杂度或高频修改）
- 项目统计（函数数、类数、平均复杂度）

---

### 5. 语义代码搜索（v1.4 Phase 4）

用自然语言搜索代码意图：

```bash
# 搜索"用户认证逻辑"
curl -X POST http://localhost:17999/api/v1/memories/search \
  -d '{
    "semantic_query": "用户认证逻辑",
    "type": "code"
  }'
```

**工作原理**:
- 提取代码关键结构（函数签名、类定义）
- 生成语义向量
- 匹配自然语言查询的语义相似度

---

## v1.4 新特性

### 调用关系追踪 🔄 开发中

追踪函数间的调用关系，支持代码依赖分析。

**功能**:

- 存储函数调用关系 (CallSymbol)
- 查询函数的调用者 (references)
- 查询函数的依赖 (dependencies)
- 支持跨文件调用追踪

**使用场景**:

- 重构前影响分析
- 理解代码依赖关系
- 查找未使用代码

### 代码地图 ✅ 已实现

项目级代码可视化，帮助理解项目结构。

**功能**:

- 文件树结构
- 模块依赖图
- 热点文件标记
- 项目统计信息

**使用场景**:

- 新成员快速了解项目
- 架构审查
- 代码健康度评估

### 代码统计 ✅ 已实现

项目级代码指标聚合，提供代码健康度快照。

**功能**:

- 函数/类/接口总数
- 平均圈复杂度
- 语言分布统计
- 文件规模分布

**使用场景**:

- 项目健康度评估
- 技术债务追踪
- 团队代码质量报告

### 分析结果缓存 ✅ 已实现

缓存代码分析结果，避免重复解析。

**功能**:

- 基于文件指纹的缓存命中
- 自动失效（文件修改后重新分析）
- 减少分析延迟

### 增强的代码指标

**新增指标**:

- 返回类型分析
- 导出/异步标记
- 类属性提取
- 接口定义提取
- 依赖分类 (internal/external/builtin)

---

## API 状态

| API | 状态 | 说明 |
|-----|------|------|
| 代码搜索 | ✅ 可用 | 基础搜索 + 过滤 |
| 代码地图 | ✅ 可用 | GET /projects/{id}/map |
| 代码统计 | ✅ 可用 | GET /projects/{id}/stats |
| 调用关系存储 | 🔄 开发中 | POST /calls/batch |
| 引用查询 | 🔄 开发中 | GET /memories/{id}/references |
| 依赖查询 | 🔄 开发中 | GET /memories/{id}/dependencies |

---

## 使用场景

### 场景 1: 重构影响分析 🔄 开发中

**问题**: 修改 `auth.ts` 会影响哪些文件？

**解决**:

```bash
# 1. 找到 auth.ts 的记忆 ID
# 2. 查询引用关系（开发中）
curl http://localhost:17999/api/v1/memories/mem_auth/references
```

### 场景 2: PR 代码审查 ✅ 已实现

**问题**: 找出 PR 中圈复杂度最高的函数

**解决**:

```bash
curl -X POST http://localhost:17999/api/v1/memories/search \
  -d '{
    "type": "code",
    "code_filter": {
      "min_complexity": 10
    },
    "sort": "code_complexity:desc"
  }'
```

### 场景 3: 项目架构概览 ✅ 已实现

**问题**: 新接手项目，快速了解结构

**解决**:

```bash
# 获取项目代码地图
curl http://localhost:17999/api/v1/projects/github.com/user/repo/map
```

---

## 数据结构

### 代码分析结果

```json
{
  "language": "typescript",
  "analyzer": "tree-sitter",
  "analyzed_at": "2026-04-06T12:00:00Z",
  "functions": [
    {
      "name": "validateUser",
      "start_line": 10,
      "end_line": 25,
      "parameters": [{"name": "email", "type": "string"}],
      "return_type": "boolean",
      "is_exported": true,
      "is_async": false
    }
  ],
  "classes": [
    {
      "name": "AuthService",
      "start_line": 1,
      "end_line": 50,
      "methods": ["validateUser", "logout"],
      "properties": ["token", "user"]
    }
  ],
  "complexity": {
    "cyclomatic": 5,
    "lines_of_code": 100,
    "function_count": 3,
    "class_count": 1
  }
}
```

---

## 隐私与安全

### 本地处理
- 所有代码分析在本地完成
- 代码不上传任何云端服务
- 分析结果存储在本地 SurrealDB

### 敏感信息过滤
自动跳过敏感文件：
- `.env` 文件
- `node_modules/` 目录
- 包含密码/API key 的文件

---

## 版本演进

| 版本 | 功能 | 状态 |
|------|------|------|
| v1.0 | 基础代码解析 | ✅ 已完成 |
| v1.2 | 复杂度分析、code_filter | ✅ 已完成 |
| **v1.4** | **调用关系、代码地图、语义搜索** | **🚧 实施中** |

---

## API 实施状态

| API | 方法 | 状态 | 计划 |
|-----|------|------|------|
| 代码上传 | POST /api/v1/memories | ✅ 已支持 | - |
| 调用关系存储 | POST /api/v1/calls/batch | 🔄 Phase 2 | 2026-04-21 |
| 引用查询 | GET /api/v1/memories/{id}/references | 🔄 Phase 2 | 2026-04-21 |
| 依赖查询 | GET /api/v1/memories/{id}/dependencies | 🔄 Phase 2 | 2026-04-21 |
| 代码地图 | GET /api/v1/projects/{id}/map | 🔄 Phase 3 | 2026-05-05 |
| 项目统计 | GET /api/v1/projects/{id}/stats | 🔄 Phase 3 | 2026-05-05 |

---

## 相关文档

- **设计文档**: `docs/CODE-ANALYSIS-DESIGN-v1.4.md`
- **API 文档**: `docs/dev/CODE_ANALYSIS_API.md`
- **开发指南**: `CONTRIBUTING.md`

---

*最后更新: 2026-04-07 | v1.4 新特性和 API 实施状态已更新*
