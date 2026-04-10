# 统一架构 v3.2 设计方案（实施版）

> **版本**: v3.2.0  
> **日期**: 2026-04-10  
> **状态**: 实施版  
> **技术栈**: Python 3.10+ + SurrealDB 3.0+ + SDK 1.0.8  
> **服务端口**: 18008  
> **核心变更**: 单租户实施 + tenant_id 预留字段（多租户物理隔离暂缓至 SDK 2.0 stable）

---

## 目录

1. [设计原则](#1-设计原则)
2. [架构概览](#2-架构概览)
3. [数据模型](#3-数据模型)
4. [数据库 Schema](#4-数据库-schema)
5. [API 接口规范](#5-api-接口规范)
6. [WebSocket 实时同步设计](#6-websocket-实时同步设计)
7. [预计算服务设计](#7-预计算服务设计)
8. [Docker 多阶段构建方案](#8-docker-多阶段构建方案)
9. [实施计划](#9-实施计划)
10. [实现状态标注](#10-实现状态标注)
11. [关键遗漏项补充](#11-关键遗漏项补充)
12. [风险与应对](#12-风险与应对)

---

## 1. 设计原则

### 1.1 核心原则（基于反馈确认）

| 原则               | 说明                                 | 决策依据                         |
| ------------------ | ------------------------------------ | -------------------------------- |
| **云端优先**       | 主要部署在云端，小型化后可本地部署   | 云端更适合 Agent 高频调用        |
| **Agent-Native**   | 为 Coding Agent 设计，不迁就人类习惯 | 无 GUI、无自动 LLM 处理          |
| **无后端 LLM**     | Agent 显式决策，系统只提供能力       | Agent 手动标注、手动建立关系     |
| **预计算加速**     | 文件保存时预计算，查询时快速响应     | 基于现有 code-analyzer.js        |
| **保留 v2.0 模型** | Atom/Entity/Relation 四层架构        | v2.0 内容全部需要                |
| **SurrealDB 3.0+** | 使用最新数据库特性                   | RELATE、COMPUTED、FULLTEXT       |
| **端口 18008**     | 新服务独立端口                       | 与现有 17999 服务隔离            |
| **Python 原生**    | 使用 Python 3.10+ + SDK 1.0.8        | 异步支持、Live Queries、类型安全 |

### 1.2 关键决策

- ✅ **多租户策略调整**: 暂缓 Namespace 物理隔离（SDK 2.0 alpha 不稳定），采用单租户 + `tenant_id` 预留字段方案，SDK 2.0 stable 后无缝迁移
- ✅ **Docker 已做**: 基础容器化已完成，Phase 1 包含多阶段构建优化
- ✅ **BACKLOG API 取消**: 融入当前设计方案，作为 Entity 类型实现
- ✅ **代码分析已实现**: CallSymbol 提取已完成（tree-sitter Query）
- ✅ **优先级**: 预计算服务优先
- ✅ **接口模式**: 插件工具 + CLI（无 MCP、无 IDE 集成）
- ✅ **SurrealDB 版本**: 使用 3.0+ 语法（`COMPUTED`, `FULLTEXT`, `type::record()`）
- ✅ **tree-sitter Query**: 验证通过，性能提升 3.32x，纳入 Phase 1
- ✅ **服务端口**: 新服务使用 18008（从 17999 迁移）
- ✅ **Agent-Native**: 无 GUI、无 IDE Integration（已取消）

---

## 2. 架构概览

### 2.1 四层架构（保留 v2.0 设计，简化表示层）

```
┌─────────────────────────────────────────────────────────────┐
│                    表示层 (Presentation)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Markdown  │  │   Wiki/MD   │  │    Obsidian Vault   │ │
│  │  (人类可读)  │  │  (双向链接)  │  │    (本地编辑)        │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│                                                              │
│  【注：Agent-Native 设计，无 GUI、无 IDE 集成】              │
│  【所有交互通过 Python FastAPI 进行】                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    原子层 (Atomic Layer)                     │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│   │  Goal   │  │  Scope  │  │  Task   │  │  Note   │       │
│   │ (目标)  │  │ (范围)  │  │ (任务)  │  │ (笔记)  │       │
│   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘       │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│   │Function │  │  Class  │  │Interface│  │ Import  │       │
│   │ (函数)  │  │  (类)   │  │ (接口)  │  │ (导入)  │       │
│   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘       │
│        └─────────────┴─────────────┴────────────┘            │
│                       │                                      │
│              ┌────────┴────────┐                           │
│              │  Atom Relations  │                           │
│              │  • depends_on    │                           │
│              │  • blocks        │                           │
│              │  • calls         │ ← 代码调用               │
│              │  • imports       │ ← 代码导入               │
│              │  • implements    │ ← 代码实现任务           │
│              └─────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  实体层 (Entity Layer)                       │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│   │  Memory  │  │  Backlog │  │   Wiki   │  │   Code   │  │
│   │  (记忆)  │  │  (任务)  │  │  (页面)  │  │  (代码)  │  │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│        └──────────────┼──────────────┴────────────┘         │
│                       │                                      │
│              ┌────────┴────────┐                           │
│              │  Entity Graph    │                           │
│              │  • wiki_link     │                           │
│              │  • implements    │ ← 代码实现任务           │
│              │  • relates_to    │                           │
│              │  • part_of       │                           │
│              └─────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   存储层 (Storage Layer)                     │
│   ┌─────────────────┐    ┌─────────────────┐               │
│   │   SurrealDB     │◄──►│   Meilisearch   │               │
│   │  (主数据库)      │    │   (搜索索引)     │               │
│   │                 │    │                 │               │
│   │  • Entities     │    │  • 全文搜索      │               │
│   │  • Atoms        │    │  • 向量搜索      │               │
│   │  • Relations    │    │  • 过滤排序      │               │
│   │  • Graph        │    │                 │               │
│   └─────────────────┘    └─────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 部署架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Cloud Deployment (主要)                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 API Gateway                          │   │
│  │  • 统一认证 (API Key)                                │   │
│  │  • 请求路由                                          │   │
│  │  • 限流/配额                                         │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Precompute Service                      │   │
│  │  • AST 解析 (tree-sitter)                            │   │
│  │  • 符号提取                                          │   │
│  │  • 引用解析                                          │   │
│  │  • 聚类 (Leiden)                                     │   │
│  │  • 执行流追踪                                        │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Plugin Tool API                         │   │
│  │  • memory_write/search/read                          │   │
│  │  • code_analyze/navigate/impact                      │   │
│  │  • task_create/update                                │   │
│  │  • graph_traverse/query                              │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │               Storage Layer                          │   │
│  │  • SurrealDB 3.0+ (符号、关系、图谱)                  │   │
│  │  • Meilisearch (混合搜索索引)                         │   │
│  │  • 纯存储，无 LLM 处理                                │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Local Deployment (小型化后)                     │
│  docker-compose up                                          │
│  • 同样的容器镜像（已存在）                                 │
│  • 本地 SurrealDB 3.0+ + Meilisearch                       │
│  • 同样的插件工具接口                                       │
│  • 服务端口: 18008                                         │
└─────────────────────────────────────────────────────────────┘
```

**端口配置更新**:

- 新服务端口: **18008** (从 17999 迁移)
- SurrealDB 端口: 8000 (保持不变)
- Meilisearch 端口: 7700 (保持不变)

---

## 3. 数据模型

### 3.1 Atom（原子单元）- 最小知识单元

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class Atom:
    """
    原子单元 - 最小知识单元

    对应 SurrealDB atom 表
    """
    # 基础字段（所有 Atom 共有）
    id: Optional[str] = None  # ULID，如 "atom:01HQ..."
    type: str  # function | class | interface | import | goal | scope | task | note
    content: str  # 内容（函数源码、任务描述等）
    tenant_id: str = "default"  # v3.2: 预留字段，当前固定为 "default"

    # 状态（用于 task 类型）
    status: Optional[str] = None  # pending | done | blocked

    # 代码特有字段（Function/Class/Interface）
    name: Optional[str] = None  # 函数/类名
    signature: Optional[str] = None  # 函数签名
    params: List[Dict] = field(default_factory=list)
    return_type: Optional[str] = None
    is_exported: Optional[bool] = None
    is_async: Optional[bool] = None
    complexity: Optional[int] = None
    max_nesting_depth: Optional[int] = None
    docstring: Optional[Dict] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None

    # 通用元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> dict:
        """转换为字典（用于 SurrealDB 插入）"""
        return {k: v for k, v in {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "tenant_id": self.tenant_id,
            "status": self.status,
            "name": self.name,
            "signature": self.signature,
            "params": self.params,
            "return_type": self.return_type,
            "is_exported": self.is_exported,
            "is_async": self.is_async,
            "complexity": self.complexity,
            "max_nesting_depth": self.max_nesting_depth,
            "docstring": self.docstring,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "metadata": self.metadata,
            "version": self.version,
        }.items() if v is not None}
```

### 3.2 Entity（实体）- 知识实体

```python
@dataclass
class Entity:
    """
    实体 - 知识实体

    对应 SurrealDB entity 表
    """
    id: Optional[str] = None  # ULID
    type: str  # memory | backlog | wiki | code
    tenant_id: str = "default"  # v3.2: 预留字段
    abstract: str  # L0: 摘要
    overview: Dict[str, Any] = field(default_factory=dict)  # L1: 结构化概览
    atoms: List[str] = field(default_factory=list)  # L2: Atom ID 列表

    # Wiki 特性
    title: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    outgoing_links: List[str] = field(default_factory=list)
    incoming_links: List[str] = field(default_factory=list)

    # Backlog 特性（融入当前设计）
    priority: Optional[str] = None  # P0 | P1 | P2 | P3
    status: Optional[str] = None  # backlog | todo | in_progress | in_review | done
    scene: Optional[str] = None
    estimated_hours: Optional[float] = None
    actual_hours: Optional[float] = None

    # Code 特性
    file_path: Optional[str] = None
    language: Optional[str] = None
    quality_score: Optional[Dict] = None
    complexity_metrics: Optional[Dict] = None

    # 通用
    tags: List[str] = field(default_factory=list)
    project: Optional[str] = None
    created_by: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
```

### 3.3 Relation（关系）- 原子/实体间关系

```python
@dataclass
class Relation:
    """
    关系 - 原子/实体间关系

    对应 SurrealDB reference 表
    """
    id: Optional[str] = None  # ULID
    type: str  # depends_on | blocks | calls | imports | implements | relates_to | wiki_link | part_of
    tenant_id: str = "default"  # v3.2: 预留字段

    from_id: str  # Atom ID 或 Entity ID
    to_id: str  # Atom ID 或 Entity ID

    # 代码特有（用于 calls/imports）
    file_path: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None

    # Backlog 特有（用于 implements）
    metadata: Optional[Dict] = None

    # 通用
    weight: float = 0.5
    created_by: Optional[str] = None
    created_at: Optional[str] = None
```

---

## 4. 数据库 Schema

### 4.1 Atom 表（SurrealDB 3.0+ 语法）

```sql
-- 使用 SurrealDB 3.0+ 语法
DEFINE NAMESPACE IF NOT EXISTS opencode;
USE NS opencode;
DEFINE DATABASE IF NOT EXISTS memory;
USE DB memory;

-- Atom 表定义
DEFINE TABLE atom TYPE NORMAL SCHEMAFULL CHANGEFEED 7d INCLUDE ORIGINAL;

DEFINE FIELD id ON atom TYPE record;
-- v3.2: tenant_id 预留字段，单租户时默认 "default"
DEFINE FIELD tenant_id ON atom TYPE string DEFAULT 'default';
DEFINE FIELD type ON atom TYPE string
    ASSERT $value IN ['function', 'class', 'interface', 'import', 'goal', 'scope', 'task', 'note'];

DEFINE FIELD content ON atom TYPE string;
DEFINE FIELD status ON atom TYPE option<string>
    ASSERT $value == NONE OR $value IN ['pending', 'done', 'blocked'];

-- 代码特有字段
DEFINE FIELD name ON atom TYPE option<string>;
DEFINE FIELD signature ON atom TYPE option<string>;
DEFINE FIELD params ON atom TYPE option<array>;
DEFINE FIELD return_type ON atom TYPE option<string>;
DEFINE FIELD is_exported ON atom TYPE option<bool>;
DEFINE FIELD is_async ON atom TYPE option<bool>;
DEFINE FIELD complexity ON atom TYPE option<int>;
DEFINE FIELD max_nesting_depth ON atom TYPE option<int>;
DEFINE FIELD docstring ON atom TYPE option<object>;
DEFINE FIELD start_line ON atom TYPE option<int>;
DEFINE FIELD end_line ON atom TYPE option<int>;

-- 通用字段（使用 COMPUTED 替代 FUTURE）
DEFINE FIELD metadata ON atom TYPE option<object> DEFAULT {};
DEFINE FIELD version ON atom TYPE int DEFAULT 1;
DEFINE FIELD created_at ON atom TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON atom TYPE datetime DEFAULT time::now();

-- 索引（含 tenant_id 复合索引，为多租户预留）
DEFINE INDEX idx_atom_type ON atom FIELDS type;
DEFINE INDEX idx_atom_name ON atom FIELDS name;
DEFINE INDEX idx_atom_project ON atom FIELDS project;
DEFINE INDEX idx_atom_tenant_type ON atom FIELDS tenant_id, type;
DEFINE INDEX idx_atom_tenant_project ON atom FIELDS tenant_id, project;
```

### 4.2 Entity 表（SurrealDB 3.0+ 语法）

```sql
DEFINE TABLE entity TYPE NORMAL SCHEMAFULL CHANGEFEED 7d INCLUDE ORIGINAL;

DEFINE FIELD id ON entity TYPE record;
DEFINE FIELD tenant_id ON entity TYPE string DEFAULT 'default';
DEFINE FIELD type ON entity TYPE string
    ASSERT $value IN ['memory', 'backlog', 'wiki', 'code'];

-- L0/L1/L2 分层
DEFINE FIELD abstract ON entity TYPE string;
DEFINE FIELD overview ON entity TYPE object;

-- v3.2: 双向引用（自动维护反向引用）
DEFINE FIELD atoms ON entity
    TYPE option<array<record<atom>>>
    REFERENCE
    ON DELETE CASCADE;

-- Wiki 特性
DEFINE FIELD title ON entity TYPE option<string>;
DEFINE FIELD aliases ON entity TYPE array<string> DEFAULT [];
DEFINE FIELD outgoing_links ON entity TYPE array<string> DEFAULT [];
DEFINE FIELD incoming_links ON entity TYPE array<string> DEFAULT [];

-- Backlog 特性
DEFINE FIELD priority ON entity TYPE option<string>
    ASSERT $value == NONE OR $value IN ['P0', 'P1', 'P2', 'P3'];
DEFINE FIELD status ON entity TYPE option<string>
    ASSERT $value == NONE OR $value IN ['backlog', 'todo', 'in_progress', 'in_review', 'done'];
DEFINE FIELD scene ON entity TYPE option<string>;
DEFINE FIELD estimated_hours ON entity TYPE option<float>;
DEFINE FIELD actual_hours ON entity TYPE option<float>;

-- Code 特性
DEFINE FIELD file_path ON entity TYPE option<string>;
DEFINE FIELD language ON entity TYPE option<string>;
DEFINE FIELD quality_score ON entity TYPE option<object>;
DEFINE FIELD complexity_metrics ON entity TYPE option<object>;

-- 通用字段
DEFINE FIELD tags ON entity TYPE array<string> DEFAULT [];
DEFINE FIELD project ON entity TYPE option<string>;
DEFINE FIELD created_by ON entity TYPE string;
DEFINE FIELD created_at ON entity TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON entity TYPE datetime DEFAULT time::now();

-- 索引
DEFINE INDEX idx_entity_type ON entity FIELDS type;
DEFINE INDEX idx_entity_project ON entity FIELDS project;
DEFINE INDEX idx_entity_status ON entity FIELDS status;
DEFINE INDEX idx_entity_tenant_type ON entity FIELDS tenant_id, type;
DEFINE INDEX idx_entity_tenant_status ON entity FIELDS tenant_id, status;
```

### 4.3 Relation 表（使用 RELATE - SurrealDB 3.0+ 推荐）

```sql
-- 使用原生图关系（替代手动关系表）
-- v3.2: 通用关系表，支持 atom-atom/atom-entity/entity-entity
DEFINE TABLE reference TYPE RELATION SCHEMAFULL CHANGEFEED 7d;

DEFINE FIELD tenant_id ON reference TYPE string DEFAULT 'default';
DEFINE FIELD type ON reference TYPE string
    ASSERT $value IN ['depends_on', 'blocks', 'calls', 'imports', 'implements', 'relates_to', 'wiki_link', 'part_of'];

-- 代码特有
DEFINE FIELD file_path ON reference TYPE option<string>;
DEFINE FIELD line ON reference TYPE option<int>;
DEFINE FIELD column ON reference TYPE option<int>;

-- Backlog 特有
DEFINE FIELD metadata ON reference TYPE option<object>;

-- 通用字段
DEFINE FIELD weight ON reference TYPE float DEFAULT 0.5;
DEFINE FIELD created_by ON reference TYPE string;
DEFINE FIELD created_at ON reference TYPE datetime DEFAULT time::now();

-- 唯一索引：避免重复关系（单租户下不含 tenant_id，预留兼容性）
DEFINE INDEX idx_unique_ref ON reference FIELDS in, out, type UNIQUE;
```

---

## 5. API 接口规范

### 5.1 基础配置

```python
# 服务端口: 18008 (从 17999 迁移)
API_BASE = "http://localhost:18008/api/v1"

# 请求头
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer {API_KEY}"
}
```

### 5.2 原子级操作

```python
# 创建 Atom（Agent 显式标注）
POST /api/v1/atoms
{
  "type": "function",
  "content": "async function analyzeCode(...) { ... }",
  "name": "analyzeCode",
  "signature": "async analyzeCode(filePath: string): Promise<AnalysisResult>",
  "params": [{"name": "filePath", "type": "string"}],
  "return_type": "Promise<AnalysisResult>",
  "is_exported": true,
  "is_async": true,
  "complexity": 5,
  "start_line": 85,
  "end_line": 125,
  "project": "backlog-api",
  "tenant_id": "default",  # v3.2: 预留字段
  "tags": ["typescript", "analysis"]
}

# 更新 Atom
PATCH /api/v1/atoms/{atom_id}
{
  "content": "更新后的内容",
  "version": 2
}

# 删除 Atom
DELETE /api/v1/atoms/{atom_id}

# 搜索 Atom（自动按 tenant_id 过滤）
GET /api/v1/atoms?query=analyzeCode&type=function&project=backlog-api&tenant_id=default
```

### 5.3 实体操作

```python
# 创建 Entity（包含 Atoms）
POST /api/v1/entities
{
  "type": "code",
  "title": "utils.ts",
  "abstract": "TypeScript file with 5 functions",
  "overview": {
    "language": "typescript",
    "lines_of_code": 150,
    "function_count": 5
  },
  "atoms": ["atom:func-1", "atom:func-2"],
  "file_path": "src/utils.ts",
  "language": "typescript",
  "project": "backlog-api",
  "tenant_id": "default",
  "tags": ["typescript", "utils"]
}

# Backlog Entity（融入当前设计）
POST /api/v1/entities
{
  "type": "backlog",
  "title": "实现文件监听",
  "abstract": "文件保存自动触发分析",
  "overview": {
    "description": "当用户保存文件时...",
    "scope": ["lib/file-watcher.js"]
  },
  "priority": "P0",
  "status": "in_progress",
  "scene": "代码分析v1.4",
  "estimated_hours": 4,
  "project": "backlog-api",
  "tenant_id": "default",
  "tags": ["代码分析", "自动化"]
}
```

### 5.4 关系操作

```python
# 创建关系（Agent 显式建立）
POST /api/v1/relations
{
  "type": "calls",
  "from_id": "atom:func-analyzeCode",
  "to_id": "atom:func-parseSync",
  "file_path": "src/utils.ts",
  "line": 95,
  "column": 20,
  "tenant_id": "default"
}

# 代码实现任务关系
POST /api/v1/relations
{
  "type": "implements",
  "from_id": "entity:backlog-BL-CA-17",
  "to_id": "entity:code-src-file-watcher",
  "metadata": {
    "implemented_functions": ["onFileSaved", "debouncedAnalyze"],
    "status": "completed"
  },
  "tenant_id": "default"
}
```

---

## 6. WebSocket 实时同步设计

详见 [BACKEND-v3.2-WEBSOCKET.md](./BACKEND-v3.2-WEBSOCKET.md)

---

## 7. 预计算服务设计

详见 [BACKEND-v3.2-PRECOMPUTE.md](./BACKEND-v3.2-PRECOMPUTE.md)

---

## 8. Docker 多阶段构建方案

详见 [BACKEND-v3.2-MIGRATION.md](./BACKEND-v3.2-MIGRATION.md)

---

## 9. 实施计划

详见 [BACKEND-v3.2-IMPLEMENTATION.md](./BACKEND-v3.2-IMPLEMENTATION.md)

---

## 10. 实现状态标注

### 10.1 验证结果总览

| 技术                           | 验证状态  | 关键发现                         | 实施优先级 |
| ------------------------------ | --------- | -------------------------------- | ---------- |
| **SurrealDB 3.0+**             | ✅ 已验证 | 7 个特性均可用                   | P0         |
| **SurrealDB Python SDK 1.0.8** | ✅ 已验证 | 稳定版，支持 live/subscribe_live | P0         |
| **SurrealDB SDK 2.0**          | ⚠️ Alpha  | 不稳定，暂缓多租户物理隔离       | P2         |
| **tree-sitter Query**          | ✅ 已验证 | 性能 3.32x                       | P0         |
| **GitHub Actions**             | ✅ 已验证 | 完全适配 Python                  | P0         |
| **Docker**                     | ✅ 已验证 | 多阶段构建可行                   | P0         |
| **Python asyncio**             | ✅ 已验证 | 替代 Node.js Worker Threads      | P1         |

### 10.2 已验证特性清单（150+ 个）

#### SurrealDB 3.0+ 特性（已验证）

- ✅ Database/Namespace 隔离
- ✅ RELATE 图关系（跨表支持，v3.1 误报已澄清）
- ✅ Live Queries（单节点，SDK 1.0.8 支持）
- ✅ Record References（REFERENCE ON DELETE CASCADE）
- ✅ DEFINE EVENT（$event 变量语法正确）
- ✅ HNSW 向量索引（EFC 参数正确，非 EF_CONSTRUCTION）
- ✅ FULLTEXT 索引（3.0+ 标准语法）
- ✅ 事务支持
- ✅ ChangeFeed（CHANGEFEED 7d）
- ✅ DEFINE FUNCTION（自定义函数）

#### tree-sitter 特性（已验证）

- ✅ Query API 可用
- ✅ 性能提升 3.32x
- ✅ 多语言支持（Python/JS/Rust/Java）
- ✅ 增量解析

#### Python 生态（已验证）

- ✅ FastAPI 异步性能
- ✅ SurrealDB SDK 1.0.8 异步支持
- ✅ tree-sitter Python 绑定
- ✅ portalocker 跨平台文件锁

---

## 11. 关键遗漏项补充

### 11.1 SurrealDB 2.x → 3.x 迁移要点

```sql
-- 1. 函数名变更
type::thing() → type::record()
duration::from::days() → duration::from_days()
string::is::email() → string::is_email()

-- 2. 全文搜索索引
SEARCH ANALYZER → FULLTEXT ANALYZER（v3.2 已使用正确语法）

-- 3. 计算字段
VALUE <future> { expr } → COMPUTED expr

-- 4. 参数声明
$var = value → LET $var = value

-- 5. MTREE 移除（使用 HNSW）
MTREE DIMENSION 768 → HNSW DIMENSION 768
```

### 11.2 增量更新策略

详见 [BACKEND-v3.2-PRECOMPUTE.md](./BACKEND-v3.2-PRECOMPUTE.md)

### 11.3 数据迁移方案（预留多租户升级）

详见 [BACKEND-v3.2-MIGRATION.md](./BACKEND-v3.2-MIGRATION.md)

---

## 12. 风险与应对（更新版）

| 风险                         | 影响 | 应对措施                                                   | 状态      |
| ---------------------------- | ---- | ---------------------------------------------------------- | --------- |
| **SurrealDB 3.x 语法变更**   | 高   | 使用新语法（COMPUTED, FULLTEXT），已验证 EFC/$event 正确性 | ✅ 已更新 |
| **tree-sitter Query 兼容性** | 中   | 版本对齐（0.22.x），Python 绑定已验证                      | ✅ 已验证 |
| **端口迁移遗漏**             | 中   | 全局搜索 + 全面测试                                        | ⚠️ 需检查 |
| **SDK 2.0 Alpha 不稳定**     | 高   | **暂缓物理隔离**，使用 tenant_id 预留字段方案              | ✅ 已调整 |
| **Live Queries 单节点限制**  | 低   | 当前架构单节点足够，未来考虑集群                           | ⚠️ 待评估 |
| **预计算性能问题**           | 高   | 异步处理、增量更新、连接池优化                             | ✅ 已补充 |
| **数据迁移失败**             | 高   | 备份、分阶段迁移、tenant_id 预留兼容                       | ✅ 已补充 |
| **并发冲突**                 | 中   | asyncio.Semaphore 限流、去重机制                           | ✅ 已补充 |
| **Docker 容器逃逸**          | 高   | **v3.2 安全加固**：no-new-privileges、read_only、非 root   | ✅ 已加固 |

---

## 参考文档

- [BACKEND-v3.2-IMPLEMENTATION.md](./BACKEND-v3.2-IMPLEMENTATION.md)
- [BACKEND-v3.2-WEBSOCKET.md](./BACKEND-v3.2-WEBSOCKET.md)
- [BACKEND-v3.2-PRECOMPUTE.md](./BACKEND-v3.2-PRECOMPUTE.md)
- [BACKEND-v3.2-MIGRATION.md](./BACKEND-v3.2-MIGRATION.md)
- [PLUGIN-v3.2-IMPLEMENTATION.md](./PLUGIN-v3.2-IMPLEMENTATION.md)
- [UNIFIED-ARCHITECTURE-v3.1.md](../v3.1/UNIFIED-ARCHITECTURE-v3.1.md)
- [API-CONTRACT.md](../API-CONTRACT.md)

---

_文档版本: v3.2.0_  
_最后更新: 2026-04-10_  
_状态: 实施版（单租户 + tenant_id 预留）_  
_SurrealDB 版本: 3.0+_  
_Python SDK: 1.0.8（稳定版）_  
_服务端口: 18008_
