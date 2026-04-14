# 数据库 ER 关系图

> **版本**: v3.2.0  
> **日期**: 2026-04-10  
> **状态**: 实施版  
> **SurrealDB 版本**: 3.0+

---

## 目录

1. [核心实体关系图](#1-核心实体关系图)
2. [图关系模型](#2-图关系模型)
3. [关系类型说明](#3-关系类型说明)
4. [数据流图](#4-数据流图)
5. [多租户模型](#5-多租户模型)

---

## 1. 核心实体关系图

### 1.1 完整 ER 图

```mermaid
erDiagram
    atom ||--o{ reference : "from"
    atom ||--o{ reference : "to"
    entity ||--o{ reference : "from"
    entity ||--o{ reference : "to"
    entity ||--o{ timeline : "indexed_by"
    
    atom {
        record id "ULID"
        string tenant_id "default"
        string type "function|class|..."
        string content
        string name
        string signature
        array params
        string return_type
        bool is_exported
        bool is_async
        int complexity
        int max_nesting_depth
        object docstring
        int start_line
        int end_line
        object metadata
        int version
        datetime created_at
        datetime updated_at
    }
    
    entity {
        record id "ULID"
        string tenant_id "default"
        string type "memory|backlog|wiki|code"
        string abstract
        object overview
        array atoms
        string title
        array aliases
        array outgoing_links
        array incoming_links
        string priority
        string status
        string scene
        float estimated_hours
        float actual_hours
        string file_path
        string language
        object quality_score
        object complexity_metrics
        array tags
        string project
        string created_by
        datetime created_at
        datetime updated_at
    }
    
    reference {
        record id "ULID"
        string tenant_id "default"
        string type "depends_on|blocks|calls|..."
        record in "from atom/entity"
        record out "to atom/entity"
        string file_path
        int line
        int column
        object metadata
        float weight
        string created_by
        datetime created_at
    }
    
    timeline {
        record id "ULID"
        string tenant_id "default"
        int year
        int month
        int day
        record entity_id
        string entity_type
        datetime created_at
    }
```

### 1.2 表结构说明

| 表名 | 类型 | 主键 | 说明 |
|------|------|------|------|
| **atom** | NORMAL | ULID | 原子级数据（函数、类、任务等） |
| **entity** | NORMAL | ULID | 实体级数据（记忆、Backlog、Wiki） |
| **reference** | RELATION | ULID | 原子/实体间关系 |
| **timeline** | NORMAL | ULID | 时间线索引 |

---

## 2. 图关系模型

### 2.1 代码关系模型

```mermaid
erDiagram
    atom ||--o{ atom : "calls"
    atom ||--o{ atom : "imports"
    atom ||--o{ entity : "part_of"
    
    atom {
        record id
        string type "function|class|interface"
        string name
        string signature
        int complexity
    }
    
    entity {
        record id
        string type "code"
        string title
        string file_path
        string language
    }
```

### 2.2 Backlog 关系模型

```mermaid
erDiagram
    entity ||--o{ entity : "depends_on"
    entity ||--o{ entity : "blocks"
    entity ||--o{ entity : "implements"
    entity ||--o{ atom : "contains"
    
    entity {
        record id
        string type "backlog|task"
        string title
        string priority
        string status
        float estimated_hours
        float actual_hours
    }
    
    atom {
        record id
        string type "task|goal|scope"
        string name
        string status
    }
```

### 2.3 Wiki 关系模型

```mermaid
erDiagram
    entity ||--o{ entity : "wiki_link"
    entity ||--o{ entity : "relates_to"
    
    entity {
        record id
        string type "wiki"
        string title
        array aliases
        array outgoing_links
        array incoming_links
    }
```

### 2.4 记忆关系模型

```mermaid
erDiagram
    entity ||--o{ entity : "relates_to"
    entity ||--o{ timeline : "indexed_by"
    
    entity {
        record id
        string type "memory"
        string abstract
        object overview
        array tags
    }
    
    timeline {
        record id
        int year
        int month
        int day
        record entity_id
    }
```

---

## 3. 关系类型说明

### 3.1 代码关系

| 关系类型 | 从 | 到 | 说明 | 权重因子 |
|----------|-----|-----|------|----------|
| `calls` | atom (function) | atom (function) | 函数调用关系 | 复杂度、频率 |
| `imports` | atom (file) | atom (module) | 模块导入关系 | 导入次数 |
| `part_of` | atom | entity (code) | 组成部分 | 文件内位置 |

### 3.2 Backlog 关系

| 关系类型 | 从 | 到 | 说明 | 权重因子 |
|----------|-----|-----|------|----------|
| `depends_on` | entity | entity | 依赖关系 | 阻塞程度 |
| `blocks` | entity | entity | 阻塞关系 | 优先级 |
| `implements` | entity (task) | entity (backlog) | 实现关系 | 完成度 |

### 3.3 Wiki 关系

| 关系类型 | 从 | 到 | 说明 | 权重因子 |
|----------|-----|-----|------|----------|
| `wiki_link` | entity | entity | Wiki 双向链接 | 链接频率 |
| `relates_to` | entity | entity | 一般关联 | 语义相似度 |

### 3.4 通用关系

| 关系类型 | 从 | 到 | 说明 | 权重因子 |
|----------|-----|-----|------|----------|
| `indexed_by` | entity | timeline | 时间索引 | 时间衰减 |

---

## 4. 数据流图

### 4.1 代码分析数据流

```mermaid
flowchart TD
    A[Source Code] --> B[tree-sitter Parser]
    B --> C[AST]
    C --> D[Symbol Extractor]
    D --> E[Atoms]
    E --> F[Relation Builder]
    F --> G[Call Relations]
    F --> H[Import Relations]
    G --> I[Reference Table]
    H --> I
    E --> J[Entity Table]
    J --> K[Atom Table]
```

### 4.2 记忆存储数据流

```mermaid
flowchart TD
    A[Memory Content] --> B[Content Splitter]
    B --> C[L0 Abstract]
    B --> D[L1 Overview]
    B --> E[L2 Content]
    C --> F[Entity Table]
    D --> F
    E --> G[External Storage]
    F --> H[Timeline Index]
    F --> I[Meilisearch Index]
```

### 4.3 实时同步数据流

```mermaid
flowchart TD
    A[Database Change] --> B[ChangeFeed]
    B --> C[WebSocket Server]
    C --> D[DIFF Generator]
    D --> E[JSON Patch]
    E --> F[ReliableWebSocket]
    F --> G[Client]
```

---

## 5. 多租户模型

### 5.1 逻辑隔离（当前）

```mermaid
erDiagram
    tenant ||--o{ atom : "owns"
    tenant ||--o{ entity : "owns"
    tenant ||--o{ reference : "owns"
    
    tenant {
        string tenant_id PK
        string name
        string plan
    }
    
    atom {
        record id PK
        string tenant_id FK
        string type
    }
    
    entity {
        record id PK
        string tenant_id FK
        string type
    }
    
    reference {
        record id PK
        string tenant_id FK
        string type
    }
```

### 5.2 物理隔离（未来）

```mermaid
erDiagram
    tenant ||--|| database : "has"
    database ||--o{ atom : "contains"
    database ||--o{ entity : "contains"
    
    tenant {
        string tenant_id PK
        string database_name
    }
    
    database {
        string name PK
        string namespace
    }
```

### 5.3 租户字段说明

| 表名 | 租户字段 | 默认值 | 索引 |
|------|----------|--------|------|
| atom | tenant_id | 'default' | ✅ 复合索引 |
| entity | tenant_id | 'default' | ✅ 复合索引 |
| reference | tenant_id | 'default' | ✅ 复合索引 |
| timeline | tenant_id | 'default' | ✅ 复合索引 |

---

## 参考文档

- [DATABASE-v3.2-SCHEMA.md](./DATABASE-v3.2-SCHEMA.md) - 完整 Schema 定义
- [BACKEND-v3.2-PRECOMPUTE.md](./BACKEND-v3.2-PRECOMPUTE.md) - 预计算服务设计
- [BACKEND-v3.2-WEBSOCKET.md](./BACKEND-v3.2-WEBSOCKET.md) - WebSocket 设计

---

_文档版本: v3.2.0_  
_最后更新: 2026-04-10_
