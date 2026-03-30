# 后端 Schema 升级方案 - 代码分析集成

> **版本**: v1.0  
> **日期**: 2026-03-31  
> **状态**: 待审核（仅文档，未动代码）  
> **关联**: CODE-ANALYSIS-DESIGN-v1.2.md

---

## 1. 当前 Schema 状态

### 1.1 现有 CodeAnalysisResult（Python）

文件: `wrapper/src/utils/code_analyzer.py` 第 26-42 行

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

### 1.2 现有 MemoryItem（Python）

文件: `wrapper/src/main.py` 第 48-58 行

```python
class MemoryItem(BaseModel):
    content: str
    abstract: str | None = None
    overview: str | None = None
    type: str = "general"
    tags: list[str] = []
    metadata: dict[str, Any] = {}   # ← code_analysis 存放于此
    project_id: str = "global"
    source: str = "api"
    source_id: str | None = None
    local_id: str | None = None
```

### 1.3 现有 to_metadata_dict() 序列化

文件: `wrapper/src/utils/code_analyzer.py` 第 44-58 行

```python
def to_metadata_dict(self) -> dict[str, Any]:
    return {
        "language": self.language,
        "functions": self.functions,
        "classes": self.classes,
        "imports": self.imports,
        "exports": self.exports,
        "comments_count": len(self.comments),
        "docstrings_count": len(self.docstrings),
        "dependencies": self.dependencies,
        "complexity": self.complexity_metrics,
        "analyzed_at": self.analyzed_at,
        "analyzer_version": self.analyzer_version,
    }
```

---

## 2. 升级目标

### 2.1 插件端数据结构（TypeScript）

```typescript
interface CodeAnalysisResult {
  language: string;
  analyzer: string;
  analyzed_at: string;
  analyzer_version: string;
  functions: FunctionSymbol[];
  classes: ClassSymbol[];
  interfaces: InterfaceSymbol[];    // 新增
  imports: ImportSymbol[];          // 升级：string[] → ImportSymbol[]
  exports: ExportSymbol[];          // 升级：string[] → ExportSymbol[]
  dependencies: DependencyInfo;     // 升级：string[] → {internal, external, builtin}
  complexity_metrics: ComplexityMetrics;  // 扩展字段
  errors?: ParseError[];            // 新增
  warnings?: ParseWarning[];        // 新增
}
```

### 2.2 需要升级的字段

| 字段 | 当前类型 | 目标类型 | 变更说明 |
|------|---------|---------|---------|
| `functions` | `List[Dict]` | `List[FunctionSymbol]` | 结构化为具名字段 |
| `classes` | `List[Dict]` | `List[ClassSymbol]` | 结构化为具名字段 |
| `imports` | `List[str]` | `List[ImportSymbol]` | 升级为对象数组 |
| `exports` | `List[str]` | `List[ExportSymbol]` | 升级为对象数组 |
| `dependencies` | `List[str]` | `DependencyInfo` | 拆分为 internal/external/builtin |
| `complexity_metrics` | `Dict[str, int]` | `ComplexityMetrics` | 新增 max/avg 复杂度 |
| `interfaces` | 不存在 | `List[InterfaceSymbol]` | 新增 |
| `analyzer` | 不存在 | `str` | 新增（区分 tree-sitter/oxc/fallback） |
| `errors` | 不存在 | `List[ParseError]` | 新增 |
| `warnings` | 不存在 | `List[ParseWarning]` | 新增 |

---

## 3. 升级方案

### 3.1 后端 Python 升级后的 CodeAnalysisResult

```python
@dataclass
class FunctionSymbol:
    """函数符号"""
    name: str
    start_line: int
    end_line: int
    params: List[Dict[str, Any]]       # [{name, type?, optional}]
    return_type: Optional[str] = None
    is_exported: bool = False
    is_async: bool = False


@dataclass
class ClassSymbol:
    """类符号"""
    name: str
    start_line: int
    end_line: int
    methods: List[str]                 # 方法名列表
    properties: List[str]              # 属性名列表


@dataclass
class InterfaceSymbol:
    """接口符号"""
    name: str
    start_line: int
    end_line: int
    methods: List[str]
    properties: List[str]


@dataclass
class ImportSymbol:
    """导入符号"""
    source: str                        # 模块路径
    imported_names: List[str]          # 导入的名称
    is_default: bool = False
    is_namespace: bool = False


@dataclass
class ExportSymbol:
    """导出符号"""
    name: str
    type: str                          # "function" | "class" | "interface" | "variable"
    is_default: bool = False


@dataclass
class DependencyInfo:
    """依赖信息"""
    internal: List[str] = field(default_factory=list)
    external: List[str] = field(default_factory=list)
    builtin: List[str] = field(default_factory=list)


@dataclass
class ComplexityMetrics:
    """复杂度指标"""
    cyclomatic: int = 0
    lines_of_code: int = 0
    function_count: int = 0
    class_count: int = 0
    max_function_complexity: int = 0
    average_function_complexity: float = 0.0


@dataclass
class ParseWarning:
    """解析警告"""
    type: str                          # "degraded" | "unsupported_syntax" | ...
    message: str = ""
    details: Optional[str] = None


@dataclass
class CodeAnalysisResult:
    """代码分析结果 v2.0 - 与插件端对齐"""

    # 基础信息
    language: str
    analyzer: str                      # "tree-sitter" | "oxc" | "fallback"
    analyzed_at: str = ""
    analyzer_version: str = "1.0.0"

    # 符号
    functions: List[Dict[str, Any]] = field(default_factory=list)
    classes: List[Dict[str, Any]] = field(default_factory=list)
    interfaces: List[Dict[str, Any]] = field(default_factory=list)   # 新增
    imports: List[Any] = field(default_factory=list)                 # 兼容 str 和 ImportSymbol
    exports: List[Any] = field(default_factory=list)                 # 兼容 str 和 ExportSymbol

    # 依赖
    dependencies: Any = field(default_factory=list)  # 兼容 List[str] 和 DependencyInfo

    # 复杂度
    complexity_metrics: Dict[str, Any] = field(default_factory=dict)

    # 错误处理
    errors: List[Dict[str, Any]] = field(default_factory=list)       # 新增
    warnings: List[Dict[str, Any]] = field(default_factory=list)     # 新增

    def to_metadata_dict(self) -> dict[str, Any]:
        """序列化为 metadata.code_analysis 字典"""
        return {
            "language": self.language,
            "analyzer": self.analyzer,
            "analyzed_at": self.analyzed_at,
            "analyzer_version": self.analyzer_version,
            "functions": self.functions,
            "classes": self.classes,
            "interfaces": self.interfaces,
            "imports": self.imports,
            "exports": self.exports,
            "dependencies": self.dependencies,
            "complexity_metrics": self.complexity_metrics,
            "errors": self.errors,
            "warnings": self.warnings,
        }
```

### 3.2 向后兼容策略

| 场景 | 处理方式 |
|------|---------|
| 插件端上传新格式 | 直接存储，所有新字段可选 |
| 后端本地解析（旧格式） | 自动转换：`imports: ["lodash"]` → `imports: [{source: "lodash", ...}]` |
| 旧数据无 `interfaces` 字段 | 默认空数组 `[]` |
| 旧数据 `dependencies` 是 `List[str]` | 保持原样，不强制转换 |
| 搜索时字段缺失 | Meilisearch 过滤条件处理 null |

**关键原则**：新字段全部 `Optional` 或 `default_factory`，旧数据不受影响。

---

## 4. SurrealDB Schema 变更

### 4.1 现有 Schema

文件: `scripts/init_surrealdb.surql`

记忆表 `memory` 使用灵活 Schema（`SCHEMAFULL` 但 `metadata` 为 `any` 类型），无需修改表结构。

### 4.2 需要的变更

**无需修改 SurrealDB Schema**。

理由：

- `metadata` 字段类型为 `any`，天然支持嵌套结构扩展
- 新增的 `code_analysis` 子字段在 `metadata` 内部
- SurrealDB 无需预定义嵌套字段

### 4.3 唯一键 Upsert 逻辑

代码记忆需要 `file_path` + `project_id` 作为唯一键：

```sql
-- 查询是否已存在同文件的代码记忆
SELECT id FROM memory
WHERE metadata.code_analysis.file_path = $file_path
  AND project_id = $project_id
  AND tenant_id = $tenant_id
LIMIT 1;

-- 如存在 → UPDATE；不存在 → CREATE
```

**实现位置**: `memory_manager.py` 的 `upload_memories()` 方法，新增 upsert 分支。

---

## 5. 变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `wrapper/src/utils/code_analyzer.py` | 修改 | 升级 `CodeAnalysisResult` 数据结构 |
| `wrapper/src/utils/memory_manager.py` | 修改 | 新增 upsert 逻辑（`file_path` + `project_id`） |
| `wrapper/src/main.py` | 无需修改 | `metadata: dict[str, Any]` 天然支持 |
| `scripts/init_surrealdb.surql` | 无需修改 | `metadata` 为 `any` 类型 |

---

## 6. 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 旧数据与新格式不兼容 | 低 | 所有新字段可选，默认空数组 |
| 后端本地解析结果格式不同 | 低 | Phase 1 插件端解析为主，后端解析为辅 |
| upsert 误覆盖 | 中 | 使用 `file_path` + `project_id` + `tenant_id` 三重校验 |

---

*文档结束 - 等待审核确认后执行代码变更*
