# 后端 v3.2 预计算服务设计

> **版本**: v3.2.0  
> **日期**: 2026-04-10  
> **状态**: 实施版  
> **目标**: 将代码分析重构为 PrecomputeService，实现服务化、批量处理、增量更新、性能监控

---

## 目录

1. [设计目标](#1-设计目标)
2. [架构设计](#2-架构设计)
3. [核心组件](#3-核心组件)
4. [实现细节](#4-实现细节)
5. [测试验证](#5-测试验证)

---

## 1. 设计目标

### 1.1 功能要求

| 功能         | 要求              | 实现方式                        |
| ------------ | ----------------- | ------------------------------- |
| **AST 解析** | tree-sitter Query | PrecomputeService.\_parse_ast() |
| **符号提取** | 函数、类、接口    | \_extract_symbols()             |
| **批量创建** | 批量插入 Atoms    | \_create_atoms_batch()          |
| **双向引用** | Entity ↔ Atoms    | SurrealDB REFERENCE             |
| **性能监控** | 耗时/内存/CPU     | PerformanceMonitor              |
| **增量更新** | SHA256 指纹       | \_calculate_fingerprint()       |
| **并发控制** | 最大 5 并发       | asyncio.Semaphore               |

### 1.2 性能指标

- 处理速度：> 1000 行/秒
- 内存占用：< 100MB（大文件）
- 批量插入：> 100 条/批次
- 增量识别率：> 95%

---

## 2. 架构设计

### 2.1 组件关系

```
┌─────────────────────────────────────────────────────────────┐
│                  PrecomputeService                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │   Parser    │───►│   Analyzer  │───►│   Storage   │    │
│  │             │    │             │    │             │    │
│  │ • tree-sit│    │ • extract   │    │ • SurrealDB │    │
│  │ • Query   │    │ • relations │    │ • Meilisearc│    │
│  │             │    │             │    │             │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
│         │                  │                  │            │
│         ▼                  ▼                  ▼            │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              PerformanceMonitor                      │  │
│  │  • duration_ms  • memory_mb  • cpu_percent          │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │           ConcurrencyControl                         │  │
│  │  • Semaphore(5)  • dedup set  • queue               │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 数据流

```
1. 触发
   File Save ──► PrecomputeService.precompute()

2. 解析
   Source Code ──► tree-sitter ──► AST

3. 提取
   AST ──► Query ──► Symbols (functions, classes)

4. 指纹
   Content ──► SHA256 ──► Fingerprint

5. 检测
   Fingerprint ──► Compare ──► Skip if unchanged

6. 批量
   Symbols ──► Batch ──► Create Atoms

7. 关联
   Atoms ──► RELATE ──► Entity

8. 监控
   Record ──► performance_log
```

---

## 3. 核心组件

### 3.1 PrecomputeService

```python
import time
import psutil
import hashlib
from tree_sitter import Language, Parser
from typing import List, Dict, Any


class PrecomputeService:
    """
    预计算服务

    功能：
    - AST 解析和符号提取
    - 批量创建 Atoms 和 Entity
    - 性能监控和增量更新
    """

    def __init__(self, db: AsyncSurreal, tenant_id: str = "default"):
        self.db = db
        self.tenant_id = tenant_id
        self.parser = Parser()
        self.languages = {}
        self._init_languages()

    def _init_languages(self):
        """初始化语言解析器"""
        try:
            from tree_sitter_python import language as python_lang
            from tree_sitter_javascript import language as js_lang
            from tree_sitter_typescript import language as ts_lang

            self.languages = {
                "python": python_lang,
                "javascript": js_lang,
                "typescript": ts_lang,
            }
        except ImportError:
            pass

    async def precompute(
        self,
        file_path: str,
        source_code: str,
        language: str = "python"
    ) -> Dict[str, Any]:
        """
        完整预计算流程

        Args:
            file_path: 文件路径
            source_code: 源代码
            language: 语言类型

        Returns:
            预计算结果
        """
        start_time = time.perf_counter()
        process = psutil.Process()
        start_mem = process.memory_info().rss / 1024 / 1024

        try:
            # 1. 计算指纹
            fingerprint = self._calculate_fingerprint(source_code)

            # 2. 检查是否变更
            last = await self._get_last_precompute(file_path)
            if last and last.get("fingerprint") == fingerprint:
                return {"skipped": True, "reason": "No changes"}

            # 3. 解析 AST
            ast = self._parse_ast(source_code, language)

            # 4. 提取符号
            symbols = self._extract_symbols(ast, file_path, source_code)

            # 5. 批量创建 Atoms
            atoms = await self._create_atoms_batch(symbols)

            # 6. 创建 Entity
            entity = await self._create_file_entity(file_path, atoms, source_code)

            # 7. 创建 Relations
            await self._create_relations(symbols, atoms)

            # 8. 更新指纹
            await self._update_fingerprint(file_path, fingerprint)

            # 性能监控
            duration = (time.perf_counter() - start_time) * 1000
            memory = process.memory_info().rss / 1024 / 1024 - start_mem
            await self._log_performance(file_path, duration, memory)

            return {
                "entity_id": entity["id"],
                "atoms_count": len(atoms),
                "duration_ms": duration,
                "memory_mb": memory,
                "success": True
            }

        except Exception as e:
            logger.error(f"Precompute error: {e}")
            raise

    def _calculate_fingerprint(self, content: str) -> str:
        """计算文件指纹"""
        return hashlib.sha256(content.encode()).hexdigest()

    def _parse_ast(self, source_code: str, language: str):
        """解析 AST"""
        if language not in self.languages:
            raise ValueError(f"Unsupported language: {language}")

        self.parser.set_language(self.languages[language])
        return self.parser.parse(bytes(source_code, "utf8"))

    def _extract_symbols(self, ast, file_path: str, source_code: str) -> List[Dict]:
        """提取符号（使用 tree-sitter Query）"""
        symbols = []
        root_node = ast.root_node

        # Query 模式匹配
        query_str = """
        (function_definition
          name: (identifier) @func_name
          parameters: (parameters) @params)

        (class_definition
          name: (identifier) @class_name)
        """

        # 递归遍历
        def visit_node(node):
            if node.type == "function_definition":
                name_node = node.child_by_field_name("name")
                name = source_code[name_node.start_byte:name_node.end_byte] if name_node else "anonymous"

                symbols.append({
                    "type": "function",
                    "name": name,
                    "content": source_code[node.start_byte:node.end_byte],
                    "file_path": file_path,
                    "start_line": node.start_point[0],
                    "end_line": node.end_point[0],
                    "tenant_id": self.tenant_id
                })

            for child in node.children:
                visit_node(child)

        visit_node(root_node)
        return symbols

    async def _create_atoms_batch(self, symbols: List[Dict]) -> List[Dict]:
        """批量创建 Atoms"""
        if not symbols:
            return []

        # SurrealDB 批量插入
        result = await self.db.query("""
            RETURN array::flatten(
                $symbols.map(|$s| CREATE atom CONTENT $s)
            )
        """, {"symbols": symbols})

        return result[0]["result"] if result else []

    async def _create_file_entity(self, file_path: str, atoms: List[Dict], source_code: str) -> Dict:
        """创建文件级 Entity"""
        entity_data = {
            "type": "code",
            "title": file_path.split("/")[-1],
            "abstract": f"File with {len(atoms)} symbols",
            "overview": {
                "language": "python",
                "lines_of_code": len(source_code.splitlines()),
                "function_count": len(atoms)
            },
            "atoms": [a["id"] for a in atoms],
            "file_path": file_path,
            "project": file_path.split("/")[0] if "/" in file_path else "default",
            "tenant_id": self.tenant_id
        }

        return await self.db.create("entity", entity_data)

    async def _create_relations(self, symbols: List[Dict], atoms: List[Dict], file_path: str):
        """
        创建调用关系
        
        分析函数调用关系并创建 SurrealDB RELATE 记录。
        支持循环调用检测和关系权重计算。
        
        算法复杂度:
        - 时间: O(n*m) n=符号数, m=每个符号的调用数
        - 空间: O(n) 用于构建调用图
        
        Args:
            symbols: 提取的符号列表
            atoms: 已创建的 atom 记录
            file_path: 源文件路径
        """
        # 构建符号名称到 atom ID 的映射
        atom_map = {a["name"]: a["id"] for a in atoms if a.get("name")}
        
        # 收集所有调用关系
        relations = []
        for symbol in symbols:
            if symbol["type"] != "function":
                continue
                
            caller_name = symbol.get("name")
            caller_id = atom_map.get(caller_name)
            
            if not caller_id:
                continue
            
            # 提取调用表达式
            calls = symbol.get("metadata", {}).get("calls", [])
            for call in calls:
                callee_name = call.get("name")
                callee_id = atom_map.get(callee_name)
                
                if callee_id and callee_id != caller_id:
                    relations.append({
                        "from_id": caller_id,
                        "to_id": callee_id,
                        "type": "calls",
                        "file_path": file_path,
                        "line": call.get("line"),
                        "column": call.get("column"),
                        "weight": self._calculate_call_weight(symbol, call)
                    })
        
        # 检测循环调用
        cycles = self._detect_cycles(relations)
        if cycles:
            logger.warning(f"Detected {len(cycles)} circular call chains in {file_path}")
            for cycle in cycles[:5]:  # 只记录前5个
                logger.warning(f"  Cycle: {' -> '.join(cycle)}")
        
        # 批量创建关系（使用 SurrealDB RELATE）
        await self._batch_create_relations(relations)
    
    def _calculate_call_weight(self, symbol: Dict, call: Dict) -> float:
        """
        计算调用关系权重
        
        基于以下因素:
        - 调用频率 (0.3)
        - 调用者复杂度 (0.3)
        - 参数复杂度 (0.2)
        - 是否跨文件 (0.2)
        
        Returns:
            权重值 (0.0 - 1.0)
        """
        weight = 0.5  # 基础权重
        
        # 调用频率（如果有统计信息）
        call_count = call.get("count", 1)
        weight += min(call_count / 10, 0.3)  # 最多 +0.3
        
        # 调用者复杂度
        complexity = symbol.get("complexity", 1)
        weight += min(complexity / 20, 0.3)  # 最多 +0.3
        
        # 参数数量
        arg_count = len(call.get("args", []))
        weight += min(arg_count / 10, 0.2)  # 最多 +0.2
        
        # 归一化到 0-1
        return min(weight, 1.0)
    
    def _detect_cycles(self, relations: List[Dict]) -> List[List[str]]:
        """
        检测循环调用
        
        使用 DFS 算法检测图中的环。
        
        算法复杂度:
        - 时间: O(V + E) V=顶点数, E=边数
        - 空间: O(V) 用于访问标记
        
        Args:
            relations: 调用关系列表
            
        Returns:
            检测到的循环链列表
        """
        # 构建邻接表
        graph = {}
        for rel in relations:
            from_id = rel["from_id"]
            to_id = rel["to_id"]
            if from_id not in graph:
                graph[from_id] = []
            graph[from_id].append(to_id)
        
        cycles = []
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    result = dfs(neighbor)
                    if result:
                        return result
                elif neighbor in rec_stack:
                    # 发现环
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
            
            path.pop()
            rec_stack.remove(node)
            return None
        
        for node in graph:
            if node not in visited:
                dfs(node)
        
        return cycles
    
    async def _batch_create_relations(self, relations: List[Dict]):
        """
        批量创建关系
        
        使用 SurrealDB RELATE 语句批量创建关系，
        每批最多 100 条以避免超时。
        
        Args:
            relations: 关系列表
        """
        if not relations:
            return
        
        BATCH_SIZE = 100
        
        for i in range(0, len(relations), BATCH_SIZE):
            batch = relations[i:i + BATCH_SIZE]
            
            # 构建 RELATE 语句
            relate_statements = []
            for rel in batch:
                stmt = f'''
                    RELATE {rel["from_id"]}->reference->{rel["to_id"]} SET
                        type = "{rel["type"]}",
                        tenant_id = "{self.tenant_id}",
                        file_path = "{rel.get("file_path", "")}",
                        line = {rel.get("line", "NULL")},
                        column = {rel.get("column", "NULL")},
                        weight = {rel["weight"]},
                        created_at = time::now()
                '''
                relate_statements.append(stmt)
            
            # 执行批量查询
            query = ";".join(relate_statements)
            await self.db.query(query)
            
            logger.debug(f"Created {len(batch)} relations (batch {i//BATCH_SIZE + 1})")

    async def _log_performance(self, file_path: str, duration_ms: float, memory_mb: float):
        """记录性能指标"""
        await self.db.query("""
            CREATE performance_log SET
                tenant_id = $tid,
                operation = 'precompute',
                file_path = $file,
                duration_ms = $duration,
                memory_mb = $memory,
                timestamp = time::now()
        """, {
            "tid": self.tenant_id,
            "file": file_path,
            "duration": duration_ms,
            "memory": memory_mb
        })
```

### 3.2 PerformanceMonitor

```python
import time
import psutil
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    duration_ms: float
    memory_mb: float
    cpu_percent: float


class PerformanceMonitor:
    """性能监控"""

    @contextmanager
    async def monitor(self, operation: str, db: AsyncSurreal):
        start_time = time.perf_counter()
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024
        start_cpu = process.cpu_percent()

        try:
            yield self
        finally:
            duration = (time.perf_counter() - start_time) * 1000
            end_memory = process.memory_info().rss / 1024 / 1024
            cpu_usage = process.cpu_percent() - start_cpu

            await db.query("""
                CREATE performance_log SET
                    operation = $op,
                    duration_ms = $duration,
                    memory_delta_mb = $memory,
                    cpu_percent = $cpu,
                    timestamp = time::now()
            """, {
                "op": operation,
                "duration": duration,
                "memory": end_memory - start_memory,
                "cpu": cpu_usage
            })
```

### 3.3 ConcurrencyControl

```python
import asyncio
from typing import Set


class ConcurrencyControl:
    """并发控制"""

    def __init__(self, max_concurrent: int = 5):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.processing: Set[str] = set()
        self.queue = asyncio.Queue()

    async def precompute(self, file_path: str, tenant_id: str, func):
        """带并发控制的预计算"""
        key = f"{tenant_id}:{file_path}"

        # 去重
        if key in self.processing:
            return {"skipped": True, "reason": "Already processing"}

        self.processing.add(key)
        try:
            async with self.semaphore:
                return await func()
        finally:
            self.processing.discard(key)
```

---

## 4. 实现细节

### 4.1 触发时机

```python
# 1. 文件保存时自动触发
async def on_file_save(file_path: str, source_code: str, tenant_id: str = "default"):
    service = PrecomputeService(db, tenant_id)
    return await service.precompute(file_path, source_code)

# 2. CLI 手动触发
# opencode-memory analyze src/utils.ts --tenant-id=default

# 3. 批量触发
# opencode-memory analyze --all --tenant-id=default
```

### 4.2 配置

```python
# config.py
class PrecomputeConfig:
    """预计算配置"""

    # 并发
    MAX_CONCURRENT = 5

    # 批量
    BATCH_SIZE = 100

    # 性能
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    TIMEOUT = 300  # 5分钟
```

---

## 5. 测试验证

### 5.1 单元测试

```python
@pytest.mark.asyncio
async def test_precompute_service():
    """测试 PrecomputeService"""
    service = PrecomputeService(db)

    result = await service.precompute(
        "test.py",
        "def hello(): pass",
        "python"
    )

    assert result["success"] is True
    assert result["atoms_count"] == 1
```

### 5.2 性能测试

```python
@pytest.mark.asyncio
async def test_precompute_performance():
    """测试预计算性能"""
    service = PrecomputeService(db)

    # 大文件测试
    large_code = "\n".join([f"def func{i}(): pass" for i in range(1000)])

    result = await service.precompute("large.py", large_code, "python")

    assert result["duration_ms"] < 10000  # < 10s
    assert result["memory_mb"] < 100  # < 100MB
```

### 4.3 关系创建测试

```python
# tests/unit/test_precompute_relations.py
import pytest
from src.services.precompute import PrecomputeService


@pytest.fixture
async def precompute_service(mock_db):
    """测试用的 PrecomputeService 实例"""
    return PrecomputeService(mock_db, tenant_id="test")


class TestCreateRelations:
    """测试调用关系创建"""
    
    async def test_create_simple_call_relation(self, precompute_service):
        """测试简单调用关系创建"""
        symbols = [
            {
                "type": "function",
                "name": "caller",
                "metadata": {
                    "calls": [{"name": "callee", "line": 10, "column": 5}]
                }
            },
            {
                "type": "function",
                "name": "callee",
                "metadata": {"calls": []}
            }
        ]
        
        atoms = [
            {"id": "atom:caller", "name": "caller"},
            {"id": "atom:callee", "name": "callee"}
        ]
        
        await precompute_service._create_relations(
            symbols, atoms, "test.py"
        )
        
        # 验证 RELATE 语句被调用
        mock_db.query.assert_called_once()
        query = mock_db.query.call_args[0][0]
        
        assert "RELATE" in query
        assert "atom:caller->reference->atom:callee" in query
        assert 'type = "calls"' in query
    
    async def test_detect_circular_calls(self, precompute_service):
        """测试循环调用检测"""
        # A -> B -> C -> A (循环)
        relations = [
            {"from_id": "A", "to_id": "B"},
            {"from_id": "B", "to_id": "C"},
            {"from_id": "C", "to_id": "A"},  # 形成环
        ]
        
        cycles = precompute_service._detect_cycles(relations)
        
        assert len(cycles) == 1
        assert cycles[0] == ["A", "B", "C", "A"]
    
    async def test_calculate_call_weight(self, precompute_service):
        """测试权重计算"""
        symbol = {
            "complexity": 10,
            "metadata": {}
        }
        call = {
            "count": 5,
            "args": ["arg1", "arg2", "arg3"]
        }
        
        weight = precompute_service._calculate_call_weight(symbol, call)
        
        assert 0.0 <= weight <= 1.0
        assert weight > 0.5  # 复杂度较高，权重应该较高
    
    async def test_batch_create_relations(self, precompute_service):
        """测试批量关系创建"""
        relations = [
            {
                "from_id": f"atom:func{i}",
                "to_id": f"atom:func{i+1}",
                "type": "calls",
                "file_path": "test.py",
                "line": i,
                "column": 0,
                "weight": 0.5
            }
            for i in range(250)  # 250 条关系，应该分成 3 批
        ]
        
        await precompute_service._batch_create_relations(relations)
        
        # 验证分批次创建（每批 100 条）
        assert mock_db.query.call_count == 3
    
    async def test_skip_self_call(self, precompute_service):
        """测试跳过自调用"""
        symbols = [
            {
                "type": "function",
                "name": "recursive",
                "metadata": {
                    "calls": [{"name": "recursive"}]  # 自调用
                }
            }
        ]
        
        atoms = [
            {"id": "atom:recursive", "name": "recursive"}
        ]
        
        await precompute_service._create_relations(
            symbols, atoms, "test.py"
        )
        
        # 自调用不应该创建关系
        mock_db.query.assert_not_called()
```

---

## 参考文档

- [UNIFIED-ARCHITECTURE-v3.2.md](./UNIFIED-ARCHITECTURE-v3.2.md)
- [BACKEND-v3.2-IMPLEMENTATION.md](./BACKEND-v3.2-IMPLEMENTATION.md)
- [BACKEND-v3.2-WEBSOCKET.md](./BACKEND-v3.2-WEBSOCKET.md)

---

_文档版本: v3.2.0_  
_最后更新: 2026-04-10_
