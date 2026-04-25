# Wrapper Service - Agent 指南

> **Scope**: 包装层核心服务 (端口 18008)  
> **Stack**: FastAPI + Pydantic v2 + SurrealDB + Meilisearch  
> **Pattern**: Router-Service-Utils 三层架构 + Mixin 组合

---

## Structure

```text
wrapper/src/
├── main.py              # FastAPI app + lifespan (352行)
├── models.py            # 17个Pydantic模型
├── config.py            # 配置dataclass (AppConfig/SurrealDBConfig/...)
├── state.py             # 全局单例 (避免循环导入)
├── routers/ (23个)      # HTTP路由层
├── services/ (14个)     # 业务逻辑层
├── utils/ (16个)        # 基础设施层
│   └── memory_manager/  # 核心Mixin (11个模块)
└── websocket/ (12个)    # WebSocket实时推送
```

---

## Where to Look

| Task | Location | Notes |
|------|----------|-------|
| 添加新API端点 | `routers/` | 每个router一个`APIRouter` |
| 修改业务逻辑 | `services/` | 纯Python，无HTTP |
| 数据库操作 | `utils/memory_manager/` | Mixin模式组合 |
| 配置变更 | `config.py` | dataclass + os.getenv |
| 全局状态 | `state.py` | lifespan初始化后可用 |
| WebSocket | `websocket/` | 可靠重连 + 心跳 |

---

## Conventions

### Import Pattern

**相对导入** (标准):

```python
from .. import state
from ..config import config
from ..exceptions import ValidationError
```

**例外** (不要学习):

```python
# crud.py 使用绝对导入 - 历史遗留
from wrapper.src.utils.exceptions import EmbeddingError
```

### Mixin 组合模式

```python
class MemoryManager(
    StubsMixin,      # 11个stub端点
    CrudMixin,       # CRUD操作
    SearchMixin,     # 向量搜索
    SyncMixin,       # 多设备同步
    RelationsMixin,  # 图关系
    DedupMixin,      # 去重
    MeiliSyncMixin,  # Meilisearch同步
    CodeAnalysisMixin,  # 代码分析
    AuditMixin,      # 审计日志
    LookupMixin,     # 查询API
):
```

**状态共享**: 通过 `self._db`, `self._embedding_service_url` 隐式耦合

### 异常处理

```python
# 使用项目自定义异常
from ..exceptions import ValidationError, ServiceUnavailableError

# 路由层捕获并转换HTTPException
except ValidationError as e:
    raise HTTPException(status_code=400, detail=str(e))
```

**禁止**: 裸 `except:` (tests/performance/benchmark.py 有4处遗留)

### 日志格式

```python
logger = logging.getLogger(__name__)

# 带前缀的日志
logger.warning("[Auto Analyze] 分析失败: %s", error)
logger.info("[FingerprintManager] 生成指纹: %s", file_path)
```

---

## Anti-Patterns (THIS PROJECT)

| 禁止项 | 原因 | 现状 |
|--------|------|------|
| 裸 `except:` | 捕获KeyboardInterrupt | benchmark.py有4处 |
| 硬编码密码 | 安全风险 | config.py有默认值"root" |
| `global`单例 | 非线程安全 | tracing.py等6处 |
| 泛型`except Exception` | 吞掉真正错误 | 141处待修复 |
| `type: ignore`滥用 | 类型债务 | 18处 |
| 绝对导入 | 破坏可移植性 | crud.py唯一例外 |

---

## Commands

```bash
# 启动服务
uv run python -m wrapper.src.main

# 或统一启动
uv run python start_services.py

# 测试
uv run pytest tests/ -m unit
uv run pytest tests/ -m integration

# 代码检查 (注意: wrapper/不在Ruff默认scope)
uv run ruff check wrapper/src/
uv run pyright wrapper/src/
```

---

## Notes

- **端口**: 18008 (v3.2从17999迁移)
- **Python**: 3.10+ required
- **Pydantic**: v2 API (`field_validator` not `validator`)
- **Ruff Scope**: `pyproject.toml`中`include = ["src/**/*.py"]` **不包含** wrapper/，需显式指定
- **Pyright**: 排除`utils/memory_manager`整个目录
- **状态初始化**: 必须在`main.py` lifespan中完成，routers通过`state`读取
