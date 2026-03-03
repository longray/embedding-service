# embedding_service 项目质量标准 - 代码表达规范

> **项目**: embedding_service (Python 3.10-3.12 + FastAPI)  
> **生成时间**: 2026-03-03  
> **基于标准**: code-quality-standard v1.2.0

---

## 1. Python 格式化规范

# Src: docs/02-FORMATTING-STANDARDS.md#2.2

### 1.1 工具选择

**主工具**: Ruff Format ≥0.6.0

**选择理由**:
- 现代化、快速（比Black快10-100倍）
- 与Ruff Lint集成，统一工具链
- 零配置，开箱即用

### 1.2 格式化配置

```toml
# pyproject.toml
[tool.ruff]
target-version = "py310"  # Python 3.10
line-length = 120          # 行宽限制
indent-width = 4           # 缩进宽度
```

**配置说明**:
- `line-length = 120`: 单行最大字符数（PEP 8推荐79，但120更实用）
- `indent-width = 4`: Python标准缩进（4个空格）
- `target-version = "py310"`: 确保语法兼容Python 3.10+

### 1.3 EditorConfig 集成

```ini
# .editorconfig
[*.py]
indent_style = space
indent_size = 4
insert_final_newline = true
trim_trailing_whitespace = true
charset = utf-8
end_of_line = lf
```

### 1.4 格式化命令

```bash
# 格式化代码
uv run ruff format src/

# 检查格式（不修改）
uv run ruff format --check src/

# 格式化特定文件
uv run ruff format src/qwen3_embedding_service/embedding_service.py
```

---

## 2. Python Linting 规范

# Src: docs/03-LINTING-STANDARDS.md#3.2

### 2.1 工具选择

**主工具**: Ruff Lint ≥0.6.0

**规则集配置**:

```toml
# pyproject.toml
[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors (PEP 8语法错误)
    "W",   # pycodestyle warnings (代码风格警告)
    "F",   # pyflakes (未定义变量、未使用导入)
    "I",   # isort (导入排序)
    "B",   # flake8-bugbear (常见Bug模式)
    "C4",  # flake8-comprehensions (列表推导优化)
    "UP",  # pyupgrade (旧语法升级建议)
    "RUF", # Ruff专用规则 (现代最佳实践)
    "S",   # flake8-bandit (安全相关规则)
]

ignore = [
    "E501",   # line too long (由line-length配置处理)
    "W293",   # blank line contains whitespace (自动格式化处理)
    "RUF001", # 忽略中文标点符号检查
    "RUF002", # 忽略docstring中的中文标点符号
    "RUF003", # 忽略注释中的中文标点符号
]
```

### 2.2 规则集说明

| 规则集 | 覆盖范围 | 优先级 |
|--------|----------|--------|
| **E/W** | PEP 8语法错误和警告 | P1 |
| **F** | 未定义变量、未使用导入 | P1 |
| **I** | 导入顺序、未使用导入 | P2 |
| **B** | 常见Bug模式 | P1 |
| **C4** | 列表推导优化 | P2 |
| **UP** | 旧语法升级建议 | P3 |
| **RUF** | 现代最佳实践 | P2-P3 |
| **S** | 安全相关规则 | P0-P1 |

### 2.3 常用规则示例

| 规则 | 说明 | 示例 | 严重程度 |
|------|------|------|----------|
| `F401` | 未使用的导入 | `import os` (未使用) | warning |
| `F841` | 未使用的变量 | `x = 1` (x未使用) | warning |
| `I001` | 导入未排序 | 导入顺序混乱 | warning |
| `B901` | 使用 `x == True` | 应使用 `if x:` | warning |
| `S101` | 使用 assert | 生产代码禁用assert | warning |

### 2.4 Linting 命令

```bash
# 检查代码
uv run ruff check src/

# 自动修复问题
uv run ruff check src/ --fix

# 检查特定文件
uv run ruff check src/qwen3_embedding_service/llm_service.py
```

---

## 3. 命名规范

# Src: docs/03-LINTING-STANDARDS.md#3.2.2

### 3.1 Python 命名约定

| 类型 | 规范 | 示例 | 规则 |
|------|------|------|------|
| **模块** | snake_case | `embedding_service.py` | 小写+下划线 |
| **类** | PascalCase | `EmbeddingService` | 首字母大写 |
| **函数** | snake_case | `get_embedding()` | 小写+下划线 |
| **变量** | snake_case | `model_name` | 小写+下划线 |
| **常量** | UPPER_CASE | `MAX_BATCH_SIZE` | 全大写+下划线 |
| **私有** | _leading_underscore | `_internal_method()` | 前导下划线 |

### 3.2 项目特定约定

**FastAPI 路由**:
```python
@app.post("/api/v1/embeddings")  # kebab-case for URLs
async def create_embeddings(request: EmbeddingRequest):
    pass
```

**配置类**:
```python
class ServiceConfig:  # PascalCase
    model_path: str
    batch_size: int = 32  # snake_case attributes
```

---

## 4. 代码风格要点

### 4.1 导入顺序

```python
# 1. 标准库
import os
import sys
from typing import List, Optional

# 2. 第三方库
from fastapi import FastAPI
from pydantic import BaseModel

# 3. 本地模块
from qwen3_embedding_service.config import Config
```

### 4.2 类型注解

# Src: docs/04-TYPE-CHECKING-STANDARDS.md#4.2

```python
# ✅ 推荐：完整类型注解
def process_text(text: str, max_length: int = 512) -> List[float]:
    pass

# ❌ 避免：缺少类型注解
def process_text(text, max_length=512):
    pass
```

### 4.3 文档字符串

```python
def get_embedding(text: str) -> List[float]:
    """获取文本的向量表示。
    
    Args:
        text: 输入文本
        
    Returns:
        向量列表（维度：768）
    """
    pass
```

---

## 5. 实施检查清单

### 5.1 格式化检查
- [ ] 所有Python文件通过 `ruff format --check`
- [ ] 行宽不超过120字符
- [ ] 使用4空格缩进
- [ ] 文件末尾有换行符

### 5.2 Linting检查
- [ ] 所有Python文件通过 `ruff check`
- [ ] 无未使用的导入
- [ ] 导入顺序正确
- [ ] 无未使用的变量

### 5.3 命名检查
- [ ] 模块名使用snake_case
- [ ] 类名使用PascalCase
- [ ] 函数/变量使用snake_case
- [ ] 常量使用UPPER_CASE

---

**文档版本**: 1.0.0  
**最后更新**: 2026-03-03  
**规范来源**: D:\github\code-quality-standard
