# Scripts - Agent 指南

> **Scope**: 运维脚本和工具 (30个Python脚本)  
> **Purpose**: 初始化、迁移、诊断、测试、运维

---

## Structure

```text
scripts/
├── init_all.py                    # 一键初始化所有服务
├── init_meilisearch.py            # Meilisearch索引初始化
├── migrate_to_meilisearch.py      # SurrealDB → Meilisearch迁移
├── migrate_v2_to_v32.py          # v2.x → v3.2 schema迁移
├── migrate_file_path.py          # 文件路径字段迁移
├── migrate_add_deduplication.py   # 去重字段迁移
├── clear_all_data.py             # 清空所有数据（调试）
├── benchmark.py                  # 性能基准测试
├── evaluate_memory_search.py     # 搜索质量评估
├── comprehensive_api_test.py     # API全链路测试
├── demo_wrapper.py               # 功能演示
├── diagnose.py                   # 综合诊断工具
├── diagnose_surrealdb.py        # SurrealDB专项诊断
├── check_*.py                    # 各种检查脚本
├── fix_*.py                      # 修复脚本
├── collect-metrics.py           # 指标采集
├── generate-report.py           # 报告生成
├── sync-standards.py            # 质量标准同步
├── update_rtm.py                # RTM文档更新
├── opencode_integration.py      # OpenCode集成测试
└── ...                          # 其他工具脚本
```

---

## Where to Look

| 任务 | 脚本 | 说明 |
|------|------|------|
| 首次部署 | `init_all.py` | 一键初始化SurrealDB+Meilisearch |
| 数据迁移 | `migrate_*.py` | 各种迁移场景 |
| 性能测试 | `benchmark.py` | 基准测试+报告生成 |
| 清空数据 | `clear_all_data.py` | 调试专用（需API Key） |
| 诊断问题 | `diagnose.py` | 综合诊断+修复建议 |
| 检查合规 | `check_design_compliance.py` | 设计规范检查 |
| 修复文档 | `fix_md031.py`, `fix_md040.py` | Markdown格式修复 |

---

## Conventions

### 脚本模板

```python
#!/usr/bin/env python3
"""脚本说明

前置条件:
    - SurrealDB运行中
    - Meilisearch运行中

用法:
    uv run python scripts/script_name.py [--option]

示例:
    uv run python scripts/init_all.py
"""

import argparse
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="脚本说明")
    parser.add_argument("--option", default="value", help="选项说明")
    args = parser.parse_args()
    
    # 脚本逻辑
    logger.info("开始执行...")


if __name__ == "__main__":
    asyncio.run(main())
```

### 命名规范

| 前缀 | 用途 | 示例 |
|------|------|------|
| `init_` | 初始化 | `init_all.py`, `init_meilisearch.py` |
| `migrate_` | 数据迁移 | `migrate_to_meilisearch.py` |
| `check_` | 检查验证 | `check_design_compliance.py` |
| `fix_` | 修复工具 | `fix_md031.py`, `fix_md040.py` |
| `diagnose_` | 诊断工具 | `diagnose.py`, `diagnose_surrealdb.py` |
| `*_test.py` | 测试脚本 | `comprehensive_api_test.py` |

### 环境变量读取

```python
import os

# 标准做法
SURREALDB_URL = os.getenv("SURREALDB_URL", "ws://localhost:18002/rpc")
MEILI_API_KEY = os.getenv("MEILI_API_KEY", "")

# 敏感信息必须验证
if not MEILI_API_KEY:
    raise ValueError("MEILI_API_KEY environment variable is required")
```

---

## Anti-Patterns (脚本开发)

| 问题 | 位置 | 说明 |
|------|------|------|
| 裸`except:` | benchmark.py | 4处，吞掉所有异常 |
| 硬编码sleep | benchmark.py | 等待服务就绪 |
| 无类型注解 | 部分脚本 | 建议添加 |
| 重复代码 | 多个migrate | 应提取公共函数 |

---

## Commands

```bash
# 一键初始化
uv run python scripts/init_all.py

# 数据迁移
uv run python scripts/migrate_to_meilisearch.py --batch-size 200

# 性能基准
uv run python scripts/benchmark.py --iterations 5

# 综合诊断
uv run python scripts/diagnose.py

# 清空数据（危险！）
export WRAPPER_MEILI_API_KEY=your_key
uv run python scripts/clear_all_data.py

# 检查合规
uv run python scripts/check_design_compliance.py

# 修复Markdown
uv run python scripts/fix_md040.py docs/
```

---

## Notes

- **Ruff Scope**: `pyproject.toml`中`exclude`包含`scripts/`，脚本不走lint检查
- **Pyright**: 脚本目录不在类型检查范围内
- **依赖**: 脚本使用项目根目录的`pyproject.toml`依赖
- **执行**: 所有脚本通过`uv run python`运行，无需激活虚拟环境
- **幂等性**: 迁移脚本应设计为可重复执行
- **日志**: 使用`logging`模块，级别INFO及以上

---

## Related

- 主服务: `wrapper/src/main.py`
- 配置: `wrapper/src/config.py`
- 根目录指南: `AGENTS.md`
