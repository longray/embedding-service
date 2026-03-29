# Python Markdown Lint 工具调研

## 推荐方案: taskipy

与 uv 完美兼容，配置简单。

```toml
[tool.taskipy.tasks]
lint-md = "uvx pre-commit run markdownlint-cli2 --all-files"
lint-md-stats = "uv run python scripts/assess_md040.py"
```

使用: `uv run task lint-md`
