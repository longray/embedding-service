# embedding_service 项目质量标准 - 开发工作流

> **项目**: embedding_service (Python 3.10-3.12 + FastAPI)  
> **生成时间**: 2026-03-03  
> **基于标准**: code-quality-standard v1.2.0

---

## 1. Pre-commit 工作流

# Src: docs/07-PRECOMMIT-STANDARDS.md#7.2

### 1.1 Hook 执行顺序

```bash
提交前自动执行（git commit）:
1. Gitleaks（密钥检测）      ← P0: 阻止密钥泄露
2. Bandit（安全扫描）        ← P0: 阻止安全问题
3. Ruff Format（格式化）     ← P2: 自动修复格式
4. Ruff Check（Lint）        ← P2: 自动修复问题
5. Pyright（类型检查）       ← P1: 阻止类型错误

推送前执行（git push）:
6. Pytest（测试）            ← P1: 推送时执行

```

### 1.2 Pre-commit 配置

```yaml
# .pre-commit-config.yaml
default_stages: [pre-commit]
fail_fast: false

exclude: '^\.venv/|^dist/|^build/|^\.ruff_cache/'

repos:
  # P0: 安全扫描
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.21.2
    hooks:
      - id: gitleaks
        name: Gitleaks Secret Detection

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.10
    hooks:
      - id: bandit
        name: Bandit Python Security Check
        args: ['-c', 'pyproject.toml']
        additional_dependencies: ['bandit[toml]']

  # P2: Python 格式化和检查
  - repo: local
    hooks:
      - id: ruff-format
        name: Ruff Format (Python)
        entry: uv run ruff format src/
        language: system
        types: [python]
        pass_filenames: false
        require_serial: true

      - id: ruff-check
        name: Ruff Check (Python)
        entry: uv run ruff check src/ --fix
        language: system
        types: [python]
        pass_filenames: false
        require_serial: true

      # P1: 类型检查
      - id: pyright
        name: Pyright Type Check
        entry: uv run pyright src/
        language: system
        types: [python]
        pass_filenames: false
        require_serial: true

      # P1: 测试（pre-push）
      - id: pytest
        name: Pytest (Python)
        entry: uv run pytest src/tests/ -v --tb=short
        language: system
        types: [python]
        pass_filenames: false
        stages: [pre-push]
        require_serial: true

```text

### 1.3 常用命令

```bash
# 安装 hooks
pre-commit install --install-hooks

# 运行所有 hooks
pre-commit run --all-files

# 运行特定 hook
pre-commit run ruff-format
pre-commit run gitleaks

# 跳过某个 hook（紧急情况）
pre-commit run --skip gitleaks

# 更新 hooks
pre-commit autoupdate

```

---

## 2. 开发流程

# Src: docs/10-DEVELOPMENT-WORKFLOW.md#10.2

### 2.1 功能开发流程

```text
1. 需求分析 → 2. 技术设计 → 3. 开发实现 → 4. 代码审查 → 5. 测试验证 → 6. 合并主分支

```

**各阶段要点**:

**1. 需求分析**:

- 需求文档
- 技术评估
- 工时评估

**2. 技术设计**:

- 架构设计
- API 设计
- 测试策略

**3. 开发实现**:

- 代码编写
- 本地测试
- Pre-commit 检查通过

**4. 代码审查**:

- 功能完整性
- 代码质量
- 安全性审查

**5. 测试验证**:

- 单元测试 ≥70%
- 集成测试
- E2E 测试（如需要）

**6. 合并主分支**:

- 所有检查通过
- 代码审查通过
- 测试覆盖率达标

### 2.2 Bug 修复流程

```text
1. 问题确认 → 2. 根因分析 → 3. 修复实现 → 4. 验证测试 → 5. 代码审查 → 6. 合并发布

```

**关键要求**:

- 添加回归测试
- 更新相关文档
- 验证修复不影响其他功能

---

## 3. 代码审查规范

# Src: docs/10-DEVELOPMENT-WORKFLOW.md#10.5

### 3.1 审查策略（小型团队）

**本项目定位**: 2-3人团队

**审查要求**:

- 至少 1 人 Approve（可自我审查后他人确认）
- PR 创建后 24 小时内完成初步审查
- 提供建设性反馈

**例外情况**:

- 紧急修复：安全热修复可简化流程，需事后补充审查
- 文档/配置修改：仅修改文档或配置可简化审查

### 3.2 审查检查清单

| 检查项 | 说明 | 优先级 |
|--------|------|--------|
| **功能完整性** | 是否实现所有需求 | 高 |
| **代码质量** | 是否符合规范（Lint 通过） | 高 |
| **安全性** | 是否存在安全问题 | 高 |
| **可维护性** | 代码是否清晰易懂 | 中 |
| **测试覆盖** | 是否添加了测试 | 高 |
| **文档更新** | 是否更新了相关文档 | 中 |

### 3.3 PR 模板

```markdown
## 变更描述

### 变更类型
- [ ] 新功能 (feat)
- [ ] Bug 修复 (fix)
- [ ] 重构 (refactor)
- [ ] 文档更新 (docs)

### 变更内容
<!-- 简要描述变更内容 -->

### 代码质量
- [ ] 所有 Lint 检查通过
- [ ] 类型检查通过
- [ ] 测试覆盖率 ≥70%
- [ ] 通过 Code Review

### 测试
- [ ] 单元测试通过
- [ ] 集成测试通过

### 相关 Issue
- #123: 相关 Issue 链接

```text

---

## 4. 分支策略

### 4.1 分支命名规范

| 分支类型 | 命名格式 | 示例 |
|---------|---------|------|
| **功能分支** | `feature/功能名称` | `feature/embedding-cache` |
| **修复分支** | `fix/问题描述` | `fix/memory-leak` |
| **重构分支** | `refactor/模块名称` | `refactor/llm-service` |

### 4.2 分支保护规则

| 规则 | 说明 |
|------|------|
| **主分支保护** | main 分支需要 PR 才能合并 |
| **检查通过要求** | Lint, 类型检查, 测试全部通过 |
| **代码审查要求** | 至少 1 个 Approve |
| **测试覆盖率要求** | 覆盖率 ≥70% |

---

## 5. Commit Message 规范

### 5.1 格式要求

```

<type>(<scope>): <subject>

<body>

<footer>

```text

### 5.2 Type 类型

| Type | 说明 | 示例 |
|------|------|------|
| **feat** | 新功能 | `feat(api): add embedding batch endpoint` |
| **fix** | Bug 修复 | `fix(cache): resolve memory leak issue` |
| **refactor** | 重构 | `refactor(llm): simplify model loading` |
| **docs** | 文档更新 | `docs(readme): update installation guide` |
| **test** | 测试相关 | `test(api): add integration tests` |
| **chore** | 构建/工具 | `chore(deps): update dependencies` |

---

## 6. 实施检查清单

### 6.1 新功能开发

- [ ] 创建功能分支 `feature/xxx`
- [ ] 实现功能代码（通过所有检查）
- [ ] 编写单元测试（覆盖率 ≥70%）
- [ ] 进行代码自检（本地运行所有检查）
- [ ] 创建 PR 并填写 PR 模板
- [ ] 等待 Code Review 审查通过
- [ ] 确保所有 CI 检查通过

### 6.2 Bug 修复

- [ ] 创建修复分支 `fix/xxx`
- [ ] 定位根因并修复代码
- [ ] 添加回归测试
- [ ] 验证修复（本地测试 + 回归测试）
- [ ] 创建 PR 并填写 PR 模板
- [ ] 等待 Code Review 审查通过
- [ ] 确保所有 CI 检查通过

### 6.3 日常提交

- [ ] 提交前运行 `pre-commit run`
- [ ] 查看所有 hooks 是否通过
- [ ] 遵循 commit message 规范（Conventional Commits）
  - 格式: `<type>(<scope>): <description>`
  - 类型: `feat/fix/docs/style/refactor/test/chore`
  - 示例: `feat(embedding): add batch size optimization`

  - # Src: docs/10-DEVELOPMENT-WORKFLOW.md#commit-message-format

- [ ] 及时处理 CI/CD 失败
- [ ] 保持代码覆盖率不下降

---

**文档版本**: 1.0.0  
**最后更新**: 2026-03-03  
**规范来源**: D:\github\code-quality-standard
