# embedding_service 项目质量标准 - 量化门禁

> **项目**: embedding_service (Python 3.10-3.12 + FastAPI)  
> **生成时间**: 2026-03-03  
> **基于标准**: code-quality-standard v1.2.0

---

## 1. 量化门禁阈值（全部可自动化检测）

# Src: docs/03-LINTING-STANDARDS.md#3.2.3

### 1.1 复杂度控制

| 指标 | 阈值 | 检测工具 | 优先级 | 处理方式 |
|------|------|---------|--------|----------|
| **McCabe 复杂度** | ≤15 | Ruff (C901) | P3 | 警告，建议简化函数逻辑 |
| **函数长度** | ≤50行 | Ruff (PLR0915) | P3 | 警告，建议拆分大函数 |
| **函数参数** | ≤5个 | Ruff (PLR0912) | P3 | 警告，建议使用参数对象 |
| **嵌套深度** | ≤5层 | 人工审查 | P3 | 警告，建议减少嵌套 |

**Ruff 配置**:
```toml
[tool.ruff.lint]
select = ["C901", "PLR0912", "PLR0913", "PLR0915"]
```

### 1.2 代码质量指标

| 指标 | 阈值 | 检测工具 | 优先级 | 规范来源 |
|------|------|---------|--------|----------|
| **未使用导入** | =0 | Ruff (F401) | P1 | docs/03-LINTING-STANDARDS.md#3.2.2 |
| **未定义变量** | =0 | Ruff (F821) | P1 | docs/03-LINTING-STANDARDS.md#3.2.2 |
| **未使用变量** | =0 | Ruff (F841) | P2 | docs/03-LINTING-STANDARDS.md#3.2.2 |
| **导入顺序** | 正确 | Ruff (I001) | P2 | docs/03-LINTING-STANDARDS.md#3.2.2 |

### 1.3 安全扫描指标

# Src: docs/05-SECURITY-SCANNING.md#5.2

| 指标 | 阈值 | 检测工具 | 优先级 | 规范来源 |
|------|------|---------|--------|----------|
| **密钥泄露** | =0 | Gitleaks v8.21.2 | P0 | docs/05-SECURITY-SCANNING.md#5.2 |
| **高危漏洞** | =0 | Bandit (B1xx系列) | P0 | docs/05-SECURITY-SCANNING.md#5.3 |
| **中危漏洞** | ≤5 | Bandit (B3xx系列) | P1 | docs/05-SECURITY-SCANNING.md#5.3 |
| **低危问题** | 警告 | Bandit (B6xx系列) | P3 | docs/05-SECURITY-SCANNING.md#5.3 |

### 1.4 测试覆盖率指标

# Src: docs/06-TESTING-STANDARDS.md#6.2.7

| 模块类型 | 覆盖率要求 | 检测工具 | 优先级 | 规范来源 |
|---------|-----------|---------|--------|----------|
| **业务应用** | ≥70% | pytest-cov | P4 | docs/06-TESTING-STANDARDS.md#6.2.7 |
| **核心库** | ≥95% | pytest-cov | P4 | docs/06-TESTING-STANDARDS.md#6.2.7 |
| **公共API** | 100% | pytest-cov | P4 | docs/06-TESTING-STANDARDS.md#6.2.7 |

**本项目目标**:
- 初期目标：≥70%（业务应用标准）
- 中期目标：≥80%
- 长期目标：≥90%

---

## 2. 优先级决策树

### 2.1 错误处理流程

```
收到 lint/安全/测试 错误
    ↓
解析 priority 字段
    ↓
┌─────────────────────────────────────┐
│ P0/P1 🔴 (阻断提交)                  │
│ - 阻断提交 + 必须修复                │
│ - 自动修复失败 → 生成 fix.patch     │
│ - 人工介入                          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ P2 🟡 (自动修复)                     │
│ - 尝试 --fix 自动修复                │
│ - 修复后重新提交                     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ P3-P5 🔵 (仅警告)                    │
│ - 记录警告到 metrics.log             │
│ - 不阻断提交                         │
│ - Code Review 时讨论                 │
└─────────────────────────────────────┘
```

### 2.2 规则优先级映射

| 规则类型 | 示例 | 优先级 | 处理方式 |
|---------|------|--------|----------|
| **密钥泄露** | Gitleaks 检测到 API key | P0 | 阻断 + 立即修复 |
| **SQL注入** | Bandit B608 | P0 | 阻断 + 立即修复 |
| **类型错误** | Pyright 类型不匹配 | P1 | 阻断 + 必须修复 |
| **未定义变量** | Ruff F821 | P1 | 阻断 + 必须修复 |
| **未使用导入** | Ruff F401 | P2 | 自动修复 |
| **格式问题** | Ruff Format | P2 | 自动修复 |
| **复杂度警告** | McCabe > 15 | P3 | 仅警告 |
| **命名建议** | 变量名不规范 | P3 | 仅警告 |

---

## 3. 性能指标

# Src: docs/01-GLOBAL-PRINCIPLES.md#1.2.5

### 3.1 检查性能目标

| 项目规模 | 文件数量 | 增量检查 | 全量检查 | 缓存命中 |
|---------|---------|---------|---------|---------|
| **本项目** | 21个 | < 3s | < 10s | < 1s |

**当前状态**:
- 文件数量：21个Python文件
- 代码行数：约2,764行
- 预期性能：小型项目标准

### 3.2 性能优化策略

**增量检查**:
- Pre-commit `pass_filenames: true`
- 只检查变更的文件

**缓存策略**:
- Ruff 缓存：`.ruff_cache/`
- 自动缓存，无需配置

**并行执行**:
- CI/CD job 并行化
- 独立任务并行运行

---

## 4. 门禁配置示例

### 4.1 Pre-commit 门禁

```yaml
# .pre-commit-config.yaml
repos:
  # P0: 密钥泄露检测
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.21.2
    hooks:
      - id: gitleaks

  # P0-P1: 安全扫描
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.10
    hooks:
      - id: bandit
        args: ['-c', 'pyproject.toml']

  # P2: 格式化（自动修复）
  - repo: local
    hooks:
      - id: ruff-format
        entry: uv run ruff format src/

  # P1-P2: Lint检查（自动修复）
  - repo: local
    hooks:
      - id: ruff-check
        entry: uv run ruff check src/ --fix

  # P1: 类型检查
  - repo: local
    hooks:
      - id: pyright
        entry: uv run pyright src/
```

### 4.2 CI/CD 门禁

```yaml
# .github/workflows/quality-check.yml
jobs:
  quality-gate:
    steps:
      - name: Security Scan (P0)
        run: |
          gitleaks detect --no-git
          uv run bandit -c pyproject.toml -r src/
      
      - name: Type Check (P1)
        run: uv run pyright src/
      
      - name: Lint Check (P1-P2)
        run: uv run ruff check src/
      
      - name: Test Coverage (P4)
        run: |
          uv run pytest --cov=src --cov-report=term --cov-fail-under=70
```

---

## 5. 监控与报告

### 5.1 质量指标追踪

**记录位置**: `.quality-state/metrics.log`

**记录内容**:
```json
{
  "timestamp": "2026-03-03T14:00:00Z",
  "commit": "abc123",
  "metrics": {
    "ruff_errors": 0,
    "ruff_warnings": 3,
    "pyright_errors": 0,
    "bandit_high": 0,
    "bandit_medium": 0,
    "test_coverage": 72.5
  }
}
```

### 5.2 趋势分析

**每周报告**:
- 错误数趋势
- 覆盖率趋势
- 复杂度趋势
- 高频违规TOP3

---

## 6. 实施检查清单

### 6.1 门禁配置
- [ ] 配置 P0 级别门禁（Gitleaks, Bandit高危）
- [ ] 配置 P1 级别门禁（Pyright, Ruff严重错误）
- [ ] 配置 P2 级别自动修复（Ruff Format, Ruff --fix）
- [ ] 配置 P3 级别警告（复杂度、命名）

### 6.2 阈值验证
- [ ] 验证复杂度阈值：McCabe ≤15
- [ ] 验证函数长度：≤50行
- [ ] 验证参数数量：≤5个
- [ ] 验证测试覆盖率：≥70%

### 6.3 监控配置
- [ ] 创建 .quality-state/ 目录
- [ ] 配置 metrics.log 记录
- [ ] 配置每周质量报告
- [ ] 配置趋势分析脚本

---

**文档版本**: 1.0.0  
**最后更新**: 2026-03-03  
**规范来源**: D:\github\code-quality-standard

# === QUALITY-SPEC-REF ===
# Repo: longray/code-quality-standard
# Files: 
#   - docs/03-LINTING-STANDARDS.md (复杂度阈值)
#   - docs/05-SECURITY-SCANNING.md (安全扫描)
#   - docs/06-TESTING-STANDARDS.md (测试覆盖率)
#   - docs/01-GLOBAL-PRINCIPLES.md (性能指标)
# Priority: P0-P5 as marked
# === /QUALITY-SPEC-REF ===
