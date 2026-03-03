# 阶段④ 验证指南

本文档提供详细的验证步骤和问题处理指南。

## 前置条件

```bash
# 确认工具已安装
pip install pre-commit
uv --version  # 确认uv已安装
```

## 验证步骤

### 1. 安装Pre-commit Hooks

```bash
cd D:\embedding_service
pre-commit install --install-hooks
```

**预期输出**:
```
pre-commit installed at .git\hooks\pre-commit
```

### 2. 首次全量检查

```bash
pre-commit run --all-files
```

**预期结果**: 首次运行会发现问题，这是正常的。

### 3. 按优先级处理错误

#### P0级别（必须立即修复）

**Gitleaks - 密钥泄露**:
```bash
# 如果检测到密钥泄露
# 1. 立即从代码中移除密钥
# 2. 将密钥添加到.gitignore
# 3. 轮换泄露的密钥
```

**Bandit - 高危安全问题**:
```bash
# 查看详细报告
uv run bandit -r src/ -f txt

# 修复高危问题（B6xx系列）
# 常见问题：SQL注入、命令注入、硬编码密码
```

#### P1级别（阻断提交）

**Pyright - 类型错误**:
```bash
# 查看类型错误
uv run pyright src/

# 常见修复：
# - 添加类型注解
# - 修复类型不匹配
# - 处理Optional类型
```

**Ruff - 严重Lint错误**:
```bash
# 查看错误
uv run ruff check src/

# 自动修复（部分）
uv run ruff check src/ --fix
```

#### P2级别（自动修复）

**Ruff Format - 格式问题**:
```bash
# 自动格式化
uv run ruff format src/
```

**Ruff Check - 可修复问题**:
```bash
# 自动修复
uv run ruff check src/ --fix
```

#### P3级别（仅警告）

复杂度警告、命名建议等，不阻断提交。

## 迭代验证循环

### 目标
连续3个commit通过所有P0/P1检查。

### 流程

```bash
# 第1次迭代
pre-commit run --all-files
# 修复P0/P1错误
# 提交

# 第2次迭代
git add .
git commit -m "fix: resolve quality issues"
# Pre-commit自动运行
# 如果失败，继续修复

# 第3次迭代
git add .
git commit -m "chore: final quality fixes"
# 连续3次通过 = 稳定
```

### 成功标准

- ✅ Gitleaks: 0个密钥泄露
- ✅ Bandit: 0个高危问题
- ✅ Pyright: 0个类型错误
- ✅ Ruff: 0个P1错误
- ⚠️ 测试覆盖率: ≥70%（目标）

## 常见问题处理

### 问题1: 工具未安装

```bash
# 安装缺失工具
pip install ruff pyright pytest pytest-cov bandit
```

### 问题2: 大量格式错误

```bash
# 批量自动修复
uv run ruff format src/
uv run ruff check src/ --fix
```

### 问题3: 类型错误过多

```bash
# 渐进式修复
# 1. 先修复核心模块
# 2. 再修复辅助模块
# 3. 最后修复测试文件
```

### 问题4: Pre-commit太慢

```bash
# 跳过某些hook（临时）
SKIP=pyright git commit -m "..."

# 或配置超时
# 在.pre-commit-config.yaml中添加timeout
```

## 验证完成标志

当满足以下条件时，验证完成：

1. ✅ Pre-commit hooks安装成功
2. ✅ 连续3次commit通过P0/P1检查
3. ✅ 质量指标记录到metrics.log
4. ✅ 无阻断性错误

## 下一步

验证完成后，继续阶段⑤：持续治理配置。
