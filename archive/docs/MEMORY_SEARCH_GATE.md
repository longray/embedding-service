# Memory Search 上线门禁（分层版）

## 目标

- `should_find` 命中率：**>= 95%**
- `should_not_find` 误报率：**<= 1%**

## 运行方式

本地手动验证：

```bash
uv run python scripts/evaluate_memory_search.py \
  --topk 5 \
  --threshold 0.7 \
  --vector-threshold 0.75 \
  --hybrid-threshold 0.75 \
  --enforce-layered-gate \
  --save-report docs/memory-search-eval-report-gate-local.json
```text

CI 门禁测试（可选开启）：

```bash
ENABLE_SEARCH_GATE=true uv run pytest tests/test_memory_search_gate.py -v --tb=short
```

## 分层规则

### 模式层（必须全部通过）

- keyword: hit >= 96.5%, fp <= 0.8%
- vector: hit >= 94.0%, fp <= 1.2%
- hybrid: hit >= 96.0%, fp <= 1.0%

### 难度层（必须全部通过）

- easy: hit >= 98%
- medium: hit >= 95%
- hard: hit >= 90%
- hard_negative: fp <= 0.5%

### 全局加权层（必须通过）

- weighted hit >= 95%
- weighted fp <= 1%

权重：keyword=0.20, vector=0.25, hybrid=0.55。

## 当前建议参数

- keyword threshold: 0.0（占位）
- vector threshold: **0.75**
- hybrid threshold: **0.75**

## 说明

- 分层评估脚本默认通过 `metadata_filters={"eval_id": ...}` 隔离评估数据。
- 若服务库中历史数据噪声较大，建议继续强化业务侧 metadata 过滤（project/session）。
