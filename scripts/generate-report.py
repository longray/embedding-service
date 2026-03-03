#!/usr/bin/env python3
"""生成质量报告

用途：分析metrics.log并生成可读报告
运行：python scripts/generate-report.py
"""

import json
from datetime import datetime
from pathlib import Path


def load_metrics() -> list[dict]:
    """加载所有指标"""
    log_file = Path(".quality-state/metrics.log")
    if not log_file.exists():
        return []

    metrics = []
    for line in log_file.read_text().splitlines():
        if line.strip():
            try:
                metrics.append(json.loads(line))
            except:
                pass
    return metrics


def generate_report(metrics: list[dict]) -> str:
    """生成Markdown报告"""
    if not metrics:
        return "# 质量报告\n\n暂无数据。\n"

    latest = metrics[-1]
    report = [
        "# 代码质量报告",
        f"\n**生成时间**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"**最新Commit**: {latest.get('commit', 'N/A')}",
        f"**数据点数**: {len(metrics)}",
        "\n## 最新指标\n",
        "| 指标 | 数值 | 目标 | 状态 |",
        "|------|------|------|------|",
    ]

    m = latest.get("metrics", {})

    # Ruff errors
    ruff_err = m.get("ruff_errors", -1)
    report.append(f"| Ruff错误 | {ruff_err} | 0 | {'✅' if ruff_err == 0 else '❌'} |")

    # Pyright errors
    py_err = m.get("pyright_errors", -1)
    report.append(f"| Pyright错误 | {py_err} | 0 | {'✅' if py_err == 0 else '❌'} |")

    # Coverage
    cov = m.get("test_coverage", 0)
    report.append(f"| 测试覆盖率 | {cov:.1f}% | ≥70% | {'✅' if cov >= 70 else '⚠️'} |")

    # Trend
    if len(metrics) >= 2:
        report.append("\n## 趋势分析\n")
        prev = metrics[-2].get("metrics", {})

        ruff_delta = ruff_err - prev.get("ruff_errors", ruff_err)
        py_delta = py_err - prev.get("pyright_errors", py_err)
        cov_delta = cov - prev.get("test_coverage", cov)

        report.append(f"- Ruff错误: {ruff_delta:+d}")
        report.append(f"- Pyright错误: {py_delta:+d}")
        report.append(f"- 覆盖率: {cov_delta:+.1f}%")

    return "\n".join(report)


if __name__ == "__main__":
    metrics = load_metrics()
    report = generate_report(metrics)

    output_file = Path("quality-report.md")
    output_file.write_text(report, encoding="utf-8")

    print(f"✅ Report generated: {output_file}")
    print("\n" + report)
