#!/usr/bin/env python3
"""质量指标监控脚本

用途：收集和记录代码质量指标
运行：python scripts/collect-metrics.py
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_command(cmd: list[str]) -> tuple[int, str]:
    """运行命令并返回退出码和输出"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode, result.stdout + result.stderr
    except Exception as e:
        return 1, str(e)


def collect_metrics() -> dict:
    """收集质量指标"""
    metrics = {"timestamp": datetime.utcnow().isoformat() + "Z", "commit": "", "metrics": {}}

    # Git commit hash
    code, output = run_command(["git", "rev-parse", "--short", "HEAD"])
    if code == 0:
        metrics["commit"] = output.strip()

    # Ruff errors
    code, output = run_command(["uv", "run", "ruff", "check", "src/", "--output-format=json"])
    if code == 0:
        try:
            ruff_data = json.loads(output) if output else []
            metrics["metrics"]["ruff_errors"] = len([x for x in ruff_data if x.get("type") == "error"])
            metrics["metrics"]["ruff_warnings"] = len([x for x in ruff_data if x.get("type") == "warning"])
        except:
            metrics["metrics"]["ruff_errors"] = -1

    # Pyright errors
    code, output = run_command(["uv", "run", "pyright", "src/", "--outputjson"])
    if code == 0:
        try:
            pyright_data = json.loads(output)
            metrics["metrics"]["pyright_errors"] = pyright_data.get("summary", {}).get("errorCount", -1)
        except:
            metrics["metrics"]["pyright_errors"] = -1

    # Test coverage
    code, output = run_command(["uv", "run", "pytest", "--cov=src", "--cov-report=json"])
    if code == 0:
        try:
            cov_file = Path(".coverage.json")
            if cov_file.exists():
                cov_data = json.loads(cov_file.read_text())
                metrics["metrics"]["test_coverage"] = cov_data.get("totals", {}).get("percent_covered", 0)
        except:
            metrics["metrics"]["test_coverage"] = 0

    return metrics


def save_metrics(metrics: dict):
    """保存指标到日志"""
    log_file = Path(".quality-state/metrics.log")
    log_file.parent.mkdir(exist_ok=True)

    with log_file.open("a") as f:
        f.write(json.dumps(metrics) + "\n")

    print(f"✅ Metrics saved: {metrics['commit']}")
    print(f"   Ruff errors: {metrics['metrics'].get('ruff_errors', 'N/A')}")
    print(f"   Pyright errors: {metrics['metrics'].get('pyright_errors', 'N/A')}")
    print(f"   Coverage: {metrics['metrics'].get('test_coverage', 'N/A')}%")


if __name__ == "__main__":
    metrics = collect_metrics()
    save_metrics(metrics)
