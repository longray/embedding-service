import os
import subprocess
import sys

import pytest


@pytest.mark.integration
@pytest.mark.slow
def test_memory_search_accuracy_gate() -> None:
    if os.getenv("ENABLE_SEARCH_GATE", "false").lower() != "true":
        pytest.skip("ENABLE_SEARCH_GATE 未开启，跳过搜索门禁测试")

    cmd = [
        sys.executable,
        "scripts/evaluate_memory_search.py",
        "--topk",
        "5",
        "--threshold",
        "0.7",
        "--vector-threshold",
        "0.75",
        "--hybrid-threshold",
        "0.75",
        "--enforce-layered-gate",
        "--save-report",
        "docs/memory-search-eval-report-gate.json",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AssertionError(f"memory search 回归门禁失败\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
