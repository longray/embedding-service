#!/usr/bin/env python3
"""
设计符合性检查脚本 (后端 Python 版)

检查代码实现是否符合设计文档

Usage: python scripts/check_design_compliance.py [options]
Options:
  --rtm          检查RTM更新状态
  --api          检查API实现完整性
  --coverage     检查代码覆盖率
  --all          运行所有检查
"""

import os
import sys
import subprocess
import re
from pathlib import Path
import argparse
from typing import List, Dict

DESIGN_DOCS_DIR = Path(__file__).parent.parent / "docs" / "v3.2"
RTM_FILE = DESIGN_DOCS_DIR / "RTM.md"


class DesignComplianceChecker:
    def __init__(self):
        self.results = []
        self.exit_code = 0

    def log(self, message: str, type_: str = "info"):
        """打印日志"""
        prefix = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}.get(type_, "ℹ️")
        print(f"{prefix} {message}")

    def check_rtm(self):
        """检查 RTM 状态"""
        self.log("Checking RTM status...", "info")

        if not RTM_FILE.exists():
            self.log("RTM.md not found", "error")
            self.exit_code = 1
            return

        content = RTM_FILE.read_text(encoding="utf-8")

        # 统计状态
        pending = len(re.findall(r"⏳", content))
        in_progress = len(re.findall(r"🔄", content))
        warning = len(re.findall(r"⚠️", content))
        completed = len(re.findall(r"✅", content))
        cancelled = len(re.findall(r"❌", content))

        total = pending + in_progress + warning + completed + cancelled
        completion_rate = (completed / total * 100) if total > 0 else 0

        self.log(f"RTM Status: {completed}/{total} completed ({completion_rate:.1f}%)", "info")
        self.log(f"  - Pending: {pending}", "info")
        self.log(f"  - In Progress: {in_progress}", "info")
        self.log(f"  - Warning: {warning}", "info")
        self.log(f"  - Completed: {completed}", "success")
        self.log(f"  - Cancelled: {cancelled}", "info")

        if pending > 0:
            self.log(f"{pending} items still pending", "warning")

        self.results.append(
            {
                "check": "RTM",
                "status": "PASS" if pending == 0 else "WARNING",
                "details": f"{completed}/{total} completed",
            }
        )

    def check_api_implementation(self):
        """检查 API 实现"""
        self.log("Checking API implementation...", "info")

        wrapper_src = Path(__file__).parent.parent / "wrapper" / "src" / "routers"
        if not wrapper_src.exists():
            self.log("Wrapper src not found", "warning")
            return

        # 检查主要路由文件
        expected_routers = ["memories.py", "code.py", "websocket.py", "sync.py", "health.py"]

        found = []
        missing = []

        for router in expected_routers:
            router_path = wrapper_src / router
            if router_path.exists():
                found.append(router)
            else:
                missing.append(router)

        self.log(f"Found {len(found)}/{len(expected_routers)} routers", "info")

        if found:
            self.log(f"  Found: {', '.join(found)}", "success")

        if missing:
            self.log(f"  Missing: {', '.join(missing)}", "warning")
            self.exit_code = 1

        self.results.append(
            {
                "check": "API Routers",
                "status": "PASS" if not missing else "FAIL",
                "details": f"{len(found)}/{len(expected_routers)} implemented",
            }
        )

    def check_code_coverage(self):
        """检查代码覆盖率"""
        self.log("Checking code coverage...", "info")

        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "--cov=wrapper.src", "--cov-report=term-missing"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
            )

            # 解析覆盖率输出
            coverage_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", result.stdout)
            if coverage_match:
                coverage = int(coverage_match.group(1))
                self.log(f"Code coverage: {coverage}%", "info")

                if coverage >= 85:
                    self.log("Coverage meets target (≥85%)", "success")
                    status = "PASS"
                else:
                    self.log(f"Coverage below target ({coverage}% < 85%)", "warning")
                    status = "WARNING"

                self.results.append({"check": "Coverage", "status": status, "details": f"{coverage}% coverage"})
            else:
                self.log("Could not parse coverage report", "warning")

        except Exception as e:
            self.log(f"Failed to check coverage: {e}", "error")

    def run_all(self):
        """运行所有检查"""
        self.check_rtm()
        self.check_api_implementation()
        self.check_code_coverage()

        print("\n" + "=" * 70)
        print("📋 Design Compliance Check Summary")
        print("=" * 70)

        for result in self.results:
            status_icon = "✅" if result["status"] == "PASS" else "⚠️" if result["status"] == "WARNING" else "❌"
            print(f"{status_icon} {result['check']}: {result['status']}")
            print(f"   Details: {result['details']}")

        print("=" * 70)

        if self.exit_code == 0:
            self.log("All checks passed!", "success")
        else:
            self.log("Some checks failed", "error")

        return self.exit_code


def main():
    parser = argparse.ArgumentParser(description="Design Compliance Checker - Backend")
    parser.add_argument("--rtm", action="store_true", help="检查RTM状态")
    parser.add_argument("--api", action="store_true", help="检查API实现")
    parser.add_argument("--coverage", action="store_true", help="检查代码覆盖率")
    parser.add_argument("--all", action="store_true", help="运行所有检查")

    args = parser.parse_args()

    checker = DesignComplianceChecker()

    if args.all:
        sys.exit(checker.run_all())
    elif args.rtm:
        checker.check_rtm()
    elif args.api:
        checker.check_api_implementation()
    elif args.coverage:
        checker.check_code_coverage()
    else:
        # 默认运行 RTM 检查
        checker.check_rtm()


if __name__ == "__main__":
    main()
