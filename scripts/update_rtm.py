#!/usr/bin/env python3
"""
RTM 自动更新脚本 (后端 Python 版)

根据 git 提交自动更新 RTM 状态

Usage: python scripts/update_rtm.py [options]
Options:
  --commit <hash>    指定提交哈希
  --dry-run          预览变更，不实际修改
  --check            检查 RTM 状态
"""

import os
import sys
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Optional
import argparse
from datetime import datetime

RTM_FILE = Path(__file__).parent.parent / "docs" / "v3.2" / "RTM.md"


class RTMUpdater:
    def __init__(self, options: Optional[Dict] = None):
        self.options = options or {}
        self.rtm_content = ""
        self.changes = []

    def log(self, message: str, type_: str = "info"):
        """打印日志"""
        prefix = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}.get(type_, "ℹ️")
        print(f"{prefix} {message}")

    def load_rtm(self):
        """加载 RTM 文件"""
        if not RTM_FILE.exists():
            raise FileNotFoundError(f"RTM file not found: {RTM_FILE}")
        self.rtm_content = RTM_FILE.read_text(encoding="utf-8")

    def save_rtm(self):
        """保存 RTM 文件"""
        if self.options.get("dry_run"):
            self.log("Dry run mode - not saving changes", "warning")
            return
        RTM_FILE.write_text(self.rtm_content, encoding="utf-8")
        self.log("RTM updated successfully", "success")

    def get_recent_commits(self, count: int = 10) -> List[Dict]:
        """获取最近的提交"""
        try:
            result = subprocess.run(
                ["git", "log", f"--pretty=format:%H|%s|%b", f"-{count}"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
            )

            commits = []
            for line in result.stdout.strip().split("\n"):
                if "|" in line:
                    parts = line.split("|")
                    commits.append(
                        {"hash": parts[0], "subject": parts[1], "body": "|".join(parts[2:]) if len(parts) > 2 else ""}
                    )
            return commits
        except Exception as e:
            self.log(f"Failed to get commits: {e}", "error")
            return []

    def extract_design_refs(self, message: str) -> List[str]:
        """从提交信息中提取 Design-Ref"""
        refs = []
        pattern = r"Design-Ref:\s*([^\n]+)"
        for match in re.finditer(pattern, message):
            refs.append(match.group(1).strip())
        return refs

    def extract_rtm_items(self, message: str) -> List[str]:
        """从提交信息中提取 RTM-Items"""
        items = []
        pattern = r"RTM-Items:\s*([^\n]+)"
        for match in re.finditer(pattern, message):
            items.append(match.group(1).strip())
        return items

    def check_rtm(self):
        """检查 RTM 状态"""
        self.log("Checking RTM status...", "info")

        if not RTM_FILE.exists():
            self.log("RTM.md not found", "error")
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

        print("\n" + "=" * 70)
        print("📊 RTM Status Summary")
        print("=" * 70)
        print(f"Total Items: {total}")
        print(f"  ⏳ Pending: {pending}")
        print(f"  🔄 In Progress: {in_progress}")
        print(f"  ⚠️ Warning: {warning}")
        print(f"  ✅ Completed: {completed}")
        print(f"  ❌ Cancelled: {cancelled}")
        print(f"Completion Rate: {completion_rate:.1f}%")

        if completion_rate >= 80:
            print("Status: ✅ ON TRACK")
        elif completion_rate >= 50:
            print("Status: 🟡 IN PROGRESS")
        else:
            print("Status: 🔴 BEHIND")

        print("=" * 70 + "\n")

    def update_from_commits(self):
        """根据提交更新 RTM"""
        self.log("Updating RTM from recent commits...", "info")

        commits = self.get_recent_commits(5)
        if not commits:
            self.log("No commits found", "warning")
            return

        self.load_rtm()

        updated = False
        for commit in commits:
            rtm_items = self.extract_rtm_items(commit["body"])
            if rtm_items:
                self.log(f"Found RTM items in commit {commit['hash'][:8]}: {rtm_items}", "info")
                # TODO: 实现自动更新 RTM 状态的逻辑
                updated = True

        if updated:
            self.save_rtm()
        else:
            self.log("No RTM items found in recent commits", "info")


def main():
    parser = argparse.ArgumentParser(description="RTM Updater - Backend")
    parser.add_argument("--commit", help="指定提交哈希")
    parser.add_argument("--dry-run", action="store_true", help="预览变更")
    parser.add_argument("--check", action="store_true", help="检查 RTM 状态")
    parser.add_argument("--all", action="store_true", help="更新所有")

    args = parser.parse_args()

    updater = RTMUpdater(vars(args))

    if args.check:
        updater.check_rtm()
    elif args.all:
        updater.update_from_commits()
    else:
        # 默认运行 RTM 检查
        updater.check_rtm()


if __name__ == "__main__":
    main()
