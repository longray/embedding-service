#!/usr/bin/env python3
"""
OpenCode Agent 集成脚本 (后端 Python 版)

为 OpenCode Agent 提供自动化的设计治理支持

功能：
1. 自动解析 Agent 的提交信息，提取 Design-Ref
2. 自动更新 RTM 状态
3. 生成提交建议

Usage: python scripts/opencode_integration.py <action> [options]
Actions:
  suggest-commit    根据当前变更生成提交建议
  update-rtm        根据最近提交更新 RTM
  verify-design     验证实现是否符合设计
"""

import os
import sys
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Optional

# RTM 文件路径
RTM_FILE = Path(__file__).parent.parent / "docs" / "v3.2" / "RTM.md"
DESIGN_DOCS_DIR = Path(__file__).parent.parent / "docs" / "v3.2"


class Colors:
    """终端颜色"""

    INFO = "\033[36mℹ️\033[0m"
    SUCCESS = "\033[32m✅\033[0m"
    WARNING = "\033[33m⚠️\033[0m"
    ERROR = "\033[31m❌\033[0m"
    AGENT = "\033[35m🤖\033[0m"


def log(message: str, type_: str = "info"):
    """打印日志"""
    prefix = getattr(Colors, type_.upper(), Colors.INFO)
    print(f"{prefix} {message}")


def get_changed_files() -> List[str]:
    """获取变更的文件列表"""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )
        return [f for f in result.stdout.strip().split("\n") if f]
    except Exception as e:
        log(f"Failed to get changed files: {e}", "error")
        return []


def analyze_changes(files: List[str]) -> Dict:
    """分析变更类型"""
    analysis = {"type": "feat", "module": "", "description": "", "files": files}

    for file in files:
        # 判断变更类型
        if "test" in file.lower():
            analysis["type"] = "test"
        elif "docs" in file.lower() or file.endswith(".md"):
            analysis["type"] = "docs"
        elif "fix" in file.lower() or "bug" in file.lower():
            analysis["type"] = "fix"
        elif "refactor" in file.lower():
            analysis["type"] = "refactor"
        elif "perf" in file.lower():
            analysis["type"] = "perf"

        # 判断模块
        if "websocket" in file.lower():
            analysis["module"] = "websocket"
        elif "precompute" in file.lower():
            analysis["module"] = "precompute"
        elif "schema" in file.lower() or "migration" in file.lower():
            analysis["module"] = "schema"
        elif "meilisearch" in file.lower() or "meili" in file.lower():
            analysis["module"] = "meilisearch"
        elif "router" in file.lower():
            analysis["module"] = "api"
        elif "docker" in file.lower() or "k8s" in file.lower():
            analysis["module"] = "deployment"

    return analysis


def find_related_design_docs(analysis: Dict) -> List[str]:
    """查找相关设计文档"""
    refs = []
    module = analysis.get("module", "")

    module_to_doc = {
        "websocket": "BACKEND-v3.2-WEBSOCKET.md",
        "precompute": "BACKEND-v3.2-PRECOMPUTE.md",
        "schema": "DATABASE-v3.2-SCHEMA.md",
        "meilisearch": "BACKEND-v3.2-MEILISEARCH.md",
        "api": "PLUGIN-v3.2-API.md",
        "deployment": "DEPLOYMENT-v3.2.md",
    }

    if module in module_to_doc:
        refs.append(module_to_doc[module])

    return refs


def find_related_rtm_items(files: List[str]) -> List[str]:
    """根据变更文件查找相关的 RTM 条目"""
    items = []

    # 文件路径到 RTM ID 的映射
    file_to_rtm = {
        "websocket": ["WS-SRV-001", "WS-SRV-002", "WS-SRV-003", "WS-SRV-007"],
        "precompute": ["PC-SRV-001", "PC-SRV-002", "PC-SRV-003"],
        "schema": ["DB-SRV-001", "DB-SRV-002", "DB-SRV-003"],
        "meilisearch": ["API-SRV-004", "VER-SRV-003"],
        "router": ["API-SRV-001", "API-SRV-002", "API-SRV-003"],
        "docker": ["DEP-SRV-001", "DEP-SRV-002"],
        "k8s": ["DEP-SRV-003"],
    }

    for file in files:
        file_lower = file.lower()
        for key, rtm_ids in file_to_rtm.items():
            if key in file_lower:
                items.extend(rtm_ids)

    return list(set(items))  # 去重


def generate_commit_suggestion(analysis: Dict, refs: List[str], rtm_items: List[str]) -> str:
    """生成提交建议"""
    type_ = analysis["type"]
    module = analysis["module"] or "backend"
    files = analysis["files"]

    suggestion = f"""{type_}({module}): 简短描述（50字符内）

详细描述（可选，每行72字符内）:
- 说明变更原因
- 说明实现方式
- 说明影响范围
"""

    # 添加 Design-Ref
    if refs:
        suggestion += "\nDesign-Ref:\n"
        for ref in refs:
            suggestion += f"  - {ref}\n"

    # 添加 RTM 关联
    if rtm_items:
        suggestion += "\nRTM-Items:\n"
        for item in rtm_items[:5]:  # 最多显示5个
            suggestion += f"  - {item}\n"

    # 添加变更文件
    suggestion += "\n变更文件:\n"
    for file in files[:5]:  # 最多显示5个文件
        suggestion += f"  - {file}\n"

    return suggestion


def suggest_commit():
    """为 Agent 生成提交建议"""
    log("Analyzing changes for commit suggestion...", "agent")

    files = get_changed_files()
    if not files:
        log("No changes detected", "warning")
        return

    log(f"Changed files: {len(files)}", "info")

    analysis = analyze_changes(files)
    refs = find_related_design_docs(analysis)
    rtm_items = find_related_rtm_items(files)
    suggestion = generate_commit_suggestion(analysis, refs, rtm_items)

    print("\n" + "=" * 70)
    print("🤖 OpenCode Agent Commit Suggestion")
    print("=" * 70)
    print("\nSuggested commit message:")
    print("-" * 70)
    print(suggestion)
    print("-" * 70)
    print("\nTo use this suggestion:")
    print("  1. Review the Design-Ref links")
    print("  2. Adjust the description if needed")
    print('  3. Commit with: git commit -m "<message>"')
    print("=" * 70 + "\n")


def update_rtm():
    """更新 RTM"""
    log("Updating RTM...", "agent")

    if not RTM_FILE.exists():
        log(f"RTM file not found: {RTM_FILE}", "error")
        return

    # TODO: 实现根据提交自动更新 RTM 状态的逻辑
    log("RTM auto-update not yet fully implemented", "warning")
    log("Please manually update RTM.md after commit", "info")


def verify_design():
    """验证设计符合性"""
    log("Verifying design compliance...", "agent")

    if not RTM_FILE.exists():
        log(f"RTM file not found: {RTM_FILE}", "error")
        return

    # 读取 RTM 并统计
    content = RTM_FILE.read_text(encoding="utf-8")

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
    print("=" * 70 + "\n")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("Usage: python scripts/opencode_integration.py <action>")
        print("Actions:")
        print("  suggest-commit    根据当前变更生成提交建议")
        print("  update-rtm        根据最近提交更新 RTM")
        print("  verify-design     验证实现是否符合设计")
        sys.exit(1)

    action = sys.argv[1]

    if action == "suggest-commit":
        suggest_commit()
    elif action == "update-rtm":
        update_rtm()
    elif action == "verify-design":
        verify_design()
    else:
        log(f"Unknown action: {action}", "error")
        sys.exit(1)


if __name__ == "__main__":
    main()
