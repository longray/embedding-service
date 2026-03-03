#!/usr/bin/env python3
"""每月规范同步脚本

用途：对比本地 quality-standards/ 与上游 code-quality-standard/docs/ 的差异
     如有更新，输出变更摘要并生成同步建议（不自动修改规则文件）

# === QUALITY-SPEC-REF ===
# Repo: longray/code-quality-standard
# File: docs/07-PRECOMMIT-STANDARDS.md
# Section: #standards-sync
# Priority: P5
# === /QUALITY-SPEC-REF ===

运行：uv run python scripts/sync-standards.py
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path


# === 配置 ===
STANDARDS_REPO = Path(r"D:\github\code-quality-standard")
PROJECT_ROOT = Path(__file__).parent.parent
QUALITY_STANDARDS_DIR = PROJECT_ROOT / "quality-standards"
STATE_DIR = PROJECT_ROOT / ".quality-state"
CHECKPOINT_FILE = STATE_DIR / "checkpoint.json"

# docs/ 与 quality-standards/ 的映射关系
SPEC_MAPPING = {
    "docs/01-GLOBAL-PRINCIPLES.md": "quality-standards/01-structure.md",
    "docs/02-FORMATTING-STANDARDS.md": "quality-standards/02-expression.md",
    "docs/03-LINTING-STANDARDS.md": "quality-standards/02-expression.md",
    "docs/05-SECURITY-SCANNING.md": "quality-standards/03-gates.md",
    "docs/06-TESTING-STANDARDS.md": "quality-standards/03-gates.md",
    "docs/07-PRECOMMIT-STANDARDS.md": "quality-standards/04-workflow.md",
    "docs/10-DEVELOPMENT-WORKFLOW.md": "quality-standards/04-workflow.md",
    "docs/11-TOOLCHAIN-TEMPLATES.md": "quality-standards/04-workflow.md",
}


def get_git_log(repo_path: Path, file: str) -> str:
    """获取文件最后一次commit信息"""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H|%ai|%s", "--", file],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def get_file_hash(repo_path: Path, file: str) -> str:
    """获取文件当前hash（用于变更检测）"""
    try:
        result = subprocess.run(
            ["git", "hash-object", file],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()[:8]
    except Exception:  # noqa: BLE001
        return "unknown"


def load_last_sync_state() -> dict:
    """加载上次同步状态"""
    if not CHECKPOINT_FILE.exists():
        return {}
    try:
        data = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        return data.get("last_sync", {})
    except Exception:  # noqa: BLE001
        return {}


def save_sync_state(hashes: dict) -> None:
    """保存当前同步状态"""
    checkpoint = {}
    if CHECKPOINT_FILE.exists():
        try:
            checkpoint = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass

    checkpoint["last_sync"] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "file_hashes": hashes,
    }
    STATE_DIR.mkdir(exist_ok=True)
    CHECKPOINT_FILE.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")


def check_standards_available() -> bool:
    """检查标准仓库是否可访问"""
    if not STANDARDS_REPO.exists():
        print(f"⚠️  标准仓库不可访问: {STANDARDS_REPO}")
        print("   请确保 D:\\github\\code-quality-standard 存在")
        return False
    return True


def scan_for_changes() -> list[dict]:
    """扫描docs/文件变更"""
    changes = []
    last_state = load_last_sync_state()
    last_hashes = last_state.get("file_hashes", {})

    for spec_file in SPEC_MAPPING:
        full_path = STANDARDS_REPO / spec_file
        if not full_path.exists():
            continue

        current_hash = get_file_hash(STANDARDS_REPO, spec_file)
        last_hash = last_hashes.get(spec_file, "")

        if current_hash != last_hash:
            commit_info = get_git_log(STANDARDS_REPO, spec_file)
            parts = commit_info.split("|") if commit_info != "unknown" else ["?", "?", "?"]
            changes.append(
                {
                    "spec_file": spec_file,
                    "local_standard": SPEC_MAPPING[spec_file],
                    "current_hash": current_hash,
                    "last_hash": last_hash,
                    "last_commit": parts[0][:8] if len(parts) > 0 else "?",
                    "last_commit_time": parts[1] if len(parts) > 1 else "?",
                    "last_commit_msg": parts[2] if len(parts) > 2 else "?",
                    "is_first_sync": not bool(last_hash),
                }
            )

    return changes


def generate_sync_report(changes: list[dict]) -> str:
    """生成同步报告（不修改任何规则文件 —— 防止AI自动漂移）"""
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# 规范同步报告 - {ts}",
        "",
        "## ⚠️ 重要说明",
        "本脚本 **只检测变更，不自动修改 quality-standards/**。",
        "所有规则更新必须经人工确认后手动同步。",
        "— 防止AI自动漂移 (docs/07-PRECOMMIT-STANDARDS.md#anti-drift)",
        "",
    ]

    if not changes:
        lines += ["## ✅ 无变更", "", "质量标准文档与上游保持一致，无需同步。"]
        return "\n".join(lines)

    first_sync = [c for c in changes if c["is_first_sync"]]
    updated = [c for c in changes if not c["is_first_sync"]]

    if updated:
        lines += [f"## 🔄 发现 {len(updated)} 处上游更新（需人工审查）", ""]
        for c in updated:
            lines += [
                f"### `{c['spec_file']}`",
                f"- **影响本地文件**: `{c['local_standard']}`",
                f"- **最��commit**: `{c['last_commit']}` ({c['last_commit_time']})",
                f"- **变更内容**: {c['last_commit_msg']}",
                "",
                "**人工操作步骤**:",
                f"1. `diff {STANDARDS_REPO}/{c['spec_file']} {c['local_standard']}`",
                "2. 评估变更是否影响阈值/规则",
                "3. 如需更新，手动修改并添加 `# Src:` 引用",
                "4. 运行验证: `grep -c '# Src:' quality-standards/*.md`",
                "",
            ]

    if first_sync:
        lines += [f"## 📝 首次同步基线（{len(first_sync)} 个文件）", ""]
        for c in first_sync:
            lines += [f"- `{c['spec_file']}` → hash `{c['current_hash']}`"]
        lines += ["", "已记录为基线，下次运行将基于此进行变更检测。"]

    lines += [
        "",
        "## 🔍 快速验证命令",
        "```bash",
        "# 检查引用覆盖率",
        "for f in quality-standards/*.md; do",
        '  echo "$f: $(grep -c "# Src:" "$f") 引用"',
        "done",
        "",
        "# 重新运行质量检查",
        "pre-commit run --all-files",
        "```",
    ]

    return "\n".join(lines)


def update_checkpoint(changes: list[dict]) -> None:
    """更新checkpoint，记录当前文件hash"""
    hashes = {}
    for spec_file in SPEC_MAPPING:
        full_path = STANDARDS_REPO / spec_file
        if full_path.exists():
            hashes[spec_file] = get_file_hash(STANDARDS_REPO, spec_file)
    save_sync_state(hashes)


def main() -> None:
    print("=== 规范同步检查 ===")
    print(f"标准仓库: {STANDARDS_REPO}")
    print(f"检查时间: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print()

    if not check_standards_available():
        return

    print("正在扫描变更...")
    changes = scan_for_changes()

    report = generate_sync_report(changes)
    print(report)

    # 保存报告到状态目录
    report_path = STATE_DIR / f"sync-report-{datetime.utcnow().strftime('%Y%m%d')}.md"
    STATE_DIR.mkdir(exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"\n📄 报告已保存: {report_path}")

    # 更新checkpoint中的hash基线
    update_checkpoint(changes)
    print("✅ 基线已更新到 .quality-state/checkpoint.json")

    if changes and any(not c["is_first_sync"] for c in changes):
        print("\n⚠️  检测到上游规范更新，请人工审查后决定是否同步。")
        print("   禁止AI自动修改 quality-standards/ 文件。")


if __name__ == "__main__":
    main()
