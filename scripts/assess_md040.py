#!/usr/bin/env python3
"""MD040 现状评估脚本"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict


def find_markdown_files(root: Path):
    """查找所有 Markdown 文件"""
    return list(root.rglob("*.md"))


def analyze_file(file_path: Path):
    """分析单个文件中的代码块"""
    content = file_path.read_text(encoding="utf-8")
    lines = content.split("\n")

    unlabeled = []
    for i, line in enumerate(lines):
        if re.match(r"^```\s*$", line):
            unlabeled.append((i + 1, line))

    return len(unlabeled), unlabeled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", type=Path, default=Path("docs/MD040_ASSESSMENT_REPORT.md"))
    args = parser.parse_args()

    md_files = find_markdown_files(Path("."))
    total = 0
    results = []

    for f in sorted(md_files):
        count, lines = analyze_file(f)
        if count > 0:
            total += count
            results.append((f, count))

    # 生成报告
    with open(args.output, "w") as f:
        f.write("# MD040 评估报告\n\n")
        f.write(f"总计: {len(md_files)} 个文件\n")
        f.write(f"无语言标记代码块: {total} 处\n\n")

        f.write("## 重灾区文件\n\n")
        for file_path, count in sorted(results, key=lambda x: -x[1])[:20]:
            f.write(f"- {file_path}: {count} 处\n")

    print(f"✅ 报告已生成: {args.output}")
    print(f"📊 总计: {total} 个无标记代码块")


if __name__ == "__main__":
    main()
