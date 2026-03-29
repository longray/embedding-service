#!/usr/bin/env python3
"""MD040 自动修复脚本"""

import re
from pathlib import Path


def infer_language(content):
    """推断代码块语言"""
    c = content.lower()
    if any(k in c for k in ["def ", "import ", "print(", "class ", "python"]):
        return "python"
    if any(k in c for k in ["#!/bin/bash", "$ ", "curl ", "pip ", "npm ", "uv ", "git "]):
        return "bash"
    if c.strip().startswith(("{", "[")) and '"' in c:
        return "json"
    if ": " in c and not c.strip().startswith(("{", "[")):
        return "yaml"
    if "├──" in c or "└──" in c:
        return "text"
    return "text"


def fix_file(path):
    """修复单个文件"""
    content = path.read_text(encoding="utf-8")
    lines = content.split("\n")
    new_lines = []
    i = 0
    fixed = 0

    while i < len(lines):
        line = lines[i]
        if re.match(r"^```\s*$", line):
            block = []
            i += 1
            while i < len(lines) and not re.match(r"^```\s*$", lines[i]):
                block.append(lines[i])
                i += 1
            lang = infer_language("\n".join(block[:3]))
            new_lines.append(f"```{lang}")
            new_lines.extend(block)
            if i < len(lines):
                new_lines.append(lines[i])
            fixed += 1
        else:
            new_lines.append(line)
        i += 1

    path.write_text("\n".join(new_lines), encoding="utf-8")
    return fixed


def main():
    total = 0
    for f in sorted(Path(".").rglob("*.md")):
        if any(x in str(f) for x in [".venv", "venv", "node_modules", "archive", "CHANGELOG"]):
            continue
        try:
            n = fix_file(f)
            if n > 0:
                print(f"Fixed {f}: {n}")
                total += n
        except Exception as e:
            print(f"Error {f}: {e}")

    print(f"\n✅ 总计修复: {total} 处")


if __name__ == "__main__":
    main()
