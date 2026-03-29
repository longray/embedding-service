#!/usr/bin/env python3
"""修复 MD031/MD032 错误（代码块前后需要空行）"""
import re
from pathlib import Path

def fix_file(path):
    """修复单个文件"""
    content = path.read_text(encoding="utf-8")
    lines = content.split("\n")
    new_lines = []
    
    for i, line in enumerate(lines):
        # 检查是否是代码块开始
        if re.match(r"^```\w*$", line):
            # 检查前一行是否为空行或者是文件开始
            if i > 0 and new_lines[-1].strip() != "":
                new_lines.append("")
        
        new_lines.append(line)
        
        # 检查是否是代码块结束
        if re.match(r"^```\s*$", line) and i < len(lines) - 1:
            # 检查后一行是否为空行
            if lines[i + 1].strip() != "":
                new_lines.append("")
    
    path.write_text("\n".join(new_lines), encoding="utf-8")

def main():
    # 修复有错误的文件
    files = [
        "docs/START_GUIDE.md",
        "docs/SURREALDB_3_UPGRADE_DESIGN.md",
        "docs/SCHEME-EVALUATION-REPORT.md",
        "docs/API_SPECIFICATION.md",
        "docs/CODE-ANALYSIS-UNIFIED-DESIGN.md",
        "docs/testing-plan.md",
        "docs/architecture/WRAPPER_SERVICE_DESIGN.md",
        "scripts/README.md",
        "quality-standards/03-gates.md",
        "quality-standards/04-workflow.md",
    ]
    
    for f in files:
        p = Path(f)
        if p.exists():
            print(f"Fixing {f}...")
            fix_file(p)

if __name__ == "__main__":
    main()
