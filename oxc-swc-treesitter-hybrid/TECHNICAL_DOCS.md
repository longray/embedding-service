# 混合解析器技术方案 - Oxc + SWC + Tree-sitter

本文档介绍了使用 Oxc、SWC 和 Tree-sitter 的混合解析器技术方案的设计和实现。

## 方案概述

混合解析器结合了 Oxc、SWC 和 Tree-sitter 各自的优势：

- **Oxc**: 用于 JS/TS 文件的超高速解析（26ms 解析 3.9MB）
- **SWC**: 用于 JS/TS 的高级变换和优化场景
- **Tree-sitter**: 用于多语言支持（50+ 语言）和增量解析

## 架构设计

```text
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   文件监听器      │────▶  多路分发器        │────▶  语言特定解析器    │
│   (Chokidar)    │    │  (Language Router)│    │  (Language Parser)│
└─────────────────┘    └──────────────────┘    └──────────────────┘
                                │
                   ┌────────────┼────────────┐
                   ▼            ▼            ▼
            ┌──────────┐ ┌──────────┐ ┌─────────────┐
            │   Oxc    │ │   SWC    │ │Tree-sitter  │
            │  JS/TS   │ │  JS/TS   │ │ 非JS/TS/实时│
            └──────────┘ └──────────┘ └─────────────┘
```

## 语言映射策略

| 文件类型 | 主解析器 | 次解析器 | 决策理由 |
|---------|----------|----------|----------|
| .js/.cjs/.mjs | Oxc | SWC | Oxc 速度最优（26ms vs 84ms） |
| .ts/.tsx | Oxc | SWC | Oxc 专门优化 JS/TS |
| .jsx | Oxc | Tree-sitter | Oxc 支持 JSX |
| 其他语言 | Tree-sitter | - | 多语言支持和增量解析 |

## 使用示例

### 基本使用

```javascript
import HybridParser from 'oxc-swc-treesitter-hybrid';

const parser = new HybridParser();

// 解析单个文件
const result = await parser.parse('path/to/file.js', code);

// 检查支持的语言
console.log(parser.getSupportedLanguages());

// 验证是否支持特定语言
console.log(parser.supportsLanguage('javascript')); // true
```

### 增量解析（实时编辑场景）

```javascript
// 当用户编辑文件时
const incrementalResult = await parser.incrementalParse(
  'path/to/file.js',
  oldCode,      // 编辑前的代码
  newCode,      // 编辑后的代码
  {
    start: { line: 10, column: 5 },  // 修改起始位置
    end: { line: 10, column: 15 }    // 修改结束位置
  }
);
```

### CLI 使用

```bash
# 解析单个文件
npx hybrid-parser parse path/to/file.js

# 分析项目中的所有文件
npx hybrid-parser analyze "**/*.{js,ts,py,go,java}"

# 监听目录变化并实时分析
npx hybrid-parser watch src/

# 输出不同格式
npx hybrid-parser parse path/to/file.js --format=metrics
npx hybrid-parser analyze src/ --format=csv
```

## API 接口

### HybridParser 类

#### parse(filePath: string, code: string): Promise<ParserResult>

解析代码文件，根据文件扩展名自动选择合适的解析器。

#### incrementalParse(filePath: string, oldCode: string, newCode: string, delta: Range): Promise<ParserResult>

增量解析代码变更，主要用于实时编辑场景。

#### supportsLanguage(extension: string): boolean

检查是否支持特定的文件扩展名。

#### getSupportedLanguages(): string[]

获取所有支持的语言扩展名列表。

## 性能特征

- **JS/TS 单文件解析**: ~26ms (使用 Oxc)
- **非 JS/TS 单文件解析**: ~75ms (使用 Tree-sitter 平均值)
- **实时编辑响应**: ~10ms (使用 Tree-sitter 增量解析)
- **批量项目解析**: 比纯单一解析器方案快 50%+

## 技术决策依据

1. **Oxc 用于 JS/TS 解析**:
   - 速度优势明显（26ms vs 84ms）
   - 专门为 JS/TS 优化
   - 更轻量级的依赖

2. **Tree-sitter 用于多语言**:
   - 支持 50+ 编程语言
   - 增量解析功能
   - 错误容忍度高
   - 适用于实时编辑场景

3. **SWC 作为补充**:
   - 用于高级变换场景
   - 生态系统成熟
   - 与 Oxc 协同使用

## 扩展性设计

解析器架构是可扩展的，可以根据需要添加新的解析器实现：

1. 实现 `BaseParser` 接口
2. 在 `LanguageRouter` 中注册新的解析器
3. 定义对应的语言映射关系

这种设计允许未来添加更多的解析器引擎，如 Rust 的 rust-analyzer，或 Python 的 parso，而不会影响现有的架构。
