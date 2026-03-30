## Hybrid Parser (Oxc + SWC + Tree-sitter)

A high-performance code analysis tool that combines the strengths of Oxc, SWC, and Tree-sitter to provide optimal parsing performance across multiple languages.

### Architecture

The hybrid parser uses a router to direct files to the most appropriate parser based on file type and use case:

- **Oxc**: Ultra-fast JS/TS parsing for static analysis
- **SWC**: JS/TS compilation and transformation tasks
- **Tree-sitter**: Multi-language support and incremental parsing

### Features

- Support for 50+ programming languages
- Ultra-fast JS/TS parsing using Oxc
- Incremental parsing for real-time analysis
- Unified AST interface across all parsers
- CLI tool for batch processing and analysis
- Performance benchmarking tools

### Installation

```bash
npm install oxc-swc-treesitter-hybrid
```

### Usage

```javascript
import { HybridParser } from 'oxc-swc-treesitter-hybrid';

const parser = new HybridParser();
const ast = parser.parse('path/to/file.js');
```

### CLI Usage

```bash
# Parse a file
npx hybrid-parser parse path/to/file.js

# Analyze a project
npx hybrid-parser analyze ./src --report

# Watch files for changes
npx hybrid-parser watch ./src --metrics
```
