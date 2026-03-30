// src/parsers/oxc-parser.ts
import { BaseParser, ParserResult, ParseError, CodeMetrics } from './base-parser';
import { ParseOptions } from '../types/common';

// Oxc 解析器的类型定义
interface OxcParseResult {
  program: any;
  errors: OxcError[];
}

interface OxcError {
  message: string;
  line: number;
  col: number;
  fatal: boolean;
}

/**
 * Oxc解析器实现（专门用于JS/TS文件的高速解析）
 */
export class OxcParser extends BaseParser {
  private oxcModule: any;
  
  constructor() {
    super('javascript'); // 主要处理JS/TS
    
    try {
      // 动态导入Oxc，如果未安装则抛出错误
      const oxc = require('@oxc-parser/wasm');
      oxc.init(); // 初始化WebAssembly
      this.oxcModule = oxc;
    } catch (error) {
      throw new Error(`Failed to initialize Oxc parser: ${error.message}`);
    }
  }

  async parse(code: string, filename: string): Promise<ParserResult> {
    try {
      // 检测语言类型
      const ext = filename.split('.').pop()?.toLowerCase() || '';
      const language = this.getLanguageFromExtension(ext);
      
      if (!this.supportsLanguage(language)) {
        throw new Error(`Oxc parser does not support ${language}`);
      }

      // 使用Oxc解析代码
      const parseResult: OxcParseResult = this.oxcModule.parse(
        code,
        filename,
        {
          sourceType: ext === 'ts' || ext === 'tsx' ? 'typescript' : 'script',
          jsx: ext === 'jsx' || ext === 'tsx',
          allowReturnOutsideFunction: true,
          allowImportExportEverywhere: true,
          allowSuperOutsideMethod: true,
          allowHashBang: true,
        }
      );

      // 转换错误格式
      const errors: ParseError[] = parseResult.errors.map(error => ({
        message: error.message,
        position: { line: error.line - 1, column: error.col - 1 }, // 转换为0基索引
        severity: error.fatal ? 'error' : 'warning'
      }));

      // 如果没有解析错误，返回AST
      return {
        ast: parseResult.program,
        language,
        filename,
        errors,
        metrics: errors.length === 0 ? await this.calculateMetrics(code, parseResult.program) : undefined
      };
    } catch (error) {
      throw new Error(`Oxc parsing failed for ${filename}: ${error.message}`);
    }
  }

  // Oxc目前不支持增量解析，但为了接口兼容保留此方法
  async incrementalParse(oldCode: string, newCode: string, delta: any): Promise<ParserResult> {
    // 由于Oxc不支持增量解析，我们退回到完全解析
    return this.parse(newCode, 'incremental.ts');
  }

  getName(): string {
    return 'OxcParser';
  }

  supportsLanguage(language: string): boolean {
    const supportedLanguages = ['javascript', 'typescript', 'jsx', 'tsx'];
    return supportedLanguages.includes(language.toLowerCase());
  }

  private getLanguageFromExtension(ext: string): string {
    switch (ext) {
      case 'js':
        return 'javascript';
      case 'ts':
        return 'typescript';
      case 'jsx':
        return 'jsx';
      case 'tsx':
        return 'tsx';
      default:
        return ext;
    }
  }

  /**
   * 计算代码的基本指标
   */
  private async calculateMetrics(code: string, ast: any): Promise<CodeMetrics> {
    // 简单LOC计算
    const loc = code.split('\n').length;

    // 计算循环复杂度（简化版本）
    let cyclomaticComplexity = 1; // 基础值
    if (ast && ast.body) {
      this.traverseNode(ast, (node: any) => {
        switch (node.type) {
          case 'IfStatement':
          case 'ConditionalExpression':
            cyclomaticComplexity += 1;
            break;
          case 'ForStatement':
          case 'ForInStatement':
          case 'ForOfStatement':
          case 'WhileStatement':
          case 'DoWhileStatement':
            cyclomaticComplexity += 1;
            break;
          case 'CatchClause':
            cyclomaticComplexity += 1;
            break;
          case 'LogicalExpression':
            if (node.operator === '||' || node.operator === '&&') {
              cyclomaticComplexity += 1;
            }
            break;
        }
      });
    }

    return {
      loc,
      cyclomaticComplexity
    };
  }

  private traverseNode(node: any, callback: (node: any) => void) {
    if (!node || typeof node !== 'object') {
      return;
    }

    callback(node);

    Object.keys(node).forEach(key => {
      const child = node[key];
      if (child && typeof child === 'object') {
        if (Array.isArray(child)) {
          child.forEach(item => this.traverseNode(item, callback));
        } else {
          this.traverseNode(child, callback);
        }
      }
    });
  }
}