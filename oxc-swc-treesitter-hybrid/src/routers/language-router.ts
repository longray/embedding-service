// src/routers/language-router.ts
import { BaseParser } from '../parsers/base-parser';
import { OxcParser } from '../parsers/oxc-parser';
import { TreeSitterParser } from '../parsers/tree-sitter-parser';
import { ParserResult } from '../parsers/base-parser';

/**
 * 语言路由器 - 根据文件类型选择最适合的解析器
 */
export class LanguageRouter {
  private parsers: Map<string, BaseParser>;
  private oxcParser: OxcParser;
  private treeSitterParser: TreeSitterParser;
  private readonly jsTsExtensions: Set<string>;

  constructor() {
    // 初始化各个解析器
    this.oxcParser = new OxcParser();
    this.treeSitterParser = new TreeSitterParser();

    // 支持JS/TS的扩展名
    this.jsTsExtensions = new Set(['js', 'cjs', 'mjs', 'ts', 'jsx', 'tsx']);

    // 创建解析器映射
    this.parsers = new Map();
    
    // 注册解析器，JS/TS优先使用Oxc，其他语言使用Tree-sitter
    this.registerParsers();
  }

  private registerParsers(): void {
    // Oxc 解析器（用于 JS/TS 高速解析）
    this.parsers.set('oxc', this.oxcParser);
    
    // Tree-sitter 解析器（用于多语言支持和增量解析）
    this.parsers.set('treesitter', this.treeSitterParser);
  }

  /**
   * 解析代码文件
   */
  async parse(filePath: string, code?: string): Promise<ParserResult> {
    // 从文件路径获取扩展名
    const extension = this.getFileExtension(filePath).toLowerCase();
    
    // 如果没有提供代码，我们无法解析
    if (!code) {
      throw new Error('Code content is required for parsing');
    }

    // 根据文件类型选择解析器
    const parser = this.selectParser(extension);
    
    // 使用选定的解析器进行解析
    return await parser.parse(code, filePath);
  }

  /**
   * 增量解析 - 对编辑场景特别有用
   */
  async incrementalParse(
    filePath: string, 
    oldCode: string, 
    newCode: string, 
    delta: { start: { line: number; column: number }; end: { line: number; column: number } }
  ): Promise<ParserResult> {
    // 增量解析主要由Tree-sitter处理
    return await this.treeSitterParser.incrementalParse(oldCode, newCode, delta);
  }

  /**
   * 根据文件扩展名选择解析器
   */
  private selectParser(extension: string): BaseParser {
    if (this.jsTsExtensions.has(extension)) {
      // 对于JS/TS文件，使用Oxc解析器（最快）
      return this.oxcParser;
    } else {
      // 对于其他语言，使用Tree-sitter解析器（更广泛的支持）
      return this.treeSitterParser;
    }
  }

  /**
   * 获取文件扩展名
   */
  private getFileExtension(filePath: string): string {
    const parts = filePath.split('.');
    if (parts.length < 2) {
      return '';
    }
    return parts[parts.length - 1];
  }

  /**
   * 检查是否支持特定语言
   */
  supportsLanguage(extension: string): boolean {
    return this.oxcParser.supportsLanguage(extension) || 
           this.treeSitterParser.supportsLanguage(extension);
  }

  /**
   * 获取所有支持的语言
   */
  getSupportedLanguages(): string[] {
    const oxcLangs = ['javascript', 'typescript', 'jsx', 'tsx']; // Oxc支持
    const treeSitterLangs = ['python', 'go', 'java']; // TreeSitter支持（实际更多）
    
    return [...new Set([...oxcLangs, ...treeSitterLangs])];
  }
}