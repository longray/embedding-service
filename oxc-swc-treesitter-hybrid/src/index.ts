// src/index.ts
import { LanguageRouter } from './routers/language-router';
import { OxcParser } from './parsers/oxc-parser';
import { TreeSitterParser } from './parsers/tree-sitter-parser';
import { BaseParser, ParserResult } from './parsers/base-parser';
import { ParseOptions } from './types/common';

export {
  LanguageRouter,
  OxcParser,
  TreeSitterParser,
  BaseParser,
  ParserResult,
  ParseOptions
};

// 主类导出
class HybridParser {
  private router: LanguageRouter;

  constructor() {
    this.router = new LanguageRouter();
  }

  /**
   * 解析代码文件
   * @param filePath 文件路径
   * @param code 文件内容（可选，如果不提供则需要在内部获取）
   */
  async parse(filePath: string, code: string): Promise<ParserResult> {
    return await this.router.parse(filePath, code);
  }

  /**
   * 增量解析 - 适用于实时编辑场景
   */
  async incrementalParse(
    filePath: string,
    oldCode: string,
    newCode: string,
    delta: { start: { line: number; column: number }; end: { line: number; column: number } }
  ): Promise<ParserResult> {
    return await this.router.incrementalParse(filePath, oldCode, newCode, delta);
  }

  /**
   * 检查是否支持特定语言
   */
  supportsLanguage(extension: string): boolean {
    return this.router.supportsLanguage(extension);
  }

  /**
   * 获取所有支持的语言
   */
  getSupportedLanguages(): string[] {
    return this.router.getSupportedLanguages();
  }
}

export default HybridParser;