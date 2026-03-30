// src/parsers/base-parser.ts
import { Position, Range } from '../types/common';

/**
 * 代码解析器的统一接口
 */
export interface ParserResult {
  ast: any;
  language: string;
  filename: string;
  errors: ParseError[];
  metrics?: CodeMetrics;
}

export interface ParseError {
  message: string;
  position: Position;
  severity: 'error' | 'warning';
}

export interface CodeMetrics {
  loc: number; // Lines of code
  cyclomaticComplexity: number;
  halsteadMetrics?: HalsteadMetrics;
  maintainabilityIndex?: number;
}

export interface HalsteadMetrics {
  n1: number; // 独特的操作符数量
  n2: number; // 独特的操作数数量
  N1: number; // 总操作符数量
  N2: number; // 总操作数数量
  vocabulary: number; // n1 + n2
  length: number; // N1 + N2
  calculatedLength: number; // n1 * log2(n1) + n2 * log2(n2)
  volume: number; // length * log2(vocabulary)
  difficulty: number; // (n1 / 2) * (N2 / n2)
  effort: number; // difficulty * volume
}

/**
 * 抽象解析器基类
 */
export abstract class BaseParser {
  protected language: string;

  constructor(language: string) {
    this.language = language;
  }

  /**
   * 解析代码文件
   * @param code 代码内容
   * @param filename 文件名
   */
  abstract parse(code: string, filename: string): Promise<ParserResult>;

  /**
   * 增量解析（仅对支持增量解析的解析器）
   * @param oldCode 旧代码
   * @param newCode 新代码
   * @param delta 变更范围
   */
  abstract incrementalParse?(oldCode: string, newCode: string, delta: Range): Promise<ParserResult>;

  /**
   * 获取解析器名称
   */
  abstract getName(): string;

  /**
   * 检查解析器是否支持指定语言
   * @param language 语言标识符
   */
  abstract supportsLanguage(language: string): boolean;
}