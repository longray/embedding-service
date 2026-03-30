// src/types/common.ts

/**
 * 位置信息
 */
export interface Position {
  line: number; // 从0开始
  column: number; // 从0开始
}

/**
 * 范围信息
 */
export interface Range {
  start: Position;
  end: Position;
}

/**
 * 解析结果选项
 */
export interface ParseOptions {
  includeMetrics?: boolean;
  includeComments?: boolean;
  includePositions?: boolean;
  jsx?: boolean;
  tsx?: boolean;
  target?: string; // ES2015, ES2016, etc.
  module?: string; // commonjs, es6, umd, etc.
}