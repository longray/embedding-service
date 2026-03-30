// src/parsers/tree-sitter-parser.ts
import Parser from 'tree-sitter';
import JavaScript from 'tree-sitter-javascript';
import TypeScript from 'tree-sitter-typescript';
import Python from 'tree-sitter-python';
import Go from 'tree-sitter-go';
import Java from 'tree-sitter-java';
import { BaseParser, ParserResult, ParseError, CodeMetrics } from './base-parser';
import { Position, Range } from '../types/common';

/**
 * Tree-sitter 解析器实现（支持多语言和增量解析）
 */
export class TreeSitterParser extends BaseParser {
  private parser: Parser;
  private readonly parsers: Map<string, any>;
  
  constructor() {
    super('');
    
    // 初始化 Tree-sitter 解析器
    this.parser = new Parser();
    this.parsers = new Map();
    
    // 注册支持的语言
    this.registerLanguage('javascript', JavaScript);
    this.registerLanguage('typescript', TypeScript.typescript);
    this.registerLanguage('tsx', TypeScript.tsx);
    this.registerLanguage('python', Python);
    this.registerLanguage('go', Go);
    this.registerLanguage('java', Java);
    
    // 为 JS/TS 设置默认解析器
    this.language = 'javascript';
  }

  /**
   * 注册一种语言的解析器
   */
  private registerLanguage(language: string, parser: any) {
    this.parsers.set(language, parser);
  }

  async parse(code: string, filename: string): Promise<ParserResult> {
    const ext = this.getLanguageFromFilename(filename);
    const parser = this.parsers.get(ext);
    
    if (!parser) {
      throw new Error(`Tree-sitter parser does not support ${ext} language`);
    }

    // 设置语言
    this.parser.setLanguage(parser);
    
    // 解析代码
    const tree = this.parser.parse(code);
    const ast = tree.rootNode;
    
    // 收集错误（Tree-sitter 错误节点）
    const errors: ParseError[] = [];
    this.findErrors(ast, code, errors);

    // 计算指标
    const metrics = errors.length === 0 ? await this.calculateMetrics(code, ast) : undefined;

    return {
      ast,
      language: ext,
      filename,
      errors,
      metrics
    };
  }

  async incrementalParse(oldCode: string, newCode: string, delta: Range): Promise<ParserResult> {
    // 获取当前树
    const oldTree = this.parser.parse(oldCode);
    
    // 获取变更信息
    const startIndex = this.positionToIndex(oldCode, delta.start);
    const oldEndIndex = this.positionToIndex(oldCode, delta.end);
    const newEndIndex = this.positionToIndex(newCode, delta.end);
    
    const startPos = { row: delta.start.line, column: delta.start.column };
    const oldEndPos = { row: delta.end.line, column: delta.end.column };
    const newEndPos = { row: delta.end.line, column: delta.end.column };
    
    // 告诉 Tree-sitter 发生了哪些变更
    oldTree.edit({
      startIndex,
      oldEndIndex,
      newEndIndex,
      startPosition: startPos,
      oldEndPosition: oldEndPos,
      newEndPosition: newEndPos
    });
    
    // 重新解析变更后的代码
    const newTree = this.parser.parse(newCode, oldTree);
    
    // 获取文件扩展名以确定语言
    const ext = this.getLanguageFromFilename('temp.ts'); // 使用默认扩展名
    const errors: ParseError[] = [];
    this.findErrors(newTree.rootNode, newCode, errors);
    
    const metrics = errors.length === 0 ? await this.calculateMetrics(newCode, newTree.rootNode) : undefined;
    
    return {
      ast: newTree.rootNode,
      language: ext,
      filename: 'temp.ts',
      errors,
      metrics
    };
  }

  getName(): string {
    return 'TreeSitterParser';
  }

  supportsLanguage(language: string): boolean {
    return this.parsers.has(language.toLowerCase());
  }

  /**
   * 从文件名获取语言类型
   */
  private getLanguageFromFilename(filename: string): string {
    const ext = filename.split('.').pop()?.toLowerCase() || '';
    switch (ext) {
      case 'js':
      case 'cjs':
      case 'mjs':
        return 'javascript';
      case 'ts':
        return 'typescript';
      case 'tsx':
        return 'tsx';
      case 'py':
        return 'python';
      case 'go':
        return 'go';
      case 'java':
        return 'java';
      default:
        return ext;
    }
  }

  /**
   * 查找解析错误
   */
  private findErrors(node: Parser.SyntaxNode, code: string, errors: ParseError[]): void {
    if (node.type === 'ERROR' || node.isError) {
      const lines = code.substring(0, node.startIndex).split('\n');
      const lastLine = lines[lines.length - 1];
      
      errors.push({
        message: `Syntax error at line ${node.startPosition.row}, column ${node.startPosition.column}`,
        position: {
          line: node.startPosition.row,
          column: node.startPosition.column
        },
        severity: 'error'
      });
    }
    
    // 递归检查子节点
    for (let i = 0; i < node.childCount; i++) {
      this.findErrors(node.child, code, errors);
    }
  }

  /**
   * 计算代码指标
   */
  private async calculateMetrics(code: string, ast: Parser.SyntaxNode): Promise<CodeMetrics> {
    // 简单LOC计算
    const loc = code.split('\n').length;
    
    // 计算圈复杂度
    let cyclomaticComplexity = 1;
    this.countComplexityNodes(ast, (nodeType: string) => {
      switch (nodeType) {
        case 'if_statement':
        case 'else_clause':
        case 'switch_statement':
        case 'for_statement':
        case 'while_statement':
        case 'do_statement':
        case 'catch_clause':
        case 'logical_expression':
          // 在JavaScript中，逻辑运算符如 && 和 || 会增加复杂度
          if(nodeType === 'logical_expression' && 
             (ast.text.includes('&&') || ast.text.includes('||'))) {
            cyclomaticComplexity++;
          } else if (nodeType !== 'else_clause') {
            // else子句不增加复杂度
            cyclomaticComplexity++;
          }
          break;
      }
    });

    return {
      loc,
      cyclomaticComplexity
    };
  }

  /**
   * 遍历节点以计算复杂度
   */
  private countComplexityNodes(node: Parser.SyntaxNode, callback: (type: string) => void): void {
    callback(node.type);
    
    for (let i = 0; i < node.childCount; i++) {
      this.countComplexityNodes(node.child(i), callback);
    }
  }

  /**
   * 将位置转换为索引
   */
  private positionToIndex(code: string, pos: Position): number {
    const lines = code.split('\n');
    let index = 0;
    
    for (let i = 0; i < pos.line; i++) {
      index += lines[i].length + 1; // +1 for newline
    }
    
    index += pos.column;
    return index;
  }
}