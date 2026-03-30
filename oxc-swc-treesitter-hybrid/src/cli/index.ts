#!/usr/bin/env node

// src/cli/index.ts
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import { HybridParser } from '../index';
import { globby } from 'globby';
import fs from 'fs/promises';
import path from 'path';

async function main() {
  const parser = new HybridParser();

  const argv = await yargs(hideBin(process.argv))
    .usage('Usage: $0 <command> [options]')
    .command('parse <file>', 'Parse a single file', (yargs) => {
      return yargs
        .positional('file', {
          describe: 'File to parse',
          type: 'string'
        })
        .option('output', {
          alias: 'o',
          type: 'string',
          description: 'Output file for results'
        })
        .option('format', {
          alias: 'f',
          type: 'string',
          default: 'json',
          choices: ['json', 'ast', 'metrics'],
          description: 'Output format'
        });
    })
    .command('analyze [patterns...]', 'Analyze files matching patterns', (yargs) => {
      return yargs
        .positional('patterns', {
          describe: 'Glob patterns to match files',
          type: 'string',
          default: ['**/*.{js,ts,jsx,tsx,py,go,java}']
        })
        .option('output', {
          alias: 'o',
          type: 'string',
          description: 'Output file for results'
        })
        .option('format', {
          alias: 'f',
          type: 'string',
          default: 'json',
          choices: ['json', 'csv', 'summary'],
          description: 'Output format'
        });
    })
    .command('watch <dir>', 'Watch directory for changes and parse in real-time', (yargs) => {
      return yargs
        .positional('dir', {
          describe: 'Directory to watch',
          type: 'string'
        })
        .option('recursive', {
          alias: 'r',
          type: 'boolean',
          default: true,
          description: 'Watch recursively'
        });
    })
    .demandCommand(1, 'Please specify a command')
    .help()
    .argv;

  switch (argv._[0]) {
    case 'parse':
      await handleParseCommand(parser, argv);
      break;
    case 'analyze':
      await handleAnalyzeCommand(parser, argv);
      break;
    case 'watch':
      await handleWatchCommand(parser, argv);
      break;
    default:
      console.error(`Unknown command: ${argv._[0]}`);
      process.exit(1);
  }
}

async function handleParseCommand(parser: HybridParser, argv: any) {
  const filePath = argv.file as string;
  
  if (!(await fileExists(filePath))) {
    console.error(`File does not exist: ${filePath}`);
    process.exit(1);
  }

  try {
    const code = await fs.readFile(filePath, 'utf-8');
    const result = await parser.parse(filePath, code);
    
    let output: string;
    switch (argv.format) {
      case 'ast':
        output = JSON.stringify(result.ast, null, 2);
        break;
      case 'metrics':
        output = JSON.stringify(result.metrics, null, 2);
        break;
      case 'json':
      default:
        output = JSON.stringify(result, null, 2);
        break;
    }
    
    if (argv.output) {
      await fs.writeFile(argv.output, output);
      console.log(`Results written to ${argv.output}`);
    } else {
      console.log(output);
    }
  } catch (error) {
    console.error(`Error parsing file: ${(error as Error).message}`);
    process.exit(1);
  }
}

async function handleAnalyzeCommand(parser: HybridParser, argv: any) {
  const patterns = argv.patterns as string[];
  
  // 使用 globby 查找匹配的文件
  const files = await globby(patterns);
  
  console.log(`Found ${files.length} files to analyze...`);
  
  const results = [];
  for (const file of files) {
    try {
      const code = await fs.readFile(file, 'utf-8');
      const result = await parser.parse(file, code);
      results.push({ file, result });
      console.log(`Analyzed: ${file}`);
    } catch (error) {
      console.error(`Error analyzing ${file}: ${(error as Error).message}`);
    }
  }
  
  let output: string;
  switch (argv.format) {
    case 'csv':
      // 简单CSV格式，输出文件名和LOC
      output = 'file,loc,cyclomatic_complexity\n';
      results.forEach(r => {
        if (r.result.metrics) {
          output += `${r.file},${r.result.metrics.loc},${r.result.metrics.cyclomaticComplexity}\n`;
        }
      });
      break;
    case 'summary':
      // 汇总信息
      const totalLoc = results.reduce((sum, r) => sum + (r.result.metrics?.loc || 0), 0);
      const avgComplexity = results.length > 0 
        ? results.reduce((sum, r) => sum + (r.result.metrics?.cyclomaticComplexity || 0), 0) / results.length 
        : 0;
        
      output = `Analysis Summary:\n`;
      output += `- Files analyzed: ${results.length}\n`;
      output += `- Total LOC: ${totalLoc}\n`;
      output += `- Average cyclomatic complexity: ${avgComplexity.toFixed(2)}\n`;
      break;
    case 'json':
    default:
      output = JSON.stringify(results, null, 2);
      break;
  }
  
  if (argv.output) {
    await fs.writeFile(argv.output, output);
    console.log(`Results written to ${argv.output}`);
  } else {
    console.log(output);
  }
}

async function handleWatchCommand(parser: HybridParser, argv: any) {
  const chokidar = (await import('chokidar')).default;
  const dirPath = argv.dir as string;
  
  console.log(`Watching directory: ${dirPath}`);
  
  const watcher = chokidar.watch(dirPath, {
    ignored: /[\/\\]\./, // 忽略隐藏文件
    persistent: true
  });
  
  watcher.on('change', async (filePath) => {
    console.log(`File changed: ${filePath}`);
    
    // 检查是否支持该语言
    const ext = path.extname(filePath).substring(1);
    if (!parser.supportsLanguage(ext)) {
      console.log(`Unsupported language for file: ${filePath}`);
      return;
    }
    
    try {
      const code = await fs.readFile(filePath, 'utf-8');
      const result = await parser.parse(filePath, code);
      console.log(`Parsed ${filePath}: ${result.errors.length} errors`);
    } catch (error) {
      console.error(`Error parsing file ${filePath}: ${(error as Error).message}`);
    }
  });
  
  console.log('Press Ctrl+C to exit.');
  
  // 监听退出信号
  process.on('SIGINT', () => {
    console.log('\nStopping file watcher...');
    process.exit(0);
  });
}

async function fileExists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

// 运行主函数
main().catch(err => {
  console.error(err);
  process.exit(1);
});

export default main;