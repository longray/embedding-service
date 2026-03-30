// tests/basic-test.ts
import { HybridParser } from '../src/index';

async function testBasicFunctionality() {
  console.log('Testing basic functionality...\n');
  
  const parser = new HybridParser();
  
  // 测试JS文件解析
  console.log('1. Testing JS file parsing...');
  const jsCode = `
    function helloWorld() {
      console.log('Hello, world!');
      return true;
    }
    
    const result = helloWorld();
  `;
  
  try {
    const jsResult = await parser.parse('test.js', jsCode);
    console.log(`   ✓ Parsed JS file with ${jsResult.errors.length} errors`);
    console.log(`   ✓ LOC: ${jsResult.metrics?.loc}, Cyclomatic Complexity: ${jsResult.metrics?.cyclomaticComplexity}`);
  } catch (error) {
    console.error(`   ✗ Failed to parse JS file: ${(error as Error).message}`);
  }
  
  // 测试TS文件解析
  console.log('\n2. Testing TS file parsing...');
  const tsCode = `
    interface User {
      id: number;
      name: string;
      email: string;
    }
    
    class UserService {
      public static validateUser(user: User): boolean {
        return user.id > 0 && user.name.length > 0 && user.email.includes('@');
      }
    }
  `;
  
  try {
    const tsResult = await parser.parse('test.ts', tsCode);
    console.log(`   ✓ Parsed TS file with ${tsResult.errors.length} errors`);
    console.log(`   ✓ LOC: ${tsResult.metrics?.loc}, Cyclomatic Complexity: ${tsResult.metrics?.cyclomaticComplexity}`);
  } catch (error) {
    console.error(`   ✗ Failed to parse TS file: ${(error as Error).message}`);
  }
  
  // 测试Python文件解析
  console.log('\n3. Testing Python file parsing...');
  const pyCode = `
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
  `;
  
  try {
    const pyResult = await parser.parse('test.py', pyCode);
    console.log(`   ✓ Parsed Python file with ${pyResult.errors.length} errors`);
    console.log(`   ✓ LOC: ${pyResult.metrics?.loc}, Cyclomatic Complexity: ${pyResult.metrics?.cyclomaticComplexity}`);
  } catch (error) {
    console.error(`   ✗ Failed to parse Python file: ${(error as Error).message}`);
  }
  
  // 测试语言支持检查
  console.log('\n4. Testing language support...');
  console.log(`   ✓ Supports JS: ${parser.supportsLanguage('js')}`);
  console.log(`   ✓ Supports TS: ${parser.supportsLanguage('ts')}`);
  console.log(`   ✓ Supports Python: ${parser.supportsLanguage('python')}`);
  console.log(`   ✓ Supported languages:`, parser.getSupportedLanguages());
  
  // 测试增量解析
  console.log('\n5. Testing incremental parsing...');
  const oldCode = 'var x = 1; function test() { return x; }';
  const newCode = 'var x = 2; function test() { return x; }'; // 只改变了变量值
  
  try {
    const incrementalResult = await parser.incrementalParse(
      'incremental-test.js',
      oldCode,
      newCode,
      {
        start: { line: 0, column: 8 },
        end: { line: 0, column: 9 }
      }
    );
    console.log(`   ✓ Incremental parsed with ${incrementalResult.errors.length} errors`);
  } catch (error) {
    console.error(`   ✗ Failed incremental parse: ${(error as Error).message}`);
  }
  
  console.log('\n✓ Basic functionality test completed!');
}

// 运行测试
testBasicFunctionality().catch(console.error);