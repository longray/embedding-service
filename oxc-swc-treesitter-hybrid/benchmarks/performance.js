// benchmarks/performance.js
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

async function runPerformanceTests() {
  console.log('Running performance tests for Hybrid Parser...\n');

  // 检查构建是否存在
  if (!fs.existsSync(path.join(__dirname, '..', 'dist'))) {
    console.log('Building project...');
    execSync('npm run build', { stdio: 'inherit' });
  }

  // 创建测试文件
  createTestFiles();

  // 准备测试
  const { HybridParser } = require('../dist/index.js');

  const parser = new HybridParser();

  // 测试数据
  const testFiles = [
    { name: 'Large JS File', path: './test-data/large-js-file.js', lang: 'javascript' },
    { name: 'Medium TS File', path: './test-data/medium-ts-file.ts', lang: 'typescript' },
    { name: 'Python File', path: './test-data/python-file.py', lang: 'python' },
    { name: 'Go File', path: './test-data/go-file.go', lang: 'go' },
    { name: 'Java File', path: './test-data/java-file.java', lang: 'java' }
  ];

  console.log('Running parser performance tests...\n');

  for (const testFile of testFiles) {
    if (!fs.existsSync(testFile.path)) {
      console.log(`Skipping ${testFile.name} - file not found`);
      continue;
    }

    const code = fs.readFileSync(testFile.path, 'utf-8');
    console.log(`Testing ${testFile.name} (${code.length} chars)...`);

    // 执行多次测试并计算平均时间
    const iterations = 10;
    let totalTime = 0;

    for (let i = 0; i < iterations; i++) {
      const startTime = process.hrtime.bigint();
      try {
        await parser.parse(testFile.path, code);
        const endTime = process.hrtime.bigint();
        totalTime += Number(endTime - startTime) / 1000000; // Convert to milliseconds
      } catch (error) {
        console.error(`Error parsing ${testFile.path}: ${error.message}`);
        break;
      }
    }

    const avgTime = totalTime / iterations;
    console.log(`  Average parsing time: ${avgTime.toFixed(2)}ms\n`);
  }

  // 测试增量解析性能
  console.log('Testing incremental parsing performance...\n');
  
  const incrementalTestData = fs.readFileSync('./test-data/incremental-test.js', 'utf-8');
  const modifiedData = incrementalTestData.replace('var x = 1;', 'var x = 2;'); // 小幅修改
  
  const delta = {
    start: { line: 0, column: 0 },
    end: { line: 0, column: incrementalTestData.split('\n')[0].length }
  };

  const startTime = process.hrtime.bigint();
  try {
    await parser.incrementalParse('./test-data/incremental-test.js', incrementalTestData, modifiedData, delta);
    const endTime = process.hrtime.bigint();
    const incrementalTime = Number(endTime - startTime) / 1000000;
    console.log(`Incremental parsing time: ${incrementalTime.toFixed(2)}ms\n`);
  } catch (error) {
    console.error(`Error in incremental parsing test: ${error.message}`);
  }

  // 测试语言支持检查性能
  console.log('Testing language detection performance...\n');
  
  const languages = ['javascript', 'typescript', 'python', 'go', 'java', 'jsx', 'tsx'];
  const startDetectTime = process.hrtime.bigint();
  
  languages.forEach(lang => {
    parser.supportsLanguage(lang);
  });
  
  const endDetectTime = process.hrtime.bigint();
  const detectTime = Number(endDetectTime - startDetectTime) / 1000000;
  console.log(`Language detection time for ${languages.length} languages: ${detectTime.toFixed(2)}ms\n`);
}

function createTestFiles() {
  const testDataDir = './test-data';
  if (!fs.existsSync(testDataDir)) {
    fs.mkdirSync(testDataDir);
  }

  // 创建大型JS文件（模拟真实场景）
  const largeJsContent = `
    function fibonacci(n) {
      if (n <= 1) return n;
      return fibonacci(n - 1) + fibonacci(n - 2);
    }

    class Calculator {
      constructor() {
        this.history = [];
      }

      add(a, b) {
        const result = a + b;
        this.history.push({ operation: 'add', operands: [a, b], result });
        return result;
      }

      subtract(a, b) {
        const result = a - b;
        this.history.push({ operation: 'subtract', operands: [a, b], result });
        return result;
      }

      multiply(a, b) {
        const result = a * b;
        this.history.push({ operation: 'multiply', operands: [a, b], result });
        return result;
      }

      divide(a, b) {
        if (b === 0) {
          throw new Error("Division by zero");
        }
        const result = a / b;
        this.history.push({ operation: 'divide', operands: [a, b], result });
        return result;
      }

      getHistory() {
        return this.history;
      }

      clearHistory() {
        this.history = [];
      }
    }

    function complexLogic() {
      let result = 0;
      for (let i = 0; i < 100; i++) {
        if (i % 2 === 0) {
          result += i;
        } else if (i % 3 === 0) {
          result -= i;
        } else {
          result *= 1.1;
        }
        
        if (result > 1000) {
          result = 0;
        }
      }
      return result;
    }

    const calc = new Calculator();
    calc.add(10, 20);
    calc.multiply(5, 6);
    console.log("Calculations completed");
  `.repeat(10); // 重复10次以增加文件大小

  // 创建中等大小的TS文件
  const mediumTsContent = `
    interface User {
      id: number;
      name: string;
      email: string;
      isActive: boolean;
      createdAt: Date;
    }

    type UserRole = 'admin' | 'editor' | 'viewer';

    interface UserRepository {
      findById(id: number): Promise<User | null>;
      findByEmail(email: string): Promise<User | null>;
      create(user: Omit<User, 'id' | 'createdAt'>): Promise<User>;
      update(id: number, updates: Partial<User>): Promise<User>;
      delete(id: number): Promise<void>;
    }

    class InMemoryUserRepository implements UserRepository {
      private users: User[] = [];

      async findById(id: number): Promise<User | null> {
        return this.users.find(u => u.id === id) || null;
      }

      async findByEmail(email: string): Promise<User | null> {
        return this.users.find(u => u.email === email) || null;
      }

      async create(userData: Omit<User, 'id' | 'createdAt'>): Promise<User> {
        const newUser: User = {
          ...userData,
          id: this.users.length + 1,
          createdAt: new Date(),
          isActive: true
        };
        this.users.push(newUser);
        return newUser;
      }

      async update(id: number, updates: Partial<User>): Promise<User> {
        const userIndex = this.users.findIndex(u => u.id === id);
        if (userIndex === -1) {
          throw new Error("User not found");
        }
        this.users[userIndex] = { ...this.users[userIndex], ...updates };
        return this.users[userIndex];
      }

      async delete(id: number): Promise<void> {
        this.users = this.users.filter(u => u.id !== id);
      }
    }

    export class UserService {
      constructor(private userRepository: UserRepository) {}

      async getUserProfile(userId: number): Promise<{ profile: User; role: UserRole }> {
        const user = await this.userRepository.findById(userId);
        if (!user) {
          throw new Error("User not found");
        }
        
        // Determine role based on some logic
        const role: UserRole = userId === 1 ? 'admin' : userId <= 5 ? 'editor' : 'viewer';
        
        return { profile: user, role };
      }
    }
  `;

  // 创建Python文件
  const pythonContent = `
    import asyncio
    import json
    from typing import List, Dict, Optional
    from dataclasses import dataclass
    from abc import ABC, abstractmethod


    @dataclass
    class Employee:
        id: int
        name: str
        department: str
        salary: float
        manager_id: Optional[int] = None


    class EmployeeRepository(ABC):
        @abstractmethod
        async def get_by_id(self, emp_id: int) -> Optional[Employee]:
            pass

        @abstractmethod
        async def get_by_department(self, department: str) -> List[Employee]:
            pass

        @abstractmethod
        async def save(self, employee: Employee) -> None:
            pass


    class InMemoryEmployeeRepository(EmployeeRepository):
        def __init__(self):
            self._employees: Dict[int, Employee] = {}
            self._id_counter = 1

        async def get_by_id(self, emp_id: int) -> Optional[Employee]:
            return self._employees.get(emp_id)

        async def get_by_department(self, department: str) -> List[Employee]:
            return [emp for emp in self._employees.values() if emp.department == department]

        async def save(self, employee: Employee) -> None:
            if employee.id == 0:
                employee.id = self._id_counter
                self._id_counter += 1
            self._employees[employee.id] = employee


    class EmployeeService:
        def __init__(self, repository: EmployeeRepository):
            self._repository = repository

        async def promote_employee(self, emp_id: int, new_dept: str, new_salary: float) -> bool:
            employee = await self._repository.get_by_id(emp_id)
            if not employee:
                return False
            
            employee.department = new_dept
            employee.salary = new_salary
            await self._repository.save(employee)
            return True

        async def get_team_hierarchy(self, manager_id: int) -> List[Dict]:
            team = []
            for emp in self._employees.values():
                if emp.manager_id == manager_id:
                    team.append({
                        'employee': emp,
                        'subordinates': await self.get_team_hierarchy(emp.id)
                    })
            return team


    async def main():
        repo = InMemoryEmployeeRepository()
        service = EmployeeService(repo)
        
        # Add some employees
        emp1 = Employee(0, "Alice", "Engineering", 75000, None)
        emp2 = Employee(0, "Bob", "Engineering", 65000, 1)
        emp3 = Employee(0, "Charlie", "Sales", 55000, 1)
        
        await repo.save(emp1)
        await repo.save(emp2)
        await repo.save(emp3)
        
        print(f"Created {len(repo._employees)} employees")


    if __name__ == "__main__":
        asyncio.run(main())
  `;

  // 创建Go文件
  const goContent = `
    package main

    import (
        "encoding/json"
        "fmt"
        "net/http"
        "sync"
    )

    type Product struct {
        ID          int     \`json:"id"\`
        Name        string  \`json:"name"\`
        Description string  \`json:"description"\`
        Price       float64 \`json:"price"\`
        Category    string  \`json:"category"\`
    }

    type ProductService struct {
        products map[int]*Product
        mutex    sync.RWMutex
        nextID   int
    }

    func NewProductService() *ProductService {
        return &ProductService{
            products: make(map[int]*Product),
            nextID:   1,
        }
    }

    func (ps *ProductService) CreateProduct(product *Product) *Product {
        ps.mutex.Lock()
        defer ps.mutex.Unlock()
        
        product.ID = ps.nextID
        ps.nextID++
        
        ps.products[product.ID] = product
        return product
    }

    func (ps *ProductService) GetProduct(id int) (*Product, bool) {
        ps.mutex.RLock()
        defer ps.mutex.RUnlock()
        
        product, exists := ps.products[id]
        return product, exists
    }

    func (ps *ProductService) GetAllProducts() []*Product {
        ps.mutex.RLock()
        defer ps.mutex.RUnlock()
        
        products := make([]*Product, 0, len(ps.products))
        for _, product := range ps.products {
            products = append(products, product)
        }
        return products
    }

    func (ps *ProductService) UpdateProduct(id int, updates map[string]interface{}) (*Product, bool) {
        ps.mutex.Lock()
        defer ps.mutex.Unlock()
        
        product, exists := ps.products[id]
        if !exists {
            return nil, false
        }
        
        if name, ok := updates["name"].(string); ok {
            product.Name = name
        }
        if desc, ok := updates["description"].(string); ok {
            product.Description = desc
        }
        if price, ok := updates["price"].(float64); ok {
            product.Price = price
        }
        if category, ok := updates["category"].(string); ok {
            product.Category = category
        }
        
        return product, true
    }

    func (ps *ProductService) DeleteProduct(id int) bool {
        ps.mutex.Lock()
        defer ps.mutex.Unlock()
        
        _, exists := ps.products[id]
        if !exists {
            return false
        }
        
        delete(ps.products, id)
        return true
    }

    func (ps *ProductService) SetupRoutes(mux *http.ServeMux) {
        mux.HandleFunc("/products", ps.handleProducts)
        mux.HandleFunc("/products/", ps.handleProduct)
    }

    func (ps *ProductService) handleProducts(w http.ResponseWriter, r *http.Request) {
        switch r.Method {
        case "GET":
            products := ps.GetAllProducts()
            json.NewEncoder(w).Encode(products)
        case "POST":
            var product Product
            if err := json.NewDecoder(r.Body).Decode(&product); err != nil {
                http.Error(w, err.Error(), http.StatusBadRequest)
                return
            }
            
            created := ps.CreateProduct(&product)
            w.WriteHeader(http.StatusCreated)
            json.NewEncoder(w).Encode(created)
        default:
            http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        }
    }

    func (ps *ProductService) handleProduct(w http.ResponseWriter, r *http.Request) {
        if r.Method != "GET" && r.Method != "PUT" && r.Method != "DELETE" {
            http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
            return
        }
        
        id := r.URL.Path[len("/products/"):]
        fmt.Fprintf(w, "Product ID: %s", id)
    }

    func main() {
        productService := NewProductService()
        
        mux := http.NewServeMux()
        productService.SetupRoutes(mux)
        
        fmt.Println("Server starting on :8080...")
        http.ListenAndServe(":8080", mux)
    }
  `;

  // 创建Java文件
  const javaContent = `
    import java.util.*;
    import java.util.concurrent.ConcurrentHashMap;
    import java.time.LocalDateTime;
    import java.time.format.DateTimeFormatter;

    interface TaskRepository {
        Optional<Task> findById(Long id);
        List<Task> findAllByStatus(TaskStatus status);
        Task save(Task task);
        boolean deleteById(Long id);
    }

    enum TaskStatus {
        TODO, IN_PROGRESS, DONE, CANCELLED
    }

    class Task {
        private Long id;
        private String title;
        private String description;
        private TaskStatus status;
        private LocalDateTime createdAt;
        private LocalDateTime updatedAt;
        private User assignedTo;

        public Task(String title, String description) {
            this.id = UUID.randomUUID().getMostSignificantBits() & Long.MAX_VALUE;
            this.title = title;
            this.description = description;
            this.status = TaskStatus.TODO;
            this.createdAt = LocalDateTime.now();
            this.updatedAt = LocalDateTime.now();
        }

        // Getters and setters
        public Long getId() { return id; }
        public void setId(Long id) { this.id = id; }
        public String getTitle() { return title; }
        public void setTitle(String title) { 
            this.title = title; 
            this.updatedAt = LocalDateTime.now();
        }
        public String getDescription() { return description; }
        public void setDescription(String description) { 
            this.description = description; 
            this.updatedAt = LocalDateTime.now();
        }
        public TaskStatus getStatus() { return status; }
        public void setStatus(TaskStatus status) { 
            this.status = status; 
            this.updatedAt = LocalDateTime.now();
        }
        public LocalDateTime getCreatedAt() { return createdAt; }
        public LocalDateTime getUpdatedAt() { return updatedAt; }
        public User getAssignedTo() { return assignedTo; }
        public void setAssignedTo(User assignedTo) { 
            this.assignedTo = assignedTo; 
            this.updatedAt = LocalDateTime.now();
        }

        @Override
        public String toString() {
            return "Task{" +
                    "id=" + id +
                    ", title='" + title + '\'' +
                    ", description='" + description + '\'' +
                    ", status=" + status +
                    ", createdAt=" + createdAt +
                    ", updatedAt=" + updatedAt +
                    ", assignedTo=" + assignedTo +
                    '}';
        }
    }

    class User {
        private Long id;
        private String name;
        private String email;

        public User(String name, String email) {
            this.id = UUID.randomUUID().getMostSignificantBits() & Long.MAX_VALUE;
            this.name = name;
            this.email = email;
        }

        // Getters and setters
        public Long getId() { return id; }
        public void setId(Long id) { this.id = id; }
        public String getName() { return name; }
        public void setName(String name) { this.name = name; }
        public String getEmail() { return email; }
        public void setEmail(String email) { this.email = email; }

        @Override
        public String toString() {
            return "User{" +
                    "id=" + id +
                    ", name='" + name + '\'' +
                    ", email='" + email + '\'' +
                    '}';
        }
    }

    public class InMemoryTaskRepository implements TaskRepository {
        private final Map<Long, Task> tasks = new ConcurrentHashMap<>();
        private final DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

        @Override
        public Optional<Task> findById(Long id) {
            return Optional.ofNullable(tasks.get(id));
        }

        @Override
        public List<Task> findAllByStatus(TaskStatus status) {
            return tasks.values().stream()
                    .filter(task -> task.getStatus() == status)
                    .collect(ArrayList::new, (list, item) -> list.add(item), (list1, list2) -> list1.addAll(list2));
        }

        @Override
        public Task save(Task task) {
            if (task.getId() == null) {
                task.setId(UUID.randomUUID().getMostSignificantBits() & Long.MAX_VALUE);
            }
            tasks.put(task.getId(), task);
            return task;
        }

        @Override
        public boolean deleteById(Long id) {
            return tasks.remove(id) != null;
        }

        public static void main(String[] args) {
            InMemoryTaskRepository repo = new InMemoryTaskRepository();

            // Create sample tasks
            User user = new User("John Doe", "john@example.com");
            Task task1 = new Task("Implement login", "Create a secure login functionality");
            task1.setAssignedTo(user);
            
            Task task2 = new Task("Design UI", "Create a user-friendly interface");
            task2.setStatus(TaskStatus.IN_PROGRESS);
            
            repo.save(task1);
            repo.save(task2);

            System.out.println("Created " + repo.tasks.size() + " tasks");
            System.out.println("Tasks in progress: " + repo.findAllByStatus(TaskStatus.IN_PROGRESS).size());
        }
    }
  `;

  // 创建增量测试文件
  const incrementalContent = `
    var x = 1;
    var y = 2;
    var z = x + y;
    
    function add(a, b) {
        return a + b;
    }
    
    function subtract(a, b) {
        return a - b;
    }
    
    var result = add(x, y);
  `;

  // 写入测试文件
  fs.writeFileSync('./test-data/large-js-file.js', largeJsContent);
  fs.writeFileSync('./test-data/medium-ts-file.ts', mediumTsContent);
  fs.writeFileSync('./test-data/python-file.py', pythonContent);
  fs.writeFileSync('./test-data/go-file.go', goContent);
  fs.writeFileSync('./test-data/java-file.java', javaContent);
  fs.writeFileSync('./test-data/incremental-test.js', incrementalContent);

  console.log('Test files created successfully.');
}

if (require.main === module) {
  runPerformanceTests().catch(console.error);
}

module.exports = { runPerformanceTests };