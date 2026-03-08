#!/bin/bash
# 端到端测试执行脚本

set -e

echo "=========================================="
echo "端到端测试执行脚本"
echo "=========================================="
echo ""

# 检查服务状态
echo "1. 检查服务状态..."
echo ""

check_service() {
    local url=$1
    local name=$2
    if curl -s -f "$url" > /dev/null 2>&1; then
        echo "✅ $name 运行中"
        return 0
    else
        echo "❌ $name 未运行"
        return 1
    fi
}

all_services_running=true
check_service "http://localhost:18000/health" "Embedding服务" || all_services_running=false
check_service "http://localhost:18001/health" "LLM服务" || all_services_running=false
check_service "http://localhost:3001/health" "包装层服务" || all_services_running=false

echo ""

if [ "$all_services_running" = false ]; then
    echo "⚠️  部分服务未运行。请先启动所有服务："
    echo "   uv run python start_services.py --with-llm"
    echo ""
    read -p "是否继续运行测试？(y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "2. 运行测试..."
echo ""

# 运行基础功能测试
echo "📋 基础功能测试..."
uv run pytest tests/test_embedding_service.py tests/test_llm_service.py tests/test_wrapper_service.py -v --tb=short

# 运行扩展测试
echo ""
echo "📋 扩展测试（边界条件和错误处理）..."
uv run pytest tests/test_embedding_service_extended.py tests/test_llm_service_extended.py tests/test_wrapper_service_extended.py -v --tb=short

# 运行性能测试
echo ""
echo "📋 性能测试..."
uv run pytest tests/test_performance.py -v --tb=short

# 运行安全测试
echo ""
echo "📋 安全测试..."
uv run pytest tests/test_security.py -v --tb=short

# 运行集成测试
echo ""
echo "📋 集成测试..."
uv run pytest tests/test_integration.py -v --tb=short

echo ""
echo "=========================================="
echo "✅ 所有测试执行完成！"
echo "=========================================="
