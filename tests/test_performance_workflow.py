"""性能测试 CI/CD 工作流验证

验证 GitHub Actions 工作流配置。
"""

import re
from pathlib import Path

import pytest
import yaml


class TestPerformanceWorkflow:
    """测试性能测试工作流配置"""

    @pytest.fixture
    def workflow_file(self):
        """工作流文件路径"""
        return Path(__file__).parent.parent / ".github" / "workflows" / "performance.yml"

    @pytest.fixture
    def workflow_data(self, workflow_file):
        """加载工作流数据"""
        with open(workflow_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_workflow_file_exists(self, workflow_file):
        """测试工作流文件存在"""
        assert workflow_file.exists(), f"Workflow file not found: {workflow_file}"

    def test_workflow_name(self, workflow_data):
        """测试工作流名称"""
        assert workflow_data.get("name") == "Performance Tests"

    def test_workflow_triggers(self, workflow_file):
        """测试工作流触发器"""
        # 直接读取 YAML 内容，避免 Python 关键字问题
        with open(workflow_file, "r", encoding="utf-8") as f:
            content = f.read()

        # 检查定时触发
        assert "schedule:" in content, "Missing schedule trigger"
        assert "cron:" in content, "Missing cron expression"
        assert "0 2 * * *" in content, "Wrong cron schedule"

        # 检查手动触发
        assert "workflow_dispatch:" in content, "Missing workflow_dispatch trigger"

    def test_workflow_jobs(self, workflow_data):
        """测试工作流任务"""
        jobs = workflow_data.get("jobs", {})

        # 检查主要任务
        assert "performance-test" in jobs, "Missing performance-test job"
        assert "performance-trend" in jobs, "Missing performance-trend job"

    def test_performance_test_job(self, workflow_data):
        """测试性能测试任务配置"""
        job = workflow_data["jobs"]["performance-test"]

        # 检查运行环境
        assert job.get("runs-on") == "ubuntu-latest"

        # 检查超时设置
        assert job.get("timeout-minutes") == 30

        # 检查服务配置
        services = job.get("services", {})
        assert "surrealdb" in services, "Missing surrealdb service"
        assert "meilisearch" in services, "Missing meilisearch service"

    def test_services_configuration(self, workflow_data):
        """测试服务配置"""
        services = workflow_data["jobs"]["performance-test"]["services"]

        # 检查 SurrealDB 配置
        surrealdb = services["surrealdb"]
        assert surrealdb.get("image") == "surrealdb/surrealdb:latest"
        assert "ports" in surrealdb
        assert "health-cmd" in surrealdb.get("options", "")

        # 检查 Meilisearch 配置
        meilisearch = services["meilisearch"]
        assert meilisearch.get("image") == "getmeili/meilisearch:latest"
        assert "ports" in meilisearch

    def test_test_steps(self, workflow_data):
        """测试执行步骤"""
        steps = workflow_data["jobs"]["performance-test"]["steps"]

        step_names = [step.get("name", "") for step in steps]

        # 检查关键步骤
        assert any("Checkout" in name for name in step_names), "Missing checkout step"
        assert any("Python" in name for name in step_names), "Missing Python setup step"
        assert any("performance" in name.lower() for name in step_names), "Missing performance test step"
        assert any("Upload" in name for name in step_names), "Missing upload step"

    def test_artifact_upload(self, workflow_data):
        """测试报告上传配置"""
        steps = workflow_data["jobs"]["performance-test"]["steps"]

        upload_step = None
        for step in steps:
            if step.get("name") == "Upload performance reports":
                upload_step = step
                break

        assert upload_step is not None, "Missing upload step"
        assert upload_step.get("if") == "always()", "Upload should run always"

        with_clause = upload_step.get("with", {})
        assert "name" in with_clause, "Missing artifact name"
        assert "path" in with_clause, "Missing artifact path"

    def test_environment_variables(self, workflow_data):
        """测试环境变量配置"""
        env = workflow_data.get("env", {})
        assert "TEST_MODE" in env, "Missing TEST_MODE environment variable"

    def test_yaml_syntax(self, workflow_file):
        """测试 YAML 语法"""
        # 读取原始内容
        with open(workflow_file, "r", encoding="utf-8") as f:
            content = f.read()

        # 验证可以解析
        data = yaml.safe_load(content)
        assert data is not None, "YAML parsing failed"

        # 检查关键语法元素
        assert "name:" in content, "Missing name field"
        assert "on:" in content, "Missing on field"
        assert "jobs:" in content, "Missing jobs field"


class TestWorkflowIntegration:
    """测试工作流集成"""

    def test_ci_workflow_exists(self):
        """测试 CI 工作流文件存在"""
        ci_file = Path(__file__).parent.parent / ".github" / "workflows" / "ci.yml"
        assert ci_file.exists(), "CI workflow file not found"

    def test_performance_scripts_exist(self):
        """测试性能测试脚本存在"""
        perf_dir = Path(__file__).parent.parent / "tests" / "performance"

        # 检查关键脚本
        assert (perf_dir / "benchmark.py").exists(), "benchmark.py not found"
        assert (perf_dir / "run_performance_tests.py").exists(), "run_performance_tests.py not found"

    def test_performance_test_scripts(self):
        """测试性能测试脚本可导入"""
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent / "tests" / "performance"))

        # 尝试导入模块
        try:
            from benchmark import PerformanceBenchmark

            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import benchmark module: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
