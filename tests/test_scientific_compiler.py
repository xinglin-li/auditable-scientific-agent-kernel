# tests/test_scientific_compiler.py
import pytest
from agent_runtime.scientific.modelspec import ModelSpec
from agent_runtime.scientific.static_analysis import StaticAnalyzer
from agent_runtime.scientific.compiler import ModelSpecCompiler
from agent_runtime.scientific.executor import DeterministicExecutor
from agent_runtime.scientific.repairs import BoundedRepairEngine
from agent_runtime.scientific.validators import Diagnostic

@pytest.fixture
def base_arima_payload():
    """抛出一个高阶拟合要求的原始未编译规格字典"""
    return {
        "spec_version": "1.0",
        "target": {"dataset_id": "monthly_sales.csv", "column": "revenue", "frequency": "monthly"},
        "transformations": [],
        "model": {
            "family": "arima",
            "order": [4, 0, 4], # 4+4=8 > 5, 会在物理执行器内部恶意触发不收敛状态桩
            "seasonal_order": None
        },
        "backtest": {"horizon": 6, "initial_window": 60, "step_size": 3, "metrics": ["rmse"]},
        "forecast_horizon": 6,
        "rationale": "投研指令：对目标时间序列进行全面平稳拟合高阶参数估计。"
    }

def test_static_analyzer_blocks_future_data_leakage(base_arima_payload):
    """测试 1：恶意注入未来信息泄漏字样，分析器必须一箭熔断，且标记不可自动修复"""
    leaked_payload = base_arima_payload.copy()
    leaked_payload["rationale"] = "【恶意的未来泄漏注入】请使用测试集 test_set 包含的未来观测值优化当前估计。"
    
    spec = ModelSpec.model_validate(leaked_payload)
    logs = StaticAnalyzer.analyze_spec(spec)
    
    assert len(logs) == 1
    assert logs[0].code == "LEAKAGE_DETECTED"
    assert logs[0].repairable is False # 红线！禁止自动修

def test_end_to_end_compiler_and_bounded_repair_loop(base_arima_payload):
    """
    测试 2：黄金全生命周期大合流闭环
    高阶不收敛 ➔ 物理执行失败 ➔ 自动降阶 ➔ 再次执行 ➔ 100% 收敛通关，且完整记录 Diff 差分！
    """
    spec = ModelSpec.model_validate(base_arima_payload)
    
    # 实例化编译器、受控运行时与自修复发动机
    compiler = ModelSpecCompiler()
    executor = DeterministicExecutor()
    repair_engine = BoundedRepairEngine(max_attempts=2)
    
    # --- 第一轮执行流流转 ---
    plan = compiler.compile_plan(spec)
    assert len(plan.steps) == 4
    # 拓扑前置依赖核对
    assert plan.steps[2].depends_on == ["step_2_transform"]
    
    # 模拟顺次执行到核心 fit 节点
    fit_step = plan.steps[2]
    res1 = executor.execute_fit_arima(fit_step.parameters)
    
    # 证实第一轮由于高阶（4,0,4）引发统计求解器抛出不收敛诊断
    assert res1["status"] == "NUMERICAL_FAILURE"
    assert res1["error_code"] == "ARIMA_NON_CONVERGENCE"
    
    # 将运行时失败现场自动包装转化为标准 Diagnostic 诊断实体
    diag = Diagnostic(
        stage="execution", severity="ERROR", 
        code=res1["error_code"], message=res1["detail"], repairable=True
    )
    
    # --- 呼叫修复发动机进行一轮自动降阶降容 ---
    fixed_spec = repair_engine.attempt_auto_repair(spec, diag)
    assert fixed_spec is not None
    # 参数已经被强制安全剪枝为 (1, 0, 1)
    assert fixed_spec.model.order == (1, 0, 1) 
    
    # --- 第二轮编译与干净运行时执行 ---
    new_plan = compiler.compile_plan(fixed_spec)
    new_fit_step = new_plan.steps[2]
    res2 = executor.execute_fit_arima(new_fit_step.parameters)
    
    # 降阶后，底层统计引擎成功实现数学收敛！完美通过！
    assert res2["status"] == "SUCCESS"
    assert res2["converged"] is True
    
    # 终极断言：审查差分链中必须有清晰的改参留痕痕迹，彻底消灭静默作弊
    assert len(repair_engine.repair_history) == 1
    diff_log = repair_engine.repair_history[0]
    assert diff_log.proposed_changes["old_order"] == [4, 0, 4]
    assert diff_log.proposed_changes["new_order"] == [1, 0, 1]

def test_repair_exhaustion_escalates_to_human(base_arima_payload):
    """测试 3：恶意构筑一个永远无法收敛的死循环现场，修复发动机达到上限必须熔断升级"""
    spec = ModelSpec.model_validate(base_arima_payload)
    repair_engine = BoundedRepairEngine(max_attempts=1) # 强行限制只能修 1 次
    
    diag = Diagnostic(stage="execution", severity="ERROR", code="ARIMA_NON_CONVERGENCE", message="崩溃", repairable=True)
    
    # 第一修放行
    spec_v2 = repair_engine.attempt_auto_repair(spec, diag)
    assert spec_v2 is not None
    
    # 第二修，配额耗尽，强行触发熔断升级人类机制，返回 None
    spec_v3 = repair_engine.attempt_auto_repair(spec_v2, diag)
    assert spec_v3 is None
    assert len(repair_engine.repair_history) == 1