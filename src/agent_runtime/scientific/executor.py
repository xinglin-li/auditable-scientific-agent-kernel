# src/agent_runtime/scientific/executor.py
from typing import Dict, Any, Callable
from agent_runtime.scientific.compiler import ExecutionPlan, ExecutionStep

class DeterministicExecutor:
    """确定性受控运行时：通过注册表机制强行限缩大模型命令边界"""

    def __init__(self):
        # 工业死锁：系统核准的算子程序白名单映射
        self.registry: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
            "load_dataset": lambda p: {"status": "SUCCESS", "rows": 120},
            "apply_transformations": lambda p: {"status": "SUCCESS", "stationarity": "PASSED"},
            "fit_arima": self.execute_fit_arima, # 指向内部具备数学求解诊断的实体方法
            "generate_forecast": lambda p: {"status": "SUCCESS", "horizon_aligned": True}
        }

    def execute_fit_arima(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """精细化模拟计量统计引擎的数值计算边界"""
        model_spec = params.get("model_spec", {})
        p, d, q = model_spec.get("order", (0, 0, 0))
        
        # 故意注入的高阶非收敛数学死穴：如果阶数总和大于 5，底层抛出非正定或不收敛错误桩
        if p + q > 5:
            return {
                "status": "NUMERICAL_FAILURE",
                "error_code": "ARIMA_NON_CONVERGENCE",
                "detail": "数值崩溃：估计的初始 MA 系数不可逆或非平稳。ARIMA 极大似然估计迭代步数耗尽未收敛。"
            }
        return {"status": "SUCCESS", "rmse": 1.15, "converged": True}

    def execute_plan_step(self, step: ExecutionStep) -> Dict[str, Any]:
        op = step.operation
        if op not in self.registry:
            return {
                "status": "FATAL",
                "error_code": "UNAUTHORIZED_OPERATION",
                "detail": f"内核熔断：企图执行系统未经审计的非法算子指令 [{op}]。"
            }
        
        # 交付确定性 handler 回调
        return self.registry[op](step.parameters)