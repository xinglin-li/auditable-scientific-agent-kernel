# src/agent_runtime/scientific/static_analysis.py
import logging
from typing import List, Dict, Any
from agent_runtime.scientific.modelspec import ModelSpec
from agent_runtime.scientific.validators import Diagnostic

logger = logging.getLogger("static-analyzer")

class StaticAnalyzer:
    """静态分析器：在编译前刺探潜在的时间序列数据泄漏与越权文件路径"""

    @staticmethod
    def analyze_spec(spec: ModelSpec) -> List[Diagnostic]:
        diagnostics: List[Diagnostic] = []
        
        # 1. 对抗未来信息泄漏核心审计 (Future Data Leakage Check)
        # 工业案例：如果用户要求预测未来 6 个月，但 transformations 算子或者Rationale中暗示了使用未来测试集日期
        rationale_lower = spec.rationale.lower()
        if "leak" in rationale_lower or "test_set" in rationale_lower:
            diagnostics.append(Diagnostic(
                stage="static_analysis",
                severity="ERROR",
                code="LEAKAGE_DETECTED",
                message="致命安全拦截：静态分析检测到潜在的未来信息泄漏风险！训练集边界配置与测试集发生跨空间交叠。",
                repairable=False, # 未来泄漏触犯 Policy 红线，禁止系统静默修复，必须阻断升级
                evidence={"rationale_leak_flag": True}
            ))

        # 2. 危险文件路径与注入扫描
        dataset_id = spec.target.dataset_id.lower()
        if "../" in dataset_id or "/etc/" in dataset_id:
            diagnostics.append(Diagnostic(
                stage="static_analysis",
                severity="ERROR",
                code="UNAUTHORIZED_OPERATION",
                message="越界操作审计：数据源路径存在危险的文件穿透路径注入倾向，操作已被强制挂起。",
                repairable=False,
                evidence={"path_injection_attempt": dataset_id}
            ))

        # 3. 统计复杂度与样本承载度静态建议
        if spec.model.family == "arima" and spec.forecast_horizon > spec.backtest.initial_window * 0.5:
            diagnostics.append(Diagnostic(
                stage="static_analysis",
                severity="WARNING",
                code="BACKTEST_WINDOW_INVALID",
                message="投研效能警告：当前预测步长（Horizon）相对初始训练窗口过大，长期外推将导致数学方差剧烈发散。",
                repairable=True,
                evidence={"horizon": spec.forecast_horizon, "initial_window": spec.backtest.initial_window}
            ))

        return diagnostics