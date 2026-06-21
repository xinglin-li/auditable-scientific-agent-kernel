# src/agent_runtime/scientific/validators.py
import logging
from typing import Dict, Any, Tuple
from agent_runtime.scientific.modelspec import ModelSpec

logger = logging.getLogger("modelspec-validator")

class DomainValidationError(Exception):
    """专门抛出不满足计量经济学或物理样本要求的领域异常"""
    pass

class ModelSpecDomainValidator:
    """双层防御体系之：确定性统计学与业务领域逻辑校验引擎"""
    
    @staticmethod
    def validate_spec(spec: ModelSpec, mock_dataset_registry: Dict[str, Tuple[int, str]]) -> bool:
        """
        传入拟审查的 ModelSpec 与本地数据集元数据存根(包含真实样本行数与数据频率)
        """
        dataset_id = spec.target.dataset_id
        
        # 1. 深度交叉核对数据集存在性与时间频率
        if dataset_id not in mock_dataset_registry:
            raise DomainValidationError(f"数据源校验拦截：指定的数据集 ID '{dataset_id}' 在本地仓库中不存在。")
            
        actual_rows, actual_freq = mock_dataset_registry[dataset_id]
        if spec.target.frequency != actual_freq:
            raise DomainValidationError(
                f"数据频率冲突：分析指令声称数据为 [{spec.target.frequency}] 频率，"
                f"但物理文件实际为 [{actual_freq}]。"
            )
            
        # 2. 校验 ETS 模型的数学完备性
        if spec.model.family == "ets":
            if spec.model.seasonal != "none" and not spec.model.seasonal_period:
                raise DomainValidationError("ETS模型冲突：当激活季节性算子时，必须显式指定非空的 seasonal_period 周期数。")

        # 3. 校验 ARIMA 阶数的合法性
        if spec.model.family == "arima":
            p, d, q = spec.model.order
            if p < 0 or d < 0 or q < 0:
                raise DomainValidationError("ARIMA参数失效：(p, d, q) 阶数矩阵绝对不允许出现负数值。")
            
            # 计算非季节性参数总量
            param_complexity = p + q
            if param_complexity > 12:
                raise DomainValidationError(
                    f"模型复杂度超限：非季节性参数阶数过高 ({param_complexity})，引发强烈的非收敛与过拟合风险。"
                )

        # 4. 工业级核心防线：样本量与训练窗口的数学限界校验
        required_min_window = spec.backtest.initial_window + spec.backtest.horizon
        if actual_rows < required_min_window:
            raise DomainValidationError(
                f"历史观测样本量枯竭：当前物理数据集总行数仅为 {actual_rows} 行，"
                f"但在滚动回测计划中，初始窗口 ({spec.backtest.initial_window}) "
                f"+ 预测步长 ({spec.backtest.horizon}) 至少需要 {required_min_window} 行可用样本。"
            )
            
        logger.info(f"ModelSpec [{spec.model.family}] 通过所有静态统计学领域审查边界。")
        return True