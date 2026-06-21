# src/agent_runtime/scientific/modelspec.py
from typing import Literal, List, Union, Tuple, Optional
from pydantic import BaseModel, Field

class SeriesRef(BaseModel):
    """时间序列数据集元数据引用约束"""
    dataset_id: str
    column: str
    frequency: Literal["daily", "weekly", "monthly", "quarterly"]

class TransformationSpec(BaseModel):
    """数据前置平稳化转换算子规约"""
    kind: Literal["none", "log", "difference", "seasonal_difference"]
    order: int = Field(default=0, ge=0, le=3)
    seasonal_period: Optional[int] = Field(default=None, gt=0)
    
class BacktestSpec(BaseModel):
    """工业级滚动回测窗口设计"""
    horizon: int = Field(gt=0, le=60, description="预测步长")
    initial_window: int = Field(gt=0, description="初始训练观测样本量")
    step_size: int = Field(gt=0, description="窗口前移步长")
    metrics: List[Literal["mae", "rmse", "mape"]]

class SeasonalNaiveSpec(BaseModel):
    """季节性天真模型规约"""
    family: Literal["seasonal_naive"] = "seasonal_naive"
    seasonal_period: int = Field(gt=0)

class ETSSpec(BaseModel):
    """状态空间误差趋势季节模型(ETS)规约"""
    family: Literal["ets"] = "ets"
    trend: Literal["none", "additive"]
    seasonal: Literal["none", "additive"]
    seasonal_period: Optional[int] = Field(default=None, gt=0)

class ARIMASpec(BaseModel):
    """自回归积分滑动平均模型(ARIMA)规格界定"""
    family: Literal["arima"] = "arima"
    order: Tuple[int, int, int] = Field(description="(p, d, q) 确定性阶数")
    seasonal_order: Optional[Tuple[int, int, int, int]] = Field(default=None, description="(P, D, Q, s)")

class ModelSpec(BaseModel):
    """
    量化内核核心 IR (Intermediate Representation)
    作为模糊自然语言向底层受控执行管道流转的唯一事实来源
    """
    spec_version: str = "1.0"
    target: SeriesRef
    transformations: List[TransformationSpec] = Field(default_factory=list)
    # 辨识联合体：通过 family 字段自动分流解析正确的子模型参数
    model: Union[SeasonalNaiveSpec, ETSSpec, ARIMASpec] = Field(..., discriminator="family")
    backtest: BacktestSpec
    forecast_horizon: int = Field(gt=0, le=60)
    require_human_approval: bool = False
    rationale: str = Field(..., description="模型选型和参数设定的经济学/统计学直觉理由")