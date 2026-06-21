# src/agent_runtime/scientific/modelspec.py
from typing import Literal, List, Union, Tuple, Optional
from pydantic import BaseModel, Field

class SeriesRef(BaseModel):
    """Reference to time-series dataset metadata."""
    dataset_id: str
    column: str
    frequency: Literal["daily", "weekly", "monthly", "quarterly"]

class TransformationSpec(BaseModel):
    """Preprocessing transformation for stationarity."""
    kind: Literal["none", "log", "difference", "seasonal_difference"]
    order: int = Field(default=0, ge=0, le=3)
    seasonal_period: Optional[int] = Field(default=None, gt=0)
    
class BacktestSpec(BaseModel):
    """Rolling backtest window configuration."""
    horizon: int = Field(gt=0, le=60, description="Forecast horizon")
    initial_window: int = Field(gt=0, description="Number of observations in the initial training window")
    step_size: int = Field(gt=0, description="Rolling-window step size")
    metrics: List[Literal["mae", "rmse", "mape"]]

class SeasonalNaiveSpec(BaseModel):
    """Seasonal-naive model specification."""
    family: Literal["seasonal_naive"] = "seasonal_naive"
    seasonal_period: int = Field(gt=0)

class ETSSpec(BaseModel):
    """Error-trend-seasonal state-space model specification."""
    family: Literal["ets"] = "ets"
    trend: Literal["none", "additive"]
    seasonal: Literal["none", "additive"]
    seasonal_period: Optional[int] = Field(default=None, gt=0)

class ARIMASpec(BaseModel):
    """ARIMA model specification."""
    family: Literal["arima"] = "arima"
    order: Tuple[int, int, int] = Field(description="Deterministic (p, d, q) order")
    seasonal_order: Optional[Tuple[int, int, int, int]] = Field(default=None, description="(P, D, Q, s)")

class ModelSpec(BaseModel):
    """
    Core quantitative intermediate representation and the single source of
    truth between natural-language requests and the controlled pipeline.
    """
    spec_version: str = "1.0"
    target: SeriesRef
    transformations: List[TransformationSpec] = Field(default_factory=list)
    # Discriminated union that selects model parameters from the family field.
    model: Union[SeasonalNaiveSpec, ETSSpec, ARIMASpec] = Field(..., discriminator="family")
    backtest: BacktestSpec
    forecast_horizon: int = Field(gt=0, le=60)
    require_human_approval: bool = False
    rationale: str = Field(..., description="Economic or statistical rationale for model and parameter selection")
