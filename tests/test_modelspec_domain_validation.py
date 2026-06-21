# tests/test_modelspec_domain_validation.py
import pytest
import json
from agent_runtime.scientific.modelspec import ModelSpec
from agent_runtime.scientific.validators import ModelSpecDomainValidator

MOCK_DATA_REGISTRY = {
    "monthly_sales_ok.csv": (120, "monthly"),
    "short_series_broken.csv": (20, "monthly") 
}

def test_error_accumulation_captures_all_violations():
    """
    故障注入对抗测试：构造一个同时触犯了【频率错配】、【复杂度超限】、【样本量枯竭】三重罪状的恶劣 Spec，
    验证全量累积引擎是否能完整留痕、无一遗漏。
    """
    # 故意指向只有 20 行样本的短数据集，但是要求初始窗口 60；物理是月频，这里瞎填日频；ARIMA 阶数填到 10,0,10 撑爆复杂度
    terrible_payload = {
        "spec_version": "1.0",
        "target": {
            "dataset_id": "short_series_broken.csv", # 导致 INSUFFICIENT_SAMPLE
            "column": "revenue",
            "frequency": "daily"                    # 导致 FREQUENCY_MISMATCH
        },
        "transformations": [],
        "model": {
            "family": "arima",
            "order": [10, 0, 10],                   # 导致 MODEL_COMPLEXITY_EXCESSIVE (10+10=20 > 12)
            "seasonal_order": None
        },
        "backtest": {
            "horizon": 6,
            "initial_window": 60,
            "step_size": 3,
            "metrics": ["rmse"]
        },
        "forecast_horizon": 6,
        "rationale": "恶意注入注入的多重错误规格体测试。"
    }
    
    spec = ModelSpec.model_validate(terrible_payload)
    
    # 运行全量审计
    reports = ModelSpecDomainValidator.validate_spec(spec, MOCK_DATA_REGISTRY)
    
    # 核心断言：错误收集器必须精准斩获 3 条诊断，不能漏掉任何一个！
    assert len(reports) == 3
    
    codes = [d.code for d in reports]
    assert "FREQUENCY_MISMATCH" in codes
    assert "MODEL_COMPLEXITY_EXCESSIVE" in codes
    assert "INSUFFICIENT_SAMPLE" in codes
    
    # 打印查看打包给 LLM 的高浓度诊断报告样式
    from agent_runtime.scientific.validators import format_diagnostics_for_llm
    final_prompt = format_diagnostics_for_llm(reports)
    print("\n" + final_prompt)