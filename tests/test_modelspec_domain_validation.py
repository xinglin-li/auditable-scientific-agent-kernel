# tests/test_modelspec_domain_validation.py
import pytest
from pydantic import ValidationError
import json
from agent_runtime.scientific.modelspec import ModelSpec
from agent_runtime.scientific.parser import ModelSpecParser
from agent_runtime.scientific.validators import ModelSpecDomainValidator, DomainValidationError

# 构建本地模拟物理数据集仓库存根元数据：{ dataset_id: (总行数, 时间频率) }
MOCK_DATA_REGISTRY = {
    "monthly_sales_ok.csv": (120, "monthly"),
    "short_series_broken.csv": (15, "monthly") # 只有 15 行的残缺数据
}

@pytest.fixture
def base_valid_arima_payload():
    """提供一个完美通过 Schema 和 Domain 的 ARIMA(1,1,1) 标准参数字典"""
    return {
        "spec_version": "1.0",
        "target": {
            "dataset_id": "monthly_sales_ok.csv",
            "column": "revenue",
            "frequency": "monthly"
        },
        "transformations": [
            {"kind": "log", "order": 0}
        ],
        "model": {
            "family": "arima",
            "order": [1, 1, 1],
            "seasonal_order": [0, 0, 0, 12]
        },
        "backtest": {
            "horizon": 6,
            "initial_window": 60,
            "step_size": 3,
            "metrics": ["rmse", "mae"]
        },
        "forecast_horizon": 6,
        "rationale": "基于数据的月度季节性表现，选用常规稳定 ARIMA 管道进行基线拟合。"
    }

def test_happy_path_modelspec_compiles(base_valid_arima_payload):
    """测试标准合法规格在全链路解析与领域审查中的畅通流转"""
    # 1. 语法与 Schema 转化放行
    spec = ModelSpecParser.parse_deterministic_json(
        import_json_string(base_valid_arima_payload)
    )
    assert spec.model.family == "arima"
    assert spec.model.order == (1, 1, 1)
    
    # 2. 领域数学校验放行
    is_ok = ModelSpecDomainValidator.validate_spec(spec, MOCK_DATA_REGISTRY)
    assert is_ok is True

def test_schema_level_violation_raises(base_valid_arima_payload):
    """故障注入 1：注入非法的家族模型代号，属于一阶一网 Schema 拦截范围"""
    bad_payload = base_valid_arima_payload.copy()
    bad_payload["model"] = {"family": "deep_learning_lstm_fake_model"} # 捏造不支持的模型
    
    with pytest.raises(ValidationError):
        ModelSpecParser.wrap_llm_structured_candidate(bad_payload)

def test_domain_level_frequency_mismatch(base_valid_arima_payload):
    """故障注入 2：注入错误的频率描述，引发 DomainValidationError 业务级拦截"""
    bad_payload = base_valid_arima_payload.copy()
    bad_payload["target"]["frequency"] = "daily" # 强行说自己是日频，但物理存根实际上是月频
    
    spec = ModelSpec.model_validate(bad_payload)
    with pytest.raises(DomainValidationError, match="数据频率冲突"):
        ModelSpecDomainValidator.validate_spec(spec, MOCK_DATA_REGISTRY)

def test_domain_level_sample_size_exhaustion(base_valid_arima_payload):
    """故障注入 3：指向只有 15 行的短残缺数据集，引发样本量枯竭国防线熔断"""
    bad_payload = base_valid_arima_payload.copy()
    bad_payload["target"]["dataset_id"] = "short_series_broken.csv" # 指向残存源
    
    spec = ModelSpec.model_validate(bad_payload)
    with pytest.raises(DomainValidationError, match="历史观测样本量枯竭"):
        ModelSpecDomainValidator.validate_spec(spec, MOCK_DATA_REGISTRY)

def test_llm_parser_privilege_stripping(base_valid_arima_payload):
    """故障注入 4：对抗大模型企图私自跳过人工审批的越权防御测试"""
    sneaky_payload = base_valid_arima_payload.copy()
    sneaky_payload["require_human_approval"] = False # 模型企图宣称自己完全不需要人看
    
    # Parser 必须暴力干预剥夺其越权声明，强行根据安全策略重设为真
    spec = ModelSpecParser.wrap_llm_structured_candidate(sneaky_payload)
    assert spec.require_human_approval is True

def import_json_string(d: dict) -> str:
    return json.dumps(d)