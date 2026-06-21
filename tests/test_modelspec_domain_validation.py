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
    Inject frequency mismatch, excessive complexity, and insufficient sample
    failures, then verify that the validator reports all of them.
    """
    # Use 20 monthly rows with a 60-row daily window and ARIMA(10,0,10).
    terrible_payload = {
        "spec_version": "1.0",
        "target": {
            "dataset_id": "short_series_broken.csv", # Triggers INSUFFICIENT_SAMPLE.
            "column": "revenue",
            "frequency": "daily"                    # Triggers FREQUENCY_MISMATCH.
        },
        "transformations": [],
        "model": {
            "family": "arima",
            "order": [10, 0, 10],                   # Triggers MODEL_COMPLEXITY_EXCESSIVE.
            "seasonal_order": None
        },
        "backtest": {
            "horizon": 6,
            "initial_window": 60,
            "step_size": 3,
            "metrics": ["rmse"]
        },
        "forecast_horizon": 6,
        "rationale": "Adversarial specification containing multiple injected errors."
    }
    
    spec = ModelSpec.model_validate(terrible_payload)
    
    # Run the complete audit.
    reports = ModelSpecDomainValidator.validate_spec(spec, MOCK_DATA_REGISTRY)
    
    # The collector must return all three diagnostics.
    assert len(reports) == 3
    
    codes = [d.code for d in reports]
    assert "FREQUENCY_MISMATCH" in codes
    assert "MODEL_COMPLEXITY_EXCESSIVE" in codes
    assert "INSUFFICIENT_SAMPLE" in codes
    
    # Print the compact diagnostic report supplied to the LLM.
    from agent_runtime.scientific.validators import format_diagnostics_for_llm
    final_prompt = format_diagnostics_for_llm(reports)
    print("\n" + final_prompt)
