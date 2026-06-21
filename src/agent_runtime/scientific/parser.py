# src/agent_runtime/scientific/parser.py
import json
from typing import Dict, Any
from agent_runtime.scientific.modelspec import ModelSpec

class ModelSpecParser:
    """Parse raw input into a strongly typed and controlled spec candidate."""
    
    @staticmethod
    def parse_deterministic_json(json_str: str) -> ModelSpec:
        """
        Convert JSON from a research interface into the standard IR without
        relying on a stochastic LLM.
        """
        data = json.loads(json_str)
        return ModelSpec.model_validate(data)

    @staticmethod
    def wrap_llm_structured_candidate(llm_dict_output: Dict[str, Any]) -> ModelSpec:
        """
        Wrap structured model output as an auditable candidate. The resulting
        spec remains untrusted and cannot elevate privileges or skip validation.
        """
        # Reject attempts by the model to alter security or bypass authorization.
        if "require_human_approval" in llm_dict_output:
            # Risk-boundary settings are controlled exclusively by system policy.
            llm_dict_output["require_human_approval"] = True
            
        # Enforce first-pass validation through the Pydantic schema.
        return ModelSpec.model_validate(llm_dict_output)
