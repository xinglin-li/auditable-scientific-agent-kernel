# src/agent_runtime/scientific/parser.py
import json
from typing import Dict, Any
from agent_runtime.scientific.modelspec import ModelSpec

class ModelSpecParser:
    """规格描述体解析防火墙：将原始输入转化为类型固化的受控 Spec 候选实体"""
    
    @staticmethod
    def parse_deterministic_json(json_str: str) -> ModelSpec:
        """
        确定性基线解析：直接将投研面板或标准化接口抛出的 JSON 转化为标准 IR 实例，不依赖任何随机 LLM
        """
        data = json.loads(json_str)
        return ModelSpec.model_validate(data)

    @staticmethod
    def wrap_llm_structured_candidate(llm_dict_output: Dict[str, Any]) -> ModelSpec:
        """
        将大模型通过 JSON Mode 或是工具调用吐出的结构化原始 Dict 包装为可审计候选实体。
        请牢记：此时的 Spec 依然是 Untrusted Candidate，绝对禁止赋予其提升权限或跳过校验的特权。
        """
        # 拦截模型私自篡改安全等级或跳过授权的企图
        if "require_human_approval" in llm_dict_output:
            # 涉及风险边界的设定，强制由系统 Policy 控制，剥夺 LLM 自主宣告免审批的权利
            llm_dict_output["require_human_approval"] = True
            
        # 强制实施 Pydantic Schema 级别一阶过滤
        return ModelSpec.model_validate(llm_dict_output)