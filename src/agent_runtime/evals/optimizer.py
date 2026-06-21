# src/agent_runtime/evals/optimizer.py
from typing import Dict, Any, List
from pydantic import BaseModel
from agent_runtime.evals.failure_episode import FailureEpisode

class CandidatePatch(BaseModel):
    """引发智能体行为修正的补丁实体"""
    patch_id: str
    target_node: str
    injected_instruction: str  # 动态注入的防御性 Prompt 指令

class ReflexionOptimizer:
    """自动化离线诊断与 Prompt 补丁生成引擎"""
    
    def __init__(self, domain_knowledge_base: Dict[str, str] = None):
        self.knowledge_base = domain_knowledge_base or {
            "max_steps_exceeded_failure": "【防死循环策略】你已经连续多次尝试相同工具。请立刻停止重复调用，如果无法获取新数据，请直接输出 Final Answer 说明局限性。",
            "tool_argument_failure": "【强类型规约】调用工具前，必须严格核对 Schema 定义的字段类型，禁止捏造参数。"
        }

    def diagnose_and_suggest(self, episode: FailureEpisode) -> CandidatePatch:
        """
        高浓度根因推演：不看海量噪音，只基于结构化失败片段进行精准定点修复
        """
        ftype = episode.failure_type
        failed_node = episode.failed_node or "agent_core"
        
        # 寻找对应的专家防御策略
        instruction = self.knowledge_base.get(
            ftype, 
            "【通用防御】运行时遭遇未知崩溃，请退回到安全节点并检查输入边界。"
        )
        
        # 组装动态自修复补丁
        patch = CandidatePatch(
            patch_id=f"patch_{episode.task_id}_{ftype}",
            target_node=failed_node,
            injected_instruction=instruction
        )
        
        # 将建议回写沉淀到资产实体的候选池中，完成审计留痕
        episode.candidate_patches.append(patch.injected_instruction)
        episode.root_cause_hypothesis = f"诊断报告：检测到类型为 [{ftype}] 的失效。根因在于模型在 [{failed_node}] 节点未能正确收敛。"
        
        return patch