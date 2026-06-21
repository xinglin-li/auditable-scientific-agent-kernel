# src/agent_runtime/evals/optimizer.py
from typing import Dict, Any, List
from pydantic import BaseModel
from agent_runtime.evals.failure_episode import FailureEpisode

class CandidatePatch(BaseModel):
    """Patch that modifies agent behavior."""
    patch_id: str
    target_node: str
    injected_instruction: str  # Defensive prompt instruction injected at runtime.

class ReflexionOptimizer:
    """Generate prompt patches from offline failure diagnostics."""
    
    def __init__(self, domain_knowledge_base: Dict[str, str] = None):
        self.knowledge_base = domain_knowledge_base or {
            "max_steps_exceeded_failure": "[LOOP PREVENTION] You have attempted the same tool repeatedly. Stop retrying and return a final answer that explains the limitation if no new data is available.",
            "tool_argument_failure": "[STRICT TYPING] Validate every tool argument against its schema before calling the tool. Never invent arguments."
        }

    def diagnose_and_suggest(self, episode: FailureEpisode) -> CandidatePatch:
        """
        Diagnose root causes from compact structured failure records instead
        of loading the full noisy context.
        """
        ftype = episode.failure_type
        failed_node = episode.failed_node or "agent_core"
        
        # Select the matching defensive strategy.
        instruction = self.knowledge_base.get(
            ftype, 
            "[GENERAL DEFENSE] An unknown runtime failure occurred. Return to a safe node and validate input boundaries."
        )
        
        # Build the dynamic repair patch.
        patch = CandidatePatch(
            patch_id=f"patch_{episode.task_id}_{ftype}",
            target_node=failed_node,
            injected_instruction=instruction
        )
        
        # Persist the suggestion in the episode for auditability.
        episode.candidate_patches.append(patch.injected_instruction)
        episode.root_cause_hypothesis = f"Diagnostic report: detected failure type [{ftype}]. The model failed to converge at node [{failed_node}]."
        
        return patch
