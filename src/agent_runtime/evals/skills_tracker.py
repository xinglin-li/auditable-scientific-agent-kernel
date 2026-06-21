# src/agent_runtime/evals/skills_tracker.py
from typing import Dict, List, Any
from agent_runtime.evals.models import TrialResult, GraderResult

class SkillsRegressionGuard:
    """Regression gate that protects previously acquired skills."""
    
    def __init__(self, golden_skills_baseline: Dict[str, float] = None):
        # Define core research skills and their minimum baseline pass rates.
        self.baseline_skills = golden_skills_baseline or {
            "skill_rolling_backtest": 1.0,      # The backtest engine must remain fully stable.
            "skill_seasonal_diagnostics": 1.0   # Required seasonal-diagnostics pass rate.
        }

    def verify_skills_integrity(self, candidate_trial_results: List[TrialResult]) -> Dict[str, Any]:
        """
        Audit whether the candidate retained established core skills.
        """
        skill_scores = {k: 0.0 for k in self.baseline_skills.keys()}
        skill_counts = {k: 0 for k in self.baseline_skills.keys()}
        
        # Scan trials for tasks labeled as core-skill tests.
        for trial in candidate_trial_results:
            # Identify skill regression cases from the task_id namespace.
            for skill_name in self.baseline_skills.keys():
                if skill_name in trial.task_id:
                    skill_counts[skill_name] += 1
                    is_passed = all(g.passed for g in trial.grader_results) if trial.grader_results else (trial.status == "completed")
                    if is_passed:
                        skill_scores[skill_name] += 1.0
                        
        report = {"integrity_passed": True, "details": {}}
        
        for skill_name, baseline_rate in self.baseline_skills.items():
            count = skill_counts[skill_name]
            if count == 0:
                # Missing coverage for an established skill is itself a critical risk.
                report["integrity_passed"] = False
                report["details"][skill_name] = "CRITICAL: the evaluation matrix has no test case for this core skill."
                continue
                
            actual_rate = skill_scores[skill_name] / count
            passed = actual_rate >= baseline_rate
            
            report["details"][skill_name] = {
                "baseline_rate": baseline_rate,
                "actual_rate": actual_rate,
                "passed": passed
            }
            
            if not passed:
                # Fail the hard gate when a patch regresses an established skill.
                report["integrity_passed"] = False
                
        return report
