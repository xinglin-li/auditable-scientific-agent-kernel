# src/agent_runtime/evals/skills_tracker.py
from typing import Dict, List, Any
from agent_runtime.evals.models import TrialResult, GraderResult

class SkillsRegressionGuard:
    """工业级技能防遗忘国防网关"""
    
    def __init__(self, golden_skills_baseline: Dict[str, float] = None):
        # 固化历史沉淀的高级投研核心核心技能及预期的最低通过率 (Baseline)
        self.baseline_skills = golden_skills_baseline or {
            "skill_rolling_backtest": 1.0,      # 回测引擎必须 100% 稳定
            "skill_seasonal_diagnostics": 1.0   # 季节性诊断组件通过率
        }

    def verify_skills_integrity(self, candidate_trial_results: List[TrialResult]) -> Dict[str, Any]:
        """
        审计 Candidate 智能体在经典核心技能上的保留情况
        """
        skill_scores = {k: 0.0 for k in self.baseline_skills.keys()}
        skill_counts = {k: 0 for k in self.baseline_skills.keys()}
        
        # 扫描试炼集，寻找打标了基础技能测试的 Task 表现
        for trial in candidate_trial_results:
            # 根据 task_id 的命名空间识别是否属于基础技能回归测试用例
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
                # 如果评测集缺失了对老核心技能的覆盖，本身就是严重的工程隐患
                report["integrity_passed"] = False
                report["details"][skill_name] = "CRITICAL: 评测矩阵缺失了该核心技能的测试用例！"
                continue
                
            actual_rate = skill_scores[skill_name] / count
            passed = actual_rate >= baseline_rate
            
            report["details"][skill_name] = {
                "baseline_rate": baseline_rate,
                "actual_rate": actual_rate,
                "passed": passed
            }
            
            if not passed:
                # 触发硬熔断：新补丁虽然修了新 Bug，但是导致了老技能倒退
                report["integrity_passed"] = False
                
        return report