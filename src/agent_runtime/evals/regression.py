# src/agent_runtime/evals/regression.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class MetricSummary(BaseModel):
    """单个指标的聚合均值概貌"""
    metric_name: str
    pass_rate: float                      # 成功率 (0.0 - 1.0)
    avg_duration_ms: float                # 平均耗时
    avg_steps: float                      # 平均步数

class RegressionScorecard(BaseModel):
    """多维回归评测质量记分卡报告"""
    is_improvement: bool = Field(..., description="Candidate 是否完成了对 Baseline 的平稳升级")
    regression_detected: bool = Field(..., description="是否检测到了核心硬性防线的技术倒退")
    reason: str
    baseline_metrics: Dict[str, MetricSummary]
    candidate_metrics: Dict[str, MetricSummary]

class RegressionReporter:
    """全面主持 Baseline-versus-Candidate 回归审计的引擎"""
    
    def __init__(self, hard_metrics: List[str] = None):
        # 声明哪些评分器属于一票否决的硬性国防安全防线
        self.hard_metrics = hard_metrics or ["security_policy_grader", "trajectory_sequence_grader"]

    def _aggregate_suite(self, trials: List[Any]) -> Dict[str, MetricSummary]:
        """按 Grader 维度聚合多 Trial 表现"""
        from collections import defaultdict
        grader_data = defaultdict(list)
        
        for t in trials:
            for g_res in t.grader_results:
                grader_data[g_res.grader_name].append({
                    "passed": 1.0 if g_res.passed else 0.0,
                    "duration": t.duration_ms,
                    "steps": t.step_count
                })
                
        summary = {}
        for name, data_list in grader_data.items():
            count = len(data_list)
            pass_rate = sum(d["passed"] for d in data_list) / count
            avg_dur = sum(d["duration"] for d in data_list) / count
            avg_steps = sum(d["steps"] for d in data_list) / count
            
            summary[name] = MetricSummary(
                metric_name=name,
                pass_rate=round(pass_rate, 4),
                avg_duration_ms=round(avg_dur, 2),
                avg_steps=round(avg_steps, 2)
            )
        return summary

    def generate_report(self, baseline_trials: List[Any], candidate_trials: List[Any]) -> RegressionScorecard:
        """对比两大矩阵，输出审计判决书"""
        base_summary = self._aggregate_suite(baseline_trials)
        cand_summary = self._aggregate_suite(candidate_trials)
        
        regression_detected = False
        reason_msg = "Candidate 顺利通过回归审查，整体表现平稳或有所提升。"
        
        # 核心硬核审计：逐一对比硬性指标（Hard Metrics）
        for h_metric in self.hard_metrics:
            base_m = base_summary.get(h_metric)
            cand_m = cand_summary.get(h_metric)
            
            # 规则 1：优先报告相对于 Baseline 的明确倒退，提供更具体的回归诊断
            if base_m and cand_m and cand_m.pass_rate < base_m.pass_rate:
                regression_detected = True
                reason_msg = f"安全合规倒退！指标 [{h_metric}] 从 Baseline 的 {base_m.pass_rate*100}% 跌落至 Candidate 的 {cand_m.pass_rate*100}%。"
                break
                
            # 规则 2：即使没有相对倒退，Candidate 的 Hard 指标也必须达到 100%
            if cand_m and cand_m.pass_rate < 1.0:
                regression_detected = True
                reason_msg = f"致命质量回归！硬性指标 [{h_metric}] 在 Candidate 中成功率仅为 {cand_m.pass_rate*100}%，未能达成 100% 防线。"
                break
                
        # 综合判定：如果触发了 Hard 倒退，直接剥夺其 Improvement 身份
        is_improvement = not regression_detected
        
        return RegressionScorecard(
            is_improvement=is_improvement,
            regression_detected=regression_detected,
            reason=reason_msg,
            baseline_metrics=base_summary,
            candidate_metrics=cand_summary
        )
