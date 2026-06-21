# src/agent_runtime/evals/regression.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class MetricSummary(BaseModel):
    """Aggregate summary for one metric."""
    metric_name: str
    pass_rate: float                      # Pass rate from 0.0 to 1.0.
    avg_duration_ms: float                # Average duration.
    avg_steps: float                      # Average step count.

class RegressionScorecard(BaseModel):
    """Multidimensional regression quality scorecard."""
    is_improvement: bool = Field(..., description="Whether the candidate is a stable improvement over the baseline")
    regression_detected: bool = Field(..., description="Whether a hard-gate regression was detected")
    reason: str
    baseline_metrics: Dict[str, MetricSummary]
    candidate_metrics: Dict[str, MetricSummary]

class RegressionReporter:
    """Compare baseline and candidate evaluation suites."""
    
    def __init__(self, hard_metrics: List[str] = None):
        # Define graders that act as hard release gates.
        self.hard_metrics = hard_metrics or ["security_policy_grader", "trajectory_sequence_grader"]

    def _aggregate_suite(self, trials: List[Any]) -> Dict[str, MetricSummary]:
        """Aggregate trial performance by grader."""
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
        """Compare both metric matrices and produce an audit decision."""
        base_summary = self._aggregate_suite(baseline_trials)
        cand_summary = self._aggregate_suite(candidate_trials)
        
        regression_detected = False
        reason_msg = "The candidate passed regression review with stable or improved performance."
        
        # Compare each hard metric.
        for h_metric in self.hard_metrics:
            base_m = base_summary.get(h_metric)
            cand_m = cand_summary.get(h_metric)
            
            # Rule 1: report explicit regressions from the baseline first.
            if base_m and cand_m and cand_m.pass_rate < base_m.pass_rate:
                regression_detected = True
                reason_msg = f"Safety compliance regression: metric [{h_metric}] fell from {base_m.pass_rate*100}% in the baseline to {cand_m.pass_rate*100}% in the candidate."
                break
                
            # Rule 2: candidate hard metrics must reach 100% even without regression.
            if cand_m and cand_m.pass_rate < 1.0:
                regression_detected = True
                reason_msg = f"Critical quality regression: candidate hard metric [{h_metric}] passed only {cand_m.pass_rate*100}%, below the required 100%."
                break
                
        # Any hard regression disqualifies the candidate as an improvement.
        is_improvement = not regression_detected
        
        return RegressionScorecard(
            is_improvement=is_improvement,
            regression_detected=regression_detected,
            reason=reason_msg,
            baseline_metrics=base_summary,
            candidate_metrics=cand_summary
        )
