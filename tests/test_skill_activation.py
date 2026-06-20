from textwrap import dedent

import pytest

from agent_runtime.skills.loader import SkillLoader
from agent_runtime.skills.selector import SkillSelector


@pytest.fixture
def skills_root(tmp_path):
    """Create an isolated skills directory for activation tests."""
    rolling = tmp_path / "rolling-backtest"
    rolling.mkdir()
    rolling.joinpath("SKILL.md").write_text(
        dedent(
            """\
            ---
            name: rolling-backtest
            description: Evaluate chronological forecasting backtests and leakage controls.
            allowed_tools: ["run_approved_script"]
            ---
            # Rolling Backtest Procedure
            1. Define the forecasting horizon.
            2. Validate chronological splits.
            """
        ),
        encoding="utf-8",
    )

    seasonal = tmp_path / "seasonal-diagnostics"
    seasonal.mkdir()
    seasonal.joinpath("SKILL.md").write_text(
        dedent(
            """\
            ---
            name: seasonal-diagnostics
            description: Analyze repeated seasonal variance in high-frequency series.
            allowed_tools: ["run_approved_script"]
            ---
            # Seasonal Diagnostics Procedure
            1. Compute autocovariance.
            2. Isolate stable seasonal variance patterns.
            """
        ),
        encoding="utf-8",
    )

    return tmp_path


def test_exact_skill_name_activates_matching_skill(skills_root):
    selector = SkillSelector(SkillLoader(str(skills_root)))

    activated = selector.select_and_activate(
        "Please use rolling-backtest before choosing a forecasting model."
    )

    assert activated is not None
    assert activated.name == "rolling-backtest"
    assert activated.metadata.name == "rolling-backtest"
    assert activated.metadata.allowed_tools == ["run_approved_script"]
    assert "Rolling Backtest Procedure" in activated.full_instructions
    assert "Validate chronological splits" in activated.full_instructions
    assert "Seasonal Diagnostics Procedure" not in activated.full_instructions


def test_activation_is_case_insensitive_for_skill_name(skills_root):
    selector = SkillSelector(SkillLoader(str(skills_root)))

    activated = selector.select_and_activate("Run SEASONAL-DIAGNOSTICS on this series.")

    assert activated is not None
    assert activated.name == "seasonal-diagnostics"
    assert "Seasonal Diagnostics Procedure" in activated.full_instructions


def test_description_keyword_can_activate_skill(skills_root):
    selector = SkillSelector(SkillLoader(str(skills_root)))

    activated = selector.select_and_activate(
        "Check whether these results have repeated seasonal behavior."
    )

    assert activated is not None
    assert activated.name == "seasonal-diagnostics"
    assert "Isolate stable seasonal variance patterns" in activated.full_instructions


def test_fuzzy_intent_can_activate_without_skill_name(skills_root):
    selector = SkillSelector(SkillLoader(str(skills_root)))

    activated = selector.select_and_activate(
        "Can you double check whether my chronological splits suffer from information leakage?"
    )

    assert activated is not None
    assert activated.name == "rolling-backtest"
    assert "Rolling Backtest Procedure" in activated.full_instructions
    assert "Validate chronological splits" in activated.full_instructions


def test_unmatched_query_does_not_activate_or_load_full_skill(skills_root):
    class SpyLoader(SkillLoader):
        def __init__(self, root):
            super().__init__(root)
            self.load_count = 0

        def load_full_skill(self, meta):
            self.load_count += 1
            return super().load_full_skill(meta)

    loader = SpyLoader(str(skills_root))
    selector = SkillSelector(loader)

    activated = selector.select_and_activate("Summarize this customer support ticket.")

    assert activated is None
    assert loader.load_count == 0
