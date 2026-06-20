# tests/test_skill_loader.py
from agent_runtime.skills.loader import SkillLoader
from agent_runtime.skills.selector import SkillSelector


def test_discovery_stage_only_metadata():
    """Discovery should read only lightweight skill metadata, not full SOP bodies."""
    loader = SkillLoader("skills")
    catalog = loader.discover_catalog()

    assert len(catalog) >= 2
    names = [m.name for m in catalog]
    assert "rolling-backtest" in names
    assert "seasonal-diagnostics" in names

    assert not hasattr(catalog[0], "full_instructions")


def test_progressive_disclosure_activation():
    """The full skill body should be loaded only after selector activation."""
    loader = SkillLoader("skills")
    selector = SkillSelector(loader)

    query = "Please help me run a rolling-backtest for ARIMA"
    activated = selector.select_and_activate(query)

    assert activated is not None
    assert activated.name == "rolling-backtest"
    assert "Core Execution Steps" in activated.full_instructions
    assert "Define the clear forecasting horizon" in activated.full_instructions


def test_semantic_skill_fallback_and_activation():
    """A fuzzy query should activate rolling-backtest without naming the skill."""
    loader = SkillLoader("skills")
    selector = SkillSelector(loader)

    fuzzy_query = "Can you check my chronological simulation rows to ensure there is no future information leakage?"

    activated = selector.select_and_activate(fuzzy_query)

    assert activated is not None
    assert activated.name == "rolling-backtest"
    assert "Core Execution Steps" in activated.full_instructions
    assert "Validate that chronological row splitting" in activated.full_instructions


def test_semantic_skill_seasonal_diagnostics():
    """A fuzzy query should activate seasonal-diagnostics."""
    loader = SkillLoader("skills")
    selector = SkillSelector(loader)

    fuzzy_query = "Please analyze the repetitive variance and components in our high-frequency series."

    activated = selector.select_and_activate(fuzzy_query)

    assert activated is not None
    assert activated.name == "seasonal-diagnostics"
