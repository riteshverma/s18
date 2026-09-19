"""Agent alias map invariants.

The loop, base agent, and planner must agree on alias resolution: every alias
target has to be a real configured agent, and the planner aliases previously
split across two divergent maps must all resolve through the single map.
"""

import yaml
from agents.base_agent import AGENT_ALIASES


def _configured_agents():
    config_path = "config/agent_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return set(yaml.safe_load(f)["agents"].keys())


def test_every_alias_target_is_a_configured_agent():
    configured = _configured_agents()
    unknown = {target for target in AGENT_ALIASES.values() if target not in configured}
    assert not unknown, f"Alias targets not present in agent_config.yaml: {sorted(unknown)}"


def test_previously_divergent_aliases_resolve():
    assert AGENT_ALIASES["SummaryAgent"] == "SummarizerAgent"
    assert AGENT_ALIASES["RagAgent"] == "RetrieverAgent"
    assert AGENT_ALIASES["SummarizationAgent"] == "SummarizerAgent"
    assert AGENT_ALIASES["ResearchAgent"] == "RetrieverAgent"
    assert AGENT_ALIASES["ResponseAgent"] == "FormatterAgent"


def test_no_alias_shadows_a_configured_agent_name():
    configured = _configured_agents()
    shadowed = set(AGENT_ALIASES) & configured
    assert not shadowed, f"Aliases that collide with real agent names: {sorted(shadowed)}"
