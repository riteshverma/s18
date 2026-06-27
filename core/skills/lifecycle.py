from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from core.skills.base import BaseSkill
from core.skills.manager import skill_manager


def _auto_match_allowed(
    integration_id: Optional[str],
    workflow_id: Optional[str],
) -> bool:
    integration = (integration_id or "default").strip().lower()
    workflow = (workflow_id or "generic").strip().lower()
    return integration == "default" and workflow == "generic"


async def resolve_skill(
    *,
    query: str,
    run_id: str,
    agent_id: str,
    explicit_skill_id: Optional[str] = None,
    integration_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
) -> Tuple[Optional[BaseSkill], str, Optional[str]]:
    """
    Resolve and initialize a skill for a run.

    Returns: (skill instance or None, effective query, resolved skill id or None)
    """
    if not skill_manager.registry_file.exists():
        skill_manager.initialize()

    resolved_skill_id = explicit_skill_id
    if not resolved_skill_id and _auto_match_allowed(integration_id, workflow_id):
        resolved_skill_id = skill_manager.match_intent(query)

    if not resolved_skill_id:
        return None, query, None

    skill = skill_manager.get_skill(resolved_skill_id)
    if not skill:
        return None, query, None

    skill.context.run_id = run_id
    skill.context.agent_id = agent_id
    skill.context.config = {"query": query}
    effective_query = await skill.on_run_start(query)
    return skill, effective_query, resolved_skill_id


async def run_skill_success(
    skill: Optional[BaseSkill],
    artifact: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not skill:
        return None
    result = await skill.on_run_success(artifact)
    return result if isinstance(result, dict) else None


async def run_skill_failure(skill: Optional[BaseSkill], error: str) -> None:
    if not skill:
        return
    await skill.on_run_failure(error)
