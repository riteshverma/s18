import re
import yaml
import json
from pathlib import Path
from typing import Optional
from core.model_manager import ModelManager
from core.json_parser import parse_llm_json, parse_llm_json_or_fallback
from core.utils import log_step, log_error
from PIL import Image
from datetime import datetime
import os

# Alias map: planner-invented agent names -> actual agents in agent_config.yaml
# Includes WISE CDSS architecture names so plans can use doc terminology (no refactor needed).
AGENT_ALIASES = {
    # General / planner-invented
    "SearchLabsAgent": "RetrieverAgent",
    "SummarizationAgent": "SummarizerAgent",
    "NoteWriterAgent": "FormatterAgent",
    "NoteWriter": "FormatterAgent",
    "RAG": "RetrieverAgent",
    "QA": "QAAgent",
    "ParserAgent": "DistillerAgent",
    "ReportGeneratorAgent": "FormatterAgent",
    "NameExtractorAgent": "RetrieverAgent",
    "InterpretationAgent": "ThinkerAgent",
    "System": "ThinkerAgent",
    # WISE CDSS architecture names -> existing S18 agents
    "ClinicalReasoningAgent": "ThinkerAgent",
    "ContextSynthesisAgent": "DistillerAgent",
    "ResearchAgent": "RetrieverAgent",
    "SafetyExplainabilityAgent": "ThinkerAgent",
    "ConfidenceScoringAgent": "ThinkerAgent",
    "SymptomAgent": "ThinkerAgent",
    "CBCAgent": "EHRDataMinerAgent",
    "TrendAgent": "EHRDataMinerAgent",
    "ActionAgent": "FormatterAgent",
}


class AgentRunner:
    def __init__(self, multi_mcp):
        self.multi_mcp = multi_mcp
        
        # Load agent configurations
        config_path = Path(__file__).parent.parent / "config/agent_config.yaml"
        with open(config_path, "r") as f:
            self.agent_configs = yaml.safe_load(f)["agents"]

        self._agent_aliases = AGENT_ALIASES
    
    def calculate_cost(self, input_text: str, output_text: str) -> dict:
        """Calculate cost and token usage"""
        # Approximate tokens = words * 1.5
        input_words = len(input_text.split()) if input_text else 0
        output_words = len(output_text.split()) if output_text else 0
        
        input_tokens = int(input_words * 1.5)
        output_tokens = int(output_words * 1.5)
        
        # Cost per million tokens
        input_cost_per_million = 0.1  # $0.1 per 1M input tokens
        output_cost_per_million = 0.4  # $0.4 per 1M output tokens
        
        input_cost = (input_tokens / 1_000_000) * input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * output_cost_per_million
        
        total_cost = input_cost + output_cost
        
        return {
            "cost": total_cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }

    def _ensure_planner_plan_graph(self, output, raw_response: str) -> dict:
        """
        Ensure PlannerAgent output always contains a plan_graph key.
        """
        if isinstance(output, list) and output and isinstance(output[0], dict):
            output = output[0]
        if not isinstance(output, dict):
            output = {"response": str(output) if output is not None else raw_response}

        if "plan_graph" not in output:
            if isinstance(output.get("plan"), dict):
                plan = output["plan"]
                output["plan_graph"] = (
                    plan.get("plan_graph", plan)
                    if isinstance(plan.get("plan_graph"), dict)
                    else plan
                )
            elif "nodes" in output:
                output["plan_graph"] = {
                    "nodes": output.get("nodes", []),
                    "edges": output.get("edges", output.get("links", [])),
                }
            else:
                output["plan_graph"] = {"nodes": [], "edges": []}

        pg = output.get("plan_graph")
        if not isinstance(pg, dict):
            pg = {"nodes": [], "edges": []}
        if not isinstance(pg.get("nodes"), list):
            pg["nodes"] = []
        if "edges" not in pg:
            pg["edges"] = pg.get("links", []) if isinstance(pg.get("links"), list) else []
        if not isinstance(pg.get("edges"), list):
            pg["edges"] = []

        # Normalize planner edge sources to the runtime bootstrap planner node.
        for edge in pg["edges"]:
            if not isinstance(edge, dict):
                continue
            source = edge.get("source")
            if source in {"ROOT", "root", "original_query", "query", "user_query"}:
                edge["source"] = "Query"

        # Ensure a valid next step id exists for downstream consumers.
        if not output.get("next_step_id"):
            if pg["nodes"] and isinstance(pg["nodes"][0], dict):
                output["next_step_id"] = pg["nodes"][0].get("id", "T001")
            else:
                output["next_step_id"] = "T001"
        output["plan_graph"] = pg
        return output

    def _extract_wise_from_text(self, text: str) -> dict:
        """
        Extract risk_level, confidence, and flags from raw LLM response text
        (e.g. markdown with "Risk Level: High", "Flags: [...]") for WISE integration.
        Returns a dict with keys risk_level, confidence, flags; missing keys or None
        mean "not found".
        """
        if not text or not isinstance(text, str):
            return {}
        out = {}
        # risk_level: "Risk Level: High", "*Risk Level:* moderate", risk_level": "low"
        for pattern in (
            r"(?i)risk_level[\"']?\s*:\s*[\"']?(\w+)",
            r"(?i)\*?\s*Risk\s+Level\s*\*?\s*:\s*(\w+)",
            r"(?i)Risk\s+Level:\s*(\w+)",
        ):
            m = re.search(pattern, text)
            if m:
                raw = m.group(1).lower().strip()
                if raw in ("low", "moderate", "high"):
                    out["risk_level"] = raw
                elif raw in ("normal", "medium"):
                    out["risk_level"] = "moderate"
                else:
                    out["risk_level"] = "moderate"
                break
        # confidence: "Confidence: 0.85", "*Confidence:* 0.85", "85%"
        for pattern in (
            r"(?i)confidence[\"']?\s*:\s*([\d.]+)",
            r"(?i)\*?\s*Confidence\s*\*?\s*:\s*([\d.]+)",
            r"(?i)Confidence:\s*([\d.]+)",
            r"(\d+)\s*%",
        ):
            m = re.search(pattern, text)
            if m:
                try:
                    v = float(m.group(1))
                    if v > 1:
                        v = v / 100.0
                    out["confidence"] = max(0.0, min(1.0, v))
                except (TypeError, ValueError):
                    pass
                if "confidence" in out:
                    break
        # flags: "Flags: [\"low_hemoglobin\", \"high_wbc\"]" or "Flags: ['low_hemoglobin', 'high_wbc']"
        for pattern in (
            r"(?i)flags[\"']?\s*:\s*\[\s*([^\]]*)\s*\]",
            r"(?i)Flags:\s*\[\s*([^\]]*)\s*\]",
        ):
            m = re.search(pattern, text)
            if m:
                inner = m.group(1).strip()
                if not inner:
                    out["flags"] = []
                else:
                    parts = [p.strip().strip('"\'') for p in re.split(r",", inner)]
                    out["flags"] = [p for p in parts if p]
                break
        return out

    def _ensure_wise_output_schema(self, output, raw_response: str) -> dict:
        """
        Ensure ThinkerAgent output includes risk_level, confidence, flags for WISE integration.
        When parsed output is missing these (e.g. fallback to {response: "..."}), extract from
        raw text before applying defaults.
        """
        if isinstance(output, list) and output and isinstance(output[0], dict):
            output = output[0]
        if not isinstance(output, dict):
            output = {"response": str(output) if output is not None else raw_response}
        source = (raw_response or "").strip() or (output.get("response") if isinstance(output.get("response"), str) else "")
        extracted = self._extract_wise_from_text(source) if source else {}
        # risk_level
        if output.get("risk_level") not in ("low", "moderate", "high"):
            output["risk_level"] = extracted.get("risk_level") or output.get("risk_level") or "moderate"
            if output["risk_level"] not in ("low", "moderate", "high"):
                output["risk_level"] = "moderate"
        # confidence
        if output.get("confidence") is None:
            output["confidence"] = extracted.get("confidence") if extracted.get("confidence") is not None else 0.5
        else:
            try:
                output["confidence"] = float(output["confidence"])
            except (TypeError, ValueError):
                output["confidence"] = extracted.get("confidence") if extracted.get("confidence") is not None else 0.5
        # flags
        if not isinstance(output.get("flags"), list):
            output["flags"] = extracted.get("flags") if isinstance(extracted.get("flags"), list) else []
        return output

    async def run_agent(self, agent_type: str, input_data: dict, image_path: Optional[str] = None) -> dict:
        """Run a specific agent with input data and optional image"""
        # Resolve planner-invented aliases to actual configured agents
        agent_type = self._agent_aliases.get(agent_type, agent_type)

        if agent_type not in self.agent_configs:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
        config = self.agent_configs[agent_type]
        
        try:
            # 1. Load prompt template
            prompt_template = Path(config["prompt_file"]).read_text(encoding="utf-8")
            
            # 2. Get tools from specified MCP servers (if any)
            tools_text = ""
            if config.get("mcp_servers"):
                tools = self.multi_mcp.get_tools_from_servers(config["mcp_servers"])
                if tools:
                    tool_descriptions = []
                    for tool in tools:
                        schema = tool.inputSchema
                        if "input" in schema.get("properties", {}):
                            inner_key = next(iter(schema.get("$defs", {})), None)
                            props = schema["$defs"][inner_key]["properties"]
                        else:
                            props = schema["properties"]

                        arg_types = []
                        for k, v in props.items():
                            t = v.get("type", "any")
                            arg_types.append(t)

                        signature_str = ", ".join(arg_types)
                        tool_descriptions.append(f"- `{tool.name}({signature_str})` # {tool.description}")
                    
                    tools_text = "\n\n### Available Tools\n\n" + "\n".join(tool_descriptions)

            
            # 3. Build full prompt
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # 3a. Inject user preferences (compact format)
            try:
                from remme.preferences import get_compact_policy
                # Map agent types to scopes for preference lookup
                scope_map = {
                    "PlannerAgent": "planning", "CoderAgent": "coding",
                    "DistillerAgent": "coding", "FormatterAgent": "formatting",
                    "RetrieverAgent": "research", "ThinkerAgent": "reasoning",
                }
                scope = scope_map.get(agent_type, "general")
                user_prefs_text = f"\n---\n## User Preferences\n{get_compact_policy(scope)}\n---\n"
            except Exception as e:
                print(f"⚠️ Could not load user preferences: {e}")
                user_prefs_text = ""
            
            full_prompt = f"CURRENT_DATE: {current_date}\n\n{prompt_template.strip()}{user_prefs_text}{tools_text}\n\n```json\n{json.dumps(input_data, indent=2)}\n```"

            print(f"🛠️ [DEBUG] Generated Tools Text for {agent_type}:\n{tools_text}\n")

            # 📝 LOGGING: Save prompt to file for debugging
            debug_log_dir = Path(__file__).parent.parent / "memory" / "debug_logs"
            debug_log_dir.mkdir(parents=True, exist_ok=True)
            (debug_log_dir / "latest_prompt.txt").write_text(f"AGENT: {agent_type}\nCONFIG: {config['prompt_file']}\n\n{full_prompt}", encoding="utf-8")
            log_step(f"🤖 {agent_type} invoked", payload={"prompt_file": config['prompt_file'], "input_keys": list(input_data.keys())}, symbol="🟦")

            # 4. Create model manager with user's selected model from settings
            # IMPORTANT: Use reload_settings() to get fresh settings from disk
            from config.settings_loader import reload_settings
            fresh_settings = reload_settings()
            agent_settings = fresh_settings.get("agent", {})
            
            # Check for per-agent overrides
            overrides = agent_settings.get("overrides", {})
            if agent_type in overrides:
                override = overrides[agent_type]
                model_provider = override.get("model_provider", "gemini")
                model_name = override.get("model", "gemini-2.5-flash")
                log_step(f"🎯 Override for {agent_type}: {model_provider}:{model_name}", symbol="✨")
            else:
                model_provider = agent_settings.get("model_provider", "gemini")
                model_name = agent_settings.get("default_model", "gemini-2.5-flash")
            
            log_step(f"📡 Using {model_provider}:{model_name}", symbol="🔌")
            model_manager = ModelManager(model_name, provider=model_provider)
            
            # 5. Generate response (with or without image)
            if image_path and os.path.exists(image_path):
                log_step(f"🖼️ {agent_type} (with image)")
                image = Image.open(image_path)
                response = await model_manager.generate_content([full_prompt, image])
            else:
                response = await model_manager.generate_text(full_prompt)
            
            # 📝 LOGGING: Save raw response
            timestamp = datetime.now().strftime("%H%M%S")
            (debug_log_dir / f"{timestamp}_{agent_type}_response.txt").write_text(response, encoding="utf-8")
            (debug_log_dir / f"{timestamp}_{agent_type}_prompt.txt").write_text(full_prompt, encoding="utf-8")

            # 6. Parse JSON response dynamically (PlannerAgent must be strict; others allow plain-text fallback)
            if agent_type == "PlannerAgent":
                try:
                    output = parse_llm_json(response)
                except Exception:
                    output = parse_llm_json_or_fallback(response, fallback_key="response")
                output = self._ensure_planner_plan_graph(output, response)
            elif agent_type == "ThinkerAgent":
                output = parse_llm_json_or_fallback(response, fallback_key="response")
                output = self._ensure_wise_output_schema(output, response)
            elif agent_type == "SummarizerAgent":
                output = parse_llm_json_or_fallback(response, fallback_key="response")
                output = self._ensure_wise_output_schema(output, response)
                # Ensure machine-readable flags are always present in markdown output
                # so downstream WISE adapters can parse consistently.
                if isinstance(output.get("response"), str) and "Flags:" not in output["response"]:
                    output["response"] = (
                        output["response"].rstrip()
                        + f"\n\nFlags: {json.dumps(output.get('flags', []))}"
                    )
            else:
                output = parse_llm_json_or_fallback(response, fallback_key="response")
            
            # Robustness: Some models (like gemma3) wrap JSON in a list
            if isinstance(output, list) and len(output) > 0 and isinstance(output[0], dict):
                output = output[0]
                
            log_step(f"🟩 {agent_type} finished", payload={"output_keys": list(output.keys()) if isinstance(output, dict) else "raw_string"}, symbol="🟩")

            # import pdb; pdb.set_trace()
            
            # Calculate input text for costing
            input_text = str(input_data)
            
            # Calculate output text for costing
            output_text = str(output)
            
            # Calculate cost and tokens
            cost_data = self.calculate_cost(input_text, output_text)
            
            # Add cost data and model info to result
            if isinstance(output, dict):
                output.update(cost_data)
                output["executed_model"] = f"{model_provider}:{model_name}"
            
            return {
                "success": True,
                "agent_type": agent_type,
                "output": output
            }
            
        except Exception as e:
            log_error(f"❌ {agent_type}: {str(e)}")
            return {
                "success": False,
                "agent_type": agent_type,
                "error": str(e),
                "cost": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            }

    def get_available_agents(self) -> list:
        """Return list of available agent types"""
        return list(self.agent_configs.keys())
