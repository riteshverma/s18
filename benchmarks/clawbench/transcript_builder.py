"""Convert S18 run events into a ClawBench Transcript."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from clawbench.schemas import ToolCall, Transcript, TranscriptMessage


@dataclass
class EventCollector:
    """Subscribe to S18 event_bus and accumulate tool-call pairs."""

    pending: dict[str, dict[str, Any]] = field(default_factory=dict)
    tool_calls: list[ToolCall] = field(default_factory=list)
    _queue: Any = None

    async def start(self) -> None:
        from core.event_bus import event_bus

        self._queue = await event_bus.subscribe(max_queue_size=500, replay_history=False)

    async def stop(self) -> None:
        self._queue = None

    async def drain(self) -> None:
        if self._queue is None:
            return
        while not self._queue.empty():
            event = self._queue.get_nowait()
            if event.get("type") != "tool_call":
                continue
            data = event.get("data") or {}
            key = f"{data.get('step_id')}:{data.get('tool_name')}"
            phase = data.get("phase")
            if phase == "start":
                self.pending[key] = data
            elif phase == "end":
                start = self.pending.pop(key, {})
                self.tool_calls.extend(
                    synthesize_tool_calls(
                        tool_name=str(data.get("tool_name") or start.get("tool_name") or "unknown"),
                        arguments=data.get("arguments") or start.get("arguments") or {},
                        output=str(data.get("result_preview") or ""),
                        success=True,
                    )
                )

    def record_tool_call(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        output: str,
        success: bool = True,
    ) -> None:
        """Record a synthetic tool call (e.g. post-run workspace materialization)."""
        self.tool_calls.extend(
            synthesize_tool_calls(
                tool_name=tool_name,
                arguments=arguments,
                output=output,
                success=success,
            )
        )

    def build_transcript(
        self,
        *,
        user_messages: list[str],
        assistant_text: str,
    ) -> Transcript:
        messages: list[TranscriptMessage] = []
        for text in user_messages:
            messages.append(TranscriptMessage(role="user", text=text))
        if self.tool_calls:
            messages.append(
                TranscriptMessage(
                    role="assistant",
                    text=assistant_text,
                    tool_calls=list(self.tool_calls),
                )
            )
        elif assistant_text.strip():
            messages.append(TranscriptMessage(role="assistant", text=assistant_text))
        return Transcript(messages=messages)


def synthesize_tool_calls(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    output: str,
    success: bool,
) -> list[ToolCall]:
    """Map S18 MCP tools (especially sandbox Python) to ClawBench tool names."""
    lowered = tool_name.lower()
    code = str(arguments.get("code") or arguments.get("script") or "")

    if lowered == "read_workspace_file":
        path = str(arguments.get("path") or "")
        return [
            ToolCall(
                name="read_file",
                input={"path": path},
                output=output,
                success=success,
            )
        ]

    if lowered == "write_workspace_file":
        path = str(arguments.get("path") or "")
        payload = {"path": path}
        content = arguments.get("content")
        if content is not None:
            payload["content"] = str(content)
        return [
            ToolCall(
                name="apply_patch",
                input=payload,
                output=output,
                success=success,
            )
        ]

    if lowered in {"run_python_script", "run_user_code"} and code.strip():
        calls: list[ToolCall] = []
        inferred = _infer_calls_from_python(code)
        for index, (inferred_name, payload) in enumerate(inferred):
            calls.append(
                ToolCall(
                    name=inferred_name,
                    input=payload,
                    output=output if index == len(inferred) - 1 else "",
                    success=success,
                )
            )
        if calls:
            return calls

    mapped_name = tool_name
    payload = dict(arguments)
    if "browser" in lowered:
        mapped_name = "browser_action"
    elif "memory" in lowered or "remme" in lowered:
        mapped_name = "memory_search" if "search" in lowered or "get" in lowered else "memory_write"
    elif "rag" in lowered or "search" in lowered:
        mapped_name = "search_files"
    elif lowered in {"run_python_script", "run_user_code"}:
        mapped_name = "exec_command"
        payload = {"cmd": code or str(arguments)}

    return [
        ToolCall(
            name=mapped_name,
            input=payload,
            output=output,
            success=success,
        )
    ]


def _infer_calls_from_python(code: str) -> list[tuple[str, dict[str, str]]]:
    inferred: list[tuple[str, dict[str, str]]] = []
    read_paths = set(re.findall(r"""open\s*\(\s*['"]([^'"]+)['"]\s*,\s*['"]r""", code))
    write_paths = set(
        re.findall(r"""open\s*\(\s*['"]([^'"]+)['"]\s*,\s*['"][wa]""", code)
        + re.findall(r"""Path\s*\(\s*['"]([^'"]+)['"]\s*\)\.write_text""", code)
    )
    for path in sorted(read_paths):
        inferred.append(("read_file", {"path": path}))
    for path in sorted(write_paths):
        inferred.append(("apply_patch", {"path": path, "code": code}))
    if re.search(r"\bpytest\b", code, re.IGNORECASE) or re.search(
        r"subprocess\.(run|call|Popen)", code
    ):
        inferred.append(("exec_command", {"cmd": code}))
    if not inferred:
        inferred.append(("exec_command", {"cmd": code}))
    return inferred


def extract_assistant_text(context: Any) -> str:
    chunks: list[str] = []
    plan_graph = getattr(context, "plan_graph", None)
    if plan_graph is None:
        return ""
    for node_id in plan_graph.nodes:
        node = plan_graph.nodes[node_id]
        if node_id == "ROOT":
            continue
        output = node.get("output")
        if output:
            chunks.append(str(output))
    return "\n".join(chunks).strip()
