"""A tool-calling agent loop driven through the Model gateway. Agents finish by calling their submit tool."""

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from agentic_jev_rag.gateway import ModelGateway

TOOL_RESULT_LIMIT = 6000
NUDGE = "Call one of your tools, or submit your result with `{submit}`."


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]

    def schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


@dataclass
class AgentResult:
    submitted: dict | None
    steps: int
    cost_usd: float
    latency_s: float
    stopped_reason: Literal["submitted", "step_limit", "no_tool_call"]
    transcript: list[dict] = field(default_factory=list)


def run_agent(
    gateway: ModelGateway,
    model: str,
    system: str,
    task: str,
    tools: list[Tool],
    submit: Tool,
    *,
    max_steps: int = 8,
) -> AgentResult:
    by_name = {t.name: t for t in tools}
    schemas = [t.schema() for t in tools] + [submit.schema()]
    messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    cost = latency = 0.0
    nudged = False

    def result(reason, submitted=None, steps=0) -> AgentResult:
        return AgentResult(submitted, steps, cost, latency, reason, messages)

    for step in range(1, max_steps + 1):
        turn = gateway.chat(model, messages, tools=schemas)
        cost += turn.cost_usd
        latency += turn.latency_s
        messages.append(turn.as_message())

        if not turn.tool_calls:
            if nudged:
                return result("no_tool_call", steps=step)
            nudged = True
            messages.append({"role": "user", "content": NUDGE.format(submit=submit.name)})
            continue

        for call in turn.tool_calls:
            if call.name == submit.name:
                missing = [k for k in submit.parameters.get("required", []) if k not in call.arguments]
                if not missing:
                    messages.append(_tool_message(call.id, {"status": "submitted"}))
                    return result("submitted", submitted=call.arguments, steps=step)
                messages.append(_tool_message(call.id, {"error": f"missing required fields: {missing}"}))
                continue
            messages.append(_tool_message(call.id, _run_tool(by_name.get(call.name), call.name, call.arguments)))

    return result("step_limit", steps=max_steps)


def _run_tool(tool: Tool | None, name: str, arguments: dict) -> Any:
    if tool is None:
        return {"error": f"unknown tool {name!r}"}
    try:
        return tool.fn(**arguments)
    except Exception as error:  # reported back to the model, which can correct itself
        return {"error": f"{type(error).__name__}: {error}"}


def _tool_message(call_id: str, content: Any) -> dict:
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
    return {"role": "tool", "tool_call_id": call_id, "content": text[:TOOL_RESULT_LIMIT]}
