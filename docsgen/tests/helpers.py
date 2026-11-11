"""Test doubles: a deterministic Strands model (no network, no API keys).

``ScriptedModel`` implements the real Strands ``Model`` interface, so agents run through the actual
Strands event loop, tool execution, hooks and structured-output validation. For each agent run it
first issues the scripted tool calls (if any), then returns the structured output for the output
model the agent requested.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from strands.models.model import Model

Payload = dict[str, Any] | Callable[[list, list[str]], dict[str, Any]]


class ScriptedModel(Model):
    def __init__(self, outputs: dict[str, Payload], tool_calls: dict[str, list[dict[str, Any]]] | None = None, text_reply: str = "OK") -> None:
        self.outputs = outputs
        self.tool_calls = tool_calls or {}
        self.text_reply = text_reply
        self.config: dict[str, Any] = {"model_id": "scripted-test-model"}
        self.invocations: list[dict[str, Any]] = []

    def update_config(self, **model_config: Any) -> None:
        self.config.update(model_config)

    def get_config(self) -> dict[str, Any]:
        return self.config

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):  # noqa: ANN001, ANN201
        raise NotImplementedError("Agents use structured_output_model on invocation.")
        yield  # pragma: no cover

    @staticmethod
    def _tool_uses(messages: list) -> list[dict]:
        uses = []
        for message in messages:
            if message.get("role") != "assistant":
                continue
            for block in message.get("content", []):
                if "toolUse" in block:
                    uses.append(block["toolUse"])
        return uses

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):  # noqa: ANN001, ANN201
        names = [t["name"] for t in (tool_specs or [])]
        output_name = next((n for n in names if n in self.outputs), None)
        self.invocations.append({"output": output_name, "tools": names, "system_prompt": system_prompt or "", "messages": messages})
        planned = self.tool_calls.get(output_name or "", [])
        done = [u for u in self._tool_uses(messages) if u.get("name") != output_name]
        yield {"messageStart": {"role": "assistant"}}
        if output_name is None:
            yield {"contentBlockDelta": {"delta": {"text": self.text_reply}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
            return
        if len(done) < len(planned):
            call = planned[len(done)]
            name, payload = call["name"], call.get("input", {})
        else:
            name = output_name
            spec = self.outputs[output_name]
            payload = spec(messages, names) if callable(spec) else spec
        yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": f"tu-{len(done)}-{name}", "name": name}}}}
        yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(payload)}}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "tool_use"}}


def last_tool_result_text(messages: list) -> str:
    for message in reversed(messages):
        for block in message.get("content", []):
            if "toolResult" in block:
                return " ".join(c.get("text", "") for c in block["toolResult"].get("content", []))
    return ""
