from __future__ import annotations

import json
from typing import Any, Protocol

from searchloop.env import SERVICES, Action, State
from searchloop.tools import ToolRegistry

DEFAULT_MODEL = "claude-sonnet-4-6"
# Switch to "claude-haiku-4-5-20251001" for the search-heavy phase; it's cheaper per call.

SYSTEM = (
    "You are an investigation agent. Reply with ONLY a JSON array of action objects, "
    "no prose and no markdown fences. Each object must be "
    '{"tool": <one of the allowed tool names>, "args": {<string>: <string>}}. '
    'For resolve use {"tool": "resolve", "args": {"target": "<service>"}}.'
)


class ProposalError(ValueError):
    pass


def default_briefing(max_steps: int) -> str:
    services = ", ".join(SERVICES)
    return (
        "You are investigating which service is the culprit. "
        f"The candidate services are: {services}. "
        "Use the available tools to gather evidence. "
        'When confident, emit a resolve action with args {"target": "<service>"}. '
        f"You have at most {max_steps} actions."
    )


def render_state(state: State) -> str:
    lines = []
    for index, observation in enumerate(state.observations, start=1):
        status = "ok" if observation.result.ok else "FAIL"
        output = (
            observation.result.output
            if observation.result.ok
            else observation.result.error or ""
        )
        evidence = observation.evidence_gained or "-"
        lines.append(
            f"#{index} {observation.action.tool} {observation.action.args_dict()} "
            f"-> {status}, out={_truncate(output, 120)}, evidence_gained={evidence}"
        )

    lines.append(f"evidence so far: {sorted(state.evidence)}")
    lines.append(f"steps used: {state.steps}")
    return "\n".join(lines)


def render_tools(registry: ToolRegistry) -> str:
    lines = []
    for name in registry.names():
        tool = registry.get(name)
        params = ", ".join(tool.params)
        lines.append(f"{tool.name}({params}) - {tool.description}")

    lines.append("resolve(target) - declare the culprit service and end the episode")
    return "\n".join(lines)


def build_user_prompt(briefing: str, registry: ToolRegistry, state: State, n: int) -> str:
    return "\n\n".join(
        [
            briefing,
            "Available tools:\n" + render_tools(registry),
            "Current transcript:\n" + render_state(state),
            f"Propose up to {n} distinct next actions as a JSON array.",
        ]
    )


def parse_actions(text: str, allowed_tools: set[str]) -> list[Action]:
    parsed = json.loads(_strip_markdown_fence(text))
    if not isinstance(parsed, list):
        raise ProposalError("Expected a JSON array of action objects")

    actions = []
    seen = set()
    for item in parsed:
        if not isinstance(item, dict):
            continue

        tool = item.get("tool")
        args = item.get("args")
        if tool not in allowed_tools or not isinstance(args, dict):
            continue
        if not all(isinstance(name, str) for name in args):
            continue

        action = Action.from_dict(tool, {name: str(value) for name, value in args.items()})
        if action in seen:
            continue

        actions.append(action)
        seen.add(action)

    return actions


class Proposer(Protocol):
    def propose(self, state: State, n: int) -> list[Action]: ...


class MockProposer(Proposer):
    def __init__(self, batches: list[list[Action]]) -> None:
        self._batches = batches
        self._index = 0

    def propose(self, state: State, n: int) -> list[Action]:
        if self._index >= len(self._batches):
            raise IndexError("MockProposer exhausted")

        batch = self._batches[self._index]
        self._index += 1
        return batch[:n]


class AnthropicProposer(Proposer):
    def __init__(
        self,
        registry: ToolRegistry,
        briefing: str,
        model: str = DEFAULT_MODEL,
        client: Any | None = None,
        max_tokens: int = 1024,
    ) -> None:
        self.registry = registry
        self.briefing = briefing
        self.model = model
        self._client = client
        self.max_tokens = max_tokens
        self.allowed = set(registry.names()) | {"resolve"}

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def propose(self, state: State, n: int) -> list[Action]:
        user = build_user_prompt(self.briefing, self.registry, state, n)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        return parse_actions(text, self.allowed)[:n]


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```") or not stripped.endswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) < 2:
        return stripped

    opener = lines[0].strip().lower()
    if opener not in {"```", "```json"}:
        return stripped

    body_lines = lines[1:]
    if body_lines and body_lines[-1].strip() == "```":
        body_lines = body_lines[:-1]

    return "\n".join(body_lines).strip()


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."
