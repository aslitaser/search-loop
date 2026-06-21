from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from searchloop.env import SERVICES, Action, State
from searchloop.tools import ToolRegistry

DEFAULT_MODEL = "claude-sonnet-4-6"
# Switch to "claude-haiku-4-5-20251001" for the search-heavy phase; it's cheaper per call.
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
# A cheap, fast model is right for a proposer called many times in search.
# If you switch to an o-series / reasoning model, the param is max_completion_tokens
# (not max_tokens) and temperature may be unsupported.

SYSTEM = (
    "You are an investigation agent. Reply with ONLY a JSON array of action objects, "
    "no prose and no markdown fences. Each object must be "
    '{"tool": <one of the allowed tool names>, "args": {<string>: <string>}}. '
    'For resolve use {"tool": "resolve", "args": {"target": "<service>"}}.'
)


class ProposalError(ValueError):
    pass


@dataclass
class UsageMeter:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def record(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.calls += 1

    def reset(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0


def default_briefing(max_steps: int) -> str:
    services = ", ".join(SERVICES)
    return (
        "You are investigating which service is the culprit. "
        f"The candidate services are: {services}. "
        "Use the available tools to gather evidence. "
        "Investigate a service by passing its name as the relevant tool argument; "
        "pointing get_logs, get_metrics, and check_deploy at the true culprit service "
        "yields evidence, which you should collect before resolving. "
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
    stripped = _strip_markdown_fence(text).strip()
    if not stripped:
        return []

    actions = []
    seen = set()
    for value in _extract_json_values(stripped):
        for item in _iter_action_dicts(value):
            action = _coerce_action(item, allowed_tools)
            if action is None or action in seen:
                continue

            actions.append(action)
            seen.add(action)

    return actions


def _extract_json_values(s: str) -> list[object]:
    decoder = json.JSONDecoder()
    values = []
    index = 0
    while index < len(s):
        starts = [
            position
            for position in (s.find("{", index), s.find("[", index))
            if position != -1
        ]
        if not starts:
            break

        start = min(starts)
        try:
            value, end = decoder.raw_decode(s, start)
        except json.JSONDecodeError:
            index = start + 1
            continue

        values.append(value)
        index = end

    return values


def _iter_action_dicts(value: object) -> Iterator[dict[str, Any]]:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yield item
        return

    if not isinstance(value, dict):
        return

    found_wrapper = False
    for key in ("actions", "tools", "calls"):
        items = value.get(key)
        if not isinstance(items, list):
            continue

        found_wrapper = True
        for item in items:
            if isinstance(item, dict):
                yield item

    if not found_wrapper and "tool" in value:
        yield value


def _coerce_action(item: object, allowed_tools: set[str]) -> Action | None:
    if not isinstance(item, dict):
        return None

    tool = item.get("tool")
    if not isinstance(tool, str) or tool not in allowed_tools:
        return None

    args = item.get("args", {})
    if not isinstance(args, dict):
        args = {}
    if not all(isinstance(name, str) for name in args):
        return None

    return Action.from_dict(tool, {name: str(value) for name, value in args.items()})


class Proposer(Protocol):
    usage: UsageMeter

    def propose(self, state: State, n: int) -> list[Action]: ...


class MockProposer(Proposer):
    def __init__(self, batches: list[list[Action]]) -> None:
        self._batches = batches
        self._index = 0
        self.usage = UsageMeter()

    def propose(self, state: State, n: int) -> list[Action]:
        self.usage.record(0, 0)
        if self._index >= len(self._batches):
            raise IndexError("MockProposer exhausted")

        batch = self._batches[self._index]
        self._index += 1
        return batch[:n]


class CachingProposer(Proposer):
    def __init__(self, inner: Proposer) -> None:
        self.inner = inner
        self.cache: dict[tuple[frozenset[str], frozenset[Action], bool], list[Action]] = {}
        self.hits = 0
        self.misses = 0

    @property
    def usage(self) -> UsageMeter:
        return self.inner.usage

    def _signature(self, state: State) -> tuple[frozenset[str], frozenset[Action], bool]:
        return (
            frozenset(state.evidence),
            frozenset(observation.action for observation in state.observations),
            state.resolved,
        )

    def propose(self, state: State, n: int) -> list[Action]:
        signature = self._signature(state)
        if signature in self.cache:
            self.hits += 1
            result = self.cache[signature]
        else:
            self.misses += 1
            result = list(self.inner.propose(state, n))
            self.cache[signature] = result

        return result[:n]

    def reset_cache(self) -> None:
        self.cache.clear()
        self.hits = 0
        self.misses = 0


# Anthropic has no seed parameter; for Anthropic ablation comparability, use temperature=0.
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
        self.usage = UsageMeter()
        self.last_raw = ""

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
        usage = getattr(resp, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        self.usage.record(input_tokens, output_tokens)
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        self.last_raw = text
        return parse_actions(text, self.allowed)[:n]


class OpenAIProposer(Proposer):
    def __init__(
        self,
        registry: ToolRegistry,
        briefing: str,
        model: str = DEFAULT_OPENAI_MODEL,
        client: Any | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> None:
        self.registry = registry
        self.briefing = briefing
        self.model = model
        self._client = client
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.allowed = set(registry.names()) | {"resolve"}
        self.usage = UsageMeter()
        self.last_raw = ""

    @property
    def client(self) -> Any:
        if self._client is None:
            import openai

            self._client = openai.OpenAI()
        return self._client

    def propose(self, state: State, n: int) -> list[Action]:
        user = build_user_prompt(self.briefing, self.registry, state, n)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
        }
        if _uses_max_completion_tokens(self.model):
            kwargs["max_completion_tokens"] = self.max_tokens
        else:
            kwargs["max_tokens"] = self.max_tokens
            kwargs["temperature"] = self.temperature

        if self.seed is not None:
            kwargs["seed"] = self.seed

        resp = self.client.chat.completions.create(**kwargs)
        usage = getattr(resp, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        self.usage.record(input_tokens, output_tokens)
        text = resp.choices[0].message.content or ""
        self.last_raw = text
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


def _uses_max_completion_tokens(model: str) -> bool:
    return model.startswith("gpt-5") or model.startswith("o")
