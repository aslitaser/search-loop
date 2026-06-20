from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: str
    latency_ms: float
    error: str | None


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    params: dict[str, str]
    failure_prob: float
    latency_mean_ms: float
    latency_std_ms: float

    def execute(self, args: dict[str, str], rng: random.Random) -> ToolResult:
        fail_roll = rng.random()
        latency = max(0.0, rng.gauss(self.latency_mean_ms, self.latency_std_ms))
        if fail_roll < self.failure_prob:
            return ToolResult(
                ok=False,
                output="",
                latency_ms=latency,
                error=f"{self.name} failed",
            )

        return ToolResult(
            ok=True,
            output=f"{self.name} ok | args={sorted(args.items())}",
            latency_ms=latency,
            error=None,
        )


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            known = ", ".join(self.names()) or "<none>"
            raise KeyError(f"Unknown tool {name!r}; known tools: {known}") from exc

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


def default_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            Tool(
                name="get_pods",
                description="List pods in a Kubernetes namespace.",
                params={"namespace": "Kubernetes namespace to inspect."},
                failure_prob=0.02,
                latency_mean_ms=80,
                latency_std_ms=30,
            ),
            Tool(
                name="get_logs",
                description="Fetch recent log lines from a pod.",
                params={"pod": "Pod name to inspect.", "lines": "Number of log lines to fetch."},
                failure_prob=0.05,
                latency_mean_ms=150,
                latency_std_ms=60,
            ),
            Tool(
                name="get_metrics",
                description="Run a metrics query against the monitoring backend.",
                params={"query": "Metrics query expression to evaluate."},
                failure_prob=0.08,
                latency_mean_ms=200,
                latency_std_ms=80,
            ),
            Tool(
                name="describe_node",
                description="Describe Kubernetes node details and recent conditions.",
                params={"node": "Node name to describe."},
                failure_prob=0.03,
                latency_mean_ms=100,
                latency_std_ms=40,
            ),
            Tool(
                name="check_deploy",
                description="Check rollout and health information for an application deployment.",
                params={"app": "Application name to check."},
                failure_prob=0.10,
                latency_mean_ms=250,
                latency_std_ms=120,
            ),
            Tool(
                name="restart_pod",
                description="Restart a pod as a recovery action.",
                params={"pod": "Pod name to restart."},
                failure_prob=0.20,
                latency_mean_ms=600,
                latency_std_ms=250,
            ),
        ]
    )
