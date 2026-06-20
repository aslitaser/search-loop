from __future__ import annotations

import random
from dataclasses import dataclass

from searchloop.tools import ToolRegistry, ToolResult

SERVICES = [
    "auction-engine",
    "catalog-service",
    "central-services",
    "risingwave",
    "bid-retrieval",
]


@dataclass(frozen=True)
class Action:
    tool: str
    args: tuple[tuple[str, str], ...]

    @classmethod
    def from_dict(cls, tool: str, args: dict[str, str]) -> Action:
        return cls(
            tool=tool,
            args=tuple(sorted((name, str(value)) for name, value in args.items())),
        )

    def args_dict(self) -> dict[str, str]:
        return dict(self.args)


@dataclass(frozen=True)
class Observation:
    action: Action
    result: ToolResult
    evidence_gained: str | None


@dataclass(frozen=True)
class State:
    observations: tuple[Observation, ...]
    evidence: frozenset[str]
    steps: int
    resolved: bool
    resolved_target: str | None

    @classmethod
    def initial(cls) -> State:
        return cls(
            observations=(),
            evidence=frozenset(),
            steps=0,
            resolved=False,
            resolved_target=None,
        )


@dataclass(frozen=True)
class Reveal:
    tool: str
    arg_value: str
    token: str


@dataclass(frozen=True)
class Task:
    culprit: str
    required_evidence: frozenset[str]
    reveals: tuple[Reveal, ...]
    max_steps: int


def revealed_token(task: Task, action: Action) -> str | None:
    action_args = action.args_dict()
    for reveal in task.reveals:
        if action.tool == reveal.tool and any(
            reveal.arg_value in value for value in action_args.values()
        ):
            return reveal.token
    return None


def step(
    task: Task,
    state: State,
    action: Action,
    registry: ToolRegistry,
    rng: random.Random,
) -> tuple[State, ToolResult]:
    evidence_gained = None

    if action.tool == "resolve":
        target = action.args_dict().get("target")
        result = ToolResult(ok=True, output=f"resolve {target}", latency_ms=0.0, error=None)
        new_evidence = state.evidence
        resolved = True
        resolved_target = target
    else:
        result = registry.get(action.tool).execute(action.args_dict(), rng)
        evidence_gained = revealed_token(task, action) if result.ok else None
        new_evidence = state.evidence | {evidence_gained} if evidence_gained else state.evidence
        resolved = state.resolved
        resolved_target = state.resolved_target

    observation = Observation(action=action, result=result, evidence_gained=evidence_gained)
    new_state = State(
        observations=state.observations + (observation,),
        evidence=new_evidence,
        steps=state.steps + 1,
        resolved=resolved,
        resolved_target=resolved_target,
    )
    return new_state, result


def is_terminal(task: Task, state: State) -> bool:
    return state.resolved or state.steps >= task.max_steps


def reward(task: Task, state: State) -> float:
    if not is_terminal(task, state):
        return 0.0

    if (
        state.resolved
        and state.resolved_target == task.culprit
        and task.required_evidence <= state.evidence
    ):
        base = 1.0
    elif state.resolved and state.resolved_target == task.culprit:
        base = 0.3
    elif state.resolved:
        base = -1.0
    else:
        base = -0.5

    return base - 0.02 * state.steps


def make_task(seed: int) -> Task:
    rng = random.Random(seed)
    culprit = rng.choice(SERVICES)
    tokens = tuple(f"ev_{culprit}_{index}" for index in range(3))
    reveals = (
        Reveal("get_logs", culprit, tokens[0]),
        Reveal("get_metrics", culprit, tokens[1]),
        Reveal("check_deploy", culprit, tokens[2]),
    )

    return Task(
        culprit=culprit,
        required_evidence=frozenset(tokens),
        reveals=reveals,
        max_steps=8,
    )
