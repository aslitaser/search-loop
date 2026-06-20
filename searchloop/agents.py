from __future__ import annotations

import random
from dataclasses import dataclass

from searchloop import env
from searchloop.env import Action, State, Task, is_terminal, reward, step
from searchloop.llm import Proposer
from searchloop.mcts import MCTSConfig, mcts_search
from searchloop.tools import ToolRegistry


@dataclass(frozen=True)
class EpisodeResult:
    final_state: State
    reward: float
    steps: int
    success: bool
    correct_culprit: bool
    proposer_calls: int


def run_greedy(
    task: Task,
    registry: ToolRegistry,
    proposer: Proposer,
    rng: random.Random,
) -> EpisodeResult:
    state = State.initial()
    proposer_calls = 0

    while not is_terminal(task, state):
        candidates = proposer.propose(state, n=1)
        proposer_calls += 1
        if candidates:
            action = candidates[0]
        else:
            action = Action.from_dict("resolve", {"target": rng.choice(env.SERVICES)})

        state, _ = step(task, state, action, registry, rng)

    success = (
        state.resolved
        and state.resolved_target == task.culprit
        and task.required_evidence <= state.evidence
    )
    correct_culprit = state.resolved and state.resolved_target == task.culprit
    episode_reward = reward(task, state)

    return EpisodeResult(
        final_state=state,
        reward=episode_reward,
        steps=state.steps,
        success=success,
        correct_culprit=correct_culprit,
        proposer_calls=proposer_calls,
    )


def run_mcts(
    task: Task,
    registry: ToolRegistry,
    proposer: Proposer,
    rng: random.Random,
    config: MCTSConfig,
) -> EpisodeResult:
    calls0 = proposer.usage.calls
    state = State.initial()

    while not is_terminal(task, state):
        action = mcts_search(task, registry, proposer, rng, config, state)
        state, _ = step(task, state, action, registry, rng)

    proposer_calls = proposer.usage.calls - calls0
    success = (
        state.resolved
        and state.resolved_target == task.culprit
        and task.required_evidence <= state.evidence
    )
    correct_culprit = state.resolved and state.resolved_target == task.culprit
    episode_reward = reward(task, state)

    return EpisodeResult(
        final_state=state,
        reward=episode_reward,
        steps=state.steps,
        success=success,
        correct_culprit=correct_culprit,
        proposer_calls=proposer_calls,
    )
