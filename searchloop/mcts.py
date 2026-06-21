from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from math import floor, log, sqrt

from searchloop import env
from searchloop.env import Action, State, Task, is_terminal, reward, step
from searchloop.llm import Proposer
from searchloop.tools import ToolRegistry
from searchloop.traces import TraceCollector, state_features


@dataclass(frozen=True)
class MCTSConfig:
    c_explore: float = 1.41421356
    n_candidates: int = 4
    max_iterations: int = 50
    seed: int | None = None
    rollout_depth: int = 8
    evidence_bonus: float = 0.25
    pw_k: float = 1.0
    pw_alpha: float = 0.5


@dataclass
class MCTSNode:
    state: State
    parent: MCTSNode | None
    action: Action | None
    terminal: bool
    children: dict[Action, MCTSNode] = field(default_factory=dict)
    untried_actions: list[Action] | None = None
    visits: int = 0
    value_sum: float = 0.0

    def q(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def is_fully_expanded(self) -> bool:
        return self.untried_actions is not None and len(self.untried_actions) == 0


def uct_score(node: MCTSNode, c: float) -> float:
    if node.visits == 0:
        return float("inf")

    assert node.parent is not None
    return node.q() + c * sqrt(log(node.parent.visits) / node.visits)


def best_uct_child(node: MCTSNode, c: float) -> MCTSNode:
    if not node.children:
        raise ValueError("Cannot choose a UCT child from a node with no children")

    return max(node.children.values(), key=lambda child: uct_score(child, c))


def pw_limit(node: MCTSNode, k: float, alpha: float) -> int:
    return max(1, floor(k * (node.visits**alpha)))


def is_expandable(node: MCTSNode, config: MCTSConfig) -> bool:
    return (not node.terminal) and (
        node.untried_actions is None
        or (
            len(node.untried_actions) > 0
            and len(node.children) < pw_limit(node, config.pw_k, config.pw_alpha)
        )
    )


def select_leaf(
    root: MCTSNode,
    c: float,
    is_expandable: Callable[[MCTSNode], bool] | None = None,
) -> MCTSNode:
    if is_expandable is None:
        def is_expandable(node: MCTSNode) -> bool:
            return (not node.terminal) and (not node.is_fully_expanded())

    node = root
    while not node.terminal and not is_expandable(node) and node.children:
        node = best_uct_child(node, c)
    return node


def expand_child(
    parent: MCTSNode,
    action: Action,
    child_state: State,
    terminal: bool,
    untried: list[Action] | None = None,
) -> MCTSNode:
    child = MCTSNode(
        state=child_state,
        parent=parent,
        action=action,
        terminal=terminal,
        untried_actions=untried,
    )
    parent.children[action] = child
    if parent.untried_actions is not None and action in parent.untried_actions:
        parent.untried_actions.remove(action)
    return child


def random_rollout_action(state: State, registry: ToolRegistry, rng: random.Random) -> Action:
    tool_names = registry.names()
    if rng.random() < 0.2 or not tool_names:
        return Action.from_dict("resolve", {"target": rng.choice(env.SERVICES)})

    tool_name = rng.choice(tool_names)
    tool = registry.get(tool_name)
    param_name = next(iter(tool.params), "query")
    return Action.from_dict(tool_name, {param_name: rng.choice(env.SERVICES)})


def rollout(
    task: Task,
    state: State,
    registry: ToolRegistry,
    rng: random.Random,
    max_depth: int,
    evidence_bonus: float,
) -> float:
    rollout_state = state
    depth = 0
    while not is_terminal(task, rollout_state) and depth < max_depth:
        action = random_rollout_action(rollout_state, registry, rng)
        rollout_state, _ = step(task, rollout_state, action, registry, rng)
        depth += 1

    collected_required = len(rollout_state.evidence & task.required_evidence)
    return reward(task, rollout_state) + evidence_bonus * collected_required


def backpropagate(node: MCTSNode, value: float) -> None:
    current: MCTSNode | None = node
    while current is not None:
        current.visits += 1
        current.value_sum += value
        current = current.parent


def mcts_search(
    task: Task,
    registry: ToolRegistry,
    proposer: Proposer,
    rng: random.Random,
    config: MCTSConfig,
    root_state: State,
    trace: TraceCollector | None = None,
    value_fn: Callable[[list[float]], float] | None = None,
) -> Action:
    root = MCTSNode(
        state=root_state,
        parent=None,
        action=None,
        terminal=is_terminal(task, root_state),
        untried_actions=None,
    )

    for _ in range(config.max_iterations):
        def can_expand(candidate: MCTSNode) -> bool:
            return is_expandable(candidate, config)

        node = select_leaf(root, config.c_explore, can_expand)
        if not node.terminal:
            if node.untried_actions is None:
                node.untried_actions = list(proposer.propose(node.state, config.n_candidates))
            if (
                node.untried_actions
                and len(node.children) < pw_limit(node, config.pw_k, config.pw_alpha)
            ):
                action = node.untried_actions.pop(0)
                child_state, _ = step(task, node.state, action, registry, rng)
                node = expand_child(
                    node,
                    action,
                    child_state,
                    is_terminal(task, child_state),
                    untried=None,
                )

        features = state_features(node.state, task.max_steps)
        if value_fn is not None:
            value = value_fn(features)
        else:
            value = rollout(
                task,
                node.state,
                registry,
                rng,
                config.rollout_depth,
                config.evidence_bonus,
            )
        if trace is not None:
            trace.record(features, value)
        backpropagate(node, value)

    if not root.children:
        return Action.from_dict("resolve", {"target": rng.choice(env.SERVICES)})

    best = max(root.children.values(), key=lambda child: child.visits)
    assert best.action is not None
    return best.action
