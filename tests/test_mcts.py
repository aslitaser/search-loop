import random
from math import log, sqrt

import pytest

from searchloop import env
from searchloop.agents import run_mcts
from searchloop.env import Action, Reveal, State, Task, reward
from searchloop.llm import UsageMeter
from searchloop.mcts import (
    MCTSConfig,
    MCTSNode,
    backpropagate,
    best_uct_child,
    expand_child,
    is_expandable,
    mcts_search,
    pw_limit,
    random_rollout_action,
    rollout,
    select_leaf,
    uct_score,
)
from searchloop.tools import Tool, ToolRegistry


def _action(name: str) -> Action:
    return Action.from_dict("tool", {"name": name})


def _node(
    *,
    parent: MCTSNode | None = None,
    action: Action | None = None,
    terminal: bool = False,
    visits: int = 0,
    value_sum: float = 0.0,
    untried_actions: list[Action] | None = None,
) -> MCTSNode:
    return MCTSNode(
        state=State.initial(),
        parent=parent,
        action=action,
        terminal=terminal,
        visits=visits,
        value_sum=value_sum,
        untried_actions=untried_actions or [],
    )


def test_q_is_zero_without_visits_and_mean_otherwise() -> None:
    assert _node(visits=0, value_sum=10.0).q() == 0.0
    assert _node(visits=4, value_sum=3.0).q() == 0.75


def test_is_fully_expanded_depends_on_untried_actions() -> None:
    assert MCTSNode(State.initial(), None, None, terminal=False).is_fully_expanded() is False
    assert _node(untried_actions=[]).is_fully_expanded() is True
    assert _node(untried_actions=[_action("a")]).is_fully_expanded() is False


def test_pw_limit_scales_with_visits() -> None:
    assert pw_limit(_node(visits=0), k=1.0, alpha=0.5) == 1
    assert pw_limit(_node(visits=4), k=1.0, alpha=0.5) == 2
    assert pw_limit(_node(visits=9), k=1.0, alpha=0.5) == 3
    assert pw_limit(_node(visits=4), k=2.0, alpha=0.5) == 4


def test_is_expandable_respects_progressive_widening_limit() -> None:
    config = MCTSConfig(pw_k=1.0, pw_alpha=0.5)
    unproposed = MCTSNode(State.initial(), None, None, terminal=False)
    under_limit = _node(visits=4, untried_actions=[_action("candidate")])
    at_limit = _node(visits=4, untried_actions=[_action("candidate")])
    terminal = _node(terminal=True, visits=4, untried_actions=[_action("candidate")])

    expand_child(under_limit, _action("child-1"), State.initial(), terminal=False)
    expand_child(at_limit, _action("child-1"), State.initial(), terminal=False)
    expand_child(at_limit, _action("child-2"), State.initial(), terminal=False)

    assert is_expandable(unproposed, config) is True
    assert is_expandable(under_limit, config) is True
    assert is_expandable(at_limit, config) is False
    assert is_expandable(terminal, config) is False


def test_uct_score_is_infinite_for_unvisited_node() -> None:
    parent = _node(visits=10)
    child = _node(parent=parent, visits=0)

    assert uct_score(child, c=1.0) == float("inf")


def test_uct_score_matches_manual_formula() -> None:
    parent = _node(visits=20)
    child = _node(parent=parent, visits=5, value_sum=3.0)
    c = 1.25

    expected = (3.0 / 5) + c * sqrt(log(20) / 5)

    assert uct_score(child, c=c) == pytest.approx(expected)


def test_best_uct_child_prefers_unvisited_child() -> None:
    parent = _node(visits=10)
    visited = expand_child(parent, _action("visited"), State.initial(), terminal=False)
    unvisited = expand_child(parent, _action("unvisited"), State.initial(), terminal=False)
    visited.visits = 5
    visited.value_sum = 5.0

    assert best_uct_child(parent, c=1.0) is unvisited


def test_best_uct_child_prefers_higher_uct_among_visited_children() -> None:
    parent = _node(visits=100)
    low = expand_child(parent, _action("low"), State.initial(), terminal=False)
    high = expand_child(parent, _action("high"), State.initial(), terminal=False)
    low.visits = 20
    low.value_sum = 2.0
    high.visits = 10
    high.value_sum = 8.0

    assert best_uct_child(parent, c=1.0) is high


def test_best_uct_child_breaks_ties_by_insertion_order() -> None:
    parent = _node(visits=10)
    first = expand_child(parent, _action("first"), State.initial(), terminal=False)
    second = expand_child(parent, _action("second"), State.initial(), terminal=False)
    first.visits = 5
    first.value_sum = 2.5
    second.visits = 5
    second.value_sum = 2.5

    assert best_uct_child(parent, c=1.0) is first


def test_best_uct_child_raises_without_children() -> None:
    with pytest.raises(ValueError, match="no children"):
        best_uct_child(_node(), c=1.0)


def test_select_leaf_descends_max_uct_path() -> None:
    root = _node(visits=50)
    left = expand_child(root, _action("left"), State.initial(), terminal=False, untried=[])
    right = expand_child(root, _action("right"), State.initial(), terminal=False, untried=[])
    left.visits = 10
    left.value_sum = 2.0
    right.visits = 10
    right.value_sum = 8.0

    right_left = expand_child(
        right,
        _action("right-left"),
        State.initial(),
        terminal=False,
        untried=[],
    )
    right_right = expand_child(
        right,
        _action("right-right"),
        State.initial(),
        terminal=False,
        untried=[],
    )
    right_left.visits = 5
    right_left.value_sum = 1.0
    right_right.visits = 5
    right_right.value_sum = 4.0

    assert select_leaf(root, c=1.0) is right_right


def test_select_leaf_stops_at_terminal_node_mid_descent() -> None:
    root = _node(visits=10)
    terminal = expand_child(root, _action("terminal"), State.initial(), terminal=True)
    terminal.visits = 1
    terminal.value_sum = 1.0
    expand_child(terminal, _action("ignored"), State.initial(), terminal=False)

    assert select_leaf(root, c=1.0) is terminal


def test_select_leaf_stops_at_not_fully_expanded_node() -> None:
    root = _node(visits=10)
    child = expand_child(
        root,
        _action("child"),
        State.initial(),
        terminal=False,
        untried=[_action("untried")],
    )
    child.visits = 1
    child.value_sum = 1.0

    assert select_leaf(root, c=1.0) is child


def test_select_leaf_with_pw_predicate_descends_past_saturated_node() -> None:
    config = MCTSConfig(pw_k=1.0, pw_alpha=0.5)
    saturated = _node(visits=4, untried_actions=[_action("extra")])
    low = expand_child(saturated, _action("low"), State.initial(), terminal=False, untried=[])
    high = expand_child(saturated, _action("high"), State.initial(), terminal=False, untried=[])
    low.visits = 1
    low.value_sum = 0.0
    high.visits = 1
    high.value_sum = 1.0

    def can_expand(node: MCTSNode) -> bool:
        return is_expandable(node, config)

    assert select_leaf(saturated, c=1.0, is_expandable=can_expand) is high


def test_select_leaf_with_pw_predicate_stops_at_node_under_limit() -> None:
    config = MCTSConfig(pw_k=1.0, pw_alpha=0.5)
    under_limit = _node(visits=9, untried_actions=[_action("extra")])
    expand_child(under_limit, _action("child-1"), State.initial(), terminal=False)
    expand_child(under_limit, _action("child-2"), State.initial(), terminal=False)

    def can_expand(node: MCTSNode) -> bool:
        return is_expandable(node, config)

    assert select_leaf(under_limit, c=1.0, is_expandable=can_expand) is under_limit


def test_expand_child_attaches_child_and_updates_untried_actions() -> None:
    action = _action("expand")
    remaining = _action("remaining")
    parent = _node(untried_actions=[action, remaining])
    child_untried = [_action("child-untried")]

    child = expand_child(parent, action, State.initial(), terminal=True, untried=child_untried)

    assert child.parent is parent
    assert child.action == action
    assert child.terminal is True
    assert parent.children[action] is child
    assert parent.untried_actions == [remaining]
    assert child.untried_actions == child_untried


def test_backpropagate_updates_leaf_mid_and_root() -> None:
    root = _node()
    mid = expand_child(root, _action("mid"), State.initial(), terminal=False, untried=[])
    leaf = expand_child(mid, _action("leaf"), State.initial(), terminal=False, untried=[])

    backpropagate(leaf, 0.75)

    assert (root.visits, root.value_sum) == (1, 0.75)
    assert (mid.visits, mid.value_sum) == (1, 0.75)
    assert (leaf.visits, leaf.value_sum) == (1, 0.75)


def test_rollout_returns_reward_for_terminal_state_without_stepping() -> None:
    task = Task(
        culprit="risingwave",
        required_evidence=frozenset({"ev"}),
        reveals=(),
        max_steps=8,
    )
    state = State(
        observations=(),
        evidence=frozenset({"ev"}),
        steps=2,
        resolved=True,
        resolved_target="risingwave",
    )

    assert rollout(
        task,
        state,
        ToolRegistry([]),
        random.Random(1),
        max_depth=8,
        evidence_bonus=0.0,
    ) == reward(task, state)


def test_rollout_adds_evidence_bonus_to_terminal_true_reward() -> None:
    required = frozenset({"ev_a", "ev_b", "ev_c"})
    task = Task(
        culprit="risingwave",
        required_evidence=required,
        reveals=(),
        max_steps=8,
    )
    state = State(
        observations=(),
        evidence=frozenset({"ev_a", "ev_c", "irrelevant"}),
        steps=3,
        resolved=True,
        resolved_target="risingwave",
    )
    true_reward = reward(task, state)

    shaped = rollout(
        task,
        state,
        ToolRegistry([]),
        random.Random(2),
        max_depth=0,
        evidence_bonus=0.25,
    )
    unshaped = rollout(
        task,
        state,
        ToolRegistry([]),
        random.Random(2),
        max_depth=0,
        evidence_bonus=0.0,
    )

    assert shaped == pytest.approx(true_reward + 0.25 * 2)
    assert unshaped == true_reward


def test_rollout_is_deterministic_and_returns_reward_like_float() -> None:
    task = Task(
        culprit="risingwave",
        required_evidence=frozenset({"ev_risingwave_0"}),
        reveals=(Reveal("get_logs", "risingwave", "ev_risingwave_0"),),
        max_steps=4,
    )
    registry = ToolRegistry([_tool("get_logs", {"pod": "Pod to inspect."})])

    first = rollout(
        task,
        State.initial(),
        registry,
        random.Random(5),
        max_depth=4,
        evidence_bonus=0.25,
    )
    second = rollout(
        task,
        State.initial(),
        registry,
        random.Random(5),
        max_depth=4,
        evidence_bonus=0.25,
    )

    assert isinstance(first, float)
    assert -2.0 <= first <= 1.25
    assert first == second


def test_random_rollout_action_uses_known_tools_or_resolve_and_is_deterministic() -> None:
    registry = ToolRegistry([_tool("get_logs", {"pod": "Pod to inspect."})])

    first = random_rollout_action(State.initial(), registry, random.Random(6))
    second = random_rollout_action(State.initial(), registry, random.Random(6))

    assert first == second
    assert first.tool in set(registry.names()) | {"resolve"}
    assert any(value in env.SERVICES for value in first.args_dict().values())


def test_mcts_end_to_end_winnable_and_deterministic() -> None:
    task = _winnable_task()
    registry = ToolRegistry(
        [
            _tool("get_logs", {"pod": "Pod to inspect."}),
            _tool("get_metrics", {"query": "Metrics query."}),
            _tool("check_deploy", {"app": "Application to inspect."}),
        ]
    )
    config = MCTSConfig(max_iterations=8, n_candidates=1, rollout_depth=2)

    first = run_mcts(task, registry, _GuidedProposer(), random.Random(7), config)
    second = run_mcts(task, registry, _GuidedProposer(), random.Random(7), config)

    assert first.success is True
    assert first.correct_culprit is True
    assert first == second


def test_empty_proposer_fallback_returns_resolve_and_run_mcts_terminates_resolved() -> None:
    task = Task(
        culprit="risingwave",
        required_evidence=frozenset({"ev"}),
        reveals=(),
        max_steps=8,
    )
    proposer = _EmptyProposer()
    config = MCTSConfig(max_iterations=3, n_candidates=2, rollout_depth=2)

    action = mcts_search(
        task,
        ToolRegistry([]),
        proposer,
        random.Random(8),
        config,
        State.initial(),
    )
    result = run_mcts(task, ToolRegistry([]), _EmptyProposer(), random.Random(8), config)

    assert action.tool == "resolve"
    assert result.final_state.resolved is True


def _tool(name: str, params: dict[str, str]) -> Tool:
    return Tool(
        name=name,
        description="Test tool.",
        params=params,
        failure_prob=0.0,
        latency_mean_ms=10,
        latency_std_ms=1,
    )


def _winnable_task() -> Task:
    culprit = "risingwave"
    return Task(
        culprit=culprit,
        required_evidence=frozenset(
            {
                "ev_risingwave_0",
                "ev_risingwave_1",
                "ev_risingwave_2",
            }
        ),
        reveals=(
            Reveal("get_logs", culprit, "ev_risingwave_0"),
            Reveal("get_metrics", culprit, "ev_risingwave_1"),
            Reveal("check_deploy", culprit, "ev_risingwave_2"),
        ),
        max_steps=8,
    )


class _GuidedProposer:
    def __init__(self) -> None:
        self.usage = UsageMeter()

    def propose(self, state: State, n: int) -> list[Action]:
        self.usage.record(0, 0)
        missing = [
            token
            for token in ("ev_risingwave_0", "ev_risingwave_1", "ev_risingwave_2")
            if token not in state.evidence
        ]
        if not missing:
            return [Action.from_dict("resolve", {"target": "risingwave"})][:n]
        if missing[0].endswith("_0"):
            return [Action.from_dict("get_logs", {"pod": "risingwave"})][:n]
        if missing[0].endswith("_1"):
            return [
                Action.from_dict(
                    "get_metrics",
                    {"query": "error_rate{service='risingwave'}"},
                )
            ][:n]
        return [Action.from_dict("check_deploy", {"app": "risingwave"})][:n]


class _EmptyProposer:
    def __init__(self) -> None:
        self.usage = UsageMeter()

    def propose(self, state: State, n: int) -> list[Action]:
        self.usage.record(0, 0)
        return []
