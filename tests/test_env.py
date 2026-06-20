import random
from dataclasses import FrozenInstanceError

import pytest

from searchloop.env import (
    Action,
    Reveal,
    State,
    Task,
    is_terminal,
    make_task,
    revealed_token,
    reward,
    step,
)
from searchloop.tools import Tool, ToolRegistry


def _tool(name: str, failure_prob: float) -> Tool:
    return Tool(
        name=name,
        description="Test tool.",
        params={"target": "Target to inspect."},
        failure_prob=failure_prob,
        latency_mean_ms=10,
        latency_std_ms=1,
    )


def _task(max_steps: int = 8) -> Task:
    return Task(
        culprit="svc",
        required_evidence=frozenset({"ev_a", "ev_b", "ev_c"}),
        reveals=(),
        max_steps=max_steps,
    )


def test_state_and_action_are_frozen() -> None:
    action = Action.from_dict("probe", {"target": "svc"})
    state = State.initial()

    with pytest.raises(FrozenInstanceError):
        action.tool = "other"

    with pytest.raises(FrozenInstanceError):
        state.steps = 1


def test_action_round_trips_and_hashes() -> None:
    first = Action.from_dict("probe", {"b": 2, "a": "one"})
    second = Action.from_dict("probe", {"a": "one", "b": "2"})

    assert first.args == (("a", "one"), ("b", "2"))
    assert first.args_dict() == {"a": "one", "b": "2"}
    assert first == second
    assert hash(first) == hash(second)
    assert {first: "value"}[second] == "value"


def test_step_does_not_mutate_input_state() -> None:
    registry = ToolRegistry([_tool("probe", failure_prob=0.0)])
    state = State.initial()
    before = (
        state.observations,
        state.evidence,
        state.steps,
        state.resolved,
        state.resolved_target,
    )

    new_state, result = step(
        _task(),
        state,
        Action.from_dict("probe", {"target": "svc"}),
        registry,
        random.Random(1),
    )

    assert result.ok
    assert (
        state.observations,
        state.evidence,
        state.steps,
        state.resolved,
        state.resolved_target,
    ) == before
    assert new_state is not state
    assert new_state.steps == 1
    assert len(new_state.observations) == 1


def test_revealed_token_matches_correct_tool_and_arg_value_substring() -> None:
    task = Task(
        culprit="risingwave",
        required_evidence=frozenset({"ev_risingwave_0"}),
        reveals=(Reveal("get_logs", "risingwave", "ev_risingwave_0"),),
        max_steps=8,
    )

    action = Action.from_dict("get_logs", {"pod": "risingwave"})

    assert revealed_token(task, action) == "ev_risingwave_0"


def test_revealed_token_matches_with_extra_args_present() -> None:
    task = Task(
        culprit="risingwave",
        required_evidence=frozenset({"ev_risingwave_0"}),
        reveals=(Reveal("get_logs", "risingwave", "ev_risingwave_0"),),
        max_steps=8,
    )

    action = Action.from_dict("get_logs", {"pod": "risingwave", "lines": "100"})

    assert revealed_token(task, action) == "ev_risingwave_0"


def test_revealed_token_matches_embedded_name_in_arg_value() -> None:
    task = Task(
        culprit="risingwave",
        required_evidence=frozenset({"ev_risingwave_1"}),
        reveals=(Reveal("get_metrics", "risingwave", "ev_risingwave_1"),),
        max_steps=8,
    )

    action = Action.from_dict("get_metrics", {"query": "error_rate{service='risingwave'}"})

    assert revealed_token(task, action) == "ev_risingwave_1"


def test_revealed_token_does_not_match_wrong_tool_or_missing_name() -> None:
    task = Task(
        culprit="risingwave",
        required_evidence=frozenset({"ev_risingwave_0"}),
        reveals=(Reveal("get_logs", "risingwave", "ev_risingwave_0"),),
        max_steps=8,
    )

    assert revealed_token(task, Action.from_dict("get_metrics", {"query": "risingwave"})) is None
    assert revealed_token(task, Action.from_dict("get_logs", {"pod": "catalog-service"})) is None


def test_successful_step_reveals_evidence() -> None:
    action = Action.from_dict("probe", {"target": "svc"})
    token = "ev_probe"
    task = Task(
        culprit="svc",
        required_evidence=frozenset({token}),
        reveals=(Reveal("probe", "svc", token),),
        max_steps=8,
    )
    registry = ToolRegistry([_tool("probe", failure_prob=0.0)])

    new_state, result = step(task, State.initial(), action, registry, random.Random(2))

    assert result.ok
    assert new_state.observations[-1].evidence_gained == token
    assert token in new_state.evidence


def test_failed_step_does_not_reveal_evidence() -> None:
    action = Action.from_dict("probe", {"target": "svc"})
    task = Task(
        culprit="svc",
        required_evidence=frozenset({"ev_probe"}),
        reveals=(Reveal("probe", "svc", "ev_probe"),),
        max_steps=8,
    )
    registry = ToolRegistry([_tool("probe", failure_prob=1.0)])

    new_state, result = step(task, State.initial(), action, registry, random.Random(3))

    assert not result.ok
    assert new_state.evidence == frozenset()
    assert new_state.steps == 1
    assert new_state.observations[-1].evidence_gained is None


def test_resolve_does_not_consult_registry() -> None:
    action = Action.from_dict("resolve", {"target": "svc"})

    new_state, result = step(
        _task(),
        State.initial(),
        action,
        ToolRegistry([]),
        random.Random(4),
    )

    assert result.ok
    assert result.output == "resolve svc"
    assert new_state.resolved is True
    assert new_state.resolved_target == "svc"
    assert new_state.evidence == frozenset()
    assert new_state.observations[-1].evidence_gained is None


def test_is_terminal() -> None:
    task = _task(max_steps=2)

    assert not is_terminal(task, State.initial())
    assert is_terminal(task, State((), frozenset(), steps=1, resolved=True, resolved_target="svc"))
    assert is_terminal(task, State((), frozenset(), steps=2, resolved=False, resolved_target=None))


def test_reward_ordering_and_step_penalty() -> None:
    required = frozenset({"ev_a", "ev_b", "ev_c"})
    task = Task(culprit="svc", required_evidence=required, reveals=(), max_steps=2)
    correct_full = State((), required, steps=2, resolved=True, resolved_target="svc")
    correct_unsupported = State((), frozenset(), steps=2, resolved=True, resolved_target="svc")
    budget_exhausted = State((), frozenset(), steps=2, resolved=False, resolved_target=None)
    wrong_resolve = State((), required, steps=2, resolved=True, resolved_target="other")

    assert (
        reward(task, correct_full)
        > reward(task, correct_unsupported)
        > reward(task, budget_exhausted)
        > reward(task, wrong_resolve)
    )

    faster = State((), required, steps=1, resolved=True, resolved_target="svc")
    slower = State((), required, steps=2, resolved=True, resolved_target="svc")

    assert reward(task, faster) > reward(task, slower)


def test_make_task_is_reproducible_and_varies_across_seeds() -> None:
    task = make_task(7)

    assert task == make_task(7)
    assert len(task.reveals) == 3
    assert [reveal.tool for reveal in task.reveals] == [
        "get_logs",
        "get_metrics",
        "check_deploy",
    ]
    assert all(reveal.arg_value == task.culprit for reveal in task.reveals)
    assert task.required_evidence == frozenset(
        f"ev_{task.culprit}_{index}" for index in range(3)
    )
    assert {reveal.token for reveal in task.reveals} == task.required_evidence

    culprits = {make_task(seed).culprit for seed in range(21)}

    assert len(culprits) > 1
