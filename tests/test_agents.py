import random

from searchloop import env
from searchloop.agents import EpisodeResult, run_greedy
from searchloop.env import Action, Task, probe_key
from searchloop.llm import MockProposer
from searchloop.tools import Tool, ToolRegistry


def _probe_tool(failure_prob: float = 0.0) -> Tool:
    return Tool(
        name="probe",
        description="Probe a test target.",
        params={"target": "Target to inspect."},
        failure_prob=failure_prob,
        latency_mean_ms=10,
        latency_std_ms=1,
    )


def test_greedy_happy_path_collects_evidence_and_resolves() -> None:
    culprit = "svc-a"
    evidence = frozenset({"ev_a", "ev_b", "ev_c"})
    probe1 = Action.from_dict("probe", {"target": "one"})
    probe2 = Action.from_dict("probe", {"target": "two"})
    probe3 = Action.from_dict("probe", {"target": "three"})
    resolve = Action.from_dict("resolve", {"target": culprit})
    task = Task(
        culprit=culprit,
        required_evidence=evidence,
        reveals=(
            (probe_key(probe1), "ev_a"),
            (probe_key(probe2), "ev_b"),
            (probe_key(probe3), "ev_c"),
        ),
        max_steps=8,
    )
    registry = ToolRegistry([_probe_tool()])
    proposer = MockProposer([[probe1], [probe2], [probe3], [resolve]])

    result = run_greedy(task, registry, proposer, random.Random(1))

    assert result.success is True
    assert result.correct_culprit is True
    assert result.steps == 4
    assert result.proposer_calls == 4
    assert result.reward == 1.0 - 0.02 * 4


def test_greedy_budget_exhaustion_without_resolve() -> None:
    action = Action.from_dict("probe", {"target": "miss"})
    task = Task(
        culprit="svc-a",
        required_evidence=frozenset({"ev_a"}),
        reveals=(),
        max_steps=3,
    )
    registry = ToolRegistry([_probe_tool()])
    proposer = MockProposer([[action] for _ in range(task.max_steps)])

    result = run_greedy(task, registry, proposer, random.Random(2))

    assert result.steps == task.max_steps
    assert result.final_state.resolved is False
    assert result.success is False
    assert result.reward == -0.5 - 0.02 * task.max_steps


def test_greedy_forces_terminal_guess_on_empty_proposal() -> None:
    task = Task(
        culprit="svc-a",
        required_evidence=frozenset({"ev_a"}),
        reveals=(),
        max_steps=8,
    )
    proposer = MockProposer([[]])

    result = run_greedy(task, ToolRegistry([]), proposer, random.Random(3))

    assert result.final_state.resolved is True
    assert result.final_state.resolved_target in env.SERVICES
    assert result.proposer_calls == 1
    assert result.steps == 1


def test_greedy_is_deterministic_for_same_seed_and_script() -> None:
    culprit = "svc-a"
    evidence = frozenset({"ev_a"})
    probe = Action.from_dict("probe", {"target": "one"})
    resolve = Action.from_dict("resolve", {"target": culprit})
    task = Task(
        culprit=culprit,
        required_evidence=evidence,
        reveals=((probe_key(probe), "ev_a"),),
        max_steps=8,
    )
    registry = ToolRegistry([_probe_tool()])

    first = run_greedy(
        task,
        registry,
        MockProposer([[probe], [resolve]]),
        random.Random(4),
    )
    second = run_greedy(
        task,
        registry,
        MockProposer([[probe], [resolve]]),
        random.Random(4),
    )

    assert _stable_result(first) == _stable_result(second)


def _stable_result(result: EpisodeResult) -> tuple[float, int, bool, frozenset[str]]:
    return (
        result.reward,
        result.steps,
        result.success,
        result.final_state.evidence,
    )
