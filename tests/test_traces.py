import random

from searchloop.agents import run_mcts
from searchloop.env import Action, Observation, Reveal, State, Task
from searchloop.llm import UsageMeter
from searchloop.mcts import MCTSConfig
from searchloop.tools import Tool, ToolRegistry, ToolResult
from searchloop.traces import TraceCollector, read_traces, state_features, write_traces


def test_state_features_are_fixed_length_agent_facing_values() -> None:
    logs = Action.from_dict("get_logs", {"pod": "risingwave"})
    metrics = Action.from_dict("get_metrics", {"query": "risingwave"})
    state = State(
        observations=(
            Observation(logs, ToolResult(ok=True, output="ok", latency_ms=1.0, error=None), "ev_a"),
            Observation(
                metrics,
                ToolResult(ok=False, output="", latency_ms=2.0, error="fail"),
                None,
            ),
            Observation(logs, ToolResult(ok=True, output="ok", latency_ms=1.0, error=None), "ev_b"),
        ),
        evidence=frozenset({"ev_a", "ev_b"}),
        steps=3,
        resolved=True,
        resolved_target="risingwave",
    )

    assert state_features(state, max_steps=8) == [2.0, 3.0, 5.0, 1.0, 3.0, 1.0, 2.0]


def test_write_read_traces_round_trips(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"
    records = [
        ([1.0, 2.0, 3.0, 0.0, 2.0, 0.0, 1.0], 0.5),
        ([2.0, 3.0, 2.0, 1.0, 3.0, 1.0, 2.0], -0.1),
    ]

    write_traces(str(path), records)

    assert read_traces(str(path)) == records


def test_run_mcts_collects_deterministic_traces() -> None:
    task = _winnable_task()
    registry = ToolRegistry(
        [
            _tool("get_logs", {"pod": "Pod to inspect."}),
            _tool("get_metrics", {"query": "Metrics query."}),
            _tool("check_deploy", {"app": "Application to inspect."}),
        ]
    )
    config = MCTSConfig(max_iterations=8, n_candidates=1, rollout_depth=2)

    first = TraceCollector()
    second = TraceCollector()

    run_mcts(task, registry, _GuidedProposer(), random.Random(7), config, trace=first)
    run_mcts(task, registry, _GuidedProposer(), random.Random(7), config, trace=second)

    assert len(first) > 0
    assert all(len(features) == 7 and isinstance(value, float) for features, value in first.records)
    assert first.records == second.records


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
