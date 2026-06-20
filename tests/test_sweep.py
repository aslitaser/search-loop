import random

from searchloop.agents import EpisodeResult
from searchloop.env import State, Task
from searchloop.llm import UsageMeter
from searchloop.mcts import MCTSConfig
from searchloop.metrics import Summary
from searchloop.sweep import SweepResult, format_sweep_table, pareto_frontier, run_sweep
from searchloop.tools import ToolRegistry


def test_pareto_frontier_excludes_dominated_and_keeps_ties() -> None:
    high_quality = _result("high", mean_reward=1.0, mean_tokens=100.0)
    dominated = _result("dominated", mean_reward=0.5, mean_tokens=150.0)
    cheap = _result("cheap", mean_reward=0.8, mean_tokens=50.0)
    tie = _result("tie", mean_reward=1.0, mean_tokens=100.0)

    frontier = pareto_frontier([high_quality, dominated, cheap, tie])

    assert frontier == [high_quality, cheap, tie]


def test_run_sweep_uses_benchmark_and_computes_mean_tokens(monkeypatch) -> None:
    def stub_run_mcts(
        task: Task,
        registry: ToolRegistry,
        proposer: _StubProposer,
        rng: random.Random,
        config: MCTSConfig,
    ) -> EpisodeResult:
        proposer.usage.record(config.max_iterations, config.n_candidates)
        final_state = State.initial()
        return EpisodeResult(
            final_state=final_state,
            reward=float(config.max_iterations),
            steps=1,
            success=False,
            correct_culprit=False,
            proposer_calls=1,
        )

    monkeypatch.setattr("searchloop.sweep.run_mcts", stub_run_mcts)
    grid = [
        MCTSConfig(max_iterations=10, n_candidates=2),
        MCTSConfig(max_iterations=20, n_candidates=4),
    ]
    proposer = _StubProposer()

    results = run_sweep(grid, seeds=[0, 1], proposer=proposer, registry=ToolRegistry([]))

    assert len(results) == 2
    assert [result.summary.n for result in results] == [2, 2]
    assert [result.mean_total_tokens for result in results] == [12.0, 24.0]
    assert [result.summary.mean_reward for result in results] == [10.0, 20.0]


def test_format_sweep_table_includes_labels_frontier_marker_and_baseline() -> None:
    baseline = _result("greedy", mean_reward=0.1, mean_tokens=5.0)
    first = _result("iter=15,cand=2,eb=0.25,pw=1/0.5", mean_reward=0.5, mean_tokens=10.0)
    second = _result("iter=30,cand=4,eb=0.25,pw=1/0.5", mean_reward=0.7, mean_tokens=30.0)

    text = format_sweep_table([first, second], frontier=[first], baseline=baseline)

    assert isinstance(text, str)
    assert "greedy" in text
    assert first.label in text
    assert second.label in text
    assert "*" in text
    assert "tok/ep" in text


def _summary(mean_reward: float, total_tokens: int, n: int = 1) -> Summary:
    return Summary(
        n=n,
        success_rate=0.0,
        correct_culprit_rate=0.0,
        mean_reward=mean_reward,
        mean_steps=1.0,
        total_input_tokens=total_tokens,
        total_output_tokens=0,
        tokens_per_success=None,
        mean_sim_latency_ms=0.0,
        mean_wall_ms=2.0,
    )


def _result(label: str, mean_reward: float, mean_tokens: float) -> SweepResult:
    summary = _summary(mean_reward, total_tokens=int(mean_tokens))
    return SweepResult(
        label=label,
        config=MCTSConfig(),
        summary=summary,
        mean_total_tokens=mean_tokens,
        mean_wall_ms=summary.mean_wall_ms,
    )


class _StubProposer:
    def __init__(self) -> None:
        self.usage = UsageMeter()

    def propose(self, state: State, n: int):
        return []
