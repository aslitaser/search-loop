import random

import pytest

from searchloop.agents import EpisodeResult
from searchloop.env import Action, Observation, State, Task
from searchloop.llm import UsageMeter
from searchloop.metrics import EpisodeMetrics, format_table, run_benchmark, summarize
from searchloop.tools import ToolRegistry, ToolResult


def test_summarize_computes_rates_means_and_tokens_per_success() -> None:
    rows = [
        EpisodeMetrics(
            seed=0,
            success=True,
            correct_culprit=True,
            reward=1.0,
            steps=4,
            proposer_calls=4,
            input_tokens=100,
            output_tokens=50,
            sim_latency_ms=30.0,
            wall_ms=10.0,
        ),
        EpisodeMetrics(
            seed=1,
            success=False,
            correct_culprit=True,
            reward=0.2,
            steps=5,
            proposer_calls=5,
            input_tokens=10,
            output_tokens=5,
            sim_latency_ms=10.0,
            wall_ms=4.0,
        ),
    ]

    summary = summarize(rows)

    assert summary.n == 2
    assert summary.success_rate == pytest.approx(0.5)
    assert summary.correct_culprit_rate == pytest.approx(1.0)
    assert summary.mean_reward == pytest.approx(0.6)
    assert summary.mean_steps == pytest.approx(4.5)
    assert summary.total_input_tokens == 110
    assert summary.total_output_tokens == 55
    assert summary.tokens_per_success == pytest.approx(165.0)
    assert summary.mean_sim_latency_ms == pytest.approx(20.0)
    assert summary.mean_wall_ms == pytest.approx(7.0)


def test_summarize_tokens_per_success_is_none_without_successes() -> None:
    summary = summarize(
        [
            EpisodeMetrics(
                seed=0,
                success=False,
                correct_culprit=False,
                reward=-1.0,
                steps=1,
                proposer_calls=1,
                input_tokens=10,
                output_tokens=5,
                sim_latency_ms=3.0,
                wall_ms=1.0,
            )
        ]
    )

    assert summary.tokens_per_success is None


def test_run_benchmark_resets_usage_and_sums_simulated_latency() -> None:
    proposer = _StubProposer()
    rows = run_benchmark(
        _stub_runner,
        proposer,
        ToolRegistry([]),
        seeds=[0, 1],
        task_factory=_task_factory,
    )

    assert len(rows) == 2
    assert [row.seed for row in rows] == [0, 1]
    assert [row.input_tokens for row in rows] == [100, 100]
    assert [row.output_tokens for row in rows] == [25, 25]
    assert [row.sim_latency_ms for row in rows] == [20.0, 20.0]
    assert all(row.wall_ms >= 0.0 for row in rows)


def test_format_table_contains_headers_and_summary_numbers() -> None:
    rows = [
        EpisodeMetrics(
            seed=3,
            success=True,
            correct_culprit=True,
            reward=0.92,
            steps=4,
            proposer_calls=4,
            input_tokens=10,
            output_tokens=2,
            sim_latency_ms=30.0,
            wall_ms=4.0,
        )
    ]
    summary = summarize(rows)

    text = format_table(rows, summary)

    assert isinstance(text, str)
    assert "seed" in text
    assert "ok?" in text
    assert "in_tok" in text
    assert "success_rate: 1.000" in text
    assert "total_input_tokens: 10" in text
    assert "tokens_per_success: 12.0" in text


class _StubProposer:
    def __init__(self) -> None:
        self.usage = UsageMeter()

    def propose(self, state: State, n: int) -> list[Action]:
        return []


def _stub_runner(
    task: Task,
    registry: ToolRegistry,
    proposer: _StubProposer,
    rng: random.Random,
) -> EpisodeResult:
    proposer.usage.record(100, 25)
    action = Action.from_dict("probe", {"target": "svc"})
    observations = (
        Observation(action, ToolResult(True, "one", 12.5, None), None),
        Observation(action, ToolResult(True, "two", 7.5, None), None),
    )
    final_state = State(
        observations=observations,
        evidence=frozenset(),
        steps=2,
        resolved=True,
        resolved_target=task.culprit,
    )
    return EpisodeResult(
        final_state=final_state,
        reward=0.1,
        steps=2,
        success=False,
        correct_culprit=True,
        proposer_calls=2,
    )


def _task_factory(seed: int) -> Task:
    return Task(
        culprit=f"svc-{seed}",
        required_evidence=frozenset(),
        reveals=(),
        max_steps=2,
    )
