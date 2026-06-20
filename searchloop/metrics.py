from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from time import perf_counter

from searchloop.agents import EpisodeResult, run_greedy
from searchloop.env import Task, make_task
from searchloop.llm import Proposer
from searchloop.tools import ToolRegistry


@dataclass(frozen=True)
class EpisodeMetrics:
    seed: int
    success: bool
    correct_culprit: bool
    reward: float
    steps: int
    proposer_calls: int
    input_tokens: int
    output_tokens: int
    sim_latency_ms: float
    wall_ms: float


@dataclass(frozen=True)
class Summary:
    n: int
    success_rate: float
    correct_culprit_rate: float
    mean_reward: float
    mean_steps: float
    total_input_tokens: int
    total_output_tokens: int
    tokens_per_success: float | None
    mean_sim_latency_ms: float
    mean_wall_ms: float


Runner = Callable[[Task, ToolRegistry, Proposer, random.Random], EpisodeResult]
TaskFactory = Callable[[int], Task]


def run_benchmark(
    runner: Runner,
    proposer: Proposer,
    registry: ToolRegistry,
    seeds: Iterable[int],
    task_factory: TaskFactory = make_task,
) -> list[EpisodeMetrics]:
    rows = []
    for seed in seeds:
        task = task_factory(seed)
        rng = random.Random(seed)
        proposer.usage.reset()
        reset_cache = getattr(proposer, "reset_cache", None)
        if callable(reset_cache):
            reset_cache()
        start = perf_counter()
        result = runner(task, registry, proposer, rng)
        wall_ms = (perf_counter() - start) * 1000
        sim_latency_ms = sum(
            observation.result.latency_ms for observation in result.final_state.observations
        )
        rows.append(
            EpisodeMetrics(
                seed=seed,
                success=result.success,
                correct_culprit=result.correct_culprit,
                reward=result.reward,
                steps=result.steps,
                proposer_calls=result.proposer_calls,
                input_tokens=proposer.usage.input_tokens,
                output_tokens=proposer.usage.output_tokens,
                sim_latency_ms=sim_latency_ms,
                wall_ms=wall_ms,
            )
        )

    return rows


def summarize(rows: list[EpisodeMetrics]) -> Summary:
    n = len(rows)
    if n == 0:
        return Summary(
            n=0,
            success_rate=0.0,
            correct_culprit_rate=0.0,
            mean_reward=0.0,
            mean_steps=0.0,
            total_input_tokens=0,
            total_output_tokens=0,
            tokens_per_success=None,
            mean_sim_latency_ms=0.0,
            mean_wall_ms=0.0,
        )

    successes = sum(row.success for row in rows)
    total_input_tokens = sum(row.input_tokens for row in rows)
    total_output_tokens = sum(row.output_tokens for row in rows)
    total_tokens = total_input_tokens + total_output_tokens

    return Summary(
        n=n,
        success_rate=successes / n,
        correct_culprit_rate=sum(row.correct_culprit for row in rows) / n,
        mean_reward=sum(row.reward for row in rows) / n,
        mean_steps=sum(row.steps for row in rows) / n,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        tokens_per_success=total_tokens / successes if successes > 0 else None,
        mean_sim_latency_ms=sum(row.sim_latency_ms for row in rows) / n,
        mean_wall_ms=sum(row.wall_ms for row in rows) / n,
    )


def format_table(rows: list[EpisodeMetrics], summary: Summary) -> str:
    header = (
        f"{'seed':>5} {'ok?':>3} {'correct?':>8} {'reward':>8} {'steps':>5} "
        f"{'calls':>5} {'in_tok':>7} {'out_tok':>7} {'sim_ms':>9} {'wall_ms':>9}"
    )
    line = "-" * len(header)
    body = [
        (
            f"{row.seed:>5} {_bool_cell(row.success):>3} "
            f"{_bool_cell(row.correct_culprit):>8} {row.reward:>8.3f} "
            f"{row.steps:>5} {row.proposer_calls:>5} "
            f"{row.input_tokens:>7} {row.output_tokens:>7} "
            f"{row.sim_latency_ms:>9.1f} {row.wall_ms:>9.1f}"
        )
        for row in rows
    ]
    tokens_per_success = (
        "None" if summary.tokens_per_success is None else f"{summary.tokens_per_success:.1f}"
    )
    summary_lines = [
        "",
        "summary",
        f"n: {summary.n}",
        f"success_rate: {summary.success_rate:.3f}",
        f"correct_culprit_rate: {summary.correct_culprit_rate:.3f}",
        f"mean_reward: {summary.mean_reward:.3f}",
        f"mean_steps: {summary.mean_steps:.3f}",
        f"total_input_tokens: {summary.total_input_tokens}",
        f"total_output_tokens: {summary.total_output_tokens}",
        f"tokens_per_success: {tokens_per_success}",
        f"mean_sim_latency_ms: {summary.mean_sim_latency_ms:.1f}",
        f"mean_wall_ms: {summary.mean_wall_ms:.1f}",
    ]

    return "\n".join([header, line, *body, *summary_lines])


def _bool_cell(value: bool) -> str:
    return "Y" if value else "N"


__all__ = [
    "EpisodeMetrics",
    "Summary",
    "format_table",
    "run_benchmark",
    "run_greedy",
    "summarize",
]
