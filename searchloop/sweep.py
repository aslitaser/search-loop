from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from searchloop.agents import EpisodeResult, run_mcts
from searchloop.env import Task
from searchloop.llm import Proposer
from searchloop.mcts import MCTSConfig
from searchloop.metrics import Summary, run_benchmark, summarize
from searchloop.tools import ToolRegistry


@dataclass(frozen=True)
class SweepResult:
    label: str
    config: MCTSConfig
    summary: Summary
    mean_total_tokens: float
    mean_wall_ms: float


def run_sweep(
    grid: list[MCTSConfig],
    seeds: Iterable[int],
    proposer: Proposer,
    registry: ToolRegistry,
) -> list[SweepResult]:
    results = []
    seed_list = list(seeds)
    for config in grid:
        def runner(
            task: Task,
            reg: ToolRegistry,
            prop: Proposer,
            random_source: random.Random,
            current_config: MCTSConfig = config,
        ) -> EpisodeResult:
            return run_mcts(task, reg, prop, random_source, current_config)

        rows = run_benchmark(runner, proposer, registry, seed_list)
        summary = summarize(rows)
        results.append(_result_from_summary(_label(config), config, summary))

    return results


def pareto_frontier(
    results: list[SweepResult],
    quality: Callable[[SweepResult], float] = lambda result: result.summary.mean_reward,
    cost: Callable[[SweepResult], float] = lambda result: result.mean_total_tokens,
) -> list[SweepResult]:
    frontier = []
    for result in results:
        dominated = any(
            quality(other) >= quality(result)
            and cost(other) <= cost(result)
            and (quality(other) > quality(result) or cost(other) < cost(result))
            for other in results
        )
        if not dominated:
            frontier.append(result)

    return frontier


def format_sweep_table(
    results: list[SweepResult],
    frontier: list[SweepResult],
    baseline: SweepResult | None = None,
) -> str:
    frontier_ids = {id(result) for result in frontier}
    header = (
        f"{'frontier':>8} {'label':<36} {'succ':>7} {'reward':>8} "
        f"{'tok/ep':>9} {'tok/succ':>9} {'wall_ms':>9}"
    )
    line = "-" * len(header)
    rows = [header, line]
    if baseline is not None:
        rows.append(_format_row(baseline, marker=""))
        rows.append(line)

    rows.extend(
        _format_row(result, marker="*" if id(result) in frontier_ids else "") for result in results
    )
    return "\n".join(rows)


def _result_from_summary(label: str, config: MCTSConfig, summary: Summary) -> SweepResult:
    total_tokens = summary.total_input_tokens + summary.total_output_tokens
    return SweepResult(
        label=label,
        config=config,
        summary=summary,
        mean_total_tokens=total_tokens / max(1, summary.n),
        mean_wall_ms=summary.mean_wall_ms,
    )


def _format_row(result: SweepResult, marker: str) -> str:
    tokens_per_success = (
        "None"
        if result.summary.tokens_per_success is None
        else f"{result.summary.tokens_per_success:.1f}"
    )
    return (
        f"{marker:>8} {result.label:<36} {result.summary.success_rate:>7.3f} "
        f"{result.summary.mean_reward:>8.3f} {result.mean_total_tokens:>9.1f} "
        f"{tokens_per_success:>9} {result.mean_wall_ms:>9.1f}"
    )


def _label(config: MCTSConfig) -> str:
    return (
        f"iter={config.max_iterations},"
        f"cand={config.n_candidates},"
        f"eb={config.evidence_bonus:g},"
        f"pw={config.pw_k:g}/{config.pw_alpha:g}"
    )
