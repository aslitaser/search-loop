#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import os

from searchloop.agents import run_greedy
from searchloop.env import make_task
from searchloop.llm import (
    DEFAULT_MODEL,
    DEFAULT_OPENAI_MODEL,
    AnthropicProposer,
    CachingProposer,
    OpenAIProposer,
    default_briefing,
)
from searchloop.mcts import MCTSConfig
from searchloop.metrics import run_benchmark, summarize
from searchloop.sweep import SweepResult, format_sweep_table, pareto_frontier, run_sweep
from searchloop.tools import default_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep MCTS hyperparameters.")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--model")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--iterations-grid", default="15,30")
    parser.add_argument("--candidates-grid", default="2,4")
    parser.add_argument("--evidence-bonus-grid", default="0.25")
    parser.add_argument("--pw-k", type=float, default=1.0)
    parser.add_argument("--pw-alpha", type=float, default=0.5)
    parser.add_argument("--include-greedy", action="store_true")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    iterations_grid = _parse_ints(args.iterations_grid)
    candidates_grid = _parse_ints(args.candidates_grid)
    evidence_bonus_grid = _parse_floats(args.evidence_bonus_grid)
    grid = [
        MCTSConfig(
            max_iterations=iterations,
            n_candidates=candidates,
            evidence_bonus=evidence_bonus,
            pw_k=args.pw_k,
            pw_alpha=args.pw_alpha,
        )
        for iterations, candidates, evidence_bonus in itertools.product(
            iterations_grid,
            candidates_grid,
            evidence_bonus_grid,
        )
    ]

    if args.provider == "openai":
        if "OPENAI_API_KEY" not in os.environ:
            print("OPENAI_API_KEY is not set; set it to run the OpenAI sweep.")
            return 0
    else:
        if "ANTHROPIC_API_KEY" not in os.environ:
            print("ANTHROPIC_API_KEY is not set; set it to run the Anthropic sweep.")
            return 0

    registry = default_registry()
    briefing = default_briefing(make_task(seeds[0]).max_steps if seeds else 8)
    model = args.model or _default_model(args.provider)
    proposer = _build_proposer(args.provider, registry, briefing, model)

    total_episodes = len(grid) * len(seeds)
    print(f"Cost warning: {len(grid)} configs x {len(seeds)} seeds = {total_episodes} episodes.")

    baseline = None
    if args.include_greedy:
        rows = run_benchmark(run_greedy, proposer, registry, seeds)
        summary = summarize(rows)
        baseline = _baseline_result(summary)

    results = run_sweep(grid, seeds, proposer, registry)
    frontier = pareto_frontier(results)
    print(format_sweep_table(results, frontier, baseline=baseline))
    return 0


def _build_proposer(provider: str, registry, briefing: str, model: str):
    if provider == "openai":
        proposer = OpenAIProposer(registry, briefing, model=model)
    else:
        proposer = AnthropicProposer(registry, briefing, model=model)
    return CachingProposer(proposer)


def _baseline_result(summary) -> SweepResult:
    config = MCTSConfig()
    total_tokens = summary.total_input_tokens + summary.total_output_tokens
    return SweepResult(
        label="greedy",
        config=config,
        summary=summary,
        mean_total_tokens=total_tokens / max(1, summary.n),
        mean_wall_ms=summary.mean_wall_ms,
    )


def _parse_seeds(value: str) -> list[int]:
    if "-" in value:
        start_text, end_text = value.split("-", maxsplit=1)
        start = int(start_text)
        end = int(end_text)
        if end < start:
            raise argparse.ArgumentTypeError("--seeds range end must be >= start")
        return list(range(start, end + 1))

    return list(range(int(value)))


def _parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _default_model(provider: str) -> str:
    if provider == "openai":
        return DEFAULT_OPENAI_MODEL
    return DEFAULT_MODEL


if __name__ == "__main__":
    raise SystemExit(main())
