#!/usr/bin/env python
from __future__ import annotations

import argparse
import os

from searchloop.agents import run_greedy, run_mcts
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
from searchloop.metrics import format_table, run_benchmark, summarize
from searchloop.tools import default_registry
from searchloop.value import load_model, make_value_fn


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark searchloop agents.")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--model")
    parser.add_argument("--seeds", default="0-49")
    parser.add_argument("--agent", choices=["greedy", "mcts"], default="greedy")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--c", type=float, default=1.41421356)
    parser.add_argument("--pw-k", type=float, default=1.0)
    parser.add_argument("--pw-alpha", type=float, default=0.5)
    parser.add_argument("--cache", dest="cache", action="store_true")
    parser.add_argument("--no-cache", dest="cache", action="store_false")
    parser.add_argument("--value-model")
    parser.set_defaults(cache=True)
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    registry = default_registry()
    briefing = default_briefing(make_task(seeds[0]).max_steps if seeds else 8)
    model = args.model or _default_model(args.provider)

    if args.provider == "openai":
        if "OPENAI_API_KEY" not in os.environ:
            print("OPENAI_API_KEY is not set; set it to run the OpenAI benchmark.")
            return 0
        proposer = OpenAIProposer(registry, briefing, model=model)
    else:
        if "ANTHROPIC_API_KEY" not in os.environ:
            print("ANTHROPIC_API_KEY is not set; set it to run the Anthropic benchmark.")
            return 0
        proposer = AnthropicProposer(registry, briefing, model=model)

    if args.cache:
        proposer = CachingProposer(proposer)

    value_fn = None
    if args.value_model:
        model, mean, std = load_model(args.value_model)
        value_fn = make_value_fn(model, mean, std)

    if args.agent == "mcts":
        config = MCTSConfig(
            c_explore=args.c,
            n_candidates=args.candidates,
            max_iterations=args.iterations,
            pw_k=args.pw_k,
            pw_alpha=args.pw_alpha,
        )

        def runner(task, reg, prop, random_source):
            return run_mcts(
                task,
                reg,
                prop,
                random_source,
                config,
                value_fn=value_fn,
            )

    else:
        runner = run_greedy
    rows = run_benchmark(runner, proposer, registry, seeds)
    print(format_table(rows, summarize(rows)))
    return 0


def _parse_seeds(value: str) -> list[int]:
    if "-" in value:
        start_text, end_text = value.split("-", maxsplit=1)
        start = int(start_text)
        end = int(end_text)
        if end < start:
            raise argparse.ArgumentTypeError("--seeds range end must be >= start")
        return list(range(start, end + 1))

    return list(range(int(value)))


def _default_model(provider: str) -> str:
    if provider == "openai":
        return DEFAULT_OPENAI_MODEL
    return DEFAULT_MODEL


if __name__ == "__main__":
    raise SystemExit(main())
