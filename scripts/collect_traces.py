#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import random

from searchloop.agents import run_mcts
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
from searchloop.tools import default_registry
from searchloop.traces import TraceCollector, write_traces


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect MCTS rollout traces.")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--model")
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--evidence-bonus", type=float, default=0.25)
    parser.add_argument("--out", default="traces.jsonl")
    args = parser.parse_args()

    if args.provider == "openai":
        if "OPENAI_API_KEY" not in os.environ:
            print("OPENAI_API_KEY is not set; set it to collect OpenAI traces.")
            return 0
    else:
        if "ANTHROPIC_API_KEY" not in os.environ:
            print("ANTHROPIC_API_KEY is not set; set it to collect Anthropic traces.")
            return 0

    seeds = _parse_seeds(args.seeds)
    registry = default_registry()
    briefing = default_briefing(make_task(seeds[0]).max_steps if seeds else 8)
    model = args.model or _default_model(args.provider)
    proposer = CachingProposer(_build_proposer(args.provider, registry, briefing, model))
    config = MCTSConfig(
        max_iterations=args.iterations,
        n_candidates=args.candidates,
        evidence_bonus=args.evidence_bonus,
    )
    collector = TraceCollector()

    for seed in seeds:
        task = make_task(seed)
        rng = random.Random(seed)
        proposer.usage.reset()
        reset_cache = getattr(proposer, "reset_cache", None)
        if callable(reset_cache):
            reset_cache()
        run_mcts(task, registry, proposer, rng, config, trace=collector)

    write_traces(args.out, collector.records)
    print(f"wrote {len(collector)} trace records to {args.out}")
    return 0


def _build_proposer(provider: str, registry, briefing: str, model: str):
    if provider == "openai":
        return OpenAIProposer(registry, briefing, model=model)
    return AnthropicProposer(registry, briefing, model=model)


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
