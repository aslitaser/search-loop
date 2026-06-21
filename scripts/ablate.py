#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from collections.abc import Callable

from searchloop.ablate import Variant, format_ablation_table, run_ablation
from searchloop.agents import EpisodeResult, run_greedy, run_mcts
from searchloop.env import Task, make_task
from searchloop.llm import (
    DEFAULT_MODEL,
    DEFAULT_OPENAI_MODEL,
    AnthropicProposer,
    OpenAIProposer,
    Proposer,
    default_briefing,
)
from searchloop.mcts import MCTSConfig
from searchloop.tools import ToolRegistry, default_registry
from searchloop.value import load_model, make_value_fn


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MCTS ablations.")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--model")
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--value-model", default="value_model.json")
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--evidence-bonus", type=float, default=0.25)
    parser.add_argument("--pw-k", type=float, default=1.0)
    parser.add_argument("--pw-alpha", type=float, default=0.5)
    parser.add_argument("--proposer-seed", type=int, default=0)
    args = parser.parse_args()

    if args.provider == "openai":
        if "OPENAI_API_KEY" not in os.environ:
            print("OPENAI_API_KEY is not set; set it to run the OpenAI ablation.")
            return 0
    else:
        if "ANTHROPIC_API_KEY" not in os.environ:
            print("ANTHROPIC_API_KEY is not set; set it to run the Anthropic ablation.")
            return 0

    seeds = _parse_seeds(args.seeds)
    registry = default_registry()
    briefing = default_briefing(make_task(seeds[0]).max_steps if seeds else 8)
    model = args.model or _default_model(args.provider)
    proposer = _build_proposer(args.provider, registry, briefing, model, args.proposer_seed)
    value_fn = _load_value_fn(args.value_model)
    variants = _build_variants(args, value_fn)

    total_episodes = len(variants) * len(seeds)
    print(
        f"Cost warning: {len(variants)} variants x {len(seeds)} seeds = "
        f"{total_episodes} episodes."
    )
    results = run_ablation(variants, proposer, registry, seeds)
    print(format_ablation_table(results, full_label="full"))
    return 0


def _build_variants(args, value_fn) -> list[Variant]:
    variants = [
        Variant("greedy", lambda: run_greedy, use_cache=True),
        Variant(
            "rollout",
            lambda: _mcts_runner(
                _config(args, evidence_bonus=args.evidence_bonus, use_pw=True),
                value_fn=None,
            ),
            use_cache=True,
        ),
    ]
    if value_fn is None:
        print(f"{args.value_model} not found; skipping value-head ablation variants.")
        return variants

    variants.extend(
        [
            Variant(
                "full",
                lambda: _mcts_runner(
                    _config(args, evidence_bonus=args.evidence_bonus, use_pw=True),
                    value_fn=value_fn,
                ),
                use_cache=True,
            ),
            Variant(
                "no-shaping",
                lambda: _mcts_runner(
                    _config(args, evidence_bonus=0.0, use_pw=True),
                    value_fn=value_fn,
                ),
                use_cache=True,
            ),
            Variant(
                "no-pw",
                lambda: _mcts_runner(
                    _config(args, evidence_bonus=args.evidence_bonus, use_pw=False),
                    value_fn=value_fn,
                ),
                use_cache=True,
            ),
            Variant(
                "no-cache",
                lambda: _mcts_runner(
                    _config(args, evidence_bonus=args.evidence_bonus, use_pw=True),
                    value_fn=value_fn,
                ),
                use_cache=False,
            ),
        ]
    )
    return variants


def _config(args, evidence_bonus: float, use_pw: bool) -> MCTSConfig:
    return MCTSConfig(
        max_iterations=args.iterations,
        n_candidates=args.candidates,
        evidence_bonus=evidence_bonus,
        pw_k=args.pw_k,
        pw_alpha=args.pw_alpha,
        use_pw=use_pw,
    )


def _mcts_runner(
    config: MCTSConfig,
    value_fn: Callable[[list[float]], float] | None,
):
    def runner(
        task: Task,
        registry: ToolRegistry,
        proposer: Proposer,
        rng,
    ) -> EpisodeResult:
        return run_mcts(task, registry, proposer, rng, config, value_fn=value_fn)

    return runner


def _load_value_fn(path: str):
    if not os.path.exists(path):
        return None
    model, mean, std = load_model(path)
    return make_value_fn(model, mean, std)


def _build_proposer(
    provider: str,
    registry: ToolRegistry,
    briefing: str,
    model: str,
    seed: int | None,
) -> Proposer:
    if provider == "openai":
        return OpenAIProposer(registry, briefing, model=model, seed=seed)
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
