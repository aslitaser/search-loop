#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import random

from searchloop.agents import run_greedy
from searchloop.env import make_task
from searchloop.llm import (
    DEFAULT_MODEL,
    DEFAULT_OPENAI_MODEL,
    AnthropicProposer,
    OpenAIProposer,
    default_briefing,
    render_state,
)
from searchloop.tools import default_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one greedy searchloop episode.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--model")
    args = parser.parse_args()

    task = make_task(args.seed)
    registry = default_registry()
    rng = random.Random(args.seed)
    briefing = default_briefing(task.max_steps)
    model = args.model or _default_model(args.provider)

    if args.provider == "openai":
        if "OPENAI_API_KEY" not in os.environ:
            print("OPENAI_API_KEY is not set; set it to run the OpenAI greedy proposer.")
            return 0
        proposer = OpenAIProposer(registry, briefing, model=model)
    else:
        if "ANTHROPIC_API_KEY" not in os.environ:
            print("ANTHROPIC_API_KEY is not set; set it to run the Anthropic greedy proposer.")
            return 0
        proposer = AnthropicProposer(registry, briefing, model=model)

    result = run_greedy(task, registry, proposer, rng)

    print(render_state(result.final_state))
    print(
        "summary: "
        f"success={result.success}, "
        f"correct_culprit={result.correct_culprit}, "
        f"reward={result.reward:.3f}, "
        f"steps={result.steps}"
    )
    return 0


def _default_model(provider: str) -> str:
    if provider == "openai":
        return DEFAULT_OPENAI_MODEL
    return DEFAULT_MODEL


if __name__ == "__main__":
    raise SystemExit(main())
