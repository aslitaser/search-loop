from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from searchloop.llm import CachingProposer, Proposer
from searchloop.metrics import Runner, Summary, run_benchmark, summarize
from searchloop.tools import ToolRegistry


@dataclass(frozen=True)
class Variant:
    label: str
    build_runner: Callable[[], Runner]
    use_cache: bool


def run_ablation(
    variants: list[Variant],
    base_proposer: Proposer,
    registry: ToolRegistry,
    seeds: Iterable[int],
) -> list[tuple[str, Summary, float, float]]:
    results = []
    seed_list = list(seeds)
    for variant in variants:
        proposer = CachingProposer(base_proposer) if variant.use_cache else base_proposer
        runner = variant.build_runner()
        rows = run_benchmark(runner, proposer, registry, seed_list)
        summary = summarize(rows)
        mean_total_tokens = (
            summary.total_input_tokens + summary.total_output_tokens
        ) / max(1, summary.n)
        results.append((variant.label, summary, mean_total_tokens, summary.mean_wall_ms))

    return results


def format_ablation_table(results: list[tuple[str, Summary, float, float]], full_label: str) -> str:
    reference = _find_result(results, full_label)
    header = (
        f"{'label':<14} {'succ':>7} {'d_succ':>8} {'reward':>8} "
        f"{'tok/ep':>9} {'d_tok':>9} {'tok/succ':>9} {'wall_ms':>9}"
    )
    line = "-" * len(header)
    rows = [header, line]
    for label, summary, mean_total_tokens, mean_wall_ms in results:
        if reference is None:
            delta_success = "n/a"
            delta_tokens = "n/a"
        else:
            _, ref_summary, ref_tokens, _ = reference
            delta_success = f"{summary.success_rate - ref_summary.success_rate:+.3f}"
            delta_tokens = f"{mean_total_tokens - ref_tokens:+.1f}"
        tokens_per_success = (
            "None" if summary.tokens_per_success is None else f"{summary.tokens_per_success:.1f}"
        )
        rows.append(
            f"{label:<14} {summary.success_rate:>7.3f} {delta_success:>8} "
            f"{summary.mean_reward:>8.3f} {mean_total_tokens:>9.1f} "
            f"{delta_tokens:>9} {tokens_per_success:>9} {mean_wall_ms:>9.1f}"
        )

    return "\n".join(rows)


def _find_result(
    results: list[tuple[str, Summary, float, float]],
    label: str,
) -> tuple[str, Summary, float, float] | None:
    for result in results:
        if result[0] == label:
            return result
    return None
