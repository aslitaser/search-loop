import random

from searchloop.ablate import Variant, format_ablation_table, run_ablation
from searchloop.agents import EpisodeResult
from searchloop.env import Action, State, Task
from searchloop.llm import CachingProposer, UsageMeter
from searchloop.mcts import MCTSConfig, MCTSNode, expand_child, is_expandable
from searchloop.metrics import Summary
from searchloop.tools import ToolRegistry


def test_use_pw_flag_disables_child_count_cap() -> None:
    node = MCTSNode(
        state=State.initial(),
        parent=None,
        action=None,
        terminal=False,
        visits=4,
        untried_actions=[Action.from_dict("tool", {"name": "candidate"})],
    )
    expand_child(node, Action.from_dict("tool", {"name": "one"}), State.initial(), False)
    expand_child(node, Action.from_dict("tool", {"name": "two"}), State.initial(), False)

    assert is_expandable(node, MCTSConfig(use_pw=True, pw_k=1.0, pw_alpha=0.5)) is False
    assert is_expandable(node, MCTSConfig(use_pw=False, pw_k=1.0, pw_alpha=0.5)) is True


def test_run_ablation_summarizes_tokens_and_respects_cache_wrapping() -> None:
    proposer = _StubProposer()
    seen_cache_flags = []
    variants = [
        Variant("cached", _build_stub_runner(10, seen_cache_flags), use_cache=True),
        Variant("plain", _build_stub_runner(20, seen_cache_flags), use_cache=False),
    ]

    results = run_ablation(variants, proposer, ToolRegistry([]), seeds=[0, 1])

    assert [label for label, _, _, _ in results] == ["cached", "plain"]
    assert [summary.n for _, summary, _, _ in results] == [2, 2]
    assert [mean_tokens for _, _, mean_tokens, _ in results] == [10.0, 20.0]
    assert seen_cache_flags == [True, True, False, False]


def test_format_ablation_table_contains_labels_and_deltas() -> None:
    full = ("full", _summary(success_rate=0.5, mean_reward=0.2, total_tokens=20), 10.0, 1.0)
    greedy = ("greedy", _summary(success_rate=0.25, mean_reward=0.1, total_tokens=10), 5.0, 2.0)

    text = format_ablation_table([greedy, full], full_label="full")

    assert "greedy" in text
    assert "full" in text
    assert "d_succ" in text
    assert "d_tok" in text
    assert "-0.250" in text
    assert "-5.0" in text


class _StubProposer:
    def __init__(self) -> None:
        self.usage = UsageMeter()

    def propose(self, state: State, n: int):
        return []


def _build_stub_runner(token_count: int, seen_cache_flags: list[bool]):
    def build_runner():
        def runner(
            task: Task,
            registry: ToolRegistry,
            proposer,
            rng: random.Random,
        ) -> EpisodeResult:
            seen_cache_flags.append(isinstance(proposer, CachingProposer))
            proposer.usage.record(token_count, 0)
            return EpisodeResult(
                final_state=State.initial(),
                reward=0.0,
                steps=0,
                success=False,
                correct_culprit=False,
                proposer_calls=1,
            )

        return runner

    return build_runner


def _summary(success_rate: float, mean_reward: float, total_tokens: int) -> Summary:
    return Summary(
        n=2,
        success_rate=success_rate,
        correct_culprit_rate=success_rate,
        mean_reward=mean_reward,
        mean_steps=1.0,
        total_input_tokens=total_tokens,
        total_output_tokens=0,
        tokens_per_success=float(total_tokens),
        mean_sim_latency_ms=0.0,
        mean_wall_ms=1.0,
    )
