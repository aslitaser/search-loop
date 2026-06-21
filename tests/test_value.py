import random

import numpy as np

from searchloop.agents import run_mcts
from searchloop.env import Action, Reveal, State, Task
from searchloop.llm import UsageMeter
from searchloop.mcts import MCTSConfig
from searchloop.tools import Tool, ToolRegistry
from searchloop.value import (
    ValueMLP,
    apply_standardizer,
    fit_standardizer,
    load_model,
    make_value_fn,
    save_model,
)


def test_value_mlp_learns_simple_function_better_than_mean() -> None:
    X, y = _linear_dataset()
    mean, std = fit_standardizer(X)
    Xs = apply_standardizer(X, mean, std)
    model = ValueMLP(input_dim=7, hidden_dim=16, seed=1)

    losses = model.train(Xs, y, epochs=1000, lr=0.03)

    baseline_mse = float(np.mean((y.mean() - y) ** 2))
    assert losses[-1] < baseline_mse * 0.2


def test_save_load_round_trip_preserves_predictions(tmp_path) -> None:
    X, y = _linear_dataset()
    mean, std = fit_standardizer(X)
    Xs = apply_standardizer(X, mean, std)
    model = ValueMLP(input_dim=7, hidden_dim=8, seed=2)
    model.train(Xs, y, epochs=300, lr=0.02)
    path = tmp_path / "value_model.json"

    save_model(str(path), model, mean, std)
    loaded, loaded_mean, loaded_std = load_model(str(path))

    np.testing.assert_allclose(loaded.predict(Xs), model.predict(Xs), atol=1e-6)
    np.testing.assert_allclose(loaded_mean, mean, atol=1e-12)
    np.testing.assert_allclose(loaded_std, std, atol=1e-12)


def test_make_value_fn_returns_deterministic_float() -> None:
    X, y = _linear_dataset()
    mean, std = fit_standardizer(X)
    Xs = apply_standardizer(X, mean, std)
    model = ValueMLP(input_dim=7, hidden_dim=8, seed=3)
    model.train(Xs, y, epochs=300, lr=0.02)
    value_fn = make_value_fn(model, mean, std)
    features = X[0].tolist()

    first = value_fn(features)
    second = value_fn(features)

    assert isinstance(first, float)
    assert first == second


def test_standardizer_clips_zero_variance_columns_without_nans() -> None:
    X = np.array(
        [
            [1.0, 2.0, 5.0],
            [3.0, 2.0, 5.0],
            [5.0, 2.0, 5.0],
        ]
    )

    mean, std = fit_standardizer(X)
    Xs = apply_standardizer(X, mean, std)

    np.testing.assert_allclose(mean, np.array([3.0, 2.0, 5.0]))
    assert std[0] > 0
    assert std[1] == 1e-8
    assert std[2] == 1e-8
    assert not np.isnan(Xs).any()


def test_run_mcts_uses_value_fn_without_rollouts_and_is_deterministic(monkeypatch) -> None:
    def poisoned_rollout(*args, **kwargs):
        raise AssertionError("rollout should not be called when value_fn is provided")

    monkeypatch.setattr("searchloop.mcts.rollout", poisoned_rollout)
    task = _winnable_task()
    registry = ToolRegistry(
        [
            _tool("get_logs", {"pod": "Pod to inspect."}),
            _tool("get_metrics", {"query": "Metrics query."}),
            _tool("check_deploy", {"app": "Application to inspect."}),
        ]
    )
    config = MCTSConfig(max_iterations=8, n_candidates=1, rollout_depth=2)

    def value_fn(features: list[float]) -> float:
        return features[0]

    first = run_mcts(
        task,
        registry,
        _GuidedProposer(),
        random.Random(7),
        config,
        value_fn=value_fn,
    )
    second = run_mcts(
        task,
        registry,
        _GuidedProposer(),
        random.Random(7),
        config,
        value_fn=value_fn,
    )

    assert first.final_state.resolved is True
    assert first.correct_culprit is True
    assert first == second


def _linear_dataset() -> tuple[np.ndarray, np.ndarray]:
    x0 = np.linspace(-2.0, 2.0, 80)
    X = np.zeros((80, 7))
    X[:, 0] = x0
    X[:, 1] = x0**2
    y = x0.reshape(-1, 1) + 0.5
    return X, y


def _tool(name: str, params: dict[str, str]) -> Tool:
    return Tool(
        name=name,
        description="Test tool.",
        params=params,
        failure_prob=0.0,
        latency_mean_ms=10,
        latency_std_ms=1,
    )


def _winnable_task() -> Task:
    culprit = "risingwave"
    return Task(
        culprit=culprit,
        required_evidence=frozenset(
            {
                "ev_risingwave_0",
                "ev_risingwave_1",
                "ev_risingwave_2",
            }
        ),
        reveals=(
            Reveal("get_logs", culprit, "ev_risingwave_0"),
            Reveal("get_metrics", culprit, "ev_risingwave_1"),
            Reveal("check_deploy", culprit, "ev_risingwave_2"),
        ),
        max_steps=8,
    )


class _GuidedProposer:
    def __init__(self) -> None:
        self.usage = UsageMeter()

    def propose(self, state: State, n: int) -> list[Action]:
        self.usage.record(0, 0)
        missing = [
            token
            for token in ("ev_risingwave_0", "ev_risingwave_1", "ev_risingwave_2")
            if token not in state.evidence
        ]
        if not missing:
            return [Action.from_dict("resolve", {"target": "risingwave"})][:n]
        if missing[0].endswith("_0"):
            return [Action.from_dict("get_logs", {"pod": "risingwave"})][:n]
        if missing[0].endswith("_1"):
            return [
                Action.from_dict(
                    "get_metrics",
                    {"query": "error_rate{service='risingwave'}"},
                )
            ][:n]
        return [Action.from_dict("check_deploy", {"app": "risingwave"})][:n]
