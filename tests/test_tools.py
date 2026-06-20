import random

import pytest

from searchloop.tools import Tool, default_registry


def _args_for(tool_name: str) -> dict[str, str]:
    return {
        "get_pods": {"namespace": "prod"},
        "get_logs": {"pod": "api-123", "lines": "50"},
        "get_metrics": {"query": "rate(http_requests_total[5m])"},
        "describe_node": {"node": "worker-1"},
        "check_deploy": {"app": "api"},
        "restart_pod": {"pod": "api-123"},
    }[tool_name]


def test_execute_is_reproducible_for_same_seed() -> None:
    registry = default_registry()
    calls = registry.names() * 5

    first_rng = random.Random(42)
    second_rng = random.Random(42)

    first_results = [registry.get(name).execute(_args_for(name), first_rng) for name in calls]
    second_results = [registry.get(name).execute(_args_for(name), second_rng) for name in calls]

    assert first_results == second_results


def test_different_seeds_generally_differ() -> None:
    registry = default_registry()
    calls = registry.names() * 20

    first_rng = random.Random(1)
    second_rng = random.Random(2)

    first_results = [registry.get(name).execute(_args_for(name), first_rng) for name in calls]
    second_results = [registry.get(name).execute(_args_for(name), second_rng) for name in calls]

    assert any(first != second for first, second in zip(first_results, second_results, strict=True))


def test_failure_probability_extremes() -> None:
    never_fails = Tool(
        name="never_fails",
        description="Always succeeds for probability-boundary testing.",
        params={},
        failure_prob=0.0,
        latency_mean_ms=10,
        latency_std_ms=1,
    )
    always_fails = Tool(
        name="always_fails",
        description="Always fails for probability-boundary testing.",
        params={},
        failure_prob=1.0,
        latency_mean_ms=10,
        latency_std_ms=1,
    )

    never_rng = random.Random(123)
    always_rng = random.Random(123)

    assert all(never_fails.execute({}, never_rng).ok for _ in range(1000))
    assert all(not always_fails.execute({}, always_rng).ok for _ in range(1000))


def test_latency_is_clamped_to_non_negative() -> None:
    tool = Tool(
        name="high_std",
        description="Produces high-variance latency for clamp testing.",
        params={},
        failure_prob=0.0,
        latency_mean_ms=5,
        latency_std_ms=500,
    )
    rng = random.Random(456)

    assert all(tool.execute({}, rng).latency_ms >= 0.0 for _ in range(1000))


def test_default_registry() -> None:
    registry = default_registry()

    expected_names = [
        "check_deploy",
        "describe_node",
        "get_logs",
        "get_metrics",
        "get_pods",
        "restart_pod",
    ]

    assert registry.names() == expected_names
    assert len(registry) == 6
    assert "get_pods" in registry

    with pytest.raises(KeyError, match="Unknown tool 'missing'"):
        registry.get("missing")
