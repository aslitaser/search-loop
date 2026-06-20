from types import SimpleNamespace

import pytest

from searchloop.env import Action, Observation, State
from searchloop.llm import (
    DEFAULT_OPENAI_MODEL,
    SYSTEM,
    AnthropicProposer,
    CachingProposer,
    MockProposer,
    OpenAIProposer,
    UsageMeter,
    parse_actions,
    render_state,
    render_tools,
)
from searchloop.tools import ToolResult, default_registry


def test_usage_meter_records_and_resets() -> None:
    meter = UsageMeter()

    meter.record(10, 3)
    meter.record(7, 2)

    assert meter.input_tokens == 17
    assert meter.output_tokens == 5
    assert meter.calls == 2

    meter.reset()

    assert meter.input_tokens == 0
    assert meter.output_tokens == 0
    assert meter.calls == 0


def test_parse_actions_happy_path() -> None:
    text = """
    [
      {"tool": "get_logs", "args": {"pod": "api", "lines": "50"}},
      {"tool": "resolve", "args": {"target": "api"}}
    ]
    """

    actions = parse_actions(text, {"get_logs", "resolve"})

    assert actions == [
        Action.from_dict("get_logs", {"pod": "api", "lines": "50"}),
        Action.from_dict("resolve", {"target": "api"}),
    ]


def test_parse_actions_skips_leading_empty_array_and_uses_later_array() -> None:
    text = '[]\n\n[{"tool":"get_pods","args":{"namespace":"default"}}]'

    actions = parse_actions(text, {"get_pods"})

    assert actions == [Action.from_dict("get_pods", {"namespace": "default"})]


def test_parse_actions_accepts_jsonl_action_objects() -> None:
    text = """
    {"tool": "get_pods", "args": {"namespace": "prod"}}
    {"tool": "resolve", "args": {"target": "auction-engine"}}
    """

    actions = parse_actions(text, {"get_pods", "resolve"})

    assert actions == [
        Action.from_dict("get_pods", {"namespace": "prod"}),
        Action.from_dict("resolve", {"target": "auction-engine"}),
    ]


def test_parse_actions_accepts_prose_around_json() -> None:
    text = """
    Here are the actions:
    [{"tool": "get_pods", "args": {"namespace": "prod"}}]
    Done.
    """

    actions = parse_actions(text, {"get_pods"})

    assert actions == [Action.from_dict("get_pods", {"namespace": "prod"})]


def test_parse_actions_accepts_wrapping_object() -> None:
    text = """
    {
      "actions": [
        {"tool": "get_pods", "args": {"namespace": "prod"}},
        {"tool": "resolve", "args": {"target": "auction-engine"}}
      ]
    }
    """

    actions = parse_actions(text, {"get_pods", "resolve"})

    assert actions == [
        Action.from_dict("get_pods", {"namespace": "prod"}),
        Action.from_dict("resolve", {"target": "auction-engine"}),
    ]


def test_parse_actions_accepts_single_action_object() -> None:
    text = '{"tool":"resolve","args":{"target":"auction-engine"}}'

    actions = parse_actions(text, {"resolve"})

    assert actions == [Action.from_dict("resolve", {"target": "auction-engine"})]


@pytest.mark.parametrize(
    "text",
    [
        '```json\n[{"tool": "get_pods", "args": {"namespace": "prod"}}]\n```',
        '```\n[{"tool": "get_pods", "args": {"namespace": "prod"}}]\n```',
    ],
)
def test_parse_actions_strips_fences(text: str) -> None:
    actions = parse_actions(text, {"get_pods"})

    assert actions == [Action.from_dict("get_pods", {"namespace": "prod"})]


def test_parse_actions_skips_invalid_elements() -> None:
    text = """
    [
      {"tool": "unknown", "args": {"pod": "api"}},
      {"tool": 123, "args": {"pod": "api"}},
      {"args": {"pod": "api"}},
      "not an object",
      {"tool": "get_logs", "args": {"pod": "api", "lines": "20"}}
    ]
    """

    actions = parse_actions(text, {"get_logs"})

    assert actions == [Action.from_dict("get_logs", {"pod": "api", "lines": "20"})]


def test_parse_actions_uses_empty_args_when_args_is_not_a_dict() -> None:
    actions = parse_actions('[{"tool": "get_logs", "args": "pod=api"}]', {"get_logs"})

    assert actions == [Action.from_dict("get_logs", {})]


def test_parse_actions_coerces_arg_values_to_strings() -> None:
    actions = parse_actions('[{"tool": "get_logs", "args": {"lines": 100}}]', {"get_logs"})

    assert actions == [Action.from_dict("get_logs", {"lines": "100"})]


def test_parse_actions_dedups_preserving_order() -> None:
    text = """
    [
      {"tool": "get_logs", "args": {"pod": "api", "lines": "50"}},
      {"tool": "get_logs", "args": {"lines": "50", "pod": "api"}},
      {"tool": "resolve", "args": {"target": "api"}}
    ]
    """

    actions = parse_actions(text, {"get_logs", "resolve"})

    assert actions == [
        Action.from_dict("get_logs", {"pod": "api", "lines": "50"}),
        Action.from_dict("resolve", {"target": "api"}),
    ]


@pytest.mark.parametrize("text", ["", "not json at all", '{"actions": "not-a-list"}'])
def test_parse_actions_returns_empty_for_unparseable_or_irrelevant_input(text: str) -> None:
    assert parse_actions(text, {"get_logs"}) == []


def test_mock_proposer_returns_scripted_batches_and_exhausts() -> None:
    first = [Action.from_dict("get_pods", {"namespace": "prod"})]
    second = [
        Action.from_dict("get_logs", {"pod": "api", "lines": "20"}),
        Action.from_dict("resolve", {"target": "api"}),
    ]
    proposer = MockProposer([first, second])

    assert proposer.propose(State.initial(), 3) == first
    assert proposer.propose(State.initial(), 1) == second[:1]
    with pytest.raises(IndexError, match="exhausted"):
        proposer.propose(State.initial(), 1)


def test_caching_proposer_signature_is_order_invariant() -> None:
    action_a = Action.from_dict("get_logs", {"pod": "risingwave"})
    action_b = Action.from_dict("get_metrics", {"query": "risingwave"})
    first = _state_with_actions([action_a, action_b], evidence=frozenset({"ev_a"}))
    second = _state_with_actions([action_b, action_a], evidence=frozenset({"ev_a"}))
    different = _state_with_actions([action_b, action_a], evidence=frozenset({"ev_b"}))
    proposer = CachingProposer(_CountingProposer([[action_a]]))

    assert proposer._signature(first) == proposer._signature(second)
    assert proposer._signature(first) != proposer._signature(different)


def test_caching_proposer_hit_and_miss_with_transposed_state() -> None:
    action_a = Action.from_dict("get_logs", {"pod": "risingwave"})
    action_b = Action.from_dict("get_metrics", {"query": "risingwave"})
    proposed = Action.from_dict("check_deploy", {"app": "risingwave"})
    first = _state_with_actions([action_a, action_b], evidence=frozenset({"ev_a"}))
    second = _state_with_actions([action_b, action_a], evidence=frozenset({"ev_a"}))
    inner = _CountingProposer([[proposed]])
    proposer = CachingProposer(inner)

    assert proposer.propose(first, 4) == [proposed]
    assert proposer.propose(second, 4) == [proposed]
    assert inner.calls == 1
    assert proposer.misses == 1
    assert proposer.hits == 1


def test_caching_proposer_usage_passthrough() -> None:
    inner = _CountingProposer([[]])
    proposer = CachingProposer(inner)

    assert proposer.usage is inner.usage


def test_caching_proposer_reset_cache_clears_and_remisses() -> None:
    state = State.initial()
    proposed = Action.from_dict("get_pods", {"namespace": "default"})
    inner = _CountingProposer([[proposed], [proposed]])
    proposer = CachingProposer(inner)

    assert proposer.propose(state, 1) == [proposed]
    assert proposer.propose(state, 1) == [proposed]
    assert inner.calls == 1
    assert proposer.hits == 1
    assert proposer.misses == 1

    proposer.reset_cache()

    assert proposer.hits == 0
    assert proposer.misses == 0
    assert proposer.cache == {}
    assert proposer.propose(state, 1) == [proposed]
    assert inner.calls == 2
    assert proposer.hits == 0
    assert proposer.misses == 1


def test_anthropic_proposer_uses_injected_client_and_truncates() -> None:
    registry = default_registry()
    text = """
    [
      {"tool": "get_pods", "args": {"namespace": "prod"}},
      {"tool": "resolve", "args": {"target": "catalog-service"}}
    ]
    """
    fake_client = _FakeClient(text)
    proposer = AnthropicProposer(
        registry,
        briefing="public briefing",
        model="configured-model",
        client=fake_client,
        max_tokens=77,
    )

    actions = proposer.propose(State.initial(), 1)

    assert actions == [Action.from_dict("get_pods", {"namespace": "prod"})]
    call = fake_client.messages.calls[0]
    assert call["model"] == "configured-model"
    assert call["max_tokens"] == 77
    assert call["system"] == SYSTEM


def test_anthropic_proposer_records_usage() -> None:
    fake_client = _FakeClient(
        '[{"tool": "get_pods", "args": {"namespace": "prod"}}]',
        usage=SimpleNamespace(input_tokens=120, output_tokens=30),
    )
    proposer = AnthropicProposer(
        default_registry(),
        briefing="public briefing",
        client=fake_client,
    )

    proposer.propose(State.initial(), 1)

    assert proposer.usage.input_tokens == 120
    assert proposer.usage.output_tokens == 30
    assert proposer.usage.calls == 1


def test_openai_proposer_uses_injected_client_and_truncates() -> None:
    registry = default_registry()
    text = """
    [
      {"tool": "get_pods", "args": {"namespace": "prod"}},
      {"tool": "resolve", "args": {"target": "catalog-service"}}
    ]
    """
    fake_client = _FakeOpenAIClient(text)
    proposer = OpenAIProposer(
        registry,
        briefing="public briefing",
        model="configured-openai-model",
        client=fake_client,
        max_tokens=88,
        temperature=0.4,
    )

    actions = proposer.propose(State.initial(), 1)

    assert actions == [Action.from_dict("get_pods", {"namespace": "prod"})]
    assert proposer.last_raw == text
    call = fake_client.chat.completions.calls[0]
    assert call["model"] == "configured-openai-model"
    assert call["max_tokens"] == 88
    assert call["temperature"] == 0.4
    assert call["messages"][0] == {"role": "system", "content": SYSTEM}
    assert call["messages"][1]["role"] == "user"


def test_openai_proposer_records_usage() -> None:
    fake_client = _FakeOpenAIClient(
        '[{"tool": "get_pods", "args": {"namespace": "prod"}}]',
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30),
    )
    proposer = OpenAIProposer(
        default_registry(),
        briefing="public briefing",
        client=fake_client,
    )

    proposer.propose(State.initial(), 1)

    assert proposer.usage.input_tokens == 120
    assert proposer.usage.output_tokens == 30
    assert proposer.usage.calls == 1


def test_openai_proposer_uses_completion_token_param_for_default_model() -> None:
    fake_client = _FakeOpenAIClient('[{"tool": "get_pods", "args": {"namespace": "prod"}}]')
    proposer = OpenAIProposer(
        default_registry(),
        briefing="public briefing",
        client=fake_client,
        max_tokens=55,
    )

    proposer.propose(State.initial(), 1)

    call = fake_client.chat.completions.calls[0]
    assert call["model"] == DEFAULT_OPENAI_MODEL
    assert call["max_completion_tokens"] == 55
    assert "max_tokens" not in call
    assert "temperature" not in call


def test_render_state_and_tools_on_empty_public_inputs() -> None:
    state_text = render_state(State.initial())
    tools_text = render_tools(default_registry())

    assert state_text
    assert tools_text
    assert "culprit" not in state_text
    assert "ev_" not in state_text
    assert "ev_" not in tools_text
    assert "resolve(target) - declare the culprit service and end the episode" in tools_text


class _FakeClient:
    def __init__(self, text: str, usage: SimpleNamespace | None = None) -> None:
        self.messages = _FakeMessages(text, usage)


def _state_with_actions(actions: list[Action], evidence: frozenset[str]) -> State:
    observations = tuple(
        Observation(
            action=action,
            result=ToolResult(ok=True, output="ok", latency_ms=1.0, error=None),
            evidence_gained=None,
        )
        for action in actions
    )
    return State(
        observations=observations,
        evidence=evidence,
        steps=len(observations),
        resolved=False,
        resolved_target=None,
    )


class _CountingProposer:
    def __init__(self, batches: list[list[Action]]) -> None:
        self._batches = batches
        self.calls = 0
        self.usage = UsageMeter()

    def propose(self, state: State, n: int) -> list[Action]:
        self.usage.record(0, 0)
        batch = self._batches[self.calls]
        self.calls += 1
        return batch[:n]


class _FakeMessages:
    def __init__(self, text: str, usage: SimpleNamespace | None = None) -> None:
        self._text = text
        self._usage = usage
        self.calls = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self._text)],
            usage=self._usage,
        )


class _FakeOpenAIClient:
    def __init__(self, text: str, usage: SimpleNamespace | None = None) -> None:
        self.chat = SimpleNamespace(completions=_FakeOpenAICompletions(text, usage))


class _FakeOpenAICompletions:
    def __init__(self, text: str, usage: SimpleNamespace | None = None) -> None:
        self._text = text
        self._usage = usage
        self.calls = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self._text),
                )
            ],
            usage=self._usage,
        )
