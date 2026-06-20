from types import SimpleNamespace

import pytest

from searchloop.env import Action, State
from searchloop.llm import (
    SYSTEM,
    AnthropicProposer,
    MockProposer,
    ProposalError,
    parse_actions,
    render_state,
    render_tools,
)
from searchloop.tools import default_registry


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
      {"tool": "get_logs"},
      {"tool": "get_logs", "args": "pod=api"},
      "not an object",
      {"tool": "get_logs", "args": {"pod": "api", "lines": "20"}}
    ]
    """

    actions = parse_actions(text, {"get_logs"})

    assert actions == [Action.from_dict("get_logs", {"pod": "api", "lines": "20"})]


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


def test_parse_actions_raises_when_top_level_is_not_list() -> None:
    with pytest.raises(ProposalError, match="Expected a JSON array"):
        parse_actions('{"tool": "get_logs", "args": {}}', {"get_logs"})


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
    def __init__(self, text: str) -> None:
        self.messages = _FakeMessages(text)


class _FakeMessages:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self._text)],
        )
