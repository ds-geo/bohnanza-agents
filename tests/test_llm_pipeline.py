"""Full-pipeline tests for the LLM agents, driven by a mock model.

Every test here runs real games with ``Agent3`` and ``Agent4`` while a scripted
transport stands in for the provider, so the pipeline is exercised without an
API key, a network call, or any cost. An autouse fixture replaces the HTTP
opener with one that fails, which turns an accidental real request into a test
failure rather than a slow, billable one.
"""

import json
import logging
import re

import pytest

from bohnanza import llm, llm_agent
from bohnanza.agents import RuleBasedAgent, StrategicPolicyAgent
from bohnanza.cards import Deck
from bohnanza.events import configure_json_logging, disable_json_logging
from bohnanza.llm import LLMClient, LLMConfig, LLMResponse
from bohnanza.llm_agent import Agent3, Agent4
from bohnanza.player import BeanField, Player
from bohnanza.sim import AGENT_TYPES, play_game, run_batch
from bohnanza.turn import Turn, TurnPhase

LEGAL_INDICES = re.compile(r"indices[^:]*: \[([0-9, ]*)\]")
FACE_UP_CARDS = re.compile(r"Face-up cards to trade away: (.+)")
RESPONDER_HAND = re.compile(r'" hand: (.+)')


class MockTransport:
    """A provider stand-in that answers each prompt with a legal move.

    The reply is chosen from the prompt itself: field questions are answered
    with the first index the prompt lists as legal, a trade swaps the first
    face-up card for the responder's first card, and every offer is accepted.
    That is a weak policy, but a valid one, which is what the pipeline needs to
    run to completion while reaching every decision the agents make.
    """

    def __init__(self, reply: str | None = None) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, prompt, *, model, timeout_seconds, max_output_tokens):
        self.prompts.append(prompt)
        return LLMResponse(
            text=self.reply if self.reply is not None else self._answer(prompt),
            input_tokens=7,
            output_tokens=3,
        )

    @classmethod
    def _answer(cls, prompt: str) -> str:
        if "Accept or reject" in prompt:
            return '{"accept": true}'
        if "Name the card indices" in prompt:
            return cls._offer(prompt)
        listed = LEGAL_INDICES.search(prompt)
        if listed and listed.group(1).strip():
            return '{"field": %s}' % listed.group(1).split(",")[0].strip()
        # No field is legal, so the agent has to fall back to harvest first.
        return '{"field": null}'

    @staticmethod
    def _offer(prompt: str) -> str:
        face_up = FACE_UP_CARDS.search(prompt)
        responder_hand = RESPONDER_HAND.search(prompt)
        if not (face_up and responder_hand):
            return '{"pass": true}'
        if "none" in (face_up.group(1), responder_hand.group(1)):
            return '{"pass": true}'
        return '{"give": [0], "give_source": "face_up", "take": [0]}'


def mock_client(transport: MockTransport) -> LLMClient:
    return LLMClient(
        LLMConfig(model="mock-model", base_url="https://mock.test/v1"),
        transport=transport,
        sleep=lambda _: None,
    )


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Make any real HTTP request fail loudly instead of reaching a provider."""

    def forbidden(*args, **kwargs):
        raise AssertionError("the test suite must not make real API calls")

    monkeypatch.setattr(llm, "urlopen", forbidden)


@pytest.fixture
def mock_llm(monkeypatch):
    """Point every no-argument LLM agent at one shared mock transport."""

    transport = MockTransport()
    monkeypatch.setattr(llm_agent, "default_client", lambda: mock_client(transport))
    return transport


def _player():
    return Player(
        name="P1",
        hand=["Blue", "Red", "Stink"],
        fields=[BeanField(["Blue", "Blue"]), BeanField()],
        coins=5,
    )


def _turn():
    players = [_player(), Player(name="P2", hand=["Chili", "Blue"])]
    turn = Turn(players=players, deck=Deck(seed=1), active_index=0)
    turn.phase = TurnPhase.DRAW_TRADE
    turn.face_up = ["Garden", "Cocoa"]
    return turn


class TestRegistration:
    """The LLM policies are selectable by name like the hand-crafted ones."""

    def test_agent_names(self):
        assert set(AGENT_TYPES) == {"Agent1", "Agent2", "Agent3", "Agent4"}

    def test_agent4_fallback_is_agent1(self):
        assert isinstance(Agent4(mock_client(MockTransport())).fallback, RuleBasedAgent)

    def test_agent3_fallback_is_agent2(self):
        fallback = Agent3(mock_client(MockTransport())).fallback
        assert isinstance(fallback, StrategicPolicyAgent)


class TestMissingConfig:
    """Model settings are required, and the error says which one is missing."""

    @pytest.fixture(autouse=True)
    def no_dotenv(self, monkeypatch):
        # Ignore any real .env so the developer's own keys cannot pass the test.
        monkeypatch.setattr(llm, "_dotenv_value", lambda name: None)
        for name in ("LLM_MODEL", "LLM_BASE_URL"):
            monkeypatch.delenv(name, raising=False)

    def test_missing_model(self):
        with pytest.raises(RuntimeError, match="LLM_MODEL"):
            Agent4()

    def test_missing_base_url(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "mock-model")
        with pytest.raises(RuntimeError, match="LLM_BASE_URL"):
            Agent3()

    def test_config_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "mock-model")
        monkeypatch.setenv("LLM_BASE_URL", "https://mock.test/v1")
        config = LLMConfig.from_env()
        assert (config.model, config.base_url) == (
            "mock-model",
            "https://mock.test/v1",
        )


class TestAgent3CallsModelOnlyToTrade:
    """Agent 3's planting and harvesting stay with the hand-crafted policy."""

    def test_plant_and_harvest_skip_the_model(self):
        transport = MockTransport()
        agent = Agent3(mock_client(transport))
        policy = StrategicPolicyAgent()
        player = _player()
        assert agent.plant(player, "Blue") == policy.plant(player, "Blue")
        assert agent.harvest(player) == policy.harvest(player)
        assert transport.prompts == []

    def test_trades_use_the_model(self):
        transport = MockTransport()
        agent = Agent3(mock_client(transport))
        offer = agent.propose(_turn(), 1)
        assert offer is not None
        assert offer.proposer_face_up is True
        assert len(transport.prompts) == 1


class TestGamesWithoutAPICalls:
    """Complete games run end to end against the mock transport."""

    @pytest.mark.parametrize(
        "matchup",
        [["Agent3", "Agent2"], ["Agent4", "Agent1"], ["Agent3", "Agent4"]],
    )
    def test_game_completes(self, mock_llm, matchup):
        result = play_game(matchup, seed=11)
        assert result.winner in matchup
        assert result.total_turns > 0
        assert set(result.final_scores) == set(matchup)
        assert mock_llm.prompts, "the matchup should have consulted the model"

    def test_batch_completes(self, mock_llm):
        summary = run_batch(["Agent4", "Agent2"], games=2, seed=3)
        assert summary.games == 2
        assert sum(summary.wins.values()) == 2

    def test_same_seed_gives_the_same_result(self, mock_llm):
        first = play_game(["Agent4", "Agent1"], seed=5)
        second = play_game(["Agent4", "Agent1"], seed=5)
        assert first.final_scores == second.final_scores
        assert first.total_turns == second.total_turns

    def test_every_decision_reaches_the_model(self, mock_llm, tmp_path):
        """Two LLM agents cover all four decisions without a single fallback.

        This is what separates a real pipeline run from one where the agents
        quietly played their fallback policy the whole game.
        """

        log_file = tmp_path / "events.jsonl"
        configure_json_logging(str(log_file))
        try:
            play_game(["Agent3", "Agent4"], seed=11)
        finally:
            disable_json_logging()

        events = [json.loads(line) for line in log_file.read_text().splitlines()]
        decisions = {
            event["decision"] for event in events if event["event"] == "llm.decision"
        }
        assert decisions == {"plant", "harvest", "propose", "respond"}
        assert not [event for event in events if event["event"] == "llm.fallback"]

    def test_usage_is_tracked(self, monkeypatch):
        transport = MockTransport()
        client = mock_client(transport)
        monkeypatch.setattr(llm_agent, "default_client", lambda: client)
        play_game(["Agent3", "Agent1"], seed=2)
        assert client.calls == len(transport.prompts)
        assert client.total_input_tokens == 7 * client.calls


class TestUnusableRepliesStillFinish:
    """A model that answers nonsense degrades to the fallback policy."""

    @pytest.fixture
    def fallback_events(self, monkeypatch, tmp_path):
        transport = MockTransport(reply="I would rather not say.")
        monkeypatch.setattr(
            llm_agent, "default_client", lambda: mock_client(transport)
        )
        log_file = tmp_path / "events.jsonl"
        configure_json_logging(str(log_file))
        try:
            yield log_file, play_game(["Agent4", "Agent1"], seed=9)
        finally:
            disable_json_logging()

    def test_game_still_finishes(self, fallback_events):
        _, result = fallback_events
        assert result.total_turns > 0

    def test_fallbacks_are_logged(self, fallback_events):
        log_file, _ = fallback_events
        events = [json.loads(line) for line in log_file.read_text().splitlines()]
        fallbacks = [event for event in events if event["event"] == "llm.fallback"]
        assert fallbacks, "unusable replies should be logged as fallbacks"
        assert {event["decision"] for event in fallbacks} <= {
            "plant",
            "harvest",
            "propose",
            "respond",
        }
        assert all(event["error"] == "LLMParseError" for event in fallbacks)

    def test_prompts_and_replies_stay_out_of_the_log(self, fallback_events):
        log_file, _ = fallback_events
        text = log_file.read_text()
        assert "You are playing Bohnanza" not in text  # no prompt text
        assert "I would rather not say" not in text  # no reply text


@pytest.fixture(autouse=True)
def quiet_event_logger():
    """Leave the shared event logger as the test session found it."""

    logger = logging.getLogger("bohnanza.events")
    handlers = logger.handlers[:]
    level = logger.level
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)
