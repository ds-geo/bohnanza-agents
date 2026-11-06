"""Tests for LLM agent prompt templates and state serialization."""

import pytest

from bohnanza.cards import Deck
from bohnanza.llm import LLMClient, LLMConfig, LLMResponse, RetryableLLMError
from bohnanza.llm_agent import (
    LLMAgent,
    LLMParseError,
    harvest_prompt,
    parse_action,
    plant_prompt,
    propose_prompt,
    respond_prompt,
    serialize_player,
    serialize_turn,
)
from bohnanza.player import BeanField, Player
from bohnanza.trade import TradeOffer
from bohnanza.turn import Turn, TurnPhase


def _player():
    return Player(
        name="P1",
        hand=["Blue", "Red", "Stink"],
        fields=[BeanField(["Blue", "Blue"]), BeanField()],
        coins=5,
    )


def _turn():
    players = [_player(), Player(name="P2", hand=["Chili", "Wax"])]
    turn = Turn(players=players, deck=Deck(seed=1), active_index=0)
    turn.phase = TurnPhase.DRAW_TRADE
    turn.face_up = ["Garden", "Cocoa"]
    return turn


class TestSerialization:
    """State snapshots are complete and JSON-friendly."""

    def test_player_state(self):
        state = serialize_player(_player(), pending=["Soy"])
        assert state == {
            "name": "P1",
            "coins": 5,
            "hand": ["Blue", "Red", "Stink"],
            "fields": [
                {"bean_type": "Blue", "count": 2, "coins_now": 0},
                {"bean_type": None, "count": 0, "coins_now": 0},
            ],
            "pending": ["Soy"],
        }

    def test_turn_state(self):
        state = serialize_turn(_turn())
        assert state["phase"] == "draw_trade"
        assert state["active_index"] == 0
        assert state["face_up"] == ["Garden", "Cocoa"]
        assert [p["name"] for p in state["players"]] == ["P1", "P2"]


class TestPrompts:
    """Prompts carry the state and legal actions needed to decide."""

    def test_plant_prompt_lists_legal_fields(self):
        prompt = plant_prompt(_player(), "Blue")
        assert "Blue" in prompt
        assert "[0, 1]" in prompt
        assert '{"field"' in prompt

    def test_plant_prompt_flags_harvest_when_blocked(self):
        prompt = plant_prompt(_player(), "Chili")
        assert "harvest" in prompt

    def test_harvest_prompt_lists_occupied_fields(self):
        prompt = harvest_prompt(_player())
        assert "[0]" in prompt
        assert "worth" in prompt

    def test_propose_prompt_includes_both_hands(self):
        prompt = propose_prompt(_turn(), 1)
        assert "0=Garden" in prompt  # face-up indices
        assert "0=Chili" in prompt  # responder hand indices
        assert '"give_source"' in prompt

    def test_respond_prompt_names_offered_beans(self):
        turn = _turn()
        offer = TradeOffer(
            proposer=0,
            responder=1,
            proposer_gives=[0],
            responder_gives=[1],
            proposer_face_up=True,
        )
        prompt = respond_prompt(turn, offer)
        assert "Garden" in prompt  # the face-up card being offered
        assert "Wax" in prompt  # the responder's card being asked for
        assert '"accept"' in prompt


class StubTransport:
    """Transport returning queued reply texts or raising queued errors."""

    def __init__(self, *results):
        self.results = list(results)
        self.prompts = []

    def complete(self, prompt, *, model, timeout_seconds, max_output_tokens):
        self.prompts.append(prompt)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return LLMResponse(text=result, input_tokens=0, output_tokens=0)


def _agent(*results):
    transport = StubTransport(*results)
    client = LLMClient(
        LLMConfig(model="m", base_url="https://example.test/v1"),
        transport=transport,
        sleep=lambda _: None,
    )
    return LLMAgent(client), transport


class TestParseAction:
    """Replies are extracted from prose and fences."""

    def test_plain_json(self):
        assert parse_action('{"field": 1}') == {"field": 1}

    def test_fenced_json(self):
        assert parse_action('```json\n{"accept": true}\n```') == {"accept": True}

    def test_prose_around_object(self):
        reply = 'I think field 0 is best. {"field": 0} Hope that helps!'
        assert parse_action(reply) == {"field": 0}

    def test_nested_object(self):
        assert parse_action('{"a": {"b": 1}}') == {"a": {"b": 1}}

    def test_no_object_raises(self):
        with pytest.raises(LLMParseError):
            parse_action("I cannot decide.")

    def test_non_object_json_raises(self):
        with pytest.raises(LLMParseError):
            parse_action("[1, 2, 3]")


class TestLLMAgent:
    """Replies are validated; bad ones fall back to the rule-based policy."""

    def test_plant_returns_valid_field(self):
        agent, _ = _agent('{"field": 0}')
        assert agent.plant(_player(), "Blue") == 0

    def test_plant_rejects_field_that_wont_accept(self):
        agent, transport = _agent('{"field": 1}')
        player = Player(
            name="P",
            hand=["Blue"],
            fields=[BeanField(["Blue"]), BeanField(["Chili"])],
        )
        # Field 1 exists but holds Chili, so the LLM's choice is illegal;
        # the fallback returns the legal Blue field instead.
        assert agent.plant(player, "Blue") == 0
        assert len(transport.prompts) == 1

    def test_plant_out_of_range_falls_back(self):
        agent, _ = _agent('{"field": 9}')
        player = _player()
        assert agent.plant(player, "Blue") in player.legal_fields("Blue")

    def test_plant_garbage_falls_back(self):
        agent, _ = _agent("no idea")
        player = _player()
        assert agent.plant(player, "Blue") in player.legal_fields("Blue")

    def test_harvest_returns_occupied_field(self):
        agent, _ = _agent('{"field": 0}')
        assert agent.harvest(_player()) == 0

    def test_harvest_empty_field_falls_back(self):
        agent, _ = _agent('{"field": 1}')  # field 1 is empty
        assert agent.harvest(_player()) == 0

    def test_propose_pass_returns_none(self):
        agent, _ = _agent('{"pass": true}')
        assert agent.propose(_turn(), 1) is None

    def test_propose_builds_valid_offer(self):
        agent, _ = _agent(
            '{"give": [0], "give_source": "face_up", "take": [1]}'
        )
        offer = agent.propose(_turn(), 1)
        assert offer is not None
        assert offer.proposer == 0
        assert offer.responder == 1
        assert offer.proposer_gives == [0]
        assert offer.proposer_face_up is True
        assert offer.responder_gives == [1]

    def test_propose_bad_index_falls_back(self):
        agent, _ = _agent('{"give": [7], "take": []}')
        assert agent.propose(_turn(), 1) is None  # fallback never proposes

    def test_propose_empty_offer_falls_back(self):
        agent, _ = _agent('{"give": [], "take": []}')
        assert agent.propose(_turn(), 1) is None

    def test_respond_accepts_boolean(self):
        agent, _ = _agent('{"accept": true}')
        turn = _turn()
        offer = TradeOffer(proposer=0, responder=1, proposer_gives=[0])
        assert agent.respond(turn, offer) is True

    def test_respond_non_boolean_falls_back(self):
        agent, _ = _agent('{"accept": "yes"}')
        turn = _turn()
        offer = TradeOffer(
            proposer=0, responder=1, proposer_gives=[0], responder_gives=[0]
        )
        # Fallback policy rejects offers that ask for cards back.
        assert agent.respond(turn, offer) is False

    def test_transport_error_falls_back(self):
        agent, _ = _agent(RetryableLLMError("down"))
        player = _player()
        assert agent.plant(player, "Blue") in player.legal_fields("Blue")
