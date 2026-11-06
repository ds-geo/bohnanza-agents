"""Tests for LLM agent prompt templates and state serialization."""

from bohnanza.cards import Deck
from bohnanza.llm_agent import (
    harvest_prompt,
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
