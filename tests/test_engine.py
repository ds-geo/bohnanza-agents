"""Sanity tests for scoring and forced-order mechanics."""

import json
import logging

import pytest

from bohnanza import (
    BEAN_TYPES,
    BEAN_TYPES_BY_NAME,
    BeanField,
    BeanType,
    Deck,
    Game,
    GameConfig,
    Player,
    TradeOffer,
    Agent1,
    Agent2,
    TradeValidator,
    Turn,
    TurnPhase,
    coins_for,
)


class TestScoring:
    """Tests for the coins_for scoring function."""

    def test_scoring_basic(self):
        """Test basic scoring for common bean types."""
        # Wax beans: 4, 7, 9, 11 for 1, 2, 3, 4 coins
        assert coins_for("Wax", 3) == 0
        assert coins_for("Wax", 4) == 1
        assert coins_for("Wax", 7) == 2
        assert coins_for("Wax", 9) == 3
        assert coins_for("Wax", 11) == 4
        assert coins_for("Wax", 15) == 4

    def test_scoring_unreachable_coins(self):
        """Test bean types with unreachable coin values."""
        # Garden beans: None, 2, 3, None (only 2 and 3 coins reachable)
        assert coins_for("Garden", 1) == 0
        assert coins_for("Garden", 2) == 2
        assert coins_for("Garden", 3) == 3
        assert coins_for("Garden", 10) == 3

    def test_scoring_zero_count(self):
        """Test that zero beans earn zero coins."""
        assert coins_for("Blue", 0) == 0

    def test_scoring_negative_count(self):
        """Test that negative counts raise an error."""
        with pytest.raises(ValueError, match="negative"):
            coins_for("Red", -1)

    def test_scoring_unknown_bean(self):
        """Test that unknown bean types raise an error."""
        with pytest.raises(ValueError, match="Unknown bean type"):
            coins_for("MagicBean", 5)

    def test_field_harvest_scoring(self):
        """Test that harvesting a field correctly calculates coins."""
        field = BeanField(beans=["Blue"] * 6)
        earned, cards = field.harvest()
        assert earned == 2  # 6 Blue beans = 2 coins (beanometer: 4, 6, 8, 10)
        assert cards == ["Blue"] * 6
        assert field.beans == []

    def test_player_harvest_adds_coins(self):
        """Test that player harvest adds coins to player total."""
        player = Player(name="Test", coins=0)
        player.fields[0].plant("Blue")
        player.fields[0].plant("Blue")
        player.fields[0].plant("Blue")
        player.fields[0].plant("Blue")

        earned, discarded = player.harvest(0)
        assert earned == 1
        assert player.coins == 1
        assert discarded == ["Blue"] * 3

    def test_harvested_coin_cards_leave_play(self):
        player = Player(name="Test")
        player.fields[0].beans = ["Blue"] * 6

        earned, discarded = player.harvest(0)

        assert earned == 2
        assert discarded == ["Blue"] * 4
        assert player.coins == 2

    def test_deck_counts_each_exhaustion_episode_once(self):
        deck = Deck(bean_types=[BeanType("Test", 1, (1, None, None, None))], seed=1)

        assert deck.draw() == "Test"
        assert deck.draw() is None
        assert deck.draw() is None
        assert deck.times_exhausted == 1

        deck.discard("Test")
        assert deck.draw() == "Test"
        assert deck.draw() is None
        assert deck.times_exhausted == 2

    def test_all_bean_types_have_valid_scoring(self):
        """Test that all defined bean types have valid scoring."""
        for bean_type in BEAN_TYPES:
            # Test at the thresholds
            for i, threshold in enumerate(bean_type.beanometer, start=1):
                if threshold is not None:
                    coins = coins_for(bean_type.name, threshold)
                    assert coins == i
                else:
                    # Unreachable threshold should not give coins
                    coins = coins_for(bean_type.name, 100)
                    assert coins <= 4


class TestForcedOrder:
    """Tests for forced hand order mechanics."""

    def test_hand_order_preserved_on_add(self):
        """Test that adding cards preserves order."""
        player = Player(name="Test")
        player.add_to_hand(["Blue", "Red", "Green"])
        assert player.hand == ["Blue", "Red", "Green"]

    def test_plant_from_front_removes_first(self):
        """Test that planting from front removes the first card."""
        player = Player(name="Test")
        player.add_to_hand(["Blue", "Red", "Green"])
        bean = player.plant_from_front(0)
        assert bean == "Blue"
        assert player.hand == ["Red", "Green"]

    def test_plant_from_front_preserves_rest(self):
        """Test that planting from front preserves order of remaining cards."""
        player = Player(name="Test")
        player.add_to_hand(["Blue", "Blue", "Blue", "Blue"])
        player.plant_from_front(0)
        assert player.hand == ["Blue", "Blue", "Blue"]
        player.plant_from_front(0)
        assert player.hand == ["Blue", "Blue"]

    def test_cannot_plant_from_empty_hand(self):
        """Test that planting from empty hand raises an error."""
        player = Player(name="Test")
        with pytest.raises(ValueError, match="empty hand"):
            player.plant_from_front(0)

    def test_remove_from_index_preserves_order(self):
        """Test that removing from an index preserves order of other cards."""
        player = Player(name="Test")
        player.add_to_hand(["Blue", "Red", "Green", "Wax"])
        bean = player.remove_from_hand(2)
        assert bean == "Green"
        assert player.hand == ["Blue", "Red", "Wax"]

    def test_forced_planting_in_turn(self):
        """Test that turn enforces planting from front of hand."""
        players = [
            Player(name="Alice", hand=["Blue", "Red", "Green"]),
            Player(name="Bob", hand=["Wax", "Chili"]),
        ]
        deck = Deck(seed=42)
        turn = Turn(players=players, deck=deck, active_index=0)

        # Plant first card
        bean = turn.plant_from_hand(0)
        assert bean == "Blue"
        assert players[0].hand == ["Red", "Green"]

        # Plant second card
        bean = turn.plant_from_hand(1)
        assert bean == "Red"
        assert players[0].hand == ["Green"]

    def test_must_plant_first_card(self):
        """Test that the first card must be planted if hand is not empty."""
        players = [
            Player(name="Alice", hand=["Blue", "Red", "Green"]),
            Player(name="Bob", hand=["Wax", "Chili"]),
        ]
        deck = Deck(seed=42)
        turn = Turn(players=players, deck=deck, active_index=0)

        # Try to end plant phase without planting
        with pytest.raises(ValueError, match="first card"):
            turn.end_plant_phase()

    def test_empty_hand_allows_skipping(self):
        """Test that empty hand allows skipping planting."""
        players = [
            Player(name="Alice", hand=[]),
            Player(name="Bob", hand=["Wax", "Chili"]),
        ]
        deck = Deck(seed=42)
        turn = Turn(players=players, deck=deck, active_index=0)

        # Should be able to end plant phase with empty hand
        face_up = turn.end_plant_phase()
        assert len(face_up) == 2  # Two face-up cards drawn

    def test_hand_cannot_be_reordered(self):
        """Test that there is no method to reorder the hand."""
        player = Player(name="Test")
        player.add_to_hand(["Blue", "Red", "Green"])

        # The only way to remove cards is by index, which preserves order
        # There is no insert, reverse, or sort method
        player.remove_from_hand(0)
        assert player.hand == ["Red", "Green"]
        player.remove_from_hand(0)
        assert player.hand == ["Green"]


class TestGameIntegration:
    """Integration tests for the full game loop."""

    def test_game_initialization(self):
        """Test that a game initializes correctly."""
        config = GameConfig(player_names=["Alice", "Bob"], seed=42)
        game = Game(config)

        assert len(game.players) == 2
        assert game.players[0].name == "Alice"
        assert game.players[1].name == "Bob"
        assert all(len(p.hand) == 5 for p in game.players)
        assert game.active_index == 0
        assert not game.is_over

    def test_turn_rotation(self):
        """Test that turns rotate correctly."""
        config = GameConfig(player_names=["Alice", "Bob", "Charlie"], seed=42)
        game = Game(config)

        assert game.active_index == 0
        game.active_index = (game.active_index + 1) % len(game.players)
        assert game.active_index == 1
        game.active_index = (game.active_index + 1) % len(game.players)
        assert game.active_index == 2
        game.active_index = (game.active_index + 1) % len(game.players)
        assert game.active_index == 0

    def test_game_state_summary(self):
        """Test that game state can be summarized."""
        config = GameConfig(player_names=["Alice", "Bob"], seed=42)
        game = Game(config)
        state = game.get_game_state()

        assert "active_player" in state
        assert "turn_count" in state
        assert "deck_exhaustions" in state
        assert "players" in state
        assert len(state["players"]) == 2

    def test_player_state_summary(self):
        """Test that player state can be summarized."""
        config = GameConfig(player_names=["Alice", "Bob"], seed=42)
        game = Game(config)
        state = game.get_player_state("Alice")

        assert state["name"] == "Alice"
        assert "coins" in state
        assert "hand" in state
        assert "fields" in state
        assert len(state["fields"]) == 2

    def test_invalid_player_count(self):
        """Test that invalid player counts are rejected."""
        with pytest.raises(ValueError, match="at least 2"):
            GameConfig(player_names=["Alice"])

        with pytest.raises(ValueError, match="at most 7"):
            GameConfig(player_names=[f"P{i}" for i in range(8)])

    def test_duplicate_player_names(self):
        """Test that duplicate player names are rejected."""
        with pytest.raises(ValueError, match="unique"):
            GameConfig(player_names=["Alice", "Alice"])

    def test_agent1_vs_agent2_full_game_conserves_cards(self):
        """One complete game should finish without illegal moves or lost cards."""
        game = Game(GameConfig(player_names=["Agent 1", "Agent 2"], seed=42))
        agents = [Agent1(), Agent2()]
        initial_card_total = sum(bean.card_count for bean in BEAN_TYPES)

        def assert_conserved(turn=None):
            cards_in_play = game.deck.total_count
            cards_in_play += sum(len(player.hand) for player in game.players)
            cards_in_play += sum(
                len(bean_field.beans)
                for player in game.players
                for bean_field in player.fields
            )
            cards_in_play += sum(player.coins for player in game.players)
            if turn is not None:
                cards_in_play += len(turn.face_up)
                cards_in_play += sum(
                    len(cards) for cards in turn.pending.values()
                )
            assert cards_in_play == initial_card_total

        assert_conserved()
        while not game.is_over:
            turn = game.start_turn()
            active_index = game.active_index
            active = game.players[active_index]
            active_agent = agents[active_index]

            for _ in range(2):
                if not active.hand:
                    break
                bean = active.hand[0]
                while not active.legal_fields(bean):
                    field_index = active_agent.harvest(active)
                    assert field_index is not None
                    turn.harvest(active_index, field_index)
                    assert_conserved(turn)
                field_index = active_agent.plant(active, bean)
                turn.plant_from_hand(field_index)
                assert_conserved(turn)

            turn.end_plant_phase()
            assert_conserved(turn)
            validator = TradeValidator(game.players, active_index, len(turn.face_up))
            for responder_index in range(len(game.players)):
                if responder_index == active_index:
                    continue
                offer = active_agent.propose(turn, responder_index)
                if offer is None:
                    continue
                is_valid, error = validator.validate(offer)
                assert is_valid, error
                if not agents[responder_index].respond(turn, offer):
                    offer.reject()
                    continue
                offer.accept()
                can_execute, error = validator.can_execute(offer)
                assert can_execute, error
                turn.transfer(
                    offer.proposer,
                    offer.responder,
                    hand_indices=() if offer.proposer_face_up else offer.proposer_gives,
                    face_up_indices=offer.proposer_gives if offer.proposer_face_up else (),
                )
                turn.transfer(
                    offer.responder,
                    offer.proposer,
                    hand_indices=offer.responder_gives,
                )
                assert_conserved(turn)

            turn.end_trade_phase()
            assert_conserved(turn)
            for player_index, player in enumerate(game.players):
                while turn.pending[player_index]:
                    bean = turn.pending[player_index][0]
                    while not player.legal_fields(bean):
                        field_index = agents[player_index].harvest(player)
                        assert field_index is not None
                        turn.harvest(player_index, field_index)
                        assert_conserved(turn)
                    field_index = agents[player_index].plant(player, bean)
                    turn.plant_pending(player_index, 0, field_index)
                    assert_conserved(turn)

            turn.end_plant_traded_phase()
            turn.draw_three()
            assert_conserved(turn)
            game.end_turn()

        result = game.finalize()
        assert result.total_turns == game.total_turns
        assert result.deck_exhaustions >= game.config.max_deck_exhaustions
        assert_conserved()


class TestStructuredLogging:
    """Structured events are valid JSON and include game state details."""

    def test_offer_status_and_player_state_are_json_logged(self, caplog):
        caplog.set_level(logging.INFO, logger="bohnanza.events")
        players = [
            Player(name="Alice", hand=["Blue"]),
            Player(name="Bob", hand=["Red"]),
        ]
        turn = Turn(players=players, deck=Deck(seed=7), active_index=0)

        accepted = TradeOffer(
            proposer=0,
            responder=1,
            proposer_gives=[0],
            responder_gives=[0],
        )
        accepted.accept()
        rejected = TradeOffer(proposer=1, responder=0, proposer_gives=[0])
        rejected.reject()
        turn.plant_from_hand(0)

        events = [json.loads(record.message) for record in caplog.records]
        event_names = [event["event"] for event in events]
        assert event_names == [
            "trade.offered",
            "trade.accepted",
            "trade.offered",
            "trade.rejected",
            "player.planted",
        ]
        planted = events[-1]
        assert planted["player"] == "Alice"
        assert planted["hand"] == []
        assert planted["fields"][0]["beans"] == ["Blue"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
