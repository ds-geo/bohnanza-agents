"""Game loop, end-of-game detection, and final scoring for Bohnanza."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .cards import Deck, coins_for
from .events import log_event
from .player import Player
from .turn import Turn, TurnPhase


@dataclass
class GameConfig:
    """Configuration options for a Bohnanza game."""

    player_names: list[str]
    """Names of the players in the game."""

    seed: int | None = None
    """Random seed for deck shuffling (for reproducible games)."""

    max_deck_exhaustions: int = 3
    """Number of times the deck can be exhausted before the game ends."""

    def __post_init__(self) -> None:
        if len(self.player_names) < 2:
            raise ValueError("Bohnanza requires at least 2 players")
        if len(self.player_names) > 7:
            raise ValueError("Bohnanza supports at most 7 players")
        if len(set(self.player_names)) != len(self.player_names):
            raise ValueError("Player names must be unique")


@dataclass
class GameResult:
    """The result of a completed game."""

    winner: str
    """Name of the winning player."""

    final_scores: dict[str, int]
    """Final coin totals for all players."""

    total_turns: int
    """Total number of turns played."""

    deck_exhaustions: int
    """Number of times the deck was exhausted."""


class Game:
    """A complete Bohnanza game with turn rotation and end detection."""

    def __init__(self, config: GameConfig) -> None:
        """Initialize a new game with the given configuration."""

        self.config = config
        self.deck = Deck(seed=config.seed)
        self.players = [
            Player(name=name) for name in config.player_names
        ]
        self.active_index = 0
        self.turn: Turn | None = None
        self.total_turns = 0
        self.is_over = False

        # Deal initial hands (5 cards per player)
        for player in self.players:
            for _ in range(5):
                card = self.deck.draw()
                if card is None:
                    raise RuntimeError("Deck exhausted during initial deal")
                player.add_to_hand([card])

    @property
    def active_player(self) -> Player:
        """The player whose turn it currently is."""

        return self.players[self.active_index]

    @property
    def deck_exhaustions(self) -> int:
        """Number of times drawing found both piles empty."""

        return self.deck.times_exhausted

    def start_turn(self) -> Turn:
        """Begin a new turn for the active player."""

        if self.is_over:
            raise RuntimeError("Cannot start a turn after the game has ended")

        self.turn = Turn(
            players=self.players,
            deck=self.deck,
            active_index=self.active_index,
        )
        return self.turn

    def end_turn(self) -> None:
        """Complete the current turn and rotate to the next player."""

        if self.turn is None:
            raise RuntimeError("No turn in progress")
        if self.turn.phase is not TurnPhase.DONE:
            raise RuntimeError("Cannot end a turn before it is complete")

        self.total_turns += 1
        self.active_index = (self.active_index + 1) % len(self.players)
        self.turn = None

        # Check for game end
        if self.deck.times_exhausted >= self.config.max_deck_exhaustions:
            self.is_over = True

    def check_game_over(self) -> bool:
        """Check if the game should end based on deck exhaustion."""

        return self.deck.times_exhausted >= self.config.max_deck_exhaustions

    def finalize(self) -> GameResult:
        """Harvest all remaining fields and compute final scores."""

        if not self.is_over:
            raise RuntimeError("Cannot finalize a game that is not over")

        final_scores = {player.name: player.coins for player in self.players}

        # Harvest all remaining fields
        for player in self.players:
            for field_index in range(len(player.fields)):
                if player.fields[field_index].beans:
                    earned, cards = player.harvest(field_index)
                    for bean in cards:
                        self.deck.discard(bean)
                    final_scores[player.name] += earned
                    log_event(
                        "player.final_harvested",
                        player=player.name,
                        field_index=field_index,
                        discarded=cards,
                        coins_earned=earned,
                        hand=list(player.hand),
                        fields=[
                            {
                                "bean_type": bean_field.bean_type,
                                "beans": list(bean_field.beans),
                            }
                            for bean_field in player.fields
                        ],
                        coins=player.coins,
                    )

        # Determine winner
        winner = max(final_scores, key=final_scores.get)
        max_score = final_scores[winner]
        winners = [name for name, score in final_scores.items() if score == max_score]

        # In case of tie, the winner is the one with the most cards in hand
        if len(winners) > 1:
            winner = max(
                winners,
                key=lambda name: len(
                    next(p for p in self.players if p.name == name).hand
                )
            )

        log_event(
            "game.scored",
            winner=winner,
            scores=final_scores,
            total_turns=self.total_turns,
            deck_exhaustions=self.deck.times_exhausted,
        )

        return GameResult(
            winner=winner,
            final_scores=final_scores,
            total_turns=self.total_turns,
            deck_exhaustions=self.deck.times_exhausted,
        )

    def get_player_state(self, player_name: str) -> dict:
        """Get a summary of a player's current state."""

        player = next(p for p in self.players if p.name == player_name)
        return {
            "name": player.name,
            "coins": player.coins,
            "hand": list(player.hand),
            "fields": [
                {
                    "bean_type": field.bean_type,
                    "count": len(field.beans),
                }
                for field in player.fields
            ],
        }

    def get_game_state(self) -> dict:
        """Get a summary of the current game state."""

        return {
            "active_player": self.active_player.name,
            "turn_count": self.total_turns,
            "deck_exhaustions": self.deck_exhaustions,
            "deck_remaining": self.deck.draw_count,
            "is_over": self.is_over,
            "players": [self.get_player_state(p.name) for p in self.players],
        }
