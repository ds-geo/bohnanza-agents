"""The four phases of a single Bohnanza turn.

``Turn`` tracks one turn's state and rejects moves made in the wrong phase. It
never decides anything: callers pass in the field and card indices, and later
agent code chooses them. Rotating the active player and detecting the end of the
game belong to the game loop.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable, Sequence

from .cards import Deck
from .player import Player

PLANTS_PER_TURN = 2
FACE_UP_CARDS = 2
CARDS_DRAWN_AT_END = 3


class TurnPhase(Enum):
    """The phases of a turn, in the order they are played."""

    PLANT = "plant"
    DRAW_TRADE = "draw_trade"
    PLANT_TRADED = "plant_traded"
    DRAW_THREE = "draw_three"
    DONE = "done"


class Turn:
    """One player's turn, advanced one phase at a time."""

    def __init__(self, players: Sequence[Player], deck: Deck, active_index: int) -> None:
        if not players:
            raise ValueError("A turn requires at least one player")
        if not 0 <= active_index < len(players):
            raise IndexError(f"No player at index {active_index}")

        self.players = list(players)
        self.deck = deck
        self.active_index = active_index
        self.phase = TurnPhase.PLANT
        self.face_up: list[str] = []
        self.pending: dict[int, list[str]] = {
            index: [] for index in range(len(self.players))
        }
        self.plants_this_phase = 0
        self._started_with_empty_hand = not self.active_player.hand

    @property
    def active_player(self) -> Player:
        """The player whose turn this is."""

        return self.players[self.active_index]

    # Phase 1: plant from the front of the hand.

    def plant_from_hand(self, field_index: int) -> str:
        """Plant the front card of the active player's hand."""

        self._require_phase(TurnPhase.PLANT)
        if self.plants_this_phase >= PLANTS_PER_TURN:
            raise ValueError(f"Already planted {PLANTS_PER_TURN} cards this turn")
        bean = self.active_player.plant_from_front(field_index)
        self.plants_this_phase += 1
        return bean

    def end_plant_phase(self) -> list[str]:
        """Finish planting and flip the face-up cards, returning them."""

        self._require_phase(TurnPhase.PLANT)
        if self.plants_this_phase == 0 and not self._started_with_empty_hand:
            raise ValueError("The first card in hand must be planted")

        for _ in range(FACE_UP_CARDS):
            bean = self.deck.draw()
            if bean is None:
                break
            self.face_up.append(bean)
        self.phase = TurnPhase.DRAW_TRADE
        return list(self.face_up)

    # Phase 2: trade and donate.

    def transfer(
        self,
        from_player: int,
        to_player: int,
        hand_indices: Iterable[int] = (),
        face_up_indices: Iterable[int] = (),
    ) -> list[str]:
        """Move cards to another player's pile of cards awaiting planting.

        This is the primitive behind both trades and donations; a trade is two
        transfers. Validated trade objects will call it in a later commit.
        """

        self._require_phase(TurnPhase.DRAW_TRADE)
        if not 0 <= from_player < len(self.players):
            raise IndexError(f"No player at index {from_player}")
        if not 0 <= to_player < len(self.players):
            raise IndexError(f"No player at index {to_player}")
        if from_player == to_player:
            raise ValueError("Cannot transfer cards to the giving player")

        hand = self._unique_indices(hand_indices, len(self.players[from_player].hand))
        face_up = self._unique_indices(face_up_indices, len(self.face_up))
        if face_up and from_player != self.active_index:
            raise ValueError("Only the active player can give away face-up cards")

        # Remove from the highest index down so earlier indices stay valid.
        moved = [self.face_up.pop(index) for index in reversed(face_up)]
        moved += [
            self.players[from_player].remove_from_hand(index)
            for index in reversed(hand)
        ]
        moved.reverse()
        self.pending[to_player].extend(moved)
        return moved

    def end_trade_phase(self) -> None:
        """Finish trading; the active player keeps any unclaimed face-up cards."""

        self._require_phase(TurnPhase.DRAW_TRADE)
        self.pending[self.active_index].extend(self.face_up)
        self.face_up = []
        self.phase = TurnPhase.PLANT_TRADED

    # Phase 3: plant the traded and face-up cards.

    def plant_pending(self, player_index: int, card_index: int, field_index: int) -> str:
        """Plant one awaiting card, which may be planted in any order."""

        self._require_phase(TurnPhase.PLANT_TRADED)
        try:
            cards = self.pending[player_index]
        except KeyError as exc:
            raise IndexError(f"No player at index {player_index}") from exc
        try:
            bean = cards[card_index]
        except IndexError as exc:
            raise IndexError(f"No awaiting card at index {card_index}") from exc

        # Validate before removing the card, so a failed planting leaves state intact.
        self.players[player_index].plant(bean, field_index)
        del cards[card_index]
        return bean

    def end_plant_traded_phase(self) -> None:
        """Finish planting traded cards once every player's pile is empty."""

        self._require_phase(TurnPhase.PLANT_TRADED)
        unplanted = [index for index, cards in self.pending.items() if cards]
        if unplanted:
            raise ValueError(
                "All traded cards must be planted before the phase ends; "
                f"players {unplanted} still hold cards"
            )
        self.phase = TurnPhase.DRAW_THREE

    # Phase 4: refill the hand.

    def draw_three(self) -> list[str]:
        """Draw cards onto the back of the active player's hand, ending the turn."""

        self._require_phase(TurnPhase.DRAW_THREE)
        drawn: list[str] = []
        for _ in range(CARDS_DRAWN_AT_END):
            bean = self.deck.draw()
            if bean is None:
                break
            drawn.append(bean)
        self.active_player.add_to_hand(drawn)
        self.phase = TurnPhase.DONE
        return drawn

    # Harvesting, which any player may do at any point in the turn.

    def harvest(self, player_index: int, field_index: int) -> int:
        """Harvest a field, discarding unconverted cards and returning coins."""

        if self.phase is TurnPhase.DONE:
            raise ValueError("Cannot harvest after the turn has ended")
        try:
            player = self.players[player_index]
        except IndexError as exc:
            raise IndexError(f"No player at index {player_index}") from exc

        earned, cards = player.harvest(field_index)
        for bean in cards:
            self.deck.discard(bean)
        return earned

    def _require_phase(self, expected: TurnPhase) -> None:
        if self.phase is not expected:
            raise ValueError(
                f"Expected the {expected.value} phase, but the turn is in the "
                f"{self.phase.value} phase"
            )

    @staticmethod
    def _unique_indices(indices: Iterable[int], size: int) -> list[int]:
        ordered = sorted(indices)
        if len(set(ordered)) != len(ordered):
            raise ValueError("Card indices must be unique")
        for index in ordered:
            if not 0 <= index < size:
                raise IndexError(f"No card at index {index}")
        return ordered
