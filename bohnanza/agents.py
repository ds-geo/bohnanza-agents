"""Decision policies for players in a Bohnanza game."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .cards import coins_for
from .player import Player
from .trade import TradeOffer
from .turn import Turn


class Agent(ABC):
    """Interface implemented by objects that choose a player's actions.

    Agents return decisions; the game or turn controller remains responsible
    for applying them and enforcing the rules.
    """

    @abstractmethod
    def plant(self, player: Player, bean: str) -> int:
        """Choose a field index for the bean being planted."""

    @abstractmethod
    def propose(self, turn: Turn, responder_index: int) -> TradeOffer | None:
        """Choose a trade offer for another player, or return ``None``."""

    @abstractmethod
    def respond(self, turn: Turn, offer: TradeOffer) -> bool:
        """Accept or reject a pending trade offer."""

    @abstractmethod
    def harvest(self, player: Player) -> int | None:
        """Choose a field to harvest, or return ``None`` if none should be."""


class RuleBasedAgent(Agent):
    """A deterministic baseline policy (Agent 1).

    It plants into a matching field first, then an empty field, and otherwise
    chooses the field with the lowest current coin yield. It only trades when
    an incoming bean can be planted directly and never proposes a trade. It
    harvests the weakest field when a planting decision needs a field freed.
    """

    def plant(self, player: Player, bean: str) -> int:
        legal = player.legal_fields(bean)
        if legal:
            # Prefer a field already growing this bean; among ties, lowest index.
            matching = [i for i in legal if player.fields[i].bean_type == bean]
            return matching[0] if matching else legal[0]
        return self._weakest_field(player)

    def propose(self, turn: Turn, responder_index: int) -> TradeOffer | None:
        return None

    def respond(self, turn: Turn, offer: TradeOffer) -> bool:
        if offer.status.value != "pending" or offer.responder_gives:
            return False
        try:
            incoming = self._incoming_beans(turn, offer)
        except (IndexError, KeyError):
            return False
        responder = turn.players[offer.responder]
        # Accept only when every received bean can be planted without a harvest.
        return bool(incoming) and all(responder.legal_fields(bean) for bean in incoming)

    def harvest(self, player: Player) -> int | None:
        occupied = [i for i, bean_field in enumerate(player.fields) if bean_field.beans]
        if not occupied:
            return None
        return min(occupied, key=lambda i: (coins_for(
            player.fields[i].bean_type or "", len(player.fields[i].beans)
        ), len(player.fields[i].beans), i))

    @staticmethod
    def _weakest_field(player: Player) -> int:
        if not player.fields:
            raise ValueError("Cannot choose a field for a player with no fields")
        return min(
            range(len(player.fields)),
            key=lambda i: (
                coins_for(player.fields[i].bean_type, len(player.fields[i].beans))
                if player.fields[i].bean_type
                else -1,
                len(player.fields[i].beans),
                i,
            ),
        )

    @staticmethod
    def _incoming_beans(turn: Turn, offer: TradeOffer) -> list[str]:
        """Resolve the offer's cards from the current turn state."""
        proposer = turn.players[offer.proposer]
        if offer.proposer_face_up:
            source = turn.face_up
        else:
            source = proposer.hand
        return [source[i] for i in offer.proposer_gives]


# A short alias makes the baseline convenient to select in game setups.
Agent1 = RuleBasedAgent

