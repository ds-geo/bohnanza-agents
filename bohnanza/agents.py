"""Decision policies for players in a Bohnanza game."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .cards import coins_for
from .player import Player
from .trade import TradeOffer, TradeStatus
from .turn import Turn

URGENCY_WINDOW = 3
"""How many cards at the front of the hand count toward urgency."""


def _urgency_weight(index: int) -> int:
    """Higher weight for cards closer to the front, 0 outside the window."""

    return max(0, URGENCY_WINDOW - index)


def _is_blocked(player: Player, bean: str) -> bool:
    """Whether planting this bean right now would require a harvest first."""

    return player.needs_harvest_to_plant(bean)


def _urgency_score(player: Player, hand: list[str] | None = None) -> int:
    """Sum of front-window weights for hand cards that are currently blocked.

    A hypothetical ``hand`` can be supplied to score a hand with cards removed
    without mutating the player. Fields are always the player's real fields.
    """

    cards = player.hand if hand is None else hand
    return sum(
        _urgency_weight(index)
        for index, bean in enumerate(cards[:URGENCY_WINDOW])
        if _is_blocked(player, bean)
    )


def _most_urgent_index(player: Player) -> int | None:
    """Index of the frontmost blocked card within the window, or ``None``."""

    for index, bean in enumerate(player.hand[:URGENCY_WINDOW]):
        if _is_blocked(player, bean):
            return index
    return None


def _offered_beans(turn: Turn, offer: TradeOffer) -> list[str]:
    """Resolve the cards the proposer is giving from the current turn state."""

    source = turn.face_up if offer.proposer_face_up else turn.players[offer.proposer].hand
    return [source[index] for index in offer.proposer_gives]


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
        if offer.status is not TradeStatus.PENDING or offer.responder_gives:
            return False
        try:
            incoming = _offered_beans(turn, offer)
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


class StrategicPolicyAgent(RuleBasedAgent):
    """A stronger hand-crafted policy (Agent 2).

    It plants like Agent 1 but adds a hand-order urgency heuristic: a bean that
    no field can accept and that is close to the front of the hand is a bean
    worth trading away, because planting it will soon force a harvest. The agent
    offers such beans in exchange for cards that fit its existing fields, avoids
    harvesting fields it is about to reinforce, and accepts any trade that is
    free to plant and does not make its own hand more urgent.
    """

    def propose(self, turn: Turn, responder_index: int) -> TradeOffer | None:
        proposer_index = turn.active_index
        proposer = turn.players[proposer_index]
        if responder_index == proposer_index:
            return None
        try:
            responder = turn.players[responder_index]
        except IndexError:
            return None

        urgent_index = _most_urgent_index(proposer)
        if urgent_index is None:
            return None

        # Ask for a bean already growing in a field, so it can be planted
        # without displacing anything.
        planted = {
            bean_field.bean_type for bean_field in proposer.fields if bean_field.bean_type
        }
        requested_index = next(
            (i for i, bean in enumerate(responder.hand) if bean in planted), None
        )
        if requested_index is None:
            return None

        return TradeOffer(
            proposer=proposer_index,
            responder=responder_index,
            proposer_gives=[urgent_index],
            responder_gives=[requested_index],
        )

    def respond(self, turn: Turn, offer: TradeOffer) -> bool:
        if offer.status is not TradeStatus.PENDING:
            return False
        try:
            incoming = _offered_beans(turn, offer)
            responder = turn.players[offer.responder]
        except (IndexError, KeyError):
            return False
        if not incoming or not self._plantable_for_free(responder, incoming):
            return False
        if not offer.responder_gives:
            return True

        outgoing = set(offer.responder_gives)
        if any(index >= len(responder.hand) for index in outgoing):
            return False
        remaining = [
            bean for index, bean in enumerate(responder.hand) if index not in outgoing
        ]
        # Giving cards away is only worth it if the hand does not get more urgent.
        return _urgency_score(responder, hand=remaining) <= _urgency_score(responder)

    def harvest(self, player: Player) -> int | None:
        occupied = [i for i, bean_field in enumerate(player.fields) if bean_field.beans]
        if not occupied:
            return None
        # Keep fields that the cards near the front of the hand will reinforce.
        upcoming = set(player.hand[:URGENCY_WINDOW])
        candidates = [
            i for i in occupied if player.fields[i].bean_type not in upcoming
        ] or occupied
        return min(
            candidates,
            key=lambda i: (
                coins_for(player.fields[i].bean_type or "", len(player.fields[i].beans)),
                len(player.fields[i].beans),
                i,
            ),
        )

    @staticmethod
    def _plantable_for_free(player: Player, beans: list[str]) -> bool:
        """Whether every bean fits now or after harvests that forfeit no coins."""

        # Track the field contents that planting would change, without mutating
        # the player.
        types: list[str | None] = [
            bean_field.bean_type for bean_field in player.fields
        ]
        counts = [len(bean_field.beans) for bean_field in player.fields]

        for bean in beans:
            matching = next((i for i, t in enumerate(types) if t == bean), None)
            if matching is not None:
                counts[matching] += 1
                continue
            empty = next((i for i, t in enumerate(types) if t is None), None)
            if empty is not None:
                types[empty] = bean
                counts[empty] = 1
                continue
            # No room: a field must be cleared, and only a field that has not
            # yet earned a coin can be cleared for free.
            free = next(
                (
                    i
                    for i, t in enumerate(types)
                    if t is not None and coins_for(t, counts[i]) == 0
                ),
                None,
            )
            if free is None:
                return False
            types[free] = bean
            counts[free] = 1
        return True


# Short aliases make the policies convenient to select in game setups.
Agent1 = RuleBasedAgent
Agent2 = StrategicPolicyAgent

