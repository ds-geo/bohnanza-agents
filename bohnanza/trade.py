"""Trade objects and validation for Bohnanza's trading phase."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from .events import log_event


class TradeStatus(Enum):
    """The status of a trade proposal."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COUNTERED = "countered"


@dataclass
class TradeOffer:
    """A card exchange offer between two players.

    The offer specifies which cards each player gives to the other. Cards can come
    from the player's hand or, for the active player, from the face-up cards.
    """

    proposer: int
    """Index of the player making the offer."""

    responder: int
    """Index of the player receiving the offer."""

    proposer_gives: list[int] = field(default_factory=list)
    """Indices of cards the proposer gives (from hand or face-up)."""

    responder_gives: list[int] = field(default_factory=list)
    """Indices of cards the responder gives (from hand only)."""

    proposer_face_up: bool = False
    """Whether proposer_gives refers to face-up cards (active player only)."""

    status: TradeStatus = TradeStatus.PENDING
    """Current status of this trade offer."""

    counter_to: TradeOffer | None = None
    """If this is a counter-offer, the original offer it counters."""

    def __post_init__(self) -> None:
        if self.proposer == self.responder:
            raise ValueError("A trade must be between different players")
        if self.proposer < 0 or self.responder < 0:
            raise ValueError("Player indices must be non-negative")
        if any(i < 0 for i in self.proposer_gives):
            raise ValueError("Card indices must be non-negative")
        if any(i < 0 for i in self.responder_gives):
            raise ValueError("Card indices must be non-negative")
        if len(set(self.proposer_gives)) != len(self.proposer_gives):
            raise ValueError("Proposer card indices must be unique")
        if len(set(self.responder_gives)) != len(self.responder_gives):
            raise ValueError("Responder card indices must be unique")
        log_event(
            "trade.offered",
            proposer=self.proposer,
            responder=self.responder,
            proposer_gives=self.proposer_gives,
            responder_gives=self.responder_gives,
            proposer_face_up=self.proposer_face_up,
            counter=self.counter_to is not None,
        )

    def is_empty(self) -> bool:
        """Whether this offer involves no cards (a donation)."""

        return not self.proposer_gives and not self.responder_gives

    def is_counter(self) -> bool:
        """Whether this offer is a counter-offer."""

        return self.counter_to is not None

    def accept(self) -> None:
        """Mark this offer as accepted."""

        if self.status is not TradeStatus.PENDING:
            raise ValueError(f"Cannot accept a trade with status {self.status}")
        self.status = TradeStatus.ACCEPTED
        self._log_status("trade.accepted")

    def reject(self) -> None:
        """Mark this offer as rejected."""

        if self.status is not TradeStatus.PENDING:
            raise ValueError(f"Cannot reject a trade with status {self.status}")
        self.status = TradeStatus.REJECTED
        self._log_status("trade.rejected")

    def counter(self, new_offer: TradeOffer) -> None:
        """Replace this offer with a counter-offer."""

        if self.status is not TradeStatus.PENDING:
            raise ValueError(f"Cannot counter a trade with status {self.status}")
        self.status = TradeStatus.COUNTERED
        new_offer.counter_to = self
        self._log_status("trade.countered")

    def _log_status(self, event: str) -> None:
        log_event(
            event,
            proposer=self.proposer,
            responder=self.responder,
            proposer_gives=self.proposer_gives,
            responder_gives=self.responder_gives,
            proposer_face_up=self.proposer_face_up,
            status=self.status.value,
        )


class TradeValidator:
    """Validates trade offers against game state."""

    def __init__(self, players: list, active_index: int, face_up_count: int) -> None:
        """Initialize with current game state.

        Args:
            players: List of Player objects.
            active_index: Index of the active player.
            face_up_count: Number of face-up cards currently available.
        """
        self.players = players
        self.active_index = active_index
        self.face_up_count = face_up_count

    def validate(self, offer: TradeOffer) -> tuple[bool, str]:
        """Check if a trade offer is valid given current game state.

        Returns:
            A tuple of (is_valid, error_message). If valid, error_message is empty.
        """

        # Check player indices
        if not 0 <= offer.proposer < len(self.players):
            return False, f"No player at index {offer.proposer}"
        if not 0 <= offer.responder < len(self.players):
            return False, f"No player at index {offer.responder}"

        # Check that only active player can give face-up cards
        if offer.proposer_face_up and offer.proposer != self.active_index:
            return False, "Only the active player can offer face-up cards"

        # Validate proposer's card indices
        if offer.proposer_face_up:
            max_index = self.face_up_count - 1
            for index in offer.proposer_gives:
                if index > max_index:
                    return False, f"No face-up card at index {index}"
        else:
            proposer_hand_size = len(self.players[offer.proposer].hand)
            for index in offer.proposer_gives:
                if index >= proposer_hand_size:
                    return False, f"No card in proposer's hand at index {index}"

        # Validate responder's card indices
        responder_hand_size = len(self.players[offer.responder].hand)
        for index in offer.responder_gives:
            if index >= responder_hand_size:
                return False, f"No card in responder's hand at index {index}"

        # Counter-offer must reverse the roles
        if offer.is_counter():
            original = offer.counter_to
            if original is None:
                return False, "Counter-offer must reference an original offer"
            if offer.proposer != original.responder:
                return False, "Counter-offer must come from the original responder"
            if offer.responder != original.proposer:
                return False, "Counter-offer must be directed to the original proposer"

        return True, ""

    def can_execute(self, offer: TradeOffer) -> tuple[bool, str]:
        """Check if an accepted offer can be executed.

        This is stricter than validate() because it also checks that the offer
        has been accepted and that the cards are still in their original positions.
        """

        if offer.status is not TradeStatus.ACCEPTED:
            return False, f"Cannot execute trade with status {offer.status}"

        # Re-validate indices (cards may have moved)
        is_valid, error = self.validate(offer)
        if not is_valid:
            return False, error

        return True, ""
