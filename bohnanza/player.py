"""Player state: ordered hand, planted bean fields, and coins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .cards import BEAN_TYPES, coins_for


_BEAN_NAMES = frozenset(bean.name for bean in BEAN_TYPES)


def _validate_bean(bean: str) -> None:
    if bean not in _BEAN_NAMES:
        raise ValueError(f"Unknown bean type: {bean}")


@dataclass
class BeanField:
    """One plot, which can contain cards of only one bean variety."""

    beans: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for bean in self.beans:
            _validate_bean(bean)
        if len(set(self.beans)) > 1:
            raise ValueError("A bean field can contain only one bean type")

    @property
    def bean_type(self) -> str | None:
        """The planted variety, or ``None`` while the field is empty."""

        return self.beans[0] if self.beans else None

    def accepts(self, bean: str) -> bool:
        """Whether this field is empty or already holds ``bean``."""

        _validate_bean(bean)
        return not self.beans or self.bean_type == bean

    def plant(self, bean: str) -> None:
        """Plant a bean if this field is empty or already has that variety."""

        if not self.accepts(bean):
            raise ValueError(
                f"Cannot plant {bean} in a field containing {self.bean_type}"
            )
        self.beans.append(bean)

    def harvest(self) -> tuple[int, list[str]]:
        """Empty this field, returning the coins earned and the cards removed."""

        if not self.beans:
            raise ValueError("Cannot harvest an empty bean field")
        cards = self.beans
        self.beans = []
        return coins_for(cards[0], len(cards)), cards


@dataclass
class Player:
    """Mutable game state for one player.

    ``hand`` is kept in deal order. Code that adds cards should append them;
    planting from the hand removes the first card and never rearranges others.
    Players start with two fields, with additional fields supplied by game rules.
    """

    name: str
    hand: list[str] = field(default_factory=list)
    fields: list[BeanField] = field(default_factory=lambda: [BeanField(), BeanField()])
    coins: int = 0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Player name cannot be empty")
        if self.coins < 0:
            raise ValueError("Player coins cannot be negative")
        for bean in self.hand:
            _validate_bean(bean)
        if not self.fields:
            raise ValueError("A player must have at least one bean field")

    def add_to_hand(self, beans: Iterable[str]) -> None:
        """Append cards to the end of the hand, preserving their order."""

        new_beans = list(beans)
        for bean in new_beans:
            _validate_bean(bean)
        self.hand.extend(new_beans)

    def remove_from_hand(self, index: int) -> str:
        """Remove one card from the hand, preserving the order of the rest."""

        try:
            return self.hand.pop(index)
        except IndexError as exc:
            raise IndexError(f"No card in hand at index {index}") from exc

    def legal_fields(self, bean: str) -> list[int]:
        """Indices of the fields that could hold ``bean`` right now."""

        return [
            index
            for index, bean_field in enumerate(self.fields)
            if bean_field.accepts(bean)
        ]

    def needs_harvest_to_plant(self, bean: str) -> bool:
        """Whether a field must be harvested before ``bean`` can be planted."""

        return not self.legal_fields(bean)

    def plant(self, bean: str, field_index: int) -> None:
        """Plant a card that did not come from the hand (traded or face-up)."""

        self._field_at(field_index).plant(bean)

    def plant_from_front(self, field_index: int = 0) -> str:
        """Plant and remove the first card in hand, preserving hand order."""

        if not self.hand:
            raise ValueError("Cannot plant from an empty hand")
        bean_field = self._field_at(field_index)

        # Validate before removing the card, so a failed planting leaves state intact.
        bean = self.hand[0]
        bean_field.plant(bean)
        del self.hand[0]
        return bean

    def harvest(self, field_index: int) -> tuple[int, list[str]]:
        """Harvest a field, adding its coins and returning the cards removed."""

        earned, cards = self._field_at(field_index).harvest()
        self.coins += earned
        return earned, cards

    def _field_at(self, field_index: int) -> BeanField:
        if field_index < 0:
            raise IndexError(f"No bean field at index {field_index}")
        try:
            return self.fields[field_index]
        except IndexError as exc:
            raise IndexError(f"No bean field at index {field_index}") from exc
