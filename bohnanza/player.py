"""Player state: ordered hand, planted bean fields, and coins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .cards import BEAN_TYPES


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

    def plant(self, bean: str) -> None:
        """Plant a bean if this field is empty or already has that variety."""

        _validate_bean(bean)
        if self.beans and self.bean_type != bean:
            raise ValueError(
                f"Cannot plant {bean} in a field containing {self.bean_type}"
            )
        self.beans.append(bean)


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

    def plant_from_front(self, field_index: int = 0) -> str:
        """Plant and remove the first card in hand, preserving hand order."""

        if not self.hand:
            raise ValueError("Cannot plant from an empty hand")
        try:
            bean_field = self.fields[field_index]
        except IndexError as exc:
            raise IndexError(f"No bean field at index {field_index}") from exc

        bean = self.hand[0]
        # Validate before removing the card, so a failed planting leaves state intact.
        if bean_field.beans and bean_field.bean_type != bean:
            raise ValueError(
                f"Cannot plant {bean} in a field containing {bean_field.bean_type}"
            )
        bean_field.plant(bean)
        del self.hand[0]
        return bean
