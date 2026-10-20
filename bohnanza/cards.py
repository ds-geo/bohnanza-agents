"""Bean card definitions and deck construction for the base game."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable


@dataclass(frozen=True, slots=True)
class BeanType:
    """A bean variety and its number of cards in the base-game deck."""

    name: str
    card_count: int

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Bean type name cannot be empty")
        if self.card_count <= 0:
            raise ValueError("Bean card count must be positive")


# Classic base-game quantities (130 cards total).
BEAN_TYPES: tuple[BeanType, ...] = (
    BeanType("Garden", 6),
    BeanType("Cocoa", 4),
    BeanType("Wax", 22),
    BeanType("Blue", 20),
    BeanType("Chili", 18),
    BeanType("Stink", 16),
    BeanType("Green", 14),
    BeanType("Soy", 12),
    BeanType("Black-eyed", 10),
    BeanType("Red", 8),
)


class Deck:
    """A draw pile with a discard pile that can be shuffled back in."""

    def __init__(
        self,
        bean_types: Iterable[BeanType] = BEAN_TYPES,
        *,
        seed: int | None = None,
    ) -> None:
        types = tuple(bean_types)
        if not types:
            raise ValueError("A deck requires at least one bean type")
        names = [bean.name for bean in types]
        if len(set(names)) != len(names):
            raise ValueError("Bean type names must be unique")

        self.bean_types = types
        self._random = random.Random(seed)
        self._draw_pile = [bean.name for bean in types for _ in range(bean.card_count)]
        self._discard_pile: list[str] = []
        self._random.shuffle(self._draw_pile)

    @property
    def draw_count(self) -> int:
        """Number of cards currently available to draw."""

        return len(self._draw_pile)

    @property
    def discard_count(self) -> int:
        """Number of cards in the discard pile."""

        return len(self._discard_pile)

    @property
    def total_count(self) -> int:
        """Number of cards in both piles."""

        return self.draw_count + self.discard_count

    def draw(self) -> str | None:
        """Draw the next bean card, returning ``None`` if the deck is empty."""

        if not self._draw_pile and not self._reshuffle_discards():
            return None
        return self._draw_pile.pop()

    def discard(self, bean: str) -> None:
        """Place a bean card in the discard pile."""

        if bean not in {item.name for item in self.bean_types}:
            raise ValueError(f"Unknown bean type: {bean}")
        self._discard_pile.append(bean)

    def _reshuffle_discards(self) -> bool:
        if not self._discard_pile:
            return False
        self._draw_pile = self._discard_pile
        self._discard_pile = []
        self._random.shuffle(self._draw_pile)
        return True
