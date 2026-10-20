"""Bean card definitions and deck construction for the base game."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable


@dataclass(frozen=True, slots=True)
class BeanType:
    """A bean variety, its deck quantity, and its beanometer."""

    name: str
    card_count: int
    beanometer: tuple[int | None, int | None, int | None, int | None]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Bean type name cannot be empty")
        if self.card_count <= 0:
            raise ValueError("Bean card count must be positive")
        if len(self.beanometer) != 4:
            raise ValueError("A beanometer needs one threshold per coin value")
        thresholds = [value for value in self.beanometer if value is not None]
        if not thresholds:
            raise ValueError("A beanometer needs at least one reachable coin value")
        if any(value <= 0 for value in thresholds):
            raise ValueError("Beanometer thresholds must be positive")
        if any(a >= b for a, b in zip(thresholds, thresholds[1:])):
            raise ValueError("Beanometer thresholds must strictly increase")


# Classic base-game quantities (130 cards total). The beanometer lists how many
# beans a harvest needs for 1, 2, 3, and 4 coins; ``None`` marks an unreachable
# coin value.
BEAN_TYPES: tuple[BeanType, ...] = (
    BeanType("Garden", 6, (None, 2, 3, None)),
    BeanType("Cocoa", 4, (None, 2, 3, 4)),
    BeanType("Wax", 22, (4, 7, 9, 11)),
    BeanType("Blue", 20, (4, 6, 8, 10)),
    BeanType("Chili", 18, (3, 6, 8, 9)),
    BeanType("Stink", 16, (3, 5, 7, 8)),
    BeanType("Green", 14, (3, 5, 6, 7)),
    BeanType("Soy", 12, (2, 4, 6, 7)),
    BeanType("Black-eyed", 10, (2, 4, 5, 6)),
    BeanType("Red", 8, (2, 3, 4, 5)),
)

BEAN_TYPES_BY_NAME: dict[str, BeanType] = {bean.name: bean for bean in BEAN_TYPES}


def coins_for(bean: str, count: int) -> int:
    """Coins earned by harvesting ``count`` cards of one bean variety."""

    try:
        bean_type = BEAN_TYPES_BY_NAME[bean]
    except KeyError as exc:
        raise ValueError(f"Unknown bean type: {bean}") from exc
    if count < 0:
        raise ValueError("Cannot harvest a negative number of beans")

    coins = 0
    for value, threshold in enumerate(bean_type.beanometer, start=1):
        if threshold is not None and count >= threshold:
            coins = value
    return coins


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
