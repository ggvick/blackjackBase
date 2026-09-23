"""Playing-card primitives and their compact NumPy representation.

The hot-path representation is a single unsigned byte in the range 0..51:
``suit * 13 + rank - 1``.  :class:`Card` is the readable boundary type; decks
and shoes deliberately store codes instead of 52 Python objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

CardArray = NDArray[np.uint8]


class Rank(IntEnum):
    """Card ranks, with the ace represented as 1."""

    ACE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13

    @property
    def blackjack_value(self) -> int:
        """Return the hard blackjack value (ace is 1, faces are 10)."""

        return min(int(self), 10)


class Suit(IntEnum):
    """The four suits in the encoding order."""

    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


_RANK_LABELS = ("", "A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
_SUIT_SYMBOLS = ("♣", "♦", "♥", "♠")
_DECK_TEMPLATE: CardArray = np.arange(52, dtype=np.uint8)
_DECK_TEMPLATE.flags.writeable = False


def _validate_code(code: int | np.integer) -> int:
    if isinstance(code, (bool, np.bool_)) or not isinstance(code, (int, np.integer)):
        raise TypeError("card code must be an integer")
    value = int(code)
    if not 0 <= value < 52:
        raise ValueError("card code must be in the range 0..51")
    return value


@dataclass(frozen=True, slots=True)
class Card:
    """An immutable, human-readable playing card.

    For simulation loops, prefer passing :attr:`code` integers and only create
    ``Card`` instances for display, logging, or external API responses.
    """

    rank: Rank
    suit: Suit

    def __post_init__(self) -> None:
        if isinstance(self.rank, (bool, np.bool_)) or not isinstance(
            self.rank, (int, np.integer)
        ):
            raise ValueError(f"invalid rank: {self.rank!r}")
        try:
            rank = Rank(self.rank)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid rank: {self.rank!r}") from exc
        if isinstance(self.suit, (bool, np.bool_)) or not isinstance(
            self.suit, (int, np.integer)
        ):
            raise ValueError(f"invalid suit: {self.suit!r}")
        try:
            suit = Suit(self.suit)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid suit: {self.suit!r}") from exc
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "suit", suit)

    @property
    def code(self) -> int:
        """Return the compact integer encoding for this card."""

        return int(self.suit) * 13 + int(self.rank) - 1

    @property
    def value(self) -> int:
        """Return the card's hard blackjack value."""

        return self.rank.blackjack_value

    @classmethod
    def from_code(cls, code: int | np.integer) -> Card:
        """Create a card from a compact code in the range 0..51."""

        value = _validate_code(code)
        return cls(Rank(value % 13 + 1), Suit(value // 13))

    def __str__(self) -> str:
        return f"{_RANK_LABELS[int(self.rank)]}{_SUIT_SYMBOLS[int(self.suit)]}"


class Deck:
    """A standard immutable 52-card deck.

    Construction is intentionally tiny: unshuffled decks share a read-only
    process-wide template.  ``codes()`` returns a defensive copy by default.
    """

    __slots__ = ("_codes",)

    def __init__(
        self,
        *,
        shuffled: bool = False,
        rng: np.random.Generator | int | None = None,
    ) -> None:
        if shuffled:
            codes = _DECK_TEMPLATE.copy()
            _coerce_rng(rng).shuffle(codes)
            codes.flags.writeable = False
            self._codes = codes
        else:
            self._codes = _DECK_TEMPLATE

    def __len__(self) -> int:
        return 52

    def __iter__(self) -> Iterator[Card]:
        return (Card.from_code(code) for code in self._codes)

    def __getitem__(self, index: int | slice) -> Card | tuple[Card, ...]:
        selected = self._codes[index]
        if np.isscalar(selected):
            return Card.from_code(selected)
        return tuple(Card.from_code(code) for code in selected)

    def codes(self, *, copy: bool = True) -> CardArray:
        """Return encoded cards, defensively copied unless ``copy=False``."""

        return self._codes.copy() if copy else self._codes

    def to_cards(self) -> tuple[Card, ...]:
        """Materialize all cards as readable objects."""

        return tuple(Card.from_code(code) for code in self._codes)


def _coerce_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None or (
        isinstance(rng, (int, np.integer)) and not isinstance(rng, (bool, np.bool_))
    ):
        # SFC64 is a statistically sound, non-cryptographic generator optimized
        # for simulation throughput. Callers can supply another Generator when
        # stream compatibility or a specific bit generator is required.
        return np.random.Generator(np.random.SFC64(rng))
    raise TypeError("rng must be a numpy.random.Generator, integer seed, or None")


def card_rank_indices(codes: NDArray[np.integer]) -> NDArray[np.uint8]:
    """Vectorized zero-based rank indices for encoded cards."""

    array = np.asarray(codes)
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise TypeError("card codes must have an integer dtype")
    if array.size and (np.any(array < 0) or np.any(array > 51)):
        raise ValueError("all card codes must be in the range 0..51")
    return np.remainder(array, 13).astype(np.uint8, copy=False)


def card_values(codes: NDArray[np.integer]) -> NDArray[np.uint8]:
    """Vectorized hard blackjack values for encoded cards."""

    ranks = card_rank_indices(codes) + np.uint8(1)
    return np.minimum(ranks, np.uint8(10))
