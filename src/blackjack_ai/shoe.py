"""Fast, NumPy-backed blackjack shoe."""

from __future__ import annotations

from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from .cards import Card, CardArray, _DECK_TEMPLATE, _coerce_rng


class ShoeEmptyError(IndexError):
    """Raised when a draw asks for more cards than remain in the shoe."""


class Shoe:
    """One or more decks with O(1) scalar draws and exact composition tracking.

    Parameters
    ----------
    num_decks:
        Number of standard 52-card decks. Casino shoes commonly use 6 or 8.
    shuffled:
        Shuffle immediately. Disable for deterministic tests or ordered deals.
    rng:
        A NumPy ``Generator`` or integer seed. Passing a seed makes shuffles
        reproducible.
    penetration:
        Fraction of cards dealt before :attr:`cut_card_reached` becomes true.
    """

    __slots__ = (
        "_cards",
        "_cursor",
        "_rank_counts",
        "_rank_counts_view",
        "_rng",
        "_revision",
        "_cut_position",
        "num_decks",
        "penetration",
    )

    def __init__(
        self,
        num_decks: int = 6,
        *,
        shuffled: bool = True,
        rng: np.random.Generator | int | None = None,
        penetration: float = 0.75,
    ) -> None:
        if (
            isinstance(num_decks, (bool, np.bool_))
            or not isinstance(num_decks, (int, np.integer))
            or int(num_decks) < 1
        ):
            raise ValueError("num_decks must be a positive integer")
        if not np.isfinite(penetration) or not 0.0 < float(penetration) <= 1.0:
            raise ValueError("penetration must be in the interval (0, 1]")

        self.num_decks = int(num_decks)
        self.penetration = float(penetration)
        if not (
            rng is None
            or isinstance(rng, np.random.Generator)
            or (
                isinstance(rng, (int, np.integer))
                and not isinstance(rng, (bool, np.bool_))
            )
        ):
            raise TypeError("rng must be a numpy.random.Generator, integer seed, or None")
        # Defer Generator construction for ordered shoes; it dominates their setup.
        self._rng = rng
        # tile is substantially faster and smaller than constructing Card objects.
        self._cards = np.tile(_DECK_TEMPLATE, self.num_decks)
        self._cursor = 0
        self._rank_counts = np.full(13, 4 * self.num_decks, dtype=np.int32)
        self._rank_counts_view = self._rank_counts.view()
        self._rank_counts_view.flags.writeable = False
        self._revision = 0
        self._cut_position = int(self.total_cards * self.penetration)
        if shuffled:
            self._generator().shuffle(self._cards)

    def _generator(self) -> np.random.Generator:
        if not isinstance(self._rng, np.random.Generator):
            self._rng = _coerce_rng(self._rng)
        return self._rng

    @property
    def total_cards(self) -> int:
        """Initial number of cards in the shoe."""

        return self.num_decks * 52

    @property
    def cards_dealt(self) -> int:
        return self._cursor

    @property
    def cards_remaining(self) -> int:
        return self.total_cards - self._cursor

    @property
    def decks_remaining(self) -> float:
        """Exact decks remaining, without casino-style rounding."""

        return self.cards_remaining / 52.0

    @property
    def fraction_dealt(self) -> float:
        return self._cursor / self.total_cards

    @property
    def cut_card_reached(self) -> bool:
        return self._cursor >= self._cut_position

    @property
    def is_empty(self) -> bool:
        return self._cursor == self.total_cards

    @property
    def revision(self) -> int:
        """Monotonic composition revision used by synchronized consumers."""

        return self._revision

    def __len__(self) -> int:
        return self.cards_remaining

    def __iter__(self) -> Iterator[Card]:
        return (Card.from_code(code) for code in self._cards[self._cursor :])

    def draw_code(self) -> int:
        """Draw one card and return its compact code (the fastest draw API)."""

        if self._cursor >= self.total_cards:
            raise ShoeEmptyError("cannot draw from an empty shoe")
        code = int(self._cards[self._cursor])
        self._cursor += 1
        self._rank_counts[code % 13] -= 1
        self._revision += 1
        return code

    def draw_codes(self, count: int, *, copy: bool = True) -> CardArray:
        """Draw a batch of encoded cards.

        ``copy=False`` returns a read-only view and is intended for short-lived
        internal/AI use. The view's contents may change after a later shuffle.
        """

        if (
            isinstance(count, (bool, np.bool_))
            or not isinstance(count, (int, np.integer))
            or int(count) < 0
        ):
            raise ValueError("count must be a non-negative integer")
        count = int(count)
        end = self._cursor + count
        if end > self.total_cards:
            raise ShoeEmptyError(
                f"cannot draw {count} cards; only {self.cards_remaining} remain"
            )
        result = self._cards[self._cursor : end]
        if count == 1:
            self._rank_counts[int(result[0]) % 13] -= 1
        elif count:
            ranks = np.remainder(result, 13)
            self._rank_counts -= np.bincount(ranks, minlength=13).astype(
                np.int32, copy=False
            )
        self._cursor = end
        if count:
            self._revision += 1
        if copy:
            return result.copy()
        view = result.view()
        view.flags.writeable = False
        return view

    def draw(self, count: int = 1) -> Card | tuple[Card, ...]:
        """Draw readable card objects; use ``draw_code(s)`` in hot loops."""

        codes = self.draw_codes(count, copy=False)
        cards = tuple(Card.from_code(code) for code in codes)
        return cards[0] if count == 1 else cards

    def peek_codes(self, count: int = 1, *, copy: bool = True) -> CardArray:
        """Inspect upcoming encoded cards without consuming them."""

        if (
            isinstance(count, (bool, np.bool_))
            or not isinstance(count, (int, np.integer))
            or int(count) < 0
        ):
            raise ValueError("count must be a non-negative integer")
        end = self._cursor + int(count)
        if end > self.total_cards:
            raise ShoeEmptyError(
                f"cannot peek at {count} cards; only {self.cards_remaining} remain"
            )
        result = self._cards[self._cursor : end]
        if copy:
            return result.copy()
        view = result.view()
        view.flags.writeable = False
        return view

    def rank_counts(self, *, copy: bool = True) -> NDArray[np.int32]:
        """Counts remaining in A, 2, ..., 10, J, Q, K order."""

        if copy:
            return self._rank_counts.copy()
        return self._rank_counts_view

    def rank_probabilities(self, *, dtype: np.dtype = np.dtype(np.float32)) -> NDArray:
        """Exact probability of the next rank in A..K order."""

        if self.cards_remaining == 0:
            return np.zeros(13, dtype=dtype)
        return self._rank_counts.astype(dtype) / self.cards_remaining

    def shuffle(self) -> None:
        """Shuffle only the undealt portion of the current shoe in-place."""

        self._generator().shuffle(self._cards[self._cursor :])

    def reset(self, *, shuffled: bool = True) -> None:
        """Restore a full shoe and optionally reshuffle it."""

        self._cursor = 0
        self._rank_counts.fill(4 * self.num_decks)
        self._revision += 1
        if shuffled:
            self._generator().shuffle(self._cards)
        else:
            self._cards[:] = np.tile(_DECK_TEMPLATE, self.num_decks)
