"""Blackjack hand evaluation using compact card codes."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .cards import Card, card_values


class Hand:
    """A mutable blackjack hand with allocation-free additions.

    A hand is backed by a fixed 12-byte array. Twelve cards is above the maximum
    possible non-busted blackjack hand, so normal play never needs to resize it.
    """

    __slots__ = ("_cards", "_size")
    MAX_CARDS = 12

    def __init__(self, cards: NDArray[np.integer] | tuple[int, ...] = ()) -> None:
        self._cards = np.empty(self.MAX_CARDS, dtype=np.uint8)
        self._size = 0
        self.extend(cards)

    def __len__(self) -> int:
        return self._size

    def add(self, card: Card | int | np.integer) -> None:
        """Add a ``Card`` or encoded card to the hand."""

        if self._size == self.MAX_CARDS:
            raise OverflowError(f"a hand cannot exceed {self.MAX_CARDS} cards")
        code = card.code if isinstance(card, Card) else card
        if (
            isinstance(code, (bool, np.bool_))
            or not isinstance(code, (int, np.integer))
            or not 0 <= int(code) < 52
        ):
            raise ValueError("card code must be an integer in the range 0..51")
        self._cards[self._size] = int(code)
        self._size += 1

    def extend(self, cards: NDArray[np.integer] | tuple[int, ...]) -> None:
        for card in cards:
            self.add(card)

    def clear(self) -> None:
        self._size = 0

    def codes(self, *, copy: bool = True) -> NDArray[np.uint8]:
        result = self._cards[: self._size]
        if copy:
            return result.copy()
        view = result.view()
        view.flags.writeable = False
        return view

    @property
    def total(self) -> int:
        """Best total at or under 21, or the smallest busted total."""

        if not self._size:
            return 0
        codes = self._cards[: self._size]
        values = card_values(codes)
        total = int(values.sum())
        aces = int(np.count_nonzero(np.remainder(codes, 13) == 0))
        if aces and total + 10 <= 21:
            total += 10
        return total

    @property
    def is_soft(self) -> bool:
        if not self._size:
            return False
        codes = self._cards[: self._size]
        hard_total = int(card_values(codes).sum())
        has_ace = bool(np.any(np.remainder(codes, 13) == 0))
        return has_ace and hard_total + 10 <= 21

    @property
    def is_blackjack(self) -> bool:
        return self._size == 2 and self.total == 21

    @property
    def is_bust(self) -> bool:
        return self.total > 21

    @property
    def can_split(self) -> bool:
        return self._size == 2 and self._cards[0] % 13 == self._cards[1] % 13

