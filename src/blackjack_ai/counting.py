"""Exact card-counting and AI-facing shoe features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .cards import Card
from .shoe import Shoe, ShoeEmptyError


class CountingSystem:
    """Immutable rank tags for a card-counting system.

    Tags are supplied in ``A, 2, ..., 10, J, Q, K`` order. Fractional tags are
    supported, so systems such as Wong Halves can be represented exactly.
    """

    __slots__ = ("name", "_tags")

    def __init__(self, name: str, tags: NDArray | tuple[float, ...]) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        array = np.asarray(tags, dtype=np.float64)
        if array.shape != (13,) or not np.all(np.isfinite(array)):
            raise ValueError("tags must contain 13 finite values in A..K order")
        array = array.copy()
        array.flags.writeable = False
        self.name = name
        self._tags = array

    @property
    def tags(self) -> NDArray[np.float64]:
        return self._tags

    @property
    def balanced(self) -> bool:
        return bool(np.isclose(self._tags.sum(), 0.0))

    def __repr__(self) -> str:
        return f"CountingSystem(name={self.name!r}, tags={self._tags.tolist()!r})"


HI_LO = CountingSystem(
    "Hi-Lo",
    (-1, 1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1),
)
OMEGA_II = CountingSystem(
    "Omega II",
    (0, 1, 1, 2, 2, 2, 1, 0, -1, -2, -2, -2, -2),
)
WONG_HALVES = CountingSystem(
    "Wong Halves",
    (-1, 0.5, 1, 1, 1.5, 1, 0.5, 0, -0.5, -1, -1, -1, -1),
)


@dataclass(frozen=True, slots=True)
class CountSnapshot:
    """A point-in-time card-count summary."""

    running_count: float
    true_count: float
    cards_seen: int
    cards_remaining: int
    decks_remaining: float
    fraction_dealt: float
    cut_card_reached: bool


class BlackjackController:
    """Own a shoe and expose drift-free counts and AI-friendly observations.

    Counts are calculated from the exact rank composition still in the shoe,
    not from a separately updated accumulator. This remains correct even when
    another caller draws directly from :attr:`shoe`.
    """

    __slots__ = (
        "shoe",
        "counting_system",
        "_initial_rank_counts",
        "_initial_running_count",
        "_running_count_cache",
        "_shoe_revision",
    )

    def __init__(
        self,
        shoe: Shoe | None = None,
        *,
        num_decks: int = 6,
        counting_system: CountingSystem = HI_LO,
        shuffled: bool = True,
        rng: np.random.Generator | int | None = None,
        penetration: float = 0.75,
    ) -> None:
        if shoe is not None and num_decks != 6:
            raise ValueError("pass either shoe or num_decks, not both")
        if not isinstance(counting_system, CountingSystem):
            raise TypeError("counting_system must be a CountingSystem")
        self.shoe = (
            shoe
            if shoe is not None
            else Shoe(
                num_decks,
                shuffled=shuffled,
                rng=rng,
                penetration=penetration,
            )
        )
        self.counting_system = counting_system
        self._initial_rank_counts = np.full(
            13, 4 * self.shoe.num_decks, dtype=np.int32
        )
        self._initial_running_count = float(
            self._initial_rank_counts @ self.counting_system.tags
        )
        self._running_count_cache = self._initial_running_count - float(
            self.shoe.rank_counts(copy=False) @ self.counting_system.tags
        )
        self._shoe_revision = self.shoe.revision

    def _synchronize_running_count(self) -> float:
        """Resync after callers mutate the public shoe directly."""

        if self._shoe_revision != self.shoe.revision:
            self._running_count_cache = self._initial_running_count - float(
                self.shoe.rank_counts(copy=False) @ self.counting_system.tags
            )
            self._shoe_revision = self.shoe.revision
        return self._running_count_cache

    @property
    def seen_rank_counts(self) -> NDArray[np.int32]:
        """Number of observed/dealt cards by rank in A..K order."""

        return self._initial_rank_counts - self.shoe.rank_counts(copy=False)

    @property
    def running_count(self) -> float:
        return self._synchronize_running_count()

    @property
    def true_count(self) -> float:
        """Running count divided by exact decks remaining.

        An empty shoe has no defined true count and raises ``ShoeEmptyError``.
        """

        if self.shoe.is_empty:
            raise ShoeEmptyError("true count is undefined for an empty shoe")
        return self.running_count / self.shoe.decks_remaining

    @property
    def high_cards_remaining(self) -> int:
        """Remaining tens and aces (useful for blackjack/insurance models)."""

        counts = self.shoe.rank_counts(copy=False)
        return int(counts[0] + counts[9:].sum())

    @property
    def ten_value_probability(self) -> float:
        """Exact probability that the next card is ten-valued."""

        if self.shoe.is_empty:
            return 0.0
        return float(self.shoe.rank_counts(copy=False)[9:].sum() / len(self.shoe))

    @property
    def ace_probability(self) -> float:
        """Exact probability that the next card is an ace."""

        if self.shoe.is_empty:
            return 0.0
        return float(self.shoe.rank_counts(copy=False)[0] / len(self.shoe))

    def draw_code(self) -> int:
        self._synchronize_running_count()
        code = self.shoe.draw_code()
        self._running_count_cache += float(self.counting_system.tags[code % 13])
        self._shoe_revision = self.shoe.revision
        return code

    def draw_codes(self, count: int, *, copy: bool = True) -> NDArray[np.uint8]:
        self._synchronize_running_count()
        codes = self.shoe.draw_codes(count, copy=copy)
        if codes.size == 1:
            self._running_count_cache += float(
                self.counting_system.tags[int(codes[0]) % 13]
            )
        elif codes.size:
            ranks = np.remainder(codes, 13)
            self._running_count_cache += float(self.counting_system.tags[ranks].sum())
        self._shoe_revision = self.shoe.revision
        return codes

    def draw(self, count: int = 1) -> Card | tuple[Card, ...]:
        codes = self.draw_codes(count, copy=False)
        cards = tuple(Card.from_code(code) for code in codes)
        return cards[0] if count == 1 else cards

    def snapshot(self) -> CountSnapshot:
        """Return an immutable count/state record."""

        if self.shoe.is_empty:
            raise ShoeEmptyError("true count is undefined for an empty shoe")
        running_count = self.running_count
        return CountSnapshot(
            running_count=running_count,
            true_count=running_count / self.shoe.decks_remaining,
            cards_seen=self.shoe.cards_dealt,
            cards_remaining=self.shoe.cards_remaining,
            decks_remaining=self.shoe.decks_remaining,
            fraction_dealt=self.shoe.fraction_dealt,
            cut_card_reached=self.shoe.cut_card_reached,
        )

    def observation(
        self,
        *,
        dtype: np.dtype = np.dtype(np.float32),
        out: NDArray | None = None,
    ) -> NDArray:
        """Return a compact 18-value AI observation vector.

        Layout: 13 next-rank probabilities, running count, true count, fraction
        dealt, ten-value probability, and ace probability. All composition
        features are exact rather than inferred from a human count.

        Pass a writable ``out`` array with shape ``(18,)`` and the requested
        dtype to reuse memory in a training loop. The returned object is then
        the same array. Reusing a buffer avoids one allocation per observation.
        """

        requested_dtype = np.dtype(dtype)
        if out is None:
            result = np.empty(18, dtype=requested_dtype)
        else:
            if not isinstance(out, np.ndarray):
                raise TypeError("out must be a numpy.ndarray")
            if out.shape != (18,):
                raise ValueError("out must have shape (18,)")
            if out.dtype != requested_dtype:
                raise TypeError(
                    f"out has dtype {out.dtype}; expected {requested_dtype}"
                )
            if not out.flags.writeable:
                raise ValueError("out must be writable")
            result = out

        # Read composition once and derive every feature in one pass. The old
        # implementation called four helpers, each traversing or summing it.
        counts = self.shoe.rank_counts(copy=False)
        remaining = self.shoe.cards_remaining
        running_count = self._synchronize_running_count()
        if remaining:
            inverse_remaining = 1.0 / remaining
            np.multiply(
                counts,
                inverse_remaining,
                out=result[:13],
                casting="unsafe",
            )
            ten_values = int(
                counts[9] + counts[10] + counts[11] + counts[12]
            )
            result[14] = running_count * 52.0 * inverse_remaining
            result[16] = ten_values * inverse_remaining
            result[17] = int(counts[0]) * inverse_remaining
        else:
            result.fill(0)

        result[13] = running_count
        result[15] = self.shoe.cards_dealt / self.shoe.total_cards
        return result

    def reset(self, *, shuffled: bool = True) -> None:
        self.shoe.reset(shuffled=shuffled)
        self._running_count_cache = 0.0
        self._shoe_revision = self.shoe.revision
