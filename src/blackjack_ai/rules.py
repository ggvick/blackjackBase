"""Configurable blackjack rules, actions, and legality policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .game import PlayerHand


class Action(IntEnum):
    """Stable action indices for agents and action masks."""

    STAND = 0
    HIT = 1
    DOUBLE = 2
    SPLIT = 3
    SURRENDER = 4
    INSURANCE = 5
    DECLINE_INSURANCE = 6


ACTION_COUNT = len(Action)


class SurrenderRule(IntEnum):
    NONE = 0
    LATE = 1
    EARLY = 2


class HoleCardRule(IntEnum):
    """When the dealer receives the second card."""

    AMERICAN = 0
    EUROPEAN = 1


class DealerBlackjackLossRule(IntEnum):
    """Exposure lost when a no-hole-card dealer later makes blackjack."""

    ALL_BETS = 0
    ORIGINAL_BET_ONLY = 1


@dataclass(frozen=True, slots=True)
class BlackjackRules:
    """All supported table and payout knobs.

    Defaults model a common six-deck game: dealer stands on soft 17, peeks for
    blackjack, double on any first two cards and after splits, split to four
    hands, one card on split aces, late surrender, and blackjack paying 3:2.
    """

    num_decks: int = 6
    penetration: float = 0.75
    dealer_hits_soft_17: bool = False
    hole_card_rule: HoleCardRule = HoleCardRule.AMERICAN
    dealer_peeks_for_blackjack: bool = True
    dealer_blackjack_loss_rule: DealerBlackjackLossRule = (
        DealerBlackjackLossRule.ALL_BETS
    )

    blackjack_payout: float = 1.5
    blackjack_after_split: bool = False
    table_minimum: float = 1.0
    table_maximum: float = 10_000.0

    double_after_split: bool = True
    double_on_any_number_of_cards: bool = False
    double_allowed_totals: tuple[int, ...] | None = None

    max_split_hands: int = 4
    split_by_value: bool = True
    resplit_aces: bool = False
    hit_split_aces: bool = False

    surrender: SurrenderRule = SurrenderRule.LATE
    surrender_after_split: bool = False

    insurance_allowed: bool = True
    insurance_fraction: float = 0.5
    insurance_payout: float = 2.0

    charlie_card_count: int | None = None
    charlie_payout: float = 1.0
    minimum_cards_before_round: int = 20

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_decks, bool)
            or not isinstance(self.num_decks, int)
            or self.num_decks < 1
        ):
            raise ValueError("num_decks must be a positive integer")
        if not math.isfinite(self.penetration) or not 0 < self.penetration <= 1:
            raise ValueError("penetration must be in (0, 1]")
        if not isinstance(self.hole_card_rule, HoleCardRule):
            raise TypeError("hole_card_rule must be a HoleCardRule")
        if not isinstance(self.dealer_blackjack_loss_rule, DealerBlackjackLossRule):
            raise TypeError(
                "dealer_blackjack_loss_rule must be a DealerBlackjackLossRule"
            )
        if not isinstance(self.surrender, SurrenderRule):
            raise TypeError("surrender must be a SurrenderRule")

        for name in (
            "blackjack_payout",
            "table_minimum",
            "table_maximum",
            "insurance_fraction",
            "insurance_payout",
            "charlie_payout",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.blackjack_payout <= 0:
            raise ValueError("blackjack_payout must be positive")
        if self.table_minimum <= 0 or self.table_maximum < self.table_minimum:
            raise ValueError("table limits must satisfy 0 < minimum <= maximum")
        if not 0 < self.insurance_fraction <= 1:
            raise ValueError("insurance_fraction must be in (0, 1]")

        if (
            isinstance(self.max_split_hands, bool)
            or not isinstance(self.max_split_hands, int)
            or self.max_split_hands < 1
        ):
            raise ValueError("max_split_hands must be a positive integer")
        if self.double_allowed_totals is not None:
            totals = tuple(self.double_allowed_totals)
            if not totals or any(
                isinstance(total, bool)
                or not isinstance(total, int)
                or not 2 <= total <= 21
                for total in totals
            ):
                raise ValueError(
                    "double_allowed_totals must contain integer totals from 2 to 21"
                )
            object.__setattr__(self, "double_allowed_totals", tuple(sorted(set(totals))))
        if self.charlie_card_count is not None and (
            isinstance(self.charlie_card_count, bool)
            or not isinstance(self.charlie_card_count, int)
            or not 3 <= self.charlie_card_count <= 12
        ):
            raise ValueError("charlie_card_count must be None or an integer from 3 to 12")
        if (
            isinstance(self.minimum_cards_before_round, bool)
            or not isinstance(self.minimum_cards_before_round, int)
            or self.minimum_cards_before_round < 0
        ):
            raise ValueError("minimum_cards_before_round must be a non-negative integer")


class RuleEngine:
    """Fast policy for dealer play and player action legality."""

    __slots__ = ("rules",)

    def __init__(self, rules: BlackjackRules | None = None) -> None:
        self.rules = rules or BlackjackRules()

    def dealer_should_hit(self, total: int, is_soft: bool) -> bool:
        return total < 17 or (
            total == 17 and is_soft and self.rules.dealer_hits_soft_17
        )

    def cards_can_split(self, first_code: int, second_code: int) -> bool:
        first_rank = first_code % 13
        second_rank = second_code % 13
        if self.rules.split_by_value:
            first_value = min(first_rank + 1, 10)
            second_value = min(second_rank + 1, 10)
            return first_value == second_value
        return first_rank == second_rank

    def can_split(
        self,
        hand: PlayerHand,
        *,
        hand_count: int,
        available_balance: float,
    ) -> bool:
        if len(hand.cards) != 2 or hand_count >= self.rules.max_split_hands:
            return False
        codes = hand.cards.codes(copy=False)
        if not self.cards_can_split(int(codes[0]), int(codes[1])):
            return False
        if hand.split_aces and not self.rules.resplit_aces:
            return False
        if int(codes[0]) % 13 == 0 and hand.from_split and not self.rules.resplit_aces:
            return False
        return available_balance >= hand.wager

    def can_double(
        self, hand: PlayerHand, *, available_balance: float
    ) -> bool:
        if hand.is_complete or available_balance < hand.wager:
            return False
        if hand.from_split and not self.rules.double_after_split:
            return False
        if hand.split_aces and not self.rules.hit_split_aces:
            return False
        if not self.rules.double_on_any_number_of_cards and len(hand.cards) != 2:
            return False
        if self.rules.double_allowed_totals is not None:
            return hand.cards.total in self.rules.double_allowed_totals
        return True

    def can_surrender(self, hand: PlayerHand, *, window_open: bool) -> bool:
        return (
            window_open
            and self.rules.surrender is not SurrenderRule.NONE
            and len(hand.cards) == 2
            and hand.actions_taken == 0
            and (not hand.from_split or self.rules.surrender_after_split)
        )

    def legal_action_bits(
        self,
        hand: PlayerHand,
        *,
        hand_count: int,
        available_balance: float,
        surrender_window_open: bool,
    ) -> int:
        """Return a compact bitset of legal playing actions."""

        if hand.is_complete:
            return 0
        bits = 1 << Action.STAND
        if hand.cards.total < 21 and (
            not hand.split_aces or self.rules.hit_split_aces
        ):
            bits |= 1 << Action.HIT
        if self.can_double(hand, available_balance=available_balance):
            bits |= 1 << Action.DOUBLE
        if self.can_split(
            hand,
            hand_count=hand_count,
            available_balance=available_balance,
        ):
            bits |= 1 << Action.SPLIT
        if self.can_surrender(hand, window_open=surrender_window_open):
            bits |= 1 << Action.SURRENDER
        return bits

    @staticmethod
    def bits_to_mask(
        bits: int, out: NDArray[np.bool_] | None = None
    ) -> NDArray[np.bool_]:
        result = np.empty(ACTION_COUNT, dtype=np.bool_) if out is None else out
        if result.shape != (ACTION_COUNT,) or result.dtype != np.bool_:
            raise ValueError(f"out must be a bool array with shape ({ACTION_COUNT},)")
        for index in range(ACTION_COUNT):
            result[index] = bool(bits & (1 << index))
        return result

    @staticmethod
    def actions_from_bits(bits: int) -> tuple[Action, ...]:
        return tuple(
            Action(index)
            for index in range(ACTION_COUNT)
            if bits & (1 << index)
        )
