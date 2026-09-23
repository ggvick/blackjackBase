"""Complete, configurable blackjack round engine for AI simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import math

import numpy as np
from numpy.typing import NDArray

from .cards import Card, _validate_code, card_rank_indices
from .cash import Cash
from .counting import BlackjackController, CountingSystem, HI_LO
from .hand import Hand
from .rules import (
    ACTION_COUNT,
    Action,
    BlackjackRules,
    DealerBlackjackLossRule,
    HoleCardRule,
    RuleEngine,
    SurrenderRule,
)
from .shoe import Shoe

GAME_OBSERVATION_SIZE = 43
VISIBLE_COMPOSITION_SIZE = 14


class RoundPhase(IntEnum):
    READY = 0
    INSURANCE = 1
    PLAYER = 2
    DEALER = 3
    SETTLED = 4


class HandOutcome(IntEnum):
    WIN = 0
    LOSS = 1
    PUSH = 2
    BLACKJACK = 3
    BUST = 4
    SURRENDER = 5
    CHARLIE = 6


@dataclass(slots=True)
class PlayerHand:
    """Mutable per-hand state used during a round."""

    cards: Hand
    wager: float
    from_split: bool = False
    split_aces: bool = False
    split_depth: int = 0
    blackjack_eligible: bool = True
    needs_split_card: bool = False
    actions_taken: int = 0
    doubled: bool = False
    stood: bool = False
    surrendered: bool = False
    charlie: bool = False

    @property
    def is_blackjack(self) -> bool:
        return self.blackjack_eligible and self.cards.is_blackjack

    @property
    def is_complete(self) -> bool:
        return (
            self.stood
            or self.surrendered
            or self.charlie
            or self.cards.is_bust
        )


@dataclass(frozen=True, slots=True)
class HandResult:
    hand_index: int
    outcome: HandOutcome
    total: int
    wager: float
    payout: float
    profit: float
    card_count: int
    from_split: bool
    doubled: bool


@dataclass(frozen=True, slots=True)
class Settlement:
    round_id: int
    hand_results: tuple[HandResult, ...]
    dealer_total: int
    dealer_blackjack: bool
    dealer_bust: bool
    insurance_wager: float
    insurance_profit: float
    dealer_blackjack_refund: float
    reward: float
    balance: float


@dataclass(frozen=True, slots=True)
class GameSnapshot:
    round_id: int
    phase: RoundPhase
    active_hand_index: int | None
    player_totals: tuple[int, ...]
    player_wagers: tuple[float, ...]
    dealer_up_code: int | None
    dealer_up_value: int | None
    legal_actions: tuple[Action, ...]
    balance: float
    reward: float | None


class VisibleComposition:
    """Rank depletion visible to a fair player, excluding hidden cards."""

    __slots__ = ("num_decks", "_seen", "_buffer")

    def __init__(self, num_decks: int) -> None:
        if (
            isinstance(num_decks, (bool, np.bool_))
            or not isinstance(num_decks, (int, np.integer))
            or int(num_decks) < 1
        ):
            raise ValueError("num_decks must be a positive integer")
        self.num_decks = int(num_decks)
        self._seen = np.zeros(13, dtype=np.int32)
        self._buffer = np.empty(VISIBLE_COMPOSITION_SIZE, dtype=np.float32)

    @property
    def seen_rank_counts(self) -> NDArray[np.int32]:
        result = self._seen.view()
        result.flags.writeable = False
        return result

    def reset(self) -> None:
        self._seen.fill(0)

    def reveal(self, code: int) -> None:
        self._reveal_unchecked(_validate_code(code))

    def _reveal_unchecked(self, code: int) -> None:
        self._seen[code % 13] += 1

    def reveal_many(self, codes: NDArray[np.integer]) -> None:
        array = np.asarray(codes)
        if array.size:
            ranks = card_rank_indices(array)
            self._seen += np.bincount(ranks, minlength=13).astype(
                np.int32, copy=False
            )
        elif not np.issubdtype(array.dtype, np.integer):
            raise TypeError("card codes must have an integer dtype")

    def encode(
        self,
        *,
        out: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32]:
        result = self._buffer if out is None else out
        if not isinstance(result, np.ndarray):
            raise TypeError("out must be a numpy.ndarray")
        if result.shape != (VISIBLE_COMPOSITION_SIZE,) or result.dtype != np.float32:
            raise ValueError(
                f"out must be a float32 array with shape ({VISIBLE_COMPOSITION_SIZE},)"
            )
        if not result.flags.writeable:
            raise ValueError("out must be writable")
        np.divide(
            self._seen,
            4 * self.num_decks,
            out=result[:13],
            casting="unsafe",
        )
        result[13] = self.num_decks / (self.num_decks + 1.0)
        return result


class Dealer:
    """Dealer policy and physical shoe owner."""

    __slots__ = (
        "controller",
        "hand",
        "up_code",
        "hole_code",
        "hole_revealed",
    )

    def __init__(
        self,
        rules: BlackjackRules,
        *,
        rng: np.random.Generator | int | None = None,
        shoe: Shoe | None = None,
        counting_system: CountingSystem = HI_LO,
    ) -> None:
        if shoe is not None:
            if shoe.num_decks != rules.num_decks:
                raise ValueError("shoe.num_decks must match rules.num_decks")
            if not math.isclose(shoe.penetration, rules.penetration):
                raise ValueError("shoe.penetration must match rules.penetration")
            self.controller = BlackjackController(
                shoe, counting_system=counting_system
            )
        else:
            self.controller = BlackjackController(
                num_decks=rules.num_decks,
                counting_system=counting_system,
                rng=rng,
                penetration=rules.penetration,
            )
        self.hand = Hand()
        self.up_code: int | None = None
        self.hole_code: int | None = None
        self.hole_revealed = False

    @property
    def shoe(self) -> Shoe:
        return self.controller.shoe

    @property
    def has_blackjack(self) -> bool:
        return self.hand.is_blackjack

    def reset_hand(self) -> None:
        self.hand.clear()
        self.up_code = None
        self.hole_code = None
        self.hole_revealed = False

    def deal_up_card(self) -> int:
        code = self.controller.draw_code()
        self.hand.add(code)
        self.up_code = code
        return code

    def deal_hole_card(self) -> int:
        if self.hole_code is not None:
            raise RuntimeError("dealer already has a hole card")
        code = self.controller.draw_code()
        self.hand.add(code)
        self.hole_code = code
        return code

    def reveal_hole_card(self) -> int | None:
        if self.hole_code is None or self.hole_revealed:
            return None
        self.hole_revealed = True
        return self.hole_code

    def draw(self) -> int:
        code = self.controller.draw_code()
        self.hand.add(code)
        return code


class BlackjackGame:
    """A complete single-player blackjack table optimized for AI rollouts.

    A game owns one dealer, one persistent shoe, a bankroll, and the full round
    state machine. One call to :meth:`start_round` starts an episode; calls to
    :meth:`act` advance player decisions. Dealer play and settlement are
    automatic once the last player hand completes.
    """

    __slots__ = (
        "rules",
        "engine",
        "cash",
        "dealer",
        "visible_composition",
        "phase",
        "hands",
        "active_hand_index",
        "round_id",
        "settlement",
        "insurance_wager",
        "_insurance_profit",
        "_insurance_resolved",
        "_round_start_balance",
        "_initial_wager",
        "_peek_complete",
        "_early_peek_pending",
        "_mask_buffer",
        "_observation_buffer",
    )

    def __init__(
        self,
        rules: BlackjackRules | None = None,
        *,
        starting_balance: float = 1_000.0,
        cash: Cash | None = None,
        rng: np.random.Generator | int | None = None,
        shoe: Shoe | None = None,
        counting_system: CountingSystem = HI_LO,
    ) -> None:
        self.rules = rules or BlackjackRules()
        self.engine = RuleEngine(self.rules)
        self.cash = cash or Cash(starting_balance)
        self.dealer = Dealer(
            self.rules,
            rng=rng,
            shoe=shoe,
            counting_system=counting_system,
        )
        self.visible_composition = VisibleComposition(self.rules.num_decks)
        self.phase = RoundPhase.READY
        self.hands: list[PlayerHand] = []
        self.active_hand_index = 0
        self.round_id = 0
        self.settlement: Settlement | None = None
        self.insurance_wager = 0.0
        self._insurance_profit = 0.0
        self._insurance_resolved = True
        self._round_start_balance = self.cash.balance
        self._initial_wager = 0.0
        self._peek_complete = False
        self._early_peek_pending = False
        self._mask_buffer = np.empty(ACTION_COUNT, dtype=np.bool_)
        self._observation_buffer = np.empty(
            GAME_OBSERVATION_SIZE, dtype=np.float32
        )

    @property
    def controller(self) -> BlackjackController:
        return self.dealer.controller

    @property
    def shoe(self) -> Shoe:
        return self.dealer.shoe

    @property
    def active_hand(self) -> PlayerHand | None:
        if self.phase is not RoundPhase.PLAYER:
            return None
        if not 0 <= self.active_hand_index < len(self.hands):
            return None
        return self.hands[self.active_hand_index]

    @property
    def is_round_over(self) -> bool:
        return self.phase is RoundPhase.SETTLED

    @property
    def reward(self) -> float | None:
        return None if self.settlement is None else self.settlement.reward

    def _validate_wager(self, wager: float) -> float:
        if isinstance(wager, (bool, np.bool_)) or not isinstance(
            wager, (int, float, np.integer, np.floating)
        ):
            raise TypeError("wager must be a number")
        value = float(wager)
        if not math.isfinite(value):
            raise ValueError("wager must be finite")
        if not self.rules.table_minimum <= value <= self.rules.table_maximum:
            raise ValueError(
                f"wager must be between {self.rules.table_minimum:g} and "
                f"{self.rules.table_maximum:g}"
            )
        if self.cash.balance < value:
            raise ValueError("insufficient balance for wager")
        return value

    def _reveal(self, code: int) -> None:
        self.visible_composition._reveal_unchecked(code)

    def _draw_player_card(self, hand: PlayerHand) -> int:
        code = self.controller.draw_code()
        hand.cards.add(code)
        self._reveal(code)
        return code

    def _reshuffle_if_needed(self) -> bool:
        if (
            self.shoe.cut_card_reached
            or self.shoe.cards_remaining
            < max(4, self.rules.minimum_cards_before_round)
        ):
            self.controller.reset(shuffled=True)
            self.visible_composition.reset()
            return True
        return False

    def start_round(
        self, wager: float, *, return_snapshot: bool = True
    ) -> GameSnapshot | None:
        """Place a wager and deal the initial cards.

        Set ``return_snapshot=False`` in hot training loops to avoid allocating
        a diagnostic snapshot after every transition.
        """

        if self.phase not in (RoundPhase.READY, RoundPhase.SETTLED):
            raise RuntimeError("the current round is still active")
        wager = self._validate_wager(wager)
        self._reshuffle_if_needed()

        self.round_id += 1
        self.settlement = None
        self.hands.clear()
        self.active_hand_index = 0
        self.insurance_wager = 0.0
        self._insurance_profit = 0.0
        self._insurance_resolved = True
        self._peek_complete = False
        self._early_peek_pending = False
        self._initial_wager = wager
        self._round_start_balance = self.cash.balance
        self.cash.debit(wager)
        self.dealer.reset_hand()

        player = PlayerHand(Hand(), wager)
        self.hands.append(player)

        self._draw_player_card(player)
        dealer_up = self.dealer.deal_up_card()
        self._reveal(dealer_up)
        self._draw_player_card(player)
        if self.rules.hole_card_rule is HoleCardRule.AMERICAN:
            self.dealer.deal_hole_card()

        if self._dealer_up_is_ace() and self.rules.insurance_allowed:
            self.phase = RoundPhase.INSURANCE
        else:
            self._after_insurance_decision()
        return self.snapshot() if return_snapshot else None

    def _dealer_up_rank(self) -> int:
        if self.dealer.up_code is None:
            raise RuntimeError("dealer has no up card")
        return self.dealer.up_code % 13

    def _dealer_up_is_ace(self) -> bool:
        return self._dealer_up_rank() == 0

    def _dealer_up_is_ten_value(self) -> bool:
        return self._dealer_up_rank() >= 9

    def _dealer_can_have_blackjack(self) -> bool:
        return self._dealer_up_is_ace() or self._dealer_up_is_ten_value()

    def _after_insurance_decision(self) -> None:
        player = self.hands[0]
        if self.rules.hole_card_rule is HoleCardRule.AMERICAN:
            can_blackjack = self._dealer_can_have_blackjack()
            if (
                can_blackjack
                and self.rules.dealer_peeks_for_blackjack
                and self.rules.surrender is SurrenderRule.EARLY
                and not player.is_blackjack
            ):
                self._early_peek_pending = True
                self.phase = RoundPhase.PLAYER
                return
            if can_blackjack and self.rules.dealer_peeks_for_blackjack:
                if self._perform_peek():
                    return
            if player.is_blackjack:
                if can_blackjack and not self._peek_complete:
                    self._reveal_dealer_hole()
                    self._resolve_insurance(self.dealer.has_blackjack)
                    self._settle_round(self.dealer.has_blackjack)
                else:
                    self._settle_round(False)
                return
        elif player.is_blackjack:
            self._play_dealer_and_settle()
            return
        self.phase = RoundPhase.PLAYER
        self._ensure_active_hand_ready()

    def _reveal_dealer_hole(self) -> None:
        code = self.dealer.reveal_hole_card()
        if code is not None:
            self._reveal(code)

    def _perform_peek(self) -> bool:
        self._peek_complete = True
        self._early_peek_pending = False
        dealer_blackjack = self.dealer.has_blackjack
        self._resolve_insurance(dealer_blackjack)
        if dealer_blackjack:
            self._reveal_dealer_hole()
            self._settle_round(True)
            return True
        return False

    def _resolve_insurance(self, dealer_blackjack: bool) -> None:
        if self._insurance_resolved:
            return
        if dealer_blackjack:
            self.cash.credit(
                self.insurance_wager * (1.0 + self.rules.insurance_payout)
            )
            self._insurance_profit = (
                self.insurance_wager * self.rules.insurance_payout
            )
        else:
            self._insurance_profit = -self.insurance_wager
        self._insurance_resolved = True

    def _surrender_window_open(self) -> bool:
        hand = self.active_hand
        if hand is None or self.rules.surrender is SurrenderRule.NONE:
            return False
        if self.rules.surrender is SurrenderRule.EARLY:
            return not self._peek_complete
        if not self._dealer_can_have_blackjack():
            return True
        return self._peek_complete

    def legal_action_bits(self) -> int:
        if self.phase is RoundPhase.INSURANCE:
            bits = 1 << Action.DECLINE_INSURANCE
            insurance_cost = self._initial_wager * self.rules.insurance_fraction
            if self.cash.balance >= insurance_cost:
                bits |= 1 << Action.INSURANCE
            return bits
        hand = self.active_hand
        if hand is None:
            return 0
        return self.engine.legal_action_bits(
            hand,
            hand_count=len(self.hands),
            available_balance=self.cash.balance,
            surrender_window_open=self._surrender_window_open(),
        )

    def legal_actions(self) -> tuple[Action, ...]:
        return self.engine.actions_from_bits(self.legal_action_bits())

    def legal_action_mask(
        self, *, out: NDArray[np.bool_] | None = None
    ) -> NDArray[np.bool_]:
        result = self._mask_buffer if out is None else out
        return self.engine.bits_to_mask(self.legal_action_bits(), result)

    def act(
        self, action: Action | int, *, return_snapshot: bool = True
    ) -> GameSnapshot | None:
        """Apply one legal action and advance the round.

        Set ``return_snapshot=False`` when state is read through
        :meth:`observation` and :meth:`legal_action_mask` instead.
        """

        if isinstance(action, (bool, np.bool_)) or not isinstance(
            action, (int, np.integer)
        ):
            raise ValueError(f"unknown action: {action!r}")
        try:
            selected = Action(int(action))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown action: {action!r}") from exc
        legal_bits = self.legal_action_bits()
        if not legal_bits & (1 << selected):
            raise ValueError(
                f"illegal {selected.name} action during {self.phase.name} phase"
            )

        if self.phase is RoundPhase.INSURANCE:
            if selected is Action.INSURANCE:
                self.insurance_wager = (
                    self._initial_wager * self.rules.insurance_fraction
                )
                self.cash.debit(self.insurance_wager)
                self._insurance_resolved = False
            self._after_insurance_decision()
            return self.snapshot() if return_snapshot else None

        hand = self.active_hand
        if hand is None:
            raise RuntimeError("there is no active player hand")

        if self._early_peek_pending and selected is not Action.SURRENDER:
            if self._perform_peek():
                return self.snapshot() if return_snapshot else None
            # The action was legal before the peek and remains legal afterward.

        if selected is Action.STAND:
            hand.actions_taken += 1
            hand.stood = True
            self._advance_active_hand()
        elif selected is Action.HIT:
            hand.actions_taken += 1
            self._draw_player_card(hand)
            self._complete_after_draw(hand, forced_stand=False)
        elif selected is Action.DOUBLE:
            self.cash.debit(hand.wager)
            hand.wager *= 2.0
            hand.doubled = True
            hand.actions_taken += 1
            self._draw_player_card(hand)
            self._complete_after_draw(hand, forced_stand=True)
        elif selected is Action.SPLIT:
            self._split_active_hand()
        elif selected is Action.SURRENDER:
            hand.actions_taken += 1
            hand.surrendered = True
            self.cash.credit(hand.wager * 0.5)
            if self._early_peek_pending:
                self._early_peek_pending = False
                if not self._insurance_resolved:
                    self._peek_complete = True
                    self._reveal_dealer_hole()
                    dealer_blackjack = self.dealer.has_blackjack
                    self._resolve_insurance(dealer_blackjack)
                    self._settle_round(dealer_blackjack)
                    return self.snapshot() if return_snapshot else None
            self._advance_active_hand()
        return self.snapshot() if return_snapshot else None

    def _complete_after_draw(self, hand: PlayerHand, *, forced_stand: bool) -> None:
        if hand.cards.is_bust:
            self._advance_active_hand()
            return
        if (
            self.rules.charlie_card_count is not None
            and len(hand.cards) >= self.rules.charlie_card_count
        ):
            hand.charlie = True
            self._advance_active_hand()
            return
        if forced_stand or hand.cards.total == 21:
            hand.stood = True
            self._advance_active_hand()

    def _split_active_hand(self) -> None:
        parent = self.active_hand
        if parent is None:
            raise RuntimeError("there is no active hand to split")
        self.cash.debit(parent.wager)
        first_code = int(parent.cards._cards[0])
        second_code = int(parent.cards._cards[1])
        split_aces = first_code % 13 == 0 and second_code % 13 == 0
        depth = parent.split_depth + 1
        eligible = self.rules.blackjack_after_split
        first = PlayerHand(
            Hand((first_code,)),
            parent.wager,
            from_split=True,
            split_aces=split_aces,
            split_depth=depth,
            blackjack_eligible=eligible,
            needs_split_card=True,
        )
        second = PlayerHand(
            Hand((second_code,)),
            parent.wager,
            from_split=True,
            split_aces=split_aces,
            split_depth=depth,
            blackjack_eligible=eligible,
            needs_split_card=True,
        )
        self.hands[self.active_hand_index] = first
        self.hands.insert(self.active_hand_index + 1, second)
        self._ensure_active_hand_ready()

    def _advance_active_hand(self) -> None:
        self.active_hand_index += 1
        self._ensure_active_hand_ready()

    def _ensure_active_hand_ready(self) -> None:
        while self.active_hand_index < len(self.hands):
            hand = self.hands[self.active_hand_index]
            if hand.needs_split_card:
                hand.needs_split_card = False
                self._draw_player_card(hand)
            if hand.cards.is_bust:
                self.active_hand_index += 1
                continue
            if hand.cards.total == 21:
                hand.stood = True
                self.active_hand_index += 1
                continue
            if hand.split_aces and not self.rules.hit_split_aces:
                can_resplit = self.engine.can_split(
                    hand,
                    hand_count=len(self.hands),
                    available_balance=self.cash.balance,
                )
                if not can_resplit:
                    hand.stood = True
                    self.active_hand_index += 1
                    continue
            self.phase = RoundPhase.PLAYER
            return
        self._finish_player_phase()

    def _finish_player_phase(self) -> None:
        requires_dealer = any(
            not hand.cards.is_bust
            and not hand.surrendered
            and not hand.charlie
            for hand in self.hands
        )
        if requires_dealer or not self._insurance_resolved:
            self._play_dealer_and_settle()
        else:
            self._settle_round(False)

    def _play_dealer_and_settle(self) -> None:
        self.phase = RoundPhase.DEALER
        if self.rules.hole_card_rule is HoleCardRule.EUROPEAN:
            self.dealer.deal_hole_card()
        self._reveal_dealer_hole()
        dealer_blackjack = self.dealer.has_blackjack
        self._resolve_insurance(dealer_blackjack)
        if not dealer_blackjack:
            needs_total = any(
                not hand.cards.is_bust
                and not hand.surrendered
                and not hand.charlie
                and not hand.is_blackjack
                for hand in self.hands
            )
            while needs_total and self.engine.dealer_should_hit(
                self.dealer.hand.total, self.dealer.hand.is_soft
            ):
                self._reveal(self.dealer.draw())
        self._settle_round(dealer_blackjack)

    def _settle_round(self, dealer_blackjack: bool) -> None:
        dealer_total = self.dealer.hand.total
        dealer_bust = self.dealer.hand.is_bust
        refund = 0.0
        if (
            dealer_blackjack
            and self.rules.dealer_blackjack_loss_rule
            is DealerBlackjackLossRule.ORIGINAL_BET_ONLY
        ):
            exposed = sum(
                hand.wager for hand in self.hands if not hand.surrendered
            )
            refund = max(0.0, exposed - self._initial_wager)
            if refund:
                self.cash.credit(refund)

        results: list[HandResult] = []
        for index, hand in enumerate(self.hands):
            wager = hand.wager
            payout = 0.0
            if hand.surrendered:
                outcome = HandOutcome.SURRENDER
                payout = wager * 0.5
                profit = -wager * 0.5
            elif hand.cards.is_bust:
                outcome = HandOutcome.BUST
                profit = -wager
            elif hand.charlie:
                outcome = HandOutcome.CHARLIE
                payout = wager * (1.0 + self.rules.charlie_payout)
                profit = wager * self.rules.charlie_payout
                self.cash.credit(payout)
            elif hand.is_blackjack and dealer_blackjack:
                outcome = HandOutcome.PUSH
                payout = wager
                profit = 0.0
                self.cash.credit(payout)
            elif hand.is_blackjack:
                outcome = HandOutcome.BLACKJACK
                payout = wager * (1.0 + self.rules.blackjack_payout)
                profit = wager * self.rules.blackjack_payout
                self.cash.credit(payout)
            elif dealer_blackjack:
                outcome = HandOutcome.LOSS
                profit = -wager
            elif dealer_bust or hand.cards.total > dealer_total:
                outcome = HandOutcome.WIN
                payout = wager * 2.0
                profit = wager
                self.cash.credit(payout)
            elif hand.cards.total == dealer_total:
                outcome = HandOutcome.PUSH
                payout = wager
                profit = 0.0
                self.cash.credit(payout)
            else:
                outcome = HandOutcome.LOSS
                profit = -wager
            results.append(
                HandResult(
                    hand_index=index,
                    outcome=outcome,
                    total=hand.cards.total,
                    wager=wager,
                    payout=payout,
                    profit=profit,
                    card_count=len(hand.cards),
                    from_split=hand.from_split,
                    doubled=hand.doubled,
                )
            )

        reward = round(
            self.cash.balance - self._round_start_balance,
            self.cash.precision,
        )
        self.phase = RoundPhase.SETTLED
        self.settlement = Settlement(
            round_id=self.round_id,
            hand_results=tuple(results),
            dealer_total=dealer_total,
            dealer_blackjack=dealer_blackjack,
            dealer_bust=dealer_bust,
            insurance_wager=self.insurance_wager,
            insurance_profit=self._insurance_profit,
            dealer_blackjack_refund=refund,
            reward=reward,
            balance=self.cash.balance,
        )

    def observation(
        self, *, out: NDArray[np.float32] | None = None
    ) -> NDArray[np.float32]:
        """Return the stable 43-value, visible-information AI state."""

        state = self._observation_buffer if out is None else out
        if not isinstance(state, np.ndarray):
            raise TypeError("out must be a numpy.ndarray")
        if state.shape != (GAME_OBSERVATION_SIZE,) or state.dtype != np.float32:
            raise ValueError(
                f"out must be a float32 array with shape ({GAME_OBSERVATION_SIZE},)"
            )
        if not state.flags.writeable:
            raise ValueError("out must be writable")
        state.fill(0.0)
        self.visible_composition.encode(out=state[:14])

        hand = self.active_hand
        if hand is None and self.hands:
            hand = self.hands[min(self.active_hand_index, len(self.hands) - 1)]
        if hand is not None:
            state[14] = min(hand.cards.total, 21) / 21.0
            state[15] = float(hand.cards.is_soft)
            state[16] = len(hand.cards) / hand.cards.MAX_CARDS
            if len(hand.cards) == 2:
                first_code = int(hand.cards._cards[0])
                second_code = int(hand.cards._cards[1])
                if self.engine.cards_can_split(first_code, second_code):
                    state[17 + first_code % 13] = 1.0

        if self.dealer.up_code is not None:
            rank = self.dealer.up_code % 13
            bucket = 0 if rank == 0 else min(rank, 9)
            state[30 + bucket] = 1.0

        bits = self.legal_action_bits()
        state[40] = float(bool(bits & (1 << Action.DOUBLE)))
        state[41] = float(bool(bits & (1 << Action.SPLIT)))
        state[42] = float(bool(bits & (1 << Action.SURRENDER)))
        return state

    def snapshot(self) -> GameSnapshot:
        dealer_value = None
        if self.dealer.up_code is not None:
            dealer_value = min(self.dealer.up_code % 13 + 1, 10)
        active = (
            self.active_hand_index
            if self.phase is RoundPhase.PLAYER
            and self.active_hand_index < len(self.hands)
            else None
        )
        return GameSnapshot(
            round_id=self.round_id,
            phase=self.phase,
            active_hand_index=active,
            player_totals=tuple(hand.cards.total for hand in self.hands),
            player_wagers=tuple(hand.wager for hand in self.hands),
            dealer_up_code=self.dealer.up_code,
            dealer_up_value=dealer_value,
            legal_actions=self.legal_actions(),
            balance=self.cash.balance,
            reward=self.reward,
        )
