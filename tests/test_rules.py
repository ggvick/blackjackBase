import numpy as np
import pytest

from blackjack_ai import (
    ACTION_COUNT,
    Action,
    BlackjackRules,
    Cash,
    Hand,
    PlayerHand,
    RuleEngine,
    SurrenderRule,
)


def player_hand(
    codes: tuple[int, ...],
    *,
    wager: float = 10,
    from_split: bool = False,
    split_aces: bool = False,
) -> PlayerHand:
    return PlayerHand(
        Hand(codes),
        wager,
        from_split=from_split,
        split_aces=split_aces,
    )


def test_dealer_soft_17_policy_is_configurable() -> None:
    stand = RuleEngine(BlackjackRules(dealer_hits_soft_17=False))
    hit = RuleEngine(BlackjackRules(dealer_hits_soft_17=True))

    assert not stand.dealer_should_hit(17, True)
    assert hit.dealer_should_hit(17, True)
    assert not hit.dealer_should_hit(17, False)
    assert hit.dealer_should_hit(16, False)


def test_split_matching_can_use_value_or_exact_rank() -> None:
    by_value = RuleEngine(BlackjackRules(split_by_value=True))
    by_rank = RuleEngine(BlackjackRules(split_by_value=False))

    assert by_value.cards_can_split(9, 12)  # ten and king
    assert not by_rank.cards_can_split(9, 12)
    assert by_rank.cards_can_split(7, 20)  # eights of different suits


def test_legal_action_bits_cover_initial_pair() -> None:
    engine = RuleEngine(BlackjackRules(surrender=SurrenderRule.LATE))
    hand = player_hand((7, 20))

    bits = engine.legal_action_bits(
        hand,
        hand_count=1,
        available_balance=100,
        surrender_window_open=True,
    )

    assert engine.actions_from_bits(bits) == (
        Action.STAND,
        Action.HIT,
        Action.DOUBLE,
        Action.SPLIT,
        Action.SURRENDER,
    )
    mask = engine.bits_to_mask(bits)
    assert mask.dtype == np.bool_
    assert mask.shape == (ACTION_COUNT,)
    assert not mask[Action.INSURANCE]


def test_cash_and_split_limit_remove_expensive_actions() -> None:
    engine = RuleEngine(BlackjackRules(max_split_hands=1))
    hand = player_hand((7, 20))

    bits = engine.legal_action_bits(
        hand,
        hand_count=1,
        available_balance=0,
        surrender_window_open=False,
    )

    assert not bits & (1 << Action.DOUBLE)
    assert not bits & (1 << Action.SPLIT)


def test_double_restrictions_and_split_rules() -> None:
    engine = RuleEngine(
        BlackjackRules(
            double_allowed_totals=(9, 10, 11),
            double_after_split=False,
        )
    )

    assert engine.can_double(player_hand((4, 17)), available_balance=10)  # 5+5
    assert not engine.can_double(player_hand((2, 16)), available_balance=10)
    assert not engine.can_double(
        player_hand((4, 17), from_split=True), available_balance=10
    )


def test_surrender_after_split_is_independently_configurable() -> None:
    hand = player_hand((4, 17), from_split=True)
    forbidden = RuleEngine(BlackjackRules(surrender_after_split=False))
    allowed = RuleEngine(BlackjackRules(surrender_after_split=True))

    assert not forbidden.can_surrender(hand, window_open=True)
    assert allowed.can_surrender(hand, window_open=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_decks": 0},
        {"penetration": 0},
        {"blackjack_payout": 0},
        {"table_minimum": 10, "table_maximum": 5},
        {"max_split_hands": 0},
        {"double_allowed_totals": (1, 22)},
        {"charlie_card_count": 2},
        {"minimum_cards_before_round": -1},
        {"num_decks": 1, "minimum_cards_before_round": 53},
    ],
)
def test_invalid_rules_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        BlackjackRules(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "name",
    [
        "dealer_hits_soft_17",
        "dealer_peeks_for_blackjack",
        "blackjack_after_split",
        "double_after_split",
        "double_on_any_number_of_cards",
        "split_by_value",
        "resplit_aces",
        "hit_split_aces",
        "surrender_after_split",
        "insurance_allowed",
    ],
)
def test_rule_flags_require_actual_booleans(name: str) -> None:
    with pytest.raises(TypeError, match="must be a bool"):
        BlackjackRules(**{name: 1})  # type: ignore[arg-type]


def test_boolean_penetration_is_not_treated_as_one() -> None:
    with pytest.raises(ValueError):
        BlackjackRules(penetration=True)


def test_optimized_legal_bits_match_individual_legality_apis() -> None:
    rng = np.random.default_rng(4242)
    rule_sets = (
        BlackjackRules(),
        BlackjackRules(
            double_after_split=False,
            double_allowed_totals=(9, 10, 11),
            split_by_value=False,
            surrender_after_split=True,
        ),
        BlackjackRules(
            double_on_any_number_of_cards=True,
            resplit_aces=True,
            hit_split_aces=True,
            max_split_hands=6,
        ),
    )

    for configured in rule_sets:
        engine = RuleEngine(configured)
        for _ in range(500):
            card_count = int(rng.integers(1, 7))
            hand = player_hand(
                tuple(int(code) for code in rng.integers(0, 52, card_count)),
                wager=float(rng.integers(1, 20)),
                from_split=bool(rng.integers(2)),
                split_aces=bool(rng.integers(2)),
            )
            hand.actions_taken = int(rng.integers(3))
            hand.stood = bool(rng.integers(8) == 0)
            hand.surrendered = bool(rng.integers(12) == 0)
            balance = float(rng.integers(0, 30))
            hand_count = int(rng.integers(1, configured.max_split_hands + 2))
            surrender_open = bool(rng.integers(2))

            expected = 0
            if not hand.is_complete:
                expected |= 1 << Action.STAND
                if hand.cards.total < 21 and (
                    not hand.split_aces or configured.hit_split_aces
                ):
                    expected |= 1 << Action.HIT
                if engine.can_double(hand, available_balance=balance):
                    expected |= 1 << Action.DOUBLE
                if engine.can_split(
                    hand,
                    hand_count=hand_count,
                    available_balance=balance,
                ):
                    expected |= 1 << Action.SPLIT
                if engine.can_surrender(hand, window_open=surrender_open):
                    expected |= 1 << Action.SURRENDER

            assert engine.legal_action_bits(
                hand,
                hand_count=hand_count,
                available_balance=balance,
                surrender_window_open=surrender_open,
            ) == expected
