from __future__ import annotations

import numpy as np
import pytest

from blackjack_ai import (
    ACTION_COUNT,
    GAME_OBSERVATION_SIZE,
    Action,
    BlackjackGame,
    BlackjackRules,
    Cash,
    DealerBlackjackLossRule,
    HandOutcome,
    HoleCardRule,
    RoundPhase,
    Shoe,
    SurrenderRule,
)


def rules(**overrides: object) -> BlackjackRules:
    values: dict[str, object] = {
        "num_decks": 1,
        "penetration": 1.0,
        "minimum_cards_before_round": 0,
    }
    values.update(overrides)
    return BlackjackRules(**values)  # type: ignore[arg-type]


def rigged_shoe(draw_order: list[int], *, num_decks: int = 1) -> Shoe:
    remaining = list(range(52)) * num_decks
    for code in draw_order:
        remaining.remove(code)
    shoe = Shoe(num_decks, shuffled=False, penetration=1.0)
    shoe._cards[:] = np.asarray(draw_order + remaining, dtype=np.uint8)  # type: ignore[attr-defined]
    return shoe


def game_for(
    draw_order: list[int],
    *,
    game_rules: BlackjackRules | None = None,
    balance: float = 100,
) -> BlackjackGame:
    configured = game_rules or rules()
    return BlackjackGame(
        configured,
        cash=Cash(balance),
        shoe=rigged_shoe(draw_order, num_decks=configured.num_decks),
    )


def test_player_natural_blackjack_pays_configured_three_to_two() -> None:
    game = game_for([0, 5, 12, 7])  # player A/K, dealer 6/8

    state = game.start_round(10)

    assert state.phase is RoundPhase.SETTLED
    assert state.reward == 15
    assert game.cash.balance == 115
    assert game.settlement is not None
    assert game.settlement.hand_results[0].outcome is HandOutcome.BLACKJACK


def test_dealer_blackjack_with_insurance_breaks_even() -> None:
    game = game_for([8, 0, 7, 12])  # player 9/8, dealer A/K

    state = game.start_round(10)
    assert state.phase is RoundPhase.INSURANCE
    assert state.legal_actions == (Action.INSURANCE, Action.DECLINE_INSURANCE)

    state = game.act(Action.INSURANCE)

    assert state.phase is RoundPhase.SETTLED
    assert state.reward == 0
    assert game.cash.balance == 100
    assert game.settlement is not None
    assert game.settlement.insurance_wager == 5
    assert game.settlement.insurance_profit == 10
    assert game.settlement.dealer_blackjack


def test_european_insurance_resolves_when_hole_card_is_dealt_later() -> None:
    configured = rules(hole_card_rule=HoleCardRule.EUROPEAN)
    game = game_for([8, 0, 7, 12], game_rules=configured)
    game.start_round(10)

    state = game.act(Action.INSURANCE)
    assert state.phase is RoundPhase.PLAYER
    state = game.act(Action.STAND)

    assert state.reward == 0
    assert game.settlement is not None
    assert game.settlement.dealer_blackjack
    assert game.settlement.insurance_profit == 10


def test_player_and_dealer_blackjack_push() -> None:
    game = game_for([0, 13, 12, 25])  # both A/K

    game.start_round(10)
    state = game.act(Action.DECLINE_INSURANCE)

    assert state.reward == 0
    assert game.cash.balance == 100
    assert game.settlement is not None
    assert game.settlement.hand_results[0].outcome is HandOutcome.PUSH


def test_hit_bust_settles_without_dealer_play() -> None:
    game = game_for([12, 5, 8, 9, 17])  # K/9 hits 5
    game.start_round(10)

    state = game.act(Action.HIT)

    assert state.phase is RoundPhase.SETTLED
    assert state.reward == -10
    assert game.hands[0].cards.total == 24
    assert game.settlement is not None
    assert game.settlement.hand_results[0].outcome is HandOutcome.BUST


def test_stand_then_dealer_bust_wins() -> None:
    game = game_for([9, 5, 7, 22, 12])  # player 18, dealer 16 then K
    game.start_round(10)

    state = game.act(Action.STAND)

    assert state.reward == 10
    assert game.cash.balance == 110
    assert game.dealer.hand.is_bust
    assert game.settlement is not None
    assert game.settlement.hand_results[0].outcome is HandOutcome.WIN


def test_double_draws_once_and_scales_reward() -> None:
    game = game_for([4, 5, 18, 9, 22, 12])  # player 11 -> 21; dealer busts
    game.start_round(10)

    state = game.act(Action.DOUBLE)

    assert state.reward == 20
    assert game.hands[0].doubled
    assert game.hands[0].wager == 20
    assert len(game.hands[0].cards) == 3
    assert game.settlement is not None
    assert game.settlement.hand_results[0].profit == 20


def test_late_surrender_returns_half_the_wager() -> None:
    game = game_for([9, 5, 18, 22])  # player 16 versus dealer 6
    game.start_round(10)

    state = game.act(Action.SURRENDER)

    assert state.reward == -5
    assert game.cash.balance == 95
    assert game.settlement is not None
    assert game.settlement.hand_results[0].outcome is HandOutcome.SURRENDER


def test_early_surrender_beats_dealer_blackjack() -> None:
    configured = rules(
        surrender=SurrenderRule.EARLY,
        insurance_allowed=False,
    )
    game = game_for([8, 0, 7, 12], game_rules=configured)
    state = game.start_round(10)
    assert Action.SURRENDER in state.legal_actions

    state = game.act(Action.SURRENDER)

    assert state.reward == -5
    assert game.settlement is not None
    assert game.settlement.hand_results[0].outcome is HandOutcome.SURRENDER


def test_early_surrender_and_insurance_settle_both_wagers() -> None:
    configured = rules(surrender=SurrenderRule.EARLY)
    game = game_for([8, 0, 7, 12], game_rules=configured)
    game.start_round(10)
    game.act(Action.INSURANCE)

    state = game.act(Action.SURRENDER)

    assert state.reward == 5  # -5 surrender +10 insurance profit
    assert game.settlement is not None
    assert game.settlement.dealer_blackjack
    assert game.settlement.insurance_profit == 10


def test_losing_insurance_and_surrender_are_both_accounted() -> None:
    game = game_for([8, 0, 7, 5])  # dealer A/6, not blackjack
    game.start_round(10)
    game.act(Action.INSURANCE)

    state = game.act(Action.SURRENDER)

    assert state.reward == -10  # -5 insurance, -5 surrender
    assert game.settlement is not None
    assert game.settlement.insurance_profit == -5


def test_split_hands_are_played_in_order_and_settled_independently() -> None:
    game = game_for([7, 5, 20, 9, 22, 8, 12])
    game.start_round(10)

    state = game.act(Action.SPLIT)
    assert len(game.hands) == 2
    assert state.active_hand_index == 0
    assert state.player_totals == (18, 8)
    assert game.cash.balance == 80

    state = game.act(Action.STAND)
    assert state.active_hand_index == 1
    assert state.player_totals == (18, 17)

    state = game.act(Action.STAND)
    assert state.reward == 20
    assert game.cash.balance == 120
    assert game.settlement is not None
    assert [result.outcome for result in game.settlement.hand_results] == [
        HandOutcome.WIN,
        HandOutcome.WIN,
    ]


def test_split_aces_receive_one_card_and_are_not_natural_blackjacks() -> None:
    game = game_for([0, 5, 13, 9, 12, 8, 2])
    game.start_round(10)

    state = game.act(Action.SPLIT)

    assert state.phase is RoundPhase.SETTLED
    assert len(game.hands) == 2
    assert all(len(hand.cards) == 2 for hand in game.hands)
    assert all(not hand.is_blackjack for hand in game.hands)
    assert game.settlement is not None
    assert all(
        result.outcome is not HandOutcome.BLACKJACK
        for result in game.settlement.hand_results
    )


def test_soft_17_rule_changes_dealer_behavior_and_result() -> None:
    order = [9, 0, 7, 5, 1]  # player 18, dealer soft 17, then dealer 2
    stands = game_for(
        order,
        game_rules=rules(dealer_hits_soft_17=False, insurance_allowed=False),
    )
    hits = game_for(
        order,
        game_rules=rules(dealer_hits_soft_17=True, insurance_allowed=False),
    )
    stands.start_round(10)
    hits.start_round(10)

    stand_result = stands.act(Action.STAND)
    hit_result = hits.act(Action.STAND)

    assert stand_result.reward == 10
    assert hit_result.reward == -10
    assert stands.dealer.hand.total == 17
    assert hits.dealer.hand.total == 19


def test_blackjack_payout_is_tunable() -> None:
    game = game_for(
        [0, 5, 12, 7],
        game_rules=rules(blackjack_payout=1.2),
    )

    state = game.start_round(10)

    assert state.reward == 12
    assert game.cash.balance == 112


def test_insurance_with_player_blackjack_behaves_as_even_money() -> None:
    game = game_for([0, 13, 12, 18])  # player A/K, dealer A/6
    game.start_round(10)

    state = game.act(Action.INSURANCE)

    assert state.reward == 10
    assert game.settlement is not None
    assert game.settlement.insurance_profit == -5
    assert game.settlement.hand_results[0].outcome is HandOutcome.BLACKJACK


def test_ten_value_cards_can_split_under_default_value_rule() -> None:
    game = game_for([9, 5, 12, 22])  # player 10/K

    state = game.start_round(10)

    assert Action.SPLIT in state.legal_actions


def test_aces_can_resplit_when_enabled() -> None:
    configured = rules(resplit_aces=True, max_split_hands=3)
    game = game_for(
        [0, 5, 13, 9, 26, 12, 8, 7, 25],
        game_rules=configured,
    )
    game.start_round(10)
    first_split = game.act(Action.SPLIT)
    assert Action.SPLIT in first_split.legal_actions

    state = game.act(Action.SPLIT)

    assert state.phase is RoundPhase.SETTLED
    assert len(game.hands) == 3
    assert all(hand.split_aces for hand in game.hands)


def test_double_after_hit_can_be_enabled() -> None:
    configured = rules(double_on_any_number_of_cards=True)
    game = game_for([1, 5, 2, 9, 15, 22, 12], game_rules=configured)
    game.start_round(10)

    state = game.act(Action.HIT)
    assert Action.DOUBLE in state.legal_actions
    state = game.act(Action.DOUBLE)

    assert state.reward == 20
    assert game.hands[0].doubled


def test_insufficient_balance_masks_double_and_split() -> None:
    game = game_for([7, 5, 20, 9], balance=10)

    state = game.start_round(10)

    assert Action.DOUBLE not in state.legal_actions
    assert Action.SPLIT not in state.legal_actions
    with pytest.raises(ValueError, match="illegal SPLIT"):
        game.act(Action.SPLIT)
    assert game.cash.balance == 0
    assert len(game.hands) == 1


def test_table_limits_and_active_round_are_enforced() -> None:
    configured = rules(table_minimum=5, table_maximum=25)
    game = game_for([9, 5, 7, 22], game_rules=configured)

    with pytest.raises(ValueError, match="between"):
        game.start_round(1)
    game.start_round(5)
    with pytest.raises(RuntimeError, match="still active"):
        game.start_round(5)


def test_visible_observation_does_not_leak_hidden_hole_rank() -> None:
    first = game_for([9, 5, 7, 0])
    second = game_for([9, 5, 7, 12])

    first.start_round(10)
    second.start_round(10)
    first_state = first.observation().copy()
    second_state = second.observation().copy()

    np.testing.assert_array_equal(first_state, second_state)
    assert first.visible_composition.seen_rank_counts.sum() == 3
    assert first.visible_composition.seen_rank_counts[0] == 0

    first.act(Action.STAND)
    assert first.visible_composition.seen_rank_counts[0] == 1


def test_game_observation_and_mask_have_stable_nn_contract() -> None:
    game = game_for([7, 5, 20, 9])
    game.start_round(10)

    observation = game.observation().copy()
    mask = game.legal_action_mask().copy()

    assert observation.shape == (GAME_OBSERVATION_SIZE,)
    assert observation.dtype == np.float32
    assert np.all(np.isfinite(observation))
    assert mask.shape == (ACTION_COUNT,)
    assert mask.dtype == np.bool_
    assert mask[Action.SPLIT]


def test_observation_and_mask_support_reusable_output_buffers() -> None:
    game = game_for([9, 5, 7, 22])
    game.start_round(10)
    observation_out = np.empty(GAME_OBSERVATION_SIZE, dtype=np.float32)
    mask_out = np.empty(ACTION_COUNT, dtype=np.bool_)

    observation = game.observation(out=observation_out)
    mask = game.legal_action_mask(out=mask_out)

    assert observation is observation_out
    assert mask is mask_out
    with pytest.raises(ValueError):
        game.observation(out=np.empty(42, dtype=np.float32))
    with pytest.raises(ValueError):
        game.legal_action_mask(out=np.empty(ACTION_COUNT, dtype=np.float32))


def test_hot_api_can_skip_snapshot_allocations() -> None:
    game = game_for([9, 5, 7, 22, 12])

    assert game.start_round(10, return_snapshot=False) is None
    assert game.act(Action.STAND, return_snapshot=False) is None
    assert game.is_round_over
    assert game.reward == 10


def test_charlie_rule_auto_wins_at_configured_card_count() -> None:
    configured = rules(charlie_card_count=3)
    game = game_for([1, 9, 14, 22, 27], game_rules=configured)
    game.start_round(10)

    state = game.act(Action.HIT)

    assert state.reward == 10
    assert game.settlement is not None
    assert game.settlement.hand_results[0].outcome is HandOutcome.CHARLIE


@pytest.mark.parametrize(
    ("loss_rule", "expected_reward", "expected_refund"),
    [
        (DealerBlackjackLossRule.ALL_BETS, -20, 0),
        (DealerBlackjackLossRule.ORIGINAL_BET_ONLY, -10, 10),
    ],
)
def test_european_no_hole_card_blackjack_exposure(
    loss_rule: DealerBlackjackLossRule,
    expected_reward: float,
    expected_refund: float,
) -> None:
    configured = rules(
        hole_card_rule=HoleCardRule.EUROPEAN,
        dealer_blackjack_loss_rule=loss_rule,
        insurance_allowed=False,
        surrender=SurrenderRule.NONE,
    )
    game = game_for([7, 0, 20, 1, 2, 12], game_rules=configured)
    game.start_round(10)
    game.act(Action.SPLIT)
    game.act(Action.STAND)
    state = game.act(Action.STAND)

    assert state.reward == expected_reward
    assert game.settlement is not None
    assert game.settlement.dealer_blackjack
    assert game.settlement.dealer_blackjack_refund == expected_refund


def test_seeded_games_produce_identical_trajectories() -> None:
    configured = BlackjackRules(surrender=SurrenderRule.NONE)
    first = BlackjackGame(configured, rng=123, starting_balance=10_000)
    second = BlackjackGame(configured, rng=123, starting_balance=10_000)

    for _ in range(100):
        left = first.start_round(1)
        right = second.start_round(1)
        assert left.player_totals == right.player_totals
        assert left.dealer_up_code == right.dealer_up_code
        while not first.is_round_over:
            action = first.legal_actions()[0]
            assert action in second.legal_actions()
            left = first.act(action)
            right = second.act(action)
        assert left.reward == right.reward


def test_cut_card_causes_between_round_reshuffle_and_visible_count_reset() -> None:
    configured = BlackjackRules(num_decks=1, penetration=0.1)
    game = BlackjackGame(configured, rng=123, starting_balance=1_000)
    game.controller.draw_codes(6)
    game.visible_composition.reveal_many(np.array([0, 1, 2], dtype=np.uint8))
    assert game.shoe.cut_card_reached

    game.start_round(1)

    assert game.shoe.cards_dealt in (3, 4)
    assert game.visible_composition.seen_rank_counts.sum() in (3, 4)


def test_visible_count_persists_across_rounds_until_shuffle() -> None:
    configured = BlackjackRules(penetration=1.0, minimum_cards_before_round=0)
    game = BlackjackGame(configured, rng=444, starting_balance=10_000)
    game.start_round(1)
    while not game.is_round_over:
        game.act(game.legal_actions()[0])
    first_round_seen = int(game.visible_composition.seen_rank_counts.sum())

    game.start_round(1)

    assert int(game.visible_composition.seen_rank_counts.sum()) >= first_round_seen + 3


def test_invalid_action_does_not_mutate_round() -> None:
    game = game_for([9, 5, 7, 22])
    before = game.start_round(10)
    balance = game.cash.balance

    with pytest.raises(ValueError, match="illegal SPLIT"):
        game.act(Action.SPLIT)

    after = game.snapshot()
    assert after == before
    assert game.cash.balance == balance


def test_thousands_of_random_legal_rounds_preserve_invariants() -> None:
    rng = np.random.default_rng(991)
    game = BlackjackGame(
        BlackjackRules(),
        rng=12345,
        starting_balance=1_000_000,
    )

    for _ in range(2_000):
        game.start_round(1)
        decisions = 0
        while not game.is_round_over:
            actions = game.legal_actions()
            assert actions
            game.act(actions[int(rng.integers(len(actions)))])
            decisions += 1
            assert decisions < 100
        assert game.settlement is not None
        assert math_is_finite(game.settlement.reward)
        assert math_is_finite(game.cash.balance)
        assert np.all(np.isfinite(game.observation()))
        assert not game.legal_action_mask().any()
        accounted = (
            sum(result.profit for result in game.settlement.hand_results)
            + game.settlement.insurance_profit
            + game.settlement.dealer_blackjack_refund
        )
        assert accounted == pytest.approx(game.settlement.reward)
        assert game.shoe.rank_counts().sum() == game.shoe.cards_remaining


@pytest.mark.parametrize(
    "configured",
    [
        BlackjackRules(dealer_hits_soft_17=True),
        BlackjackRules(
            hole_card_rule=HoleCardRule.EUROPEAN,
            dealer_blackjack_loss_rule=DealerBlackjackLossRule.ALL_BETS,
        ),
        BlackjackRules(
            hole_card_rule=HoleCardRule.EUROPEAN,
            dealer_blackjack_loss_rule=DealerBlackjackLossRule.ORIGINAL_BET_ONLY,
        ),
        BlackjackRules(surrender=SurrenderRule.EARLY),
        BlackjackRules(surrender=SurrenderRule.NONE, insurance_allowed=False),
        BlackjackRules(
            double_after_split=False,
            double_allowed_totals=(9, 10, 11),
            split_by_value=False,
        ),
        BlackjackRules(
            max_split_hands=6,
            resplit_aces=True,
            hit_split_aces=True,
            surrender_after_split=True,
        ),
        BlackjackRules(charlie_card_count=5, blackjack_payout=1.2),
    ],
)
def test_randomized_rules_matrix_preserves_accounting_and_legality(
    configured: BlackjackRules,
) -> None:
    action_rng = np.random.default_rng(8800 + hash(configured) % 1_000)
    game = BlackjackGame(configured, rng=77, starting_balance=1_000_000)

    for _ in range(300):
        game.start_round(1, return_snapshot=False)
        decisions = 0
        while not game.is_round_over:
            bits = game.legal_action_bits()
            mask = game.legal_action_mask()
            actions = game.legal_actions()
            assert actions
            assert all(mask[action] for action in actions)
            assert bits == sum(1 << action for action in actions)
            game.act(
                actions[int(action_rng.integers(len(actions)))],
                return_snapshot=False,
            )
            decisions += 1
            assert decisions < 100

        settlement = game.settlement
        assert settlement is not None
        accounted = (
            sum(result.profit for result in settlement.hand_results)
            + settlement.insurance_profit
            + settlement.dealer_blackjack_refund
        )
        assert settlement.reward == pytest.approx(accounted)
        assert not game.legal_action_mask().any()
        assert np.all(np.isfinite(game.observation()))


def math_is_finite(value: float) -> bool:
    return bool(np.isfinite(value))
