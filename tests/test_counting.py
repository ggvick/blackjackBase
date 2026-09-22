import numpy as np
import pytest

from blackjack_ai import (
    COMPOSITION_COUNT_SIZE,
    HI_LO,
    BlackjackController,
    CountingSystem,
    Shoe,
    ShoeEmptyError,
)


def test_hi_lo_count_and_true_count() -> None:
    controller = BlackjackController(num_decks=1, shuffled=False)

    controller.draw_code()  # ace: -1
    assert controller.running_count == -1
    assert controller.true_count == pytest.approx(-1 / (51 / 52))

    controller.draw_code()  # two: +1
    assert controller.running_count == 0
    assert controller.true_count == 0


def test_count_cannot_drift_when_shoe_is_drawn_directly() -> None:
    shoe = Shoe(1, shuffled=False)
    controller = BlackjackController(shoe)

    shoe.draw_codes(6)  # A, 2, 3, 4, 5, 6 => -1 + five low cards
    assert controller.running_count == 4
    np.testing.assert_array_equal(
        controller.seen_rank_counts[:7], [1, 1, 1, 1, 1, 1, 0]
    )


def test_count_resynchronizes_after_direct_reset_and_draw() -> None:
    shoe = Shoe(1, shuffled=False)
    controller = BlackjackController(shoe)
    shoe.draw_codes(2)
    assert controller.running_count == 0

    shoe.reset(shuffled=False)
    shoe.draw_code()  # ace

    assert controller.running_count == -1


def test_full_balanced_shoe_finishes_at_zero_and_true_count_is_undefined() -> None:
    controller = BlackjackController(num_decks=2, shuffled=False)
    controller.draw_codes(104)

    assert controller.running_count == 0
    with pytest.raises(ShoeEmptyError):
        _ = controller.true_count
    observation = controller.observation()
    assert np.all(np.isfinite(observation))
    assert observation[14] == 0


def test_exact_composition_features() -> None:
    controller = BlackjackController(num_decks=1, shuffled=False)

    assert controller.high_cards_remaining == 20
    assert controller.ten_value_probability == pytest.approx(16 / 52)
    assert controller.ace_probability == pytest.approx(4 / 52)
    observation = controller.observation()
    assert observation.shape == (18,)
    assert observation.dtype == np.float32
    assert observation[:13].sum() == pytest.approx(1.0)


def test_observation_can_reuse_output_buffer() -> None:
    controller = BlackjackController(num_decks=1, shuffled=False)
    controller.draw_codes(7)
    expected = controller.observation()
    output = np.empty(18, dtype=np.float32)

    actual = controller.observation(out=output)

    assert actual is output
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("output", "error"),
    [
        (np.empty(17, dtype=np.float32), ValueError),
        (np.empty(18, dtype=np.float64), TypeError),
        ([0.0] * 18, TypeError),
    ],
)
def test_observation_rejects_invalid_output_buffer(
    output: object, error: type[Exception]
) -> None:
    controller = BlackjackController(num_decks=1, shuffled=False)

    with pytest.raises(error):
        controller.observation(out=output)  # type: ignore[arg-type]


def test_terminal_observation_preserves_unbalanced_running_count() -> None:
    system = CountingSystem("always one", tuple([1.0] * 13))
    controller = BlackjackController(
        num_decks=1, shuffled=False, counting_system=system
    )
    controller.draw_codes(52)

    observation = controller.observation()

    assert np.all(np.isfinite(observation))
    assert observation[13] == 52
    assert observation[14] == 0
    assert observation[15] == 1


def test_composition_count_is_bounded_fixed_and_exact() -> None:
    controller = BlackjackController(num_decks=1, shuffled=False)
    controller.draw_codes(2)  # ace and two of clubs

    count = controller.composition_count()

    assert count.shape == (COMPOSITION_COUNT_SIZE,)
    assert count.dtype == np.float32
    assert np.all(np.isfinite(count))
    assert np.all((0 <= count) & (count <= 1))
    np.testing.assert_allclose(count[:2], [0.25, 0.25])
    np.testing.assert_array_equal(count[2:13], np.zeros(11, dtype=np.float32))
    assert count[:13].mean() == pytest.approx(2 / 52)
    assert count[13] == pytest.approx(0.5)


def test_composition_count_reconstructs_true_count_without_information_loss() -> None:
    controller = BlackjackController(num_decks=6, shuffled=False)
    controller.draw_codes(73)
    composition = controller.composition_count(dtype=np.float64)

    rank_depletion = composition[:13]
    reconstructed_running = float(
        4 * controller.shoe.num_decks
        * (rank_depletion @ controller.counting_system.tags)
    )
    reconstructed_decks_remaining = controller.shoe.num_decks * (
        1.0 - float(rank_depletion.mean())
    )

    assert reconstructed_running == pytest.approx(controller.running_count)
    assert reconstructed_decks_remaining == pytest.approx(
        controller.shoe.decks_remaining
    )
    assert reconstructed_running / reconstructed_decks_remaining == pytest.approx(
        controller.true_count
    )


def test_composition_count_can_reuse_output_buffer() -> None:
    controller = BlackjackController(num_decks=8, rng=42)
    controller.draw_codes(37)
    output = np.empty(COMPOSITION_COUNT_SIZE, dtype=np.float32)

    result = controller.composition_count(out=output)

    assert result is output
    np.testing.assert_array_equal(result, controller.composition_count())


def test_snapshot_and_reset() -> None:
    controller = BlackjackController(num_decks=1, shuffled=False, penetration=0.5)
    controller.draw_codes(26)
    snapshot = controller.snapshot()

    assert snapshot.cards_seen == 26
    assert snapshot.cards_remaining == 26
    assert snapshot.decks_remaining == 0.5
    assert snapshot.fraction_dealt == 0.5
    assert snapshot.cut_card_reached

    controller.reset(shuffled=False)
    assert controller.running_count == 0
    assert controller.shoe.cards_dealt == 0


def test_custom_fractional_count_system() -> None:
    system = CountingSystem("test", tuple([0.5] * 13))
    controller = BlackjackController(
        num_decks=1, shuffled=False, counting_system=system
    )
    controller.draw_codes(4)
    assert controller.running_count == 2.0
    assert not system.balanced
    assert HI_LO.balanced


def test_counting_system_defensively_owns_tags() -> None:
    tags = np.zeros(13)
    system = CountingSystem("zero", tags)
    tags[0] = 99

    assert system.tags[0] == 0
    assert not system.tags.flags.writeable
