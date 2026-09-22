import numpy as np
import pytest

from blackjack_ai import Card, Shoe, ShoeEmptyError


def test_multi_deck_shoe_has_correct_size_and_composition() -> None:
    shoe = Shoe(8, shuffled=False)

    assert shoe.total_cards == 416
    assert len(shoe) == 416
    assert shoe.decks_remaining == 8.0
    np.testing.assert_array_equal(shoe.rank_counts(), np.full(13, 32))


def test_seeded_shoe_is_reproducible() -> None:
    first = Shoe(6, rng=8675309).peek_codes(312)
    second = Shoe(6, rng=8675309).peek_codes(312)

    np.testing.assert_array_equal(first, second)


def test_scalar_and_batch_draws_update_exact_composition() -> None:
    shoe = Shoe(1, shuffled=False)

    assert shoe.draw_code() == 0
    np.testing.assert_array_equal(shoe.draw_codes(3), [1, 2, 3])
    assert shoe.cards_dealt == 4
    assert shoe.cards_remaining == 48
    assert shoe.rank_counts()[0] == 3
    np.testing.assert_array_equal(shoe.rank_counts()[1:4], [3, 3, 3])


def test_human_facing_draw_returns_cards() -> None:
    shoe = Shoe(1, shuffled=False)

    assert shoe.draw() == Card.from_code(0)
    assert shoe.draw(2) == (Card.from_code(1), Card.from_code(2))


def test_draw_is_atomic_when_not_enough_cards_remain() -> None:
    shoe = Shoe(1, shuffled=False)
    shoe.draw_codes(51)

    with pytest.raises(ShoeEmptyError):
        shoe.draw_codes(2)
    assert shoe.cards_remaining == 1
    assert shoe.draw_code() == 51
    with pytest.raises(ShoeEmptyError):
        shoe.draw_code()


def test_read_only_zero_copy_view_cannot_corrupt_shoe() -> None:
    shoe = Shoe(1, shuffled=False)
    view = shoe.peek_codes(3, copy=False)

    assert not view.flags.writeable
    with pytest.raises(ValueError):
        view[0] = 12
    np.testing.assert_array_equal(shoe.peek_codes(3), [0, 1, 2])


def test_cut_card_and_reset() -> None:
    shoe = Shoe(1, shuffled=False, penetration=0.5)
    shoe.draw_codes(25)
    assert not shoe.cut_card_reached
    shoe.draw_code()
    assert shoe.cut_card_reached

    shoe.reset(shuffled=False)
    assert shoe.cards_dealt == 0
    assert not shoe.cut_card_reached
    np.testing.assert_array_equal(shoe.peek_codes(4), [0, 1, 2, 3])


def test_remaining_rank_probabilities_are_exact() -> None:
    shoe = Shoe(1, shuffled=False)
    probabilities = shoe.rank_probabilities(dtype=np.float64)
    np.testing.assert_allclose(probabilities, np.full(13, 1 / 13))

    shoe.draw_code()  # ace
    probabilities = shoe.rank_probabilities(dtype=np.float64)
    assert probabilities[0] == pytest.approx(3 / 51)
    assert probabilities[1] == pytest.approx(4 / 51)
    assert probabilities.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("num_decks", [0, -1, 1.5, True])
def test_invalid_deck_counts_are_rejected(num_decks: object) -> None:
    with pytest.raises(ValueError):
        Shoe(num_decks)  # type: ignore[arg-type]

