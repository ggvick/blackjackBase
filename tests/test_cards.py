import numpy as np
import pytest

from blackjack_ai import Card, Deck, Rank, Suit, card_rank_indices, card_values


def test_every_card_round_trips_through_compact_code() -> None:
    cards = [Card.from_code(code) for code in range(52)]

    assert [card.code for card in cards] == list(range(52))
    assert len(set(cards)) == 52
    assert str(Card(Rank.ACE, Suit.SPADES)) == "A♠"
    assert Card(Rank.KING, Suit.HEARTS).value == 10


@pytest.mark.parametrize("bad_code", [-1, 52, 1.5, True])
def test_invalid_card_codes_are_rejected(bad_code: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        Card.from_code(bad_code)  # type: ignore[arg-type]


def test_deck_has_each_card_and_rank_once_per_suit() -> None:
    deck = Deck()
    codes = deck.codes()

    assert len(deck) == 52
    assert codes.dtype == np.uint8
    np.testing.assert_array_equal(codes, np.arange(52, dtype=np.uint8))
    np.testing.assert_array_equal(
        np.bincount(card_rank_indices(codes), minlength=13), np.full(13, 4)
    )


def test_unshuffled_deck_storage_is_read_only_and_copy_is_safe() -> None:
    deck = Deck()
    shared = deck.codes(copy=False)
    copied = deck.codes()

    assert not shared.flags.writeable
    with pytest.raises(ValueError):
        shared[0] = 51
    copied[0] = 51
    assert deck.codes(copy=False)[0] == 0


def test_seeded_shuffled_decks_are_reproducible() -> None:
    first = Deck(shuffled=True, rng=123).codes(copy=False)
    second = Deck(shuffled=True, rng=123).codes(copy=False)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, np.arange(52, dtype=np.uint8))


def test_vectorized_card_values() -> None:
    # Ace, nine, ten, jack, queen, king of clubs.
    codes = np.array([0, 8, 9, 10, 11, 12], dtype=np.uint8)
    np.testing.assert_array_equal(card_values(codes), [1, 9, 10, 10, 10, 10])

