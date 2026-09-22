import pytest

from blackjack_ai import Card, Hand, Rank, Suit


@pytest.mark.parametrize(
    ("codes", "total", "soft"),
    [
        ((0, 9), 21, True),  # A + 10
        ((0, 13, 8), 21, True),  # A + A + 9
        ((0, 13, 8, 12), 21, False),  # A + A + 9 + K
        ((9, 22, 35), 30, False),  # three tens
        ((), 0, False),
    ],
)
def test_hand_totals(codes: tuple[int, ...], total: int, soft: bool) -> None:
    hand = Hand(codes)
    assert hand.total == total
    assert hand.is_soft is soft
    assert hand.is_bust is (total > 21)


def test_blackjack_and_split() -> None:
    blackjack = Hand((0, 12))
    pair = Hand((7, 20))  # eights of different suits

    assert blackjack.is_blackjack
    assert not pair.is_blackjack
    assert pair.can_split


def test_hand_accepts_card_objects_and_can_be_cleared() -> None:
    hand = Hand()
    hand.add(Card(Rank.ACE, Suit.SPADES))
    hand.add(Card(Rank.KING, Suit.CLUBS))

    assert hand.total == 21
    assert hand.is_blackjack
    hand.clear()
    assert len(hand) == 0
    assert hand.total == 0

