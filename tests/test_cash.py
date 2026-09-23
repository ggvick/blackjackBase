import pytest

from blackjack_ai import Cash


def test_cash_tracks_balance_profit_and_turnover() -> None:
    cash = Cash(100.0)

    assert cash.can_afford(100)
    assert not cash.can_afford(101)
    assert cash.debit(25) == 75
    assert cash.credit(40.5) == 115.5

    snapshot = cash.snapshot()
    assert snapshot.balance == 115.5
    assert snapshot.initial_balance == 100
    assert snapshot.net_profit == 15.5
    assert snapshot.total_debited == 25
    assert snapshot.total_credited == 40.5


def test_cash_rejects_invalid_or_unaffordable_transactions() -> None:
    cash = Cash(10)

    with pytest.raises(ValueError, match="insufficient"):
        cash.debit(11)
    with pytest.raises(ValueError):
        cash.credit(float("nan"))
    with pytest.raises(ValueError):
        cash.debit(0)
    with pytest.raises(TypeError):
        cash.credit(True)
    assert cash.balance == 10


def test_cash_never_uses_rounding_tolerance_to_allow_an_overdraft() -> None:
    cash = Cash(10, precision=0)

    assert not cash.can_afford(10.01)
    with pytest.raises(ValueError, match="insufficient"):
        cash.debit(10.01)
    assert cash.balance == 10


def test_cash_reset_starts_new_accounting_period() -> None:
    cash = Cash(100)
    cash.debit(10)
    cash.credit(15)

    cash.reset(250)

    assert cash.balance == 250
    assert cash.initial_balance == 250
    assert cash.net_profit == 0
    assert cash.total_credited == 0
    assert cash.total_debited == 0
