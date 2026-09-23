import numpy as np
import pytest

from blackjack_ai import CASH_HISTORY_DTYPE, Cash, CashTransactionType


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


def test_cash_accepts_numpy_numeric_scalars_but_not_numpy_booleans() -> None:
    cash = Cash(np.float32(100))

    cash.debit(np.int64(10))
    cash.credit(np.float64(2.5))

    assert cash.balance == 92.5
    with pytest.raises(TypeError):
        cash.credit(np.bool_(True))


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


def test_opt_in_history_records_every_successful_debit_and_credit() -> None:
    cash = Cash(100, record_history=True, history_capacity=1)

    cash.debit(25)
    cash.credit(10.5)
    cash.debit(5)

    assert cash.transaction_count == 3
    assert cash.history_capacity >= 3
    assert cash.latest_transaction is not None
    assert cash.latest_transaction.type is CashTransactionType.DEBIT
    assert cash.latest_transaction.balance == 80.5
    records = cash.transactions()
    assert [record.sequence for record in records] == [1, 2, 3]
    assert [record.type for record in records] == [
        CashTransactionType.DEBIT,
        CashTransactionType.CREDIT,
        CashTransactionType.DEBIT,
    ]
    assert [record.amount for record in records] == [25, 10.5, 5]
    assert [record.balance for record in records] == [75, 85.5, 80.5]


def test_numpy_history_api_is_compact_safe_and_can_avoid_a_copy() -> None:
    cash = Cash(100, record_history=True)
    cash.debit(20)
    cash.credit(5)

    copied = cash.history()
    view = cash.history(copy=False)

    assert copied.dtype == CASH_HISTORY_DTYPE
    assert copied.dtype.itemsize <= 32
    np.testing.assert_array_equal(copied["sequence"], [1, 2])
    np.testing.assert_array_equal(
        copied["type"],
        [CashTransactionType.DEBIT, CashTransactionType.CREDIT],
    )
    np.testing.assert_array_equal(copied["amount"], [20, 5])
    np.testing.assert_array_equal(copied["balance"], [80, 85])
    assert copied.flags.writeable
    assert not view.flags.writeable
    with pytest.raises(ValueError):
        view["amount"][0] = 999


def test_disabled_history_has_no_storage_and_can_be_enabled_later() -> None:
    cash = Cash(100)
    cash.debit(10)

    assert not cash.history_enabled
    assert cash.history_capacity == 0
    assert cash.transaction_count == 0
    assert cash.transactions() == ()
    assert cash.history(copy=False).shape == (0,)

    cash.set_history_enabled(True, history_capacity=2)
    cash.credit(5)
    cash.set_history_enabled(False)
    cash.debit(1)
    cash.set_history_enabled(True, history_capacity=10)
    cash.credit(2)

    assert [record.amount for record in cash.transactions()] == [5, 2]
    assert [record.sequence for record in cash.transactions()] == [1, 2]
    assert cash.history_capacity == 10
    assert cash.balance == 96


def test_failed_transactions_are_not_recorded_and_history_can_be_cleared() -> None:
    cash = Cash(10, record_history=True)

    with pytest.raises(ValueError):
        cash.debit(11)
    assert cash.transaction_count == 0

    cash.credit(1)
    cash.clear_history(release_memory=True)

    assert cash.transaction_count == 0
    assert cash.history_capacity == 0
    assert cash.latest_transaction is None
    assert cash.history_enabled
    cash.debit(1)
    assert cash.transactions()[0].sequence == 1


def test_reset_clears_history_by_default_or_can_preserve_it() -> None:
    cash = Cash(100, record_history=True)
    cash.debit(10)

    cash.reset(clear_history=False)
    assert cash.transaction_count == 1
    cash.credit(5)
    assert cash.transactions()[-1].sequence == 2

    cash.reset(250)
    assert cash.transaction_count == 0
    assert cash.latest_transaction is None


def test_preserved_history_records_an_explicit_balance_reset() -> None:
    cash = Cash(100, record_history=True)
    cash.debit(10)

    cash.reset(250, clear_history=False)

    reset = cash.latest_transaction
    assert reset is not None
    assert reset.sequence == 2
    assert reset.type is CashTransactionType.RESET
    assert reset.amount == 250
    assert reset.balance == 250


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"record_history": 1}, TypeError),
        ({"history_capacity": -1}, ValueError),
        ({"history_capacity": True}, ValueError),
    ],
)
def test_history_configuration_is_validated(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        Cash(**kwargs)  # type: ignore[arg-type]
