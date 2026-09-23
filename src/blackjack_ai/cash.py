"""Fast bankroll accounting for blackjack simulations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

import numpy as np
from numpy.typing import NDArray


class CashTransactionType(IntEnum):
    """Compact, stable transaction identifiers."""

    DEBIT = 0
    CREDIT = 1


CASH_HISTORY_DTYPE = np.dtype(
    [
        ("sequence", np.uint64),
        ("type", np.int8),
        ("amount", np.float64),
        ("balance", np.float64),
    ]
)
"""Structured NumPy dtype returned by :meth:`Cash.history`."""

_EMPTY_HISTORY = np.empty(0, dtype=CASH_HISTORY_DTYPE)
_EMPTY_HISTORY.flags.writeable = False


@dataclass(frozen=True, slots=True)
class CashTransaction:
    """Readable representation of one successful balance mutation."""

    sequence: int
    type: CashTransactionType
    amount: float
    balance: float


@dataclass(frozen=True, slots=True)
class CashSnapshot:
    """Immutable account totals at a point in time."""

    balance: float
    initial_balance: float
    net_profit: float
    total_credited: float
    total_debited: float


class Cash:
    """A lightweight bankroll with deterministic rounded arithmetic.

    Amounts are stored as floats for fast reward generation and rounded after
    each transaction. ``precision`` can match the smallest supported chip or
    currency denomination. This is accounting for simulation, not a financial
    ledger; applications handling real funds should use integer minor units.
    """

    __slots__ = (
        "_balance",
        "_initial_balance",
        "_total_credited",
        "_total_debited",
        "_history",
        "_history_size",
        "_history_enabled",
        "_history_capacity_hint",
        "_next_sequence",
        "precision",
    )

    def __init__(
        self,
        balance: float = 1_000.0,
        *,
        precision: int = 8,
        record_history: bool = False,
        history_capacity: int = 64,
    ) -> None:
        if not isinstance(precision, int) or isinstance(precision, bool):
            raise TypeError("precision must be an integer")
        if not 0 <= precision <= 12:
            raise ValueError("precision must be between 0 and 12")
        if not isinstance(record_history, bool):
            raise TypeError("record_history must be a bool")
        self._validate_history_capacity(history_capacity)
        value = self._validate_amount(balance, allow_zero=True, name="balance")
        self.precision = precision
        self._balance = round(value, precision)
        self._initial_balance = self._balance
        self._total_credited = 0.0
        self._total_debited = 0.0
        self._history_capacity_hint = history_capacity
        self._history = (
            np.empty(history_capacity, dtype=CASH_HISTORY_DTYPE)
            if record_history
            else None
        )
        self._history_size = 0
        self._history_enabled = record_history
        self._next_sequence = 1

    @staticmethod
    def _validate_history_capacity(capacity: int) -> None:
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or capacity < 0
        ):
            raise ValueError("history_capacity must be a non-negative integer")

    @staticmethod
    def _validate_amount(
        amount: float, *, allow_zero: bool, name: str = "amount"
    ) -> float:
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise TypeError(f"{name} must be a number")
        value = float(amount)
        if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be a finite {qualifier} number")
        return value

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def initial_balance(self) -> float:
        return self._initial_balance

    @property
    def net_profit(self) -> float:
        return round(self._balance - self._initial_balance, self.precision)

    @property
    def total_credited(self) -> float:
        return self._total_credited

    @property
    def total_debited(self) -> float:
        return self._total_debited

    @property
    def history_enabled(self) -> bool:
        """Whether successful credits and debits are currently recorded."""

        return self._history_enabled

    @property
    def transaction_count(self) -> int:
        return self._history_size

    @property
    def history_capacity(self) -> int:
        return 0 if self._history is None else len(self._history)

    def _record(self, transaction_type: CashTransactionType, amount: float) -> None:
        if not self._history_enabled:
            return
        history = self._history
        if history is None:
            history = np.empty(
                self._history_capacity_hint, dtype=CASH_HISTORY_DTYPE
            )
            self._history = history
        if self._history_size == len(history):
            capacity = max(8, len(history) * 2)
            expanded = np.empty(capacity, dtype=CASH_HISTORY_DTYPE)
            expanded[: self._history_size] = history
            history = expanded
            self._history = history
        history[self._history_size] = (
            self._next_sequence,
            int(transaction_type),
            amount,
            self._balance,
        )
        self._history_size += 1
        self._next_sequence += 1

    def set_history_enabled(
        self,
        enabled: bool,
        *,
        clear: bool = False,
        history_capacity: int | None = None,
    ) -> None:
        """Enable or pause recording without changing the bankroll.

        Existing entries are retained when recording is paused. Pass
        ``clear=True`` to begin a fresh record. Capacity grows geometrically,
        so it is useful only as a preallocation hint for long simulations.
        """

        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        if history_capacity is not None:
            self._validate_history_capacity(history_capacity)
            self._history_capacity_hint = history_capacity
            if (
                self._history is not None
                and history_capacity > len(self._history)
            ):
                expanded = np.empty(history_capacity, dtype=CASH_HISTORY_DTYPE)
                expanded[: self._history_size] = self._history[
                    : self._history_size
                ]
                self._history = expanded
        if clear:
            self.clear_history()
        if enabled and self._history is None:
            self._history = np.empty(
                self._history_capacity_hint, dtype=CASH_HISTORY_DTYPE
            )
        self._history_enabled = enabled

    def clear_history(self, *, release_memory: bool = False) -> None:
        """Remove every transaction while leaving the balance unchanged."""

        self._history_size = 0
        self._next_sequence = 1
        if release_memory:
            self._history = None

    def history(
        self, *, copy: bool = True
    ) -> NDArray[np.void]:
        """Return transactions as a structured NumPy array.

        Fields are ``sequence``, ``type``, ``amount``, and post-transaction
        ``balance``. The default copy is safe to retain. ``copy=False`` returns
        a read-only, allocation-light view for immediate analytics.
        """

        if self._history is None or self._history_size == 0:
            return _EMPTY_HISTORY.copy() if copy else _EMPTY_HISTORY
        records = self._history[: self._history_size]
        if copy:
            return records.copy()
        view = records.view()
        view.flags.writeable = False
        return view

    def transactions(self) -> tuple[CashTransaction, ...]:
        """Return an immutable, readable transaction sequence."""

        if self._history is None:
            return ()
        return tuple(
            CashTransaction(
                sequence=int(record["sequence"]),
                type=CashTransactionType(int(record["type"])),
                amount=float(record["amount"]),
                balance=float(record["balance"]),
            )
            for record in self._history[: self._history_size]
        )

    @property
    def latest_transaction(self) -> CashTransaction | None:
        """Return the most recent transaction, if one has been recorded."""

        if self._history is None or self._history_size == 0:
            return None
        record = self._history[self._history_size - 1]
        return CashTransaction(
            sequence=int(record["sequence"]),
            type=CashTransactionType(int(record["type"])),
            amount=float(record["amount"]),
            balance=float(record["balance"]),
        )

    def can_afford(self, amount: float) -> bool:
        value = self._validate_amount(amount, allow_zero=True)
        return self._balance >= value

    def debit(self, amount: float) -> float:
        """Remove funds and return the new balance."""

        value = self._validate_amount(amount, allow_zero=False)
        if not self.can_afford(value):
            raise ValueError(
                f"insufficient balance: need {value:g}, have {self._balance:g}"
            )
        self._balance = round(self._balance - value, self.precision)
        self._total_debited = round(
            self._total_debited + value, self.precision
        )
        self._record(CashTransactionType.DEBIT, value)
        return self._balance

    def credit(self, amount: float) -> float:
        """Add funds and return the new balance."""

        value = self._validate_amount(amount, allow_zero=True)
        self._balance = round(self._balance + value, self.precision)
        self._total_credited = round(
            self._total_credited + value, self.precision
        )
        self._record(CashTransactionType.CREDIT, value)
        return self._balance

    def reset(
        self, balance: float | None = None, *, clear_history: bool = True
    ) -> None:
        """Reset totals, optionally with a new starting balance.

        History is cleared by default because reset begins a new accounting
        period. Pass ``clear_history=False`` to retain the earlier record.
        """

        value = self._balance if balance is None else self._validate_amount(
            balance, allow_zero=True, name="balance"
        )
        self._balance = round(value, self.precision)
        self._initial_balance = self._balance
        self._total_credited = 0.0
        self._total_debited = 0.0
        if clear_history:
            self.clear_history()

    def snapshot(self) -> CashSnapshot:
        return CashSnapshot(
            balance=self._balance,
            initial_balance=self._initial_balance,
            net_profit=self.net_profit,
            total_credited=self._total_credited,
            total_debited=self._total_debited,
        )
