"""Fast bankroll accounting for blackjack simulations."""

from __future__ import annotations

from dataclasses import dataclass
import math


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
        "precision",
    )

    def __init__(self, balance: float = 1_000.0, *, precision: int = 8) -> None:
        if not isinstance(precision, int) or isinstance(precision, bool):
            raise TypeError("precision must be an integer")
        if not 0 <= precision <= 12:
            raise ValueError("precision must be between 0 and 12")
        value = self._validate_amount(balance, allow_zero=True, name="balance")
        self.precision = precision
        self._balance = round(value, precision)
        self._initial_balance = self._balance
        self._total_credited = 0.0
        self._total_debited = 0.0

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
        return self._balance

    def credit(self, amount: float) -> float:
        """Add funds and return the new balance."""

        value = self._validate_amount(amount, allow_zero=True)
        self._balance = round(self._balance + value, self.precision)
        self._total_credited = round(
            self._total_credited + value, self.precision
        )
        return self._balance

    def reset(self, balance: float | None = None) -> None:
        """Reset totals, optionally with a new starting balance."""

        value = self._balance if balance is None else self._validate_amount(
            balance, allow_zero=True, name="balance"
        )
        self._balance = round(value, self.precision)
        self._initial_balance = self._balance
        self._total_credited = 0.0
        self._total_debited = 0.0

    def snapshot(self) -> CashSnapshot:
        return CashSnapshot(
            balance=self._balance,
            initial_balance=self._initial_balance,
            net_profit=self.net_profit,
            total_credited=self._total_credited,
            total_debited=self._total_debited,
        )
