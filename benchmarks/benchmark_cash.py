"""Cash transaction-history throughput benchmark.

Run from the repository root with:
    python benchmarks/benchmark_cash.py
"""

from __future__ import annotations

from statistics import median
from time import perf_counter

from blackjack_ai import Cash


def run_transactions(
    transaction_pairs: int,
    *,
    record_history: bool,
    preallocate: bool,
) -> tuple[float, Cash]:
    transaction_count = transaction_pairs * 2
    cash = Cash(
        transaction_pairs + 1,
        record_history=record_history,
        history_capacity=transaction_count if preallocate else 64,
    )
    started = perf_counter()
    for _ in range(transaction_pairs):
        cash.debit(1)
        cash.credit(1)
    return perf_counter() - started, cash


def report(
    label: str, *, record_history: bool, preallocate: bool = False
) -> None:
    transaction_pairs = 250_000
    runs = [
        run_transactions(
            transaction_pairs,
            record_history=record_history,
            preallocate=preallocate,
        )
        for _ in range(5)
    ]
    elapsed = median(run[0] for run in runs)
    cash = runs[-1][1]
    transactions = transaction_pairs * 2
    memory = cash.history(copy=False).nbytes
    print(
        f"{label:30} {transactions / elapsed:12,.0f} transactions/s  "
        f"{memory / (1024 * 1024):8.2f} MiB history"
    )


def main() -> None:
    report("history disabled", record_history=False)
    report("history enabled/growing", record_history=True)
    report(
        "history enabled/preallocated",
        record_history=True,
        preallocate=True,
    )


if __name__ == "__main__":
    main()
