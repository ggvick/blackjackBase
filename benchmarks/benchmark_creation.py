"""Dependency-free microbenchmark for the main construction and hot paths.

Run from the repository root with:
    python benchmarks/benchmark_creation.py
"""

from __future__ import annotations

import gc
import statistics
import timeit
from typing import Any

from blackjack_ai import BlackjackController, Deck, Shoe


def benchmark(
    label: str,
    statement: str,
    *,
    number: int,
    repeat: int = 7,
    context: dict[str, Any] | None = None,
) -> None:
    namespace = dict(globals())
    namespace.update(context or {})
    timer = timeit.Timer(statement, globals=namespace)
    gc.disable()
    try:
        samples = timer.repeat(repeat=repeat, number=number)
    finally:
        gc.enable()
    per_operation = [sample / number for sample in samples]
    median = statistics.median(per_operation)
    print(f"{label:32} {median * 1e6:10.3f} µs/op  ({1 / median:,.0f} ops/s)")


def main() -> None:
    print("Median of 7 runs (lower is better)")
    benchmark("Deck()", "Deck()", number=100_000)
    benchmark("Shoe(6), ordered", "Shoe(6, shuffled=False)", number=10_000)
    benchmark("Shoe(6), shuffled", "Shoe(6)", number=5_000)
    benchmark(
        "Controller(6), shuffled", "BlackjackController()", number=5_000
    )

    shoe = Shoe(8, shuffled=False)
    benchmark(
        "draw_code + periodic reset",
        "shoe.draw_code() if len(shoe) else shoe.reset(shuffled=False)",
        number=500_000,
        context={"shoe": shoe},
    )

    controller = BlackjackController(shuffled=False)
    benchmark(
        "AI observation",
        "controller.observation()",
        number=20_000,
        context={"controller": controller},
    )


if __name__ == "__main__":
    main()
