"""End-to-end game-engine benchmark.

Run from the repository root with:
    python benchmarks/benchmark_game.py
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from blackjack_ai import (
    Action,
    BlackjackGame,
    BlackjackRules,
    Cash,
    SurrenderRule,
)


def run_policy(
    rounds: int,
    *,
    random_actions: bool,
    record_cash_history: bool = False,
) -> tuple[float, int]:
    game = BlackjackGame(
        BlackjackRules(surrender=SurrenderRule.NONE),
        cash=Cash(
            10_000_000,
            record_history=record_cash_history,
            history_capacity=rounds * 4 if record_cash_history else 0,
        ),
        rng=12345,
    )
    rng = np.random.default_rng(999)
    observation = np.empty(43, dtype=np.float32)
    mask = np.empty(7, dtype=np.bool_)
    decisions = 0
    started = perf_counter()
    for _ in range(rounds):
        game.start_round(1, return_snapshot=False)
        while not game.is_round_over:
            actions = game.legal_actions()
            if random_actions:
                action = actions[int(rng.integers(len(actions)))]
            else:
                action = Action.STAND if Action.STAND in actions else actions[0]
            game.observation(out=observation)
            game.legal_action_mask(out=mask)
            game.act(action, return_snapshot=False)
            decisions += 1
    return perf_counter() - started, decisions


def report(
    label: str,
    rounds: int,
    *,
    random_actions: bool,
    record_cash_history: bool = False,
) -> None:
    elapsed, decisions = run_policy(
        rounds,
        random_actions=random_actions,
        record_cash_history=record_cash_history,
    )
    print(
        f"{label:30} {rounds / elapsed:12,.0f} rounds/s  "
        f"{decisions / elapsed:12,.0f} decisions/s"
    )


def main() -> None:
    report("stand policy", 50_000, random_actions=False)
    report("random legal policy", 20_000, random_actions=True)
    report(
        "random legal + cash history",
        20_000,
        random_actions=True,
        record_cash_history=True,
    )


if __name__ == "__main__":
    main()
