"""End-to-end game-engine benchmark.

Run from the repository root with:
    python benchmarks/benchmark_game.py
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from blackjack_ai import Action, BlackjackGame, BlackjackRules, SurrenderRule


def run_policy(rounds: int, *, random_actions: bool) -> tuple[float, int]:
    game = BlackjackGame(
        BlackjackRules(surrender=SurrenderRule.NONE),
        starting_balance=10_000_000,
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


def report(label: str, rounds: int, *, random_actions: bool) -> None:
    elapsed, decisions = run_policy(rounds, random_actions=random_actions)
    print(
        f"{label:30} {rounds / elapsed:12,.0f} rounds/s  "
        f"{decisions / elapsed:12,.0f} decisions/s"
    )


def main() -> None:
    report("stand policy", 50_000, random_actions=False)
    report("random legal policy", 20_000, random_actions=True)


if __name__ == "__main__":
    main()
