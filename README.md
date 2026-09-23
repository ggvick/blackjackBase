# Blackjack AI Core

A compact, NumPy-backed blackjack model intended for simulations and AI
training. Cards are stored as one-byte integer codes, so a six-deck shoe uses
312 bytes for card storage instead of hundreds of Python objects.

## Install and test

```bash
python -m pip install -e '.[test]'
pytest
python benchmarks/benchmark_creation.py
python benchmarks/benchmark_game.py
```

NumPy is the only runtime dependency. Pytest is only needed to run tests.

## Quick start

```python
import numpy as np

from blackjack_ai import BlackjackController, Hand, Shoe

# Reproducible six-deck shoe, shuffled on construction.
shoe = Shoe(num_decks=6, rng=42, penetration=0.75)
controller = BlackjackController(shoe)

# Use integer codes in training loops; this is the fastest API.
player = Hand(controller.draw_codes(2))
dealer_up_code = controller.draw_code()

print(player.total, player.is_soft, player.is_blackjack)
print(controller.running_count)      # Hi-Lo by default
print(controller.true_count)         # exact decks remaining, no rounding
print(controller.snapshot())

# 18 float32 values: 13 rank probabilities, running/true counts,
# fraction dealt, ten-value probability, and ace probability.
state = controller.observation()

# Avoid allocating a new array every step in a hot training loop.
observation_buffer = np.empty(18, dtype=np.float32)
state = controller.observation(out=observation_buffer)

# A bounded, lossless computer count: exact A..K depletion plus shoe size.
# This retains composition information that any scalar true count discards.
computer_count = controller.composition_count()
```

For display or external APIs, `shoe.draw()` returns a `Card` (or a tuple when
the requested count is greater than one). In performance-sensitive code,
`draw_code()` and `draw_codes()` avoid creating Python card objects.

## Complete game engine

`BlackjackGame` owns a persistent shoe, dealer, bankroll, visible-card tracker,
legal-action engine, split hands, settlement, and rewards. All table rules are
immutable constructor settings, which makes an experiment's ruleset easy to
record and reproduce.

```python
from blackjack_ai import (
    Action,
    BlackjackGame,
    BlackjackRules,
    DealerBlackjackLossRule,
    HoleCardRule,
    SurrenderRule,
)

rules = BlackjackRules(
    num_decks=6,
    penetration=0.75,
    dealer_hits_soft_17=False,
    blackjack_payout=1.5,
    max_split_hands=4,
    resplit_aces=False,
    hit_split_aces=False,
    double_after_split=True,
    double_allowed_totals=None,  # any first two cards
    surrender=SurrenderRule.LATE,
    insurance_allowed=True,
    hole_card_rule=HoleCardRule.AMERICAN,
    dealer_blackjack_loss_rule=DealerBlackjackLossRule.ALL_BETS,
)
game = BlackjackGame(rules, starting_balance=10_000, rng=42)
game.start_round(10, return_snapshot=False)

while game.settlement is None:
    mask = game.legal_action_mask()  # reusable bool array with 7 entries
    state = game.observation()       # reusable float32 array with 43 entries
    legal = game.legal_actions()
    game.act(legal[0], return_snapshot=False)

print(game.settlement.reward, game.cash.balance)
```

The stable action indices are `STAND`, `HIT`, `DOUBLE`, `SPLIT`, `SURRENDER`,
`INSURANCE`, and `DECLINE_INSURANCE`. Insurance actions are exposed only during
the insurance phase. Dealer play and settlement happen automatically after the
last player hand completes. `settlement.reward` is the exact change in bankroll
for the round, including split wagers, doubles, surrender, insurance, blackjack
payouts, Charlie rules, and optional original-bet-only protection in a European
no-hole-card game.

The 43-value game observation contains only information visible to the player:
visible rank depletion, active-hand facts, dealer up-card, and action context.
The hidden hole card is excluded until reveal. Pass `out=` buffers to
`observation()` and `legal_action_mask()`, and use `return_snapshot=False`, to
avoid allocations in rollout loops. Readable snapshots remain available for
debugging and user interfaces.

The principal tuning knobs include deck count, cut-card penetration, S17/H17,
American or European hole-card dealing, dealer peek, all-bets or
original-bet-only dealer-blackjack loss, 3:2/6:5-style blackjack payouts,
double restrictions, DAS, exact-rank versus equal-value splits, split limits,
ace resplits/hits, early/late/no surrender, insurance, table limits, and
optional Charlie rules. Invalid combinations and out-of-range settings fail at
construction time.

## Card encoding

Codes are `suit * 13 + rank - 1`, where ranks are ace through king and suits
are clubs, diamonds, hearts, and spades. Helpers `card_rank_indices()` and
`card_values()` work on entire NumPy arrays.

## Counting and composition

`BlackjackController` includes Hi-Lo, Omega II, and Wong Halves systems, and
accepts a custom `CountingSystem`. It computes its count from the exact dealt
and remaining rank arrays. The result stays correct even if code draws directly
from the attached `Shoe`.

### Computer composition count

`controller.composition_count()` is the recommended count input for a neural
network. Its fixed 14-value layout is:

- values 0–12: fraction of each rank dealt, ordered A, 2, ..., K;
- value 13: starting deck count encoded as `decks / (decks + 1)`.

Every value is finite and bounded in `[0, 1]`, including an exhausted shoe. The
mean of the first 13 values is penetration. This vector is lossless with respect
to rank composition: it can reconstruct Hi-Lo, Omega II, Wong Halves, or any
other linear count, but it does not collapse different rank compositions to the
same scalar as true count does. An `out=` buffer is supported for allocation-free
training loops.

The controller also exposes information a computer can use beyond a human
running count:

- exact remaining counts and probabilities for all 13 ranks;
- cards/decks remaining, penetration, and cut-card state;
- exact ace, ten-value, and high-card composition;
- a ready-to-consume `float32` observation vector.

For allocation-stable training loops, allocate one `(18,)` `float32` array per
environment and pass it as `observation(out=buffer)`. The allocating form has
similar raw throughput and is convenient when each returned state must be kept.

`true_count` raises `ShoeEmptyError` when the shoe is empty because a true count
cannot be defined with zero decks remaining. The AI observation substitutes
zero in that terminal state so it always contains finite values.

## Performance notes

- `Deck()` shares an immutable 52-byte template.
- A `Shoe` is built with `numpy.tile` and shuffled in-place with
  `numpy.random.Generator.shuffle`; integer seeds use the fast SFC64 bit
  generator, while caller-provided NumPy generators are honored unchanged.
- Scalar draws are O(1); batch composition updates use `numpy.bincount`.
- Controller draws update a cached running count in O(1). A monotonic shoe
  revision detects direct shoe mutations and triggers an exact resynchronization,
  so the optimization does not sacrifice count correctness.
- `Card` objects are deliberately created only at the readable API boundary.
- Pass `copy=False` only for short-lived, read-only views when eliminating a
  small defensive copy matters. Such views may change after a shoe reset.

Microbenchmarks vary by hardware and should be treated as comparative, not as
hard performance guarantees. The benchmark script prints median time and
throughput for each path.

## Rules references

The configurable rules are based on published casino/regulatory rules rather
than one assumed universal table: Nevada's GameAce rules document configurable
deck counts, S17/H17, doubling, splitting, surrender, blackjack payouts, and
Charlie variants; Colorado's blackjack regulations cover splitting, doubling,
insurance/even money, surrender, and original-bet handling. Casino rules vary,
so record the complete `BlackjackRules` value with every experiment.

- [Nevada Gaming Control Board GameAce Live Blackjack rules](https://www.gaming.nv.gov/siteassets/content/divisions/technology/rules-of-play/gameace-live-blackjack.pdf)
- [Colorado blackjack regulations](https://www.law.cornell.edu/regulations/colorado/1-CCR-207-1-8)
