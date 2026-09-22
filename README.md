# Blackjack AI Core

A compact, NumPy-backed blackjack model intended for simulations and AI
training. Cards are stored as one-byte integer codes, so a six-deck shoe uses
312 bytes for card storage instead of hundreds of Python objects.

## Install and test

```bash
python -m pip install -e '.[test]'
pytest
python benchmarks/benchmark_creation.py
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
```

For display or external APIs, `shoe.draw()` returns a `Card` (or a tuple when
the requested count is greater than one). In performance-sensitive code,
`draw_code()` and `draw_codes()` avoid creating Python card objects.

## Card encoding

Codes are `suit * 13 + rank - 1`, where ranks are ace through king and suits
are clubs, diamonds, hearts, and spades. Helpers `card_rank_indices()` and
`card_values()` work on entire NumPy arrays.

## Counting and composition

`BlackjackController` includes Hi-Lo, Omega II, and Wong Halves systems, and
accepts a custom `CountingSystem`. It computes its count from the exact dealt
and remaining rank arrays. The result stays correct even if code draws directly
from the attached `Shoe`.

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
  `numpy.random.Generator.shuffle`.
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
