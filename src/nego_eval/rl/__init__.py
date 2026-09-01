"""Training-side wiring: the board as a reinforcement-learning environment.

Kept apart from `game/` on purpose. Everything under `game/` and `sim/` runs with
no dependency beyond httpx, and the tests and every published measurement rely on
that. This package is the only place that knows about `verifiers`, and it imports
it lazily, so a missing trainer cannot break scoring.
"""
