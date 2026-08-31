"""Board-game layer: a finite action set, and memory that outlives a game.

`denom` is a leaf and `world` depends on it, so nothing heavier may be re-exported
here — importing `Match` at package level would make `world` import `match` import
`world`. Take it from `nego_eval.game.match` directly.
"""
from nego_eval.game.denom import STEP, shares, snap

__all__ = ["STEP", "shares", "snap"]
