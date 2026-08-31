"""Money comes in denominations, because an enumerable action set is the point.

A continuous share is not a move. Snapping every price and every share to a grid
turns bargaining into a finite game: with L = 120 and a step of 10 there are
thirteen legal answers to "who pays what", which a table can hold and a scorer
can check. It also stops an agent from winning a comparison by proposing 61
instead of 60 — a distinction no counterparty in this game can act on.

The grid must divide the loss exactly, or the settlement identity L_B + L_S = L
stops closing and the world starts refusing rounds.
"""

from __future__ import annotations

STEP = 10


def snap(x: int, step: int = STEP, lo: int = 0, hi: int | None = None) -> int:
    """Nearest legal amount, clamped. Exact midpoints go up.

    Not `round`, which is banker's rounding: it sends 65 to 60 and 75 to 80, so
    whether a concession is rounded toward the buyer or the seller depends on
    which multiple of the step it happens to sit between. Half-up is arbitrary
    too, but it is the same arbitrary rule everywhere.
    """
    v = int(x // step + (1 if x % step * 2 >= step else 0)) * step
    if hi is not None:
        v = min(v, hi)
    return max(lo, v)


def shares(loss: int, step: int = STEP) -> list[int]:
    """Every legal seller share, low to high. The buyer's is the remainder."""
    if loss % step:
        raise ValueError(f"loss {loss} is not a whole number of {step}s")
    return list(range(0, loss + step, step))
