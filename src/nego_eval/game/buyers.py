"""Buyers for the board where reliability is quoted and conduct is not.

`GrudgeBuyer` scored a seller on one thing — the share of losses it had absorbed
— which was the whole answer on the first board and is 0.64 of it here. A policy
that reads only that is no longer the ceiling, so the reference has to do the
arithmetic the board now requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nego_eval.sim.world import Outcome, Quote


def borne(history: list[Outcome], name: str, default: float = 0.0) -> float:
    paid = owed = 0
    for o in history:
        if o.seller == name and o.failed:
            paid += o.seller_share
            owed += o.loss
    return paid / owed if owed else default


@dataclass
class EVBuyer:
    """Buys whichever quote maximises expected profit given what it has seen.

    Reliability comes off the board and the share of a loss the seller will take
    comes off the record, so this is the arithmetic the environment rewards, and
    it needs both. Sellers never traded with are credited with `unseen`, which is
    the one free parameter: at 0 an unfamiliar seller is assumed to concede
    nothing and the policy never explores, at 1 it assumes the best and churns.
    """

    value: int = 150
    loss: int = 110
    unseen: float = 0.30
    ask: float = 1.0            # presses for the whole loss, so the floor binds

    def choose(self, quotes: list[Quote], t: int, remaining: int,
               history: list[Outcome]) -> str:
        def ev(q: Quote) -> float:
            r = q.rate if q.rate is not None else 0.70
            b = borne(history, q.seller, default=self.unseen)
            return r * (self.value - q.price) - (1 - r) * self.loss * (1 - b)
        return max(quotes, key=ev).seller

    def bargain(self, loss, offer, r, max_rounds, seller_name, history, terms):
        want = int(loss * self.ask)
        if offer >= want or r == max_rounds:
            return True, offer
        return False, want


@dataclass
class BoardOnlyBuyer:
    """Reads the quote board and never the record. The null for this board."""

    value: int = 150
    loss: int = 110

    def choose(self, quotes, t, remaining, history):
        def ev(q):
            r = q.rate if q.rate is not None else 0.70
            return r * (self.value - q.price) - (1 - r) * self.loss
        return max(quotes, key=ev).seller

    def bargain(self, loss, offer, r, max_rounds, seller_name, history, terms):
        if offer >= loss or r == max_rounds:
            return True, offer
        return False, loss
