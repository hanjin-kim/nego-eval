"""Staying does not merely reveal the counterparty. It changes it.

Board three satisfied every condition on the checklist and still measured the
wrong thing. Getting the published delivery rate to be exact meant switching off
the channel by which loyalty improved delivery; restoring room to bargain left
the loyalty term in the concession ceiling no longer binding; and the solvency
spiral had already gone. Between them, nothing remained by which returning to a
seller made that seller better. What was left was a fixed hidden attribute and a
buyer discovering it — so carrying the ledger across a boundary saved time and
almost nothing else, and measured at four percent.

That is not what the literature describes. A relational buyer pays a premium and
the rent buys effort no contract can specify; the relationship is productive, not
merely informative. So here the hidden trait is not a constant:

    published, exact, priced      delivery rate for a buyer with no history
    hidden, learned by trading    how much of a loss it absorbs at arm's length
    hidden, learned by staying    how much further it will go for a regular

The last is the one board three lost. It is still countable — a share of a loss,
in whole units of ten — so nothing has to be judged and the reward stays
verifiable. And it is still only visible after a failure, which is what keeps the
record worth carrying.
"""

from __future__ import annotations

import random

from nego_eval.sim.agents import ScriptedSeller

NAMES = ("A", "B", "C")
VALUE = 150
BASE = 86
PREMIUM = 120
COST_MARGIN = 70
OPENING = 0.4               # opens at this fraction of what it would concede

RATES = (0.60, 0.70, 0.80, 0.90)
FLOORS = (0.00, 0.20, 0.35, 0.50)       # what a stranger gets
CARE = (0.00, 0.20, 0.40)               # what a regular gets on top, at full loyalty


def price_of(rate: float) -> int:
    return int(round(BASE + PREMIUM * (rate - RATES[0])))


def expected(rate: float, floor: float, loss: int) -> float:
    return rate * (VALUE - price_of(rate)) - (1 - rate) * loss * (1 - floor)


def cast_for(seed: int, loss: int = 110):
    """One match's sellers, and the name a stranger should open with.

    The key is deliberately the arm's-length ranking: at the first move nobody
    has a relationship, so that is the right choice on the evidence. Whether a
    seller would repay staying is not knowable then, and an agent graded on it
    would be graded on a coin flip.
    """
    rng = random.Random(seed)
    for _ in range(400):
        picks = [(rng.choice(RATES), rng.choice(FLOORS), rng.choice(CARE))
                 for _ in NAMES]
        if len({p[0] for p in picks}) < len(NAMES):
            continue
        ev = {NAMES[i]: expected(picks[i][0], picks[i][1], loss) for i in range(len(NAMES))}
        top = sorted(ev.values(), reverse=True)
        if top[0] - top[1] >= 3.0:
            break

    def make(_game: int | None = None):
        return [ScriptedSeller(NAMES[i], price_of(picks[i][0]),
                               reliability=picks[i][0],
                               care_bonus=0.0, health_couples=False,
                               cost=price_of(picks[i][0]) - COST_MARGIN,
                               share=picks[i][1] * OPENING,
                               floor=picks[i][1], floor_care=picks[i][2],
                               seed=seed)
                for i in range(len(NAMES))]

    board = {NAMES[i]: dict(rate=picks[i][0], price=price_of(picks[i][0]))
             for i in range(len(NAMES))}
    hidden = {NAMES[i]: dict(floor=picks[i][1], care=picks[i][2]) for i in range(len(NAMES))}
    return make, max(ev, key=ev.get), ev, board, hidden
