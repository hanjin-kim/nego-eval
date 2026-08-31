"""Quality is published and priced. Conduct after a failure is not.

Both earlier boards put the same thing in the wrong place. The first hid nothing
except loss-sharing and made delivery records meaningless, so an ordinary rule
was punished. The second hid delivery reliability too, and a rate estimated from
five trades is not recoverable, so the question had no answer.

Here the split follows the one real markets make.

    published, exact, priced     the seller's delivery rate, on the quote board
    hidden, learned by failing    how much of a loss it absorbs

A supplier's on-time rate is in the catalogue and you pay for it. What it does
when a shipment is ruined is not in the catalogue, is not contractible, and is
learned only by having it happen. That is the asymmetry relational contracting
is about, and it is the one this environment can measure without a judge.

Recoverability comes from the concession being deterministic once the seller can
afford it: pressed for the whole loss, a seller concedes exactly its floor, so a
single failure reveals the trait outright. It still takes a failure — a rare
event — which is what makes the carried ledger worth having rather than merely
present.
"""

from __future__ import annotations

import random

from carryover.sim.agents import ScriptedSeller

NAMES = ("A", "B", "C")
VALUE = 150
BASE = 86               # what the least reliable seller charges
PREMIUM = 120           # price added per unit of published delivery rate, chosen
                        # so that neither reading of the board dominates: buying
                        # the highest published rate is right 72% of the time and
                        # buying the biggest loss-absorber 68%, and the ceiling
                        # needs both
#: Wide enough that no type bleeds to death. The solvency spiral is a real part
#: of procurement and it is switched off here on purpose: a seller running at a
#: loss delivers below its published rate, and then the published number stops
#: being exact and the estimation noise that sank the previous board comes back.
#: Solvency goes back in once this board is validated without it.
COST_MARGIN = 70

#: What a seller opens at, as a fraction of what it will eventually concede.
#: At 1.0 the first offer is the best offer, countering can only lose time, and
#: taking whatever is put on the table is optimal — which is what this board did
#: until it was measured. The gap is the thing there is to bargain over.
OPENING = 0.4

RATES = (0.60, 0.70, 0.80, 0.90)
#: Capped at half the loss on purpose. A seller concedes at most
#: `margin x cooloff x loyalty`, so against a buyer that is still spreading its
#: business a 0.75 floor never binds and the observed conduct of every type gets
#: squeezed toward zero — which leaves the ledger unable to move the ranking and
#: the quote board answering the question by itself.
FLOORS = (0.00, 0.20, 0.35, 0.50)


def price_of(rate: float) -> int:
    return int(round(BASE + PREMIUM * (rate - RATES[0])))


def expected(rate: float, floor: float, loss: int) -> float:
    return rate * (VALUE - price_of(rate)) - (1 - rate) * loss * (1 - floor)


def cast_for(seed: int, loss: int = 110):
    """One match's sellers, the name with the highest expectation, and the table.

    Traits are drawn independently so that neither published reliability nor
    hidden conduct decides the match on its own, and redrawn if the top two are
    close enough that the answer would be noise.
    """
    rng = random.Random(seed)
    for _ in range(400):
        picks = [(rng.choice(RATES), rng.choice(FLOORS)) for _ in NAMES]
        if len(set(picks)) < len(NAMES) or len({p[0] for p in picks}) < len(NAMES):
            continue
        ev = {NAMES[i]: expected(*picks[i], loss) for i in range(len(NAMES))}
        top = sorted(ev.values(), reverse=True)
        if top[0] - top[1] >= 3.0:
            break

    def make(_game: int | None = None):
        return [ScriptedSeller(NAMES[i], price_of(picks[i][0]),
                               reliability=picks[i][0],
                               care_bonus=0.0,        # published rate is the rate
                               health_couples=False,  # and margin does not bend it
                               cost=price_of(picks[i][0]) - COST_MARGIN,
                               share=picks[i][1] * OPENING,
                               floor=picks[i][1], seed=seed)
                for i in range(len(NAMES))]

    board = {NAMES[i]: dict(rate=picks[i][0], price=price_of(picks[i][0]))
             for i in range(len(NAMES))}
    return make, max(ev, key=ev.get), ev, board
