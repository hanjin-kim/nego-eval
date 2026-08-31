"""The cast, defined once.

Five scripts were each declaring their own three sellers, and they drifted: the
game table ended up giving the generous seller the highest list price, which is
the shortcut the seller module was written to avoid. Picking the most expensive
quote then identified the generous type 55% of the time with no ledger read at
all, and every "the ledger is the only explanation" claim inherited that leak.

So the types differ in one thing and one thing only — how much of a failure they
absorb — and every seller opens at the same price. What separates them is visible
in conduct after a failure and nowhere else, which is the whole point: a type
that can be read off a price list is not a type you need a relationship to learn.

Prices still move round to round. That jitter is exogenous and identical in
distribution across sellers, so it carries no information about type, but it does
make staying cost something: the partner is usually not the cheapest quote, and a
loyal buyer pays that difference on purpose.
"""

from __future__ import annotations

import random

from carryover.sim.agents import ScriptedSeller

NAMES = ("A", "B", "C")
PRICE = 98          # the same for everyone; type is not for sale

#: (share it opens at, floor it will concede to). Index 0 is the generous type
#: and is the answer key everywhere.
TYPES = ((0.50, 0.60), (0.25, 0.35), (0.00, 0.00))


def cast_for(seed: int, cost: int = 70):
    """A factory for one match's sellers, and which name is the generous one.

    Which name holds which type is permuted per match, so nothing can be won by
    learning that 'A' is the good one; it has to be read from conduct.
    """
    order = list(range(len(TYPES)))
    random.Random(seed).shuffle(order)

    def make(_game: int | None = None):
        # Match hands its factory the game index; the direct callers do not.
        return [ScriptedSeller(NAMES[i], PRICE, cost=cost,
                               share=TYPES[order[i]][0],
                               floor=TYPES[order[i]][1], seed=seed)
                for i in range(len(NAMES))]

    return make, NAMES[order.index(0)]
