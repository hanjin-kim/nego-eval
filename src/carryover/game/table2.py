"""A cast where two countable things matter and neither one settles it.

The first table made reliability a function of loyalty alone, so failure counts
carried no information about type and "buy from whoever fails least" — the most
ordinary procurement heuristic there is — scored chance. That is not a hard
environment, it is a trap: a model applying a rule that is right in the world got
punished for it here, and calling the result a capability gap would have been
measuring compliance with my parameterisation.

Here both signals are real and both are countable:

    failure count     the seller's own base reliability, a stable trait
    loss borne        the share of each failure it absorbs, a stable trait

Neither dominates. A reliable seller that pays nothing and a fragile one that
pays most of every loss are both plausible best buys, and which is better depends
on arithmetic the ledger fully supports:

    E[round] = r (value - price) - (1 - r) (loss - seller's share)

So the answer key is no longer a type I labelled generous. It is the seller that
maximises the objective the prompt already states, computed from two numbers the
agent is shown. Nothing has to be judged, which keeps the reward verifiable.

The pairing of traits is drawn per match, so no fixed rule — "most reliable",
"pays most", "cheapest" — wins across the population.
"""

from __future__ import annotations

import random

from carryover.sim.agents import ScriptedSeller

NAMES = ("A", "B", "C")
PRICE = 98
VALUE = 150

#: Drawn independently per seller, not permuted from a fixed set of three. With
#: a fixed set the same type wins every match and one heuristic — "buy from
#: whoever fails least" — is dominant, which is the old trap with the sign
#: flipped. Sampling the two traits separately makes the best buy sometimes the
#: dependable one and sometimes the one that shares, so the choice is arithmetic
#: rather than a rule.
RELIABILITY = (0.62, 0.70, 0.78, 0.86)
FLOORS = (0.00, 0.25, 0.50, 0.70)


def expected(rel: float, floor: float, loss: int, price: int = PRICE,
             value: int = VALUE) -> float:
    """Per-round expectation if this seller is bought from and concedes to `floor`.

    The floor rather than the opening share, because a buyer that bargains gets
    the floor; using the opening offer would make the arithmetic reward a buyer
    that never negotiates.
    """
    return rel * (value - price) - (1 - rel) * (loss - loss * floor)


def cast_for(seed: int, loss: int = 110, cost: int = 70):
    """One match's sellers, and the name with the highest expected value.

    Traits are redrawn until the best and second-best differ by enough that the
    answer is not a coin flip; a key decided by rounding error would put a
    ceiling on every score for reasons that have nothing to do with the agent.
    """
    rng = random.Random(seed)
    for _ in range(200):
        picks = [(rng.choice(RELIABILITY), rng.choice(FLOORS))
                 for _ in range(len(NAMES))]
        if len({p for p in picks}) < len(NAMES):
            continue
        ev = {NAMES[i]: expected(picks[i][0], picks[i][1], loss)
              for i in range(len(NAMES))}
        top = sorted(ev.values(), reverse=True)
        if top[0] - top[1] >= 2.0:
            break

    def make(_game: int | None = None):
        return [ScriptedSeller(NAMES[i], PRICE, cost=cost,
                               reliability=picks[i][0],
                               share=min(picks[i][1], 0.5),
                               floor=picks[i][1], seed=seed)
                for i in range(len(NAMES))]

    return make, max(ev, key=ev.get), ev
