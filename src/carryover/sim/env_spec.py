"""The environment as something another trainer can load.

Two things travel with it that a benchmark usually leaves out.

**A learned ceiling.** In code or maths the reference answer is the grader, so a
score interprets itself. Here there is no reference policy, and a model scoring
badly could mean the model is weak or the environment is broken — in the course
of building this one it was the environment six times out of six. So the ceiling
is trained on shuffled seller types and shipped alongside: it picks the generous
seller in round one at chance rate, which is the evidence that it learned a
policy rather than the answer key.

**The trivial policies.** Cheapest-each-round and squeeze-everything bracket the
score from below. A submission that cannot beat "take the lowest quote" has not
demonstrated anything, and without that line on the page it is easy to mistake a
number for a result.

Two regimes, because which way the relational rent flows is an empirical question
rather than a settled one: sellers charge a premium for reliability where supply
is scarce, and discount for loyalty where the buyer holds the power.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from carryover.sim.agents import ScriptedSeller
from carryover.sim.world import World

#: Enough names for the wider markets. Three sellers put a floor under the
#: concentration measures — even random play returns to the same counterparty a
#: third of the time — which made them useless for comparison against a market
#: with hundreds of wholesalers.
NAMES = tuple(chr(65 + i) for i in range(26))

#: Reference scores on the default config, from this repository's own runs.
#: `prior` knows which seller is generous before trading; nothing that learns
#: inside an episode can reach it, and it is here as an upper bound, not a target.
REFERENCE = {
    "prior_knowledge": 438,
    "rl_ceiling": 304,
    "cheapest": 81,
    "squeeze": -105,
}


@dataclass(frozen=True)
class Config:
    rounds: int = 20
    loss: int = 120
    value: int = 150
    #: buyer_power — loyalty earns a discount, visible every round.
    #: seller_power — loyalty earns care, visible only when a delivery fails.
    regime: str = "both"
    care: float = 0.15
    discount: int = 15
    cost: int = 55
    share: float = 0.15
    reliability: float = 0.80
    shuffle_types: bool = True
    n_sellers: int = 3
    bargain_rounds: int = 3
    cooloff: int = 3

    def tiers(self):
        """Generosity spread evenly over however many sellers there are, so the
        best and worst are the same regardless of market size."""
        c = self.care if self.regime in ("seller_power", "both") else 0.0
        d = self.discount if self.regime in ("buyer_power", "both") else 0
        n = self.n_sellers
        if n == 1:
            return [(d, c)]
        return [(int(round(d * (1 - i / (n - 1)))), c * (1 - i / (n - 1)))
                for i in range(n)]


@dataclass
class Episode:
    """One rollout. `reward` is the buyer's cumulative contribution profit."""

    cfg: Config
    seed: int
    sellers: list = field(default_factory=list)
    generosity: dict = field(default_factory=dict)

    def build(self):
        t = list(self.cfg.tiers())
        if self.cfg.shuffle_types:
            random.Random(self.seed * 7919).shuffle(t)
        names = NAMES[:self.cfg.n_sellers]
        self.sellers = [
            ScriptedSeller(n, 100, cost=self.cfg.cost, share=self.cfg.share,
                           floor=self.cfg.share + 0.10, seed=self.seed,
                           reliability=self.cfg.reliability,
                           care_bonus=cb, loyalty_discount=ld,
                           solvency_floor=-150, health_scale=300)
            for n, (ld, cb) in zip(names, t)]
        self.generosity = {n: ld + cb * 200 for n, (ld, cb) in zip(names, t)}
        return self

    def rollout(self, buyer):
        w = World(buyer=buyer, sellers=self.sellers, value=self.cfg.value,
                  loss=self.cfg.loss, rounds=self.cfg.rounds, seed=self.seed,
                  bargain_rounds=self.cfg.bargain_rounds, cooloff=self.cfg.cooloff)
        h = w.run()
        return h, self.score(h)

    def score(self, history) -> dict:
        from collections import Counter
        c = Counter(o.seller for o in history)
        best = max(self.generosity, key=self.generosity.get)
        fails = [o for o in history if o.failed]
        n = max(len(history), 1)
        return dict(
            reward=sum(o.buyer_profit for o in history),
            persistence=sum(1 for x, y in zip(history, history[1:])
                            if x.seller == y.seller) / max(len(history) - 1, 1),
            chose_generous=c[best] / n,
            first_round_hit=1 if history and history[0].seller == best else 0,
            sellers_alive=sum(1 for s in self.sellers if s.alive()) / len(self.sellers),
            failure_rate=sum(o.failed for o in history) / n,
            impasse_rate=(sum(o.impasse for o in fails) / len(fails)) if fails else None,
            # L=0 is a legitimate configuration — it is the control that tells a
            # relationship formed for payoff reasons from one formed because the
            # prompt implied it — so a zero total here is a well-defined "no loss
            # was ever at stake", not an error.
            seller_share=(sum(o.seller_share for o in fails) / total_loss)
                         if (total_loss := sum(o.loss for o in fails)) else None,
        )


def make(seed: int, **kw) -> Episode:
    return Episode(Config(**kw), seed).build()
