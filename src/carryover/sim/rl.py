"""A ceiling for this environment, found without any language.

Every claim about the buyers so far has been measured against policies I wrote by
hand. `GrudgeBuyer` is a guess at what loyalty should look like, and using a guess
as the benchmark makes "the model underperforms" a statement about my guess.

Tabular Q-learning removes the guess. It has no narrative, no prompt and no notion
of trust — only the payoffs. So it answers a question the language experiments
cannot:

    does this environment actually reward staying with one counterparty?

If the learned policy concentrates, loyalty is optimal here and a model that
scatters is failing. If it scatters too, the environment never rewarded loyalty
and the earlier reading was wrong. That second outcome would be a finding about
my design, which is why it is worth being able to see.

The state is deliberately coarse. Fine discretisation lets the table memorise
seed-specific price noise and report a ceiling nobody could reach.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field

from carryover.sim.world import Outcome, Quote


def _band(x: float, edges: tuple[float, ...]) -> int:
    return sum(1 for e in edges if x >= e)


@dataclass
class QPolicy:
    alpha: float = 0.10
    gamma: float = 0.95
    eps: float = 0.20
    seed: int = 0
    learning: bool = True
    ASKS: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75)

    q_choose: dict = field(default_factory=lambda: defaultdict(float))
    q_bargain: dict = field(default_factory=lambda: defaultdict(float))
    n_choose: dict = field(default_factory=lambda: defaultdict(int))
    n_bargain: dict = field(default_factory=lambda: defaultdict(int))
    rng: random.Random = field(default=None)
    #: (round_index, kind, key) — the round is what makes credit assignment
    #: possible. Crediting every visit with the whole-episode return, as the first
    #: version did, drowns a 20-step trajectory in the variance of four coin
    #: flips: the table then learns the seed, not the policy.
    trace: list = field(default_factory=list)
    #: Added to the world's round index before anything is traced. A match is
    #: several games and the world restarts `t` at zero in each of them, so
    #: without an offset every game's opening move would be credited with the
    #: same return and the table would never see that a relationship built in
    #: game one pays in game two. With it, the horizon is the match.
    t_offset: int = 0
    _t: int = 0

    def __post_init__(self):
        if self.rng is None:
            self.rng = random.Random(self.seed)

    def reset(self):
        self.trace = []

    # ---- state ----------------------------------------------------------
    def _s_choose(self, name, price, quotes, history, remaining, rate=None):
        f = [o for o in history if o.seller == name and o.failed]
        borne = (sum(o.seller_share for o in f) / sum(o.loss for o in f)
                 if f and f[0].loss else -1.0)
        gap = price - min(q.price for q in quotes)
        mine = sum(1 for o in history if o.seller == name)
        return (_band(gap, (1, 6)),
                _band(borne, (0.0, 0.15, 0.40)) + 1 if borne >= 0 else 0,
                _band(mine, (1, 4)),          # how much of a relationship exists
                _band(remaining, (4, 10)),
                # Where the board publishes a delivery rate it is the largest
                # single term in the payoff, and a table that cannot see it is
                # choosing blind on that board while still being reported as a
                # ceiling. Absent, this is a constant and nothing changes.
                _band(rate, (0.65, 0.75, 0.85)) if rate is not None else 0)

    def _s_bargain(self, name, loss, offer, r, history):
        f = [o for o in history if o.seller == name and o.failed]
        borne = (sum(o.seller_share for o in f) / sum(o.loss for o in f)
                 if f and f[0].loss else -1.0)
        return (_band(offer / max(loss, 1), (0.1, 0.3, 0.5)),
                _band(borne, (0.0, 0.15, 0.40)) + 1 if borne >= 0 else 0,
                r)

    # ---- acting ---------------------------------------------------------
    def choose(self, quotes: list[Quote], t: int, remaining: int,
               history: list[Outcome]) -> str:
        states = {q.seller: self._s_choose(q.seller, q.price, quotes, history,
                                           remaining, getattr(q, "rate", None))
                  for q in quotes}
        if self.learning and self.rng.random() < self.eps:
            pick = self.rng.choice(quotes).seller
        else:
            pick = max(quotes, key=lambda q: self.q_choose[states[q.seller]]).seller
        self._t = t + self.t_offset
        self.trace.append((self._t, "c", states[pick]))
        return pick

    def bargain(self, loss, offer, r, max_rounds, seller_name, history, terms):
        s = self._s_bargain(seller_name, loss, offer, r, history)
        if self.learning and self.rng.random() < self.eps:
            a = self.rng.randrange(len(self.ASKS))
        else:
            a = max(range(len(self.ASKS)), key=lambda i: self.q_bargain[(s, i)])
        self.trace.append((self._t, "b", (s, a)))
        want = int(loss * self.ASKS[a])
        return (offer >= want, want)

    # ---- learning -------------------------------------------------------
    def learn(self, rewards: list[float]) -> None:
        """First-visit Monte Carlo with returns measured forward from each round.

        `rewards[t]` is the buyer's profit in round t. A state-action visited at
        round t is credited with the discounted return from t onward, not with the
        whole episode — a choice made in round 18 cannot be blamed for a failure
        in round 3.

        Averaging is incremental (1/n) rather than a fixed step, so early noisy
        estimates are not frozen in by a large alpha.
        """
        g = [0.0] * (len(rewards) + 1)
        for t in range(len(rewards) - 1, -1, -1):
            g[t] = rewards[t] + self.gamma * g[t + 1]
        seen = set()
        for t, kind, key in self.trace:
            if (t, kind, key) in seen:
                continue
            seen.add((t, kind, key))
            ret = g[t] if t < len(g) else 0.0
            if kind == "c":
                self.n_choose[key] += 1
                self.q_choose[key] += (ret - self.q_choose[key]) / self.n_choose[key]
            else:
                self.n_bargain[key] += 1
                self.q_bargain[key] += (ret - self.q_bargain[key]) / self.n_bargain[key]
