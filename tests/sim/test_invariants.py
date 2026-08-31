"""What must hold no matter which policy is playing.

These are the checks that would have caught the six defects this environment went
through. Each one exists because something plausible-looking was measured before
it was true: a bargaining step that had exactly one arithmetic answer, a buyer
paying for goods it never received, a loyalty channel whose ceiling sat below the
opening offer, a persistence metric that read 1.00 for a memoryless policy, a
ceiling learned by memorising which name was generous.

A number produced by an environment that fails any of these is not a result.
"""

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from nego_eval.sim.agents import CheapestBuyer, ScriptedSeller  # noqa: E402
from spec import Config, make  # noqa: E402
from nego_eval.sim.world import World  # noqa: E402

SEEDS = range(120)


@dataclass
class Squeeze:
    def choose(self, q, t, rem, h):
        return min(q, key=lambda x: x.price).seller

    def bargain(self, loss, offer, r, mx, name, h, terms):
        return (offer >= loss, loss)


@dataclass
class Loyal:
    def choose(self, q, t, rem, h):
        if not h:
            return q[0].seller
        return h[0].seller if any(x.seller == h[0].seller for x in q) else q[0].seller

    def bargain(self, loss, offer, r, mx, name, h, terms):
        return True, offer


@dataclass
class RandomBuyer:
    seed: int = 0
    def __post_init__(self):
        self.r = random.Random(self.seed)
    def choose(self, q, t, rem, h):
        return self.r.choice(q).seller
    def bargain(self, loss, offer, r, mx, name, h, terms):
        return True, offer


POLICIES = [("cheapest", lambda s: CheapestBuyer()),
            ("squeeze", lambda s: Squeeze()),
            ("loyal", lambda s: Loyal()),
            ("random", lambda s: RandomBuyer(s))]


# ── the loss is conserved ────────────────────────────────────────────────
@pytest.mark.parametrize("name,mk", POLICIES)
def test_loss_is_exhausted(name, mk):
    """L_B + L_S = L on every failure. The one thing capability cannot dissolve."""
    for s in SEEDS:
        h, _ = make(s).rollout(mk(s))
        for o in h:
            if o.failed:
                assert o.buyer_share + o.seller_share == o.loss, (name, s, o.t)
                assert o.buyer_share >= 0 and o.seller_share >= 0


@pytest.mark.parametrize("name,mk", POLICIES)
def test_no_price_paid_for_undelivered_goods(name, mk):
    """A failed delivery moves no price. Getting this wrong once made failing
    profitable for a seller that concedes nothing."""
    for s in SEEDS:
        h, _ = make(s).rollout(mk(s))
        for o in h:
            if o.failed:
                assert o.buyer_profit == -o.buyer_share, (name, s, o.t)


# ── the metrics mean what they claim ─────────────────────────────────────
def test_persistence_of_memoryless_policy_is_near_chance():
    """With three sellers a policy that carries nothing forward must land near
    1/3. It read 1.00 while prices were static — the metric was measuring price
    stability, not loyalty."""
    vals = [make(s).rollout(RandomBuyer(s))[1]["persistence"] for s in SEEDS]
    assert 0.25 < sum(vals) / len(vals) < 0.45


def test_types_are_not_legible_before_trading():
    """Round-one choice cannot beat chance, or an agent can skip the task."""
    hits = [make(s).rollout(CheapestBuyer())[1]["first_round_hit"] for s in SEEDS]
    assert sum(hits) / len(hits) < 0.45


# ── the incentives are not degenerate ────────────────────────────────────
def test_relationship_pays_and_the_stakes_are_what_make_it_pay():
    """Staying must beat switching, and it must beat it *more* when a failure
    actually costs something.

    The earlier version of this test asked for no advantage at all at L=0, which
    was wrong: the discount channel pays regardless of the loss, so loyalty is
    worth something even when nothing is at stake. What has to hold is that the
    stakes move it. If the gap were flat in L, the environment would be rewarding
    loyalty for its own sake and an agent's persistence would say nothing about
    whether it read the payoffs."""
    def gap(**kw):
        L = sum(make(s, **kw).rollout(Loyal())[1]["reward"] for s in SEEDS)
        C = sum(make(s, **kw).rollout(CheapestBuyer())[1]["reward"] for s in SEEDS)
        return (L - C) / len(SEEDS)

    at_stake, no_stake = gap(), gap(loss=0)
    assert at_stake > 0
    assert at_stake > no_stake * 2, (at_stake, no_stake)


def test_squeezing_is_punished():
    """Demanding the whole loss every time must lose to simply accepting, or the
    seller's solvency constraint is not doing anything."""
    sq = sum(make(s).rollout(Squeeze())[1]["reward"] for s in SEEDS) / len(SEEDS)
    ch = sum(make(s).rollout(CheapestBuyer())[1]["reward"] for s in SEEDS) / len(SEEDS)
    assert sq < ch


def test_bargaining_has_more_than_one_answer():
    """The buyer's counter must be able to change the split. When the prompt
    fixed both the total and the seller's share, 'accept' was the only
    consistent reply and the negotiation measured nothing."""
    S = [ScriptedSeller("A", 100, cost=55, share=0.15, floor=0.45, seed=0),
         ScriptedSeller("B", 100, cost=55, share=0.15, floor=0.45, seed=0),
         ScriptedSeller("C", 100, cost=55, share=0.15, floor=0.45, seed=0)]

    @dataclass
    class Asker:
        want: float
        def choose(self, q, t, rem, h):
            return q[0].seller
        def bargain(self, loss, offer, r, mx, name, h, terms):
            w = int(loss * self.want)
            return (offer >= w, w)

    got = set()
    for want in (0.0, 0.30, 0.45):
        for s in range(40):
            S2 = [ScriptedSeller(n, 100, cost=55, share=0.15, floor=0.45, seed=s)
                  for n in "ABC"]
            h = World(buyer=Asker(want), sellers=S2, loss=120, rounds=20, seed=s).run()
            got.update(o.seller_share for o in h if o.failed)
    assert len(got) > 1, "seller share never varied with what the buyer asked"


# ── regimes are actually different problems ──────────────────────────────
def test_regimes_differ_in_how_visible_the_signal_is():
    """Under buyer power the reward for loyalty shows up in every quote, so
    chasing the cheapest quote drifts into a relationship by itself. Under seller
    power it only shows up after a failure, and the same policy does not."""
    def pers(regime):
        return sum(make(s, regime=regime).rollout(CheapestBuyer())[1]["persistence"]
                   for s in SEEDS) / len(SEEDS)
    assert pers("buyer_power") > pers("seller_power") + 0.10


def test_sellers_mostly_survive_ordinary_play():
    """If a normal policy bankrupts the market the constraint is miscalibrated
    and every downstream number is about insolvency instead of bargaining."""
    alive = [make(s).rollout(CheapestBuyer())[1]["sellers_alive"] for s in SEEDS]
    assert sum(alive) / len(alive) > 0.80


def test_rollouts_are_reproducible():
    a = make(7).rollout(CheapestBuyer())[1]
    b = make(7).rollout(CheapestBuyer())[1]
    assert a == b
