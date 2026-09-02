"""What a given rate of picking the expected-value best seller is worth.

The models see the delivery rates and still fall below a rule that uses nothing
else, and persistence does not explain it: the references concentrate on one
seller as much as the models do. That leaves which seller and how the failure
is settled. This prices the first of the two.

A buyer that picks the expected-value best seller with probability p and
otherwise picks at random, settling exactly as the null does, so the only thing
varying is arithmetic reliability. Sweeping p turns the readme's model deficits
into a claim about accuracy that can be checked against a model directly.
"""
import argparse
import random
import statistics as st
import sys

sys.path.insert(0, 'src')

from nego_eval.game.match import Match
from nego_eval.game.table4 import VALUE, cast_for

L, R, G, S = 110, 12, 4, 10


class Slips:
    """The null's policy, with a fixed chance of getting the pick wrong.

    How wrong matters. A model that slips probably takes the runner-up, not a
    seller drawn from a hat, and a runner-up costs less — so the two modes
    bracket the real thing rather than estimating it. `mode='random'` is the
    expensive bound, `mode='second'` the cheap one.
    """

    def __init__(self, p, seed, mode='random'):
        self.p, self.mode = p, mode
        self.rng = random.Random(seed)

    def choose(self, quotes, t, remaining, history):
        def ev(q):
            r = q.rate if q.rate is not None else 0.70
            return r * (VALUE - q.price) - (1 - r) * L
        ranked = sorted(quotes, key=ev, reverse=True)
        if self.rng.random() < self.p:
            return ranked[0].seller
        if self.mode == 'second':
            return ranked[1].seller if len(ranked) > 1 else ranked[0].seller
        return self.rng.choice([q.seller for q in quotes])

    def bargain(self, loss, offer, r, max_rounds, seller_name, history, terms):
        if offer >= loss or r == max_rounds:
            return True, offer
        return False, loss


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=24)
    ap.add_argument('--start', type=int, default=910_000)
    ap.add_argument('--reps', type=int, default=3)
    a = ap.parse_args()

    def score(p, mode):
        prof = []
        for k in range(a.reps):
            for i in range(a.seeds):
                seed = a.start + i
                make = cast_for(seed, loss=L)[0]
                r = Match(buyer_factory=lambda: Slips(p, seed * 100 + k, mode),
                          seller_factory=make, games=G, rounds=R, loss=L,
                          value=VALUE, seed=seed, carry_over=True, step=S).run()
                if r.rounds:
                    prof.append(r.profit)
        return st.mean(prof)

    perfect = score(1.0, 'random')
    print(f"  every pick right: {perfect:.0f}\n")
    print(f"  {'pick accuracy':<16}{'slips to 2nd':>14}{'slips at random':>17}")
    for p in (0.95, 0.90, 0.85, 0.80, 0.70, 0.50, 1 / 3):
        b = score(p, 'second') - perfect
        w = score(p, 'random') - perfect
        print(f"  {p:<16.2f}{b:>+14.0f}{w:>+17.0f}")
    print("\n  readme, models against the quote-sheet rule: +6 to -342")
