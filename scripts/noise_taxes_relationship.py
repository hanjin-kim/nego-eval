"""Does noise in picking cost the relational margin more than it costs the board?

The relational value on this board is bought by staying. Reliability is
`base + bonus x loyalty` and the share a seller absorbs is
`floor + floor_care x loyalty`, where loyalty is this buyer's share of the last
eight rounds. So a buyer whose picks wobble does not merely pick worse — it
never accumulates the loyalty that the relational payoff is priced in.

If that is right, the two channels are not additive. Adding the same slip rate
should cost the record-reading policy more than the quote-sheet one, and the
margin between them — the 165 this board exists to measure — should shrink as
accuracy falls, closing entirely before accuracy does.

Same slips, same seeds, two policies: `BoardOnlyBuyer`, which never consults
the record, and `EVBuyer`, which does.
"""
import argparse
import random
import statistics as st
import sys

sys.path.insert(0, 'src')

from nego_eval.game.buyers import BoardOnlyBuyer, EVBuyer
from nego_eval.game.match import Match
from nego_eval.game.table4 import VALUE, cast_for

L, R, G, S = 110, 12, 4, 10


class Slipping:
    """Some reference policy, with a fixed chance of taking the runner-up.

    The runner-up rather than a random seller, because that is the cheaper and
    more plausible error, and it keeps the comparison conservative.
    """

    def __init__(self, inner, p, seed):
        self.inner, self.p = inner, p
        self.rng = random.Random(seed)

    def choose(self, quotes, t, remaining, history):
        want = self.inner.choose(quotes, t, remaining, history)
        if self.rng.random() < self.p or len(quotes) < 2:
            return want
        return self.rng.choice([q.seller for q in quotes if q.seller != want])

    def bargain(self, *a, **k):
        return self.inner.bargain(*a, **k)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=24)
    ap.add_argument('--start', type=int, default=910_000)
    ap.add_argument('--reps', type=int, default=3)
    a = ap.parse_args()

    def score(kind, p):
        prof = []
        for k in range(a.reps):
            for i in range(a.seeds):
                seed = a.start + i
                make = cast_for(seed, loss=L)[0]
                inner = BoardOnlyBuyer() if kind == 'board' else EVBuyer()
                r = Match(buyer_factory=lambda: Slipping(inner, p, seed * 100 + k),
                          seller_factory=make, games=G, rounds=R, loss=L,
                          value=VALUE, seed=seed, carry_over=True, step=S).run()
                if r.rounds:
                    prof.append(r.profit)
        return st.mean(prof)

    print(f"  {'pick accuracy':<16}{'quote sheet':>13}{'reads record':>14}"
          f"{'margin':>9}")
    top = None
    for p in (1.0, 0.95, 0.90, 0.85, 0.80, 0.70, 0.50):
        b, e = score('board', p), score('rule', p)
        top = (e - b) if top is None else top
        print(f"  {p:<16.2f}{b:>13.0f}{e:>14.0f}{e - b:>+9.0f}")
    print("\n  margin at perfect picking is what the record is worth here;")
    print("  the readme prices it at 165 across the whole optimum.")
