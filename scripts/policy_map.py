"""Where a surplus number sits on the map of policies.

A trained model's score means nothing on its own. -650 could be "close to
random", in which case there is a lot for an optimiser to find, or it could be
well below random, in which case the model is doing something actively costly
and the question is what. This places the simple policies on the same scale,
through the same `Driver` the model is measured with, so the comparison is not
across two code paths.

The two channels are scored apart on purpose: picking a seller and settling a
failed delivery are separate decisions, and a policy can be fine at one while
throwing the match away on the other.
"""
import argparse
import random
import statistics as st
import sys

sys.path.insert(0, 'src')

from nego_eval.rl.vf_env import EVAL, Driver


def _cheapest(pos):
    return min(pos.data['quotes'], key=lambda q: q.price).seller


def _sticky(pos):
    h = pos.data['history']
    return h[0].seller if h else _cheapest(pos)


def _evbest(pos):
    """What the null picks: expected value on the posted delivery rate."""
    def ev(q):
        r = q.rate if q.rate is not None else 0.70
        return r * (150 - q.price) - (1 - r) * 110
    return max(pos.data['quotes'], key=ev).seller


def _random(pos, rng):
    return rng.choice(list(pos.legal))


def _accept(pos):
    return {'accept': True, 'ask': pos.data['offer']}


def _half(pos):
    """Ask for half the loss once, then take what is there."""
    want = pos.data['loss'] // 2
    if pos.data['offer'] >= want or pos.data['r'] == pos.data['max_rounds']:
        return {'accept': True, 'ask': pos.data['offer']}
    return {'accept': False, 'ask': want}


def _null(pos):
    """What the null does: hold out for the whole loss, concede at the buzzer."""
    if pos.data['offer'] >= pos.data['loss'] or pos.data['r'] == pos.data['max_rounds']:
        return {'accept': True, 'ask': pos.data['offer']}
    return {'accept': False, 'ask': pos.data['loss']}


def _greedy(pos):
    """Never settle for less than the whole loss. A plausible failure mode."""
    return {'accept': False, 'ask': pos.data['loss']}


PICKS = {'evbest': _evbest, 'cheapest': _cheapest,
         'sticky': _sticky, 'random': _random}
SETTLES = {'null': _null, 'half': _half, 'accept': _accept, 'greedy': _greedy}


def score(pick, settle, n, start, preset, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        d = Driver(start + i, **preset)
        while not d.done:
            p = d.pending
            if p.kind == 'choose':
                got = {'seller': pick(p, rng) if pick is _random else pick(p)}
            else:
                got = settle(p)
            d.step(got)
        out.append(d.reward())
    return st.mean(out), st.stdev(out) / n ** 0.5 if n > 1 else 0.0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', type=int, default=48)
    ap.add_argument('--start', type=int, default=900_000)
    a = ap.parse_args()

    print(f"  eval preset, {a.n} boards from seed {a.start}."
          f"  0 is a policy that ignores the record.\n")
    print(f"  {'pick':<10}{'settle':<10}{'surplus':>12}")
    rows = []
    for pn, pf in PICKS.items():
        for sn, sf in SETTLES.items():
            m, se = score(pf, sf, a.n, a.start, EVAL)
            rows.append((m, pn, sn, se))
            print(f"  {pn:<10}{sn:<10}{m:>+9.0f} ± {se:.0f}")
    rows.sort(reverse=True)
    print(f"\n  best  {rows[0][1]}/{rows[0][2]}  {rows[0][0]:+.0f}")
    print(f"  worst {rows[-1][1]}/{rows[-1][2]}  {rows[-1][0]:+.0f}")

    # The null is evbest/null and scores 0 by construction. Splitting its lead
    # says which of the two decisions an optimiser has to fix first.
    at = {(pn, sn): m for m, pn, sn, _ in rows}
    print("\n  널의 우위를 채널로 쪼개면:")
    for hold, label in (('null', '합의를 널로 고정, 고르기만 바꿈'),
                        ('accept', '합의는 수용, 고르기만 바꿈')):
        d = at[('evbest', hold)] - at[('cheapest', hold)]
        print(f"    {label:<34}{d:+8.0f}")
    for hold, label in (('evbest', '고르기를 널로 고정, 합의만 바꿈'),
                        ('cheapest', '고르기는 최저가, 합의만 바꿈')):
        d = at[(hold, 'null')] - at[(hold, 'accept')]
        print(f"    {label:<34}{d:+8.0f}")
