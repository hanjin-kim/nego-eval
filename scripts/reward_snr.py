"""Does truncating the horizon actually make one decision visible?

The claim behind the windowed reward is that a branch's whole-match total
carries thirty-odd later decisions' noise, so a group of eight cannot resolve
the 10-50 that the branched decision is worth. That is a claim about a ratio,
and it is checkable here with no GPU: freeze a prefix, take each legal action
at the cut, and play the tail with a stochastic policy many times.

The tail policy is random-pick-and-accept because that is where the trained
model actually sits on `policy_map.py` — the noise being measured is the noise
this run would really face, not a worst case invented for the argument.
"""
import argparse
import random
import statistics as st
import sys

sys.path.insert(0, 'src'); sys.path.insert(0, 'deploy')

from grpo import Roll, window_reward
from nego_eval.rl.vf_env import TRAIN, Driver


def _tail(pos, rng):
    if pos.kind == 'choose':
        return {'seller': rng.choice(list(pos.legal))}
    return {'accept': True, 'ask': pos.data['offer']}


def _prefix(seed, preset, cut, rng):
    """Answers up to `cut`, played by the same stochastic policy."""
    d = Driver(seed, **preset)
    out = []
    while not d.done and len(out) < cut:
        out.append(_tail(d.pending, rng))
        d.step(out[-1])
    return out


HORIZONS = (0, 2, 4, 6, 8, 12, 24)      # 0 is the whole match


def _first_choose(seed, preset, at, rng_seed=7):
    """The first turn at or after `at` where a seller is picked."""
    d = Driver(seed, **preset)
    out = []
    while not d.done:
        if len(out) >= at and d.pending.kind == 'choose':
            return out, list(d.pending.legal)
        out.append(_tail(d.pending, random.Random(rng_seed + len(out))))
        d.step(out[-1])
    return None, None


def sample(seed, preset, cut, pre, action, reps, base_seed=7):
    """Every horizon off one set of random tails, so the sweep is free."""
    out = {h: [] for h in HORIZONS}
    for k in range(reps):
        rng = random.Random(base_seed * 1_000_003 + k)
        r = Roll(seed, preset, pre + [action])
        while not r.d.done:
            r._record(_tail(r.d.pending, rng))
        for h in HORIZONS:
            out[h].append(window_reward(r, cut, h))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=24)
    ap.add_argument('--reps', type=int, default=40)
    ap.add_argument('--cut', type=int, default=4)
    a = ap.parse_args()

    tot = {h: [] for h in HORIZONS}
    used = 0
    for s in range(a.seeds):
        seed = 500_000 + s
        pre, legal = _first_choose(seed, TRAIN, a.cut)
        if pre is None:
            continue
        used += 1
        per = {h: {'m': [], 's': []} for h in HORIZONS}
        for name in legal:
            got = sample(seed, TRAIN, len(pre), pre, {'seller': name}, a.reps)
            for h in HORIZONS:
                per[h]['m'].append(st.mean(got[h]))
                per[h]['s'].append(st.stdev(got[h]))
        for h in HORIZONS:
            sd = st.mean(per[h]['s'])
            tot[h].append((max(per[h]['m']) - min(per[h]['m'])) / sd if sd else 0.0)

    print(f"  TRAIN preset, {used} boards, {a.reps} random tails per action.")
    print("  How far apart the best and worst seller are, in tail standard"
          " deviations.\n")
    print(f"  {'horizon':<12}{'effect/SD':>11}")
    base = st.mean(tot[0])
    for h in HORIZONS:
        v = st.mean(tot[h])
        tag = '  (전체 매치)' if h == 0 else f'  x{v / base:.1f}'
        print(f"  {(str(h) + ' rounds') if h else 'whole match':<12}{v:>11.3f}{tag}")
