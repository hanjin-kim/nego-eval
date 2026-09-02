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
from policy_map import _evbest, _null
from nego_eval.rl.vf_env import TRAIN, Driver


EPS = 1.0     # probability the tail plays at random; rebound from the CLI


def _tail(pos, rng):
    """The policy that plays the turns after the branched one.

    `EPS = 1` is random-pick-and-accept, which is where policy_map.py puts the
    trained model. `EPS = 0` is the null's own play. Everything in between is
    the question the sweep asks: how competent does the tail have to be before
    the decision under it becomes visible.
    """
    noisy = rng.random() < EPS
    if pos.kind == 'choose':
        return {'seller': rng.choice(list(pos.legal)) if noisy else _evbest(pos)}
    return ({'accept': True, 'ask': pos.data['offer']} if noisy else _null(pos))


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


def measure(seeds, reps, cut):
    """Effect and tail SD per horizon, averaged over boards."""
    acc = {h: {'eff': [], 'sd': []} for h in HORIZONS}
    used = 0
    for i in range(seeds):
        seed = 500_000 + i
        pre, legal = _first_choose(seed, TRAIN, cut)
        if pre is None:
            continue
        used += 1
        per = {h: {'m': [], 's': []} for h in HORIZONS}
        for name in legal:
            got = sample(seed, TRAIN, len(pre), pre, {'seller': name}, reps)
            for h in HORIZONS:
                per[h]['m'].append(st.mean(got[h]))
                per[h]['s'].append(st.stdev(got[h]))
        for h in HORIZONS:
            acc[h]['eff'].append(max(per[h]['m']) - min(per[h]['m']))
            acc[h]['sd'].append(st.mean(per[h]['s']))
    return used, {h: (st.mean(v['eff']), st.mean(v['sd'])) for h, v in acc.items()}


def _ratio(eff, sd):
    return f"{eff / sd:>8.2f}" if sd > 1e-9 else "       -"


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=24)
    ap.add_argument('--reps', type=int, default=40)
    ap.add_argument('--cut', type=int, default=4)
    ap.add_argument('--eps', type=float, nargs='*',
                    default=[1.0, 0.7, 0.4, 0.2, 0.0],
                    help='how often the tail plays at random; 1 is the model, '
                         '0 is the null')
    a = ap.parse_args()

    print(f"  TRAIN preset, {a.reps} tails per action, cut at the first pick"
          f" from turn {a.cut}.")
    print("  Best minus worst seller, and the tail spread it has to be seen"
          " through.\n")
    print(f"  {'tail random':<14}{'horizon':<14}{'effect':>9}{'tail SD':>10}"
          f"{'effect/SD':>10}")
    for eps in a.eps:
        globals()['EPS'] = eps
        used, got = measure(a.seeds, a.reps, a.cut)
        for h in (0, 8):
            eff, sd = got[h]
            label = 'whole match' if h == 0 else '8 rounds'
            head = f"{eps:.1f}" if h == 0 else ''
            print(f"  {head:<14}{label:<14}{eff:>9.0f}{sd:>10.0f}"
                  f"{_ratio(eff, sd)}")
        print()
    print(f"  ({used} boards)")
