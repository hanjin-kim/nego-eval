"""Terminal reward or dense — which does the same learner do better with?

A match is about thirty-five decisions and the reward arrives once, at the end.
Thirty-five turns is a long way for credit to travel, and it is the largest risk
in putting this on a GPU: a run can fail for want of signal shape rather than for
want of compute, and it fails expensively.

The dense form costs nothing to produce. The baseline that is already subtracted
— what a policy reading only the quote sheet earns on this seed — is played round
by round, so its per-round profit is available too:

    terminal   r = (match profit) - (baseline match profit),  once
    dense      r_t = (my profit at t) - (baseline profit at t),  every round

They sum to the same number and are the same control variate. Only the shape
differs. If the dense form does not help a tabular learner, it will not save a
policy-gradient run either, and that is worth knowing for free.
"""
import sys, json, statistics as st
sys.path.insert(0, 'src')
from nego_eval.game.buyers import BoardOnlyBuyer
from nego_eval.game.table4 import VALUE, cast_for
from nego_eval.sim.rl import QPolicy
from nego_eval.sim.world import World

LOSS, STEP = 110, 10
TRAIN_EPS, EVAL_EPS = 20000, 500
#: The shape a GPU run would use, not the one the published numbers use.
ROUNDS, GAMES = 4, 6
_base: dict[int, list[float]] = {}


def baseline_rounds(seed):
    """Per-round profit of the record-blind policy on this seed."""
    if seed not in _base:
        make = cast_for(seed, loss=LOSS)[0]
        prior, out = [], []
        for g in range(GAMES):
            h = World(buyer=BoardOnlyBuyer(), sellers=make(), value=VALUE,
                      loss=LOSS, rounds=ROUNDS, seed=seed * 1000 + g, step=STEP,
                      prior=list(prior)).run()
            by_t = {o.t: o.buyer_profit for o in h}
            out += [by_t.get(t, 0.0) for t in range(ROUNDS)]
            prior = prior + h
        _base[seed] = out
    return _base[seed]


def play(pol, seed):
    make, key = cast_for(seed, loss=LOSS)[:2]
    pol.reset()
    prior, rewards, games = [], [], []
    for g in range(GAMES):
        pol.t_offset = g * ROUNDS
        h = World(buyer=pol, sellers=make(), value=VALUE, loss=LOSS,
                  rounds=ROUNDS, seed=seed * 1000 + g, step=STEP,
                  prior=list(prior)).run()
        games.append(h)
        by_t = {o.t: o.buyer_profit for o in h}
        rewards += [by_t.get(t, 0.0) for t in range(ROUNDS)]
        prior = prior + h
    return games, rewards, key


def train(shape, seed=0):
    pol = QPolicy(seed=seed, eps=0.30)
    for ep in range(TRAIN_EPS):
        pol.eps = max(0.02, 0.30 * (1 - ep / (TRAIN_EPS * 0.85)))
        s = ep * 7 + seed
        _, raw, _ = play(pol, s)
        b = baseline_rounds(s)
        if shape == 'dense':
            r = [x - y for x, y in zip(raw, b)]
        elif shape == 'terminal':
            # the same total, delivered only on the last round
            r = [0.0] * (len(raw) - 1) + [sum(raw) - sum(b)]
        else:
            r = list(raw)
        pol.learn(r)
    pol.learning = False
    return pol


def evaluate(pol, start=700000):
    surp, gk, g1 = [], [], []
    for i in range(EVAL_EPS):
        s = start + i
        games, raw, key = play(pol, s)
        if not any(games):
            continue
        surp.append(sum(raw) - sum(baseline_rounds(s)))
        played = [g for g in games if g]
        g1.append(int(played[0][0].seller == key))
        if len(played) > 1:
            gk.append(sum(int(g[0].seller == key) for g in played[1:]) / (len(played) - 1))
    m = lambda v: st.mean(v) if v else float('nan')
    se = lambda v: (st.stdev(v) / len(v) ** 0.5) if len(v) > 1 else 0.0
    return dict(surplus=m(surp), se=se(surp), g1=m(g1), gk=m(gk))


print(f"{ROUNDS}x{GAMES} (학습 프리셋) · 학습 {TRAIN_EPS} · 평가 {EVAL_EPS}")
print(f"보상은 셋 다 같은 총합. 도착하는 시점만 다름.\n")
print(f"  {'보상 형태':<12}{'잉여':>8}{'±':>6}{'G1첫수':>8}{'G2+첫수':>9}{'변화':>7}")
out = {}
for shape, lbl in (('raw', '원점수'), ('terminal', '종단'), ('dense', '조밀')):
    r = evaluate(train(shape))
    out[lbl] = r
    print(f"  {lbl:<12}{r['surplus']:>8.0f}{r['se']:>6.0f}{r['g1']:>8.2f}"
          f"{r['gk']:>9.2f}{r['gk']-r['g1']:>+7.2f}", flush=True)
    json.dump(out, open('data/dense_reward.json', 'w'), indent=1)
print(f"\n  잉여 0 = 원장을 안 읽는 정책과 동점. 이 프리셋에서 환경이 주는 최대는 약 +52.")
