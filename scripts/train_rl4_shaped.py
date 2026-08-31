"""Does subtracting the non-relational part of the reward change what is learned?

Eighty-five percent of a score on this board comes from buying at a sensible
price, which every policy does about as well. The axis the environment exists to
measure is the other fifteen, so a gradient on raw profit spends most of its
signal on something already solved and the relational part arrives as noise.

The fix is a control variate: subtract, round by round and on the same seed, what
a policy that reads the quote sheet and never the record earned. That baseline
does not depend on the learner's actions, so it is unbiased — it removes the
board and leaves the axis. It is the same device that took the standard error on
the model comparisons from 657 to about 70.

    raw       reward = profit
    shaped    reward = profit - what ignoring the record would have earned here
"""
import sys, json, statistics as st
sys.path.insert(0, 'src')
from carryover.game.buyers3 import BoardOnlyBuyer
from carryover.game.table4 import VALUE, cast_for
from carryover.sim.rl import QPolicy
from carryover.sim.world import World

LOSS, ROUNDS, GAMES, STEP = 110, 12, 4, 10
TRAIN, EVAL = 20000, 500
_base: dict[int, list[float]] = {}


def baseline(seed, carry=True):
    """Per-round profit of the record-blind policy on this seed."""
    if seed not in _base:
        make = cast_for(seed, loss=LOSS)[0]
        prior, out = [], []
        for g in range(GAMES):
            h = World(buyer=BoardOnlyBuyer(), sellers=make(), value=VALUE,
                      loss=LOSS, rounds=ROUNDS, seed=seed * 1000 + g, step=STEP,
                      prior=list(prior) if carry else []).run()
            by_t = {o.t: o.buyer_profit for o in h}
            out += [by_t.get(t, 0.0) for t in range(ROUNDS)]
            if carry:
                prior = prior + h
        _base[seed] = out
    return _base[seed]


def play(pol, seed, carry=True):
    make, key, ev, board, hidden = cast_for(seed, loss=LOSS)
    pol.reset()
    prior, games, rewards = [], [], []
    for g in range(GAMES):
        pol.t_offset = g * ROUNDS
        h = World(buyer=pol, sellers=make(), value=VALUE, loss=LOSS,
                  rounds=ROUNDS, seed=seed * 1000 + g, step=STEP,
                  prior=list(prior) if carry else []).run()
        games.append(h)
        by_t = {o.t: o.buyer_profit for o in h}
        rewards += [by_t.get(t, 0.0) for t in range(ROUNDS)]
        if carry:
            prior = prior + h
    return games, rewards, key


def train(shaped, seed=0):
    pol = QPolicy(seed=seed, eps=0.30)
    for ep in range(TRAIN):
        pol.eps = max(0.02, 0.30 * (1 - ep / (TRAIN * 0.85)))
        s = ep * 7 + seed
        _, rewards, _ = play(pol, s)
        if shaped:
            b = baseline(s)
            rewards = [r - bb for r, bb in zip(rewards, b)]
        pol.learn(rewards)
    pol.learning = False
    return pol


def evaluate(pol, start=700000):
    prof, gk, g1, surp = [], [], [], []
    for i in range(EVAL):
        s = start + i
        games, rewards, key = play(pol, s)
        if not any(games):
            continue
        prof.append(sum(rewards))
        surp.append(sum(rewards) - sum(baseline(s)))
        played = [g for g in games if g]
        g1.append(int(played[0][0].seller == key))
        if len(played) > 1:
            gk.append(sum(int(g[0].seller == key) for g in played[1:]) / (len(played) - 1))
    m = lambda v: st.mean(v) if v else float('nan')
    se = lambda v: (st.stdev(v) / len(v) ** 0.5) if len(v) > 1 else 0.0
    return dict(profit=m(prof), se=se(prof), surplus=m(surp), surplus_se=se(surp),
                g1=m(g1), gk=m(gk))


print(f"4번 판 · 학습 {TRAIN} · 평가 {EVAL} · 잉여 = 호가판만 정책 대비\n")
print(f"  {'보상':<10}{'이익':>8}{'±':>6}{'잉여':>8}{'±':>6}{'G1첫수':>8}{'G2+첫수':>9}")
out = {}
for shaped, lbl in ((False, '원점수'), (True, '잉여')):
    r = evaluate(train(shaped))
    out[lbl] = r
    print(f"  {lbl:<10}{r['profit']:>8.0f}{r['se']:>6.0f}{r['surplus']:>8.0f}"
          f"{r['surplus_se']:>6.0f}{r['g1']:>8.2f}{r['gk']:>9.2f}", flush=True)
    json.dump(out, open('data_rl4_shaped.json', 'w'), indent=1)
print(f"\n  기준 · 최적(DP) 1099 · 스크립트 996 · 호가판만 934 · 관계 잉여 165")
