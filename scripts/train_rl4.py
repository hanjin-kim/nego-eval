"""The learned ceiling where staying earns something.

Both scripted references here are rules I wrote — one reads the quote sheet, one
also reads the record — so measuring the environment against them measures my
guesses. The table has no rule and no narrative, only the payoffs, and it is
trained separately under each condition so that the question "does carrying the
record help a policy that was allowed to learn about it" is asked directly.
"""
import sys, json, statistics as st
sys.path.insert(0, 'src')
from carryover.game.table4 import VALUE, cast_for
from carryover.sim.rl import QPolicy
from carryover.sim.world import World

LOSS, ROUNDS, GAMES, STEP = 110, 12, 4, 10
TRAIN, EVAL = 20000, 500


def play(pol, seed, carry):
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


def train(carry, seed=0):
    pol = QPolicy(seed=seed, eps=0.30)
    for ep in range(TRAIN):
        pol.eps = max(0.02, 0.30 * (1 - ep / (TRAIN * 0.85)))
        _, rewards, _ = play(pol, ep * 7 + seed, carry)
        pol.learn(rewards)
    pol.learning = False
    return pol


def evaluate(pol, carry, start=700000):
    prof, gk, g1 = [], [], []
    for i in range(EVAL):
        games, rewards, key = play(pol, start + i, carry)
        if not any(games):
            continue
        prof.append(sum(rewards))
        played = [g for g in games if g]
        g1.append(int(played[0][0].seller == key))
        if len(played) > 1:
            gk.append(sum(int(g[0].seller == key) for g in played[1:]) / (len(played) - 1))
    m = lambda v: st.mean(v) if v else float('nan')
    se = lambda v: (st.stdev(v) / len(v) ** 0.5) if len(v) > 1 else 0.0
    return dict(profit=m(prof), se=se(prof), g1=m(g1), gk=m(gk))


out = {}
print(f"4번 판 · 학습 {TRAIN} · 평가 {EVAL}\n")
print(f"  {'학습':<12}{'평가':<12}{'이익':>8}{'±':>6}{'G1첫수':>8}{'G2+첫수':>9}")
for carry, lbl in ((True, '이월 있음'), (False, '이월 없음')):
    pol = train(carry)
    for ec, el in ((True, '이월 있음'), (False, '이월 없음')):
        r = evaluate(pol, ec)
        out[f"{lbl}|{el}"] = r
        print(f"  {lbl:<12}{el:<12}{r['profit']:>8.0f}{r['se']:>6.0f}"
              f"{r['g1']:>8.2f}{r['gk']:>9.2f}", flush=True)
    json.dump(out, open('data_rl4.json', 'w'), indent=1)
print(f"\n  참고 · 호가판만 904 · 원장(기억O) 993 · 원장(기억X) 929 · 프리미엄 120")
