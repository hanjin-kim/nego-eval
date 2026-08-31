"""Is the policy answering from an empty table, or from the wrong page of it?

Handed a carried ledger it never trained with, the table picks the right opening
0.13 of the time — below the 0.33 it would get by choosing at random, which an
empty table would produce. So the states it lands in are not blank. They were
filled in a different context and are being read in this one.
"""
import sys, statistics as st
sys.path.insert(0, 'src')
from collections import Counter
from carryover.game.table3 import VALUE, cast_for
from carryover.sim.rl import QPolicy
from carryover.sim.world import World

LOSS, ROUNDS, GAMES, STEP = 110, 12, 4, 10
TRAIN, EVAL = 20000, 200


def play(pol, seed, carry, record=None):
    make, key, ev, board = cast_for(seed, loss=LOSS)
    pol.reset()
    prior, rewards, first = [], [], []
    for g in range(GAMES):
        pol.t_offset = g * ROUNDS
        before = len(pol.trace)
        h = World(buyer=pol, sellers=make(), value=VALUE, loss=LOSS,
                  rounds=ROUNDS, seed=seed * 1000 + g, step=STEP,
                  prior=list(prior) if carry else []).run()
        if record is not None and g > 0 and len(pol.trace) > before:
            record.append(pol.trace[before])       # the opening state of this game
        by_t = {o.t: o.buyer_profit for o in h}
        rewards += [by_t.get(t, 0.0) for t in range(ROUNDS)]
        if carry:
            prior = prior + h
    return rewards, key


pol = QPolicy(seed=0, eps=0.30)
for ep in range(TRAIN):
    pol.eps = max(0.02, 0.30 * (1 - ep / (TRAIN * 0.85)))
    r, _ = play(pol, ep * 7, carry=False)
    pol.learn(r)
pol.learning = False
trained_states = set(pol.q_choose)
visits = dict(pol.n_choose)

for carry in (False, True):
    seen = []
    for i in range(EVAL):
        play(pol, 700000 + i, carry, record=seen)
    keys = [k for _, kind, k in seen if kind == 'c']
    known = [k for k in keys if k in trained_states]
    n = Counter(visits.get(k, 0) for k in keys)
    thin = sum(v for k, v in n.items() if k < 20)
    print(f"  이월 {'있음' if carry else '없음'} · 게임2+ 첫 수에서 조회한 상태 {len(keys)}개")
    print(f"    학습에서 가본 적 있음      {len(known)/len(keys):>6.0%}")
    print(f"    방문 20회 미만인 얇은 칸    {thin/len(keys):>6.0%}")
    print(f"    방문 횟수 중앙값           {st.median([visits.get(k,0) for k in keys]):>6.0f}")
