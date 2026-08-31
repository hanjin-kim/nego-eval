"""The models play the fourth board, and the comparison is paired.

Everything said about language models here has come from one frozen position at
a time, which measures reading rather than playing. It is not the same thing: on
the first board four models played whole matches and two of them finished level
with a policy that consults the record on every move, despite scoring near chance
on single positions.

Match-level spread is several hundred, so a model's mean against a reference mean
computed on other seeds would mostly measure which boards were drawn — one early
seed on the first board was worth +530 to a policy that consults nothing. Every
profit is therefore recorded per seed and the references are replayed on exactly
those seeds. The board cancels; what is left is the agent.
"""
import sys, json, time, threading, statistics as st
sys.path.insert(0, 'src')
from collections import Counter
from carryover.game.match import Match
from carryover.game.table4 import VALUE, cast_for
from carryover.sim import llm
from carryover.sim.agents import LLMBuyer

LOSS, ROUNDS, GAMES, STEP = 110, 12, 4, 10
START = 910000
import ast
#: [(model, matches), ...] and the carry-over condition, both from the command
#: line so one script serves every arm.
PLAN = ast.literal_eval(sys.argv[1]) if len(sys.argv) > 1 else [('deepseek/deepseek-chat', 12)]
CARRY = (sys.argv[2] != 'off') if len(sys.argv) > 2 else True
TAG = '' if CARRY else '_off'
#: One request in flight per arm. Two arms running at once was enough to draw
#: rate limits that cost whole matches.
GATE = threading.Semaphore(1)
lock = threading.Lock()


def run(model, n):
    t0 = time.time()
    prof, gk, g1, top, per_seed, made = [], [], [], [], {}, []
    for i in range(n):
        seed = START + i
        make, key, ev, board, hidden = cast_for(seed, loss=LOSS)

        def factory():
            b = LLMBuyer(model=model); made.append(b); return b

        with GATE:
            try:
                r = Match(buyer_factory=factory, seller_factory=make, games=GAMES,
                          rounds=ROUNDS, loss=LOSS, value=VALUE, seed=seed,
                          carry_over=CARRY, step=STEP).run()
            except Exception as e:
                with lock: print(f"  {model} m{i} ERR {str(e)[:50]}", flush=True)
                continue
        if not r.rounds:
            continue
        prof.append(r.profit); per_seed[seed] = r.profit
        c = Counter(o.seller for o in r.rounds)
        top.append(max(c.values()) / len(r.rounds))
        played = [g for g in r.games if g]
        g1.append(int(played[0][0].seller == key))
        if len(played) > 1:
            gk.append(sum(int(g[0].seller == key) for g in played[1:]) / (len(played) - 1))
        with lock:
            json.dump(dict(model=model, done=len(prof), per_seed=per_seed),
                      open(f"data_f4{TAG}_{model.split('/')[-1]}.partial.json", 'w'))
    m = lambda v: (st.mean(v) if v else float('nan'))
    se = lambda v: (st.stdev(v) / len(v) ** 0.5) if len(v) > 1 else 0.0
    d = dict(model=model, matches=len(prof), profit=m(prof), se=se(prof),
             g1=m(g1), gk=m(gk), top=m(top), per_seed=per_seed,
             fallbacks=sum(b.fallbacks for b in made),
             tokens=sum(b.tokens for b in made),
             calls=llm.per_model.get(model, {}).get('n', 0),
             secs=round(time.time() - t0))
    with lock:
        json.dump(d, open(f"data_f4{TAG}_{model.split('/')[-1]}.json", 'w'), indent=1)
        print(f"  {model.split('/')[-1]:<20}{d['profit']:>8.0f}{d['se']:>6.0f}"
              f"{d['g1']:>8.2f}{d['gk']:>9.2f}{d['top']:>9.2f}"
              f"{d['matches']:>5}{d['fallbacks']:>6}"
              f"{d['tokens']/max(d['matches'],1):>9,.0f}{d['secs']:>7}s", flush=True)


print(f"4번 판 · 4게임 x 12라운드 · 이월 {'있음' if CARRY else '없음'} · 시드 {START}부터\n")
print(f"  {'모델':<20}{'이익':>8}{'±':>6}{'G1첫수':>8}{'G2+첫수':>9}{'최다비중':>9}"
      f"{'n':>5}{'폴백':>6}{'토큰/매치':>10}{'시간':>8}")
ts = [threading.Thread(target=run, args=c) for c in PLAN]
for t in ts: t.start()
for t in ts: t.join()
print("\nDONE")
