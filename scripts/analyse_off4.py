"""Where does the model's surplus come from — inside a game, or across the bell?

A model can capture the value of reading a record without ever carrying one: a
twelve-round game is long enough to learn a seller from scratch. The two arms run
identical seeds and differ only in whether the ledger survives the boundary, so
the difference between them is the part that is about relationships rather than
about within-game learning.

References are replayed under each arm's own condition, and per seed, so neither
the board nor the condition leaks into the comparison.
"""
import sys, json, glob, statistics as st
sys.path.insert(0, 'src')
from nego_eval.game.buyers import BoardOnlyBuyer, EVBuyer
from nego_eval.game.match import Match
from nego_eval.game.table4 import VALUE, cast_for

LOSS, ROUNDS, GAMES, STEP = 110, 12, 4, 10
_c = {}


def ref(factory, carry, seed, tag):
    if (tag, carry, seed) not in _c:
        make = cast_for(seed, loss=LOSS)[0]
        r = Match(buyer_factory=factory, seller_factory=make, games=GAMES,
                  rounds=ROUNDS, loss=LOSS, value=VALUE, seed=seed,
                  carry_over=carry, step=STEP).run()
        _c[(tag, carry, seed)] = r.profit if r.rounds else None
    return _c[(tag, carry, seed)]


def load(pattern):
    out = {}
    for f in glob.glob(pattern):
        d = json.load(open(f))
        per = {int(k): v for k, v in (d.get('per_seed') or {}).items()}
        if per:
            out[d['model'].split('/')[-1]] = per
    return out


on = load('data_f4_[!o]*.json')
off = load('data_f4_off_*.json')
both = sorted(set(on) & set(off))
print("이월 있음 / 없음, 같은 시드에서\n")
print(f"  {'모델':<20}{'이월O':>8}{'이월X':>8}{'차이':>8}{'±':>6}{'n':>4}")
for m in both:
    shared = sorted(set(on[m]) & set(off[m]))
    if len(shared) < 3:
        print(f"  {m:<20}  공유 시드 {len(shared)}개 — 부족")
        continue
    d = [on[m][s] - off[m][s] for s in shared]
    print(f"  {m:<20}{st.mean(on[m][s] for s in shared):>8.0f}"
          f"{st.mean(off[m][s] for s in shared):>8.0f}{st.mean(d):>8.0f}"
          f"{st.stdev(d)/len(d)**0.5 if len(d)>1 else 0:>6.0f}{len(shared):>4}")
    seeds = shared
    for nm, fac in (('호가판만', lambda: BoardOnlyBuyer()), ('원장', lambda: EVBuyer())):
        a = st.mean(ref(fac, True, s, nm) for s in seeds)
        b = st.mean(ref(fac, False, s, nm) for s in seeds)
        print(f"      기준 {nm:<8}{a:>8.0f}{b:>8.0f}{a-b:>8.0f}")
if not both:
    print("  아직 두 조건이 모두 있는 모델이 없습니다.")
