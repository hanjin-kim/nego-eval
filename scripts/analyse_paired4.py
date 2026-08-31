"""Model against reference on the same board, not on the same average."""
import sys, json, glob, statistics as st
sys.path.insert(0, 'src')
from nego_eval.game.buyers import BoardOnlyBuyer, EVBuyer
from nego_eval.game.match import Match
from nego_eval.game.table4 import VALUE, cast_for

LOSS, ROUNDS, GAMES, STEP = 110, 12, 4, 10
REFS = [('호가판만', lambda: BoardOnlyBuyer(), True),
        ('원장·기억X', lambda: EVBuyer(), False),
        ('원장·기억O', lambda: EVBuyer(), True)]
_cache = {}


def ref_on(name, factory, carry, seed):
    if (name, seed) not in _cache:
        make = cast_for(seed, loss=LOSS)[0]
        r = Match(buyer_factory=factory, seller_factory=make, games=GAMES,
                  rounds=ROUNDS, loss=LOSS, value=VALUE, seed=seed,
                  carry_over=carry, step=STEP).run()
        _cache[(name, seed)] = r.profit if r.rounds else None
    return _cache[(name, seed)]


rows = []
#: Prefer a finished run, fall back to its snapshot. A run stopped for budget
#: still has every completed match in the partial file, and dropping it would
#: throw away paid-for measurements.
finals = {x for x in glob.glob('data_f4_*.json') if '.partial.' not in x}
files = sorted(finals | {x for x in glob.glob('data_f4_*.partial.json')
                         if x.replace('.partial', '') not in finals})
for f in files:
    d = json.load(open(f))
    per = {int(k): v for k, v in (d.get('per_seed') or {}).items()}
    if len(per) < 3:
        continue
    seeds = sorted(per)
    row = dict(model=d['model'].split('/')[-1], n=len(seeds),
               raw=st.mean(per[s] for s in seeds), fallbacks=d.get('fallbacks', 0))
    for name, fac, carry in REFS:
        diffs = [per[s] - ref_on(name, fac, carry, s) for s in seeds
                 if ref_on(name, fac, carry, s) is not None]
        row[name] = (st.mean(diffs),
                     st.stdev(diffs) / len(diffs) ** 0.5 if len(diffs) > 1 else 0.0)
    rows.append(row)

if not rows:
    print("아직 짝지을 매치가 부족합니다."); raise SystemExit

print("같은 판에서의 차이 (모델 − 기준선). 양수면 모델이 나음.\n")
hdr = f"  {'모델':<20}{'n':>4}{'원점수':>9}"
for name, _, _ in REFS:
    hdr += f"{name:>20}"
print(hdr)
for r in rows:
    line = f"  {r['model']:<20}{r['n']:>4}{r['raw']:>9.0f}"
    for name, _, _ in REFS:
        m, se = r[name]
        line += f"{m:>13.0f}±{se:<6.0f}"
    print(line)
print(f"\n  폴백(호출 실패로 최저가 대체): "
      f"{ {r['model']: r['fallbacks'] for r in rows if r['fallbacks']} or '없음'}")
json.dump(rows, open('data_paired4.json', 'w'), indent=1, default=str)
