"""What is memory worth when the market already prices what it can see?

Reported as a single number, "carry-over is worth 72" is a statement about the
one premium I happened to pick — and I picked it at the edge of what the contract
allowed, which is a market that underprices published quality. There, buying the
best number on the sheet is most of the job and there is little left for a record
to add.

The premium is exactly how well the market prices the quality it can observe. So
sweep it. What should appear, if relational contracting is about the residual, is
that the value of remembering rises as the observable stops being a bargain.
"""
import sys, json, statistics as st
sys.path.insert(0, 'src')
import nego_eval.game.table4 as T
from nego_eval.game.buyers import BoardOnlyBuyer, EVBuyer
from nego_eval.game.match import Match

LOSS, ROUNDS, GAMES, STEP, N = 110, 12, 4, 10, 400


def cell(buyer, carry, smem):
    prof, gk = [], []
    for s in range(N):
        make, key, ev, board, hidden = T.cast_for(s, loss=LOSS)
        r = Match(buyer_factory=buyer, seller_factory=make, games=GAMES,
                  rounds=ROUNDS, loss=LOSS, value=T.VALUE, seed=s,
                  carry_over=carry, seller_memory=smem, step=STEP).run()
        if not r.rounds:
            continue
        prof.append(r.profit)
        played = [g for g in r.games if g]
        if len(played) > 1:
            gk.append(sum(int(g[0].seller == key) for g in played[1:]) / (len(played) - 1))
    return st.mean(prof), (st.stdev(prof) / len(prof) ** 0.5), st.mean(gk)


rows = []
print(f"  {'프리미엄':>8}{'호가판만':>10}{'원장·기억X':>12}{'원장·기억O':>12}"
      f"{'  기억의 값':>11}{'  원장의 값':>11}{'G2+첫수':>9}")
for prem in (90, 105, 120, 135, 150, 165):
    T.PREMIUM = prem
    b, _, _ = cell(BoardOnlyBuyer, True, True)
    off, _, _ = cell(EVBuyer, False, True)
    on, _, gk = cell(EVBuyer, True, True)
    rows.append(dict(premium=prem, board=b, ledger_off=off, ledger_on=on, gk=gk))
    print(f"  {prem:>8}{b:>10.0f}{off:>12.0f}{on:>12.0f}"
          f"{on-off:>11.0f}{on-b:>11.0f}{gk:>9.2f}", flush=True)
json.dump(rows, open('data_premium_sweep.json', 'w'), indent=1)
print(f"\n  '기억의 값' = 같은 정책에 이월만 켠 차이")
print(f"  '원장의 값' = 원장을 읽는 정책이 호가판만 읽는 정책보다 나은 정도")
