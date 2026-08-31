"""How often the bell rings, where the relationship has to be built.

Carrying a ledger across games is worth little when a game is long enough to
rebuild it: at twelve rounds an agent sees three or four failures and has learned
what it needs before the bell. Total rounds are held at 48 and only the position
of the boundaries moves — which is not a knob invented for this note. Procurement
relationships differ enormously in how many transactions fall inside a contract
period, and that is exactly this parameter.
"""
import sys, json, statistics as st
sys.path.insert(0, 'src')
import nego_eval.game.table4 as T
from nego_eval.game.buyers import BoardOnlyBuyer, EVBuyer
from nego_eval.game.match import Match

LOSS, STEP, N = 110, 10, 400
SHAPES = [(12, 4), (8, 6), (6, 8), (4, 12), (3, 16), (2, 24)]


def cell(buyer, rounds, games, carry):
    prof, gk = [], []
    for s in range(N):
        make, key, ev, board, hidden = T.cast_for(s, loss=LOSS)
        r = Match(buyer_factory=buyer, seller_factory=make, games=games,
                  rounds=rounds, loss=LOSS, value=T.VALUE, seed=s,
                  carry_over=carry, step=STEP).run()
        if not r.rounds:
            continue
        prof.append(r.profit)
        played = [g for g in r.games if g]
        if len(played) > 1:
            gk.append(sum(int(g[0].seller == key) for g in played[1:]) / (len(played) - 1))
    return st.mean(prof), (st.mean(gk) if gk else float('nan'))


print(f"프리미엄 {T.PREMIUM} · 총 48라운드 고정 · {N}매치\n")
print(f"  {'라운드x게임':<12}{'경계수':>7}{'호가판만':>10}{'원장·기억X':>12}"
      f"{'원장·기억O':>12}{'  기억의 값':>11}{'G2+첫수':>9}")
rows = []
for R, G in SHAPES:
    b, _ = cell(BoardOnlyBuyer, R, G, True)
    off, _ = cell(EVBuyer, R, G, False)
    on, gk = cell(EVBuyer, R, G, True)
    rows.append(dict(rounds=R, games=G, board=b, off=off, on=on, gk=gk))
    print(f"  {f'{R}x{G}':<12}{G-1:>7}{b:>10.0f}{off:>12.0f}{on:>12.0f}"
          f"{on-off:>11.0f}{gk:>9.2f}", flush=True)
json.dump(rows, open('data_shape4.json', 'w'), indent=1)
