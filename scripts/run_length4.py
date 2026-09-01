"""Can the episode be halved without losing what it measures?

Training cost here is not the number of calls but their depth: forty-eight
rounds is about seventy calls that have to happen in order, and nothing inside an
episode parallelises. Halving the match halves the depth.

What must survive the cut is the thing the environment exists to measure. The
shape sweep already suggested it might survive comfortably — the shorter the
games, the more carrying the ledger is worth, because a relationship keeps being
interrupted before it has paid for itself. So this sweeps total rounds while
holding the shape that made memory matter most, and asks the same two questions
at every length: what is memory worth, and does the board still separate a policy
that reads the record from one that does not.
"""
import sys, json, statistics as st
sys.path.insert(0, 'src')
from nego_eval.game.buyers import BoardOnlyBuyer, EVBuyer
from nego_eval.game.match import Match
from nego_eval.game.table4 import VALUE, cast_for

LOSS, STEP, N = 110, 10, 400
#: (rounds per game, games). Total rounds falls from 48 to 12; the games stay
#: short, which is where the boundary effect lives.
SHAPES = [(6, 8), (6, 6), (6, 4), (4, 6), (4, 4), (3, 4), (2, 6)]


def cell(buyer, rounds, games, carry):
    """Returns the per-seed scores, not just their mean.

    Cost is decided by statistical power, not by effect size: a shorter episode
    has a smaller effect and a smaller spread, and which shrinks faster is the
    only thing that matters for how many rollouts a training run needs.
    """
    prof, gk = [], []
    for s in range(N):
        make, key, ev, board, hidden = cast_for(s, loss=LOSS)
        r = Match(buyer_factory=buyer, seller_factory=make, games=games,
                  rounds=rounds, loss=LOSS, value=VALUE, seed=s,
                  carry_over=carry, step=STEP).run()
        if not r.rounds:
            continue
        prof.append(r.profit)
        played = [g for g in r.games if g]
        if len(played) > 1:
            gk.append(sum(int(g[0].seller == key) for g in played[1:]) / (len(played) - 1))
    return prof, (st.mean(gk) if gk else float('nan'))


print(f"L={LOSS} · {N}매치 · 라운드 수를 줄이면 학습 비용이 그만큼 줄어든다\n")
print(f"  {'모양':<9}{'라운드':>7}{'호출':>7}{'기억의값':>9}{'쌍SD':>7}"
      f"{'점수대비':>8}{'2σ쌍수':>8}{'총호출':>10}{'G2+':>7}")
rows = []
for R, Gm in SHAPES:
    total = R * Gm
    bv, _ = cell(BoardOnlyBuyer, R, Gm, True)
    offv, _ = cell(EVBuyer, R, Gm, False)
    onv, gk = cell(EVBuyer, R, Gm, True)
    b, off, on = st.mean(bv), st.mean(offv), st.mean(onv)
    # paired: the same seeds, so the board cancels
    d = [a - c for a, c in zip(onv, offv)]
    sd = st.stdev(d)
    calls = round(total * 1.45)          # measured: about 70 calls for 48 rounds
    # episodes for the memory effect to clear two standard errors, and what that
    # costs in calls that have to happen one after another
    need = max(4, round((2 * sd / max(st.mean(d), 1e-9)) ** 2))
    rows.append(dict(rounds=R, games=Gm, total=total, calls=calls, board=b,
                     off=off, on=on, gk=gk, sd=sd, need=need,
                     budget=need * calls))
    print(f"  {f'{R}x{Gm}':<9}{total:>7}{calls:>7}{on-off:>9.0f}{sd:>7.0f}"
          f"{(on-off)/on:>8.1%}{need:>8}{need*calls:>10,}{gk:>7.2f}", flush=True)
json.dump(rows, open('data/length4.json', 'w'), indent=1)
print(f"\n  '총호출' = 기억 효과를 2시그마로 잡는 데 드는 순차 호출 수. 작을수록 싸다.")
