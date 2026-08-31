"""The carry-over manipulation where staying makes the counterparty better.

The numbers this project rests on — memory worth 411 at twelve rounds a game,
rising to 1205 at three — were all measured where one column of the ledger
decided the answer outright. On a board where the quote sheet alone gets 0.75 of
the way there, memory has much less left to be worth, and whether the effect
survives at all is the question.

Same three conditions as before. `BoardOnlyBuyer` never reads the record, so its
choice channel must come out at exactly zero however much memory it is handed;
that is the design's own null, not an estimate.
"""
import sys, json, statistics as st
sys.path.insert(0, 'src')
from carryover.game.buyers3 import BoardOnlyBuyer, EVBuyer, borne
from carryover.game.match import Match
from carryover.game.table4 import VALUE, cast_for

LOSS, ROUNDS, GAMES, STEP, N = 110, 12, 4, 10, 500
CONDITIONS = [('없음', False, True), ('구매자만', True, False), ('양쪽', True, True)]


def cell(buyer, carry, smem):
    prof, gk, g1 = [], [], []
    for s in range(N):
        make, key, ev, board, hidden = cast_for(s, loss=LOSS)
        r = Match(buyer_factory=buyer, seller_factory=make, games=GAMES,
                  rounds=ROUNDS, loss=LOSS, value=VALUE, seed=s,
                  carry_over=carry, seller_memory=smem, step=STEP).run()
        if not r.rounds:
            continue
        prof.append(r.profit)
        played = [g for g in r.games if g]
        g1.append(int(played[0][0].seller == key))
        if len(played) > 1:
            gk.append(sum(int(g[0].seller == key) for g in played[1:]) / (len(played) - 1))
    m = lambda v: st.mean(v) if v else float('nan')
    se = lambda v: (st.stdev(v) / len(v) ** 0.5) if len(v) > 1 else 0.0
    return dict(profit=m(prof), se=se(prof), g1=m(g1), gk=m(gk), gk_se=se(gk))


out = {}
print(f"3번 판 · L={LOSS} · {N}매치 · 정답은 참값 기준 최적 셀러\n")
print(f"  {'구매자':<12}{'기억':<10}{'이익':>8}{'±':>6}{'G1첫수':>8}{'G2+첫수':>9}{'±':>6}")
for lbl, B in (('호가판만', BoardOnlyBuyer), ('원장+호가판', EVBuyer)):
    for cl, carry, smem in CONDITIONS:
        r = cell(B, carry, smem)
        out[f"{lbl}|{cl}"] = r
        print(f"  {lbl:<12}{cl:<10}{r['profit']:>8.0f}{r['se']:>6.0f}"
              f"{r['g1']:>8.2f}{r['gk']:>9.2f}{r['gk_se']:>6.2f}", flush=True)

g = lambda k: out[f"원장+호가판|{k}"]['profit']
c = lambda k: out[f"호가판만|{k}"]['profit']
print(f"\n  원장을 읽는 구매자")
print(f"    선택 채널   (구매자만 − 없음)  {g('구매자만') - g('없음'):>+7.0f}")
print(f"    판매자 채널 (양쪽 − 구매자만)  {g('양쪽') - g('구매자만'):>+7.0f}")
print(f"    합계        (양쪽 − 없음)      {g('양쪽') - g('없음'):>+7.0f}")
print(f"  호가판만 읽는 구매자 (설계상 선택 채널이 0이어야 함)")
print(f"    선택 채널                      {c('구매자만') - c('없음'):>+7.0f}")
print(f"    판매자 채널                    {c('양쪽') - c('구매자만'):>+7.0f}")
print(f"\n  3번 판(관계가 정보뿐): 합계 +28 · 1번 판(조작됨): +412")
json.dump(out, open('data_carry4.json', 'w'), indent=1)
