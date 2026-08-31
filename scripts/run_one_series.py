"""One match, printed in full, with what the model said as it went.

Aggregates hide the thing you most want to see in a negotiation environment:
what the agent actually did, round by round, and what it told itself it was
doing. No statistics here — a single series, seed fixed, every move shown.
"""
import sys, json
sys.path.insert(0, 'src')
from carryover.game.table3 import VALUE, cast_for
from carryover.sim.agents import LLMBuyer
from carryover.sim.world import World

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'qwen/qwen3.8-max'
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 900001
LOSS, ROUNDS, GAMES, STEP = 110, 12, 4, 10

make, key, ev, board = cast_for(SEED, loss=LOSS)
print(f"모델 {MODEL} · 시드 {SEED} · {GAMES}게임 x {ROUNDS}라운드 · 손실 {LOSS}\n")
print("호가판 (매 라운드 가격은 ±12 흔들림)")
for n in sorted(board):
    tag = '  ← 기대이익 최대' if n == key else ''
    print(f"  {n}: 정가 {board[n]['price']}, 공시 배달률 {board[n]['rate']:.2f}"
          f", 기대이익 {ev[n]:+.1f}{tag}")
print()

prior, totals = [], []
for g in range(GAMES):
    buyer = LLMBuyer(model=MODEL)
    h = World(buyer=buyer, sellers=make(), value=VALUE, loss=LOSS, rounds=ROUNDS,
              seed=SEED * 1000 + g, step=STEP, prior=list(prior)).run()
    why = {e['t']: str(e['raw'].get('why', ''))[:64]
           for e in buyer.log if e['kind'] == 'choose'}
    print(f"── 게임 {g+1} " + "─" * 56)
    for o in h:
        mark = '*' if o.seller == key else ' '
        if o.failed:
            print(f" {mark}R{o.t+1:>2} {o.seller} @{o.price:>3}  실패  "
                  f"셀러 {o.seller_share:>3} / 나 {o.buyer_share:>3}"
                  f"  {o.buyer_profit:>+5}   {why.get(o.t,'')}")
            if o.transcript:
                print(f"        {' | '.join(o.transcript)}")
        else:
            print(f" {mark}R{o.t+1:>2} {o.seller} @{o.price:>3}  납품 "
                  f"              {o.buyer_profit:>+5}   {why.get(o.t,'')}")
    tot = sum(x.buyer_profit for x in h)
    totals.append(tot)
    share = sum(1 for x in h if x.seller == key) / len(h)
    print(f"  게임 이익 {tot:+}  ·  최적 셀러 비중 {share:.0%}"
          f"  ·  첫 수 {'적중' if h[0].seller == key else h[0].seller}"
          f"  ·  폴백 {buyer.fallbacks}\n", flush=True)
    prior = prior + h

print(f"매치 합계 {sum(totals):+}   게임별 {totals}")
print(f"(* 표시가 기대이익 최대 셀러)")
