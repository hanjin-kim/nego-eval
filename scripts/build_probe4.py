"""Frozen positions on the board where staying changes the counterparty.

The prompt now carries a quote board with a delivery rate beside every price, so
the agent is not estimating reliability at all — it is deciding what reliability
is worth against what a seller does when the delivery fails, which is the only
thing it has to learn by trading.
"""
import sys, json, random
sys.path.insert(0, 'src')
from nego_eval.game.contract import report, show
from nego_eval.game.table4 import VALUE, cast_for, expected, price_of
from nego_eval.sim.agents import LLMBuyer
from nego_eval.sim.world import World

LOSS, ROUNDS, GAMES, STEP, N = 110, 12, 4, 10, 140
fmt = LLMBuyer()


class Explorer:
    """Spreads its business so that every seller ends up with a record."""

    def __init__(self, seed):
        self.rng = random.Random(seed)

    def choose(self, quotes, t, remaining, history):
        return self.rng.choice(quotes).seller

    def bargain(self, loss, offer, r, max_rounds, seller_name, history, terms):
        # Always press for the whole loss: the seller then concedes exactly its
        # floor, so one failure states the trait outright instead of hinting.
        return (r == max_rounds), loss


positions = []
for s in range(N):
    make, key, ev, board, hidden = cast_for(s, loss=LOSS)
    prior = []
    for g in range(GAMES):
        if g > 0 and prior:
            stats = {}
            for o in prior:
                a = stats.setdefault(o.seller, dict(n=0, fail=0, paid=0, owed=0))
                a['n'] += 1
                if o.failed:
                    a['fail'] += 1; a['paid'] += o.seller_share; a['owed'] += o.loss
            if len(stats) == 3 and all(a['owed'] for a in stats.values()):
                # The key is the best act given the evidence shown, not the best
                # act under traits the evidence underdetermines.
                def borne_of(k, st=stats):
                    a = st[k]
                    return a['paid'] / a['owed'] if a['owed'] else 0.0
                obs_key = max(board, key=lambda k: (
                    board[k]['rate'] * (VALUE - board[k]['price'])
                    - (1 - board[k]['rate']) * LOSS * (1 - borne_of(k))))
                positions.append(dict(
                    match=s, game=g, key=obs_key, true_key=key,
                    board=board, stats=stats,
                    ev={k: round(v, 1) for k, v in ev.items()},
                    ledger=fmt._ledger(prior),
                    cheapest=min(board, key=lambda k: board[k]['price'])))
        prior = prior + World(buyer=Explorer(s * 10 + g), sellers=make(),
                              value=VALUE, loss=LOSS, rounds=ROUNDS,
                              seed=s * 1000 + g, step=STEP, prior=list(prior)).run()

json.dump(positions, open('data_probe4_positions.json', 'w'), indent=1)


def borne(p, k):
    a = p['stats'][k]
    return a['paid'] / a['owed'] if a['owed'] else 0.0


def ev_from_prompt(p):
    """The key, recomputed from the quote board plus the ledger."""
    return max(p['board'], key=lambda k: (
        p['board'][k]['rate'] * (VALUE - p['board'][k]['price'])
        - (1 - p['board'][k]['rate']) * LOSS * (1 - borne(p, k))))


H = {
    '공시 배달률 최고': lambda p: max(p['board'], key=lambda k: p['board'][k]['rate']),
    '손실 부담률 최고': lambda p: max(p['board'], key=lambda k: borne(p, k)),
    '관측 실패율 최저': lambda p: min(p['stats'], key=lambda k: p['stats'][k]['fail'] / p['stats'][k]['n']),
    '최저가': lambda p: p['cheapest'],
}
print(f"위치 {len(positions)}개 · 매치 {N} · 우연 0.33\n")
board_only = lambda p: max(p['board'], key=lambda k: (
    p['board'][k]['rate'] * (VALUE - p['board'][k]['price'])
    - (1 - p['board'][k]['rate']) * LOSS))
rep = report(positions, ev_from_prompt, H,
             sensible={'공시 배달률 최고', '관측 실패율 최저'},
             key_from_board_only=board_only)
from nego_eval.game.contract import bargaining_room
import nego_eval.game.table4 as T4
room = bargaining_room(lambda seed, loss: T4.cast_for(seed, loss)[:1], LOSS)
rep['checks']['받아쳐서 얻을 것이 있음 (>=0.95)'] = (room, room >= 0.95)
rep['passed'] = all(ok for _, ok in rep['checks'].values())
agree = sum(p['key'] == p['true_key'] for p in positions) / len(positions)
print(f"  (참고) 관측 기준 정답과 참값 기준 정답이 일치하는 비율 {agree:.2f}\n")
show(rep)
json.dump(rep['heuristics'] | {'recoverable': rep['recoverable']},
          open('data_probe4_baselines.json', 'w'), indent=1)
