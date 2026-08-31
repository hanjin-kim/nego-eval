"""How much room is there above the best rule I wrote?

Two models now play this board indistinguishably from the scripted policy that
reads the record, which would be the end of the story if that policy were the
ceiling. It is not obviously anything of the kind: it is a rule I wrote, the
learned baseline scores below it, and neither knows what the sellers actually
are.

An oracle does. It is handed each seller's hidden floor and its hidden care trait
and plays the expectation exactly, including the part of the concession that only
loyalty unlocks. It cannot be reached by any agent — nothing in the prompt states
those traits — so it is an upper bound rather than a target. The gap between it
and the best real policy is the headroom, and headroom is the difference between
an environment worth training in and a leaderboard.
"""
import sys, json, statistics as st
sys.path.insert(0, 'src')
from carryover.game.buyers3 import BoardOnlyBuyer, EVBuyer, borne
from carryover.game.match import Match
from carryover.game.table4 import VALUE, cast_for

LOSS, ROUNDS, GAMES, STEP, N = 110, 12, 4, 10, 400


class Oracle:
    """Knows every hidden trait; still bound by the rules and the dice."""

    def __init__(self, hidden, window=8, cap=0.80):
        self.h, self.w, self.cap = hidden, window, cap

    def _earned(self, name, history):
        recent = history[-self.w:]
        loyalty = sum(1 for o in recent if o.seller == name) / self.w
        t = self.h[name]
        return min(self.cap, t['floor'] + t['care'] * loyalty)

    def choose(self, quotes, t, remaining, history):
        def ev(q):
            r = q.rate
            # What it would concede next time, after this purchase lifts loyalty.
            f = self._earned(q.seller, history + [type('o', (), {'seller': q.seller})()])
            return r * (VALUE - q.price) - (1 - r) * LOSS * (1 - f)
        return max(quotes, key=ev).seller

    def bargain(self, loss, offer, r, max_rounds, seller_name, history, terms):
        return (r == max_rounds), loss     # press for everything, never impasse


def score(factory, carry=True):
    prof, gk = [], []
    for s in range(N):
        make, key, ev, board, hidden = cast_for(s, loss=LOSS)
        r = Match(buyer_factory=lambda h=hidden: factory(h), seller_factory=make,
                  games=GAMES, rounds=ROUNDS, loss=LOSS, value=VALUE, seed=s,
                  carry_over=carry, step=STEP).run()
        if not r.rounds:
            continue
        prof.append(r.profit)
        played = [g for g in r.games if g]
        if len(played) > 1:
            gk.append(sum(int(g[0].seller == key) for g in played[1:]) / (len(played) - 1))
    return st.mean(prof), st.stdev(prof) / len(prof) ** 0.5, (st.mean(gk) if gk else float('nan'))


rows = [('오라클 (숨은 특성을 앎)', lambda h: Oracle(h)),
        ('원장+호가판 (스크립트)', lambda h: EVBuyer()),
        ('호가판만', lambda h: BoardOnlyBuyer())]
print(f"4번 판 · {N}매치 · 이월 있음\n")
print(f"  {'정책':<26}{'이익':>8}{'±':>6}{'G2+첫수':>9}")
out = {}
for name, f in rows:
    m, se, gk = score(f)
    out[name] = dict(profit=m, se=se, gk=gk)
    print(f"  {name:<26}{m:>8.0f}{se:>6.0f}{gk:>9.2f}", flush=True)
top = out['오라클 (숨은 특성을 앎)']['profit']
best = out['원장+호가판 (스크립트)']['profit']
print(f"\n  오라클 − 최고 스크립트 = {top - best:+.0f}   ← 학습으로 파고들 여유")
print(f"  참고 · 전체 플레이 실측: gemini 979 · gpt 927 · haiku 663 · deepseek 610")
json.dump(out, open('data_oracle4.json', 'w'), indent=1)
