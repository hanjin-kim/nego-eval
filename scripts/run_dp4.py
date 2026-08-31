"""The actual optimum, by dynamic programming — not a greedy proxy for it.

The "oracle" that put the ceiling at 1034 looked one round ahead: it valued a
purchase by the concession that purchase would unlock, and never considered
paying now to open something later. That is a lower bound on the optimum, and
using it to claim there is no headroom would be claiming a limit of my own
policy.

The optimum here is computable. A seller's concession depends only on this
buyer's share of the last `window` rounds, delivery rates are fixed and
published, and prices are a fixed sequence per seed. So the state is (round,
the last eight picks) — 3^8 x 48, about three hundred thousand — and the value
function solves exactly:

    V(t, w) = max over sellers of  r (value - p) - (1 - r) (loss - concession)
                                   + V(t + 1, w')

Failure is stochastic but its probability is known, so the expectation is taken
rather than sampled. Nothing here is reachable by an agent — it needs the hidden
traits and the price sequence — which is what makes it a bound.
"""
import sys, json, statistics as st
from functools import lru_cache
sys.path.insert(0, 'src')
from nego_eval.game.table4 import VALUE, cast_for

LOSS, ROUNDS, GAMES, STEP, WINDOW, COOLOFF = 110, 12, 4, 10, 8, 3
N = 120
NAMES = ('A', 'B', 'C')
IDX = {n: i for i, n in enumerate(NAMES)}


def snap(x, step=STEP):
    return int(x // step + (1 if x % step * 2 >= step else 0)) * step


def solve(seed, carry=True):
    make, key, ev, board, hidden = cast_for(seed, loss=LOSS)
    sellers = {x.name: x for x in make()}
    T = GAMES * ROUNDS

    # Prices: every seller quotes every round, so the draw order is fixed.
    prices = {n: [] for n in NAMES}
    for g in range(GAMES):
        fresh = {x.name: x for x in make()}
        for t in range(ROUNDS):
            for n in NAMES:
                prices[n].append(snap(fresh[n].quote(t, ROUNDS - t)))

    margin = {n: sellers[n].base_price - sellers[n].cost for n in NAMES}
    cap, floor, care = 0.80, {}, {}
    for n in NAMES:
        floor[n], care[n] = hidden[n]['floor'], hidden[n]['care']

    def concession(n, window, t):
        """What the seller pays when pressed for the whole loss."""
        remaining = ROUNDS - (t % ROUNDS)
        loyalty = sum(1 for w in window if w == IDX[n]) / WINDOW
        earned = min(cap, floor[n] + care[n] * loyalty)
        standing = 0.4 + 0.6 * loyalty
        worth = margin[n] * min(COOLOFF, max(remaining, 0)) * standing
        return snap(min(int(round(LOSS * earned)), int(worth)))

    @lru_cache(maxsize=None)
    def V(t, window):
        if t >= T:
            return 0.0
        # Without carry-over the window is cleared at every game boundary.
        w = window if (carry or t % ROUNDS) else ()
        best = None
        for n in NAMES:
            r = sellers[n].reliability
            paid = concession(n, w, t)
            nxt = (w + (IDX[n],))[-WINDOW:]
            here = r * (VALUE - prices[n][t]) - (1 - r) * (LOSS - paid)
            v = here + V(t + 1, nxt)
            if best is None or v > best:
                best = v
        return best

    return V(0, ())


print(f"4번 판 · 시드 {N}개 · 동적계획법 정확해\n")
for carry, lbl in ((True, '이월 있음'), (False, '이월 없음')):
    vals = [solve(s, carry) for s in range(N)]
    print(f"  최적 (DP) · {lbl:<8}{st.mean(vals):>8.0f}  ±{st.stdev(vals)/len(vals)**0.5:>4.0f}")
print(f"\n  비교 · 근시안 오라클 1034 · 스크립트 996 · 호가판만 934")
print(f"        실측 · gemini 979 · gpt 927 · haiku 663 · deepseek 610")
