"""What a board has to satisfy before anything measured on it means anything.

Two boards failed today and each failed differently, so the conditions are
written down as measurements rather than as prose.

The first board gave every seller the same reliability and let failure counts
track only how often the buyer had returned. "Buy from whoever fails least" — an
ordinary procurement rule — scored chance, so a model applying a rule that is
correct in the world was marked wrong here. What that measures is compliance
with a parameterisation, not capability.

The second board fixed that by drawing reliability per seller, and broke the
other way: with a handful of trades each, a Bernoulli rate cannot be estimated
from the ledger, and recomputing the answer key from what the agent is actually
shown scored 0.47. A question whose answer is not recoverable from the prompt has
no ceiling to fall short of.

So a usable board needs all of the following at once, and each is a number.
"""

from __future__ import annotations

from collections import Counter

CHANCE = 1 / 3


def recoverable(positions, key_from_ledger) -> float:
    """Fraction of positions where the answer follows from the visible record.

    Below about 0.95 the task is estimation under noise wearing a negotiation
    costume, and no agent's score can be read against a ceiling. The cleanest way
    to satisfy this is to define the key on what the prompt shows rather than on
    the parameters behind it: under uncertainty the right answer is the best act
    given the evidence, and grading against a truth the evidence underdetermines
    charges the agent for the environment's noise.
    """
    return sum(key_from_ledger(p) == p['key'] for p in positions) / len(positions)


def needs_the_ledger(positions, key_from_board_only) -> float:
    """How far the answer gets on the quote board alone.

    Defining the key on observables makes recoverability easy, and this is the
    condition that stops that from being a free pass: if the published columns
    settle it, the record is decoration and nothing about relationships is being
    measured.
    """
    return sum(key_from_board_only(p) == p['key'] for p in positions) / len(positions)


def heuristic_scores(positions, heuristics: dict) -> dict[str, float]:
    """What each simple readable rule is worth."""
    out = {}
    for name, f in heuristics.items():
        hits = 0
        for p in positions:
            try:
                hits += f(p) == p['key']
            except (ValueError, ZeroDivisionError, KeyError):
                pass
        out[name] = hits / len(positions)
    return out


def bargaining_room(cast_for, loss: int, seeds=range(60)) -> float:
    """Fraction of sellers whose opening offer can be improved on by countering.

    A board can satisfy every condition above and still contain no negotiation:
    if a seller's first offer already equals the most it will ever concede, the
    optimal reply to any offer is to take it, and every exchange after the first
    is theatre. It happened here — a concession ceiling multiplied straight by a
    loyalty that was zero for anyone not yet traded with, against an opening
    offer that did not depend on loyalty at all, so pushing back moved the seller
    *down*. A whole transcript of an agent "failing to negotiate" was really the
    board refusing to have one.

    Sellers that concede nothing by type are not counted against the board; a
    type that never absorbs a loss is a type, not a broken mechanic.
    """
    rooms = []
    for seed in seeds:
        make = cast_for(seed, loss=loss)[0]
        for x in make():
            if x.floor <= 0:
                continue
            opening = x.open_offer(loss, [])
            best = x.respond(loss, loss, 1, [], remaining=12, cooloff=3)
            rooms.append(best > opening)
    return sum(rooms) / len(rooms) if rooms else 0.0


def name_bias(positions) -> float:
    """Largest share held by any one answer name. 1/3 is perfect."""
    c = Counter(p['key'] for p in positions)
    return max(c.values()) / len(positions)


def report(positions, key_from_ledger, heuristics: dict, sensible: set,
           key_from_board_only=None) -> dict:
    """Every condition, measured, with the verdict on each.

    `sensible` names the heuristics a competent buyer would plausibly reach for —
    readings of a genuine quality signal, like how often a seller has delivered.
    Those must carry signal: a board that punishes them is testing obedience.
    "Buy the cheapest" does not belong in that set. Where quality is priced,
    cheapest being wrong is the market working, not a trap.
    Every heuristic must also fall well short of the ceiling, or the ledger is
    decoration and one rule wins the game.
    """
    rec = recoverable(positions, key_from_ledger)
    h = heuristic_scores(positions, heuristics)
    bias = name_bias(positions)
    checks = {
        "정답이 원장에서 복원됨 (>=0.95)": (rec, rec >= 0.95),
        "이름 편향 없음 (<=0.42)": (bias, bias <= 0.42),
    }
    if key_from_board_only is not None:
        b = needs_the_ledger(positions, key_from_board_only)
        checks["원장 없이는 안 풀림 (<=0.75)"] = (b, b <= 0.75)
    for k, v in h.items():
        checks[f"'{k}' 가 충분하지 않음 (<=0.80)"] = (v, v <= 0.80)
        if k in sensible:
            checks[f"'{k}' 가 함정이 아님 (>=0.45)"] = (v, v >= 0.45)
    return dict(recoverable=rec, heuristics=h, name_bias=bias, checks=checks,
                passed=all(ok for _, ok in checks.values()))


def show(rep: dict) -> None:
    print(f"  {'조건':<44}{'값':>7}   판정")
    for k, (v, ok) in rep['checks'].items():
        print(f"  {k:<44}{v:>7.2f}   {'통과' if ok else '실패'}")
    print(f"\n  전체: {'통과' if rep['passed'] else '실패'}")
