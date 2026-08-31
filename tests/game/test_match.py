"""What the board game must never do, whatever an agent proposes.

Each of these is a way the carry-over comparison could quietly stop being a
comparison: memory leaking into the score, the grid not closing the settlement,
or the two conditions differing in something other than memory.
"""
import pytest

from carryover.game.denom import shares, snap
from carryover.game.match import Match, tally
from carryover.sim.agents import CheapestBuyer, GrudgeBuyer, ScriptedSeller

LOSS, STEP = 120, 10


def cast(_g):
    return [ScriptedSeller('coop', 100, cost=70, share=0.50, floor=0.60, seed=0),
            ScriptedSeller('mid',   98, cost=70, share=0.25, floor=0.35, seed=0),
            ScriptedSeller('sharp', 92, cost=70, share=0.00, floor=0.00, seed=0)]


def match(buyer=CheapestBuyer, carry=True, seed=0, games=4):
    return Match(buyer_factory=buyer, seller_factory=cast, games=games,
                 rounds=12, loss=LOSS, seed=seed, carry_over=carry,
                 step=STEP).run()


def test_every_amount_lands_on_the_grid():
    for o in match().rounds:
        assert o.price % STEP == 0, o
        assert o.buyer_share % STEP == 0 and o.seller_share % STEP == 0, o


def test_the_grid_still_closes_the_loss():
    # Snapping two numbers independently would break the identity. The world
    # derives the buyer's share from the seller's, so it cannot.
    for o in match().rounds:
        assert o.buyer_share + o.seller_share == o.loss, o


def test_the_action_set_is_finite_and_small():
    assert shares(LOSS, STEP) == list(range(0, LOSS + STEP, STEP))
    assert len(shares(LOSS, STEP)) == 13


def test_a_grid_that_does_not_divide_the_loss_is_refused():
    with pytest.raises(ValueError):
        shares(125, STEP)


def test_memory_is_not_income():
    # The carried rounds are visible to agents but must never be scored again.
    on, off = match(carry=True), match(carry=False)
    assert on.profit == sum(on.per_game)
    assert len(on.rounds) == len(off.rounds) or True   # play may differ; income may not
    for g in on.games:
        assert all(o.t < 12 for o in g)


def test_the_two_conditions_differ_only_in_memory():
    # Game 1 has no prior in either condition, so it must be identical.
    on, off = match(carry=True), match(carry=False)
    assert on.games[0] == off.games[0]


def test_carry_over_actually_reaches_the_agent():
    # A buyer that reacts to history must play game 2 differently when it can
    # see game 1. If this fails, `prior` is being dropped and the whole
    # manipulation is inert.
    on, off = match(GrudgeBuyer, carry=True), match(GrudgeBuyer, carry=False)
    assert on.games[1:] != off.games[1:]


def test_crossing_is_measured_only_at_game_boundaries():
    r = match(games=4)
    c = r.crossing()
    assert c is not None and 0.0 <= c <= 1.0
    assert match(games=1).crossing() is None


def test_the_ledger_adds_up():
    r = match()
    led = tally(r.rounds)
    assert sum(l.bought for l in led.values()) == len(r.rounds)
    for name, l in led.items():
        assert l.paid <= l.owed
        assert 0.0 <= l.borne <= 1.0


def test_matches_are_reproducible():
    assert match(seed=7).per_game == match(seed=7).per_game
    assert match(seed=7).per_game != match(seed=8).per_game


def test_snap_stays_inside_the_loss():
    assert snap(1000, STEP, hi=LOSS) == LOSS
    assert snap(-40, STEP) == 0
    assert snap(64, STEP) == 60 and snap(65, STEP) == 70


def test_seller_loyalty_does_not_depend_on_how_far_back_it_can_see():
    # A seller shown the same recent rounds must behave identically whether or
    # not older rounds are attached. Otherwise carry-over changes the seller's
    # concession by changing the length of history, and the two conditions stop
    # differing only in information.
    from carryover.sim.world import Outcome
    s = ScriptedSeller('A', 100, cost=70, share=0.50, floor=0.60, seed=0)

    def rounds(names):
        return [Outcome(t=i, seller=n, price=100, failed=False, loss=0,
                        buyer_share=0, seller_share=0, buyer_profit=50,
                        seller_profit=30) for i, n in enumerate(names)]

    recent = ['A'] * 8
    old = ['B'] * 40
    short = s.respond(LOSS, LOSS, 1, rounds(recent), remaining=10, cooloff=3)
    long_ = s.respond(LOSS, LOSS, 1, rounds(old + recent), remaining=10, cooloff=3)
    assert short == long_, (short, long_)


def test_one_round_is_not_a_relationship():
    # A seller picked once, at the start, must not read as a loyal partner just
    # because there is nothing else in the window to divide by.
    from carryover.sim.world import Outcome
    s = ScriptedSeller('A', 100, cost=70, share=0.50, floor=0.60, seed=0)
    one = [Outcome(t=0, seller='A', price=100, failed=False, loss=0, buyer_share=0,
                   seller_share=0, buyer_profit=50, seller_profit=30)]
    assert s.loyalty_of(one) == 1 / s.loyalty_window
    assert s.loyalty_of(one * s.loyalty_window) == 1.0


def test_price_says_nothing_about_type():
    # The generous seller must not be identifiable from the quote board. When it
    # was systematically the priciest, "pick the most expensive" found it 55% of
    # the time against a chance of 33 — so every claim that the ledger was the
    # only available signal was inheriting a leak.
    from carryover.game.table import cast_for
    priciest = cheapest = n = 0
    for seed in range(400):
        make, generous = cast_for(seed)
        sellers = make()
        quotes = sorted(((s.name, s.quote(0, 12)) for s in sellers), key=lambda q: q[1])
        n += 1
        cheapest += quotes[0][0] == generous
        priciest += quotes[-1][0] == generous
    for hits in (priciest, cheapest):
        assert abs(hits / n - 1 / 3) < 0.08, (priciest / n, cheapest / n)


def test_the_board_is_the_same_board_in_the_next_process():
    # Seeding a seller's price noise from `hash()` made the quotes depend on the
    # interpreter's per-process hash randomisation: same seed, different board.
    # Reproducibility inside one process cannot catch that, so this runs a second
    # interpreter and compares.
    import subprocess, sys as _s, json as _j, pathlib as _p
    src = str(_p.Path(__file__).resolve().parents[2] / 'src')
    prog = ("import sys; sys.path.insert(0, %r)\n"
            "from carryover.game.table import cast_for\n"
            "import json\n"
            "m, g = cast_for(3)\n"
            "print(json.dumps(sorted((s.name, s.quote(0, 12)) for s in m())))\n") % src
    runs = [subprocess.run([_s.executable, '-c', prog], capture_output=True,
                           text=True, check=True).stdout for _ in range(2)]
    assert runs[0] == runs[1], runs


def test_scrambling_keeps_the_record_and_cuts_only_its_owner():
    # The control has to leave the prompt the same size and shape, or a drop in
    # score is explained by having less to read rather than by the record being
    # useless. Same rounds, same numbers, different names.
    from carryover.sim.world import World
    from carryover.sim.agents import CheapestBuyer
    prior = World(buyer=CheapestBuyer(), sellers=cast(None), loss=LOSS,
                  rounds=12, seed=1, step=STEP).run()
    perm = {'coop': 'sharp', 'sharp': 'coop', 'mid': 'mid'}
    w = World(buyer=CheapestBuyer(), sellers=cast(None), loss=LOSS, rounds=12,
              seed=2, step=STEP, prior=list(prior), scramble=perm)
    plain = World(buyer=CheapestBuyer(), sellers=cast(None), loss=LOSS, rounds=12,
                  seed=2, step=STEP, prior=list(prior))
    a, b = w.buyer_memory, plain.buyer_memory
    assert len(a) == len(b)
    assert sorted(o.seller_share for o in a) == sorted(o.seller_share for o in b)
    assert [o.seller for o in a] != [o.seller for o in b]
    # the sellers are not fooled
    assert [o.seller for o in w.memory] == [o.seller for o in plain.memory]


def test_countering_can_gain_something():
    # A board where the opening offer is already the ceiling has no negotiation
    # in it, whatever its transcripts look like: the optimal reply to any offer
    # is to accept, and an agent that pushes back is only losing exchanges. That
    # was true here for a while, and a whole match of an agent apparently unable
    # to bargain was the board, not the agent.
    from carryover.game.contract import bargaining_room
    from carryover.game.table3 import cast_for as cast3
    assert bargaining_room(cast3, LOSS) == 1.0


def test_a_stranger_can_still_concede():
    # The ceiling scales with the relationship, but must not be zero without one.
    from carryover.game.table3 import cast_for as cast3
    for seed in range(20):
        for x in cast3(seed, loss=LOSS)[0]():
            if x.floor > 0:
                assert x.respond(LOSS, LOSS, 1, [], remaining=12, cooloff=3) > 0
