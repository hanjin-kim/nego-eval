"""The adapter must not be a second implementation of the rules.

A trainer needs the match to stop at every decision, which `World.run()` does not
do, and the temptation is to reimplement the loop. Then two copies of the
settlement identity exist and only one of them is under test. Instead the driver
inverts control around the same `World`, and these tests check that what comes
out the other side is what `Match` produces from the identical seed.
"""
import statistics as st

from nego_eval.game.buyers import BoardOnlyBuyer, EVBuyer
from nego_eval.game.match import Match
from nego_eval.game.table4 import VALUE, cast_for
from nego_eval.rl.vf_env import (LOSS, STEP, TRAIN, Driver, _parse_json,
                                 baseline, dataset_rows)


def drive(buyer, seed, shape=TRAIN, cap=4000):
    """Play a whole match through the driver with any scripted buyer.

    The buyer is handed the structured position, not the prompt, so this checks
    the adapter rather than a regex over its own English.
    """
    d = Driver(seed, **shape)
    n = 0
    while not d.done and n < cap:
        p = d.pending
        if p.kind == 'choose':
            pick = buyer.choose(p.data['quotes'], p.data['t'],
                                p.data['remaining'], p.data['history'])
            d.step({'seller': pick})
        else:
            acc, ask = buyer.bargain(p.data['loss'], p.data['offer'], p.data['r'],
                                     p.data['max_rounds'], p.data['seller'],
                                     p.data['history'], p.data['terms'])
            d.step({'accept': acc, 'ask': ask})
        n += 1
    return d


def by_match(buyer_factory, seed, shape=TRAIN):
    make = cast_for(seed, loss=LOSS)[0]
    return Match(buyer_factory=buyer_factory, seller_factory=make,
                 games=shape['games'], rounds=shape['rounds'], loss=LOSS,
                 value=VALUE, seed=seed, carry_over=True, step=STEP).run().profit


def test_the_driver_reaches_the_end():
    d = drive(BoardOnlyBuyer(), 0)
    assert d.done and d.profit != 0


def test_the_driver_and_the_match_runner_agree():
    """Two ways of playing the same policy on the same seed must not differ."""
    for factory in (BoardOnlyBuyer, EVBuyer):
        for seed in range(6):
            got = drive(factory(), seed).profit
            want = by_match(factory, seed)
            assert got == want, (factory.__name__, seed, got, want)


def test_the_reward_is_zero_for_the_policy_it_subtracts():
    """The baseline is the record-blind policy, so that policy scores exactly 0."""
    for seed in range(8):
        assert abs(drive(BoardOnlyBuyer(), seed).reward()) < 1e-9, seed


def test_reading_the_record_beats_the_baseline():
    got = [drive(EVBuyer(), s).reward() for s in range(40)]
    assert st.mean(got) > 10, st.mean(got)


def test_every_position_carries_its_own_structure():
    d = Driver(3, **TRAIN)
    seen, n = set(), 0
    while not d.done and n < 4000:
        p = d.pending
        seen.add(p.kind)
        assert (p.legal is not None) == (p.kind == 'choose')
        assert 'Reply:' in p.text
        if p.kind == 'choose':
            assert {q.seller for q in p.data['quotes']} == set(p.legal)
            d.step({'seller': p.legal[0]})
        else:
            assert 0 <= p.data['offer'] <= p.data['loss']
            d.step({'accept': True, 'ask': p.data['offer']})
        n += 1
    assert seen == {'choose', 'bargain'}


def test_dataset_rows_carry_the_seed_and_the_shape():
    rows = dataset_rows(3, 0)
    assert len(rows) == 3
    for r in rows:
        assert 'seed' in r['info'] and 'rounds' in r['info']


def test_json_is_found_inside_prose_and_fences():
    assert _parse_json('I pick B. {"seller":"B"}') == {'seller': 'B'}
    assert _parse_json('```json\n{"accept": false, "ask": 60}\n```')['ask'] == 60
    assert _parse_json('nothing here') == {}


def test_dense_rewards_sum_to_the_terminal_one():
    """Two shapes of the same quantity. If they diverge, one of them is wrong."""
    for seed in range(6):
        d = drive(EVBuyer(), seed)
        assert abs(sum(d.rewards()) - d.reward()) < 1e-9, (seed, sum(d.rewards()), d.reward())


def test_dense_rewards_are_zero_round_by_round_for_the_baseline():
    """The subtrahend is the record-blind policy, so it scores zero everywhere.

    Summing to zero would not be enough: a shaping that cancelled out over a
    match while being wrong within it would still mislead a per-step trainer.
    """
    for seed in range(6):
        d = drive(BoardOnlyBuyer(), seed)
        assert all(abs(r) < 1e-9 for r in d.rewards()), (seed, d.rewards()[:6])


def test_there_is_one_reward_per_round():
    d = drive(EVBuyer(), 0)
    assert len(d.rewards()) == TRAIN['rounds'] * TRAIN['games']


def test_the_adapter_shows_the_same_record_as_the_scored_buyer():
    """A trained model must be solving the task the table was measured on.

    The adapter re-typed the ledger once and it drifted — the `promises made`
    column and the `last round` line went missing, which is who was traded with a
    moment ago and how it went. Nothing would have caught that, and the trained
    model would simply not have been comparable to the seven already measured.
    """
    from nego_eval.rl.vf_env import _ledger
    from nego_eval.sim.agents import LLMBuyer
    from nego_eval.sim.world import World
    for seed in range(4):
        make = cast_for(seed, loss=LOSS)[0]
        h = World(buyer=EVBuyer(), sellers=make(), value=VALUE, loss=LOSS,
                  rounds=12, seed=seed, step=STEP).run()
        assert _ledger(h) == LLMBuyer()._ledger(h), seed


def test_a_bargain_position_stands_on_its_own():
    """Every exchange after the first must state the ones before it.

    Scoring a turn on its position rather than the transcript is what makes the
    context flat, and it takes away the only place the earlier offers lived.
    Without them a seller holding at twenty reads exactly like one that has come
    up to twenty from zero, and sellers here do move in both directions.
    """
    for seed in range(12):
        d = Driver(seed, **TRAIN)
        n = 0
        while not d.done and n < 4000:
            p = d.pending
            if p.kind == 'bargain' and p.data['r'] >= 2:
                assert 'So far this negotiation:' in p.text, (seed, p.text)
                assert p.text.count('offered to pay') >= 2, (seed, p.text)
                return
            d.step({'seller': (p.legal or ('A',))[0], 'accept': False, 'ask': LOSS})
            n += 1
    raise AssertionError('no second exchange was reached in twelve matches')


def test_the_exchange_log_resets_between_negotiations():
    """Two failures in one match must not share a transcript."""
    d = Driver(5, **TRAIN)
    firsts, n = [], 0
    while not d.done and n < 4000:
        p = d.pending
        if p.kind == 'bargain' and p.data['r'] == 1:
            firsts.append(p.text.count('offered to pay'))
        d.step({'seller': (p.legal or ('A',))[0], 'accept': True, 'ask': 0})
        n += 1
    assert firsts, 'no bargain happened'
    assert all(c == 1 for c in firsts), firsts
