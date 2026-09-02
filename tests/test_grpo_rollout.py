"""The two assumptions the GRPO rollout is built on, pinned.

Neither is checkable on the pod without spending a GPU hour to find out, and
both are silent when they break: a prefix that rebuilds to the wrong position
would train the model on a board it never saw, and a logprob read off the
wrong index would corrupt the importance ratio without raising anything.
"""
import pytest

from grpo import Roll, _sampled_logprobs, window_reward
from nego_eval.rl.vf_env import Driver

PRESET = dict(rounds=2, games=2)
SEED = 4271


def _scripted(pos):
    """A fixed policy, so the recorded run is a property of the rules alone."""
    if pos.kind == 'choose':
        return {'seller': pos.legal[pos.data['t'] % len(pos.legal)]}
    return {'accept': True, 'ask': pos.data['offer']}


def _record(seed=SEED, preset=PRESET):
    d = Driver(seed, **preset)
    texts, answers = [], []
    while not d.done:
        texts.append(d.pending.text)
        answers.append(_scripted(d.pending))
        d.step(answers[-1])
    return texts, answers, d.reward()


def test_replaying_answers_rebuilds_the_same_position():
    """The credit assignment rests on this: a frozen prefix is a real prefix.

    Every branch in a group is rebuilt by replaying the recorded answers, and
    the group mean only removes the prefix's contribution if the prefix each
    branch replays is the one that was actually played.
    """
    texts, answers, _ = _record()
    for cut in range(len(answers)):
        assert Roll(SEED, PRESET, answers[:cut]).d.pending.text == texts[cut]


def test_a_fully_replayed_match_earns_what_it_earned():
    texts, answers, reward = _record()
    r = Roll(SEED, PRESET, answers)
    assert r.d.done
    assert r.d.reward() == reward


def test_replay_is_stable_across_instances():
    _, answers, _ = _record()
    cut = len(answers) // 2
    assert (Roll(SEED, PRESET, answers[:cut]).d.pending.text
            == Roll(SEED, PRESET, answers[:cut]).d.pending.text)


def test_first_picks_are_the_opening_move_of_each_game():
    _, answers, _ = _record()
    r = Roll(SEED, PRESET, answers)
    assert len(r.firsts) == PRESET['games']


def test_sampled_logprob_is_found_by_token_id_not_position():
    """vLLM includes the sampled token but it need not head the top-k."""
    got = _sampled_logprobs(
        completion_ids=[[11, 22]],
        logprobs=[[[-0.5, -9.0], [-2.0, -0.1]]],
        token_ids=[[[99, 11], [22, 77]]])
    assert got == [[-9.0, -2.0]]


def test_sampled_logprob_falls_back_to_the_first_entry():
    got = _sampled_logprobs(completion_ids=[[11]], logprobs=[[[-0.5]]], token_ids=None)
    assert got == [[-0.5]]


def test_flat_logprobs_pass_through():
    got = _sampled_logprobs(completion_ids=[[11, 22]], logprobs=[[-0.5, -2.0]], token_ids=None)
    assert got == [[-0.5, -2.0]]


def test_every_answer_maps_to_a_round():
    _, answers, _ = _record()
    r = Roll(SEED, PRESET, answers)
    assert len(r.round_of) == len(answers)
    assert r.round_of[0] == 0
    assert r.round_of == sorted(r.round_of)
    assert max(r.round_of) == PRESET['rounds'] * PRESET['games'] - 1


def test_a_bargain_is_credited_to_the_round_that_opened_it():
    """Only a pick carries `t`, so a settlement inherits the round above it."""
    d = Driver(SEED, **PRESET)
    r = Roll(SEED, PRESET)
    kinds = []
    while not r.d.done:
        kinds.append(r.d.pending.kind)
        r._record(_scripted(r.d.pending))
    for i, kind in enumerate(kinds):
        if kind == 'bargain':
            assert r.round_of[i] == r.round_of[i - 1]


def test_no_horizon_is_the_whole_match():
    _, answers, reward = _record()
    r = Roll(SEED, PRESET, answers)
    assert window_reward(r, 0, 0) == reward
    assert window_reward(r, 0, -1) == reward


def test_a_horizon_past_the_end_is_the_whole_match():
    """The dense residuals sum to the terminal one, so a wide window is it."""
    _, answers, reward = _record()
    r = Roll(SEED, PRESET, answers)
    wide = PRESET['rounds'] * PRESET['games'] + 5
    assert window_reward(r, 0, wide) == pytest.approx(reward)


def test_a_narrow_window_is_a_strict_slice():
    _, answers, _ = _record()
    r = Roll(SEED, PRESET, answers)
    per = r.d.rewards()
    assert window_reward(r, 0, 1) == pytest.approx(per[0])
    assert window_reward(r, 0, 2) == pytest.approx(sum(per[:2]))
