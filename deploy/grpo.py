"""Multi-turn GRPO on the board, one turn to a row.

Not the usual flattened multi-turn sequence, and the reason matters. The
environment hands the model `[system] + the current position` and nothing
else — that choice lives in `vf_env.load_environment`, and every published
figure was measured under it. Concatenating a match into one training
sequence would let the answer at round 2 attend to round 1's text, which at
inference it cannot see. Training a model to lean on information it will not
have is the precise failure this board exists to measure, so a row is a turn.

Credit assignment survives the split. For each board the rollout plays a
prefix, freezes it, and samples `num_generations` continuations of the same
position; each continuation then plays the match out. The prefix is identical
across the group, so its contribution to the match total is a constant, and
GRPO subtracts the group mean. Whole-match reward and return-to-go therefore
differ by a constant and yield identical advantages — which is why the reward
below is simply `Driver.reward()`.

`Driver` is deterministic in (seed, answers), so replaying a recorded answer
list rebuilds a position exactly and spends no model call doing it.

Generation is lockstep rather than threaded: colocate vLLM is one in-process
engine, so the batching has to be ours. Every live rollout contributes its
pending position to one `generate` call, and finished ones drop out.
"""
import sys

sys.path.insert(0, 'src')

from nego_eval.game.table4 import cast_for
from nego_eval.rl.vf_env import SYSTEM, Driver, _parse_json

MAX_NEW = 64          # the budget every published figure was measured under
LOSS = 110


def _templated(tok, text: str) -> list[int]:
    """One position as token ids, thinking off.

    Qwen3 opens with `<think>` and does not reach the JSON inside any budget
    worth paying for; 31 of 31 replies came back truncated mid-thought on the
    first real rollout. The flag is ignored by templates that lack it.
    """
    out = tok.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM},
         {'role': 'user', 'content': text}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False)
    # transformers 5 hands back a BatchEncoding, and iterating one yields its
    # keys — feeding that to vLLM compares a str against the vocab size.
    return list(out['input_ids'] if hasattr(out, 'keys') else out)


def _sampled_logprobs(completion_ids, logprobs, token_ids):
    """One logprob per generated token.

    vLLM returns `(length, top_k)` per sequence and TRL wants a flat scalar per
    position, so the sampled token's entry is the one to keep. It is not always
    at index 0 — the sampled token is always included but may fall outside the
    top-k — hence the lookup by id rather than a positional guess.
    """
    ids = token_ids or [None] * len(completion_ids)
    out = []
    for comp, seq, tids in zip(completion_ids, logprobs, ids):
        row = []
        for pos, tok in enumerate(comp):
            cand = seq[pos]
            if not isinstance(cand, (list, tuple)):
                row.append(cand)
                continue
            k = 0
            if tids is not None:
                k = next((i for i, t in enumerate(tids[pos]) if t == tok), 0)
            row.append(cand[k])
        out.append(row)
    return out


def generate(trainer, texts: list[str]):
    """One completion per text, batched. Rows come back aligned to `texts`."""
    tok = trainer.processing_class
    # Colocate vLLM hardcodes `n=1` and ignores its num_generations argument —
    # TRL's own path hands it prompts that are already duplicated. Repeating a
    # position is therefore done by repeating it in `texts`.
    ids = [_templated(tok, t) for t in texts]
    prompt_ids, completion_ids, logprobs, token_ids = trainer.vllm_generation.generate(
        ids, None, 1)
    return (prompt_ids, completion_ids,
            _sampled_logprobs(completion_ids, logprobs, token_ids))


class Roll:
    """A match, rebuilt from a recorded answer list and then carried forward."""

    def __init__(self, seed: int, preset: dict, replay: list[dict] | None = None):
        self.seed = seed
        self.d = Driver(seed, **preset)
        self.answers: list[dict] = []
        self.firsts: list[str] = []
        self.unparsed = 0
        self.round_of: list[int] = []     # answer index -> index into rewards()
        self._game, self._t = -1, 0
        for a in replay or []:
            self._record(a)

    def _record(self, answer: dict) -> None:
        p = self.d.pending
        if p is not None and p.kind == 'choose':
            if p.data['t'] == 0:
                self._game += 1
                self.firsts.append(str(answer.get('seller', '')).strip())
            self._t = p.data['t']
        # a bargain belongs to the round of the pick that opened it
        self.round_of.append(max(self._game, 0) * self.d.rounds + self._t)
        self.answers.append(answer)
        self.d.step(answer)

    def play(self, trainer, cap: int = 4000) -> None:
        """Only meaningful inside `play_all`; kept for a single rollout."""
        play_all(trainer, [self], cap)


def play_all(trainer, rolls: list[Roll], cap: int = 4000) -> None:
    """Carry every roll to the end, one batched generate per lockstep turn."""
    tok = trainer.processing_class
    for _ in range(cap):
        live = [r for r in rolls if not r.d.done and r.d.pending is not None]
        if not live:
            return
        _, completions, _ = generate(trainer, [r.d.pending.text for r in live])
        for r, c in zip(live, completions):
            answer = _parse_json(tok.decode(c, skip_special_tokens=True))
            r.unparsed += not answer
            r._record(answer)
    raise RuntimeError('rollouts did not terminate inside the step cap')


def evaluate(trainer, preset: dict, n: int, start: int = 900_000,
             temp: float = 0.7) -> dict:
    """Surplus and the two recall numbers, on the published eval seeds.

    Sampled at the temperature the seven published models were called at, not
    at the training temperature, and at the same one before and after — the
    only thing that may differ between the two passes is the weights.
    """
    import statistics as st

    was = trainer.vllm_generation.temperature
    trainer.vllm_generation.temperature = temp
    try:
        rolls = [Roll(start + i, preset) for i in range(n)]
        play_all(trainer, rolls)
    finally:
        trainer.vllm_generation.temperature = was
    surp = [r.d.reward() for r in rolls]
    g1, gk = [], []
    for r in rolls:
        key = cast_for(r.seed, loss=LOSS)[1]
        if r.firsts:
            g1.append(int(r.firsts[0] == key))
            if len(r.firsts) > 1:
                gk.append(sum(int(f == key) for f in r.firsts[1:]) / (len(r.firsts) - 1))
    return dict(
        surplus=st.mean(surp), n=len(surp),
        unparsed=sum(r.unparsed for r in rolls) / max(sum(len(r.answers) for r in rolls), 1),
        se=st.stdev(surp) / len(surp) ** 0.5 if len(surp) > 1 else 0.0,
        g1=st.mean(g1) if g1 else float('nan'),
        gk=st.mean(gk) if gk else float('nan'))


def window_reward(roll: 'Roll', cut: int, horizon: int) -> float:
    """The dense residual over the rounds a decision can still reach.

    A branch's whole-match total is the return-to-go up to a constant the group
    mean removes, which is correct and still nearly useless: the decision being
    scored is worth 10-50 and the thirty-odd that follow it carry a match SD of
    200-650, so one sample says almost nothing. Truncating the horizon trades
    the credit that arrives late for a variance a group of eight can actually
    resolve.

    `horizon <= 0` restores the whole-match total, so the two are one flag apart
    and a run that changes it changes nothing else.
    """
    if horizon <= 0:
        return roll.d.reward()
    per = roll.d.rewards()
    start = roll.round_of[cut]
    return sum(per[start:start + horizon])


def curve_callback(trainer, preset: dict, n: int, every: int, path: str):
    """Re-measure the eval seeds every `every` steps.

    Two points cannot show a trend, and the per-step training reward cannot
    either: each step draws different boards, and board variance (a match-level
    SD of 200-650) swamps whatever the policy is doing. Only the same seeds,
    re-measured, say whether the thing is improving.
    """
    import json

    from transformers import TrainerCallback

    class Curve(TrainerCallback):
        rows: list[dict] = []

        def on_step_end(self, args, state, control, **kw):
            if every <= 0 or state.global_step % every:
                return
            # the engine syncs at the start of a rollout, so it is holding the
            # policy from before this step until told otherwise
            trainer.vllm_generation.sync_weights()
            out = evaluate(trainer, preset, n)
            out['step'] = state.global_step
            self.rows.append(out)
            json.dump(self.rows, open(path, 'w'), indent=1)
            print(f"  [평가] step {state.global_step:>3}"
                  f"  잉여 {out['surplus']:+7.0f} ± {out['se']:.0f}"
                  f"  g1 {out['g1']:.2f}  gk {out['gk']:.2f}"
                  f"  미파싱 {out['unparsed']:.1%}", flush=True)

    return Curve()


def make_rollout(preset: dict, horizon: int = 0, log=print):
    """The `rollout_func` TRL calls once per generation batch.

    Rows are grouped by board rather than assumed: TRL hands over the batch it
    wants completions for, already carrying `num_generations` repeats of each
    board, and one row must come back per row given. Every row sharing a seed
    therefore shares one prefix and one cut point, which is what makes the
    group mean a clean baseline for the branch that follows it.
    """
    import random

    def rollout(prompts, trainer):
        seeds = [int(p if isinstance(p, str) else p[-1]['content']) for p in prompts]
        uniq = list(dict.fromkeys(seeds))
        tok = trainer.processing_class
        rng = random.Random(trainer.state.global_step * 1_000_003 + uniq[0])

        # 1. one pass with the current policy per board, to have a prefix worth
        #    freezing; boards play in lockstep so the engine sees full batches
        played = {s: Roll(s, preset) for s in uniq}
        play_all(trainer, list(played.values()))
        cuts = {s: rng.randrange(len(played[s].answers)) for s in uniq}

        # 2. rebuild each board to its cut and branch once per row
        stems = {s: Roll(s, preset, played[s].answers[:cuts[s]]) for s in uniq}
        prompt_ids, completion_ids, logprobs = generate(
            trainer, [stems[s].d.pending.text for s in seeds])

        # 3. every branch plays its match out. The prefix is shared inside a
        #    board, so its contribution is a constant the group mean removes:
        #    the whole-match total is the return-to-go up to that constant.
        branches = [Roll(s, preset, played[s].answers[:cuts[s]] +
                         [_parse_json(tok.decode(c, skip_special_tokens=True))])
                    for s, c in zip(seeds, completion_ids)]
        play_all(trainer, branches)
        rewards = [window_reward(b, cut, horizon)
                   for b, cut in zip(branches, [cuts[s] for s in seeds])]

        turns = sum(len(b.answers) for b in branches)
        bad = sum(b.unparsed for b in branches)
        log(f"  step {trainer.state.global_step:>3}  rows {len(seeds)}"
            f"  boards {len(uniq)}  reward {sum(rewards) / len(rewards):+7.0f}"
            f"  미파싱 {bad / max(turns, 1):.1%}", flush=True)
        return dict(prompt_ids=prompt_ids, completion_ids=completion_ids,
                    logprobs=logprobs, rollout_reward=rewards)

    return rollout


def reward_from_rollout(completions, rollout_reward=None, **kw):
    """The reward is decided in the rollout; this hands it to the optimiser."""
    return list(rollout_reward)
