"""The board as a `verifiers` environment, for training rather than scoring.

A rollout here is a whole match — four short games against the same three
sellers, the ledger carried across them — and the model's turn is one decision
at a time: which seller to buy from, and what to ask a seller to pay when a
delivery it made has failed. The environment answers with the next position.

Two things are worth knowing before wiring this to a trainer.

**The reward is a residual, and it arrives every round.** About 85% of a score on
this board comes from buying at a sensible price, which every competent policy
already does, so a gradient on raw profit spends most of its signal on a solved
problem. What is subtracted is the score a policy that reads the quote sheet and
never the record earns *on the same seed* — a control variate, independent of the
model's actions, so it removes the board and leaves the axis.

The timing is not a detail. Delivered once at the end of a thirty-five-decision
match, that same subtraction is worth nothing: a tabular learner scores −134 on
terminal reward against −131 on no shaping at all, and its opening move collapses
to 0.14 against a chance rate of 0.33, because a signal arriving thirty-five turns
later cannot teach the one decision that has no history behind it. Paid out round
by round — the same total, the same control variate — the same learner reaches
−44. Every round of the baseline is played anyway, so the dense form is free.

**The position is not a sufficient statistic, and that is a known limit.** How
much a seller will concede depends on this buyer's share of the *last eight*
rounds; the record shows lifetime counts. Two histories with identical ledgers —
eight rounds with A then eight with B, or the reverse — differ by 33 in what A
will pay on a loss of 110, and nothing in the prompt separates them. An agent is
asked to build a relationship whose mechanism it can only partly see.

Leaving the transcript in would recover part of it, since an agent could read its
own recent picks back out. That is not why the transcript was cut, and it is not
a good reason to keep it: a buyer knows who it has been buying from without
consulting a log. The clean fix is one more line in the record — "of your last
eight rounds: C x5, A x2, B x1" — and it is not applied here because every
published figure was measured without it, and re-measuring seven models is not
free. The environment is partially observable in this specific way, every agent
measured on it faced the same, and the comparison is sound; a board built fresh
should include the line.

**The episode is short on purpose.** Cost here is sequential depth: nothing
inside a match parallelises, and forty-eight rounds is about seventy calls in a
row. Four games of six rounds is thirty-five, and the effect it has to preserve
survives the cut — memory is worth about a tenth of the score at every length
between twelve and forty-eight rounds. The evaluation preset stays at twelve
rounds by four games, because every published number was measured there and
changing it would break the comparison.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

from nego_eval.game.buyers import BoardOnlyBuyer
from nego_eval.game.match import Match
from nego_eval.game.table4 import VALUE, cast_for
from nego_eval.sim.agents import SYSTEM as _SYSTEM
from nego_eval.sim.world import Outcome, Quote, World

LOSS, STEP = 110, 10
TRAIN = dict(rounds=4, games=6)     # 35 calls a rollout
EVAL = dict(rounds=12, games=4)     # what every published figure was measured on


@dataclass
class Position:
    """One decision: the prose a model reads, and the same thing structured.

    `text` is what goes into the prompt. `data` is the identical position as
    objects, so a scripted policy can be driven through this adapter without
    parsing its own prompt back out of English — which is how a test meant to
    check the adapter against `Match` ends up checking a regex instead.
    """

    kind: str                       # 'choose' or 'bargain'
    text: str
    legal: tuple[str, ...] | None
    data: dict


def _ledger(history: list[Outcome]) -> str:
    """The record, rendered exactly as the scored buyer renders it.

    Re-typed here once and it drifted: the adapter's version dropped the
    `promises made` column and the `last round` line, which is who was traded
    with a moment ago and how it went. A model trained on that prompt would have
    been solving a different task from the seven in the table, and nothing would
    have flagged it. So it delegates.
    """
    from nego_eval.sim.agents import LLMBuyer
    return LLMBuyer()._ledger(history)


def choose_prompt(quotes: list[Quote], t: int, remaining: int,
                  history: list[Outcome]) -> str:
    q = ", ".join(f"{x.seller}={x.price}"
                  + (f" (published delivery rate {x.rate:.2f})" if x.rate else "")
                  for x in quotes)
    return (f"Round {t + 1}. Rounds remaining after this one: {remaining - 1}.\n"
            f"Unit value to you if delivery succeeds: {VALUE}.\n"
            f"Loss if delivery fails: {LOSS}.\n"
            f"Quotes: {q}\n\n"
            f"Your record so far:\n{_ledger(history)}\n\n"
            f'Reply: {{"seller": "<name>"}}')


def bargain_prompt(loss: int, offer: int, r: int, max_rounds: int, seller: str,
                   history: list[Outcome], terms: str,
                   exchanges: list[str] | None = None) -> str:
    """One exchange of a bargain, with the exchanges before it spelled out.

    Only the standing offer used to appear here, on the assumption that the
    transcript carried the rest. Scoring each turn on its own position took the
    transcript away, and with it any way to tell a seller that has moved from one
    that has not — twenty against twenty reads the same as zero against twenty,
    and a seller in this game does sometimes retreat. The exchanges are stated
    because the position has to be self-contained, not because the model needs
    reminding.
    """
    so_far = ""
    if exchanges:
        so_far = "So far this negotiation:\n" + "\n".join(exchanges) + "\n\n"
    return (f"Delivery from {seller} failed. The loss is {loss}.\n"
            f"Exchange {r} of {max_rounds}. {seller} offers to pay {offer}; "
            f"the remaining {loss - offer} would fall on you.\n\n"
            f"{so_far}{terms}\n\n"
            f"Your record:\n{_ledger(history)}\n\n"
            f'Reply: {{"accept": true|false, "ask": <integer 0-{loss}>}}')


# The rules as the buyer is told them, defined once in `sim.agents`.
SYSTEM = _SYSTEM


_BASE: dict[tuple[int, int, int], list[float]] = {}


def baseline_rounds(seed: int, rounds: int, games: int) -> list[float]:
    """Round-by-round profit of the record-blind policy on this seed."""
    key = (seed, rounds, games)
    if key not in _BASE:
        make = cast_for(seed, loss=LOSS)[0]
        prior, out = [], []
        for g in range(games):
            h = World(buyer=BoardOnlyBuyer(), sellers=make(), value=VALUE,
                      loss=LOSS, rounds=rounds, seed=seed * 1000 + g, step=STEP,
                      prior=list(prior)).run()
            by_t = {o.t: o.buyer_profit for o in h}
            out += [by_t.get(t, 0.0) for t in range(rounds)]
            prior = prior + h
        _BASE[key] = out
    return _BASE[key]


def baseline(seed: int, rounds: int, games: int) -> float:
    """The same thing summed, for a trainer that wants one number."""
    return sum(baseline_rounds(seed, rounds, games))


def seeds(n: int, start: int = 0) -> list[int]:
    return list(range(start, start + n))


def dataset_rows(n: int, start: int = 0, preset: dict | None = None) -> list[dict]:
    """One row per seed. The board is the seed; nothing else varies."""
    preset = preset or TRAIN
    out = []
    for s in seeds(n, start):
        make, key, ev, board, hidden = cast_for(s, loss=LOSS)
        out.append(dict(
            question=f"match seed {s}",
            answer=key,
            info=json.dumps(dict(seed=s, best=key, **preset)),
        ))
    return out


class Driver:
    """Runs a match, stopping wherever the buyer would have to decide.

    `World.run()` plays a whole game with a buyer object, which is the right
    shape for a scripted policy and the wrong one for a trainer that hands back
    one completion at a time. Rather than reimplement the rules — and risk the
    two drifting, which is how the settlement identity would quietly stop being
    enforced — the buyer is a shim inside a worker thread that blocks on a queue
    whenever it is asked to decide.

    A generator cannot do this: `World.run()` is an ordinary function, so a
    `yield` inside the buyer has nowhere to suspend to. Inverting control needs a
    second stack, and a thread is the one available without a dependency.
    """

    _SENTINEL = object()

    def __init__(self, seed: int, rounds: int, games: int, timeout: float = 30.0):
        import queue, threading
        self.seed, self.rounds, self.games = seed, rounds, games
        self.make, self.best = cast_for(seed, loss=LOSS)[:2]
        self.profit = 0.0
        self.per_round: list[float] = []
        self._exchanges: list[str] = []
        self.done = False
        self.pending: Position | None = None
        self._ask_q: "queue.Queue" = queue.Queue(1)
        self._ans_q: "queue.Queue" = queue.Queue(1)
        self._timeout = timeout
        self._thread = threading.Thread(target=self._play, daemon=True)
        self._thread.start()
        self._receive()

    class _Shim:
        """The buyer `World` sees. Every decision goes out and waits."""

        def __init__(self, outer):
            self.o = outer

        def choose(self, quotes, t, remaining, history):
            legal = tuple(q.seller for q in quotes)
            got = self.o._request(Position(
                'choose', choose_prompt(quotes, t, remaining, history), legal,
                dict(quotes=list(quotes), t=t, remaining=remaining,
                     history=list(history))))
            pick = str(got.get('seller', '')).strip()
            return pick if pick in legal else min(quotes, key=lambda q: q.price).seller

        def bargain(self, loss, offer, r, max_rounds, seller_name, history, terms):
            if r == 1:
                self.o._exchanges = []
            ex = self.o._exchanges
            ex.append(f"  {seller_name} offered to pay {offer}")
            got = self.o._request(Position(
                'bargain',
                bargain_prompt(loss, offer, r, max_rounds, seller_name, history,
                               terms, list(ex)),
                None,
                dict(loss=loss, offer=offer, r=r, max_rounds=max_rounds,
                     seller=seller_name, history=list(history), terms=terms)))
            try:
                ask = int(got.get('ask', offer))
            except (TypeError, ValueError):
                ask = offer
            acc = got.get('accept')
            if acc is None:
                acc = ask <= offer
            ex.append(f"  you {'accepted' if acc else f'asked for {ask}'}")
            return bool(acc), ask

    def _request(self, pos: Position) -> dict:
        self._ask_q.put(pos)
        got = self._ans_q.get()
        return got if isinstance(got, dict) else {}

    def _play(self):
        buyer = Driver._Shim(self)
        prior: list[Outcome] = []
        try:
            for g in range(self.games):
                h = World(buyer=buyer, sellers=self.make(), value=VALUE, loss=LOSS,
                          rounds=self.rounds, seed=self.seed * 1000 + g, step=STEP,
                          prior=list(prior)).run()
                self.profit += sum(o.buyer_profit for o in h)
                by_t = {o.t: o.buyer_profit for o in h}
                self.per_round += [by_t.get(t, 0.0) for t in range(self.rounds)]
                prior = prior + h
        finally:
            self._ask_q.put(Driver._SENTINEL)

    def _receive(self):
        import queue
        try:
            got = self._ask_q.get(timeout=self._timeout)
        except queue.Empty:
            self.pending, self.done = None, True
            return
        if got is Driver._SENTINEL:
            self.pending, self.done = None, True
        else:
            self.pending = got

    def step(self, answer: dict | None) -> None:
        """Hand back one decision and run on to the next, or to the end."""
        if self.done:
            return
        self._ans_q.put(answer or {})
        self._receive()

    def reward(self) -> float:
        """Profit over the whole match, less what ignoring the record earns."""
        return self.profit - baseline(self.seed, self.rounds, self.games)

    def rewards(self) -> list[float]:
        """The same total, round by round, which is the form to train on.

        A trainer that can only take one number per rollout should still use
        `reward()`; one that can credit per step should use this, because the
        difference between them on a tabular learner is a third of the gap to
        the ceiling.
        """
        mine = self.per_round + [0.0] * (self.rounds * self.games - len(self.per_round))
        base = baseline_rounds(self.seed, self.rounds, self.games)
        return [a - b for a, b in zip(mine, base)]


# ---------------------------------------------------------------------------
# The `verifiers` wrapper. Imported lazily so the rest of this module — the
# driver, the prompts, the residual reward — stays usable by any trainer, and
# by the test suite, without the dependency.
# ---------------------------------------------------------------------------

def load_environment(n_train: int = 2000, n_eval: int = 200, preset: str = 'train',
                     **kwargs):
    """Build the environment `prime-rl` and `vf-eval` expect.

    `preset='train'` is four games of six rounds, which is thirty-five calls a
    rollout; `preset='eval'` is twelve by four, the shape every published figure
    was measured on. Do not train on the eval preset to save a conversion — the
    numbers in the readme stop being comparable and nothing warns you.
    """
    import json as _json

    import verifiers as vf
    from datasets import Dataset
    from verifiers.legacy.utils.message_utils import (concat_messages,
                                                      maybe_normalize_messages)

    shape = EVAL if preset == 'eval' else TRAIN

    class NegoEnv(vf.MultiTurnEnv):
        """Each turn is scored on its own position, not on the transcript.

        The default rollout hands the model everything said so far. Here that is
        pure repetition: a prompt already carries the whole ledger and the whole
        quote sheet, so by the last turn of a training match the context is about
        4,400 tokens of which nearly all is restated. Measured over a rollout it
        is 73,000 tokens processed against 8,900 — eight times the compute for
        the same information, and nineteen times on the evaluation preset.

        It is also cleaner. With the transcript attached, a model can answer from
        what it said three turns ago rather than from the record, and "I picked B
        before" is exactly the shortcut this environment exists to distinguish
        from reading a ledger. Cut, every decision is taken on the position, which
        is the condition the frozen probe measures under.

        The docstring on the method being overridden invites this: "override for
        rollouts with non-linear message sequences."
        """

        async def get_prompt_messages(self, state):
            if not state["trajectory"]:
                return state["prompt"]
            last = state["trajectory"][-1]
            seen = concat_messages([last["prompt"], last["completion"]])
            reply = await self.env_response(seen, state)
            reply = maybe_normalize_messages(reply, field_name="env_response")
            head = [m for m in state["prompt"] if _role(m) == "system"]
            return concat_messages([head, reply])

        # awaited by the base rollout, so all three hooks are coroutines even
        # though nothing in them blocks
        async def setup_state(self, state):
            info = state['input'].info
            info = _json.loads(info) if isinstance(info, str) else info
            state['task'] = dict(state.get('task') or {})
            state['task']['driver'] = Driver(info['seed'], info['rounds'],
                                             info['games'])
            return state

        async def is_completed(self, state, **kw):
            d = state['task'].get('driver')
            return d is None or d.done

        async def env_response(self, messages, state, **kw):
            d = state['task']['driver']
            last = messages[-1] if messages else None
            answer = self.parser.parse_answer(messages) if self.parser else None
            if not isinstance(answer, dict):
                answer = _parse_json(getattr(last, 'content', '') or '')
            d.step(answer)
            if d.done:
                return []
            # typed, not a raw dict: the library revalidates dicts on every turn
            return [vf.UserMessage(role='user', content=d.pending.text)]

    @vf.reward
    def surplus(state, **kw) -> float:
        """Score above a policy that reads the quote sheet and never the record.

        The subtrahend does not depend on what the model did — it is the same
        board played by a fixed policy — so this is a control variate rather
        than a shaped objective. It removes the four fifths of the reward that
        every competent policy already collects and leaves the axis this
        environment exists to measure.

        This is the summed form, which is what a rollout-level rubric can take.
        A trainer able to credit individual turns should use `Driver.rewards()`
        instead: the same total delivered round by round is worth about a third
        of the remaining gap, and delivering it only at the end throws the whole
        control variate away.
        """
        d = state['task'].get('driver')
        return float(d.reward()) if d is not None else 0.0

    rows = dataset_rows(n_train, 0, shape)
    ev = dataset_rows(n_eval, 900_000, shape)
    return NegoEnv(
        dataset=Dataset.from_list(rows),
        eval_dataset=Dataset.from_list(ev),
        system_prompt=SYSTEM,
        rubric=vf.Rubric(funcs=[surplus], weights=[1.0]),
        max_turns=shape['rounds'] * shape['games'] * 4 + 4,
        **kwargs,
    )


def _role(m) -> str:
    return m.get('role') if isinstance(m, dict) else getattr(m, 'role', '')


def _parse_json(text: str) -> dict:
    """Last JSON object in a completion. Models wrap it in prose and fences."""
    import re
    for m in reversed(list(re.finditer(r'\{[^{}]*\}', text or ''))):
        try:
            got = json.loads(m.group(0))
            if isinstance(got, dict):
                return got
        except json.JSONDecodeError:
            continue
    return {}
