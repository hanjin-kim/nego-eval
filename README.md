**English** · [한국어](README.ko.md) · [中文](README.zh.md)

# nego-eval

A negotiation environment where an agent **chooses** its counterparty and **meets it again**.

Every negotiation benchmark that lets an agent pick whom to deal with throws the
record away when the session ends, and every one that keeps a record assigns the
counterparty. So the thing a relationship is actually made of — what happened
last time, with someone you chose — has never been the variable.

Here it is the variable, and it is one bit.

|  | Picks its partner | Memory across games |
|---|---|---|
| [Cattle Trade](https://arxiv.org/html/2605.14537v1) | yes | no |
| [M3-Bench](https://arxiv.org/pdf/2601.08462) | fixed dyad | within game |
| [RLVR negotiation](https://arxiv.org/abs/2604.09855) | one seller | no |
| [TERMS-Bench](https://arxiv.org/abs/2605.13909) | assigned | within one negotiation |
| **nego-eval** | **yes** | **yes** |

TERMS-Bench is worth reading next to the numbers below. It diagnoses thirteen
frontier agents inside a single bilateral negotiation and finds that they
**saturate deal rate while diverging in surplus extraction** — an aggregate
outcome hiding the differences that matter. The same thing happens here one level
up: total profit puts four models within noise of each other, and the share of
the relational surplus they capture spreads them from 25% to below zero. Its
counterpart is assigned and its horizon is one negotiation; the divergence it
finds inside a deal is the divergence this board looks for between them.

---

## The game

Three sellers quote openly. The buyer takes one unit a round. Delivery sometimes
fails, and a failure creates a loss `L` that settles only when two integers add
up to it:

```
L_buyer + L_seller = L
```

The two sides bargain over that split across up to three exchanges. If they never
agree, the loss stays where it landed — on the buyer — and the pair stops trading
for three rounds. Both are told this before they start, which is what gives the
seller something to concede against. Every amount is on a grid of ten, so at
`L = 110` there are exactly twelve legal splits.

A **match** is four games of twelve rounds with the same three sellers. The
manipulation is whether each game begins holding the match ledger or among
strangers. Seeds, cast, rules and length are identical between the two.

### What is published and what is not

| | Where it lives | Priced? |
|---|---|---|
| delivery rate for a buyer with no history | on the quote sheet, exact | yes |
| share of a loss it absorbs at arm's length | learned by failing with it | no |
| how much further it goes for a regular | learned by staying | no |

A supplier's on-time rate is in the catalogue and you pay for it. What it does
when a shipment is ruined is not in the catalogue, is not contractible, and is
found out by having it happen. All of it is countable in whole units, so the
reward is arithmetic — **no judge model anywhere in the loop.**

---

## What it is an analogy for

Every mechanic answers to something a buyer does, and it is worth being explicit
about which ones, because a board game that resembles nothing is a puzzle rather
than an environment.

| In the game | In procurement |
|---|---|
| a published delivery rate, and a higher price for a better one | on-time-delivery figures, quality certifications, SLA tiers — disclosed, and you pay for them |
| the loss when a delivery fails | the consequential loss: a stopped line, expedited freight, a sale that did not happen, rework |
| two integers that must sum to the loss | somebody bears it. Contracts are usually silent on consequential damages, and silence puts it on the buyer |
| three exchanges over the split | the call after the incident — a credit note, a free replacement, splitting the freight |
| impasse: the buyer bears it all, and the pair stops trading | no agreement, so you eat it, and you stop putting that supplier on the next RFQ |
| how much of a loss it absorbs at arm's length | not in the catalogue. Found out by having a shipment ruined |
| how much further it goes for a regular | the unwritten difference between a new account and a ten-year one |
| the ledger carried across games | a purchase history surviving a fiscal year, a contract renewal, or the buyer changing desks |
| prices that jitter each round | spot movement. Staying with your supplier costs you the difference some weeks, which is what makes staying a decision |

The asymmetry in the middle of the table is the point, and it is not invented
here. What a supplier discloses is priced into what it charges; what it does once
something has gone wrong is neither disclosed nor contractible, and is learned
only by having it happen. That gap is what
[Brown, Falk & Fehr](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1468-0262.2004.00511.x)
put in a laboratory and what Macchiavello and Morjaria measure in a real supply
chain: relationships form there because nothing else enforces the part of the
bargain a contract cannot reach.

### For an agent, "no carry-over" is not hypothetical

The manipulation has a second reading that needs no supply chain at all. A fresh
context window is a buyer with no history. So is a new session, a summarised
transcript that dropped the ledger, a tool call that returns only the current
quote. The condition this environment toggles is one that agent systems are in by
default, and the measurement is what it costs.

### Where the analogy breaks

Stated because a reader will find these anyway, and should find them here first.

- **Three sellers, not a market.** No entry, no exit, no competition between
  buyers for a good supplier's attention.
- **One product, one unit a round.** No baskets, no lead times, no quality
  grades, no minimum order quantities.
- **Failure is exogenous and its rate is published.** In reality neither holds:
  reliability responds to how a supplier is treated, and nobody publishes it
  honestly.
- **Solvency is switched off.** Real suppliers squeezed below cost cut corners
  and then disappear. That mechanism was in an earlier board and is off here for
  a measurement reason given below; it is a real omission.
- **No courts, no insurance, no Incoterms.** Consequential loss often *is*
  allocated in advance in real contracts. This board is the case where it was
  not.
- **The record is not a sufficient statistic.** How much a seller concedes
  depends on the buyer's share of the *last eight* rounds; the record shows
  lifetime counts. Two histories with identical ledgers can differ by 33 in what
  a seller will pay on a loss of 110. Every agent measured here faced the same
  gap, so the comparison holds — but an agent is being asked to build a
  relationship whose mechanism it can only partly see. One line stating the
  recent window would close it, at the cost of re-measuring everything.
- **The horizon is known.** The agent is told how many rounds remain, which
  sharpens endgame behaviour in a way an open-ended relationship does not.

---

## Quick start

```bash
pip install -e ".[dev]"
pytest                                   # 34 invariant tests, no API key needed

python scripts/run_carry4.py             # the manipulation, scripted policies
python scripts/run_dp4.py                # the exact optimum, by dynamic programming
python scripts/train_rl4_shaped.py       # tabular Q-learning, two reward shapings
```

Language-model buyers need a key; nothing else does.

```bash
cp .env.example .env && $EDITOR .env
python scripts/run_probe4.py             # one move per frozen position
python scripts/run_fullplay4.py "[('deepseek/deepseek-chat', 12)]" on
python scripts/run_fullplay4.py "[('deepseek/deepseek-chat', 12)]" off 910012
python scripts/analyse_paired4.py        # models against references, same seeds
```

The third argument is the first seed, so an arm continues where an earlier one
stopped rather than replaying matches already paid for.

---

## Reference points

At `L = 110`, four games of twelve rounds, premium 120:

| Policy | Score | Share of the relational surplus |
|---|---|---|
| optimum, by dynamic programming | **1099** | 100% |
| optimum without carry-over | 1031 | 59% |
| hand-written, reads the record | 996 | 38% |
| hand-written, quote sheet only | 934 | 0% |
| tabular Q-learning, shaped reward | 898 | — |

And the models, playing whole matches, paired against those references on the
same seeds. Zero is the hand-written rule; every model sits to the left of it.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/models.dark.svg">
  <img alt="Each model against the hand-written rule, one standard error" src="docs/models.light.svg">
</picture>

| Model | n | Score | vs the rule | vs quote sheet only |
|---|---|---|---|---|
| gpt-5.6-terra | 12 | 958 | −127 ±68 | +6 ±75 |
| gpt-5.6-sol | 10 | 897 | −153 ±75 | −27 ±96 |
| claude-fable-5 | 10 | 876 | −174 ±63 | −48 ±82 |
| gemini-3.7-flash | 24 | 762 | −118 ±35 | −66 ±34 |
| qwen3.8-max | 8 | 786 | −251 ±101 | −190 ±105 |
| claude-haiku-4.5 | 12 | 663 | −421 ±72 | −288 ±79 |
| deepseek-chat | 12 | 610 | −474 ±87 | −342 ±91 |

Those rows were measured with a system prompt that misstated the rules twice:
it told models a failed delivery costs them the price, which `World` does not
charge, and that sellers do not publish their failure rates, one message before
printing all three. Both are corrected in `sim/agents.py`, so a rerun today will
not reproduce these numbers exactly. Tested on one model, the contradiction did
not change the opening pick — 0.88 with the old text against 0.75 with the new,
paired, −1.3 sigma — but the misstatement about the price flips which seller
maximises expected value on 7 of 48 boards, so the table should be re-measured
before it is leaned on.

**A deficit here does not say what it is made of.** A score is one number and
three things move it: how reliably the buyer picks the expected-value best
seller, how it settles a failure, and what the relationship earns it. Two very
different accounts fit gemini's −66 equally well.

| | picking | settling | relationship | total |
|---|---|---|---|---|
| one story | 0.91 accuracy, −71 | like the null, 0 | 0 | −66 |
| another | perfect, 0 | takes what is offered, −194 | **+128** | −66 |

Under the second, the model is *gaining* from the relationship and losing more
than that elsewhere. The score cannot tell them apart, and it does not even fix
the sign of the relational term. The three figures come from
`scripts/accuracy_to_profit.py`, `scripts/policy_map.py` and
`scripts/noise_taxes_relationship.py`, all replayed on the same board.

So these rows should not be read as measuring how well a model uses the record.
They measure a sum. Splitting it needs the pick and the settlement logged every
round — the model's choice, the expected-value best at that moment, and whether
the offer was taken or countered — which leaves the relationship as a residual
with a sign. `run_fullplay4.py` keeps only per-seed profit and discards the
rounds, so it cannot be recovered from `data/`; it has to be re-measured.

**What is forced, though, is that the information was not the limit.** Every
choose prompt prints the delivery rate beside each price, and the 934 reference
consults nothing else — no record, no memory, three expected values and a
maximum. Six of seven models see that same sheet *plus* a ledger and score below
it. However the split between picking and settling turns out, the shortfall is
in executing on visible numbers rather than in access to them, so arithmetic
reliability across the fifty-odd comparisons a match asks for is an axis this
board is sensitive to, and one worth reporting apart from any claim about the
record.

For two rows that much is already pinned. Picking at chance costs about 276
(`scripts/accuracy_to_profit.py`); haiku-4.5 at −288 and deepseek-chat at −342
are further down than picking at chance would put them, so the settlement has to
be leaking as well. Nothing similar is pinned for the rest.

Pairing matters more than it looks. Match-level spread is two to six hundred, and
one early seed was worth +530 to a policy that consults nothing at all — so a
model's mean against a reference mean computed on other seeds would mostly
measure which boards were drawn.

The surplus denominator is `1099 − 934 = 165`: what reading the record is worth
when read perfectly. Total profit is a poor scale for this environment because
roughly 85% of it comes from buying at a sensible price, which every competent
policy does about as well.

**The optimum is computable.** A seller's concession depends only on the buyer's
share of the last eight rounds, delivery rates are published, and prices are a
fixed sequence per seed — so the state is `(round, last eight picks)` and value
iteration solves it exactly. Very few environments can report a true ceiling
rather than a best-known score.

---

## The contract

Three boards failed before one worked, and each failed differently. The
conditions are therefore a module, `nego_eval.game.contract`, run against any
candidate board — every one is a measurement, not a guideline.

| Condition | Threshold | Why |
|---|---|---|
| answer recoverable from the prompt | ≥ 0.95 | otherwise the task is estimation under noise and no score can be read against a ceiling |
| quote sheet alone does not settle it | ≤ 0.75 | otherwise the record is decoration |
| "fewest failures" is not a trap | ≥ 0.45 | a board where an ordinary heuristic scores chance measures compliance with its author, not capability |
| no single heuristic suffices | ≤ 0.80 | |
| no answer-by-name | ≤ 0.42 | |
| countering can gain something | = 1.00 | if the opening offer is already the ceiling, there is no negotiation to fail at |

Three earlier boards are **not** in this repository. Each was withdrawn for a
reason the corresponding condition now encodes, and shipping a package with four
boards in it — three of them wrong — only invites someone to build on the wrong
one. `table4.py` is the board. What the discarded ones were, and what each got
wrong, is in `notes/`.

### The condition a checklist cannot hold

Board three satisfied every measurable condition and still measured the wrong
thing: making the published rate exact had switched off the channel by which
loyalty improved delivery, restoring room to bargain left the loyalty term in the
concession ceiling non-binding, and between them nothing remained by which
returning to a seller made that seller *better*. What was left was a fixed hidden
attribute and a buyer discovering it. Carrying the ledger then saved time and
almost nothing else — which is exactly what it measured.

No number computed on frozen positions catches that. It took reading a
transcript.

---

## What is known so far

Measured on this board unless noted. See `notes/note.html` for the full
write-up, including five figures that earlier revisions got wrong.

**Carrying the record is worth the gap between how long a relationship takes to
build and how soon it is cut off.** Holding total rounds at 48 and moving only
where the boundaries fall, its value rises monotonically from +61 at four games
to +181 at twenty-four. With memory the score is flat; without it, each game
restarts a relationship that never gets far enough to pay.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/shape.dark.svg">
  <img alt="What carrying the ledger is worth as boundaries multiply" src="docs/shape.light.svg">
</picture>

**Handing a record to an agent that was not built around one is worse than
withholding it.** A tabular policy trained without carry-over, given a carried
ledger at test time, picks the right opening 0.14 of the time against a chance
rate of 0.33 — a third of its lookups land in states training never visited and
the rest carry a median of 178 visits against 55,161. The same shape shows up
twice more: appending a pre-computed ratio to the ledger cost one model 0.15 and
collapsed its use of the column it had been reading, and six of the seven models
below choose worse once a ledger exists than before there was one.

**Subtracting the non-relational baseline from the reward is worth +49** to an
otherwise identical learner — the control variate removes the 85% of the signal
that is already solved.

**Seven language models play whole matches, and six of them choose worse once
the ledger arrives than before it.** The opening pick of games 2+ against the
opening pick of game 1: qwen3.8-max goes 0.88 → 0.29, below the 0.33 of chance,
having spent an hour a match to do it; claude-fable-5 0.70 → 0.47; three others
fall a little. Only gemini-3.7-flash improves, 0.29 → 0.61, and it is the worst
of the seven when there is no history to read. The hand-written rule goes
0.52 → 0.71.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/ledger.dark.svg">
  <img alt="Opening pick before and after the ledger exists" src="docs/ledger.light.svg">
</picture>

**Every model loses to that rule**, by 118 to 474 paired on the same seeds, at
two standard errors or better. Against the weaker reference — a policy that
ignores the record entirely — only the bottom two are clearly behind, so "frontier
models score below the null" is *not* what this shows.

**Scale does not predict any of it.** A flash-tier model is the only one the
record helps; the two top-tier models measured sit mid-table and last. Three
models from one family spanning generations and sizes scored 0.48 / 0.47 / 0.46
on the same probe.

**Whether a model earns anything from carrying the ledger is still open.** One
model was run in both conditions on 35 shared seeds: +40 ±35, against the +64 the
environment offers a policy that uses it perfectly. Consistent with both,
separated from neither. At sixteen pairs the same measurement read +88 ±43 and
looked settled; it was not.

---

## Training on it

The board is packaged as a `verifiers` environment in `src/nego_eval/rl/`, and
`deploy/` holds what it takes to run GRPO against it on one rented H100. Three
runs happened. None of them improved anything, and the useful part is why.

**A row is a turn, not a match.** The environment hands the model `[system] +
the current position` and nothing else. Flattening a match into one training
sequence would let round 2 attend to round 1's text, which at inference it
cannot see — training a model to lean on information it will not have is the
failure this board exists to measure. So the rollout freezes a prefix and
branches `num_generations` ways from one cut point. The prefix is shared inside
a group, so GRPO's group mean removes its contribution exactly, and the
whole-match total is the return-to-go up to that constant.

**Run 2 moved nothing, and the shape of the nothing was predicted.** Eight
evaluation points on fixed seeds fit a trend of −29.8 ± 24.4. The verdict in
`scripts/verdict.py`, written before any of it ran, failed two of three: `g1`
fell 0.375 to 0.323 while `gk − g1` rose 0.080 to 0.128 — the opening move with
no history behind it got worse while the ledger appeared to start helping. That
is the terminal-reward signature the tabular study recorded, and whole-match
reward is terminal reward.

**The reason is a ratio.** `scripts/reward_snr.py` freezes a prefix, takes each
legal seller, and plays the tail out. Best minus worst, in tail standard
deviations: 0.44 on the whole match, 0.85 over eight rounds, 4.75 over two. A
group of eight has a mean standard error of 0.35 SD, so the whole-match reward
offers about 1.2 sigma per group. Eight rounds is the shortest window that
still spans a game boundary, which is where carry-over lives.

**The model was not playing the board.** Thinking was turned off to fit a
64-token budget. On the opening pick — no history, the answer a pure function
of the quote sheet — Qwen3-8B scores 0.44 with thinking off against a bar of
0.53, because naming one seller every time already scores that. With reasoning
it scores 0.75–0.88, and its spread of picks matches the answer key's rather
than a name preference. So the runs were optimising a model with no way to do
the board's arithmetic, which is where `scripts/policy_map.py` had placed it
independently: on top of random-pick-and-accept-everything.

Turning it back on is not free. One decision with reasoning costs upward of
four thousand tokens against a rollout of thirty-five decisions, which puts
GRPO with thinking on outside a rented-hour budget rather than merely dearer.
That trade is not resolved here.

`deploy/README.md` carries all three runs, what changed between them, and the
prediction each was measured against.

## Layout

```
src/nego_eval/
  sim/       world, bargaining, agents, the learned baseline, the LLM client
  game/      the board, the denomination grid, the match runner, the contract
  rl/        the board as a verifiers environment, for training on
scripts/     every measurement in the write-up, one file each
data/        the numbers those scripts produced
tests/       66 invariants, including the prompt against the rules it describes
notes/       the write-up
deploy/      running GRPO against it on a rented card
```

Each module's docstring says what went wrong before it looked like that. That is
deliberate: most of the design here is scar tissue, and the scars are the part
worth reading.

## What would move this forward

Two questions are open, and neither is open for want of effort.

**Splitting a score into its channels.** The model table reports a sum — picking
arithmetic, settlement, and whatever the relationship earned — and one number
cannot separate them, so those rows cannot be read as measuring how well a model
uses the record. Separating them needs the pick and the settlement logged every
round: the model's choice, the expected-value best seller at that moment, and
whether the offer was taken or countered. That is a re-run of the seven models,
about 850 calls each for twelve matches, and the cost is dominated by reasoning
tokens rather than by call count. `run_fullplay4.py` keeps only per-seed profit,
so it cannot be recovered from `data/`.

**Training something that can play it.** The board is packaged for RLVR in
`src/nego_eval/rl/` and three GRPO runs are recorded in `deploy/`. None
improved anything, for a reason no reward shape reaches: the models with the
pick accuracy — all above 0.9 implied — are the ones whose weights are closed,
and the ones that can be fine-tuned score at or below what naming one seller
every time would score. Reasoning is what buys the arithmetic, and leaving it on
costs about seventy times the tokens.

If you have inference credits, access to open weights that clear roughly 0.80
pick accuracy on `scripts/thinking_probe.py`, or you want to use the board for
something else, open an issue. The commit log has an address if that is easier.

---

## License

[Business Source License 1.1](LICENSE) — source-available, not OSI open source.

**Evaluating any model or agent with it is free**, including your own models,
including internally at a company, including publishing what you find. So is
research and teaching. What the licence withholds is using it as a training
environment in a pipeline that ships commercial weights, or reselling it as a
product.

It converts to Apache 2.0 on 2030-09-01, or four years after any given version
is published, whichever comes first. For anything the grant does not cover,
contact the licensor.
