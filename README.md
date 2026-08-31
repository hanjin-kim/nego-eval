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
| **nego-eval** | **yes** | **yes** |

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
```

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
write-up, including four figures that earlier revisions got wrong.

**Carrying the record is worth the gap between how long a relationship takes to
build and how soon it is cut off.** Holding total rounds at 48 and moving only
where the boundaries fall, its value rises monotonically from +61 at four games
to +181 at twenty-four. With memory the score is flat; without it, each game
restarts a relationship that never gets far enough to pay.

**Handing a record to an agent that was not built around one is worse than
withholding it.** A tabular policy trained without carry-over, given a carried
ledger at test time, picks the right opening 0.14 of the time against a chance
rate of 0.33 — a third of its lookups land in states training never visited and
the rest carry a median of 178 visits against 55,161. The same shape appears in
language models: every model tested picks *worse* after the ledger arrives than
before it, and appending a pre-computed ratio to the ledger cost one model 0.15.

**Subtracting the non-relational baseline from the reward is worth +49** to an
otherwise identical learner — the control variate removes the 85% of the signal
that is already solved.

**Models are separated by roughly 400 points, and the ordering is stable** across
two boards with entirely different answer structures. It does not track model
scale: a flash-tier model leads, and three models from one family spanning
generations and sizes scored 0.48 / 0.47 / 0.46 on the same probe.

---

## Layout

```
src/nego_eval/
  sim/       world, bargaining, agents, the learned baseline, the LLM client
  game/      the board, the denomination grid, the match runner, the contract
scripts/     every measurement in the write-up, one file each
data/        the numbers those scripts produced
tests/       34 invariants — 16 on the settlement rules, 18 on the game
notes/       the write-up
```

Each module's docstring says what went wrong before it looked like that. That is
deliberate: most of the design here is scar tissue, and the scars are the part
worth reading.
