# Running it

    ./deploy/bootstrap.sh            # deps, weights, vLLM, tests, one real rollout
    python deploy/train.py           # before → train → after → verdict

`bootstrap.sh` ends with a single rollout against the served model. Read its
last line before starting anything long: it reports seconds per rollout, and 512
of them per iteration is the whole cost model.

## The image

`ubuntu_22_cuda_12`, or whatever plain CUDA image the provider offers. Not a
PyTorch one: those stop at torch 2.4, vllm 0.28 pins 2.13 exactly, so pip
replaces it regardless and the preinstalled copy is a download thrown away. Some
providers do not offer the PyTorch images at all.

## What is decided here rather than in advance

**Which trainer.** `prime-rl` and TRL's `GRPOTrainer` both take a `verifiers`
environment, and which one installs cleanly against the CUDA and vLLM on the
image is not knowable from a laptop. Both are in the bootstrap. The one that
imports is the one used.

**Batch and group.** 64 x 8 is what the RLVR negotiation paper used and what the
cost estimate assumes. If the smoke test reports much more than two seconds a
rollout, halve the batch before halving the iterations — fewer, larger steps beat
more, noisier ones when the reward is already a residual.

**Memory split.** vLLM is started at 45% of the card so the trainer has room.
On 80GB with an 8B and LoRA that is comfortable; on a smaller card, serve a
smaller model rather than squeezing both.

## What must not be changed

The evaluation preset is twelve rounds by four games. Every published figure was
measured there, including the seven models the trained one will be compared to.
Training uses four by six, which is a third of the sequential depth and preserves
the effect; that is the only place the two differ.

The verdict is three numbers and all three are required. It is in
`scripts/verdict.py` and was written before any of this ran, because the tabular
study showed how easily one of them can be bought with another: terminal reward
made the ledger appear to start helping while the surplus sat at -134 and the
opening move collapsed from 0.70 to 0.14.

## Runs, in the order they happened

Written while the second one was still going, so the record is not assembled
after the numbers are known.

There were two, and they are two experiments rather than one. Reporting only the
second would be selective, so both are here.

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| rollout temperature | 0.9 | 1.2 | 1.2 |
| learning rate | 5e-6 | 1e-5 | 1e-5 |
| reward horizon | whole match | whole match | 8 rounds |
| evaluation | before, after | + every 15 steps | + every 15 steps |
| outcome | stopped at step 41 | verdict failed, 2 of 3 | — |

Run 3 differs from run 2 in the horizon and nothing else.

Run 1 was stopped rather than finished. Two things came out of it, both worth
keeping:

- `frac_reward_zero_std` sat between 0.25 and 0.5, meaning a quarter to a half
  of the groups came back with every branch scoring identically. A group with no
  spread contributes no gradient, so that fraction of the compute was buying
  nothing. That is what the temperature change addresses, and it is a measured
  fault in the optimiser's input rather than a search for a better number.
- The per-step training reward bounced between -170 and -364 with no visible
  trend. It could not have shown one: every step draws different boards and the
  match-level SD is 200-650, so board variance swamps the policy. Only the same
  seeds, re-measured, can show a trend — which is why run 2 has a curve and run 1
  did not.

### Run 2, finished

Verdict: failed. `surplus > 0` at -618, `g1` down 0.052 against a tolerance of
0.05, `gk - g1` up to +0.13 and passing. Over 96 paired seeds the surplus went
-573.5 to -618.1 and the eight-point curve fits a trend of -29.8 +/- 24.4, so
nothing improved and the point estimate is mildly the other way.

The shape of it is the part worth keeping. g1, the opening move with no history
behind it, fell; gk - g1, the gain once the ledger arrives, rose. That is the
signature `verdict.py` was written against, from the tabular study: terminal
reward took the opening move from 0.70 to 0.14 while lifting the second-game
figure, which passes a naive "the ledger helps now" test while the surplus sits
underwater. Whole-match reward is terminal reward, and the pre-registered
criterion caught the failure it was written to catch — far weaker here, 0.375
to 0.323 rather than 0.70 to 0.14, but the same direction on both numbers at
once.

So run 2 is not just a flat control. It is a weak replication of the pathology
that motivated the dense reward in the first place, which is what run 3 tests.

### What run 3 has to do, written before it ran

Run 2 is a control, and a control is only worth the hour if it makes the next
run refutable. Over steps 15-75 it moved the surplus by -5.2 +/- 39.0, fitted
on the scatter of the points around a line rather than on the per-point error
bars, which are the spread across boards and cancel because every point is
measured on the same boards.

So run 2 rules out movement larger than roughly forty. `scripts/reward_snr.py`
puts the eight-round horizon above the whole-match one, 0.85 against 0.44 on
the ratio of averages. If the signal-to-noise of the reward is what was
binding, run 3 should move the surplus by more than +40 over the same span,
which run 2 has already excluded for itself.

An earlier reading of the same script said three times rather than twice. That
was a mean of per-board ratios, which overweights boards whose tail spread
happens to be small; the ratio of averages is what an optimiser actually sees
across a batch, and it is the more conservative of the two. The +40 line comes
from run 2's exclusion rather than from the ratio, so it stands — but a
doubling is a thinner reed than a tripling, and if run 3 lands just outside the
interval that is weak evidence rather than a demonstration.

If run 3 also lands inside +/- 40, then the ratio was not the binding
constraint and the next thing to suspect is the model: Qwen3-8B with thinking
disabled sits at -574, below every model in the published table, and the
handicap was introduced by this deployment rather than by the board.

Those two suspects are not independent, and the same script says how much they
overlap. Sweeping how often the tail plays at random, with the branched
decision held fixed:

| tail random | effect | tail SD | whole match | 8 rounds |
|---|---|---|---|---|
| 1.0 (where the model sits) | 64 | 148 | 0.44 | 0.85 |
| 0.4 | 60 | 124 | 0.49 | 0.98 |
| 0.2 | 64 | 93 | 0.69 | 1.42 |
| 0.0 (the null's own play) | 65 | 0 | — | — |

The decision is worth the same 60-65 whichever policy follows it. What changes
is the spread it has to be seen through, and that spread is manufactured by the
policy's own play over the thirty-odd turns after the branch. A weak policy
makes its own credit assignment hard, which is a genuine chicken and egg.

The size of it is the surprise. Going from a fully random tail to a
sixty-percent-competent one buys almost nothing, 0.44 to 0.49; the tail has to
be past eighty percent before it matters. At the level this model is actually
at, the horizon is doing more than the baseline is. The baseline is the larger
problem for what the run can conclude — `surplus > 0` is out of reach and 85%
of the gap is quote-sheet competence rather than the relational margin the
board measures — and the smaller one for whether it can learn at all.

This prediction is separate from the verdict in `scripts/verdict.py`, which is
unchanged and still requires all three of its numbers.

### The gap that made a second run necessary

The harness was built to adjudicate a finished run and not to diagnose one in
progress. `verdict.py` is a pre-registered pass/fail gate, which is the right
instrument for deciding whether to believe a result and the wrong one for
deciding whether the thing is learning. There was no learning curve, no
gradient-signal diagnostic, and no exploration measure, so the first honest
question asked of the run — is it improving? — had nothing to answer it.

The assumption behind that omission was that the tabular study had already
settled whether this environment trains, leaving the GPU run to confirm it at
scale. A tabular learner and an 8B model are far enough apart that it had not.

### What did not change between the runs

The three verdict criteria, their thresholds, the evaluation preset, and the
evaluation seeds (900,000 upward, disjoint from the training seeds 0-1023).
Temperature and learning rate are training-side knobs and were turned after
seeing results; that is legitimate for fixing an optimiser but it is exactly the
move that turns into a garden of forking paths if the runs are not all reported.
Any claim from run 2 names run 2.

## The model was not doing the task

Asked whether the -574 was caused by turning thinking off. It was.
`scripts/thinking_probe.py` asks only the opening pick, where there is no
history and the answer is a pure function of the quote sheet, and scores it
against the seller the null takes. 32 boards, and the bar is not one in three:
the EV-best seller is not uniform, so naming the same seller every time already
scores 0.53.

| | accuracy | picks | key |
|---|---|---|---|
| name only, 64 tokens, no thinking | 0.44 | C 21, A 9, B 2 | C 17, A 10, B 5 |
| plus a `why` clause, 256 tokens | 0.56 | C 24, A 5, B 3 | |
| thinking, 16k tokens | **0.78** | C 13, A 11, B 8 | |

The distribution decides the reading, not the accuracy. `name only` is below a
constant answer. `with why` clears 0.53 by three points while picking C on 75%
of boards against a key that is C on 53% — it is riding the name preference,
not computing. Only the thinking condition discriminates: its spread is close
to the key's, and it clears the bar by 25 points.

So the policy that runs 1 to 3 were optimising was not a weak player at the
board. It was a model with no way to do the arithmetic the board is scored on,
answering from a name preference — which is exactly where `policy_map.py`
placed it independently, on top of random-pick-and-accept.

That is a handicap this deployment introduced. Thinking was turned off to fit a
64-token budget; every model in the published table was called with its own
default and had its reasoning tokens counted rather than capped. Comparisons
made in this file between -574 and the published range of +6 to -342 were not
like for like, and they are withdrawn.

**Why the runs did not simply switch it back on.** The thinking condition
spends 17,102 characters on one decision, upward of four thousand tokens. A
rollout is thirty-five decisions. That is seventy times the tokens the current
setup uses, which puts GRPO with thinking on outside this budget rather than
merely more expensive. The trade is real and it is not resolved here.

### The published prompt contradicts the board, and it did not matter

`LLMBuyer.SYSTEM`, which produced every model row in the readme, says two
things the shipped rules do not support:

- "you pay the price and a share of a loss" — `world.py` charges only the
  share, with a comment saying so explicitly: no delivery, no price. Believing
  the sentence inflates the cost of failure by the price, and on 7 of 48 eval
  boards (15%) that flips which seller maximises expected value.
- "Sellers do not publish their failure rates" — the very next message reads
  `A=90 (published delivery rate 0.60), B=120 (0.80), C=110 (0.90)`.

The second looked like it might explain the headline, because the hand-written
rule uses nothing except those printed rates, so a sentence telling models to
disregard them would hand the rule its win. Tested on the pod: same model, same
32 boards, thinking on, the two system prompts against the seller the null
takes.

| | accuracy |
|---|---|
| thinking + the published prompt | 0.88 |
| thinking + the corrected prompt | 0.75 |

Paired, 24/32 against 28/32, three boards won against seven lost, -1.3 sigma.
The contradictory prompt did not hurt; it scored higher inside the noise. The
hypothesis is not supported and is withdrawn.

The noise is worth stating because it bounds what was ruled out. The same
thinking condition scored 0.78 in one probe and 0.88 in the next — same model,
same boards, same settings, different samples at temperature 0.7. At n = 32
nothing smaller than about 0.15 is visible here.

Two limits. This is one model, one decision, and only the opening pick, so the
prompt could still matter for the loss-split reasoning or for other models. And
the arithmetic above stands on its own: the prompt does misstate the rules, and
the fix belongs in the code whether or not it moved this number.

**What the model table actually looks like.** The scores against the rule run
-118 to -474 in a clean ordering by model strength. A defective sentence every
model received equally would take a uniform bite; a spread that tracks
capability is the signature of something the stronger models do better. The
probe offers the candidate: the expected-value comparison is not free. Qwen3-8B
with reasoning gets the opening pick right 0.75-0.88 of the time on a problem
with no history in it and three printed pairs to compare.

That is a claim about the models, not about the board. The 85% was known and
is the reason the readme reports surplus against the quote-sheet rule rather
than total profit — the arithmetic is already netted out of the scale, and the
model table's columns are "vs the rule" and "vs quote sheet only" rather than a
share of the 165. An earlier draft of this section read the split as though it
were a discovery about what the benchmark measures. It is a design decision the
benchmark already made, and that reading is withdrawn.

What the probe does add is about this deployment: the model these runs trained
had no way to do the arithmetic at all, which is why its -574 says nothing
about the relational margin.

## Cost

Roughly 3,300 tokens processed per training rollout after the context cut. At
64 x 8 that is a few minutes an iteration on one H100, so 40-120 iterations is
$13-30 at $2-3.50 an hour. The pod bills while it exists, not while it computes —
terminate it from the console when the run ends.
