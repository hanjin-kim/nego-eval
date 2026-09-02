# Running it

    ./deploy/bootstrap.sh            # deps, weights, vLLM, tests, one real rollout
    python deploy/train.py           # before → train → after → verdict

`bootstrap.sh` ends with a single rollout against the served model. Read its
last line before starting anything long: it reports seconds per rollout, and 512
of them per iteration is the whole cost model.

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

## Cost

Roughly 3,300 tokens processed per training rollout after the context cut. At
64 x 8 that is a few minutes an iteration on one H100, so 40-120 iterations is
$13-30 at $2-3.50 an hour. The pod bills while it exists, not while it computes —
terminate it from the console when the run ends.
