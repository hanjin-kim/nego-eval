"""GRPO on the board, and the before/after that decides whether it worked.

The evaluation runs twice on the same seeds — once before a gradient is taken,
once after — because a single after-number has nothing to be compared against
and every reference in the readme was measured on a different machine at a
different time. Both passes go through the trainer's own in-process engine, so
the only thing that differs between them is the weights. The verdict is
applied by `scripts/verdict.py`, which was written before any of this ran.

vLLM is colocated rather than served: one GPU, and the standalone server has no
weight-sync endpoint, so a served policy would evaluate the base model after
every step and quietly report that training did nothing.
"""
import argparse, json, os, sys, time

sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts'); sys.path.insert(0, 'deploy')

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='Qwen/Qwen3-8B')
ap.add_argument('--iters', type=int, default=120)
ap.add_argument('--eval-n', type=int, default=96)
ap.add_argument('--boards', type=int, default=1024, help='training seeds to draw from')
ap.add_argument('--out', default='deploy/out')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from grpo import evaluate, make_rollout, reward_from_rollout
from nego_eval.rl.vf_env import EVAL, TRAIN

GEN = 8                                  # branches per frozen prefix
cfg = GRPOConfig(
    output_dir=f"{a.out}/run",
    max_steps=a.iters,
    learning_rate=5e-6,                  # LoRA; 1e-5 halved entropy in one step
    num_generations=GEN,
    per_device_train_batch_size=GEN,
    gradient_accumulation_steps=GEN,     # 64 branches, 8 boards, per step
    max_completion_length=64,
    temperature=0.9,
    bf16=True,
    gradient_checkpointing=True,
    use_vllm=True,
    vllm_mode='colocate',
    vllm_gpu_memory_utilization=0.30,
    logging_steps=1,
    save_strategy='no',
    report_to=[],
)

rows = Dataset.from_dict({'prompt': [str(s) for s in range(a.boards)]})
trainer = GRPOTrainer(
    model=a.model,
    reward_funcs=reward_from_rollout,
    args=cfg,
    train_dataset=rows,
    rollout_func=make_rollout(TRAIN),
    peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                           target_modules='all-linear', task_type='CAUSAL_LM'),
)


def measure(tag):
    t0 = time.time()
    out = evaluate(trainer, EVAL, a.eval_n)
    out['seconds'] = round(time.time() - t0, 1)
    json.dump(out, open(f"{a.out}/{tag}.json", 'w'), indent=1)
    print(f"== {tag}\n{json.dumps(out, indent=1)}", flush=True)
    return out


before = measure('before')
trainer.train()

# The engine syncs weights at the start of a rollout, so after the last
# optimizer step it still holds the policy from before it. Without this the
# after-evaluation measures a stale model and reports that nothing happened.
trainer.vllm_generation.sync_weights()
after = measure('after')

from verdict import show
show(before, after)
