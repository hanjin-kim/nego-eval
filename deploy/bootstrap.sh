#!/usr/bin/env bash
# Everything the pod needs, in one pass. Run it once after connecting.
#
# The clock starts when the pod does, so nothing here waits on a decision:
# versions are pinned, the model is fetched before anything imports it, and the
# smoke test runs a real rollout against the local server so a broken wiring
# shows up in minutes rather than after an hour of training.
set -euo pipefail
# Say where it stopped. The first run died silently after piping the uv
# installer into sh under nohup, and a log that simply ends is a log that costs
# a round trip to diagnose while the pod bills.
trap 'echo "FAILED at line $LINENO: $BASH_COMMAND" >&2' ERR

MODEL="${MODEL:-Qwen/Qwen3-8B}"
# Pinned to the newest vllm whose torch is still a CUDA 12 build.
#
# vllm 0.28 requires torch==2.13.0 exactly, and torch 2.13 — like 2.12 and 2.11 —
# now ships as +cu130 on PyPI. CUDA 13 wants a 580-series driver; the pods on
# offer run 570, which tops out at CUDA 12.8. So the newest torch that runs here
# at all is 2.10.0+cu128, and vllm 0.19.1 is the release that pins it.
#
# Raising this means checking two things, not one: what torch the vllm wants, and
# what CUDA build that torch ships as. `torch.version.cuda` against the driver in
# `nvidia-smi` is the whole test.
VLLM="${VLLM:-0.19.1}"
PORT="${PORT:-8000}"
VENV="${VENV:-/root/venv}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

echo "== 1/5  system"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python3 -V
# Take the plain CUDA image, not a PyTorch one. The PyTorch images stop at torch
# 2.4 and vllm 0.28 requires 2.13 exactly, so whichever is chosen pip replaces it
# — the PyTorch image saves a download that is then thrown away, and on some
# providers it is not offered at all. Print what is here before pip decides.
python3 - <<'EOF' || true   # system python, just for the record
import torch
print(f"  torch {torch.__version__} · cuda {torch.version.cuda} · "
      f"available {torch.cuda.is_available()}")
EOF

echo "== 2/5  python deps"
# The image ships Python 3.10 and verifiers requires >=3.11, so the work happens
# in a 3.12 environment rather than the system one. uv builds it in seconds and
# fetches the interpreter itself, which is faster than arguing with apt.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  # Installed in its own shell, not piped into this one: piping the installer
  # into sh under nohup ended the script here once, without a word in the log.
  curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-install.sh
  sh /tmp/uv-install.sh >/dev/null 2>&1
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null || { echo "uv still missing"; exit 1; }
uv venv --python 3.12 --allow-existing "$VENV" -q
PY="$VENV/bin/python"

# vllm first and alone: it is the package with an opinion about torch, and
# letting it resolve on its own makes a replacement obvious instead of a
# surprise three installs later.
# Logged to a file and summarised from it. Piping into `head` under `pipefail`
# kills the script the moment head closes the pipe — which it does, by design,
# and which cost one silent run to find.
uv pip install --python "$PY" "vllm==$VLLM" > /tmp/vllm-install.log 2>&1
grep -Ei "^(Resolved|Installed)|error" /tmp/vllm-install.log | head -5 || true
uv pip install --python "$PY" -q "verifiers==0.3.1" "trl>=0.14" "peft>=0.14" \
               "transformers>=4.48" "accelerate>=1.3" "datasets>=3.2" huggingface_hub
uv pip install --python "$PY" -q -e "$HERE[dev]"   # dev extras carry pytest

# A real multiply on the card, not is_available(). torch arrives with its own
# CUDA runtime, so the only thing that can still be wrong is the host driver,
# which is a property of the machine and cannot be chosen from the console.
"$PY" - <<'EOF'
import sys, torch
print(f"  torch {torch.__version__} · cuda available {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("torch lost the GPU during install — stop, do not pay for a CPU run")
try:
    x = torch.randn(1024, 1024, device="cuda") @ torch.randn(1024, 1024, device="cuda")
    torch.cuda.synchronize()
    print(f"  matmul on device ok · {torch.cuda.get_device_name(0)}")
except Exception as e:
    sys.exit(f"the card is visible but will not compute ({e}) — driver too old. "
             f"Terminate the pod.")
EOF

echo "== 3/5  weights  ($MODEL)"
# `huggingface-cli` is retired and now only prints a hint; `hf` is the command.
"$VENV/bin/hf" download "$MODEL" --quiet

echo "== 4/5  serving"
nohup "$VENV/bin/vllm" serve "$MODEL" --port "$PORT" --max-model-len 4096 \
      --gpu-memory-utilization 0.45 > /tmp/vllm.log 2>&1 &
for i in $(seq 1 90); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null \
  || { echo "vllm did not come up"; tail -30 /tmp/vllm.log; exit 1; }
echo "  up on :$PORT"

echo "== 5/5  smoke test"
cd "$HERE"
"$VENV/bin/python" -m pytest -q
"$VENV/bin/python" deploy/smoke.py --port "$PORT" --model "$MODEL"
echo
echo "ready.  next:  $VENV/bin/python deploy/train.py --model $MODEL --port $PORT"
