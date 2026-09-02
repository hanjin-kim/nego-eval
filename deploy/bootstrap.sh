#!/usr/bin/env bash
# Everything the pod needs, in one pass. Run it once after connecting.
#
# The clock starts when the pod does, so nothing here waits on a decision:
# versions are pinned, the model is fetched before anything imports it, and the
# smoke test runs a real rollout against the local server so a broken wiring
# shows up in minutes rather than after an hour of training.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
# Pinned, not floated. vllm 0.28.0 requires torch==2.13.0 exactly — an equality,
# not a range — so a newer vllm appearing next week would quietly pull a
# different torch, and if the pod image ships 2.13 that is a two-gigabyte
# download nobody asked for. Raise this deliberately, having checked the pin.
VLLM="${VLLM:-0.28.0}"
PORT="${PORT:-8000}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

echo "== 1/5  system"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -V
# Take the plain CUDA image, not a PyTorch one. The PyTorch images stop at torch
# 2.4 and vllm 0.28 requires 2.13 exactly, so whichever is chosen pip replaces it
# — the PyTorch image saves a download that is then thrown away, and on some
# providers it is not offered at all. Print what is here before pip decides.
python3 - <<'EOF' || true
import torch
print(f"  torch {torch.__version__} · cuda {torch.version.cuda} · "
      f"available {torch.cuda.is_available()}")
EOF

echo "== 2/5  python deps"
pip install -q --upgrade pip
# vllm first and alone: it is the package with an opinion about torch, and
# letting it resolve on its own makes a replacement obvious instead of a
# surprise three installs later. If the image already carries the torch it
# wants, this is a small install; if not, it is the big one, and better here
# than halfway through.
pip install "vllm==$VLLM" 2>&1 | grep -Ei "torch|error" | head -5 || true
pip install -q "verifiers==0.3.1" "trl>=0.14" "peft>=0.14" \
               "transformers>=4.48" "accelerate>=1.3" "datasets>=3.2" huggingface_hub
pip install -q -e "$HERE"
# A real multiply on the card, not is_available(). The image list tops out at
# torch 2.4 while vllm wants 2.13, so pip replaces torch and brings its own CUDA
# runtime with it — the only thing that can still be wrong is the host driver,
# which is a property of the machine and cannot be chosen from the console.
# is_available() has been known to return True and then fault on first use.
python3 - <<'EOF'
import sys, torch
print(f"  after install: torch {torch.__version__} · cuda available {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("torch lost the GPU during install — stop, do not pay for a CPU run")
try:
    x = torch.randn(1024, 1024, device="cuda")
    torch.cuda.synchronize()
    print(f"  matmul on device ok · {torch.cuda.get_device_name(0)}")
except Exception as e:
    sys.exit(f"the card is visible but will not compute ({e}) — likely a host "
             f"driver too old for this torch. Terminate the pod.")
EOF

echo "== 3/5  weights  ($MODEL)"
huggingface-cli download "$MODEL" --quiet

echo "== 4/5  serving"
nohup vllm serve "$MODEL" --port "$PORT" --max-model-len 4096 \
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
python -m pytest -q
python deploy/smoke.py --port "$PORT" --model "$MODEL"
echo
echo "ready.  next:  python deploy/train.py --model $MODEL --port $PORT"
