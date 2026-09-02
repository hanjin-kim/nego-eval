#!/usr/bin/env bash
# Everything the pod needs, in one pass. Run it once after connecting.
#
# The clock starts when the pod does, so nothing here waits on a decision:
# versions are pinned, the model is fetched before anything imports it, and the
# smoke test runs a real rollout against the local server so a broken wiring
# shows up in minutes rather than after an hour of training.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
PORT="${PORT:-8000}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

echo "== 1/5  system"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -V

echo "== 2/5  python deps"
pip install -q --upgrade pip
pip install -q "vllm>=0.8" "verifiers==0.3.1" "trl>=0.14" "peft>=0.14" \
               "transformers>=4.48" "accelerate>=1.3" "datasets>=3.2" huggingface_hub
pip install -q -e "$HERE"

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
