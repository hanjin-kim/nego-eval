"""One real rollout against the served model before any training starts.

Cheaper to find a broken prompt, a parser that returns nothing, or a reward stuck
at zero here than forty minutes into a run that is being paid for by the hour.
"""
import argparse, json, sys, time
sys.path.insert(0, 'src')

from nego_eval.rl.vf_env import Driver, TRAIN, SYSTEM, _parse_json

p = argparse.ArgumentParser()
p.add_argument('--port', default='8000')
p.add_argument('--model', default='Qwen/Qwen3-8B')
a = p.parse_args()

import httpx
url = f"http://127.0.0.1:{a.port}/v1/chat/completions"


def ask(text):
    r = httpx.post(url, json={
        'model': a.model, 'temperature': 0.7, 'max_tokens': 64,
        # Reasoning off. Qwen3 opens with <think> and does not finish inside 512
        # tokens on this prompt, so every reply came back truncated mid-thought
        # with no JSON in it — 31 of 31 unparsed on the first real rollout.
        # Harmless where the server does not know the flag.
        'chat_template_kwargs': {'enable_thinking': False},
        'messages': [{'role': 'system', 'content': SYSTEM},
                     {'role': 'user', 'content': text}]}, timeout=60)
    r.raise_for_status()
    return _parse_json(r.json()['choices'][0]['message']['content'])


t0 = time.time()
d = Driver(0, **TRAIN)
turns, blank = 0, 0
while not d.done and turns < 400:
    got = ask(d.pending.text)
    if not got:
        blank += 1
    d.step(got)
    turns += 1
took = time.time() - t0
print(f"  turns {turns} · unparsed {blank} · profit {d.profit:.0f} · reward {d.reward():+.0f}")
print(f"  dense rewards sum to terminal: {abs(sum(d.rewards()) - d.reward()) < 1e-9}")
print(f"  {took:.0f}s for one rollout → 512 rollouts ≈ {512 * took / 60:.0f} min unbatched")
if blank > turns * 0.2:
    sys.exit(f"too many unparsed replies ({blank}/{turns}) — check the prompt")
