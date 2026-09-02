"""GRPO on the board, and the before/after that decides whether it worked.

The evaluation runs twice on the same seeds — once before a gradient is taken,
once after — because a single after-number has nothing to be compared against
and every reference in the readme was measured on a different machine at a
different time. The verdict is applied by `scripts/verdict.py`, which was
written before any of this ran.
"""
import argparse, json, os, statistics as st, sys, time
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')

from nego_eval.rl.vf_env import EVAL, TRAIN, Driver, SYSTEM, _parse_json
from nego_eval.game.table4 import cast_for

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='Qwen/Qwen3-8B')
ap.add_argument('--port', default='8000')
ap.add_argument('--iters', type=int, default=60)
ap.add_argument('--eval-n', type=int, default=60)
ap.add_argument('--out', default='deploy/out')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

import httpx
URL = f"http://127.0.0.1:{a.port}/v1/chat/completions"


def ask(text, temp=0.7):
    r = httpx.post(URL, json={
        'model': a.model, 'temperature': temp, 'max_tokens': 64,
        'messages': [{'role': 'system', 'content': SYSTEM},
                     {'role': 'user', 'content': text}]}, timeout=90)
    r.raise_for_status()
    return _parse_json(r.json()['choices'][0]['message']['content'])


def evaluate(tag, n, start=900_000):
    """The eval preset, on the seeds the seven models were measured on."""
    surp, g1, gk = [], [], []
    for i in range(n):
        seed = start + i
        d = Driver(seed, **EVAL)
        key = cast_for(seed, loss=110)[1]
        firsts, turns = [], 0
        while not d.done and turns < 2000:
            p = d.pending
            if p.kind == 'choose' and p.data['t'] == 0:
                firsts.append(None)             # filled after the pick
            got = ask(p.text)
            if p.kind == 'choose' and p.data['t'] == 0:
                firsts[-1] = str(got.get('seller', '')).strip()
            d.step(got)
            turns += 1
        surp.append(d.reward())
        if firsts:
            g1.append(int(firsts[0] == key))
            if len(firsts) > 1:
                gk.append(sum(int(f == key) for f in firsts[1:]) / (len(firsts) - 1))
        print(f"    {tag} {i + 1}/{n}  surplus {st.mean(surp):+.0f}", flush=True)
    out = dict(surplus=st.mean(surp), g1=st.mean(g1) if g1 else float('nan'),
               gk=st.mean(gk) if gk else float('nan'), n=len(surp),
               se=st.stdev(surp) / len(surp) ** 0.5 if len(surp) > 1 else 0.0)
    json.dump(out, open(f"{a.out}/{tag}.json", 'w'), indent=1)
    return out


print("== before")
before = evaluate('before', a.eval_n)
print(json.dumps(before, indent=1))

print("\n== train")
print("  GRPO wiring goes here; see deploy/README.md for the two options and")
print("  why the choice is made on the pod rather than guessed from a laptop.")

print("\n== after")
after = evaluate('after', a.eval_n)
print(json.dumps(after, indent=1))

from verdict import show
show(before, after)
