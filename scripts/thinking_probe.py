"""Is the board-reading gap a reading problem or an arithmetic one?

The null's whole advantage in picking is one calculation: for each seller,
`rate * (value - price) - (1 - rate) * loss`, then take the largest. Every term
is printed on the quote sheet. That is multi-step arithmetic, which is what a
reasoning pass buys and what turning it off takes away — and thinking was
turned off by this deployment, to fit a 64-token budget, while every model in
the published table was called normally.

So the confound is testable without playing a match. Ask only the opening pick,
where there is no history to reason about and the answer is a pure function of
the sheet, and score it against the seller the null would take. One call per
board instead of thirty-five.
"""
import argparse, concurrent.futures as cf, json, statistics as st, sys

sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts'); sys.path.insert(0, 'deploy')

import httpx

from nego_eval.rl.vf_env import EVAL, SYSTEM, Driver, _parse_json
from nego_eval.sim.agents import LLMBuyer
from policy_map import _evbest

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='Qwen/Qwen3-8B')
ap.add_argument('--port', default='8000')
ap.add_argument('-n', type=int, default=48)
ap.add_argument('--start', type=int, default=900_000)
ap.add_argument('--budget', type=int, default=2048, help='tokens when thinking')
ap.add_argument('--out', default='deploy/thinking.json')
a = ap.parse_args()
URL = f"http://127.0.0.1:{a.port}/v1/chat/completions"


PLAIN_REPLY = 'Reply: {"seller": "<name>"}'
WHY_REPLY = 'Reply: {"seller": "<name>", "why": "<one short clause>"}'

# Three protocols, not two. The published table was measured through
# `LLMBuyer`, which asks for a `why` clause and caps nothing; this deployment
# asks for the name alone inside 64 tokens with thinking off. A model with no
# reasoning pass and no `why` field has nowhere to do the arithmetic, so
# "cannot compute" and "was given no room to" look identical unless they are
# separated.
# The published table was measured through `LLMBuyer`, whose system prompt
# contradicts the board twice: it says sellers do not publish their failure
# rates while the user message prints all three, and it says a failed delivery
# costs the buyer the price when world.py charges only the loss share. Whether
# that mattered can only show where the model is computing at all, which is
# with thinking on — so the two system prompts are compared there.
CONDITIONS = [
    ('plain', SYSTEM, PLAIN_REPLY, 64, False),      # what runs 1-3 trained on
    ('parity', SYSTEM, WHY_REPLY, 256, False),      # the published reply format
    ('think', SYSTEM, PLAIN_REPLY, None, True),     # reasoning, with room
    ('t_pub', LLMBuyer.SYSTEM, WHY_REPLY, None, True),   # the published prompt
    ('t_fix', SYSTEM, WHY_REPLY, None, True),            # the same, corrected
]


def ask(text, system, reply_line, budget, thinking):
    body = {'model': a.model, 'temperature': 0.7,
            'max_tokens': budget or a.budget,
            'chat_template_kwargs': {'enable_thinking': thinking},
            'messages': [{'role': 'system', 'content': system},
                         {'role': 'user', 'content':
                          text.replace(PLAIN_REPLY, reply_line)}]}
    j = httpx.post(URL, json=body, timeout=900).json()['choices'][0]
    return j['message']['content'], j['finish_reason']


def probe(i):
    """One board, both conditions. Independent, so the boards run together."""
    seed = a.start + i
    pos = Driver(seed, **EVAL).pending          # the opening pick, no history
    row = dict(seed=seed, want=_evbest(pos))
    for name, system, reply, budget, thinking in CONDITIONS:
        txt, fin = ask(pos.text, system, reply, budget, thinking)
        got = _parse_json(txt)
        pick = str(got.get('seller', '')).strip()
        row[name] = dict(pick=pick, hit=int(pick == row['want']),
                         ok=bool(got), truncated=fin == 'length', chars=len(txt))
    return row


with cf.ThreadPoolExecutor(max_workers=12) as pool:
    rows = []
    for row in pool.map(probe, range(a.n)):
        rows.append(row)
        print(f"  {len(rows)}/{a.n}  want {row['want']}  "
              + "  ".join(f"{n} {row[n]['pick'] or '-'}"
                          for n, *_ in CONDITIONS), flush=True)

out = {}
for key, *_ in CONDITIONS:
    got = [r for r in rows if r[key]['ok']]
    out[key] = dict(
        accuracy=st.mean(r[key]['hit'] for r in rows),
        # a budget too small to reach the JSON scores zero and says nothing;
        # this is the score among the replies that actually answered
        of_answered=st.mean(r[key]['hit'] for r in got) if got else float('nan'),
        parsed=st.mean(r[key]['ok'] for r in rows),
        truncated=st.mean(r[key]['truncated'] for r in rows),
        chars=st.mean(r[key]['chars'] for r in rows))
out['n'] = len(rows)
out['both'] = st.mean(int(r['plain']['hit'] == r['think']['hit']) for r in rows)

# The bar is not 1/3. If the EV-best seller is not uniform across boards then
# answering the same name every time already beats chance, and a model with a
# name preference would collect that for free.
want = {}
for r in rows:
    want[r['want']] = want.get(r['want'], 0) + 1
out['answer_key'] = want
out['always_one_name'] = max(want.values()) / len(rows)
for key, *_ in CONDITIONS:
    picks = {}
    for r in rows:
        picks[r[key]['pick'] or '-'] = picks.get(r[key]['pick'] or '-', 0) + 1
    out[key]['picks'] = picks
json.dump(dict(summary=out, rows=rows), open(a.out, 'w'), indent=1)

print(f"\n  opening pick against the EV-best seller, n={len(rows)}\n")
print(f"  {'':<10}{'accuracy':>10}{'of answered':>13}{'parsed':>9}"
      f"{'truncated':>11}{'chars':>8}")
LABELS = (('plain', 'name only'), ('parity', 'with why'), ('think', 'thinking'),
          ('t_pub', 'published'), ('t_fix', 'corrected'))
for key, label in LABELS:
    d = out[key]
    print(f"  {label:<10}{d['accuracy']:>10.2f}{d['of_answered']:>13.2f}"
          f"{d['parsed']:>9.2f}{d['truncated']:>11.2f}{d['chars']:>8.0f}")
print(f"\n  answer key {out['answer_key']}"
      f"  -> always naming one seller scores {out['always_one_name']:.2f},"
      f" chance 0.33")
for key, label in LABELS:
    print(f"  {label:<10}picked {out[key]['picks']}")
print(f"  the two agree on {out['both']:.0%} of boards.")
