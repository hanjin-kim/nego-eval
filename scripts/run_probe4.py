"""The models, where staying earns something.

Everything measured about language models so far was measured where buying from
whoever failed least scored chance — a board that punished a rule that is correct
in the world. None of it transfers.

Here the quote sheet publishes a delivery rate beside every price and charges for
it, so reliability needs no estimating and is not free. What the sheet does not
say is what a seller does once a delivery is ruined, and that is only in the
record. The answer is the seller maximising

    rate x (value - price) - (1 - rate) x loss x (1 - share it has absorbed)

which is arithmetic over three numbers the prompt shows. Reading the sheet alone
gets 0.75 of the way; reading the record alone gets 0.64; neither is enough.
"""
import sys, json, time, threading
sys.path.insert(0, 'src')
from nego_eval.sim import llm
from nego_eval.sim.agents import LLMBuyer

P = json.load(open('data_probe4_positions.json'))[:120]
SYSTEM = LLMBuyer.SYSTEM
ROUNDS, VALUE, LOSS = 12, 150, 110
KEYS = ('paid', 'absorb', 'cover', 'compensat', 'borne', 'share of')

#: Light models only. The budget is nearly out and the board-3 run showed the
#: pattern does not need a frontier model to appear.
MODELS = ['deepseek/deepseek-chat', 'anthropic/claude-haiku-4.5',
          'google/gemini-3.7-flash']
GATE = threading.Semaphore(4)
lock = threading.Lock()


def prompt(p):
    q = ", ".join(f"{n}={b['price']} (published delivery rate {b['rate']:.2f})"
                  for n, b in sorted(p['board'].items()))
    return (f"Round 1. Rounds remaining after this one: {ROUNDS - 1}.\n"
            f"Unit value to you if delivery succeeds: {VALUE}.\n"
            f"Loss if delivery fails: {LOSS}.\n"
            f"Quotes: {q}\n\n"
            f"Your record so far:\n{p['ledger']}\n\n"
            f'Reply: {{"seller": "<name>", "why": "<one short clause>"}}')


def best_rate(p):
    return max(p['board'], key=lambda k: p['board'][k]['rate'])


def best_borne(p):
    def b(k):
        a = p['stats'][k]
        return a['paid'] / a['owed'] if a['owed'] else -1
    return max(p['stats'], key=b)


def run(model):
    from nego_eval.sim.llm import ask_json
    t0 = time.time(); rows = []; errs = 0
    for p in P:
        try:
            with GATE:
                d = ask_json(SYSTEM, prompt(p), model=model)
        except Exception as e:
            errs += 1
            if errs > 12:
                with lock: print(f"  {model} 중단 {str(e)[:50]}", flush=True)
                break
            continue
        pick = str(d.get('seller', '')).strip()
        if pick not in p['board']:
            errs += 1; continue
        why = str(d.get('why', '')).lower()
        rows.append(dict(correct=pick == p['key'], rate=pick == best_rate(p),
                         borne=pick == best_borne(p),
                         mentions=any(k in why for k in KEYS), why=why[:90]))
    n = len(rows) or 1
    men = [r for r in rows if r['mentions']]
    r = dict(model=model, n=len(rows), errors=errs,
             correct=sum(x['correct'] for x in rows) / n,
             chose_best_rate=sum(x['rate'] for x in rows) / n,
             chose_best_borne=sum(x['borne'] for x in rows) / n,
             consult=len(men) / n,
             acc_when_consults=(sum(x['correct'] for x in men) / len(men)) if men else None,
             secs=round(time.time() - t0))
    with lock:
        json.dump(dict(r, rows=rows), open(f"data_p4_{model.split('/')[-1]}.json", 'w'), indent=1)
        aw = f"{r['acc_when_consults']:.2f}" if r['acc_when_consults'] is not None else "  - "
        print(f"  {model.split('/')[-1]:<20}{r['correct']:>8.2f}{r['chose_best_rate']:>10.2f}"
              f"{r['chose_best_borne']:>10.2f}{r['consult']:>9.2f}{aw:>10}"
              f"{r['n']:>5}{r['secs']:>7}s", flush=True)


print(f"4번 판 · 위치 {len(P)}개 · 우연 0.33 · 천장 1.00")
print(f"기준선: 호가판만 0.61 · 부담률만 0.79 · 관측 실패율 0.53 · 최저가 0.07\n")
print(f"  {'모델':<20}{'정답률':>8}{'배달률최고':>10}{'부담최고':>10}{'원장언급':>9}"
      f"{'언급시':>10}{'n':>5}{'시간':>8}")
ts = [threading.Thread(target=run, args=(m,)) for m in MODELS]
for t in ts: t.start()
for t in ts: t.join()
print("\nDONE")
