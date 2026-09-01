"""Every gemini match, both conditions, merged on the seed.

Runs were extended rather than restarted, so the same arm lives in more than one
file. Pairing is on the seed and nothing else: the two conditions face the same
board, the same cast and the same price sequence, and differ only in whether the
ledger survives the boundary. Unpaired means the board's own spread — about 190
per match — sits on top of an effect the environment values at 64.
"""
import glob, json, statistics as st

def load(carry):
    """Every file for one arm. The tag sits before the model name, so a glob on
    the model alone misses the continuation runs entirely — which it did, and the
    pairing silently stayed at eleven while more matches were being paid for."""
    out = {}
    for f in glob.glob('data_f4*gemini-3.7-flash*.json'):
        if ('_off' in f) == carry:
            continue
        for k, v in json.load(open(f))['per_seed'].items():
            out[int(k)] = v
    return out

on, off = load(True), load(False)
shared = sorted(set(on) & set(off))
d = [on[s] - off[s] for s in shared]
print(f"gemini-3.7-flash · 이월 있음 {len(on)}매치 · 없음 {len(off)}매치 · 짝 {len(d)}쌍\n")
if len(d) > 1:
    se = st.stdev(d) / len(d) ** 0.5
    print(f"  이월 있음   {st.mean(on[s] for s in shared):>7.0f}")
    print(f"  이월 없음   {st.mean(off[s] for s in shared):>7.0f}")
    print(f"  차이       {st.mean(d):>+7.0f} ± {se:.0f}   ({st.mean(d)/se:.1f} 시그마)")
    print(f"\n  환경이 주는 값 (스크립트 정책) +64")
    need = (st.stdev(d) / 32) ** 2
    print(f"  현재 쌍당 표준편차 {st.stdev(d):.0f} → 2시그마로 가르려면 약 {need:.0f}쌍")
else:
    print("  아직 짝이 부족합니다.")
