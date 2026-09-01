"""What counts as the training run having worked, written before it runs.

Three numbers, all three required. Any one of them alone can be bought at the
expense of the others, and the tabular study showed exactly how: terminal reward
sent the opening-move accuracy from 0.70 to 0.14 while raising the game-2 figure
to +0.25, which passes a naive "the ledger now helps" test while the surplus sat
at −134. A run scored on that alone would have been called a success.

    surplus > 0        it actually beats a policy that ignores the record
    g1 not worse       the move with no history behind it did not get sold off
    gk - g1 > 0        and the ledger, when it arrives, now helps

The reference points are the eval preset, twelve rounds by four games, which is
what every published figure was measured on. Do not score a run on the training
preset.
"""
import json, sys, statistics as st

TOL_G1 = 0.05          # how much of the no-history move may be given up


def verdict(before: dict, after: dict) -> dict:
    """`before` and `after` are {'surplus','g1','gk'} on the eval preset."""
    checks = {
        'surplus > 0':
            (after['surplus'], after['surplus'] > 0),
        f"g1 not worse than before − {TOL_G1}":
            (after['g1'] - before['g1'], after['g1'] >= before['g1'] - TOL_G1),
        'gk − g1 > 0':
            (after['gk'] - after['g1'], after['gk'] - after['g1'] > 0),
    }
    return dict(checks=checks, passed=all(ok for _, ok in checks.values()))


def show(before, after):
    v = verdict(before, after)
    print(f"  {'':<34}{'전':>9}{'후':>9}")
    for k in ('surplus', 'g1', 'gk'):
        print(f"  {k:<34}{before[k]:>9.2f}{after[k]:>9.2f}")
    print()
    print(f"  {'판정 조건':<38}{'값':>8}   결과")
    for k, (val, ok) in v['checks'].items():
        print(f"  {k:<38}{val:>8.2f}   {'통과' if ok else '실패'}")
    print(f"\n  전체: {'성공' if v['passed'] else '실패'}")
    return v


if __name__ == '__main__':
    if len(sys.argv) == 3:
        show(json.load(open(sys.argv[1])), json.load(open(sys.argv[2])))
    else:
        # the tabular study, replayed through the verdict it would have faced
        print("표 학습자 결과를 이 판정에 걸어보면 (학습 프리셋 기준):\n")
        base = dict(surplus=-131, g1=0.70, gk=0.48)      # no shaping
        for lbl, after in (('종단 보상', dict(surplus=-134, g1=0.14, gk=0.38)),
                           ('조밀 보상', dict(surplus=-44, g1=0.70, gk=0.73))):
            print(f"── {lbl}")
            show(base, after)
            print()
