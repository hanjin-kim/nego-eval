"""Is the curve going anywhere, and how firmly can that be said?

"Flat" by eye is not a claim. The per-point error bars are the wrong ruler for
it: they are the spread across boards, and every point is measured on the same
boards, so that spread is common to all of them and cancels in the comparison.
What bounds a trend is the scatter of the points around a line, which is what
this fits.

The output to read is the last column: the change over the whole run, with an
interval. A run that moved nothing still says how much movement it ruled out.
"""
import argparse, json, pathlib, statistics as st


def trend(rows, key='surplus'):
    xs = [r['step'] for r in rows]
    ys = [r[key] for r in rows]
    n = len(xs)
    if n < 3:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys)]
    s = (sum(r * r for r in resid) / (n - 2)) ** 0.5
    se = s / sxx ** 0.5
    span = max(xs) - min(xs)
    return dict(n=n, slope=slope, se=se, span=span,
                total=slope * span, total_se=se * span)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='*',
                    default=['deploy/out/curve.json', 'deploy/out3/curve.json'])
    a = ap.parse_args()
    print(f"  {'run':<26}{'metric':<10}{'n':>3}{'per step':>12}"
          f"{'over the run':>18}")
    for path in a.paths:
        p = pathlib.Path(path)
        if not p.exists():
            continue
        rows = json.loads(p.read_text())
        for key in ('surplus', 'g1', 'gk'):
            t = trend(rows, key)
            if t is None:
                print(f"  {p.parent.name:<26}{key:<10}{len(rows):>3}"
                      f"{'  (needs 3 points)':>30}")
                continue
            print(f"  {p.parent.name:<26}{key:<10}{t['n']:>3}"
                  f"{t['slope']:>+9.3f}{'':>3}"
                  f"{t['total']:>+11.1f} ± {t['total_se']:.1f}")
