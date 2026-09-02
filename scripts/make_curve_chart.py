"""The two training runs, on one pair of axes, against the policy map.

A learning curve on its own says whether a number moved. It does not say
whether the number had anywhere to move to, and for this board that is the
question a reader arrives with: the null sits at zero, the trained model sits
six hundred below it, and roughly 85% of that gap is competence at reading the
quote sheet rather than anything relational. So the references from
`policy_map.py` are drawn in, and the curves are read against them.

Run 2 credits a decision with the whole match. Run 3 credits it with the eight
rounds it can still reach. The runs are identical otherwise, so the distance
between the curves is that one flag.
"""
import json, pathlib, sys

sys.path.insert(0, 'scripts')
from svg_audit import report

OUT = pathlib.Path('docs'); OUT.mkdir(exist_ok=True)
W, H = 720, 400
PAD = dict(l=64, r=150, t=42, b=48)

LIGHT = dict(bg='#FFFFFF', ink='#131A20', dim='#57646F', faint='#8A97A2',
             rule='#D4DCE1', live='#0E6E6B', loss='#A8322A', soft='#E2F0EF')
DARK = dict(bg='#0E1317', ink='#E4EAEF', dim='#97A5AF', faint='#6C7A85',
            rule='#2A353D', live='#46B8B0', loss='#E08074', soft='#123330')

# measured by scripts/policy_map.py on the same eval seeds
MARKS = [(0, 'the null: EV + counter'),
         (-194, 'null picks, accepts'),
         (-641, 'random picks, accepts')]

RUNS = [('deploy/out/curve.json', 'run 2 · whole match', 'loss'),
        ('deploy/out3/curve.json', 'run 3 · eight rounds', 'live')]


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def load():
    out = []
    for path, label, key in RUNS:
        p = pathlib.Path(path)
        if not p.exists():
            continue
        rows = json.loads(p.read_text())
        if rows:
            out.append((label, key, rows))
    return out


def draw(runs, p):
    if not runs:
        return None
    xs = [r['step'] for _, _, rows in runs for r in rows]
    ys = [r['surplus'] - r['se'] for _, _, rows in runs for r in rows]
    ys += [r['surplus'] + r['se'] for _, _, rows in runs for r in rows]
    ys += [m for m, _ in MARKS]
    x0, x1 = 0, max(xs + [120])
    y0, y1 = min(ys) - 40, max(ys) + 40
    px = lambda v: PAD['l'] + (v - x0) / (x1 - x0) * (W - PAD['l'] - PAD['r'])
    py = lambda v: H - PAD['b'] - (v - y0) / (y1 - y0) * (H - PAD['t'] - PAD['b'])

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'font-family="ui-sans-serif,system-ui,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="{p["bg"]}"/>',
         f'<text x="{PAD["l"]}" y="24" font-size="13" font-weight="600" '
         f'fill="{p["ink"]}">Surplus over the record-blind rule, during training</text>']

    for val, label in MARKS:                       # the map, behind the curves
        y = py(val)
        o.append(f'<line x1="{px(x0)}" y1="{y:.1f}" x2="{px(x1)}" y2="{y:.1f}" '
                 f'stroke="{p["rule"]}" stroke-width="1" stroke-dasharray="3 3"/>')
        o.append(f'<text x="{px(x1) + 8}" y="{y + 3:.1f}" font-size="9" '
                 f'fill="{p["faint"]}">{esc(label)}</text>')

    for val in (x0, 30, 60, 90, x1):               # x ticks
        o.append(f'<text x="{px(val):.1f}" y="{H - PAD["b"] + 18}" font-size="10" '
                 f'text-anchor="middle" fill="{p["dim"]}">{val}</text>')
    o.append(f'<text x="{px((x0 + x1) / 2):.1f}" y="{H - 12}" font-size="10" '
             f'text-anchor="middle" fill="{p["dim"]}">optimizer step</text>')

    for val, _ in MARKS:                           # y ticks, on the references
        o.append(f'<text x="{PAD["l"] - 10}" y="{py(val) + 3:.1f}" font-size="10" '
                 f'text-anchor="end" fill="{p["dim"]}">{val:+d}</text>')

    for label, key, rows in runs:
        col = p[key]
        pts = [(px(r['step']), py(r['surplus'])) for r in rows]
        band = ([f"{px(r['step']):.1f},{py(r['surplus'] + r['se']):.1f}" for r in rows]
                + [f"{px(r['step']):.1f},{py(r['surplus'] - r['se']):.1f}"
                   for r in reversed(rows)])
        o.append(f'<polygon points="{" ".join(band)}" fill="{col}" opacity="0.13"/>')
        o.append('<polyline points="' + ' '.join(f'{a:.1f},{b:.1f}' for a, b in pts)
                 + f'" fill="none" stroke="{col}" stroke-width="2"/>')
        for a, b in pts:
            o.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="3" fill="{col}"/>')
        # inside the plot, anchored to the last point, so the right margin
        # stays the references' and the two never fight for it
        dy = -11 if key == RUNS[0][2] else 19
        o.append(f'<text x="{pts[-1][0] - 6:.1f}" y="{pts[-1][1] + dy:.1f}" '
                 f'text-anchor="end" font-size="10" font-weight="600" '
                 f'fill="{col}">{esc(label)}</text>')

    o.append('</svg>')
    return '\n'.join(o)


if __name__ == '__main__':
    runs = load()
    if not runs:
        raise SystemExit('no curve.json yet')
    for name, pal in (('light', LIGHT), ('dark', DARK)):
        (OUT / f'curve.{name}.svg').write_text(draw(runs, pal))
    print(f"  {len(runs)} run(s), "
          f"{', '.join(f'{lab} {len(rows)} points' for lab, _, rows in runs)}")
    if report(OUT.glob('curve.*.svg')):
        raise SystemExit('labels collide')
