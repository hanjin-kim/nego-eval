"""Three charts, drawn from the data rather than transcribed from it.

Tables are fine for looking a number up and poor for the three things this
project actually wants a reader to see at a glance: where the models sit
relative to two references and how uncertain that is, which direction each one
moves when a ledger appears, and how the value of memory tracks how often the
relationship is interrupted.

Everything is recomputed here — the references are replayed on the same seeds the
models played, so nothing is copied from a previous run and a stale number cannot
survive a rerun. Output is plain SVG in two palettes, because a chart that is
legible on GitHub in the morning should still be legible in dark mode at night.
"""
import json, glob, statistics as st, pathlib, sys
sys.path.insert(0, 'src')
from nego_eval.game.buyers import BoardOnlyBuyer, EVBuyer
from nego_eval.game.match import Match
from nego_eval.game.table4 import VALUE, cast_for

L, R, G, S = 110, 12, 4, 10
OUT = pathlib.Path('docs'); OUT.mkdir(exist_ok=True)

LIGHT = dict(bg='#FFFFFF', ink='#131A20', dim='#57646F', faint='#8A97A2',
             rule='#D4DCE1', live='#0E6E6B', loss='#A8322A', soft='#E2F0EF')
DARK = dict(bg='#0E1317', ink='#E4EAEF', dim='#97A5AF', faint='#6C7A85',
            rule='#2A353D', live='#46B8B0', loss='#E08074', soft='#123330')

_ref = {}


def ref(kind, seed):
    if (kind, seed) not in _ref:
        make = cast_for(seed, loss=L)[0]
        fac = BoardOnlyBuyer if kind == 'board' else EVBuyer
        r = Match(buyer_factory=fac, seller_factory=make, games=G, rounds=R,
                  loss=L, value=VALUE, seed=seed, carry_over=True, step=S).run()
        _ref[(kind, seed)] = r.profit if r.rounds else None
    return _ref[(kind, seed)]


def models():
    out = []
    for f in glob.glob('data/f4_*.json'):
        if '_off' in f:
            continue
        d = json.load(open(f))
        per = {int(k): v for k, v in (d.get('per_seed') or {}).items()}
        if len(per) < 5:
            continue
        seeds = sorted(per)
        rule = [per[s] - ref('rule', s) for s in seeds]
        board = [per[s] - ref('board', s) for s in seeds]
        se = lambda v: st.stdev(v) / len(v) ** 0.5
        out.append(dict(name=d['model'].split('/')[-1], n=len(seeds),
                        rule=st.mean(rule), rule_se=se(rule),
                        board=st.mean(board), board_se=se(board),
                        g1=d.get('g1'), gk=d.get('gk')))
    return sorted(out, key=lambda r: -r['rule'])


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;')


def spread(items, gap=13.0):
    """Push labels apart so ties do not stack.

    Three models end games 2+ on exactly 0.61 and two open on 0.67. Drawn at the
    value they land on, those labels sit on top of each other and the chart reads
    as though two models were missing. Positions move; the dot each one belongs
    to does not, so a leader line is drawn where a label had to travel.
    """
    order = sorted(range(len(items)), key=lambda i: items[i])
    out = list(items)
    for k in range(1, len(order)):
        a, b = order[k - 1], order[k]
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def chart_models(rows, p):
    """Dot-and-whisker against the two references, with an axis under it.

    Labelling the reference lines in place kept failing: the two lines are fifty
    pixels apart, their names are three times that, and moving one to the right
    of its line ran it off the canvas. They are a legend now, which cannot
    collide with anything, and the scale they are on finally has ticks.
    """
    W, ML, MR, TOP, RH = 760, 176, 92, 84, 34
    H = TOP + RH * len(rows) + 52
    lo, hi = -520, 60
    x = lambda v: ML + (v - lo) / (hi - lo) * (W - ML - MR)
    base = TOP + RH * len(rows) + 6
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
         f'height="{H}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace">',
         f'<rect width="{W}" height="{H}" fill="{p["bg"]}"/>',
         f'<text x="8" y="18" font-size="12" fill="{p["ink"]}">'
         f'Every model against the rule it loses to</text>',
         f'<text x="8" y="34" font-size="11" fill="{p["dim"]}">'
         f'Whole matches, paired on the same seeds. Bars are one standard error.</text>']
    # legend, on its own row, where nothing can reach it
    for i, (lab, col) in enumerate((('the hand-written rule, 996', p['live']),
                                    ('quote sheet only, 934', p['faint']))):
        lx = 8 + i * 250
        o.append(f'<line x1="{lx}" y1="{TOP-30}" x2="{lx+18}" y2="{TOP-30}" '
                 f'stroke="{col}" stroke-width="1.4" stroke-dasharray="3 3"/>')
        o.append(f'<text x="{lx+24}" y="{TOP-26}" font-size="10.5" fill="{col}">{lab}</text>')
    for v, col in ((0, p['live']), (-62, p['faint'])):
        o.append(f'<line x1="{x(v):.0f}" y1="{TOP-14}" x2="{x(v):.0f}" y2="{base}" '
                 f'stroke="{col}" stroke-width="1" stroke-dasharray="3 3"/>')
    for i, r in enumerate(rows):
        y = TOP + i * RH + 10
        col = p['loss'] if r['rule'] + r['rule_se'] < 0 else p['ink']
        o.append(f'<text x="{ML-14}" y="{y+4}" font-size="12" fill="{p["ink"]}" '
                 f'text-anchor="end">{esc(r["name"])}</text>')
        o.append(f'<text x="{ML-14}" y="{y+16}" font-size="9.5" fill="{p["faint"]}" '
                 f'text-anchor="end">n={r["n"]}</text>')
        a, b = x(r['rule'] - r['rule_se']), x(r['rule'] + r['rule_se'])
        o.append(f'<line x1="{a:.0f}" y1="{y}" x2="{b:.0f}" y2="{y}" '
                 f'stroke="{col}" stroke-width="1.6"/>')
        for e in (a, b):
            o.append(f'<line x1="{e:.0f}" y1="{y-4}" x2="{e:.0f}" y2="{y+4}" '
                     f'stroke="{col}" stroke-width="1.6"/>')
        o.append(f'<circle cx="{x(r["rule"]):.0f}" cy="{y}" r="4" fill="{col}"/>')
        # values live in the right margin, in a column, clear of the plot
        o.append(f'<text x="{W-10}" y="{y+4}" font-size="11" fill="{col}" '
                 f'text-anchor="end">{r["rule"]:+.0f} \u00b1{r["rule_se"]:.0f}</text>')
    o.append(f'<line x1="{ML}" y1="{base}" x2="{W-MR}" y2="{base}" '
             f'stroke="{p["rule"]}" stroke-width="1"/>')
    for v in range(-500, 61, 100):
        o.append(f'<line x1="{x(v):.0f}" y1="{base}" x2="{x(v):.0f}" y2="{base+5}" '
                 f'stroke="{p["rule"]}" stroke-width="1"/>')
        o.append(f'<text x="{x(v):.0f}" y="{base+18}" font-size="10" fill="{p["faint"]}" '
                 f'text-anchor="middle">{v}</text>')
    o.append(f'<text x="{ML}" y="{base+36}" font-size="10" fill="{p["faint"]}">'
             f'score, relative to the hand-written rule</text>')
    o.append('</svg>')
    return '\n'.join(o)


def chart_slope(rows, p):
    """Opening pick before a ledger exists, and after."""
    W, H, ML, MR, TOP, BOT = 780, 400, 214, 196, 58, 330
    y = lambda v: BOT - v * (BOT - TOP) / 1.0
    rows = [r for r in rows if r.get('g1') is not None and r.get('gk') is not None]
    rows = rows + [dict(name='the hand-written rule', g1=0.52, gk=0.71)]
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
         f'height="{H}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace">',
         f'<rect width="{W}" height="{H}" fill="{p["bg"]}"/>',
         f'<text x="8" y="18" font-size="12" fill="{p["ink"]}">'
         f'Which way each one moves when a ledger appears</text>',
         f'<text x="8" y="34" font-size="11" fill="{p["dim"]}">'
         f'Opening pick of the first game, and of every game after it.</text>']
    o.append(f'<line x1="{ML}" y1="{y(0.333):.0f}" x2="{W-MR}" y2="{y(0.333):.0f}" '
             f'stroke="{p["rule"]}" stroke-dasharray="3 3"/>')
    o.append(f'<text x="{ML-12}" y="{y(0.333)+4:.0f}" font-size="10" '
             f'fill="{p["faint"]}" text-anchor="end">chance 0.33</text>')
    for xx, lab in ((ML, 'game 1 - no history'), (W - MR, 'games 2+ - ledger carried')):
        o.append(f'<text x="{xx}" y="{BOT+30}" font-size="11" fill="{p["dim"]}" '
                 f'text-anchor="middle">{lab}</text>')
    ly = spread([y(r['g1']) for r in rows])
    ry = spread([y(r['gk']) for r in rows])
    for r, a, b in zip(rows, ly, ry):
        up = r['gk'] > r['g1']
        col = p['live'] if up else p['loss']
        big = abs(r['gk'] - r['g1']) > 0.15
        o.append(f'<line x1="{ML}" y1="{y(r["g1"]):.0f}" x2="{W-MR}" y2="{y(r["gk"]):.0f}" '
                 f'stroke="{col}" stroke-width="{2.2 if big else 1.2}" '
                 f'opacity="{0.95 if big else 0.5}"/>')
        for cx, cy in ((ML, y(r['g1'])), (W - MR, y(r['gk']))):
            o.append(f'<circle cx="{cx}" cy="{cy:.0f}" r="3.4" fill="{col}"/>')
        # leaders, only where a label had to be moved off its dot
        if abs(a - y(r['g1'])) > 1.5:
            o.append(f'<line x1="{ML-6}" y1="{y(r["g1"]):.0f}" x2="{ML-12}" y2="{a:.0f}" '
                     f'stroke="{p["rule"]}" stroke-width="0.8"/>')
        if abs(b - y(r['gk'])) > 1.5:
            o.append(f'<line x1="{W-MR+6}" y1="{y(r["gk"]):.0f}" x2="{W-MR+12}" y2="{b:.0f}" '
                     f'stroke="{p["rule"]}" stroke-width="0.8"/>')
        o.append(f'<text x="{ML-16}" y="{a+4:.0f}" font-size="11" fill="{p["ink"]}" '
                 f'text-anchor="end">{esc(r["name"])}  {r["g1"]:.2f}</text>')
        o.append(f'<text x="{W-MR+16}" y="{b+4:.0f}" font-size="11" fill="{col}">'
                 f'{r["gk"]:.2f}  {r["gk"]-r["g1"]:+.2f}</text>')
    o.append('</svg>')
    return '\n'.join(o)


def chart_shape(p):
    """What memory is worth as the bell rings more often.

    The title moved off the plot's top-left corner, where it ran into the
    highest gridline label, and the axis caption moved out from under the tick
    labels it was sitting on.
    """
    b4 = json.load(open('data/shape4.json'))
    W, H, ML, TOP, BOT = 760, 330, 132, 62, 236
    xs = [r['games'] - 1 for r in b4]
    vals4 = [r['on'] - r['off'] for r in b4]
    hi = 200
    x = lambda g: ML + (g - min(xs)) / (max(xs) - min(xs)) * (W - ML - 40)
    y = lambda v: BOT - v / hi * (BOT - TOP)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
         f'height="{H}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace">',
         f'<rect width="{W}" height="{H}" fill="{p["bg"]}"/>',
         f'<text x="8" y="18" font-size="12" fill="{p["ink"]}">'
         f'What carrying the ledger is worth</text>',
         f'<text x="8" y="34" font-size="11" fill="{p["dim"]}">'
         f'Total rounds held at 48; only where the boundaries fall changes.</text>']
    for v in (0, 50, 100, 150, 200):
        o.append(f'<line x1="{ML}" y1="{y(v):.0f}" x2="{W-40}" y2="{y(v):.0f}" '
                 f'stroke="{p["rule"]}" stroke-width="0.8"/>')
        o.append(f'<text x="{ML-10}" y="{y(v)+4:.0f}" font-size="10" fill="{p["faint"]}" '
                 f'text-anchor="end">{v}</text>')
    pts = ' '.join(f'{x(g):.0f},{y(v):.0f}' for g, v in zip(xs, vals4))
    o.append(f'<polyline points="{pts}" fill="none" stroke="{p["live"]}" stroke-width="2.2"/>')
    for g, v, r in zip(xs, vals4, b4):
        o.append(f'<circle cx="{x(g):.0f}" cy="{y(v):.0f}" r="4" fill="{p["live"]}"/>')
        o.append(f'<text x="{x(g):.0f}" y="{y(v)-12:.0f}" font-size="10.5" '
                 f'fill="{p["live"]}" text-anchor="middle">+{v:.0f}</text>')
        o.append(f'<text x="{x(g):.0f}" y="{BOT+22:.0f}" font-size="10.5" '
                 f'fill="{p["dim"]}" text-anchor="middle">{r["rounds"]}\u00d7{r["games"]}</text>')
        o.append(f'<text x="{x(g):.0f}" y="{BOT+38:.0f}" font-size="9.5" '
                 f'fill="{p["faint"]}" text-anchor="middle">{g}</text>')
    o.append(f'<text x="{ML-22}" y="{BOT+22:.0f}" font-size="9.5" fill="{p["faint"]}" '
             f'text-anchor="end">rounds \u00d7 games</text>')
    o.append(f'<text x="{ML-22}" y="{BOT+38:.0f}" font-size="9.5" fill="{p["faint"]}" '
             f'text-anchor="end">boundaries</text>')
    o.append('</svg>')
    return '\n'.join(o)


rows = models()
for tag, pal in (('light', LIGHT), ('dark', DARK)):
    (OUT / f'models.{tag}.svg').write_text(chart_models(rows, pal))
    (OUT / f'ledger.{tag}.svg').write_text(chart_slope(rows, pal))
    (OUT / f'shape.{tag}.svg').write_text(chart_shape(pal))
print(f"{len(rows)}개 모델 · docs/ 에 SVG 6개")
for r in rows:
    print(f"  {r['name']:<20}{r['rule']:>7.0f}±{r['rule_se']:<4.0f}"
          f"  {r['g1']:.2f}→{r['gk']:.2f}")


from svg_audit import report

if report(OUT.glob('*.svg')):
    raise SystemExit("labels collide")
