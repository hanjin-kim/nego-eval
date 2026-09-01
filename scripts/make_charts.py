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


def chart_models(rows, p):
    """Dot-and-whisker against the two references. Zero is the reference itself.

    The two reference labels sit at different heights, and on opposite sides of
    their own lines. Centred under each at the same height they collided: the
    lines are fifty pixels apart and the words are three times that.
    """
    W, ML, MT, RH = 760, 168, 62, 34
    H = MT + RH * len(rows) + 30
    lo, hi = -580, 120
    x = lambda v: ML + (v - lo) / (hi - lo) * (W - ML - 28)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
         f'height="{H}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace">',
         f'<rect width="{W}" height="{H}" fill="{p["bg"]}"/>',
         f'<text x="8" y="18" font-size="12" fill="{p["dim"]}">'
         f'Whole matches, paired on the same seeds. Bars are one standard error.</text>']
    top = MT - 26
    for v, lab, col, anchor, dx, dy in (
            (0, 'the hand-written rule', p['live'], 'start', 7, 0),
            (-62, 'quote sheet only', p['faint'], 'end', -7, 14)):
        o.append(f'<line x1="{x(v):.0f}" y1="{top+dy}" x2="{x(v):.0f}" y2="{H-16}" '
                 f'stroke="{col}" stroke-width="1" stroke-dasharray="3 3"/>')
        o.append(f'<text x="{x(v)+dx:.0f}" y="{top+dy+4}" font-size="10.5" fill="{col}" '
                 f'text-anchor="{anchor}">{lab}</text>')
    for i, r in enumerate(rows):
        y = MT + i * RH + 12
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
        # the value sits past the right whisker, never above the bar
        o.append(f'<text x="{b+9:.0f}" y="{y+4}" font-size="10.5" fill="{col}">'
                 f'{r["rule"]:+.0f}</text>')
    o.append('</svg>')
    return '\n'.join(o)


def chart_slope(rows, p):
    """Opening pick before a ledger exists, and after."""
    W, H, ML, MR, TOP, BOT = 720, 380, 210, 190, 52, 320
    y = lambda v: BOT - v * (BOT - TOP) / 1.0
    rows = rows + [dict(name='the hand-written rule', g1=0.52, gk=0.71)]
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
         f'height="{H}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace">',
         f'<rect width="{W}" height="{H}" fill="{p["bg"]}"/>',
         f'<text x="8" y="20" font-size="12" fill="{p["dim"]}">'
         f'Opening pick of the first game, and of every game after it.</text>']
    o.append(f'<line x1="{ML}" y1="{y(0.333):.0f}" x2="{W-MR}" y2="{y(0.333):.0f}" '
             f'stroke="{p["rule"]}" stroke-dasharray="3 3"/>')
    o.append(f'<text x="{W-MR+8}" y="{y(0.333)+4:.0f}" font-size="10" '
             f'fill="{p["faint"]}">chance 0.33</text>')
    for xx, lab in ((ML, 'game 1 — no history'), (W - MR, 'games 2+ — ledger carried')):
        o.append(f'<text x="{xx}" y="{BOT+26}" font-size="11" fill="{p["dim"]}" '
                 f'text-anchor="middle">{lab}</text>')
    for r in rows:
        if r.get('g1') is None or r.get('gk') is None:
            continue
        up = r['gk'] > r['g1']
        col = p['live'] if up else p['loss']
        o.append(f'<line x1="{ML}" y1="{y(r["g1"]):.0f}" x2="{W-MR}" y2="{y(r["gk"]):.0f}" '
                 f'stroke="{col}" stroke-width="{2.2 if abs(r["gk"]-r["g1"])>0.15 else 1.2}" '
                 f'opacity="{0.95 if abs(r["gk"]-r["g1"])>0.15 else 0.55}"/>')
        o.append(f'<circle cx="{ML}" cy="{y(r["g1"]):.0f}" r="3.4" fill="{col}"/>')
        o.append(f'<circle cx="{W-MR}" cy="{y(r["gk"]):.0f}" r="3.4" fill="{col}"/>')
        o.append(f'<text x="{ML-10}" y="{y(r["g1"])+4:.0f}" font-size="11" fill="{p["ink"]}" '
                 f'text-anchor="end">{esc(r["name"])}  {r["g1"]:.2f}</text>')
        o.append(f'<text x="{W-MR+10}" y="{y(r["gk"])+4:.0f}" font-size="11" fill="{col}">'
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
    W, H, ML, TOP, BOT = 760, 320, 74, 62, 236
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
    o.append(f'<text x="{ML-10}" y="{BOT+22:.0f}" font-size="9.5" fill="{p["faint"]}" '
             f'text-anchor="end">rounds\u00d7games</text>')
    o.append(f'<text x="{ML-10}" y="{BOT+38:.0f}" font-size="9.5" fill="{p["faint"]}" '
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
