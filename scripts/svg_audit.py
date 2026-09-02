"""Measure every label against the canvas and against every other label.

Lives on its own because two generators need it. Three rounds of "the
labels overlap" were caught by eye and fixed by eye, which is how the
fourth got through.
"""
import pathlib


def audit(path):
    """Returns (off-canvas, overlapping) labels.

    Text width is approximated from the character count and the font size —
    crude, and enough to catch a collision a reader would notice.
    """
    import re
    s = pathlib.Path(path).read_text()
    W, H = map(int, re.search(r'viewBox="0 0 (\d+) (\d+)"', s).groups())
    box = []
    for m in re.finditer(r'<text\b([^>]*)>([^<]*)</text>', s):
        a, txt = m.group(1), m.group(2)
        num = lambda k, d: float(re.search(rf'{k}="([-\d.]+)"', a).group(1)) \
            if re.search(rf'{k}="([-\d.]+)"', a) else d
        anc = re.search(r'text-anchor="(\w+)"', a)
        anc = anc.group(1) if anc else 'start'
        x, y, fs = num(r'\bx', 0), num(r'\by', 0), num('font-size', 11)
        w = len(txt) * fs * 0.6
        left = x if anc == 'start' else (x - w if anc == 'end' else x - w / 2)
        box.append((left, left + w, y - fs * 0.8, y + fs * 0.25, txt))
    off = [b for b in box if b[0] < 0 or b[1] > W or b[2] < 0 or b[3] > H]
    hit = [(a[4], b[4]) for i, a in enumerate(box) for b in box[i + 1:]
           if a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]]
    return off, hit



def report(paths) -> bool:
    """Print what collides. True if anything did."""
    bad = False
    for f in sorted(paths):
        off, hit = audit(f)
        if off or hit:
            bad = True
            print(f"  {f.name}: {len(off)} off-canvas, {len(hit)} overlapping")
            for t in off[:3]:
                print(f"     off:     {t[4]!r}")
            for a, b in hit[:4]:
                print(f"     overlap: {a!r} / {b!r}")
    if not bad:
        print("  labels: all clear")
    return bad
