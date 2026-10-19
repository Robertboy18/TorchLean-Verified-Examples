"""Draw the error-versus-compute figure for the matching-sampling benchmark from results-msp.json.

Depth 2, 64 sign inputs, sigma_w^2 = 2, sigma_b = 0.1. Each panel is one width. The sampling line is
the exact sampling error at each compute budget; the markers are the structural estimators.
"""

import json
import math
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
INK, MUTED, GRID, PAPER = "#252b32", "#69737d", "#e0e5e8", "#ffffff"
SERIES = {"C": "#2a78d6", "D": "#eb6834", "A8": "#1baf7a"}  # validated categorical slots 1-3
NAMES = {"C": "covariance propagation (Gaussian closure)", "D": "diagonal closure (no covariances)",
         "A8": "averaging, 8 of 64 signs kept (certified bound)"}
SHORT = {"C": "covariance", "D": "diagonal", "A8": "averaging"}
SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
FONT = "IBM Plex Sans, system-ui, sans-serif"
X0, X1 = 1e3, 1e13
Y0, Y1 = 1e-3, 2.0


def rows():
    record = json.load(open(HERE / "results-msp.json"))
    out = []
    for setting in record["settings"]:
        s = setting["setting"]
        if (s["depth"], s["input"], s["sigma_w2"], s["sigma_b"]) != (2, "sign", 2.0, 0.1):
            continue
        t = setting["summary"]["T1"]
        per_sample = t["S@Cx1"]["rel_rmse_expected"] * math.sqrt(t["S@Cx1"]["n"])
        out.append(dict(width=s["width"], sample_flops=setting["S_flops_per_sample"]["T1"],
                        per_sample=per_sample,
                        points={k: (t[k]["flops"], t[k]["rel_rmse"]) for k in SERIES},
                        gain=t["S@Cx1"]["rel_rmse_expected"] / t["C"]["rel_rmse"]))
    return sorted(out, key=lambda r: r["width"])


def marker(kind, x, y, label):
    color = SERIES[kind]
    tip = f"<title>{escape(label)}</title>"
    if kind == "C":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{color}" stroke="{PAPER}" stroke-width="2">{tip}</circle>'
    if kind == "D":
        return (f'<rect x="{x-5:.1f}" y="{y-5:.1f}" width="10" height="10" rx="1.5" fill="{color}" '
                f'stroke="{PAPER}" stroke-width="2">{tip}</rect>')
    return (f'<path d="M{x:.1f},{y-6.5:.1f} L{x+6:.1f},{y+4.5:.1f} L{x-6:.1f},{y+4.5:.1f} Z" fill="{color}" '
            f'stroke="{PAPER}" stroke-width="2" stroke-linejoin="round">{tip}</path>')


def panel(r, left, top, w, h, labels):
    def px(v): return left + w * (math.log10(v) - math.log10(X0)) / (math.log10(X1) - math.log10(X0))
    def py(v): return top + h * (math.log10(Y1) - math.log10(v)) / (math.log10(Y1) - math.log10(Y0))
    out = [f'<text x="{left}" y="{top-24}" font-size="13" font-weight="600" fill="{INK}">width {r["width"]}</text>',
           f'<text x="{left}" y="{top-9}" font-size="11.5" fill="{MUTED}">covariance: {r["gain"]:.1f}× less error '
           f'than equal-cost sampling</text>']
    for e in range(3, 14, 2):
        x = px(10 ** e)
        out.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+h}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{top+h+15}" font-size="10.5" fill="{MUTED}" text-anchor="middle">10{str(e).translate(SUP)}</text>')
    for v, lab in ((1e-3, "0.1%"), (1e-2, "1%"), (1e-1, "10%"), (1.0, "100%")):
        y = py(v)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+w}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{left-6}" y="{y+3.5:.1f}" font-size="10.5" fill="{MUTED}" text-anchor="end">{lab}</text>')
    out.append(f'<rect x="{left}" y="{top}" width="{w}" height="{h}" fill="none" stroke="{GRID}"/>')
    c0, c1 = r["sample_flops"], X1
    seg = [(c, r["per_sample"] / math.sqrt(c / r["sample_flops"])) for c in (c0, c1)]
    seg = [(c, max(e, Y0)) for c, e in seg]
    x_end = px(min(c1, r["sample_flops"] * (r["per_sample"] / Y0) ** 2))
    out.append(f'<line x1="{px(seg[0][0]):.1f}" y1="{py(min(seg[0][1], Y1)):.1f}" x2="{x_end:.1f}" '
               f'y2="{py(Y0):.1f}" stroke="{MUTED}" stroke-width="2" stroke-dasharray="5 4">'
               f'<title>Sampling: exact error {r["per_sample"]*100:.0f}% at one sample, falling as 1/sqrt(compute)</title></line>')
    for kind in ("A8", "D", "C"):
        flops, rel = r["points"][kind]
        x, y = px(flops), py(rel)
        out.append(marker(kind, x, y, f'{NAMES[kind]}, width {r["width"]}: {rel*100:.3g}% error at {flops:.3g} FLOPs'))
        if labels:
            dx, dy, anchor = {"C": (0, 22, "middle"), "D": (0, 22, "middle"), "A8": (10, 4, "start")}[kind]
            out.append(f'<text x="{x+dx:.1f}" y="{y+dy:.1f}" font-size="11" fill="{INK}" text-anchor="{anchor}">'
                       f'{SHORT[kind]}</text>')
    if labels:
        at = r["sample_flops"] * (r["per_sample"] / 0.25) ** 2
        out.append(f'<text x="{px(at)+8:.1f}" y="{py(0.25):.1f}" font-size="11" fill="{MUTED}">sampling</text>')
    return out


def legend(left, top, compact):
    items = [("S", "sampling (exact error at that compute)")] + [(k, NAMES[k]) for k in ("C", "D", "A8")]
    out, x, y = [], left, top
    for kind, name in items:
        if kind == "S":
            out.append(f'<line x1="{x}" y1="{y-4}" x2="{x+22}" y2="{y-4}" stroke="{MUTED}" stroke-width="2" stroke-dasharray="5 4"/>')
            tx = x + 28
        else:
            out.append(marker(kind, x + 8, y - 4, name)); tx = x + 20
        out.append(f'<text x="{tx}" y="{y}" font-size="11.5" fill="{INK}">{escape(name)}</text>')
        if compact:
            y += 18
        elif kind in ("C",):
            x, y = left, y + 18
        else:
            x = left + 330
    return out, y


def draw(compact):
    data = rows()
    W = 360 if compact else 720
    out_legend, ly = legend(18, 26, compact)
    top0 = ly + 74
    ytitle = (f'<text x="18" y="{top0-48}" font-size="11.5" fill="{MUTED}">error, relative to the spread of the '
              f'true mean across networks</text>')
    pw, ph = (W - 78, 210) if compact else (282, 190)
    cells = [(60, top0 + i * (ph + 92)) for i in range(4)] if compact else \
            [(60 + (i % 2) * (pw + 74), top0 + (i // 2) * (ph + 96)) for i in range(4)]
    H = cells[-1][1] + ph + 52
    body = []
    for i, (r, (l, t)) in enumerate(zip(data, cells)):
        body += panel(r, l, t, pw, ph, labels=(i == 0))
    meta = escape(json.dumps(data))
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
            f'aria-labelledby="title description" font-family="{FONT}">'
            f'<title id="title">Estimation error against compute on random ReLU networks</title>'
            f'<desc id="description">Four panels, one per width, for depth-2 random ReLU networks with 64 sign inputs. '
            f'Each shows relative error on a log scale against floating-point operations on a log scale. The dashed line '
            f'is the exact error of plain sampling at each compute budget. Covariance propagation lies below the line at '
            f'every width; means-and-variances lies below it at its small cost; certified averaging lies above it.</desc>'
            f'<metadata id="measurements">{meta}</metadata>'
            f'<rect width="{W}" height="{H}" fill="{PAPER}"/>')
    foot = (f'<text x="{W/2:.0f}" y="{H-12}" font-size="11.5" fill="{MUTED}" text-anchor="middle">'
            f'compute per estimate (floating-point operations)</text></svg>')
    return head + "".join(out_legend + [ytitle] + body) + foot


if __name__ == "__main__":
    (HERE / "msp-error-vs-cost.svg").write_text(draw(False) + "\n")
    (HERE / "msp-error-vs-cost-mobile.svg").write_text(draw(True) + "\n")
    print("wrote msp-error-vs-cost.svg and msp-error-vs-cost-mobile.svg")
