"""Draw the README figures in assets/ from the frozen paper records.

    python scripts/readme_figures.py

Reads only paper/reference_scores.csv, paper/models.csv and paper/kmer_floor.json, and writes
plain SVG (transparent background; text and axes switch colour with the viewer's light/dark
scheme). No plotting library needed.
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
ASSETS = ROOT / "assets"

LIME, LIME_EDGE, GOLD = "#8EDE3D", "#384C10", "#CA8407"
STYLE = """<style>
  text { font-family: Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; fill: #29332E; font-size: 12px; }
  .num { font-family: 'IBM Plex Mono', ui-monospace, Menlo, monospace; font-size: 11px; }
  .muted { fill: #5E6A64; }
  .title { font-size: 14px; font-weight: 600; }
  .axis { stroke: #29332E; stroke-width: 1; }
  .grid { stroke: #29332E; stroke-opacity: 0.12; stroke-width: 1; }
  .floor { stroke: #5E6A64; stroke-width: 1.2; stroke-dasharray: 5 4; }
  .link { stroke: #5E6A64; stroke-opacity: 0.5; stroke-width: 1.5; }
  @media (prefers-color-scheme: dark) {
    text { fill: #E6E9E7; } .muted { fill: #A3ADA8; }
    .axis { stroke: #E6E9E7; } .grid { stroke: #E6E9E7; }
    .floor, .link { stroke: #A3ADA8; }
  }
</style>"""


def load():
    scores = defaultdict(dict)  # model -> (task, metric, layer) -> value
    with open(PAPER / "reference_scores.csv") as f:
        for r in csv.DictReader(f):
            layer = int(r["layer"]) if r["layer"] else None
            scores[r["model_id"]][(r["task"], r["metric"], layer)] = float(r["value"])
    with open(PAPER / "models.csv") as f:
        params = {r["model_id"]: int(r["params_measured"]) for r in csv.DictReader(f)}
    floor = json.loads((PAPER / "kmer_floor.json").read_text())["tasks"]
    return scores, params, floor


def svg(w, h, title, desc, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        f'role="img" aria-labelledby="t d">\n<title id="t">{title}</title>\n<desc id="d">{desc}</desc>\n'
        f"{STYLE}\n" + "\n".join(body) + "\n</svg>\n"
    )


def is_loam(m):
    return m.startswith("LOAM")


def dot(x, y, m, hollow=False, r=5):
    fill, edge = (LIME, LIME_EDGE) if is_loam(m) else (GOLD, GOLD)
    if hollow:
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="none" stroke="{fill if not is_loam(m) else LIME_EDGE}" stroke-width="1.8"/>'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{edge}" stroke-width="1"/>'


def layer_curves(scores, floor):
    """LOAM essentiality AUROC at every tap, against relative depth."""
    w, h, l, r, t, b = 640, 360, 60, 120, 40, 50
    x0, x1, y0, y1 = 0.0, 1.0, 0.45, 0.80
    X = lambda v: l + (v - x0) / (x1 - x0) * (w - l - r)
    Y = lambda v: h - b - (v - y0) / (y1 - y0) * (h - t - b)
    task, metric = "bacbench-essentiality-layer-sweep", "macro_mean_auroc"
    out = [f'<text class="title" x="{l}" y="22">Gene essentiality: LOAM AUROC at every layer</text>']
    for v in (0.5, 0.6, 0.7, 0.8):
        out += [f'<line class="grid" x1="{l}" x2="{w - r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>',
                f'<text class="num muted" x="{l - 8}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:.2f}</text>']
    for v in (0, 0.25, 0.5, 0.75, 1):
        out.append(f'<text class="num muted" x="{X(v):.1f}" y="{h - b + 18}" text-anchor="middle">{v:g}</text>')
    out += [f'<line class="axis" x1="{l}" x2="{w - r}" y1="{h - b}" y2="{h - b}"/>',
            f'<text x="{(l + w - r) / 2}" y="{h - 12}" text-anchor="middle">relative depth (tap ÷ last tap)</text>',
            f'<text x="16" y="{(t + h - b) / 2}" text-anchor="middle" transform="rotate(-90 16 {(t + h - b) / 2})">macro mean AUROC</text>']
    fv = floor["bacbench-essentiality"]["floor"]["value"]
    out += [f'<line class="floor" x1="{l}" x2="{w - r}" y1="{Y(fv):.1f}" y2="{Y(fv):.1f}"/>',
            f'<text class="muted" x="{w - r + 6}" y="{Y(fv) + 4:.1f}">k-mer floor {fv:.4f}</text>']
    models, ends = ["LOAM-25M", "LOAM-100M", "LOAM-340M", "LOAM-624M"], []
    for i, m in enumerate(models):
        pts = sorted((k[2], v) for k, v in scores[m].items() if k[0] == task and k[1] == metric and k[2] is not None)
        last = pts[-1][0]
        xy = [(X(tap / last), Y(v)) for tap, v in pts]
        op = 0.4 + 0.2 * i  # larger model, stronger line
        out.append(f'<polyline fill="none" stroke="{LIME_EDGE}" stroke-opacity="{op:.1f}" stroke-width="{1.2 + 0.6 * i:.1f}" '
                   'points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in xy) + '"/>')
        out += [dot(x, y, m, r=2.5) for x, y in xy]
        ends.append((Y(pts[-1][1]), m[5:]))
    prev = -1e9
    for y, name in sorted(ends):  # stack the end labels top-down, at least 13 px apart
        prev = max(y, prev + 13)
        out.append(f'<text x="{w - r + 6}" y="{prev + 4:.1f}">{name}</text>')
    desc = ("Line chart of macro mean AUROC on BacBench gene essentiality at every probed layer of the four "
            "LOAM models, against relative depth. All four start near 0.48 at the embedding layer, rise "
            f"steeply in the first layers and reach 0.73 to 0.77 at depth; the dashed line is the k-mer floor, {fv:.4f}.")
    return svg(w, h, "LOAM essentiality AUROC by layer", desc, out)


def last_vs_best(scores, floor, params):
    """Last-layer and best-tap probe score for every model, two panels."""
    models = sorted(scores, key=lambda m: (not is_loam(m), params[m]))
    panels = [("Gene essentiality (AUROC)", "bacbench-essentiality", "macro_mean_auroc", 0.5, 0.8, 6),
              ("Enzyme class (macro-F1)", "dgeb-ec-classification-dna", "f1", 0.0, 0.4, 4)]
    row, top, lab, pw, gap = 22, 84, 190, 250, 40
    w, h = lab + 2 * pw + gap + 30, top + row * len(models) + 50
    out = [f'<text class="title" x="16" y="22">Last layer vs best layer, every model</text>',
           f'<text class="muted" x="16" y="40">filled = last layer · ring = best tap (chosen on the test split, an upper bound)</text>']
    for i, m in enumerate(models):
        out.append(f'<text x="{lab - 10}" y="{top + i * row + 4}" text-anchor="end">{m}</text>')
    for p, (name, task, metric, lo, hi, n) in enumerate(panels):
        px = lab + p * (pw + gap)
        X = lambda v: px + (v - lo) / (hi - lo) * pw
        out.append(f'<text x="{px}" y="{top - 14}" font-weight="600">{name}</text>')
        for k in range(n + 1):
            v = lo + k * (hi - lo) / n
            out += [f'<line class="grid" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{top - 8}" y2="{h - 40}"/>',
                    f'<text class="num muted" x="{X(v):.1f}" y="{h - 24}" text-anchor="middle">{v:.2f}</text>']
        fv = floor[task]["floor"]["value"]
        out += [f'<line class="floor" x1="{X(fv):.1f}" x2="{X(fv):.1f}" y1="{top - 8}" y2="{h - 40}"/>',
                f'<text class="muted" x="{X(fv) + 4:.1f}" y="{h - 8}">k-mer floor {fv:.4f}</text>']
        for i, m in enumerate(models):
            y = top + i * row
            a, b = scores[m][(task, metric, None)], scores[m][(task + "-layer-sweep", metric, None)]
            out += [f'<line class="link" x1="{X(a):.1f}" x2="{X(b):.1f}" y1="{y}" y2="{y}"/>',
                    dot(X(b), y, m, hollow=True), dot(X(a), y, m)]
    desc = ("Dot plot of the 13 models, LOAM first then the comparators by size. For gene essentiality AUROC and "
            "enzyme-class macro-F1 each model shows its last-layer score (filled) and its best-tap score (ring); the "
            "dashed lines are the k-mer floors. On EC most comparators sit near the floor at their last layer and "
            "gain mostly at an intermediate tap; LOAM's last-layer EC scores are 0.18 to 0.30.")
    return svg(w, h, "Last-layer vs best-layer probe scores", desc, out)


def rnagym_vs_size(scores, params):
    """Zero-shot RNAGym Spearman against measured parameter count."""
    w, h, l, r, t, b = 640, 380, 60, 30, 40, 50
    lx0, lx1, y0, y1 = math.log10(8e6), math.log10(1e10), 0.0, 0.4
    X = lambda p: l + (math.log10(p) - lx0) / (lx1 - lx0) * (w - l - r)
    Y = lambda v: h - b - (v - y0) / (y1 - y0) * (h - t - b)
    out = [f'<text class="title" x="{l}" y="22">RNAGym variant effects (zero-shot) vs model size</text>']
    for v in (0, 0.1, 0.2, 0.3, 0.4):
        out += [f'<line class="grid" x1="{l}" x2="{w - r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>',
                f'<text class="num muted" x="{l - 8}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:.1f}</text>']
    for p, s in ((1e8, "100M"), (1e9, "1B"), (1e10, "10B"), (1e7, "10M")):
        out.append(f'<text class="num muted" x="{X(p):.1f}" y="{h - b + 18}" text-anchor="middle">{s}</text>')
    out += [f'<line class="axis" x1="{l}" x2="{w - r}" y1="{h - b}" y2="{h - b}"/>',
            f'<text x="{(l + w - r) / 2}" y="{h - 12}" text-anchor="middle">parameters (measured, log scale)</text>',
            f'<text x="16" y="{(t + h - b) / 2}" text-anchor="middle" transform="rotate(-90 16 {(t + h - b) / 2})">Spearman ρ</text>']
    loam = sorted((params[m], scores[m][("rnagym-dms", "macro_spearman", None)]) for m in scores if is_loam(m))
    out.append(f'<polyline fill="none" stroke="{LIME_EDGE}" stroke-opacity="0.6" stroke-width="1.5" points="'
               + " ".join(f"{X(p):.1f},{Y(v):.1f}" for p, v in loam) + '"/>')
    nudge = {"ProkBERT-mini": (6, 16), "ProkBERT-mini-c": (-8, -9), "LOAM-25M": (10, 4), "LOAM-100M": (0, -9),
             "GenomeOcean-100M": (0, 14), "gLM2-150M": (0, 14), "Evo 1.5 (8k, 7B)": (-8, 14),
             "Evo2-7B (residual stream)": (-8, -9), "GenomeOcean-500M": (8, 14), "LOAM-340M": (-8, -9),
             "LOAM-624M": (8, -9)}
    for m in sorted(scores, key=is_loam):  # LOAM drawn last, on top
        p, v = params[m], scores[m][("rnagym-dms", "macro_spearman", None)]
        out.append(dot(X(p), Y(v), m, r=6 if is_loam(m) else 5))
        dx, dy = nudge.get(m, (8, 4))
        anchor = "end" if dx < 0 else ("start" if dx > 0 else "middle")
        name = {"Evo2-7B (residual stream)": "Evo2-7B", "Evo 1.5 (8k, 7B)": "Evo 1.5"}.get(m, m)
        out.append(f'<text x="{X(p) + dx:.1f}" y="{Y(v) + dy:.1f}" text-anchor="{anchor}" font-size="11">{name}</text>')
    desc = ("Scatter of RNAGym Spearman correlation, averaged over 11 assays with no training, against measured "
            "parameter count on a log scale. LOAM models (green, joined by a line) rise from 0.13 at 25M to 0.32 at "
            "624M; Evo 1.5 and Evo2-7B, about ten times larger, reach 0.32 and 0.34. Masked models are scored by "
            "masked-marginal LLR, causal ones by log-likelihood.")
    return svg(w, h, "RNAGym Spearman vs parameters", desc, out)


def main():
    scores, params, floor = load()
    ASSETS.mkdir(exist_ok=True)
    (ASSETS / "essentiality_by_layer.svg").write_text(layer_curves(scores, floor))
    (ASSETS / "last_vs_best_layer.svg").write_text(last_vs_best(scores, floor, params))
    (ASSETS / "rnagym_vs_size.svg").write_text(rnagym_vs_size(scores, params))


if __name__ == "__main__":
    main()
