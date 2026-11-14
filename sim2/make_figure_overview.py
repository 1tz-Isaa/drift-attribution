"""Overview figure of the framework (Figure 1 of the draft). A schematic: reads no result file.

    python3 sim2/make_figure_overview.py

(a) the two periods and the mechanism the deployment adds, (b) what a monitor
without action logs does with the alert's effect, (c) the proposed
decomposition, the comparison that identifies each term, and what each term
informs. Bar lengths in (b) are illustrative, not results.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
FIG_DIRS = (ROOT / "figures", ROOT / "aaai/figures")
CX, CP, CY, CG = "#4477AA", "#228833", "#EE6677", "#888888"   # X, policy, Y|X(,A), grey
LX, LP, LY, LG = "#DCE6F2", "#DDEEDD", "#FBE0E4", "#EEEEEE"   # light fills of the same hues
INK = "#222222"
plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})


def node(ax, x, y, label, r=8.5, fc="white", ec=INK):
    ax.add_patch(plt.Circle((x, y), r, fc=fc, ec=ec, lw=0.8, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=7.5, zorder=4)


def arrow(ax, p, q, color=INK, lw=0.8, ls="-", rad=0.0, shrink=9.5, head=6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=head, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=shrink, shrinkB=shrink, zorder=2))


def box(ax, x, y, w, h, fc, ec, text, size=6.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=3", fc=fc, ec=ec, lw=0.8, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight=weight, zorder=3,
            linespacing=1.25)


def graph(ax, x0, y0, deployed):
    """Causal graph of one period; the score acts on A only after deployment."""
    X, R, A, Y = (x0 + 14, y0), (x0 + 66, y0 + 31), (x0 + 118, y0), (x0 + 178, y0)
    node(ax, *X, "$X$", fc=LX, ec=CX)
    node(ax, *R, "$\\hat r$", fc=LP if deployed else "white", ec=CP if deployed else CG)
    node(ax, *A, "$A$")
    node(ax, *Y, "$Y$", fc=LY, ec=CY)
    arrow(ax, X, A)
    arrow(ax, A, Y)
    arrow(ax, X, Y, rad=0.36)
    arrow(ax, X, R, color=INK if deployed else CG)
    if deployed:
        arrow(ax, R, A, color=CP, lw=1.6, head=8)
        ax.text(R[0] + 33, R[1] - 9, "alert policy $\\pi_1$", color=CP, fontsize=7, ha="left", va="center")
    else:
        arrow(ax, R, A, color=CG, ls=(0, (2, 2)))
        ax.text(R[0] + 33, R[1] - 9, "score not acted on", color=CG, fontsize=7, ha="left", va="center")
    ax.text((X[0] + A[0]) / 2 + 14, y0 + 3, "$\\pi_0$", fontsize=7.5, ha="center", va="bottom", color=INK)


def bar(ax, x, y, parts, h=11):
    """Stacked bar; parts are (width, face colour, edge colour, label)."""
    for width, fc, ec, label in parts:
        ax.add_patch(Rectangle((x, y), width, h, fc=fc, ec=ec, lw=0.8, zorder=2))
        ax.text(x + width / 2, y + h / 2, label, ha="center", va="center", fontsize=7, zorder=3)
        x += width


def main():
    from common import save_figure
    fig = plt.figure(figsize=(7.0, 2.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 700)
    ax.set_ylim(0, 250)
    ax.axis("off")
    title = dict(fontsize=8, weight="bold", ha="left", va="top")
    T = 7   # body text size

    # (a) setting
    ax.text(4, 246, "(a) Deployment adds a link", **title)
    ax.text(4, 222, "Before deployment ($e_0$)", fontsize=T, va="center")
    graph(ax, 6, 170, deployed=False)
    ax.text(4, 124, "After deployment ($e_1$)", fontsize=T, va="center")
    graph(ax, 6, 74, deployed=True)
    ax.text(4, 4, "$\\pi_0$: standard care\n$\\Delta R$: change in loss of the fixed score", fontsize=T, va="bottom",
            linespacing=1.25)
    ax.plot([207, 207], [4, 246], color="#CCCCCC", lw=0.6)

    # (b) problem
    x0 = 215
    ax.text(x0, 246, "(b) What the monitor reports", **title)
    ax.text(x0, 222, "Only the alert changed. True cause:", fontsize=T, va="center")
    bar(ax, x0, 200, [(200, LP, CP, "alert policy $\\pi$: all of $\\Delta R$")], h=14)
    ax.text(x0, 186, "A monitor without action logs reports:", fontsize=T, va="center")
    bar(ax, x0, 164, [(200, LY, CY, "outcome mechanism $P(Y\\mid X)$")], h=14)
    ax.text(x0, 143, "Misinterpreting this attribution\ncan lead to:", fontsize=T, va="center", linespacing=1.25)
    cx = x0
    chain = (("naive\nrefit", 52), ("alerts\nremoved", 62), ("benefit\nlost", 62))
    for i, (label, w) in enumerate(chain):
        box(ax, cx, 96, w, 30, "white", CG, label, size=T)
        if i < len(chain) - 1:
            arrow(ax, (cx + w, 111), (cx + w + 12, 111), color=CG, shrink=0.5, head=5)
        cx += w + 12
    ax.text(x0, 62, "No support at $\\varepsilon = 0$: old data cannot\nshow what the alert does.",
            fontsize=T, va="center", linespacing=1.25)
    ax.plot([437, 437], [4, 246], color="#CCCCCC", lw=0.6)

    # (c) method
    x0 = 445
    ax.text(x0, 246, "(c) Policy-last decomposition", **title)
    bw, bh, gap = 76, 30, 11
    xs = [x0, x0 + bw + gap, x0 + 2 * (bw + gap)]
    ax.text(xs[0], 224, "before deployment", fontsize=T, ha="left", va="center", color=CG)
    ax.text((xs[1] + xs[2] + bw) / 2, 224, "after deployment", fontsize=T, ha="center", va="center", color=CG)
    box(ax, xs[0], 184, bw, bh, LG, CG, "standard care\n$\\pi_0$", size=6.8)
    box(ax, xs[1], 184, bw, bh, LG, CG, "unalerted arm\n$\\pi_0$, randomised", size=6.4)
    box(ax, xs[2], 184, bw, bh, LP, CP, "alerted arm\n$\\pi_1$", size=6.8)
    for a, b, color in ((xs[0], xs[1], CG), (xs[1], xs[2], CP)):
        ax.add_patch(FancyArrowPatch((a + bw + 0.5, 199), (b - 0.5, 199), arrowstyle="<|-|>", mutation_scale=4.5,
                                     color=color, lw=0.9, shrinkA=0, shrinkB=0, zorder=2))
    mid1, mid2 = xs[0] + bw + gap / 2, xs[1] + bw + gap / 2
    for mid, color in ((mid1, CG), (mid2, CP)):
        ax.plot([mid, mid], [195, 175], color=color, lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.text(mid1, 168, "reference-care change", fontsize=T, ha="center", va="center", color=INK)
    ax.text(mid2 + 6, 168, "policy term", fontsize=T, ha="center", va="center", color=CP)
    ax.text(mid1 - 38, 149, "$\\Delta R\\;=$", fontsize=8.5, ha="right", va="center")
    ax.text(mid1, 149, "$\\Delta_X + \\Delta_{Y|X,A}$", fontsize=8.5, ha="center", va="center")
    ax.text((mid1 + mid2) / 2 + 8, 149, "$+$", fontsize=8.5, ha="center", va="center")
    ax.text(mid2 + 6, 149, "$\\Delta_\\pi$", fontsize=8.5, ha="center", va="center", color=CP)
    ax.text(x0, 126, "Attribution informs update evaluation:", fontsize=T, va="center")
    rows = ((CP, LP, "$\\Delta_\\pi$", "assess policy benefit"),
            (CY, LY, "$\\Delta_{Y|X,A}$", "assess outcome drift"),
            (CX, LX, "$\\Delta_X$", "assess case-mix change"))
    for i, (ec, fc, sym, text) in enumerate(rows):
        y = 92 - 27 * i
        box(ax, x0, y, 48, 19, fc, ec, sym, size=7.5)
        ax.text(x0 + 56, y + 9.5, text, fontsize=T, va="center")

    save_figure(fig, "fig_overview.pdf", FIG_DIRS)
    plt.close(fig)
    print("wrote fig_overview.pdf")


if __name__ == "__main__":
    main()
