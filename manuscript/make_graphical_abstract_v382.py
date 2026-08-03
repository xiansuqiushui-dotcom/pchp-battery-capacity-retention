from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


NAVY = "#173A5E"
BLUE = "#4777AA"
TEAL = "#1D9A8A"
GOLD = "#D6A52A"
CORAL = "#E76F51"
INK = "#162B40"
MUTED = "#647789"
BORDER = "#B9C6D1"
PALE_BLUE = "#EEF4F8"
PALE_TEAL = "#E8F5F2"
PALE_GOLD = "#FBF5E7"
PALE_CORAL = "#FCEEEA"
WHITE = "#FFFFFF"


def box(ax, x, y, w, h, *, face=WHITE, edge=BORDER, radius=0.8, lw=1.4):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.25,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, x1, y1, x2, y2, *, color=MUTED, lw=1.8, mutation=12, style="-|>"):
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(patch)
    return patch


def panel_header(ax, x, number, title):
    ax.text(x, 46.5, str(number), color=NAVY, fontsize=17, fontweight="bold", va="center")
    ax.text(x + 3.1, 46.5, title, color=INK, fontsize=15, fontweight="bold", va="center")


def metric_card(ax, y, accent, title, scope, result):
    box(ax, 93.3, y, 34.0, 8.2, face=WHITE, edge=BORDER, radius=0.7)
    ax.plot([95.1, 101.8], [y + 6.9, y + 6.9], color=accent, linewidth=4.2, solid_capstyle="round")
    ax.text(95.0, y + 5.6, title, fontsize=9.8, fontweight="bold", color=INK, va="center")
    ax.text(95.0, y + 3.5, scope, fontsize=9.0, color=MUTED, va="center")
    ax.text(95.0, y + 1.35, result, fontsize=9.2, fontweight="bold", color=INK, va="center")


def build_graphical_abstract(output_dir: Path) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.unicode_minus": True,
            "mathtext.fontset": "dejavusans",
        }
    )
    fig, ax = plt.subplots(figsize=(13, 5), dpi=180)
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 50)
    ax.axis("off")

    ax.plot([40.5, 40.5], [3, 47], color="#DCE4EA", linewidth=1.2)
    ax.plot([90.5, 90.5], [3, 47], color="#DCE4EA", linewidth=1.2)

    # Panel 1: operational failure.
    panel_header(ax, 1.2, 1, "Delayed-outcome gap")
    ax.text(4.2, 40.7, "Partial charge records arrive first", fontsize=10.9, color=INK, fontweight="bold")
    ax.text(4.2, 38.1, "reference capacity labels arrive later", fontsize=9.4, color=MUTED)
    ax.plot([5.0, 36.0], [25.0, 25.0], color="#9CADBA", linewidth=2.1)
    for x in (8.0, 17.0):
        ax.scatter([x], [25.0], s=105, color=BLUE, zorder=3)
    for x in (28.0, 35.0):
        ax.scatter([x], [25.0], s=105, color=GOLD, zorder=3)
    ax.text(8.0, 21.2, r"$t-1$", ha="center", color=NAVY, fontsize=10.3)
    ax.text(17.0, 21.2, r"$t$", ha="center", color=NAVY, fontsize=10.3)
    ax.text(28.0, 21.2, r"$t+1$", ha="center", color=MUTED, fontsize=10.3)
    ax.text(35.0, 21.2, r"$t+2$", ha="center", color=MUTED, fontsize=10.3)
    ax.text(13.5, 29.1, "issued record", ha="center", color=NAVY, fontsize=9.8, fontweight="bold")
    ax.text(31.0, 29.1, "future charge data", ha="center", color=MUTED, fontsize=9.2)
    revision = FancyArrowPatch(
        (31.0, 31.2),
        (17.8, 27.0),
        arrowstyle="-|>",
        mutation_scale=12,
        connectionstyle="arc3,rad=0.25",
        linewidth=2.0,
        linestyle=(0, (4, 3)),
        color=CORAL,
    )
    ax.add_patch(revision)
    ax.text(24.0, 34.3, "retrospective revision", color=CORAL, fontsize=9.7, ha="center")
    ax.plot([16.2, 18.0], [26.5, 28.3], color=CORAL, linewidth=2.5)
    ax.plot([16.2, 18.0], [28.3, 26.5], color=CORAL, linewidth=2.5)
    box(ax, 2.7, 7.3, 35.4, 7.0, face=PALE_CORAL, edge="#F0B9AA", radius=0.8)
    ax.text(20.4, 11.6, "Future observations must not rewrite", ha="center", color=INK, fontsize=9.3)
    ax.text(20.4, 9.2, "an already issued health record", ha="center", color=INK, fontsize=9.3, fontweight="bold")

    # Panel 2: method and guarantee.
    panel_header(ax, 42.0, 2, "Prefix-causal projection")
    ax.text(45.1, 43.6, "with an exact harm budget", color=MUTED, fontsize=9.5, va="center")
    box(ax, 44.2, 33.5, 17.0, 8.2, face=PALE_BLUE, edge="#AFC3D4", radius=0.7)
    ax.plot([45.7, 51.2], [40.4, 40.4], color=BLUE, linewidth=4.0, solid_capstyle="round")
    ax.text(45.6, 38.4, "Protected state", color=INK, fontsize=10.2, fontweight="bold")
    ax.text(45.6, 35.9, r"prefix-only $b_t$", color=NAVY, fontsize=10.0)

    box(ax, 69.7, 33.5, 17.0, 8.2, face=PALE_TEAL, edge="#A9D3C9", radius=0.7)
    ax.plot([71.2, 76.7], [40.4, 40.4], color=TEAL, linewidth=4.0, solid_capstyle="round")
    ax.text(71.1, 38.4, "Learned candidate", color=INK, fontsize=10.2, fontweight="bold")
    ax.text(71.1, 35.9, r"record-specific $c_t$", color="#177C70", fontsize=10.0)

    arrow(ax, 53.0, 33.2, 59.0, 29.1, color=BLUE)
    arrow(ax, 78.0, 33.2, 72.5, 29.1, color=TEAL)

    box(ax, 49.2, 17.0, 32.5, 12.0, face=WHITE, edge="#8EA7BA", radius=0.8, lw=1.6)
    ax.text(65.45, 26.8, "Exact online viability set", ha="center", color=INK, fontsize=11.2, fontweight="bold")
    ax.text(
        65.45,
        22.8,
        r"$[b_t-\delta,\,b_t+\delta]\;\cap\;\mathcal{Y}\;\cap\;(-\infty,p_{t-1}]$",
        ha="center",
        color=NAVY,
        fontsize=12.2,
    )
    ax.text(65.45, 19.2, "project candidate to the closest viable output", ha="center", color=MUTED, fontsize=9.3)
    arrow(ax, 65.45, 16.7, 65.45, 13.0, color=NAVY, lw=2.0)
    box(ax, 58.8, 8.0, 13.3, 4.8, face=PALE_GOLD, edge="#E3C66F", radius=0.7)
    ax.text(65.45, 10.4, r"output $p_t$", ha="center", va="center", color=INK, fontsize=11.3, fontweight="bold")
    box(ax, 44.8, 1.8, 41.3, 4.7, face=PALE_TEAL, edge="#9DCDC2", radius=0.7)
    ax.text(
        65.45,
        4.15,
        r"For any hidden outcome: $|p_t-y|-|b_t-y|\leq\delta$",
        ha="center",
        va="center",
        color=INK,
        fontsize=10.5,
        fontweight="bold",
    )

    # Panel 3: evidence tied to the claim.
    panel_header(ax, 92.0, 3, "Evidence across settings")
    metric_card(ax, 35.0, BLUE, "Nested cross-domain", "12 development domains  |  586 cells", r"11/12 improved  |  $\Delta$MAE $-0.00486$")
    metric_card(ax, 24.3, TEAL, "External mechanism confirmation", "6 datasets  |  659 cells", r"5/6 improved  |  $\Delta$MAE $-0.00690$")
    metric_card(ax, 13.6, CORAL, "Decision-cost connection", r"cost ratio $r=5{:}1$", "7.54% lower  |  binary decision stable")
    box(ax, 93.3, 3.0, 34.0, 7.4, face=PALE_GOLD, edge="#E4CA7A", radius=0.7)
    ax.text(110.3, 7.5, "Risk-controlled updates", ha="center", color=INK, fontsize=10.6, fontweight="bold")
    ax.text(110.3, 5.0, "before reference capacity is available", ha="center", color=MUTED, fontsize=9.2)

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "graphical_abstract_applied_energy_v388.png"
    pdf = output_dir / "graphical_abstract_applied_energy_v388.pdf"
    fig.savefig(png, dpi=180, facecolor=WHITE, bbox_inches=None, pad_inches=0)
    fig.savefig(pdf, facecolor=WHITE, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the V388 Applied Energy graphical abstract.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    png, pdf = build_graphical_abstract(args.output_dir)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
