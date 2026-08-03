"""Create the bilingual BaSyTec external-validation and harm-control figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CELL = (
    ROOT
    / "external_basytec_v352"
    / "scored_v354"
    / "basytec_cell_metrics_v354.csv"
)
REPORT = (
    ROOT
    / "external_basytec_v352"
    / "scored_v354"
    / "basytec_frozen_confirmation_v354_report.json"
)
AUDIT = (
    ROOT
    / "external_basytec_v352"
    / "scored_v354"
    / "basytec_external_statistics_audit_v357.json"
)
OUT = HERE / "figures"

NAVY = "#17365D"
TEAL = "#168C80"
GOLD = "#D3A52E"
CORAL = "#E76F51"
SLATE = "#566573"
LIGHT_TEAL = "#E7F4F1"
LIGHT_GRAY = "#EEF1F4"


def setup(zh: bool) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": (
                ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
                if zh
                else ["Arial", "DejaVu Sans"]
            ),
            "font.size": 9.2,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.edgecolor": "#7B8794",
            "axes.linewidth": 0.8,
            "xtick.color": "#34495E",
            "ytick.color": "#34495E",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw(zh: bool) -> None:
    setup(zh)
    cell = pd.read_csv(CELL).sort_values("pchp_minus_baseline").reset_index(drop=True)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))

    primary = report["accuracy"]["comparisons"]["pchp_minus_baseline"]
    mean = float(primary["cell_equal_mean"])
    lower, upper = map(float, primary["cell_bootstrap_ci95"])
    accuracy = report["accuracy"]
    contract = report["contract"]
    fixed_mae = float(
        audit["reviewer_risk_diagnostics"]["cell_equal_constant_safe_up_mae"]
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.6, 4.35),
        gridspec_kw={"width_ratios": [1.24, 1.0], "wspace": 0.24},
    )

    ax = axes[0]
    values = cell["pchp_minus_baseline"].to_numpy(float)
    y = np.arange(len(values))
    ax.axvspan(-0.011, 0.0, color=LIGHT_TEAL, zorder=0)
    ax.axvline(0.0, color="#7B8794", lw=1.0, ls="--", zorder=1)
    ax.scatter(values, y, s=24, color=TEAL, edgecolor="white", linewidth=0.45, zorder=3)
    ax.errorbar(
        mean,
        -3.2,
        xerr=np.array([[mean - lower], [upper - mean]]),
        fmt="D",
        ms=6.2,
        color=NAVY,
        ecolor=NAVY,
        elinewidth=1.7,
        capsize=3.2,
        zorder=4,
    )
    ax.set_ylim(-5.0, len(values) + 1)
    ax.set_xlim(-0.011, 0.001)
    ax.set_yticks([0, 10, 20, 30, 40, 44])
    ax.set_yticklabels(["1", "11", "21", "31", "41", "45"])
    ax.grid(axis="x", color="#D8DEE4", lw=0.65, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    if zh:
        ax.set_title("A  结果盲、协议锁定外部验证：45/45 个电芯改善", loc="left", fontweight="bold")
        ax.set_xlabel("电芯内 MAE 差（PCHP − 受保护基线）")
        ax.set_ylabel("按效应排序的物理电芯")
        note = (
            rf"电芯等权均值 $={mean:.5f}$" + "\n" +
            rf"$95\%$ 电芯 bootstrap 区间 $=[{lower:.5f},{upper:.5f}]$"
        )
        ax.text(-0.01075, 41.5, note, color=NAVY, fontsize=8.7, va="top")
        ax.text(-0.00935, -3.2, "均值与区间", color=NAVY, fontsize=8.2, va="center")
    else:
        ax.set_title("A  Outcome-blind, protocol-locked validation: 45/45 cells improve", loc="left", fontweight="bold")
        ax.set_xlabel("Within-cell MAE difference (PCHP − protected baseline)")
        ax.set_ylabel("Physical cells, ordered by effect")
        note = (
            rf"Cell-equal mean $={mean:.5f}$" + "\n" +
            rf"$95\%$ cell bootstrap $=[{lower:.5f},{upper:.5f}]$"
        )
        ax.text(-0.01075, 41.5, note, color=NAVY, fontsize=8.7, va="top")
        ax.text(-0.00935, -3.2, "mean and interval", color=NAVY, fontsize=8.2, va="center")

    ax = axes[1]
    baseline_mae = float(accuracy["cell_equal_baseline_mae"])
    pchp_mae = float(accuracy["cell_equal_pchp_mae"])
    comparator_mae = float(accuracy["cell_equal_source_tuned_candidate_mae"])
    max_pchp = float(contract["maximum_pchp_displacement"])
    max_comparator = float(contract["maximum_source_tuned_candidate_displacement"])
    ax.axvspan(0.0, 0.01, color=LIGHT_TEAL, zorder=0)
    ax.axvspan(0.01, 0.31, color=LIGHT_GRAY, alpha=0.6, zorder=0)
    ax.axvline(0.01, color=TEAL, lw=1.15, ls="--")
    points = [
        (0.0, baseline_mae, NAVY, "o", 58),
        (max_pchp, pchp_mae, TEAL, "o", 68),
        (0.01, fixed_mae, GOLD, "D", 58),
        (max_comparator, comparator_mae, CORAL, "X", 76),
    ]
    for x, yy, color, marker, size in points:
        ax.scatter(x, yy, s=size, marker=marker, color=color, edgecolor="white", linewidth=0.7, zorder=4)
    ax.plot([0.0, max_pchp], [baseline_mae, pchp_mae], color=TEAL, lw=1.4, zorder=2)
    ax.set_xlim(-0.012, 0.305)
    ax.set_ylim(0.045, 0.187)
    ax.set_xticks([0.01, 0.10, 0.20, 0.285])
    ax.set_xticklabels(["0.01", "0.10", "0.20", "0.285"])
    ax.grid(color="#D8DEE4", lw=0.65, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    if zh:
        ax.set_title("B  损害控制—精度权衡", loc="left", fontweight="bold")
        ax.set_xlabel("相对基线的最大逐点位移")
        ax.set_ylabel("电芯等权 MAE（越低越好）")
        ax.text(0.014, 0.115, r"预算边界 $\delta=0.01$", color=TEAL, fontsize=8.1, va="center", rotation=90)
        ax.annotate("受保护基线", (0.0, baseline_mae), xytext=(0.064, 0.181), textcoords="data", fontsize=8.4, color=NAVY, arrowprops={"arrowstyle": "-", "color": NAVY, "lw": 0.7})
        ax.annotate("PCHP", (max_pchp, pchp_mae), xytext=(0.055, 0.153), textcoords="data", fontsize=8.5, color=TEAL, arrowprops={"arrowstyle": "-", "color": TEAL, "lw": 0.7})
        ax.annotate("固定 +0.01（事后）", (0.01, fixed_mae), xytext=(0.083, 0.169), textcoords="data", fontsize=8.2, color="#8A6A00", arrowprops={"arrowstyle": "-", "color": GOLD, "lw": 0.7})
        ax.annotate("源域调优无保护比较器", (max_comparator, comparator_mae), xytext=(0.15, 0.069), textcoords="data", fontsize=8.2, color=CORAL, arrowprops={"arrowstyle": "-", "color": CORAL, "lw": 0.8})
        ax.text(0.151, 0.052, "MAE 大幅降低，但违反 45/45 个电芯的预算", fontsize=8.0, color=SLATE)
    else:
        ax.set_title("B  Harm-control--accuracy trade-off", loc="left", fontweight="bold")
        ax.set_xlabel("Maximum pointwise displacement from baseline")
        ax.set_ylabel("Cell-equal MAE (lower is better)")
        ax.text(0.014, 0.115, r"budget boundary $\delta=0.01$", color=TEAL, fontsize=8.1, va="center", rotation=90)
        ax.annotate("protected baseline", (0.0, baseline_mae), xytext=(0.064, 0.181), textcoords="data", fontsize=8.4, color=NAVY, arrowprops={"arrowstyle": "-", "color": NAVY, "lw": 0.7})
        ax.annotate("PCHP", (max_pchp, pchp_mae), xytext=(0.055, 0.153), textcoords="data", fontsize=8.5, color=TEAL, arrowprops={"arrowstyle": "-", "color": TEAL, "lw": 0.7})
        ax.annotate("fixed +0.01 (post hoc)", (0.01, fixed_mae), xytext=(0.083, 0.169), textcoords="data", fontsize=8.2, color="#8A6A00", arrowprops={"arrowstyle": "-", "color": GOLD, "lw": 0.7})
        ax.annotate("source-tuned unprotected", (max_comparator, comparator_mae), xytext=(0.15, 0.069), textcoords="data", fontsize=8.2, color=CORAL, arrowprops={"arrowstyle": "-", "color": CORAL, "lw": 0.8})
        ax.text(0.151, 0.052, "large MAE gain, but violates all 45 cell budgets", fontsize=8.0, color=SLATE)

    fig.patch.set_facecolor("white")
    for axis in axes:
        axis.set_facecolor("white")
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.17, top=0.92)
    suffix = "_zh" if zh else ""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"fig6_basytec_confirmation{suffix}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"fig6_basytec_confirmation{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    draw(False)
    draw(True)
