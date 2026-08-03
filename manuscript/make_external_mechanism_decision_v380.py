"""Create the frozen V380 external-mechanism and decision-cost figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "figures"
RESULT = ROOT / "external_mechanism_decision_v380"

NAVY = "#17365D"
TEAL = "#1F9D8A"
ORANGE = "#E76F51"
GOLD = "#D8AE45"
INK = "#19324A"
GRAY = "#657481"
MID = "#A8B4BF"
GRID = "#DDE4EA"
PALE_TEAL = "#E4F3F0"
WHITE = "#FFFFFF"


def set_style(lang: str) -> None:
    mpl.rcParams.update(
        {
            "font.family": "Microsoft YaHei" if lang == "zh" else "DejaVu Sans",
            "font.size": 9.3,
            "axes.titlesize": 10.7,
            "axes.labelsize": 9.4,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axis(ax: plt.Axes, *, axis: str) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MID)
    ax.tick_params(colors="#425466", length=3)
    ax.grid(axis=axis, color=GRID, linewidth=0.75, zorder=0)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
    )


def make_figure(lang: str = "en") -> None:
    set_style(lang)
    surface = pd.read_csv(RESULT / "external_mechanism_surface_metrics_v380.csv")
    decision = pd.read_csv(RESULT / "external_decision_summary_v380.csv")
    effect = -0.00689716916991253
    interval = (-0.011921668904922808, -0.00211749268631288)

    display = {
        "AMPERE_A123_LFP": "AMPERE LFP",
        "IMPERIAL_M50T": "Imperial M50T",
        "ISU_ILCC_NMC": "ISU-ILCC NMC",
        "LUH_NMC_SIO": "LUH NMC/SiO",
        "MULTISTAGE_50E": "Multistage 50E",
        "STANFORD_CALENDAR": "Stanford calendar",
    }
    surface["label"] = surface["domain"].map(display)
    surface = surface.sort_values("pchp_minus_fixed_shift", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.55), gridspec_kw={"width_ratios": [1.03, 1.17]})
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.19, top=0.86, wspace=0.28)

    ax = axes[0]
    y = np.arange(len(surface))
    values = surface["pchp_minus_fixed_shift"].to_numpy(float)
    colors = [TEAL if value < 0.0 else ORANGE for value in values]
    ax.axvline(0.0, color=GRAY, linewidth=1.0, linestyle="--", zorder=1)
    ax.hlines(y, 0.0, values, color=colors, linewidth=2.2, zorder=2)
    ax.scatter(values, y, s=58, color=colors, edgecolor=WHITE, linewidth=0.8, zorder=3)
    mean_y = len(surface) + 0.45
    ax.errorbar(
        effect,
        mean_y,
        xerr=np.asarray([[effect - interval[0]], [interval[1] - effect]]),
        fmt="D",
        markersize=7.2,
        color=NAVY,
        ecolor=NAVY,
        elinewidth=2.0,
        capsize=3.5,
        zorder=4,
    )
    ax.axhline(len(surface) - 0.15, color=GRID, linewidth=1.0)
    ax.set_yticks([*y, mean_y])
    mean_label = "外部数据集等权均值" if lang == "zh" else "Dataset-equal mean"
    ax.set_yticklabels([*surface["label"], mean_label])
    ax.invert_yaxis()
    ax.set_xlim(-0.021, 0.006)
    ax.set_xlabel(
        "PCHP 相对常数偏移的 MAE 差值"
        if lang == "zh"
        else "PCHP minus constant-offset MAE"
    )
    ax.set_title(
        "记录级候选信息优于统一常数偏移"
        if lang == "zh"
        else "Record-level candidate information\nexceeds a uniform constant offset",
        loc="left",
        pad=11,
        color=INK,
        fontweight="bold",
    )
    if lang == "zh":
        ax.title.set_text("")
        ax.set_title(
            "记录级候选信息\n优于统一常数偏移",
            loc="left",
            pad=11,
            color=INK,
            fontweight="bold",
        )
    note = (
        r"$5/6$ 个外部数据集改善；两阶段 $95\%$ 区间"
        if lang == "zh"
        else r"$5/6$ external datasets improved; two-stage $95\%$ interval"
    )
    ax.text(0.0, -0.19, note, transform=ax.transAxes, color=GRAY, fontsize=8.2)
    clean_axis(ax, axis="x")
    panel_label(ax, "A")

    ax = axes[1]
    subset = decision.loc[
        (decision["threshold"] == 0.8)
        & decision["method"].isin(["protected_state", "fixed_shift", "pchp_method"])
    ].copy()
    labels = {
        "protected_state": "受保护状态" if lang == "zh" else "Protected state",
        "fixed_shift": "常数偏移" if lang == "zh" else "Constant offset",
        "pchp_method": "PCHP",
    }
    styles = {
        "protected_state": (NAVY, "o", "--"),
        "fixed_shift": (GOLD, "s", "-."),
        "pchp_method": (TEAL, "D", "-"),
    }
    ax.axvspan(4.55, 5.45, color=PALE_TEAL, zorder=0)
    for method in ("protected_state", "fixed_shift", "pchp_method"):
        rows = subset.loc[subset["method"] == method].sort_values("cost_ratio")
        color, marker, linestyle = styles[method]
        ax.plot(
            rows["cost_ratio"],
            rows["surface_equal_continuous_cost"],
            label=labels[method],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.0,
            markersize=5.4,
            markeredgecolor=WHITE,
            markeredgewidth=0.7,
            zorder=3,
        )
    primary = subset.loc[subset["cost_ratio"] == 5.0].set_index("method")
    fixed_value = float(primary.loc["fixed_shift", "surface_equal_continuous_cost"])
    pchp_value = float(primary.loc["pchp_method", "surface_equal_continuous_cost"])
    ax.annotate(
        f"{fixed_value:.3f} $\\rightarrow$ {pchp_value:.3f}",
        xy=(5.0, pchp_value),
        xytext=(5.7, pchp_value - 0.11),
        color=TEAL,
        fontsize=8.7,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": TEAL, "lw": 1.1},
    )
    action_note = "τ = 0.80：二元复核决策未改变" if lang == "zh" else "τ = 0.80: binary review decision unchanged"
    ax.text(
        0.035,
        0.955,
        action_note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.1,
        color=GRAY,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": WHITE, "edgecolor": GRID},
    )
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10], labels=["1", "2", "5", "10"])
    ax.set_xlim(0.9, 11.2)
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel(
        r"漏检退化 / 不必要复核代价比 $r$"
        if lang == "zh"
        else r"Missed-degradation / unnecessary-review cost ratio $r$"
    )
    ax.set_ylabel(
        "数据集等权连续非对称代价"
        if lang == "zh"
        else "Dataset-equal continuous asymmetric cost"
    )
    ax.set_title(
        "保守更新降低决策相关风险严重度"
        if lang == "zh"
        else "Conservative updates reduce\ndecision-relevant risk severity",
        loc="left",
        pad=11,
        color=INK,
        fontweight="bold",
    )
    if lang == "zh":
        ax.title.set_text("")
        ax.set_title(
            "保守更新降低\n决策相关风险严重度",
            loc="left",
            pad=11,
            color=INK,
            fontweight="bold",
        )
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), frameon=False)
    clean_axis(ax, axis="y")
    panel_label(ax, "B")

    stem = "fig7_external_mechanism_decision"
    if lang == "zh":
        stem += "_zh"
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", facecolor=WHITE, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / f"{stem}.png", dpi=360, facecolor=WHITE, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    make_figure("en")
    make_figure("zh")
