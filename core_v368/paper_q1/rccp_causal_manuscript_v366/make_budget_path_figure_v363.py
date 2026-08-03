from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SUMMARY = ROOT / "budget_path_candidate_control_v361" / "budget_path_summary_v361.csv"
AUC = ROOT / "budget_path_candidate_control_v361" / "budget_path_domain_auc_v361.csv"
AUDIT = ROOT / "budget_path_anchor_audit_v362" / "budget_path_anchor_audit_v362_report.json"
FIGURES = HERE / "figures"

EXPECTED_SHA256 = {
    SUMMARY: "340adf8ed60a68c748b5b03aa0fa20ede6ed068a93561bd2a8ca164d747e4240",
    AUC: "f3c1275be42e9de2d38a4d516c3518a3777bad5a0341f4d07a2852d28693c416",
    AUDIT: "8e02eca68f7016d0ec08100195689e0129cccc90fc8add7b6ecc4a275b323608",
}

NAVY = "#173B63"
TEAL = "#159A8C"
ORANGE = "#E76F51"
INK = "#172B3A"
MID = "#52677B"
GRID = "#D9E2E8"
PALE_TEAL = "#EAF6F3"
PALE_ORANGE = "#FCEFEA"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_pair(value: str) -> tuple[float, float]:
    values = ast.literal_eval(value)
    if len(values) != 2:
        raise ValueError(value)
    return float(values[0]), float(values[1])


def configure_font(chinese: bool) -> None:
    if chinese:
        font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
        font_manager.fontManager.addfont(font_path)
        family = font_manager.FontProperties(fname=font_path).get_name()
    else:
        family = "DejaVu Sans"
    mpl.rcParams.update(
        {
            "font.family": family,
            "font.size": 8.5,
            "axes.titlesize": 9.3,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "axes.unicode_minus": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AEBCC7")
    ax.spines["bottom"].set_color("#AEBCC7")
    ax.tick_params(colors=MID, width=0.8, length=3)


def make_figure(summary: pd.DataFrame, auc: pd.DataFrame, chinese: bool, stem: str) -> None:
    configure_font(chinese)
    s = summary.copy()
    s["budget_pp"] = 100.0 * s["budget"]
    s["effect_pp"] = 100.0 * s["domain_equal_pchp_minus_control"]
    cis = s["descriptive_domain_bootstrap_ci95_pchp_minus_control"].map(parse_pair)
    s["lo_pp"] = [100.0 * item[0] for item in cis]
    s["hi_pp"] = [100.0 * item[1] for item in cis]

    d = auc[["domain", "budget_normalized_auc_pchp_minus_control"]].copy()
    d["effect_pp"] = 100.0 * d["budget_normalized_auc_pchp_minus_control"]
    d = d.sort_values("effect_pp", ascending=True).reset_index(drop=True)

    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    report361 = json.loads(
        (ROOT / "budget_path_candidate_control_v361" / "budget_path_candidate_control_v361_report.json").read_text(
            encoding="utf-8"
        )
    )
    mean_pp = 100.0 * float(report361["primary"]["domain_equal_mean"])
    lo_pp, hi_pp = [100.0 * float(x) for x in report361["primary"]["domain_bootstrap_ci95"]]
    if not audit["audit_pass"]:
        raise RuntimeError("V362 anchor audit did not pass")

    fig = plt.figure(figsize=(7.35, 4.05), constrained_layout=False)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.17, 1.0], wspace=0.34)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    fig.subplots_adjust(left=0.105, right=0.985, top=0.895, bottom=0.19)

    # Panel A: one protocol-locked trajectory, with per-budget intervals shown descriptively.
    y_min = min(-2.0, float(s["lo_pp"].min()) - 0.08)
    y_max = max(0.17, float(s["hi_pp"].max()) + 0.08)
    ax1.axhspan(y_min, 0.0, color=PALE_TEAL, zorder=0)
    ax1.axhspan(0.0, y_max, color=PALE_ORANGE, alpha=0.48, zorder=0)
    ax1.axhline(0.0, color=INK, linewidth=1.05, zorder=1)
    ax1.axvline(1.0, color=NAVY, linestyle=(0, (3, 2)), linewidth=1.0, alpha=0.85, zorder=1)
    yerr = np.vstack([s["effect_pp"] - s["lo_pp"], s["hi_pp"] - s["effect_pp"]])
    ax1.errorbar(
        s["budget_pp"],
        s["effect_pp"],
        yerr=yerr,
        fmt="o-",
        color=TEAL,
        ecolor=TEAL,
        linewidth=1.8,
        elinewidth=1.0,
        capsize=2.4,
        markersize=4.5,
        markeredgecolor="white",
        markeredgewidth=0.7,
        zorder=3,
    )
    main = s.loc[np.isclose(s["budget"], 0.01, rtol=0.0, atol=1e-12)].iloc[0]
    ax1.scatter([main["budget_pp"]], [main["effect_pp"]], marker="D", s=38, color=NAVY, edgecolor="white", linewidth=0.7, zorder=4)
    if chinese:
        title_a = "A  不同损害预算下的候选信息增益"
        xlab_a = "损害预算 $\\delta$（SOH 百分点）"
        ylab_a = "PCHP − 精确安全偏移\nMAE（百分点）"
        better = "下方区域表示 PCHP 更优"
        main_label = "正文预算\n$\\delta=1$ 个百分点"
    else:
        title_a = "A  Candidate-information gain across harm budgets"
        xlab_a = "Harm budget $\\delta$ (SOH percentage points)"
        ylab_a = "PCHP − exact safe shift\nMAE (percentage points)"
        better = "Lower region favors PCHP"
        main_label = "Main setting\n$\\delta=1$ pp"
    ax1.set_title(title_a, loc="left", color=INK, pad=7)
    ax1.set_xlabel(xlab_a, color=INK, labelpad=7)
    ax1.set_ylabel(ylab_a, color=INK, labelpad=7)
    ax1.set_xlim(-0.08, 3.10)
    ax1.set_ylim(y_min, y_max)
    ax1.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax1.yaxis.grid(True, color=GRID, linewidth=0.75, zorder=0)
    ax1.text(0.035, 0.06, better, transform=ax1.transAxes, color=TEAL, fontsize=7.2, weight="bold")
    ax1.annotate(
        main_label,
        xy=(1.0, float(main["effect_pp"])),
        xytext=(1.20, -0.23),
        textcoords="data",
        color=NAVY,
        fontsize=7.0,
        ha="left",
        va="top",
        arrowprops={"arrowstyle": "-", "color": NAVY, "lw": 0.8},
    )
    clean_axis(ax1)

    # Panel B: domain effects for the single prespecified integrated estimand.
    y = np.arange(len(d))
    low = min(-2.35, float(d["effect_pp"].min()) - 0.12, lo_pp - 0.12)
    high = max(0.72, float(d["effect_pp"].max()) + 0.12, hi_pp + 0.12)
    ax2.axvspan(low, 0.0, color=PALE_TEAL, zorder=0)
    ax2.axvspan(0.0, high, color=PALE_ORANGE, alpha=0.48, zorder=0)
    ax2.axvline(0.0, color=INK, linewidth=1.05, zorder=1)
    colors = np.where(d["effect_pp"] < 0.0, TEAL, ORANGE)
    for yi, effect, color in zip(y, d["effect_pp"], colors):
        ax2.plot([0.0, effect], [yi, yi], color=color, linewidth=1.45, zorder=2)
    ax2.scatter(d["effect_pp"], y, c=colors, s=27, edgecolor="white", linewidth=0.65, zorder=3)
    mean_y = len(d) + 0.65
    ax2.axhline(len(d) - 0.18, color=GRID, linewidth=0.9)
    ax2.errorbar(
        mean_pp,
        mean_y,
        xerr=[[mean_pp - lo_pp], [hi_pp - mean_pp]],
        fmt="D",
        color=NAVY,
        ecolor=NAVY,
        linewidth=1.7,
        elinewidth=1.4,
        capsize=3.0,
        markersize=5.7,
        markeredgecolor="white",
        markeredgewidth=0.65,
        zorder=4,
    )
    if chinese:
        title_b = "B  预算路径的逐域积分效应"
        xlab_b = "预算归一化 AUC 差值\n（百分点）"
        mean_label = "域等权均值"
    else:
        title_b = "B  Integrated effects across domains"
        xlab_b = "Budget-normalized AUC difference\n(percentage points)"
        mean_label = "Domain-equal mean"
    ax2.set_title(title_b, loc="left", color=INK, pad=7)
    ax2.set_xlabel(xlab_b, color=INK, labelpad=7)
    labels = list(d["domain"]) + [mean_label]
    ax2.set_yticks(list(y) + [mean_y])
    ax2.set_yticklabels(labels)
    ax2.set_xlim(low, high)
    ax2.set_ylim(-0.7, mean_y + 0.72)
    ax2.invert_yaxis()
    ax2.xaxis.grid(True, color=GRID, linewidth=0.75, zorder=0)
    clean_axis(ax2)

    legend_items = [
        Patch(facecolor=PALE_TEAL, edgecolor="none", label=("PCHP 更优" if chinese else "PCHP favored")),
        Line2D([0], [0], marker="D", color=NAVY, markerfacecolor=NAVY, linewidth=1.4, markersize=5, label=("域等权估计" if chinese else "Domain-equal estimate")),
    ]
    fig.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.015),
        ncol=2,
        frameon=False,
        fontsize=7.3,
        handlelength=1.5,
        columnspacing=1.8,
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    observed = {path.relative_to(ROOT).as_posix(): sha256(path) for path in EXPECTED_SHA256}
    expected = {
        path.relative_to(ROOT).as_posix(): value
        for path, value in EXPECTED_SHA256.items()
    }
    checks = {path: observed[path] == expected[path] for path in observed}
    if not all(checks.values()):
        raise RuntimeError(f"Input hash mismatch: {checks}; observed={observed}")
    summary = pd.read_csv(SUMMARY)
    auc = pd.read_csv(AUC)
    make_figure(summary, auc, False, "fig4_candidate_control")
    make_figure(summary, auc, True, "fig4_candidate_control_zh")
    outputs = {
        str(path): sha256(path)
        for path in [
            FIGURES / "fig4_candidate_control.pdf",
            FIGURES / "fig4_candidate_control.png",
            FIGURES / "fig4_candidate_control_zh.pdf",
            FIGURES / "fig4_candidate_control_zh.png",
        ]
    }
    (HERE / "figure_budget_path_v363_report.json").write_text(
        json.dumps(
            {
                "version": "V363",
                "input_hash_checks": checks,
                "v362_anchor_audit_required_and_passed": True,
                "deterministic_png_outputs": {
                    Path(path).name: digest
                    for path, digest in outputs.items()
                    if path.endswith(".png")
                },
                "pdf_outputs_generated_but_not_byte_hash_pinned": [
                    Path(path).name for path in outputs if path.endswith(".pdf")
                ],
                "reporting_boundary": "The integrated AUC is primary for the protocol-locked retrospective budget-path analysis; per-budget intervals are descriptive.",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
