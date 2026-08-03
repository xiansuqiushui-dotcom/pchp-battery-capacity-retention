"""Generate the five causal PCHP manuscript figures from frozen artifacts.

The script reads authoritative V325, V327, V333, and NASA V331 outputs. It
does not refit a model, alter a prediction, or select a post-outcome subgroup.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "figures"

NAVY = "#17365D"
TEAL = "#1F9D8A"
ORANGE = "#E76F51"
GOLD = "#D8AE45"
BLUE = "#4C78A8"
INK = "#19324A"
GRAY = "#657481"
MID = "#A8B4BF"
GRID = "#DDE4EA"
PALE_BLUE = "#EAF1F7"
PALE_TEAL = "#E4F3F0"
PALE_ORANGE = "#FBEAE5"
WHITE = "#FFFFFF"


def set_style(lang: str = "en") -> None:
    mpl.rcParams.update(
        {
            "font.family": "Microsoft YaHei" if lang == "zh" else "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.2,
            "xtick.labelsize": 8.1,
            "ytick.labelsize": 8.1,
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig: plt.Figure, stem: str, lang: str = "en") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if lang == "zh":
        stem = f"{stem}_zh"
    fig.savefig(OUT / f"{stem}.pdf", facecolor=WHITE)
    fig.savefig(OUT / f"{stem}.png", dpi=360, facecolor=WHITE)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.075,
        1.03,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )


def clean_axis(ax: plt.Axes, *, xgrid: bool = True) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MID)
    ax.tick_params(colors="#425466", length=3)
    if xgrid:
        ax.grid(axis="x", color=GRID, linewidth=0.75, zorder=0)


def card(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    color: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.0,
        edgecolor="#B8C3CD",
        facecolor="#F8FAFC",
    )
    ax.add_patch(patch)
    ax.add_patch(
        Rectangle(
            (x + 0.018, y + height - 0.029),
            width - 0.036,
            0.011,
            color=color,
            clip_on=False,
        )
    )
    ax.text(
        x + 0.025,
        y + height - 0.055,
        title,
        ha="left",
        va="top",
        fontsize=8.4,
        fontweight="bold",
        color=NAVY,
    )
    ax.text(
        x + 0.025,
        y + height - 0.118,
        body,
        ha="left",
        va="top",
        fontsize=7.35,
        color=INK,
        linespacing=1.35,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = GRAY,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10.5,
            linewidth=1.15,
            color=color,
            shrinkA=1,
            shrinkB=1,
        )
    )


def make_workflow(lang: str = "en") -> None:
    development = load_json(
        ROOT
        / "nested_prefix_causal_outer_v327"
        / "nested_prefix_causal_outer_v327_report.json"
    )
    candidate_control = load_json(
        ROOT
        / "candidate_information_control_v333"
        / "candidate_information_control_v333_report.json"
    )
    nasa = load_json(
        ROOT
        / "external_nasa_v329"
        / "scored_v331"
        / "nasa_frozen_confirmation_v331_report.json"
    )
    development_effect = float(
        development["primary_comparison"]["domain_equal_mean_difference"]
    )
    development_interval = (
        float(development["primary_comparison"]["ci95_lower"]),
        float(development["primary_comparison"]["ci95_upper"]),
    )
    control_effect = float(
        candidate_control["comparison"]["domain_equal_pchp_minus_control"]
    )
    control_interval = tuple(
        float(value)
        for value in candidate_control["comparison"][
            "domain_cluster_percentile_bootstrap_ci95"
        ]
    )
    nasa_effect = float(nasa["cell_equal_method_minus_baseline"])
    nasa_interval = tuple(
        float(value)
        for value in nasa["cell_cluster_percentile_bootstrap"]["ci95"]
    )
    if not all(
        interval[1] < 0
        for interval in (development_interval, control_interval, nasa_interval)
    ):
        raise RuntimeError("a headline paired-effect interval no longer excludes zero")
    development_wins = int(development["primary_comparison"]["domain_wins"])
    development_domains = int(development["domains"])
    nasa_wins = int(nasa["cell_wins_ties_losses"][0])
    nasa_cells = int(nasa["independent_cells"])

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.15))
    fig.subplots_adjust(wspace=0.20, hspace=0.34)

    ax = axes[0, 0]
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel_label(ax, "A")
    ax.set_title(
        "部署失效：未来数据改写历史" if lang == "zh" else "Deployment failure: future data revise the past",
        loc="left",
        pad=10,
    )
    ax.plot([0.08, 0.90], [0.54, 0.54], color=MID, linewidth=1.3)
    for x, label in [(0.16, r"$t-1$"), (0.40, r"$t$"), (0.70, r"$t+1$"), (0.88, r"$t+2$")]:
        ax.scatter([x], [0.54], s=46, color=BLUE if x <= 0.40 else GOLD, zorder=3)
        ax.text(x, 0.43, label, ha="center", color=INK)
    ax.annotate(
        "已发布记录" if lang == "zh" else "issued record",
        xy=(0.40, 0.54),
        xytext=(0.40, 0.79),
        ha="center",
        color=NAVY,
        arrowprops={"arrowstyle": "->", "color": NAVY, "lw": 1.1},
    )
    ax.annotate(
        "未来充电数据" if lang == "zh" else "future charge data",
        xy=(0.78, 0.54),
        xytext=(0.78, 0.79),
        ha="center",
        color=GRAY,
        arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 1.1},
    )
    ax.text(
        0.50,
        0.16,
        (
            "在线要求：时刻 $t$ 已发布的前缀必须保持不变"
            if lang == "zh"
            else "Online requirement: the prefix already emitted at $t$ must remain invariant"
        ),
        ha="center",
        color=INK,
        fontsize=8.1,
    )

    ax = axes[0, 1]
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel_label(ax, "B")
    ax.set_title("前缀因果状态与学习候选" if lang == "zh" else "Prefix-causal state and learned proposal", loc="left", pad=10)
    card(
        ax,
        0.02,
        0.30,
        0.26,
        0.43,
        "原始基线" if lang == "zh" else "Raw baseline",
        (r"仅用源域 $r_t$" "\n可提前获得") if lang == "zh" else (r"source-only $r_t$" "\nearly available"),
        BLUE,
    )
    card(
        ax,
        0.37,
        0.30,
        0.26,
        0.43,
        "因果状态" if lang == "zh" else "Causal state",
        ("拒绝向上更新" + "\n" + r"以 $\alpha$ 吸收向下更新") if lang == "zh" else ("reject upward" + "\n" + r"absorb downward at $\alpha$"),
        GOLD,
    )
    card(
        ax,
        0.72,
        0.30,
        0.26,
        0.43,
        "候选预测" if lang == "zh" else "Candidate",
        (r"逐记录 $c_t$" "\n投运参照变化") if lang == "zh" else (r"record-specific $c_t$" "\ncommissioning change"),
        TEAL,
    )
    arrow(ax, (0.285, 0.365), (0.365, 0.365))
    arrow(ax, (0.715, 0.365), (0.635, 0.365), TEAL)
    ax.text(
        0.50,
        0.15,
        (
            r"$b_t=b_{t-1}+\alpha\,\min(r_t-b_{t-1},0)$ 仅使用已观测前缀"
            if lang == "zh"
            else r"$b_t=b_{t-1}+\alpha\,\min(r_t-b_{t-1},0)$ uses only the observed prefix"
        ),
        ha="center",
        color=NAVY,
        fontsize=8.5,
    )

    ax = axes[1, 0]
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel_label(ax, "C")
    ax.set_title("递归可行区间与精确损害管" if lang == "zh" else "Recursive feasible interval and exact harm tube", loc="left", pad=10)
    y = 0.56
    ax.plot([0.08, 0.92], [y, y], color=MID, linewidth=1.2)
    lo, b, hi, previous, candidate = 0.30, 0.48, 0.66, 0.72, 0.86
    ax.add_patch(Rectangle((lo, y - 0.07), hi - lo, 0.14, facecolor=PALE_TEAL, edgecolor=TEAL, linewidth=1.2))
    ax.axvline(previous, ymin=0.30, ymax=0.70, color=GOLD, linewidth=1.7)
    ax.scatter([b], [y], marker="D", s=50, color=NAVY, zorder=4)
    ax.scatter([candidate], [y], marker="o", s=50, color=ORANGE, zorder=4)
    ax.scatter([hi], [y], marker="D", s=55, color=TEAL, zorder=5)
    ax.text(b, 0.72, (r"状态 $b_t$" if lang == "zh" else r"state $b_t$"), ha="center", color=NAVY)
    ax.text(previous, 0.34, (r"上一输出 $p_{t-1}$" if lang == "zh" else r"previous $p_{t-1}$"), ha="center", color=GOLD)
    ax.text(candidate, 0.72, (r"候选 $c_t$" if lang == "zh" else r"candidate $c_t$"), ha="center", color=ORANGE)
    ax.text((lo + hi) / 2, 0.48, r"$|p_t-b_t|\leq\delta$", ha="center", color=TEAL)
    arrow(ax, (candidate - 0.01, 0.40), (hi + 0.01, 0.40), ORANGE)
    ax.text(
        0.50,
        0.15,
        r"$p_t=\Pi_{[\max(0,b_t-\delta),\,\min(1.3,b_t+\delta,p_{t-1})]}(c_t)$",
        ha="center",
        color=NAVY,
        fontsize=8.6,
    )

    ax = axes[1, 1]
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel_label(ax, "D")
    ax.set_title("证据链：不同问题使用不同单位" if lang == "zh" else "Evidence chain: distinct questions, distinct units", loc="left", pad=10)
    development_body = (
        f"{development_domains} 个数据域\n$\\Delta$MAE {development_effect:.5f}\n"
        f"{development_wins}/{development_domains} 个域改善"
        if lang == "zh"
        else f"{development_domains} dataset domains\n$\\Delta$MAE {development_effect:.5f}\n"
        f"{development_wins}/{development_domains} improved"
    )
    control_body = (
        f"{development_domains} 个数据域\n$\\Delta$MAE {control_effect:.5f}\n"
        "优于常数安全偏移"
        if lang == "zh"
        else f"{development_domains} dataset domains\n$\\Delta$MAE {control_effect:.5f}\n"
        "vs constant safe shift"
    )
    nasa_body = (
        f"{nasa_cells} 个电池\n$\\Delta$MAE {nasa_effect:.5f}\n"
        f"{nasa_wins}/{nasa_cells} 个电池改善"
        if lang == "zh"
        else f"{nasa_cells} batteries\n$\\Delta$MAE {nasa_effect:.5f}\n"
        f"{nasa_wins}/{nasa_cells} improved"
    )
    card(
        ax,
        0.02,
        0.30,
        0.27,
        0.43,
        "嵌套主效应" if lang == "zh" else "Nested effect",
        development_body,
        BLUE,
    )
    card(
        ax,
        0.365,
        0.30,
        0.27,
        0.43,
        "匹配控制" if lang == "zh" else "Matched control",
        control_body,
        GOLD,
    )
    card(
        ax,
        0.71,
        0.30,
        0.27,
        0.43,
        "NASA 一次性检验" if lang == "zh" else "NASA one-shot",
        nasa_body,
        TEAL,
    )
    arrow(ax, (0.295, 0.365), (0.36, 0.365))
    arrow(ax, (0.64, 0.365), (0.705, 0.365))
    ax.text(
        0.50,
        0.15,
        (
            "三项 $95\%$ 区间均排除零；全部确定性证书成立"
            if lang == "zh"
            else "All three $95\%$ intervals excluded zero; all deterministic certificates held"
        ),
        ha="center",
        color=INK,
        fontsize=8.0,
    )

    save(fig, "fig1_method_workflow", lang)


def forest_plot(
    values: pd.Series,
    mean: float,
    interval: tuple[float, float],
    stem: str,
    xlabel: str,
    title: str,
    note: str,
    lang: str = "en",
) -> None:
    ordered = values.sort_values()
    labels = list(ordered.index.astype(str)) + ["域等权均值" if lang == "zh" else "Domain-equal mean"]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ax.axvspan(ax.get_xlim()[0], 0, color=PALE_TEAL, alpha=0.45, zorder=0)
    for pos, value in enumerate(ordered.to_numpy(float)):
        color = TEAL if value < 0 else ORANGE
        ax.hlines(pos, 0, value, color=color, linewidth=1.5, zorder=2)
        ax.scatter(value, pos, s=42, color=color, edgecolor=WHITE, linewidth=0.6, zorder=3)
    aggregate_y = len(ordered) + 0.2
    ax.errorbar(
        mean,
        aggregate_y,
        xerr=np.array([[mean - interval[0]], [interval[1] - mean]]),
        fmt="D",
        markersize=7,
        color=NAVY,
        ecolor=NAVY,
        elinewidth=2.0,
        capsize=4,
        zorder=4,
    )
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.set_yticks(list(range(len(ordered))) + [aggregate_y], labels)
    ax.set_ylim(-0.8, aggregate_y + 0.8)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="bold", pad=9)
    clean_axis(ax)
    xmin, xmax = ax.get_xlim()
    ax.axvspan(xmin, 0, color=PALE_TEAL, alpha=0.45, zorder=-1)
    fig.tight_layout()
    save(fig, stem, lang)


def make_domain_effects(lang: str = "en") -> None:
    report = load_json(ROOT / "nested_prefix_causal_outer_v327" / "nested_prefix_causal_outer_v327_report.json")
    comparison = report["primary_comparison"]
    values = pd.Series(comparison["per_domain_difference"], name="difference")
    forest_plot(
        values,
        float(comparison["domain_equal_mean_difference"]),
        (float(comparison["ci95_lower"]), float(comparison["ci95_upper"])),
        "fig2_domain_effects",
        "PCHP 相对因果基线的电芯宏平均 MAE 差值" if lang == "zh" else "PCHP minus causal baseline cell-macro MAE",
        "仅用源域的嵌套选择在 12 个域中的 11 个取得改善" if lang == "zh" else "Nested source-only selection improves 11 of 12 domains",
        r"$n=12$ 个完整数据域；负值表示 PCHP 更优" if lang == "zh" else r"$n=12$ complete dataset domains; negative favors PCHP",
        lang,
    )


def make_prefix_causality(lang: str = "en") -> None:
    revision = pd.read_csv(ROOT / "prefix_causal_falsification_v325" / "future_revision_domain_v325.csv")
    shock = pd.read_csv(ROOT / "prefix_causal_falsification_v325" / "directional_shock_domain_v325.csv")
    domains = sorted(revision["domain"].astype(str))
    revision = revision.set_index("domain").loc[domains]
    shock = shock.set_index("domain").loc[domains]
    y = np.arange(len(domains))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.15), gridspec_kw={"width_ratios": [0.92, 1.08]})
    fig.subplots_adjust(wspace=0.40)

    ax = axes[0]
    ax.barh(y, revision["fraction_cells_revised"], color=BLUE, height=0.58, zorder=2)
    ax.scatter(np.zeros_like(y), y, marker="D", s=28, color=TEAL, zorder=3)
    ax.axvline(0.5, color=ORANGE, linewidth=1.1, linestyle="--")
    ax.set_yticks(y, domains)
    ax.invert_yaxis()
    ax.set_xlim(-0.035, 1.05)
    ax.set_xlabel("历史被修订的电芯比例" if lang == "zh" else "Fraction of cells with revised history")
    ax.set_title("未来扩展会修订离线历史" if lang == "zh" else "Future extension revises offline history", loc="left", fontweight="bold")
    clean_axis(ax)
    panel_label(ax, "A")
    ax.text(0.505, 0.99, "0.5 门槛" if lang == "zh" else "0.5 gate", transform=ax.get_xaxis_transform(), color=ORANGE, ha="left", va="top", fontsize=7.8)

    ax = axes[1]
    x_fast = shock["median_cumulative_disturbance_alpha_1"].to_numpy(float)
    x_slow = shock["median_cumulative_disturbance_alpha_0p02"].to_numpy(float)
    for pos, fast, slow in zip(y, x_fast, x_slow):
        ax.hlines(pos, slow, fast, color=MID, linewidth=1.4, zorder=1)
    ax.scatter(x_fast, y, s=42, marker="o", color=GRAY, zorder=3, label=r"$\alpha=1$")
    ax.scatter(x_slow, y, s=46, marker="D", color=TEAL, zorder=3, label=r"$\alpha=0.02$")
    ax.set_xscale("log")
    ax.set_yticks(y, domains)
    ax.invert_yaxis()
    ax.set_xlabel("累计扰动中位数（SOH 单位，对数尺度）" if lang == "zh" else "Median cumulative disturbance (SOH units, log scale)")
    ax.set_title("受控吸收减弱冲击" if lang == "zh" else "Controlled assimilation attenuates shocks", loc="left", fontweight="bold")
    clean_axis(ax)
    panel_label(ax, "B")
    ax.legend(loc="upper right", frameon=False)

    save(fig, "fig3_prefix_causality", lang)


def make_candidate_control(lang: str = "en") -> None:
    report = load_json(ROOT / "candidate_information_control_v333" / "candidate_information_control_v333_report.json")
    comparison = report["comparison"]
    domain = pd.read_csv(ROOT / "candidate_information_control_v333" / "candidate_information_domain_metrics_v333.csv")
    values = domain.set_index("domain")["pchp_minus_control"]
    ci = comparison["domain_cluster_percentile_bootstrap_ci95"]
    forest_plot(
        values,
        float(comparison["domain_equal_pchp_minus_control"]),
        (float(ci[0]), float(ci[1])),
        "fig4_candidate_control",
        "PCHP 相对候选无关控制的电芯宏平均 MAE 差值" if lang == "zh" else "PCHP minus candidate-free control cell-macro MAE",
        "逐记录候选信息优于常数安全偏移" if lang == "zh" else "Record-specific candidate information exceeds a constant safe shift",
        r"匹配 $\alpha$ 与 $\delta=0.01$；$n=12$ 个域" if lang == "zh" else r"matched $\alpha$ and $\delta=0.01$; $n=12$ domains",
        lang,
    )


def make_nasa_stress(lang: str = "en") -> None:
    report = load_json(ROOT / "external_nasa_v329" / "scored_v331" / "nasa_frozen_confirmation_v331_report.json")
    cells = pd.read_csv(ROOT / "external_nasa_v329" / "scored_v331" / "nasa_cell_metrics_v331.csv")
    labels = pd.read_parquet(ROOT / "external_nasa_v329" / "scored_v331" / "nasa_released_labels_v331.parquet")
    cells = cells.sort_values("method_minus_baseline").reset_index(drop=True)
    order = cells["cell_id"].astype(str).tolist()
    y_lookup = {cell: pos for pos, cell in enumerate(order)}
    y = np.arange(len(cells))

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 7.25), gridspec_kw={"width_ratios": [1.03, 1.07]})
    fig.subplots_adjust(wspace=0.20)

    ax = axes[0]
    effects = cells["method_minus_baseline"].to_numpy(float)
    colors = np.where(effects < 0, TEAL, ORANGE)
    ax.hlines(y, 0, effects, color=colors, linewidth=1.35, zorder=2)
    ax.scatter(effects, y, c=colors, s=34, edgecolor=WHITE, linewidth=0.5, zorder=3)
    mean = float(report["cell_equal_method_minus_baseline"])
    ci = report["cell_cluster_percentile_bootstrap"]["ci95"]
    mean_y = len(cells) + 0.6
    ax.errorbar(
        mean,
        mean_y,
        xerr=np.array([[mean - ci[0]], [ci[1] - mean]]),
        fmt="D",
        color=NAVY,
        ecolor=NAVY,
        markersize=6.5,
        elinewidth=1.9,
        capsize=3.5,
        zorder=4,
    )
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.set_yticks(list(y) + [mean_y], order + ["电池等权均值" if lang == "zh" else "Cell-equal mean"])
    ax.set_ylim(-0.8, mean_y + 0.8)
    ax.invert_yaxis()
    ax.set_xlabel("PCHP 相对基线的 MAE 差值" if lang == "zh" else "PCHP minus baseline MAE")
    ax.set_title("冻结相对效应" if lang == "zh" else "Frozen relative effect", loc="left", fontweight="bold")
    clean_axis(ax)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3f"))
    panel_label(ax, "A")
    ax.text(
        0.98,
        0.98,
        "27 个改善 / 6 个恶化" if lang == "zh" else "27 improved / 6 worsened",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=GRAY,
    )

    ax = axes[1]
    valid = labels.loc[
        labels["cell_id"].astype(str).isin(order)
        & np.isfinite(labels["soh_initial5"].to_numpy(float))
        & (labels["soh_initial5"].to_numpy(float) > 0),
        ["cell_id", "soh_initial5"],
    ].copy()
    valid["y"] = valid["cell_id"].astype(str).map(y_lookup)
    inside = valid["soh_initial5"].to_numpy(float) <= 1.3
    ax.scatter(
        valid.loc[inside, "soh_initial5"],
        valid.loc[inside, "y"],
        s=8,
        color=BLUE,
        alpha=0.28,
        linewidth=0,
        rasterized=True,
        label=(r"已发布标签 $\leq1.3$" if lang == "zh" else r"released label $\leq1.3$"),
    )
    ax.scatter(
        valid.loc[~inside, "soh_initial5"],
        valid.loc[~inside, "y"],
        s=13,
        color=ORANGE,
        alpha=0.62,
        linewidth=0,
        rasterized=True,
        label=(r"已发布标签 $>1.3$" if lang == "zh" else r"released label $>1.3$"),
    )
    positive_min = float(valid["soh_initial5"].min())
    ax.axvspan(max(positive_min * 0.85, 1e-3), 1.3, color=PALE_BLUE, alpha=0.7, zorder=-1)
    ax.axvline(1.3, color=NAVY, linewidth=1.1, linestyle="--", label="冻结模型上界" if lang == "zh" else "frozen model upper bound")
    ax.set_xscale("log")
    ax.set_xlim(max(positive_min * 0.8, 1e-3), float(valid["soh_initial5"].max()) * 1.18)
    ax.set_ylim(-0.8, mean_y + 0.8)
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xlabel("已发布归一化容量（对数尺度）" if lang == "zh" else "Released normalized capacity (log scale)")
    ax.set_title("绝对目标兼容性边界" if lang == "zh" else "Absolute target compatibility boundary", loc="left", fontweight="bold")
    clean_axis(ax)
    panel_label(ax, "B")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markersize=5, alpha=0.55, label=(r"标签 $\leq1.3$" if lang == "zh" else r"label $\leq1.3$")),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, markersize=5, label=(r"标签 $>1.3$" if lang == "zh" else r"label $>1.3$")),
        Line2D([0], [0], color=NAVY, linestyle="--", linewidth=1.1, label="模型上界" if lang == "zh" else "model upper bound"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    ax.annotate(
        "B0041: 21.77",
        xy=(21.770609322382242, y_lookup["B0041"]),
        xytext=(7.0, y_lookup["B0041"] - 2.2),
        color=ORANGE,
        arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.0},
        fontsize=8.0,
    )

    save(fig, "fig5_nasa_stress", lang)


def main() -> None:
    for lang in ("en", "zh"):
        set_style(lang)
        make_workflow(lang)
        make_domain_effects(lang)
        make_prefix_causality(lang)
        make_candidate_control(lang)
        make_nasa_stress(lang)
    print(f"wrote causal manuscript figures to {OUT}")


if __name__ == "__main__":
    main()
