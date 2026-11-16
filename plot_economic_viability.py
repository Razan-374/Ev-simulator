"""Plot economic viability against mean maximum V2G peak reduction."""

import argparse

import matplotlib.pyplot as plt
import numpy as np


SLOPE_EUR_PER_KW = 149.285
FIXED_COST_EUR_PER_YEAR = 30753
BREAK_EVEN_KW = FIXED_COST_EUR_PER_YEAR / SLOPE_EUR_PER_KW


def make_plot(
    outfile="economic_viability_v2g.png",
    fmt=None,
):
    peak_reduction_kw = np.array([107.0, 175.7])
    annual_net_benefit_eur = np.array([-14779.0, -4523.0])
    labels = ["11/10 kW", "22/22 kW"]
    colors = ["#d62728", "#1f77b4"]

    x_min = 75.0
    x_max = 250.0
    x_line = np.linspace(x_min, x_max, 400)
    benefit_line = SLOPE_EUR_PER_KW * x_line - FIXED_COST_EUR_PER_YEAR

    fig, ax = plt.subplots(figsize=(9.2, 7))
    ax.axhspan(0, benefit_line.max() + 2500, color="#dff0d8", alpha=0.65)
    ax.axhspan(benefit_line.min() - 2500, 0, color="#f8d7da", alpha=0.65)
    ax.plot(
        x_line,
        benefit_line,
        color="#333333",
        linewidth=2,
        label="Annual net-benefit line",
    )
    ax.axhline(0, color="#555555", linewidth=1)
    ax.axvline(
        BREAK_EVEN_KW,
        color="#555555",
        linestyle=":",
        linewidth=1.5,
        label=f"Break-even = {BREAK_EVEN_KW:.0f} kW",
    )

    for x_value, y_value, label, color in zip(
        peak_reduction_kw,
        annual_net_benefit_eur,
        labels,
        colors,
    ):
        ax.scatter(x_value, y_value, color=color, s=85, zorder=3, label=label)
        ax.annotate(
            f"{label}\n{x_value:.1f} kW\n{y_value:,.0f} EUR/year",
            (x_value, y_value),
            xytext=(8, 10),
            textcoords="offset points",
            color=color,
            fontsize=11,
            va="bottom",
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-20000, 20000)
    ax.set_xlabel("Mean maximum facility peak reduction (kW)", fontsize=18)
    ax.set_ylabel("Annual net benefit (EUR/year)", fontsize=18)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=11, loc="upper left")
    fig.tight_layout()

    save_kwargs = {"dpi": 200, "bbox_inches": "tight"}
    if fmt is not None:
        save_kwargs["format"] = fmt
    fig.savefig(outfile, **save_kwargs)
    fig.savefig("economic_viability_v2g.eps", format="eps", bbox_inches="tight")
    print(f"Break-even: {BREAK_EVEN_KW:.2f} kW")
    print(f"Saved figure to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot V2G economic viability")
    parser.add_argument("--out", default="economic_viability_v2g.png")
    parser.add_argument("--format", default=None, help="Optional format, e.g. eps or png")
    args = parser.parse_args()
    make_plot(args.out, fmt=args.format)
