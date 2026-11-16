"""Plot corrected V2G HC90 and mean facility peak reduction."""

import argparse

import matplotlib.pyplot as plt
import numpy as np


def make_plot(
    outfile="hosting_capacity_peak_reduction_v2g_11_10kw.png",
    fmt=None,
):
    outlets = np.array([3, 5, 7, 9, 11, 13, 15, 17])

    # Panel (a): corrected 11/10 kW V2G HC90 from the 10-seed run.
    hc_v2g_avg = np.array([15, 35, 100, 100, 100, 100, 100, 100])
    hc_v2g_worst = np.array([10, 20, 30, 40, 100, 100, 100, 100])

    # Panel (b): 10-seed mean maximum facility peak reduction.
    peak_v2g_avg = np.array([
        7.9333, 24.1644, 37.5845, 52.0165,
        66.0010, 77.4158, 90.9344, 107.0061,
    ])
    peak_v2g_worst = np.array([
        8.7805, 30.7805, 53.7805, 73.7805,
        91.7805, 107.6805, 125.5805, 145.3805,
    ])

    v2g_color = "#d62728"
    worst_color = "#1f77b4"
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.94, wspace=0.41)

    label_fontsize = 20
    tick_fontsize = 14
    legend_fontsize = 14

    ax = axes[0]
    ax.plot(
        outlets, hc_v2g_avg, marker="o", color=v2g_color,
        linestyle="-", lw=2, label="V2G HC90 (average)",
    )
    ax.plot(
        outlets, hc_v2g_worst, marker="s", color=worst_color,
        linestyle="--", lw=2, label="V2G HC90 (worst-peak)",
    )
    ax.set_xlabel("Available controllable outlets", fontsize=label_fontsize)
    ax.set_ylabel(r"$HC_{90}$ (EVs)", fontsize=label_fontsize)
    ax.set_xticks(outlets)
    ax.set_yticks(np.arange(10, 111, 10))
    ax.set_ylim(0, 110)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=legend_fontsize)

    ax = axes[1]
    ax.plot(
        outlets, peak_v2g_avg, marker="o", color=v2g_color,
        linestyle="-", lw=2, label="Average profile",
    )
    ax.plot(
        outlets, peak_v2g_worst, marker="s", color=worst_color,
        linestyle="--", lw=2, label="Worst-peak profile",
    )
    ax.set_xlabel("Available controllable outlets", fontsize=label_fontsize)
    ax.set_ylabel("Mean maximum facility peak reduction (kW)", fontsize=label_fontsize)
    ax.set_xticks(outlets)
    ax.set_ylim(0, 160)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=legend_fontsize)

    save_kwargs = {"dpi": 200, "bbox_inches": "tight"}
    if fmt is not None:
        save_kwargs["format"] = fmt
    fig.savefig(outfile, **save_kwargs)
    fig.savefig("hosting_capacity_peak_reduction_v2g_11_10kw.eps", format="eps", bbox_inches="tight")
    print(f"Saved figure to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create corrected V2G HC90 and peak-reduction figure")
    parser.add_argument(
        "--out",
        default="hosting_capacity_peak_reduction_v2g_11_10kw.png",
        help="Output file",
    )
    parser.add_argument("--format", default=None, help="Optional output format, e.g. eps, png")
    args = parser.parse_args()
    make_plot(args.out, fmt=args.format)
