"""Plot V1G HC90 across available outlets for 11 and 22 kW configurations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def make_plot(outfile="v1g_hc90_available_outlets.png"):
    outlets = np.array([3, 5, 7, 9, 11, 13, 15, 17])

    series = [
        (
            "11 kW V1G — Average",
            np.array([10, 20, 100, 100, 100, 100, 100, 100]),
            "#d62728",
            "o",
            "-",
        ),
        (
            "11 kW V1G — Worst peak",
            np.array([5, 5, 5, 5, 5, 5, 5, 5]),
            "#1f77b4",
            "s",
            "--",
        ),
        (
            "22 kW V1G — Average",
            np.array([90, 100, 100, 100, 100, 100, 100, 100]),
            "#2ca02c",
            "^",
            "-",
        ),
        (
            "22 kW V1G — Worst peak",
            np.array([20, 25, 25, 25, 25, 25, 25, 25]),
            "#9467bd",
            "D",
            "--",
        ),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 6.8))

    for label, values, color, marker, linestyle in series:
        ax.plot(
            outlets,
            values,
            marker=marker,
            color=color,
            linestyle=linestyle,
            linewidth=2.2,
            markersize=7,
            label=label,
        )

    ax.set_xlabel("Available controllable outlets", fontsize=15)
    ax.set_ylabel(r"HC$_{90}$ (EVs)", fontsize=15)
    ax.set_xticks(outlets)
    ax.set_yticks(np.arange(0, 111, 10))
    ax.set_ylim(0, 110)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=11, framealpha=0.9)

    fig.tight_layout()
    output_path = Path(outfile)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".eps"), format="eps", bbox_inches="tight")
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    make_plot()
