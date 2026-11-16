"""Plot 22 kW V2G HC90 and mean facility peak reduction.

Suggested caption:
Impact of available controllable outlets on (a) HC90 and (b) mean maximum
facility peak reduction for the 22/22 kW V2G configuration under the
average-load and historical worst-peak profiles.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np


FIGURE_CAPTION = (
	"Impact of available controllable outlets on (a) HC90 and (b) mean "
	"maximum facility peak reduction for the 22/22 kW V2G configuration "
	"under the average-load and historical worst-peak profiles."
)


def make_plot(outfile="hosting_capacity_peak_reduction_v2g_22kw.png", fmt=None):
	outlets = np.array([3, 5, 7, 9, 11, 13, 15, 17])

	hc_v2g_avg = np.array([90, 100, 100, 100, 100, 100, 100, 100])
	hc_v2g_worst = np.array([35, 100, 100, 100, 100, 100, 100, 100])

	peak_v2g_avg = np.array([
		45.9333, 75.5365, 107.9615, 146.0626,
		169.3247, 159.0889, 163.9998, 175.7038,
	])
	peak_v2g_worst = np.array([
		49.7805, 93.7805, 137.7805, 181.7805,
		225.7805, 268.6019, 302.6986, 320.9743,
	])

	average_color = "#d62728"
	worst_color = "#1f77b4"
	fig, axes = plt.subplots(1, 2, figsize=(16, 7))
	fig.subplots_adjust(left=0.08, right=0.98, top=0.94, wspace=0.41)

	label_fontsize = 20
	tick_fontsize = 14
	legend_fontsize = 14

	ax = axes[0]
	ax.plot(
		outlets, hc_v2g_avg, marker="o", color=average_color,
		linestyle="-", lw=2, label="V2G HC90 (average)",
	)
	ax.plot(
		outlets, hc_v2g_worst, marker="s", color=worst_color,
		linestyle="--", lw=2, label="V2G HC90 (worst-peak)",
	)
	ax.set_xlabel("Available controllable outlets", fontsize=label_fontsize)
	ax.set_ylabel(r"$HC_{90}$ (EVs)", fontsize=label_fontsize)
	ax.set_xticks(outlets)
	ax.set_yticks(np.arange(0, 111, 10))
	ax.set_ylim(0, 110)
	ax.tick_params(axis="both", labelsize=tick_fontsize)
	ax.grid(True, linestyle="--", alpha=0.5)
	ax.legend(fontsize=legend_fontsize)

	ax = axes[1]
	ax.plot(
		outlets, peak_v2g_avg, marker="o", color=average_color,
		linestyle="-", lw=2, label="Average profile",
	)
	ax.plot(
		outlets, peak_v2g_worst, marker="s", color=worst_color,
		linestyle="--", lw=2, label="Worst-peak profile",
	)
	ax.set_xlabel("Available controllable outlets", fontsize=label_fontsize)
	ax.set_ylabel("Mean maximum facility peak reduction (kW)", fontsize=label_fontsize)
	ax.set_xticks(outlets)
	ax.set_ylim(0, 350)
	ax.tick_params(axis="both", labelsize=tick_fontsize)
	ax.grid(True, linestyle="--", alpha=0.5)
	ax.legend(fontsize=legend_fontsize)

	save_kwargs = {"dpi": 200, "bbox_inches": "tight"}
	if fmt is not None:
		save_kwargs["format"] = fmt
	fig.savefig(outfile, **save_kwargs)
	fig.savefig("hosting_capacity_peak_reduction_v2g_22kw.eps", format="eps", bbox_inches="tight")
	print(f"Saved figure to {outfile}")
	print(f"Caption: {FIGURE_CAPTION}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Create 22 kW V2G HC90 and peak-reduction figure")
	parser.add_argument("--out", default="hosting_capacity_peak_reduction_v2g_22kw.png")
	parser.add_argument("--format", default=None, help="Optional output format, e.g. eps, png")
	args = parser.parse_args()
	make_plot(args.out, fmt=args.format)
