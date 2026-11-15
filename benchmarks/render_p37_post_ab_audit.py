"""Render the regressed task using frozen diagnostic curves, not new inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.audit_dir
    controls = json.loads((run / "case_controls.json").read_text())
    case = json.loads((run / "case_report.json").read_text())
    with np.load(run / "case_curves.npz") as saved:
        curves = {name: saved[name] for name in saved.files}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig = plt.figure(figsize=(13, 7.8), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(1, 1.65))
    map_ax = fig.add_subplot(grid[:, 0])
    progress_ax = fig.add_subplot(grid[0, 1])
    support_ax = fig.add_subplot(grid[1, 1])
    image, xy, arc = curves["image"], curves["xy"], curves["arc"]
    map_ax.imshow(image, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    path = curves["old_accepted_path"]
    map_ax.plot(path[:, 0], path[:, 1], color="#1476ac", lw=1.8, label="Old accepted safe path")
    missing = curves["B_route_increment_occupancy"][1] <= .1
    map_ax.scatter(xy[missing, 0], xy[missing, 1], s=6, c="#ed9a24", label="B candidate 1: low support", zorder=4)
    colors = ["#d1495b", "#e69126", "#7554a3", "#c85e99"]
    stages = case["arms"]["B_route_increment"]["bounded_fallback"]["trace"]["adaptive_stages"]
    events = case["arms"]["B_route_increment"]["bounded_fallback"]["trace"]["adaptive_rejections"]
    for event in events[stages[-1]["events_begin"]:stages[-1]["events_end"]]:
        c = event["candidate"]
        x, y = event["previous_px"]
        map_ax.scatter(x, y, marker="x", s=50, color=colors[c], lw=2, zorder=6)
    map_ax.scatter(*curves["start"], marker="o", s=40, color="#16826d", edgecolor="white", zorder=7, label="Start")
    map_ax.scatter(*curves["goal"], marker="*", s=110, color="#1476ac", edgecolor="white", zorder=7, label="Goal")
    map_ax.text(15, 15, "All B candidates stop here", fontsize=9, va="top", color="#ab3042",
                bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none"})
    map_ax.annotate("", xy=(347, 101), xytext=(190, 40), arrowprops={"arrowstyle": "->", "color": "#ab3042"})
    map_ax.set(title=f"Task {case['task_id']} | 768 x 512", xlabel="x (pixels)", ylabel="y (pixels)")
    map_ax.legend(loc="lower left", fontsize=8, framealpha=.95)

    old_c = controls["old_successful_candidate"]
    portion = arc <= 250
    progress_ax.plot(arc[portion], curves["epoch15_progress"][old_c, portion], color="#1476ac", lw=2.4,
                     label="Old successful candidate")
    for c, color in enumerate(colors):
        progress_ax.plot(arc[portion], curves["B_route_increment_progress"][c, portion], color=color, lw=1.4,
                         label=f"B candidate {c}")
    progress_ax.axvspan(158, 166, color="#555555", alpha=.12)
    progress_ax.text(170, .025, "B stops near this turn", fontsize=9, color="#555555")
    progress_ax.set(title="Predicted progress on the same old safe path (early section)",
                    xlabel="Arc length along old safe path (pixels)", ylabel="Corrected predicted progress", xlim=(0, 250))
    progress_ax.grid(axis="y", alpha=.2)
    progress_ax.legend(loc="upper left", fontsize=8, ncol=2)

    for c, color in enumerate(colors):
        support_ax.plot(arc, curves["B_route_increment_occupancy"][c].clip(0, 1), color=color,
                        lw=1.8 if c == 1 else 1, alpha=1 if c == 1 else .5, label=f"B candidate {c}")
    support_ax.axhline(.1, color="#444444", ls="--", lw=1, label="Support threshold 0.1")
    gap = controls["measurements_on_old_accepted_safe_path"]["B_route_increment"]["candidates"][1]["missing_intervals_arc_px"][0]
    support_ax.axvspan(*gap, color="#ed9a24", alpha=.12)
    support_ax.text(arc[-1] * .42, .28, "Candidate 1: ~58 px below support threshold", fontsize=9, color="#a86a13")
    support_ax.set(title="B support on the same old safe path (full route)",
                   xlabel="Arc length along old safe path (pixels)", ylabel="Predicted support",
                   xlim=(0, arc[-1]), ylim=(-.05, 1.05))
    support_ax.grid(axis="y", alpha=.2)
    support_ax.legend(loc="lower left", fontsize=8, ncol=3)
    fig.suptitle("Task 17831 regression: frozen-field diagnosis", fontsize=15, weight="bold")
    fig.savefig(run / "task17831_diagnosis.png", dpi=160)
    fig.savefig(run / "task17831_diagnosis.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
