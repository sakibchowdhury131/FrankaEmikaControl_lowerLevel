"""
plot_joint_traces_tau.py
========================
Plot joint positions, velocities, accelerations, and torques
from joint_traces_tau.csv.

Usage:
    python plot_joint_traces_tau.py
    python plot_joint_traces_tau.py --save
    python plot_joint_traces_tau.py --file path/to/file.csv
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

JOINTS = [f"J{i}" for i in range(7)]
COLORS = plt.cm.tab10.colors[:7]

def load(csv_path: Path) -> dict:
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    return data

def make_figure(data, title: str, ylabel: str, cols: list[str], save: bool, out_path: Path):
    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    t = data["t"]
    for i, (ax, col, color) in enumerate(zip(axes, cols, COLORS)):
        ax.plot(t, data[col], color=color, linewidth=1.2)
        ax.set_ylabel(f"J{i}\n{ylabel}", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel("Time [s]", fontsize=10)
    fig.tight_layout()
    if save:
        fig.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")
    return fig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="joint_traces_tau.csv", help="CSV file path")
    parser.add_argument("--save", action="store_true", help="Save figures as PNG")
    args = parser.parse_args()

    csv_path = Path(args.file)
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    data = load(csv_path)

    groups = [
        ("Joint Positions",     "pos [rad]",   [f"q{i}"   for i in range(7)], "fig_traces_positions.png"),
        ("Joint Velocities",    "vel [rad/s]", [f"dq{i}"  for i in range(7)], "fig_traces_velocities.png"),
        ("Joint Accelerations", "acc [rad/s²]",[f"ddq{i}" for i in range(7)], "fig_traces_accelerations.png"),
        ("Joint Torques",       "tau [Nm]",    [f"tau{i}" for i in range(7)], "fig_traces_torques.png"),
    ]

    figs = []
    for title, ylabel, cols, fname in groups:
        out = csv_path.parent / fname
        figs.append(make_figure(data, title, ylabel, cols, args.save, out))

    plt.show()

if __name__ == "__main__":
    main()
