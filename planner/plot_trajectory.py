"""
plot_trajectory.py
==================
Plot planned vs actual joint positions, velocities, torques and
runtime monitoring signals (sigma_min, tau_ext).

Usage:
    python plot_trajectory.py                                      # default paths
    python plot_trajectory.py trajectory.csv actual_trajectory.csv
    python plot_trajectory.py trajectory.csv actual_trajectory.csv --save

Output:
    - Interactive matplotlib window (one tab per signal group)
    - Optionally saves figures as PNG files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────────────────
#  Style
# ─────────────────────────────────────────────────────────────────────────────

PLANNED_COLOR  = "#2563EB"   # blue
ACTUAL_COLOR   = "#DC2626"   # red
LIMIT_COLOR    = "#6B7280"   # grey
ESTOP_COLOR    = "#F59E0B"   # amber

JOINT_NAMES = [f"Joint {i+1}" for i in range(7)]

# Franka Panda hardware limits
TAU_MAX  = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)
VEL_MAX  = np.array([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610])
SIGMA_MIN_ESTOP = 0.02
TAU_EXT_ESTOP   = 10.0


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_planned(path: str) -> dict:
    """
    Load trajectory.csv (29 columns):
      t | q0..q6 | dq0..dq6 | ddq0..ddq6 | tau0..tau6
    """
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return {
        "t":   data[:, 0],
        "q":   data[:, 1:8],
        "dq":  data[:, 8:15],
        "ddq": data[:, 15:22],
        "tau": data[:, 22:29],
    }


def load_actual(path: str) -> dict | None:
    """
    Load actual_trajectory.csv (36 columns):
      t | q0..q6 | dq0..dq6 | tau_ext0..tau_ext6 | tau_cmd0..tau_cmd6
      | sigma_min | tau_ext_max
    """
    p = Path(path)
    if not p.exists():
        print(f"[WARN] Actual trajectory not found: {path}")
        print("       Run the executor first to generate it.")
        return None

    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[np.newaxis, :]

    return {
        "t":           data[:, 0],
        "q":           data[:, 1:8],
        "dq":          data[:, 8:15],
        "tau_ext":     data[:, 15:22],
        "tau_cmd":     data[:, 22:29],
        "sigma_min":   data[:, 29],
        "tau_ext_max": data[:, 30],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Shared plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _legend(ax, planned_label="Planned", actual_label="Actual"):
    handles = [
        Line2D([0],[0], color=PLANNED_COLOR, lw=1.8, label=planned_label),
        Line2D([0],[0], color=ACTUAL_COLOR,  lw=1.2, label=actual_label, alpha=0.85),
        Line2D([0],[0], color=LIMIT_COLOR,   lw=1.0, ls="--", label="Limit"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper right")


def _style(ax, ylabel, xlim=None):
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlabel("Time [s]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.spines[["top","right"]].set_visible(False)
    if xlim is not None:
        ax.set_xlim(xlim)


def _add_limits(ax, lim_pos, lim_neg=None):
    ax.axhline( lim_pos, color=LIMIT_COLOR, lw=0.9, ls="--", alpha=0.7)
    ax.axhline(-lim_pos if lim_neg is None else lim_neg,
               color=LIMIT_COLOR, lw=0.9, ls="--", alpha=0.7)


def _tracking_error(ax, t_plan, planned, t_act, actual, ylabel):
    """Plot tracking error = actual - planned (interpolated onto planned time)."""
    if t_act is None or actual is None:
        return
    actual_interp = np.interp(t_plan, t_act, actual)
    error = actual_interp - planned
    ax.plot(t_plan, error, color=ACTUAL_COLOR, lw=0.9, alpha=0.85)
    ax.axhline(0, color=PLANNED_COLOR, lw=0.8, ls="--", alpha=0.5)
    ax.fill_between(t_plan, error, alpha=0.15, color=ACTUAL_COLOR)
    _style(ax, ylabel)
    rms = float(np.sqrt(np.mean(error**2)))
    ax.set_title(f"RMS error = {rms*1000:.3f} m-rad", fontsize=7, pad=2)


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 1 — Joint positions
# ─────────────────────────────────────────────────────────────────────────────

def plot_positions(planned: dict, actual: dict | None, save: bool):
    fig, axes = plt.subplots(4, 7, figsize=(20, 10))
    fig.suptitle("Joint Positions — Planned vs Actual", fontsize=13, y=0.98)

    for j in range(7):
        # Top row: overlay
        ax = axes[0, j]
        ax.plot(planned["t"], planned["q"][:, j],
                color=PLANNED_COLOR, lw=1.8, label="Planned")
        if actual is not None:
            ax.plot(actual["t"], actual["q"][:, j],
                    color=ACTUAL_COLOR, lw=1.2, alpha=0.85, label="Actual")
        ax.set_title(JOINT_NAMES[j], fontsize=9, pad=3)
        _style(ax, "Position [rad]")
        if j == 0: _legend(ax)

        # Second row: planned only (zoomed)
        ax2 = axes[1, j]
        ax2.plot(planned["t"], planned["q"][:, j],
                 color=PLANNED_COLOR, lw=1.5)
        _style(ax2, "Planned [rad]")

        # Third row: actual only
        ax3 = axes[2, j]
        if actual is not None:
            ax3.plot(actual["t"], actual["q"][:, j],
                     color=ACTUAL_COLOR, lw=1.2, alpha=0.85)
        _style(ax3, "Actual [rad]")

        # Bottom row: tracking error
        ax4 = axes[3, j]
        if actual is not None:
            _tracking_error(ax4,
                            planned["t"], planned["q"][:, j],
                            actual["t"],  actual["q"][:, j],
                            "Error [rad]")
        else:
            _style(ax4, "Error [rad]")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save:
        fig.savefig("fig_positions.png", dpi=150)
        print("[SAVE] fig_positions.png")


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 2 — Joint velocities
# ─────────────────────────────────────────────────────────────────────────────

def plot_velocities(planned: dict, actual: dict | None, save: bool):
    fig, axes = plt.subplots(3, 7, figsize=(20, 8))
    fig.suptitle("Joint Velocities — Planned vs Actual", fontsize=13, y=0.98)

    for j in range(7):
        ax = axes[0, j]
        ax.plot(planned["t"], planned["dq"][:, j],
                color=PLANNED_COLOR, lw=1.8)
        if actual is not None:
            ax.plot(actual["t"], actual["dq"][:, j],
                    color=ACTUAL_COLOR, lw=1.2, alpha=0.85)
        _add_limits(ax, VEL_MAX[j])
        ax.set_title(JOINT_NAMES[j], fontsize=9, pad=3)
        _style(ax, "Velocity [rad/s]")
        if j == 0: _legend(ax)

        ax2 = axes[1, j]
        # Peak velocity marker
        peak_idx = np.argmax(np.abs(planned["dq"][:, j]))
        ax2.plot(planned["t"], planned["dq"][:, j],
                 color=PLANNED_COLOR, lw=1.5)
        ax2.axvline(planned["t"][peak_idx], color=LIMIT_COLOR,
                    lw=0.8, ls=":", alpha=0.6)
        peak_v = planned["dq"][peak_idx, j]
        ax2.annotate(f"{peak_v:.3f}",
                     xy=(planned["t"][peak_idx], peak_v),
                     fontsize=6, color=PLANNED_COLOR)
        _add_limits(ax2, VEL_MAX[j])
        _style(ax2, "Planned [rad/s]")

        ax3 = axes[2, j]
        if actual is not None:
            _tracking_error(ax3,
                            planned["t"], planned["dq"][:, j],
                            actual["t"],  actual["dq"][:, j],
                            "Error [rad/s]")
        else:
            _style(ax3, "Error [rad/s]")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save:
        fig.savefig("fig_velocities.png", dpi=150)
        print("[SAVE] fig_velocities.png")


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 3 — Joint torques
# ─────────────────────────────────────────────────────────────────────────────

def plot_torques(planned: dict, actual: dict | None, save: bool):
    fig, axes = plt.subplots(3, 7, figsize=(20, 8))
    fig.suptitle("Joint Torques — Planned (RNEA feedforward) vs Commanded",
                 fontsize=13, y=0.98)

    for j in range(7):
        ax = axes[0, j]
        ax.plot(planned["t"], planned["tau"][:, j],
                color=PLANNED_COLOR, lw=1.8, label="RNEA (planned)")
        if actual is not None:
            ax.plot(actual["t"], actual["tau_cmd"][:, j],
                    color=ACTUAL_COLOR, lw=1.2, alpha=0.85,
                    label="Commanded")
        _add_limits(ax, TAU_MAX[j])
        ax.set_title(JOINT_NAMES[j], fontsize=9, pad=3)
        _style(ax, "Torque [Nm]")
        if j == 0:
            handles = [
                Line2D([0],[0],color=PLANNED_COLOR,lw=1.8,label="RNEA (planned)"),
                Line2D([0],[0],color=ACTUAL_COLOR, lw=1.2,label="Commanded"),
                Line2D([0],[0],color=LIMIT_COLOR,  lw=1.0,ls="--",label="Limit"),
            ]
            ax.legend(handles=handles, fontsize=7, loc="upper right")

        # External torque
        ax2 = axes[1, j]
        if actual is not None:
            ax2.plot(actual["t"], actual["tau_ext"][:, j],
                     color=ACTUAL_COLOR, lw=1.0, alpha=0.85)
            ax2.axhline(0, color=PLANNED_COLOR, lw=0.8, ls="--", alpha=0.5)
            ax2.fill_between(actual["t"], actual["tau_ext"][:, j],
                             alpha=0.15, color=ACTUAL_COLOR)
        ax2.set_title("tau_ext", fontsize=8, pad=2)
        _style(ax2, "Ext. torque [Nm]")

        # Torque difference: commanded - planned_ff
        ax3 = axes[2, j]
        if actual is not None:
            _tracking_error(ax3,
                            planned["t"], planned["tau"][:, j],
                            actual["t"],  actual["tau_cmd"][:, j],
                            "Cmd - FF [Nm]")
        else:
            _style(ax3, "Cmd - FF [Nm]")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save:
        fig.savefig("fig_torques.png", dpi=150)
        print("[SAVE] fig_torques.png")


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 4 — Runtime monitoring signals
# ─────────────────────────────────────────────────────────────────────────────

def plot_monitoring(planned: dict, actual: dict | None, save: bool):
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("Runtime Safety Monitoring", fontsize=13, y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # 1. sigma_min
    ax1 = fig.add_subplot(gs[0, 0])
    if actual is not None:
        ax1.plot(actual["t"], actual["sigma_min"],
                 color=ACTUAL_COLOR, lw=1.2)
        ax1.axhline(SIGMA_MIN_ESTOP, color=ESTOP_COLOR,
                    lw=1.2, ls="--", label=f"E-stop ({SIGMA_MIN_ESTOP})")
        ax1.axhline(0.04, color=LIMIT_COLOR,
                    lw=0.9, ls=":", label="Warn (0.04)")
        ax1.legend(fontsize=7)
        ax1.set_ylim(bottom=0)
    _style(ax1, "σ_min [m/rad]")
    ax1.set_title("Minimum singular value of Jacobian", fontsize=9)

    # 2. tau_ext max
    ax2 = fig.add_subplot(gs[0, 1])
    if actual is not None:
        ax2.plot(actual["t"], actual["tau_ext_max"],
                 color=ACTUAL_COLOR, lw=1.2)
        ax2.axhline(TAU_EXT_ESTOP, color=ESTOP_COLOR,
                    lw=1.2, ls="--", label=f"E-stop ({TAU_EXT_ESTOP} Nm)")
        ax2.legend(fontsize=7)
        ax2.set_ylim(bottom=0)
    _style(ax2, "max |τ_ext| [Nm]")
    ax2.set_title("Max external torque (contact detection)", fontsize=9)

    # 3. Position tracking error RMS across all joints
    ax3 = fig.add_subplot(gs[1, 0])
    if actual is not None:
        rms_per_step = []
        t_common = planned["t"]
        for j in range(7):
            q_interp = np.interp(t_common, actual["t"], actual["q"][:, j])
            rms_per_step.append((q_interp - planned["q"][:, j])**2)
        rms = np.sqrt(np.mean(rms_per_step, axis=0))
        ax3.plot(t_common, rms * 1000, color=ACTUAL_COLOR, lw=1.2)
        ax3.fill_between(t_common, rms * 1000, alpha=0.15, color=ACTUAL_COLOR)
    _style(ax3, "RMS error [mrad]")
    ax3.set_title("Position tracking error (RMS across joints)", fontsize=9)

    # 4. Velocity tracking error RMS
    ax4 = fig.add_subplot(gs[1, 1])
    if actual is not None:
        rms_vel = []
        for j in range(7):
            dq_interp = np.interp(t_common, actual["t"], actual["dq"][:, j])
            rms_vel.append((dq_interp - planned["dq"][:, j])**2)
        rms_v = np.sqrt(np.mean(rms_vel, axis=0))
        ax4.plot(t_common, rms_v * 1000, color=ACTUAL_COLOR, lw=1.2)
        ax4.fill_between(t_common, rms_v * 1000,
                         alpha=0.15, color=ACTUAL_COLOR)
    _style(ax4, "RMS error [mrad/s]")
    ax4.set_title("Velocity tracking error (RMS across joints)", fontsize=9)

    if save:
        fig.savefig("fig_monitoring.png", dpi=150)
        print("[SAVE] fig_monitoring.png")


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 5 — Summary stats table
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary(planned: dict, actual: dict | None, save: bool):
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle("Trajectory Summary Statistics", fontsize=13)
    ax.axis("off")

    headers = ["Joint", "Peak vel (plan)",  "Peak vel (act)",
                         "Peak tau (plan)",  "Peak tau_cmd (act)",
                         "Pos RMS error",    "Vel RMS error"]
    rows = []
    for j in range(7):
        pv_plan = f"{np.max(np.abs(planned['dq'][:,j])):.4f} rad/s"
        pt_plan = f"{np.max(np.abs(planned['tau'][:,j])):.2f} Nm"

        if actual is not None:
            pv_act = f"{np.max(np.abs(actual['dq'][:,j])):.4f} rad/s"
            pt_act = f"{np.max(np.abs(actual['tau_cmd'][:,j])):.2f} Nm"

            q_int  = np.interp(planned["t"], actual["t"], actual["q"][:,j])
            dq_int = np.interp(planned["t"], actual["t"], actual["dq"][:,j])
            pos_rms = f"{np.sqrt(np.mean((q_int  - planned['q'][:,j])**2))*1000:.3f} mrad"
            vel_rms = f"{np.sqrt(np.mean((dq_int - planned['dq'][:,j])**2))*1000:.3f} mrad/s"
        else:
            pv_act = pt_act = pos_rms = vel_rms = "—"

        rows.append([JOINT_NAMES[j], pv_plan, pv_act,
                     pt_plan, pt_act, pos_rms, vel_rms])

    table = ax.table(cellText=rows, colLabels=headers,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    # Header styling
    for j in range(len(headers)):
        table[0, j].set_facecolor("#1E3A5F")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternating row colours
    for i, row in enumerate(rows):
        for j in range(len(headers)):
            table[i+1, j].set_facecolor("#EFF6FF" if i % 2 == 0 else "white")

    fig.tight_layout()
    if save:
        fig.savefig("fig_summary.png", dpi=150)
        print("[SAVE] fig_summary.png")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot planned vs actual Panda trajectory"
    )
    parser.add_argument("planned_csv", nargs="?",
                        default="../trajectory.csv")
    parser.add_argument("actual_csv", nargs="?",
                        default="../actual_trajectory.csv")
    parser.add_argument("--save", action="store_true",
                        help="Save figures as PNG files")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't display interactive window (use with --save)")
    args = parser.parse_args()

    print(f"Loading planned : {args.planned_csv}")
    try:
        planned = load_planned(args.planned_csv)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    print(f"  {len(planned['t'])} samples, duration={planned['t'][-1]:.3f}s")

    print(f"Loading actual  : {args.actual_csv}")
    actual = load_actual(args.actual_csv)
    if actual is not None:
        print(f"  {len(actual['t'])} samples, duration={actual['t'][-1]:.3f}s")
    else:
        print("  Plotting planned trajectory only.")

    if args.no_show:
        matplotlib.use("Agg")

    plot_positions(planned, actual, args.save)
    plot_velocities(planned, actual, args.save)
    plot_torques(planned, actual, args.save)
    plot_monitoring(planned, actual, args.save)
    plot_summary(planned, actual, args.save)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
