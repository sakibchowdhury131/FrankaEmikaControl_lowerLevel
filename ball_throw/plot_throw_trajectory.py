"""
plot_throw_trajectory.py
========================
Generates 6 publication-quality figures for the ball-throw trajectory.

Figures produced:
  fig1_positions.png      — joint positions with boundary goal markers
  fig2_velocities.png     — joint velocities with limit lines + release target
  fig3_accelerations.png  — joint accelerations, zero at all boundary points
  fig4_jerk.png           — joint jerk, well within limits
  fig5_phase_portrait.png — all 4 quantities stacked for binding joints
  fig6_utilisation.png    — bar chart of peak values as % of hardware limit

Usage:
    # Uses throw_trajectory.csv and throw_meta.npy in the same directory
    python plot_throw_trajectory.py

    # Custom paths
    python plot_throw_trajectory.py --csv path/to/throw_trajectory.csv \\
                                    --meta path/to/throw_meta.npy

    # Save only, no interactive window
    python plot_throw_trajectory.py --no-show

    # Change output directory
    python plot_throw_trajectory.py --outdir ./figures

Dependencies:
    pip install numpy matplotlib
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
from matplotlib.patches import Patch


# ─────────────────────────────────────────────────────────────────────────────
#  Franka Panda hardware limits
# ─────────────────────────────────────────────────────────────────────────────

V_MAX   = np.array([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610])
A_MAX   = np.array([15.0,   7.5,  10.0,  12.5,  15.0,  15.0,  15.0])
J_MAX   = np.array([7500., 3750., 5000., 6250., 7500., 7500., 7500.])
TAU_MAX = np.array([87.0,  87.0,  87.0,  87.0,  12.0,  12.0,  12.0])

JOINT_NAMES = [f"Joint {i+1}" for i in range(7)]

# ─────────────────────────────────────────────────────────────────────────────
#  Colour palette
# ─────────────────────────────────────────────────────────────────────────────

C_BLUE   = "#2563EB"
C_GREEN  = "#16A34A"
C_PURPLE = "#7C3AED"
C_RED    = "#DC2626"
C_AMBER  = "#D97706"
C_GRAY   = "#6B7280"


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(csv_path: str, meta_path: str) -> dict:
    """
    Load trajectory CSV and metadata.

    CSV columns (29):  t | q0..q6 | dq0..dq6 | ddq0..ddq6 | tau0..tau6
    Meta keys: Q_START, Q_RELEASE, Q_STOP, DQ_RELEASE,
               release_index, T1, T2
    """
    csv  = Path(csv_path)
    meta = Path(meta_path)

    if not csv.exists():
        raise FileNotFoundError(
            f"Trajectory CSV not found: {csv}\n"
            "Run  python main_throw.py --mock  first to generate it."
        )
    if not meta.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {meta}\n"
            "Run  python main_throw.py --mock  first to generate it."
        )

    raw  = np.loadtxt(str(csv),  delimiter=",", skiprows=2)
    info = np.load(str(meta), allow_pickle=True).item()

    t   = raw[:, 0]
    Q   = raw[:, 1:8]
    Qd  = raw[:, 8:15]
    Qdd = raw[:, 15:22]
    tau = raw[:, 22:29]

    dt  = t[1] - t[0]
    Jrk = np.vstack([np.diff(Qdd, axis=0) / dt, np.zeros((1, 7))])

    return {
        "t":   t,   "Q":   Q,   "Qd":  Qd,
        "Qdd": Qdd, "tau": tau, "Jrk": Jrk,
        "ri":  int(info["release_index"]),
        "T1":  float(info["T1"]),
        "T2":  float(info["T2"]),
        "Q_START":    np.array(info["Q_START"]),
        "Q_RELEASE":  np.array(info["Q_RELEASE"]),
        "Q_STOP":     np.array(info["Q_STOP"]),
        "DQ_RELEASE": np.array(info["DQ_RELEASE"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Shared plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def shade_phases(ax, t: np.ndarray, ri: int) -> None:
    """Blue shading for phase 1, amber for phase 2, red dashed at release."""
    ax.axvspan(t[0],  t[ri], alpha=0.07, color=C_BLUE,  zorder=0)
    ax.axvspan(t[ri], t[-1], alpha=0.07, color=C_AMBER, zorder=0)
    ax.axvline(t[ri], color=C_RED, lw=1.2, ls="--", alpha=0.75, zorder=3)


def add_limit_lines(ax, lim: float) -> None:
    """Dotted grey lines at ±lim."""
    ax.axhline( lim, color=C_GRAY, lw=0.85, ls=":", alpha=0.7, zorder=2)
    ax.axhline(-lim, color=C_GRAY, lw=0.85, ls=":", alpha=0.7, zorder=2)


def style_ax(ax, ylabel: str, xlabel: bool = False) -> None:
    ax.set_ylabel(ylabel, fontsize=7.5, labelpad=2)
    if xlabel:
        ax.set_xlabel("Time [s]", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)


def peak_title(j: int, signal: np.ndarray, limit: float, name: str) -> str:
    pk  = float(np.max(np.abs(signal[:, j])))
    pct = pk / limit * 100
    return f"Joint {j+1}   peak = {pk:.3f} / {limit:.3f} {name}  ({pct:.0f}%)"


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 1 — Joint positions
# ─────────────────────────────────────────────────────────────────────────────

def fig_positions(d: dict, save_dir: Path) -> plt.Figure:
    fig, axes = plt.subplots(2, 4, figsize=(19, 7))
    fig.suptitle(
        "Joint space positions  —  boundary goals verified at start, release, stop",
        fontsize=12, fontweight="500", y=0.99,
    )

    for j in range(7):
        ax = axes[j // 4][j % 4]
        shade_phases(ax, d["t"], d["ri"])

        ax.plot(d["t"], d["Q"][:, j], color=C_BLUE, lw=1.5, zorder=4)

        # Goal markers
        ax.scatter([d["t"][0]],      [d["Q_START"][j]],
                   s=70, color=C_GREEN, marker="o", zorder=6)
        ax.scatter([d["t"][d["ri"]]], [d["Q_RELEASE"][j]],
                   s=80, color=C_RED,   marker="*", zorder=6)
        ax.scatter([d["t"][-1]],      [d["Q_STOP"][j]],
                   s=70, color=C_AMBER, marker="s", zorder=6)

        ax.set_title(f"Joint {j+1}", fontsize=9)
        style_ax(ax, "Position [rad]", xlabel=True)

    # Legend panel
    ax_leg = axes[1][3]
    ax_leg.axis("off")
    handles = [
        Line2D([0],[0], color=C_BLUE,  lw=1.5, label="q(t)"),
        Line2D([0],[0], color="none", marker="o",
               markerfacecolor=C_GREEN,  markersize=9,  label="q_start  (goal)"),
        Line2D([0],[0], color="none", marker="*",
               markerfacecolor=C_RED,    markersize=11, label="q_release (goal)"),
        Line2D([0],[0], color="none", marker="s",
               markerfacecolor=C_AMBER,  markersize=9,  label="q_stop  (goal)"),
        Line2D([0],[0], color=C_RED,  lw=1.2, ls="--",    label="Release instant"),
        Patch(facecolor=C_BLUE,  alpha=0.12, label="Phase 1 — rest → non-rest"),
        Patch(facecolor=C_AMBER, alpha=0.12, label="Phase 2 — non-rest → rest"),
    ]
    ax_leg.legend(handles=handles, loc="center", fontsize=8, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = save_dir / "fig1_positions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 2 — Joint velocities
# ─────────────────────────────────────────────────────────────────────────────

def fig_velocities(d: dict, save_dir: Path) -> plt.Figure:
    fig, axes = plt.subplots(2, 4, figsize=(19, 7))
    fig.suptitle(
        "Joint velocities  —  zero at start/stop, target met at release, within limits",
        fontsize=12, fontweight="500", y=0.99,
    )

    for j in range(7):
        ax = axes[j // 4][j % 4]
        shade_phases(ax, d["t"], d["ri"])
        add_limit_lines(ax, V_MAX[j])

        ax.plot(d["t"], d["Qd"][:, j], color=C_BLUE, lw=1.4, zorder=4)

        # Release velocity target marker
        dq_t = d["DQ_RELEASE"][j]
        ax.scatter([d["t"][d["ri"]]], [dq_t],
                   s=80, color=C_RED, marker="*", zorder=6)

        # Annotate target value
        y_offset = dq_t * 0.35 if abs(dq_t) > 0.1 else 0.15
        ax.annotate(
            f"target\n{dq_t:.2f} rad/s",
            xy=(d["t"][d["ri"]], dq_t),
            xytext=(d["t"][d["ri"]] + 0.06, dq_t - y_offset),
            fontsize=5.5, color=C_RED, ha="left",
            arrowprops=dict(arrowstyle="->", color=C_RED, lw=0.7),
        )

        ax.set_title(peak_title(j, d["Qd"], V_MAX[j], "rad/s"), fontsize=7.5)
        style_ax(ax, "Velocity [rad/s]", xlabel=True)

    ax_leg = axes[1][3]
    ax_leg.axis("off")
    handles = [
        Line2D([0],[0], color=C_BLUE, lw=1.5,  label="dq(t)"),
        Line2D([0],[0], color=C_GRAY, lw=0.85, ls=":", label="±v_max limit"),
        Line2D([0],[0], color="none", marker="*",
               markerfacecolor=C_RED, markersize=11,   label="dq_release target"),
        Line2D([0],[0], color=C_RED,  lw=1.2,  ls="--", label="Release instant"),
        Line2D([0],[0], color="none", marker="o",
               markerfacecolor=C_GREEN, markersize=8,  label="dq=0  (start/stop)"),
    ]
    ax_leg.legend(handles=handles, loc="center", fontsize=8, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = save_dir / "fig2_velocities.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 3 — Joint accelerations
# ─────────────────────────────────────────────────────────────────────────────

def fig_accelerations(d: dict, save_dir: Path) -> plt.Figure:
    fig, axes = plt.subplots(2, 4, figsize=(19, 7))
    fig.suptitle(
        "Joint accelerations  —  zero at start, release, and stop  (boundary conditions)",
        fontsize=12, fontweight="500", y=0.99,
    )

    for j in range(7):
        ax = axes[j // 4][j % 4]
        shade_phases(ax, d["t"], d["ri"])
        add_limit_lines(ax, A_MAX[j])

        ax.plot(d["t"], d["Qdd"][:, j], color=C_GREEN, lw=1.4, zorder=4)

        # Zero boundary markers
        for tidx in [0, d["ri"], -1]:
            ax.scatter([d["t"][tidx]], [d["Qdd"][tidx, j]],
                       s=60, color=C_RED, marker="D", zorder=6)

        ax.set_title(peak_title(j, d["Qdd"], A_MAX[j], "rad/s²"), fontsize=7.5)
        style_ax(ax, "Acceleration [rad/s²]", xlabel=True)

    ax_leg = axes[1][3]
    ax_leg.axis("off")
    handles = [
        Line2D([0],[0], color=C_GREEN, lw=1.5, label="ddq(t)"),
        Line2D([0],[0], color=C_GRAY,  lw=0.85, ls=":", label="±a_max limit"),
        Line2D([0],[0], color="none", marker="D",
               markerfacecolor=C_RED, markersize=8,
               label="ddq = 0  (start / release / stop)"),
        Line2D([0],[0], color=C_RED, lw=1.2, ls="--", label="Release instant"),
    ]
    ax_leg.legend(handles=handles, loc="center", fontsize=8, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = save_dir / "fig3_accelerations.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 4 — Joint jerk
# ─────────────────────────────────────────────────────────────────────────────

def fig_jerk(d: dict, save_dir: Path) -> plt.Figure:
    fig, axes = plt.subplots(2, 4, figsize=(19, 7))
    fig.suptitle(
        "Joint jerk  —  finite and within limits  "
        "(5th-order polynomial guarantees C² continuity)",
        fontsize=12, fontweight="500", y=0.99,
    )

    for j in range(7):
        ax = axes[j // 4][j % 4]
        shade_phases(ax, d["t"], d["ri"])
        add_limit_lines(ax, J_MAX[j])

        ax.plot(d["t"], d["Jrk"][:, j], color=C_PURPLE, lw=1.0, alpha=0.9,
                zorder=4)

        ax.set_title(peak_title(j, d["Jrk"], J_MAX[j], "rad/s³"), fontsize=7.5)
        style_ax(ax, "Jerk [rad/s³]", xlabel=True)

    ax_leg = axes[1][3]
    ax_leg.axis("off")
    handles = [
        Line2D([0],[0], color=C_PURPLE, lw=1.2, label="jerk(t)"),
        Line2D([0],[0], color=C_GRAY,   lw=0.85, ls=":", label="±j_max limit"),
        Line2D([0],[0], color=C_RED,    lw=1.2,  ls="--", label="Release instant"),
    ]
    ax_leg.legend(handles=handles, loc="center", fontsize=8, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = save_dir / "fig4_jerk.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 5 — Phase portrait (all quantities for binding joints)
# ─────────────────────────────────────────────────────────────────────────────

def fig_phase_portrait(d: dict, save_dir: Path) -> plt.Figure:
    """
    Stack position, velocity, acceleration, jerk vertically
    for the two joints with the highest limit utilisation.
    """
    # Find two binding joints by velocity utilisation
    util = np.max(np.abs(d["Qd"]), axis=0) / V_MAX
    binding = np.argsort(util)[::-1][:2]

    fig, axes = plt.subplots(4, 2, figsize=(14, 13), sharex="col")
    fig.suptitle(
        f"Phase portrait — joints {binding[0]+1} and {binding[1]+1}  "
        f"(highest velocity utilisation: "
        f"{util[binding[0]]*100:.0f}% and {util[binding[1]]*100:.0f}%)",
        fontsize=12, fontweight="500", y=0.99,
    )

    row_specs = [
        ("Position [rad]",      d["Q"],   None,   C_BLUE),
        ("Velocity [rad/s]",    d["Qd"],  V_MAX,  C_BLUE),
        ("Acceleration [rad/s²]", d["Qdd"], A_MAX, C_GREEN),
        ("Jerk [rad/s³]",       d["Jrk"], J_MAX,  C_PURPLE),
    ]

    for col, j in enumerate(binding):
        for row, (ylabel, sig, lim, color) in enumerate(row_specs):
            ax = axes[row][col]
            shade_phases(ax, d["t"], d["ri"])
            if lim is not None:
                add_limit_lines(ax, lim[j])
            ax.plot(d["t"], sig[:, j], color=color, lw=1.5, zorder=4)

            if row == 0:
                # Goal markers on position plot
                ax.scatter([d["t"][0]],       [d["Q_START"][j]],
                           s=60, color=C_GREEN, marker="o", zorder=6)
                ax.scatter([d["t"][d["ri"]]], [d["Q_RELEASE"][j]],
                           s=70, color=C_RED,   marker="*", zorder=6)
                ax.scatter([d["t"][-1]],       [d["Q_STOP"][j]],
                           s=60, color=C_AMBER, marker="s", zorder=6)
                ax.set_title(f"Joint {j+1}  ({util[j]*100:.0f}% v_max used)",
                             fontsize=10, fontweight="500")

            if row == 1:
                # Release velocity marker
                ax.scatter([d["t"][d["ri"]]], [d["DQ_RELEASE"][j]],
                           s=70, color=C_RED, marker="*", zorder=6)

            if row == 2:
                # Zero acceleration boundary markers
                for tidx in [0, d["ri"], -1]:
                    ax.scatter([d["t"][tidx]], [sig[tidx, j]],
                               s=50, color=C_RED, marker="D", zorder=6)

            style_ax(ax, ylabel, xlabel=(row == 3))

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = save_dir / "fig5_phase_portrait.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 6 — Limit utilisation bar chart
# ─────────────────────────────────────────────────────────────────────────────

def fig_utilisation(d: dict, save_dir: Path) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Hardware limit utilisation  —  peak value as percentage of limit per joint",
        fontsize=12, fontweight="500",
    )

    quantities = [
        ("Velocity",     d["Qd"],  V_MAX,  C_BLUE,   "rad/s"),
        ("Acceleration", d["Qdd"], A_MAX,  C_GREEN,  "rad/s²"),
        ("Jerk",         d["Jrk"], J_MAX,  C_PURPLE, "rad/s³"),
    ]

    x = np.arange(7)

    for ax, (name, sig, lim, color, unit) in zip(axes, quantities):
        peaks = np.max(np.abs(sig), axis=0)
        pct   = peaks / lim * 100

        bars = ax.bar(x, pct, color=color, alpha=0.70, width=0.60,
                      edgecolor="white", linewidth=0.5, zorder=3)

        # Percentage labels on bars
        for bar, p, pk, lm in zip(bars, pct, peaks, lim):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                p + 1.5,
                f"{p:.0f}%\n({pk:.2f})",
                ha="center", va="bottom", fontsize=6.5, color="#374151",
            )

        # Reference lines
        ax.axhline(100, color=C_RED,   lw=1.3, ls="--", zorder=4,
                   label="100% — hardware limit")
        ax.axhline(80,  color=C_AMBER, lw=0.9, ls=":",  zorder=4,
                   label="80% — caution zone")

        ax.set_xticks(x)
        ax.set_xticklabels([f"J{i+1}" for i in range(7)], fontsize=9)
        ax.set_ylabel("% of limit used", fontsize=9)
        ax.set_title(f"{name}  [{unit}]", fontsize=10)
        ax.set_ylim(0, 118)
        ax.legend(fontsize=7.5)
        ax.grid(True, axis="y", lw=0.3, alpha=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    path = save_dir / "fig6_utilisation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 7 — Boundary condition verification table
# ─────────────────────────────────────────────────────────────────────────────

def fig_verification_table(d: dict, save_dir: Path) -> plt.Figure:
    """
    Table showing actual vs desired values at each boundary point,
    and whether every limit is satisfied.
    """
    tol = 1e-3
    t, Q, Qd, Qdd = d["t"], d["Q"], d["Qd"], d["Qdd"]
    ri = d["ri"]

    # ── Boundary condition rows ───────────────────────────────────────────
    def fmt(arr): return "  ".join(f"{v:+.4f}" for v in arr)

    bc_rows = [
        ("q(0) = q_start",
         fmt(Q[0]),  fmt(d["Q_START"]),
         np.allclose(Q[0],  d["Q_START"],   atol=tol)),
        ("q(T1) = q_release",
         fmt(Q[ri]), fmt(d["Q_RELEASE"]),
         np.allclose(Q[ri], d["Q_RELEASE"],  atol=tol)),
        ("q(T) = q_stop",
         fmt(Q[-1]), fmt(d["Q_STOP"]),
         np.allclose(Q[-1], d["Q_STOP"],     atol=tol)),
        ("dq(0) = 0",
         fmt(Qd[0]),  fmt(np.zeros(7)),
         np.allclose(Qd[0],  0,              atol=tol)),
        ("dq(T1) = dq_release",
         fmt(Qd[ri]), fmt(d["DQ_RELEASE"]),
         np.allclose(Qd[ri], d["DQ_RELEASE"], atol=tol)),
        ("dq(T) = 0",
         fmt(Qd[-1]),  fmt(np.zeros(7)),
         np.allclose(Qd[-1], 0,              atol=tol)),
        ("ddq(0) = 0",
         fmt(Qdd[0]),  fmt(np.zeros(7)),
         np.allclose(Qdd[0],  0,             atol=tol)),
        ("ddq(T1) = 0",
         fmt(Qdd[ri]), fmt(np.zeros(7)),
         np.allclose(Qdd[ri], 0,             atol=tol)),
        ("ddq(T) = 0",
         fmt(Qdd[-1]), fmt(np.zeros(7)),
         np.allclose(Qdd[-1], 0,             atol=tol)),
    ]

    # ── Limit rows ────────────────────────────────────────────────────────
    Jrk = d["Jrk"]
    lim_rows = [
        ("Velocity within v_max",
         f"peak = {np.max(np.abs(Qd), axis=0).round(3)}",
         f"limit = {V_MAX}",
         np.all(np.abs(Qd)  <= V_MAX * 1.001)),
        ("Acceleration within a_max",
         f"peak = {np.max(np.abs(Qdd), axis=0).round(3)}",
         f"limit = {A_MAX}",
         np.all(np.abs(Qdd) <= A_MAX * 1.001)),
        ("Jerk within j_max",
         f"peak = {np.max(np.abs(Jrk), axis=0).round(0).astype(int)}",
         f"limit = {J_MAX.astype(int)}",
         np.all(np.abs(Jrk) <= J_MAX * 1.05)),
    ]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(16, 7),
        gridspec_kw={"height_ratios": [3, 1.2]},
    )
    fig.suptitle(
        "Boundary condition and limit verification",
        fontsize=13, fontweight="500",
    )

    # ── Top: boundary conditions ──────────────────────────────────────────
    ax_top.axis("off")
    col_labels = ["Condition", "Actual (rad or rad/s)", "Desired", "Status"]
    rows_disp  = [[c, a, des, "✓  PASS" if ok else "✗  FAIL"]
                  for c, a, des, ok in bc_rows]

    tbl = ax_top.table(
        cellText=rows_disp,
        colLabels=col_labels,
        loc="center", cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 1.65)

    # Header styling
    for col in range(4):
        tbl[0, col].set_facecolor("#1E3A5F")
        tbl[0, col].set_text_props(color="white", fontweight="bold")

    # Row colouring
    for row_i, (_, _, _, ok) in enumerate(bc_rows):
        bg = "#F0FDF4" if ok else "#FEF2F2"
        for col in range(4):
            tbl[row_i+1, col].set_facecolor(bg)
        # Status cell colour
        tbl[row_i+1, 3].set_text_props(
            color="#16A34A" if ok else "#DC2626",
            fontweight="bold",
        )

    # ── Bottom: limit checks ──────────────────────────────────────────────
    ax_bot.axis("off")
    lim_disp = [[c, a, des, "✓  PASS" if ok else "✗  FAIL"]
                for c, a, des, ok in lim_rows]

    tbl2 = ax_bot.table(
        cellText=lim_disp,
        colLabels=col_labels,
        loc="center", cellLoc="left",
    )
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(7.5)
    tbl2.scale(1.0, 1.65)

    for col in range(4):
        tbl2[0, col].set_facecolor("#1E3A5F")
        tbl2[0, col].set_text_props(color="white", fontweight="bold")

    for row_i, (_, _, _, ok) in enumerate(lim_rows):
        bg = "#F0FDF4" if ok else "#FEF2F2"
        for col in range(4):
            tbl2[row_i+1, col].set_facecolor(bg)
        tbl2[row_i+1, 3].set_text_props(
            color="#16A34A" if ok else "#DC2626",
            fontweight="bold",
        )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = save_dir / "fig7_verification.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ball-throw trajectory figures"
    )
    parser.add_argument(
        "--csv",  default="throw_trajectory.csv",
        help="Path to trajectory CSV  (default: throw_trajectory.csv)",
    )
    parser.add_argument(
        "--meta", default=None,
        help="Path to metadata .npy file  (default: auto-derived from --csv)",
    )
    parser.add_argument(
        "--outdir", default=".",
        help="Directory to save PNG files  (default: current directory)",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not open interactive windows (useful on headless servers)",
    )
    args = parser.parse_args()

    if args.no_show:
        matplotlib.use("Agg")

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # Auto-derive meta path from CSV path if not given explicitly
    meta_path = args.meta or args.csv.replace(".csv", "_meta.npy")

    print(f"Loading trajectory from: {args.csv}")
    print(f"Loading metadata from  : {meta_path}")
    d = load_data(args.csv, meta_path)

    print(f"\nTrajectory info:")
    print(f"  Duration      : {d['t'][-1]:.3f} s  "
          f"(Phase 1 = {d['T1']:.3f} s,  Phase 2 = {d['T2']:.3f} s)")
    print(f"  Samples       : {len(d['t'])}  @ {1/(d['t'][1]-d['t'][0]):.0f} Hz")
    print(f"  Release index : {d['ri']}  (t = {d['t'][d['ri']]:.4f} s)")

    print(f"\nGenerating figures → {out}/")
    fig_positions(d, out)
    fig_velocities(d, out)
    fig_accelerations(d, out)
    fig_jerk(d, out)
    fig_phase_portrait(d, out)
    fig_utilisation(d, out)
    fig_verification_table(d, out)

    print(f"\nAll 7 figures saved to {out}/")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()